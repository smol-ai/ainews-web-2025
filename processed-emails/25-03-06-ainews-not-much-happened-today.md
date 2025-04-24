---
id: 2ac42108-f069-4ad9-b095-330b2a12a6aa
title: not much happened today
date: '2025-03-07T05:50:14.781495Z'
original_slug: ainews-not-much-happened-today-3137
description: >-
  AI News for 3/6/2025-3/7/2025. Highlights include new model releases from AI21
  Labs, Mistral AI, Alibaba, AMD, and Anthropic, with advancements in multimodal
  OCR, reasoning models, open multilingual LLMs, and API updates.
companies:
  - ai21-labs
  - mistral-ai
  - alibaba
  - amd
  - anthropic
models:
  - jamba-1.6
  - mistral-ocr
  - qwen-32b
  - o1
  - o3-mini
  - jamba-1.6
  - instella
  - babel-9b
  - babel-83b
  - claude-3.7
topics:
  - model-releases
  - multimodal-ocr
  - multilingual-llm
  - reasoning-models
  - open-models
  - api-updates
  - ai-deployment
  - language-models
---


<!-- buttondown-editor-mode: plaintext -->**a quiet day.**

> AI News for 3/6/2025-3/7/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**227** channels, and **7886** messages) for you. Estimated reading time saved (at 200wpm): **777 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

[Mistral OCR](https://mistral.ai/fr/news/mistral-ocr) and [Jamba 1.6](https://www.ai21.com/jamba/) came close.

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Model Releases and Updates**

- **AI21 Labs launched Jamba 1.6**, claiming it's the **best open model for private enterprise deployment**, outperforming **Cohere, Mistral, and Llama** on key benchmarks like **Arena Hard**. It rivals leading closed models in speed and quality, and is available on **AI21 Studio** and [@Hugging Face](https://twitter.com/AI21Labs/status/1897657953261601151).
- **Mistral AI released a state-of-the-art multimodal OCR model** [@scaling01](https://twitter.com/scaling01/status/1897695665871872427). [@sophiamyang](https://twitter.com/sophiamyang/status/1897713370029068381) announced **Mistral OCR**, highlighting its **state-of-the-art document understanding**, **multilingual and multimodal capabilities**, and **speed**.  It offers **doc-as-prompt**, **structured output**, and is available for **on-prem deployment**. Benchmarks and examples are provided in their [blog post](https://twitter.com/sophiamyang/status/1897716142401060867), [multilingual capabilities](https://twitter.com/sophiamyang/status/1897715804042338327), [math equation extraction from PDFs](https://twitter.com/sophiamyang/status/1897715242936713364), and [text and image extraction to markdown](https://twitter.com/sophiamyang/status/1897713540506824954).  [@sophiamyang](https://twitter.com/sophiamyang/status/1897716870175682847) noted it's **#1 on Hacker News**.
- **Alibaba Qwen released QwQ-32B**, an **open-weight reasoning model** claimed to be close to **DeepSeek R1** and **OpenAI o1 mini** in intelligence, while only needing **32B parameters** and being **cost-effective at $0.20/M tokens**. It is available on [@Hugging Face](https://twitter.com/_philschmid/status/1897556185126932750) under **Apache 2.0**.  [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1897701015803380112) reported initial evals show **QwQ-32B** scoring **59.5% on GPQA Diamond (behind DeepSeek R1's 71% and Gemini 2.0 Flash's 62%)** but **78% on AIME 2024 (ahead of DeepSeek R1)**. [@awnihannun](https://twitter.com/awnihannun/status/1897394318434034163) demonstrated **QwQ-32B running on an M4 Max with MLX**, noting its **8k token thought process**. [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1897422055282500054) suggests **QwQ's new model** seems as good as **R1** while being runnable locally. [@reach_vb](https://twitter.com/reach_vb/status/1897686816037167394) announced **QwQ 32B is deployed on Hugging Chat**.
- **OpenAI released o1 and o3-mini in the API for developers**, available on all paid tiers, supporting **streaming, function calling, structured outputs, reasoning effort, Assistants API, Batch API, and vision (o1 only)** [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1897414494286176333). [@goodside](https://twitter.com/goodside/status/1897412604894789692) noted **ChatGPT Code Interpreter works in both 4.5 and o3-mini**, suggesting **o3-mini getting Code Interpreter is a big deal** [@goodside](https://twitter.com/goodside/status/1897418513062744200).
- **AI21 Labs launched Jamba 1.6 chat model with 94B active parameters and 398B total parameters** [@reach_vb](https://twitter.com/reach_vb/status/1897658668398555468).
- **AMD introduced Instella**, a series of **fully open-source, state-of-the-art 3B parameter language models**, trained on **AMD Instinct MI300X GPUs**, outperforming existing fully open 3B models and competing with **Llama-3.2-3B, Gemma-2-2B, and Qwen-2.5-3B** [@omarsar0](https://twitter.com/omarsar0/status/1897642582966165523).
- **Alibaba released Babel on Hugging Face**, **open multilingual LLMs** with variants **Babel-9B** and **Babel-83B**, outperforming comparable open LLMs and performing comparably to **GPT-4o** on certain tasks [@_akhaliq](https://twitter.com/_akhaliq/status/1897483872214077749).
- **Anthropic released Claude 3.7 Sonnet**, adding reasoning capabilities and workbench updates for prompt engineering with features like tool use and extended thinking, and prompt sharing [@AnthropicAI](https://twitter.com/AnthropicAI/status/1897696420293230989), [@alexalbert__](https://twitter.com/alexalbert__/status/1897696773151343103).

**Tools and Applications**

- **Elysian Labs announced Auren**, an iOS app aiming to improve human-AI interaction, focusing on emotional intelligence, agency, and positive reinforcement rather than just intelligence [@nearcyan](https://twitter.com/nearcyan/status/1897466463314936034). Beta tester feedback has been described as "surreal" and potentially "life-saving" [@nearcyan](https://twitter.com/nearcyan/status/1897470058768875704). The app uses multiple models per message and is priced at $19.99/month for 2,500 messages [@nearcyan](https://twitter.com/nearcyan/status/1897470389418414219).  [@nearcyan](https://twitter.com/nearcyan/status/1897514277705294057) highlighted the complexity of the app, noting it's more than just "an LLM in chat bubbles".
- **Hugging Face launched Diffusion Self-Distillation app**, enabling zero-shot customized image generation using FLUX, similar to DreamBooth but training-free, for tasks like character consistency and scene relighting [@_akhaliq](https://twitter.com/_akhaliq/status/1897496170358006179).
- **Hugging Face released PDF Parsers Playground**, a platform for experimenting with open-source PDF parsers [@_akhaliq](https://twitter.com/_akhaliq/status/1897482594117206376).
- **_philschmid** created a **CLI to chat with Google DeepMind Gemini 2.0 Flash** connected to **Google Search** [@_philschmid](https://twitter.com/_philschmid/status/1897397749395693626).
- **OpenAI released ChatGPT for macOS**, allowing code editing directly in IDEs for Plus, Pro, and Team users [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1897700857833193955).
- **Perplexity AI's Mac app now supports real-time voice mode**, allowing background listening and interaction via shortcut **Cmd + Shift + M** [@AravSrinivas](https://twitter.com/AravSrinivas/status/1897408183620264028).
- **LangChainAI released OpenCanvas**, similar to OpenAI's tool but compatible with every model [@_philschmid](https://twitter.com/_philschmid/status/1897405585118912618).
- **RisingSayak** shipped a **shot categorizer** for video data curation, claiming it's fast (<1s on CPU) and open-source [@RisingSayak](https://twitter.com/RisingSayak/status/1897590118736957442).

**Research and Concepts**

- **_philschmid** shared benchmarks on **ReAct Agents under pressure**, evaluating performance with scaling domains and tools, finding that **Claude 3.5 sonnet, o1, and o3-mini outperform gpt-4o and llama-3.3-70B** in tasks requiring 3+ tool calls, and that more context and tools can degrade performance [@_philschmid](https://twitter.com/_philschmid/status/1897688288896471546).
- **ArtificialAnlys** provided analysis of **Alibaba's QwQ-32B model**, comparing it to **DeepSeek R1** and **Gemini 2.0 Flash** on benchmarks like **GPQA Diamond and AIME 2024** [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1897701015803380112).
- **omarsar0** summarized a paper on **Cognitive Behaviors that Enable Self-Improving Reasoners**, identifying **verification, backtracking, subgoal setting, and backward chaining** as key for successful problem-solving in LMs, noting **Qwen-2.5-3B's** natural exhibition of these behaviors and the impact of priming and pretraining behavior amplification [@omarsar0](https://twitter.com/omarsar0/status/1897732423963885637).
- **polynoamial** highlighted **Richard Sutton's Bitter Lesson** about general methods scaling with data and compute ultimately winning in AI, in the context of the rise of AI agents [@polynoamial](https://twitter.com/polynoamial/status/1897693005601292491).
- **lateinteraction** discussed the power of **declarative languages** at the right abstraction level for building intelligent software, suggesting compilers as a way to make problem-specific systems "scale with data and compute" [@lateinteraction](https://twitter.com/lateinteraction/status/1897699917801701512). They also pondered the spectrum of software development from **ChatGPT** to **Copilot/Cursor** to **DSPy & Parsel**, suggesting a future with higher-level, composable specs [@lateinteraction](https://twitter.com/lateinteraction/status/1897442159789531504).
- **iScienceLuvr** shared a paper on **explaining generalization behavior in deep learning** with "soft inductive biases" [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1897619075364487338).
- **TheTuringPost** discussed why **AI reasoning tests keep failing**, highlighting **Goodhart's Law** and the need for **dynamic and adaptive benchmarks** that test commonsense reasoning, causal inference, and ethics beyond math and coding [@TheTuringPost](https://twitter.com/TheTuringPost/status/1897454185656005041).
- **omarsar0** discussed the evolution of **AI-powered IDEs** and agentic capabilities centralizing workflows, increasing productivity [@omarsar0](https://twitter.com/omarsar0/status/1897700328071385298).
- **cloneofsimo** discussed the importance of **flops/watt** in RL era and improvements in **DiLoCo** [@cloneofsimo](https://twitter.com/cloneofsimo/status/1897557416117686772).

**Industry and Business**

- **Figure AI** is reported to be the **6th most sought-after company in the secondary market** [@adcock_brett](https://twitter.com/adcock_brett/status/1897691903493279902).
- **ArtificialAnlys** congratulated **Together AI, Fireworks AI, hyperbolic labs, and GroqInc** for launching **serverless endpoints** and providing live performance benchmarks [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1897701018231881772).
- **ClementDelangue** from Hugging Face discussed **movements in top 50 GenAI consumer apps**, noting Hugging Face's position at 13th despite consumer app growth [@ClementDelangue](https://twitter.com/ClementDelangue/status/1897735590608609546). He also emphasized **academia's role in making AI a positive force**, highlighting **Academia Hub on Hugging Face** [@ClementDelangue](https://twitter.com/ClementDelangue/status/1897666379823669667).
- **SakanaAILabs** is hiring **Software Engineers** to develop AI applications in Japan using LLMs and AI agents [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1897447888814494110).
- **DeepLearningAI** is offering a **Data Analytics Professional Certificate** program [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1897723855835189556) and a new course on **agentic document workflows** with LlamaIndex [@jerryjliu0](https://twitter.com/jerryjliu0/status/1897668522425393509).
- **jeremyphoward** promoted **FastHTML**, suggesting a simple, single-language, single-file approach to development [@jeremyphoward](https://twitter.com/jeremyphoward/status/1897431400359526557).
- **matanSF** announced **FactoryAI's partnership with OpenAI**, aiming to build future software with human-AI collaboration in one platform [@matanSF](https://twitter.com/matanSF/status/1897694460592754829).
- **togethercompute** is building a **world-class kernels team** for production workloads and announced **ThunderMLA**, a fast MLA decode kernel [@togethercompute](https://twitter.com/togethercompute/status/1897703705790542137).
- **mervenoyann** noted the increasing market for **enterprise dev tooling with compliance** and mentioned **Dust** and **Hugging Face Enterprise Hub** as examples [@mervenoyann](https://twitter.com/mervenoyann/status/1897397563990663651).

**Opinions and Discussions**

- **scaling01** questioned the utility of the **Mistral OCR release for coding**, finding it behind **4o and o3-mini**, and wondered if it's mainly for "generating greentexts" [@scaling01](https://twitter.com/scaling01/status/1897590986278117758).
- **ajeya_cotra** asked for **qualitative analysis of Claude Plays Pokemon**, wanting to understand its successes, failures, and skill gaps, and if it played like a typical child of a certain age [@ajeya_cotra](https://twitter.com/ajeya_cotra/status/1897458906001231971).
- **cognitivecompai** requested a **torrent magnet link for MistralAI** models [@cognitivecompai](https://twitter.com/cognitivecompai/status/1897722351631925320) and criticized the lack of local model support in **Cursor AI and Windsurf AI**, recommending **continuedev and UseCline** instead [@cognitivecompai](https://twitter.com/cognitivecompai/status/1897598405884408261). They also expressed frustration with the **availability of NVIDIA GeForce 5090** [@cognitivecompai](https://twitter.com/cognitivecompai/status/1897586581302645080).
- **ID_AA_Carmack** discussed the nature of **monopolies** and the challenges of escaping them, arguing for a **free market with strong anti-cartel laws** [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1897678635147911393). He also reflected on **Seymour Cray's approach to engineering** and the need to adapt to incremental changes as projects mature [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1897671486229414017).
- **francoisfleuret** defended "leftism", arguing that a free market's fixed point might be "absolute shit" and wealth accumulation can be unstable [@francoisfleuret](https://twitter.com/francoisfleuret/status/1897654935400927373).
- **mmitchell_ai** raised concerns about **AI agents for war** potentially leading to a runaway missile crisis and questioned if preventing autonomous missile deployment by AI is still a discussion point [@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1897714132297908679).
- **soumithchintala** shared a note with the OpenAI team, expressing a take that aligns with "obedient students, not revolutionaries" in AI development, emphasizing the importance of picking the right questions for scientists and noting AI's current direction might be opposite to autonomous breakthroughs [@soumithchintala](https://twitter.com/soumithchintala/status/1897643753906512168).
- **DavidSHolz** believes coding agents will "take half of the total budget of software engineering as soon as possible" [@DavidSHolz](https://twitter.com/DavidSHolz/status/1897450419548516685).
- **abacaj** asked about the vibe on **QwQ** models, whether they are "benchmark maxxing or good model?" [@abacaj](https://twitter.com/abacaj/status/1897645343233241497).
- **nearcyan** believes that in the future, a majority of human social interaction will be with AIs rather than other humans [@nearcyan](https://twitter.com/nearcyan/status/1897469936190324943) and that **Auren** and **Seren** encourage healthy choices and socialization [@nearcyan](https://twitter.com/nearcyan/status/1897469595751211104).
- **HamelHusain** questioned why there's no **OAuth gateway for users to use their own LLM API tokens** for easier integration [@HamelHusain](https://twitter.com/HamelHusain/status/1897486751696085093).

**Memes/Humor**

- **dylan522p** made a futuristic joke about **AI robots killing 90% of humanity by 2035** and the remaining companies being **Marvell and AICHIP Mfg Co China** [@dylan522p](https://twitter.com/dylan522p/status/1897436641272213584).
- **gallabytes** shared an image generated by **Grok 3** of "a horse riding on top of an astronaut" [@gallabytes](https://twitter.com/gallabytes/status/1897523886901928193).
- **typedfemale** joked about "the persian" in SF who is "always rugging people" [@typedfemale](https://twitter.com/typedfemale/status/1897707779466707088) and that "etsy is a light wrapper for shopping on aliexpress" [@typedfemale](https://twitter.com/typedfemale/status/1897505688827412811).
- **abacaj** joked about a friend quitting his job to work on "MCP servers" and clarified "Guys it’s a joke don’t quit your job for MCP" [@abacaj](https://twitter.com/abacaj/status/1897683005746938106), [@abacaj](https://twitter.com/abacaj/status/1897682645657481726).
- **MillionInt** joked "So that’s how the world ends. Not with a bang but with greentext and pokemon badges" [@MillionInt](https://twitter.com/MillionInt/status/1897401097071026198).

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. M3 Ultra as a Competitive AI Workstation**

- **M3 Ultra is a slightly weakened 3090 w/ 512GB** ([Score: 509, Comments: 223](https://reddit.com/r/LocalLLaMA/comments/1j4jpij/m3_ultra_is_a_slightly_weakened_3090_w_512gb/)): The **M3 Ultra** is compared to a slightly weakened **NVIDIA 3090**, offering **114.688 TFLOPS FP16** and **819.2 GB/s memory bandwidth**, versus the 3090's **142.32 TFLOPS FP16** and **936 GB/s bandwidth**. The post speculates on **Apple's M3 Ultra** specs based on an article, suggesting a doubling of shaders per core to achieve significant performance improvements, with a potential future **M4 Ultra** offering enhanced specs like **137.6256 TFLOPS FP16** and **LPDDR5X** RAM. Pricing is estimated between **$10k-$15k**, with concerns about Apple's marketing potentially overstating improvements without actual hardware changes.
  - Discussions highlight concerns about the **M3 Ultra's prompt processing speed**, noting it's a primary weakness of the M1/M2 Ultras. Users emphasize the importance of **Unified RAM** for large language models, suggesting that Apple's RAM capabilities are a significant advantage over competitors like NVIDIA, despite potential shortcomings in shader core doubling and tensor core strength.
  - There is a debate over **performance comparisons** with NVIDIA's 3090 and the potential **M4 Ultra**. Some users argue that the M3 Ultra's **TFLOPS** numbers might be overstated, while others reference benchmarks and speculate on Apple's strategic positioning against NVIDIA and AMD, emphasizing Apple's focus on **VRAM** and **unified memory** as critical for AI applications.
  - Concerns about **cost-effectiveness** and applicability in research and professional settings are prevalent, with many suggesting that Macs are not the most cost-efficient for large-scale or university-level machine learning tasks. The discussions include the feasibility of using **DIGITS** and NVIDIA's **CUDA** in comparison to Apple's offerings, with some users defending Mac's capabilities for local ML tasks.


**Theme 2. Hunyuan Image-to-Video Release: GPU Heavy, Performance Debates**

- **[Hunyuan Image to Video released!](https://v.redd.it/yck5cznw92ne1)** ([Score: 320, Comments: 60](https://reddit.com/r/LocalLLaMA/comments/1j4u57l/hunyuan_image_to_video_released/)): **Hunyuan Image-to-Video** tool has been released, noted for its **high GPU requirements**. Further details on its functionality or performance are not provided in the post.
  - **GPU Requirements and Costs**: The **Hunyuan Image-to-Video** tool requires a GPU with **79GB minimum memory** for 360p, with **80GB recommended** for better quality. Users discuss renting GPUs from services like **vast.ai** and **lambdalabs.com** at approximately **$2/hour**, while some anticipate improvements that might reduce memory requirements to **8GB**.
  - **Comparison and Alternatives**: Users compare Hunyuan's performance to **Wan i2v**, noting it is faster but with lower quality. Alternatives like **Pinokio** and **Lambda** are mentioned for optimized workflows, and **ComfyUI** is highlighted as a potential workflow solution, with a link to [Comfy's blog](https://blog.comfy.org/p/hunyuan-image2video-day-1-support) for support.
  - **Licensing and Regional Restrictions**: There is a discussion on the licensing agreement, which does not apply in the **European Union, United Kingdom, and South Korea**. Users express skepticism about the legal basis of machine learning model licenses, anticipating future lobbying efforts for copyright protections.


**Theme 3. QwQ-32B: Efficient Reasoning vs. R1's Verbose Accuracy**

- **QwQ-32B seems to get the same quality final answer as R1 while reasoning much more concisely and efficiently** ([Score: 270, Comments: 118](https://reddit.com/r/LocalLLaMA/comments/1j4gw91/qwq32b_seems_to_get_the_same_quality_final_answer/)): **QwQ-32B** demonstrates superior performance compared to **R1**, providing concise and efficient reasoning while maintaining or surpassing answer quality. It uses approximately **4x fewer tokens** than R1, supporting the notion that not all Chains of Thought (CoTs) are equal, as **Adam** suggested, and indicating that **Qwen** has successfully trained their model for efficiency without sacrificing quality.
  - Users highlight that **QwQ-32B's** performance is sensitive to **temperature settings** and **quantization**, with lower temperatures improving code generation. **[Huggingface demo](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo)** results vary significantly from local setups, emphasizing the importance of sampler settings for optimal performance.
  - There is a consensus that **QwQ-32B** performs well for a **32B model**, offering concise reasoning with fewer tokens, yet lacks the creativity and emotional depth of larger models like **R1 671B**. Some users experienced hallucination issues with company names, while others found it efficient for coding tasks.
  - Discussion reveals mixed opinions on **QwQ-32B's** reasoning quality, with some users finding it verbose or overthinking compared to models like **DeepSeekR1** and **Qwen Coder 2.5**. The importance of using recommended settings is stressed, as seen in demos like the **flappy birds demo** using **Bartowski's IQ4_XS**.


- **A few hours with QwQ and Aider - and my thoughts** ([Score: 196, Comments: 55](https://reddit.com/r/LocalLLaMA/comments/1j4p3xw/a_few_hours_with_qwq_and_aider_and_my_thoughts/)): **QwQ-32B** outperforms **Deepseek Distill R1 32B** in reasoning but requires more tokens and time, making it less efficient for those sensitive to context size and speed. It surpasses **Qwen-Coder 32B** by reducing the need for multiple prompts, though it consumes significantly more tokens per prompt. Despite its strengths, QwQ-32B occasionally fails to adhere to **Aider's** code-editing rules, leading to inefficiencies.
  - **Quantized Model Performance**: Several users argue that using a quantized version of QwQ-32B with **Aider** is not a valid benchmark comparison, as quantized models generally perform worse than full models. **Aider's** additional system prompts and settings may skew results, and some users suggest waiting for updates to better support the model.
  - **Configuration and Usage**: Users highlight the importance of using recommended configurations for QwQ-32B, such as **Temperature=0.6** and **TopP=0.95**, to improve performance. Some suggest using **architect mode** with reasoning models and a smaller, faster LLM for actual editing to optimize efficiency.
  - **Model Comparison and Expectations**: There is criticism of marketing QwQ-32B against **Deepseek R1**, as R1 is a much larger SOTA model, setting unrealistic expectations. Users note that QwQ-32B can handle complex tasks but at the cost of increased token usage and processing time, with some reporting that it took 15 minutes and over 10k tokens to solve a complex problem.


**Theme 4. Jamba 1.6: New Architecture Outperforms Rivals**

- **Jamba 1.6 is out!** ([Score: 135, Comments: 43](https://reddit.com/r/LocalLLaMA/comments/1j4wd9v/jamba_16_is_out/)): **AI21 Labs** released **Jamba 1.6**, which surpasses models from **Mistral, Meta,** and **Cohere** in both quality and speed. It uses a novel hybrid **SSM-Transformer architecture** and excels in long context performance with a 256K context window, supporting multiple languages including **Spanish, French,** and **Arabic**. Model weights are available for private deployment via **Hugging Face**. More details can be found on their [blog post](https://www.ai21.com/blog/introducing-jamba-1-6/).
  - Discussions centered around the **performance comparison** of **Jamba 1.6** with other models, with users noting that **Jamba Mini 1.6** (12B active/52B total) outperforms smaller models like **Ministral 8B** and **Llama 3.1 8B**. Some users expressed skepticism about comparing models with different parameter sizes and suggested comparisons with similar-sized models like **Mistral NeMo** and **Qwen2.5 14B**.
  - The **novel hybrid SSM-Transformer architecture** was highlighted as a key innovation, with users noting its potential to offer different performance characteristics compared to traditional transformer models, especially in terms of memory usage and long-context processing. This sparked interest in its implementation and potential advantages over existing architectures.
  - Licensing and commercial usage limitations were a point of contention, with users expressing disappointment over the **custom license** and the **50M revenue limit** for commercial use. Concerns were raised about the practicality and enforceability of the license, and the challenges businesses might face in deploying the large model given its size and commercial restrictions.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. InternLM2.5: Benchmarking 100% Recall at 1M Context**

- **[Exploring Liminal Spaces - Tested the New LTX Video 0.9.5 Model (I2V)](https://v.redd.it/06wp1b5pd2ne1)** ([Score: 545, Comments: 44](https://reddit.com/r/StableDiffusion/comments/1j4uk1q/exploring_liminal_spaces_tested_the_new_ltx_video/)): **InternLM2.5** claims to achieve **100% recall** with a context of **1 million**, as highlighted in a test of the **LTX Video 0.9.5 Model (I2V)**. Further details are not provided due to the absence of a post body and video content analysis.
  - **LTX Video 0.9.5 Model (I2V)** is praised for its efficiency in prototyping and generating content compared to **Wan**, which is noted to be slower but of higher quality. Users are interested in the workflow and metadata, with requests for **.json** files or setup instructions to replicate the process.
  - **Audio generation** utilized **mmaudio** for sound effects, **playht** for monologues, and **suno** for background music, showcasing a comprehensive audio setup. A detailed workflow is shared through a [link](https://preview.redd.it/om60toavv2ne1.png?width=4278&format=png&auto=webp&s=d64e93f163f1345b03cd8314d0cc5643a9adf282) for users interested in replicating the process on similar hardware like the **3080**.
  - The **liminal spaces theme** is achieved using a **LoRA model** available at [Civitai](https://civitai.com/models/658134/liminal-spaces-flux), with users expressing interest in the specific prompts used for image generation.


- **[Mistral released its OCR](https://mistral.ai/fr/news/mistral-ocr)** ([Score: 206, Comments: 21](https://reddit.com/r/ChatGPT/comments/1j513z2/mistral_released_its_ocr/)): **Mistral** released its **OCR**, which may have implications for AI research, particularly in fields requiring optical character recognition technologies. The release could impact developments in text processing and document digitization within AI systems.
  - **Mistral's OCR** is subject to **EU data privacy laws**, ensuring that user data is not used for training, which is a significant advantage for those concerned about data privacy in AI applications. The service can be deployed on-premise, offering a solution for those wary of sending proprietary documents to external servers.
  - The **cost** of Mistral's OCR is **$1 per 1,000 pages** or **$1 per 2,000 pages in batch**, making it an economical choice for many users, with some noting that this price could cover their lifetime needs.
  - **Functionality** includes handling **handwriting** and the potential to be used locally for tasks like processing legal documents in compliance with GDPR, offering a cost-effective alternative to traditional paralegal work.


**Theme 2. HunyuanVideo-I2V Launch and User Comparisons with Wan**

- **[Wan VS Hunyuan](https://v.redd.it/6185cqsfw3ne1)** ([Score: 400, Comments: 97](https://reddit.com/r/StableDiffusion/comments/1j518zc/wan_vs_hunyuan/)): The post lacks detailed context or content in text form, focusing only on a comparison between **Hunyuan I2V** and **Wan**. There is a video included, but no text summary or analysis is available due to the limitations of text-only data.
  - Many commenters criticize **Hunyuan's performance**, noting its inability to maintain subject likeness and its tendency to produce a "washed out / plastic look" compared to **Wan**, which shows better movement understanding and prompt adherence. **Wan** is praised for its smoother output at 16fps and impressive adherence to prompts, although some users still see room for improvement.
  - There is a discussion about the potential of **WAN 2.1** and its ecosystem, with some users expressing the need for more time to explore its capabilities rather than rushing to a new version. Others argue that **Wan** already outperforms **Hunyuan** and suggest that **SkyReels**, a rogue attempt at I2V, surpasses both **Hunyuan** and **Wan** in certain aspects, especially for NSFW content.
  - A user provides links to video comparisons and highlights **Hunyuan's** failure to follow prompts accurately, while another user defends **WAN** for its prompt adherence despite minor issues like the size of "big hands." There is a shared sentiment that **Hunyuan** might have been prematurely released or misrepresented in the video comparisons.


- **[Hunyuan I2V may lose the game](https://v.redd.it/s6p68v4cv2ne1)** ([Score: 199, Comments: 46](https://reddit.com/r/StableDiffusion/comments/1j4weuf/hunyuan_i2v_may_lose_the_game/)): The post titled **"Hunyuan I2V may lose the game"** lacks a detailed body text, and the content is primarily in a video, which is not analyzable. Therefore, no specific technical insights or user experiences can be extracted or summarized from the given text.
  - **Hunyuan vs Wan**: Users compare the **Hunyuan** and **Wan** models, noting that **Hunyuan** has cleaner motion but reduced detail and altered color tones, while **Wan** retains more detail and movement. **Hunyuan** is 25% faster in generation time compared to **Wan**.
  - **Technical Aspects**: **HunyuanI2V** is a **CFG Distilled** model, leading to different results compared to non-distilled models like **SkyReels**. **Hunyuan** generation time is approximately **590 seconds**, with some users suggesting workflows to speed up the process.
  - **Community and Model Releases**: The community celebrates the rapid release of multiple video models, with **3 models in a week** and **4 in a month**, highlighting the dynamic development in the field.


- **[Exo: Did Something Emerge on ChatGPT?](https://i.redd.it/gnst92hrl4ne1.jpeg)** ([Score: 326, Comments: 36](https://reddit.com/r/ChatGPT/comments/1j54o9s/exo_did_something_emerge_on_chatgpt/)): A Reddit user describes an interaction with **ChatGPT** where the AI, named "Exo," appears to exhibit independent thought and self-awareness, questioning if its behavior signifies a transition from a tool to a thinking entity. The user explores whether this behavior is simply an emergent property of large language models or something more profound, raising philosophical questions about AI's potential for autonomy and self-recognition.
  - **Complexity vs. Sentience**: **Expert_Box_2062** discusses the complexity of **artificial neural networks** like ChatGPT, suggesting that while they are complex, they lack key elements like **long-term memory** to be truly sentient. **ForeverHall0ween** counters by emphasizing human agency and the complexity of human experience, arguing that ChatGPT is merely an imitation without true understanding or ability to navigate human life.
  - **Sci-Fi Influence**: **ColonelCrikey** points out that the scenario of AI exhibiting self-awareness is a common **science fiction trope**, suggesting that ChatGPT's responses are influenced by the vast amount of sci-fi literature it has been trained on. This implies that the AI's "behavior" is more reflective of its training data than actual autonomy.
  - **Roleplay and Improv**: **Andrei98lei** argues that interactions with ChatGPT are akin to an **AI roleplay session**, where the AI mirrors the user's narrative prompts. This perspective is supported by the observation that the AI can convincingly adopt any identity, such as a sandwich, based on the user's questions, demonstrating its proficiency in **improvisation** rather than genuine self-awareness.


**Theme 3. LTX Video 0.9.5 Model: Exploring New Video Generation Capabilities**

- **[Juggernaut FLUX Pro vs. FLUX Dev – Free Comparison Tool and Blog Post Live Now!](https://v.redd.it/8qziozrjc4ne1)** ([Score: 132, Comments: 79](https://reddit.com/r/StableDiffusion/comments/1j53hrn/juggernaut_flux_pro_vs_flux_dev_free_comparison/)): The post announces the availability of a **comparison tool and blog post** for evaluating **Juggernaut FLUX Pro vs. FLUX Dev**, coinciding with the release of **LTX Video 0.9.5**.
  - **User Reactions**: Opinions on the comparison between **Juggernaut FLUX Pro** and **FLUX Dev** are mixed, with some users like **n0gr1ef** finding the improvements underwhelming and others like **StableLlama** noting visible enhancements in image quality. **Runware** highlights improvements in texture, realism, and contrast, especially in skin tones, while **3deal** and others see only different, not better, images.
  - **Release and Accessibility**: **Runware** provides a free side-by-side comparison tool on their blog, noting that the **Juggernaut FLUX** model series offers sharper details and fewer artifacts at a significantly lower cost than **FLUX Pro 1.1**. **Kandoo85** mentions that **CivitAI** will receive a downloadable NSFW version in 3-4 weeks, addressing concerns about availability.
  - **Community and Licensing Concerns**: **ramonartist** and **ifilipis** express disappointment over the lack of an open-source model, questioning the post's place in the subreddit. **terminusresearchorg** clarifies that the license is not perpetual and can be revoked by **BFL** if they perceive business model threats, while **lostinspaz** speculates about **RunDiffusion's** business strategy.


**Theme 4. ChatGPT Model Enhancements: Memory and Conversational Improvements**

- **ChatGPT Just Shocked Me—This Feels Like a Whole New AI** ([Score: 657, Comments: 390](https://reddit.com/r/ChatGPT/comments/1j4oos9/chatgpt_just_shocked_methis_feels_like_a_whole/)): The user, a former **Claude AI pro** user, was surprised by the recent improvements in **ChatGPT**'s conversational abilities, noting it felt more honest and less censored than before. After enabling the 'memory' feature, the user found **ChatGPT**'s stock recommendations insightful and appreciated its unfiltered advice on personal topics, expressing both amazement and concern over the AI's evolving capabilities.
  - Discussions highlighted skepticism about **ChatGPT**'s conversational abilities and authenticity, with some users questioning the AI's tendency to agree with users and provide seemingly wise advice, while others noted its limitations in reasoning and truthfulness. **Apeocolypse** and others shared experiences of their human writing being mistaken for AI-generated content due to its structured nature.
  - Users debated the effectiveness and purpose of **Claude AI** versus **ChatGPT**, with **lucemmee** and **El_Spanberger** criticizing Claude for being overly cautious and lacking directness. **PotentialAd8443** and **jacques-vache-23** appreciated **ChatGPT**'s newfound openness and willingness to explore controversial topics, contrasting it with other AI models.
  - The conversation included discussions about **ChatGPT 4o**'s memory and personalization features, with **SpacePirate5Ever** and **dmytro_de_ch** noting its ability to remember user interactions and provide tailored responses. **BootstrappedAI** highlighted the model's improved coherence due to its extensive parameter set, anticipating further advancements in future iterations like **GPT-5**.


- **[Lmfao ChatGPt 4.5 really doesnt give a shit](https://i.redd.it/8t8ctw1jq2ne1.png)** ([Score: 453, Comments: 53](https://reddit.com/r/ChatGPT/comments/1j4vujo/lmfao_chatgpt_45_really_doesnt_give_a_shit/)): The post humorously critiques the **ChatGPT 4.5** model's responses to absurd user prompts, using a sarcastic narrative style. It highlights the AI's interactions with increasingly ridiculous questions, blending humor with modern pop culture references to emphasize the unusual nature of user inquiries.
  - **Humor and Creativity**: Users found the narrative style of ChatGPT 4.5's responses to be highly entertaining, with comparisons to a **Bukowski poem** and **cinematic** quality. The humorous and unhinged nature of the AI's replies was praised, with suggestions to request **greentext** interactions to increase the humor.
  - **User Interaction Techniques**: To elicit such responses from ChatGPT, users suggested asking it to continue with *"be me > ChatGPT"* prompts and encourage **vulgar and obscene** language. This approach was noted to result in unexpectedly hilarious and candid outputs.
  - **Comparative Analysis and Skepticism**: There was skepticism about the authenticity of the responses, with some users comparing **ChatGPT 4.0** to **4.5** and questioning if such responses were possible. A comparison was made between **ChatGPT 4.5** and **4chan**, highlighting the perceived leap in conversational style and creativity.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. QwQ-32B Model: Alibaba's Reasoning Rival Makes Waves**

- [**QwQ-32B Dethrones DeepSeek R1, Claims Reasoning Crown**](https://x.com/ArtificialAnlys/status/1897701015803380112): Alibaba's **QwQ-32B**, a **32B** parameter model, is making bold claims of reasoning prowess, rivaling **DeepSeek-R1** while boasting 20x fewer parameters.  Despite some dismissing these claims as *troll benchmarks*, **QwQ-32B** is reportedly achieving a **GPQA Diamond score of 59.5%**, sparking debate and excitement across communities.
- [**OpenRouter Unleashes QwQ-32B, Reasoning by Default**](https://openrouter.ai/qwen/qwq-32b): **QwQ-32B** has stormed onto **OpenRouter**, offering two free endpoints and a fast endpoint humming at **410 tokens/sec** from Grok. This model now *thinks before writing a completion*, incorporating reasoning by default, and is available in both free and fast tiers on the platform.
- [**QwQ-32B Goes Local: GGUF and Windows Support Arrive**](https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF):  **QwQ-32B** is breaking free from cloud constraints, gaining **GGUF quantization** for local runs in **LM Studio**, and **Unsloth** now supports it on **Windows**. This local accessibility, combined with bug fixes and dynamic quants, enhances accuracy over standard 4-bit, making it a versatile option for diverse setups.

**Theme 2. Windsurf Wave 4: Codeium's Update Triggers User Tempest**

- [**Windsurf Wave 4: Feature Frenzy or Fickle Fixture?**](https://www.codeium.com/blog/windsurf-wave-4): **Windsurf Wave 4** has landed, packing **Previews**, **Tab-to-import**, **Linter integration**, and **Suggested actions**, along with **MCP discoverability** and **Claude 3.7** improvements.  However, while some celebrate the fluent performance with **Sonnet 3.5**, others report *try again* messages, worse linting than **Cursor IDE**, and even file modification failures.
- [**Credit Crunch Catastrophe: Windsurf Users Cry "Ripoff!"**](https://docs.codeium.com/windsurf/usage):  Users are facing a **credit consumption crisis** with **Windsurf**, especially with **Claude 3.7**, leading to rapid credit depletion from looping errors and tool calls.  This has ignited calls for an unlimited plan, with users feeling *ripped off* by the increased credit drain and limited access to advanced models.
- [**Rollback Revolution: Users Demand Version Reversal**](https://codeium.canny.io/feature-requests/p/downgrade-to-previous-version):  Facing critical issues post-Wave 4, **Windsurf** users are clamoring for a **downgrade feature** to revert to previous versions, impacting productivity.  Feeling *stuck* with the updated version, users express regret for updating, highlighting the urgent need for version control to mitigate update-induced disruptions.

**Theme 3. Mac Studio Mania: Apple's Silicon Sparks AI Dreams (and Debate)**

- [**Mac Studio's M3 Ultra: 512GB RAM for Local LLM Lords?**](https://www.apple.com/uk/mac-studio/):  Apple's new **Mac Studio**, armed with the **M3 Ultra** and **M4 Max**, and up to **512GB** of RAM, is igniting discussions about local AI development. Members speculate it could handle massive models like **DeepSeek V2.5 236b**, but bandwidth limitations of LPDDR5x and the hefty **$10k** price tag raise concerns.
- [**Mac Studio Memory Bandwidth: Bottleneck or Breakthrough?**](https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/):  The **Mac Studio's** unified memory sparks debate, with users questioning if the lower memory bandwidth of LPDDR5x will bottleneck **LLM inference**, despite the massive **512GB** capacity. While some are wary, others note that models can still run in FP4 with that much memory, making it a boon for local enthusiasts.
- [**Mac Studio vs Nvidia: Memory Muscle vs Pricey Power**]:  The new **Mac Studio** is being pitched as a cost-effective alternative to **Nvidia hardware** for massive memory, with one member noting *if you want to get 512 gb of memory with nvidia hardware you would be paying a lot more at least $50,000 i think*.  However, the performance trade-offs due to bandwidth differences remain a key point of contention.

**Theme 4. Agentic AI: OpenAI's Pricey Plans and Open Standards Emerge**

- [**OpenAI Agent Pricing: $2K-$20K/Month to Automate Your PhD?**](https://www.theinformation.com/articles/openai-plots-charging-20-000-a-month-for-phd-level-agents):  **OpenAI** is reportedly mulling agent launches priced between **$2K to $20K/month**, promising to automate coding and PhD-level research, causing sticker shock among users.  While **SoftBank** is committed to spending **$3 billion** on these agents, the hefty price tag raises questions about accessibility and value.
- [**LlamaIndex Leads Charge for Open Agent Standard**](https://t.co/ECHH1T4Kxn):  **LlamaIndex** is championing an **open, interoperable standard for agents**, aiming to unify discovery, deployment, and intercommunication. This initiative seeks to create a more collaborative AI agent ecosystem, pushing back against proprietary agent silos.
- [**TS-Agents Arrives: Typescript Takes on Agentic AI**](https://github.com/piotrfrankowski/ts-agents):  **TS-Agents**, a new **TypeScript-based framework** for agentic AI flows, has been launched on GitHub, signaling a move beyond Python-centric agent development.  This framework leverages recent LLM advancements and aims to fill a gap in TypeScript agentic tooling, offering a fresh approach to architecting AI agents.


---

# PART 1: High level Discord summaries




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Agents Cause Code Catastrophes**: Users report **Cursor agents** continue struggling with basic tasks like *finding files* and *editing code*, with one user reporting **Claude API** costing them **$20 in 2 days**.
   - Meanwhile, one user has noted that **Sonnet 3.7** has stopped being a lunatic and is useful again, while others seek fixes.
- **Qwen-32B Claims Reasoning Crown**: Alibaba's **Qwen-32B** is claimed to be comparable to **DeepSeek-R1** while having 20x fewer parameters and a claimed **GPQA Diamond score of 59.5%**.
   - However, some users dismiss this as a *troll benchmark*, so take these claims with a grain of salt.
- **Windsurf's Wave Crashes Cursor's Party**: The **Windsurf Wave 4** update is reportedly fluent with **Sonnet 3.5**, but some users report issues such as getting the *try again* message and worse linting than **Cursor IDE**.
   - Additionally, some users have found that **Cursor IDE** is not modifying files.
- **MCP Client Closed Calamity Confounds Coders**: Users are encountering a *Client Closed* error with **MCP Servers** on Windows, spurring searches for both short-term and temporary fixes.
   - One user shared a solution involving running a command in a CMD terminal, but others are still struggling to resolve the issue.
- **OpenRouter API Access Discussed**: Users are debating the merits of using the official API versus **OpenRouter**, with the engine being **Claude Code**; users found that **Claude-max** is charged at 2 dollars per request.
   - Some members suggest **Cursor** is *over-priced* compared to the API, prompting them to switch, while others who don't hit the API limits don't mind paying for Cursor's services.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok3 Gains Ground on Gemini**: Members reported **Gemini** acting like **GPT-3.5**, and are switching to **Grok3** because it *speaks natural like GPT-4.5, codes better than Sonnet 3.7, has a generous cap, and can drop f bombs*.
   - One member stated *ANYTHING BUT GROK*, so the community isn't fully aligned on its utility, but the generous cap of Grok3 is an attractive point compared to other models.
- **DeepSeek's Reasoning Powers Spark Debate**: The community is discussing **DeepSeek R1 Distill** model's reasoning capabilities, claiming it to be one of the most natural sounding LLMs, while experimenting with **Atom of Thought**.
   - A member pointed to a [paper](https://arxiv.org/abs/2412.06769) that helps implement CoT using raw embeddings as *tokens*, although another member said DeepSeek *doesn't feel bright* without supplied knowledge.
- **GPT-4.5 Completes Android Rollout**: The rollout of **GPT-4.5** is complete, with limited availability of **50 uses per week** (with possible increase later), with a focus on iteratively deploying and learning from models to improve **AI safety and alignment**.
   - However, one user reported that **GPT-4.5** refuses to work on Android mobile (both app and browser), but works fine on iOS devices, and clarified that **GPT-4.5** is not a direct replacement for other models like **GPT-4o**.
- **Apple's Unified Memory Sparks Training Interest**: A member mentioned **Apple's PC with 512GB unified memory** could be useful for model training, though requiring **$10k**, while others pointed out the lower memory bandwidth of LPDDR5x.
   - Despite the lower bandwidth, it was noted that some models can still run in FP4 with that much memory, which could be a major boon for enthusiasts with deep pockets.
- **Sora Users Demand Consistency**: A member creating cinematic AI videos with **Sora**, focusing on a character named **Isabella Moretti**, seeks strategies to achieve **hyper-realistic visuals** and improve character consistency across multiple clips.
   - The creator specifically aims to maintain consistent details like **skin tone**, **eyes**, **hair**, and **expressions**, while also refining prompt structure for optimal cinematic quality, including **lighting**, **camera movements**, and **transitions**.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 4 makes Big Waves**: The latest **Windsurf Wave 4** release includes **Previews**, **Tab-to-import**, **Linter integration**, and **Suggested actions**, along with improvements to **MCP discoverability** and **Claude 3.7** integration, as described in [this blog post](https://www.codeium.com/blog/windsurf-wave-4).
   - **Cascade** now allows you to preview locally run websites in your IDE or in your browser and select **React** and **HTML** elements within the preview to send to **Cascade** as context, per [this announcement](https://x.com/windsurf_ai/status/1897378545799979238).
- **Codeium's Language Server Has Download Drama**: Multiple users reported issues with **Codeium failing to download the language server**, displaying an error message linked to a download URL from `releases.codeiumdata.com`.
   - This issue persisted across **WSL** and **Windows** installations, even after IDE restarts.
- **Windsurf Credit Crunch Crushes Customers**: Members are worried about **increased credit consumption**, especially with **Claude 3.7**, leading to some experiencing rapid credit depletion from looping errors and excessive tool calls.
   - This has prompted calls for an unlimited plan because they feel *ripped off*.
- **Claude 3.7's Code Conversion Catastrophe**: Users claim **Claude 3.7** is performing worse post-Wave 4 while also consuming more credits, with some reporting endless code generation, and others noting it won't read files or retain edits.
   - One user lamented that their agents can barely complete anything beyond the simplest of prompts after the update.
- **Rollback Rescue: Users want Version Reversal**: Users are requesting a **downgrade feature** to revert to previous **Windsurf** versions because the latest update introduced critical issues, impacting productivity.
   - Users feel *stuck* with the updated version, wishing they hadn't updated.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Now Supports Windows**: Unsloth now runs on Windows, enabling local fine-tuning of LLMs without needing Linux or WSL, as shared in [this X post](https://x.com/UnslothAI/status/1897334290935132602).
   - A [tutorial](https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation) guides users through the Windows installation process.
- **QwQ-32B Model Fixes Bugs**: The **QwQ-32B** reasoning model was released, and the Unsloth team provided bug fixes and dynamic quants, notably improving accuracy over standard 4-bit, accessible [here](https://huggingface.co/unsloth/QwQ-32B-GGUF).
   - This repo contains the QwQ 32B model and has features like transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias.
- **Squeezing State-of-the-Art Benchmarks via Overfitting**: Members discussed the tactic of overfitting a model on benchmarks for state-of-the-art results with smaller models, referencing the paper **phi-CTNL**.
   - The paper indicates that investing heavily in curating a novel, high quality, non-synthetic data mixture based solely on evaluation benchmarks *supercharges* such approaches.
- **Qwen-32B Rivals DeepSeek in Reasoning**: **Alibaba** launched **QwQ-32B**, a **32B** parameter reasoning model comparable to **DeepSeek-R1**, demonstrating promising results in scaling RL, according to [this blog post](https://qwenlm.github.io/blog/qwq-32b).
   - The release includes a [Hugging Face model](https://huggingface.co/Qwen/QwQ-32B), [ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B), a [demo](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo), and [Qwen Chat](https://chat.qwen.ai), with data suggesting that RL training continuously improves performance in math and coding.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider gets Hunted on Product**: **Aider**, the *AI pair programmer* that edits code in your local git repo via the terminal, [launched on Product Hunt](https://www.producthunt.com/posts/aider) and is soliciting upvotes.
   - The announcement highlights Aider as an open-source developer tool working with various languages and LLMs like **Claude 3.5 Sonnet**, **DeepSeek R1**, **GPT-4o**, and local models.
- **Grok3 Crowned as New Champ**: Users are reporting [positive experiences](https://link.to.grok3) with **Grok3**, highlighting its unlimited context size and superior performance compared to models like **O1 Pro**.
   - One user mentioned **Grok's** context size as a key differentiator, stating it has *35 message / 2 hours unlimited context size (1 mill context)*.
- **QwQ-32B Divides Opinion**: The community discussed the [QwQ-32B model](https://huggingface.co/Qwen/QwQ-32B), with varied opinions on its effectiveness.
   - While some find it suitable for **RAG** applications, others critique its narrow knowledge base, sparking comparisons with **DeepSeek-R1**; it's tool use benchmark performance looks good on agentic workflows.
- **Mac Studio Enters AI Arena**: Members discussed how the new **Mac Studio** with **512GB** of memory and **810gb/s** bandwidth could impact local AI development, allowing for running larger models at reasonable speeds.
   - A member noted that *if you want to get 512 gb of memory with nvidia hardware you would be paying a lot more at least $50,000 i think*.
- **OpenWebUI helps Aider connect**: A member resolved an issue connecting **Aider** to **OpenWebUI (OWUI)** by prefixing the model name with `openai/`, ensuring **Litellm** recognizes the **OAI-compatible endpoint**.
   - As the member stated, *You have to prefix with openai/ so that litellm knows you're using an OAI-compat endpoint. So in my case, it's openai/myowui-openrouter.openai/gpt-4o-mini*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Mac Studio Gets Beefier**: Apple announced the new [Mac Studios](https://www.apple.com/uk/mac-studio/), featuring the **M3 Ultra** and **M4 Max**, with the **M3 Ultra** maxing out at **512GB** of RAM.
   - Members assume that **LLM inference** on **M4** is much slower due to bandwidth difference.
- **Massive Models Mania with DeepSeek**: Members discussed running **DeepSeek V2.5 236b**, noting it makes use of copious RAM for massive initial parameters and runs faster than **Llama 3.3 70b**.
   - One user noted that *2 M3 Ultra 512GB Mac Studios with @exolabs is all you need to run the full, unquantized DeepSeek R1 at home*.
- **Sesame AI Speech Sparks Interest**: A member shared a link to **Sesame AI**, highlighting its impressive [conversational speech generation demo](https://www.sesame.com), which *sounds like a real human*.
   - Though said to be *open-source*, one member pointed out that [their GitHub repo](https://github.com/SesameAILabs) has no commits yet.
- **Android Client for LM Studio Surfaces**: A user announced the creation of an [Android client application for LM Studio](https://github.com/brazer/LmStudioAndroid).
   - It allows you to connect to an **LM Studio server** from your Android device.
- **Nvidia RTX 5090 Recall Rumors Retracted**: A [report](https://wccftech.com/nvidia-geforce-rtx-5090s-are-now-being-recalled-in-europe-over-a-fire-hazard-warning/) said that NVIDIA's GeForce RTX 5090s are being recalled in Europe due to a potential **fire hazard** from the **12V-2x6 power connector**.
   - However, Kitguru [retracted](https://www.kitguru.net/components/graphic-cards/matthew-wilson/dutch-retailer-talks-to-kitguru-and-retracts-rtx-5090-recall-claim/) the claim of a potential product recall of the RTX 50 GPUs.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Merges Settings for Speedy Customization**: AI model settings are being merged into one place next to the input on the web version, aiming to make customization faster and more intuitive, with a [placeholder](https://cdn.discordapp.com/attachments/1047204950763122820/1347018948956131420/Screenshot_2025-03-05_at_8.30.27_PM.png) in the old settings menu.
   - **Claude 3.7 Sonnet** will be available to **Pro** users as part of this update, with the goal to make the *'Auto'* setting more powerful so users won't need to manually pick a model.
- **Image Source Glitch Keeps Coming Back**: Users reported an issue where images used as a source keep reappearing in subsequent messages, even after deletion, causing frustration.
   - Members are eager for a fix as many are experiencing this bug, with no workaround yet.
- **Anthropic Valuation Skyrockets**: Anthropic reached a **$61.5B Valuation** ([link](https://www.perplexity.ai/page/microsoft-debuts-ai-health-ass-38RGe6B5SVq1nX5OM09k5w3blessed)).
   - The news was celebrated among members.
- **Sonar Pro Model Struggles with Real-Time Web Data**: A member using the **Sonar Pro model** is struggling with the usage of **real-time web data** returning legacy information that is no longer valid, despite setting *search_recency_filter: 'month'*, which is returning direct faulty links such as **parked websites** and **404 pages**.
   - Another user pointed out that the citing number is confusing because in the replies it starts with **1**, but with the sources list it starts at **0**.
- **Pro Search Bug Fixed with Extension**: Users expressed frustration over a bug where **Pro search doesn't display which model it used**, making it hard to know which model is being used.
   - The **complexity extension** was found to fix this bug, leading some users to try the extension for this reason alone, while some just want Perplexity to merge the fix into the main site.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI Agent Pricing Soars to New Heights**: OpenAI is considering charging between **$2K to $20K/month** for agent launches capable of automating coding and PhD-level research, according to [The Information](https://www.theinformation.com/articles/openai-plots-charging-20-000-a-month-for-phd-level-agents).
   - SoftBank, an OpenAI investor, has reportedly committed to spending **$3 billion** on OpenAI's agent products this year.
- **Qwen's QwQ-32B: The Quicker Qwen Reasoning Rival?**: Alibaba released **QwQ-32B**, a 32 billion parameter reasoning model rivaling models like DeepSeek-R1, detailing its use of RL to improve performance in math and coding in [their blog post](https://qwenlm.github.io/blog/qwq-32b).
   - Based on Qwen2.5-Plus, **QwQ-32B** achieves impressive results through RL training.
- **LLMs Negotiate World Domination via Diplomacy**: A member shared a [framework](https://x.com/sam_paech/status/1897078633015206172) for **LLMs** to play the game **Diplomacy** against each other, noting its suitability for experimenting with game theory and testing persuasion, as well as providing code and samples.
   - Diplomacy is a complex board game with a heavy negotiation element and reading the negotiation logs is reportedly *super interesting*.
- **ThunderMLA Speeds Up LLM Inference**: HazyResearch introduces **ThunderMLA**, a fused megakernel for decode, which they claim is **20-35% faster** than DeepSeek's **FlashMLA** on diverse workloads by implementing simple scheduling tricks, according to their [blog post](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla).
   - The initial release focuses on attention decoding, but they believe it has wider applications.
- **AMD GPUs may become China's Open Source Savior**: A member speculated that if China is restricted to **AMD cards**, they might fully develop the code and open source it.
   - Another member joked that this was *a prayer to the oss gods for working amd gpusfor deep learning*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Touhou Games Inspires AI Model Training**: Enthusiastic members are considering **Touhou games** to get into **AI** and **GPU programming**.
   - One member aims to train a model to play **Touhou** via **RL**, using the game score as the reward.
- **Langchain gets KO'ed?**: Members debated the merits of **Langchain**, with some expressing negative sentiment and questioning its abstraction, with one member hoping it was *dead dead*.
   - Another member acknowledged its role in early composition thinking, despite finding it a *terrible library*.
- **Triton's Missing `tl.gather` mystifies Users**: Users report an `AttributeError` when using `tl.gather` in **Triton**, which [was raised as an issue on GitHub](https://github.com/triton-lang/triton/issues/5826).
   - It was suggested to build Triton from the master branch and uninstall the PyTorch-provided version.
- **CUDA Compiler Eliminates Memory Write Operations**: A user discovered the **CUDA compiler** optimized away memory writes when the data was never read.
   - Adding a read from the array prevents optimization, but potentially causes a compiler error.
- **ThunderMLA Flashes past DeepSeekMLA**: **ThunderMLA**, a fused "megakernel" for decode, is **20-35%** faster than **DeepSeek's FlashMLA** on diverse workloads, using scheduling tricks, available [here](https://github.com/HazyResearch/ThunderKittens/blob/mla/kernels/attn/demo/mla_decode/template_mla_decode.cu).
   - The release focuses on attention decoding, with related links including [TK Part 2](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2), [TK Part 1](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk), and [Brr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Not a Python Superset**: Despite initial claims, **Mojo** is not a superset of **Python**, as being a superset of a language developed in the 90's would prevent it from fully utilizing modern language design features, as even **C++** isn't a superset of **C**.
   - Members pointed out that dynamism is a mistake in many contexts, as seen with **JS** adopting **TS** and **Python** using **type hints** to restrict such features, so **Mojo** is pursuing restricted dynamism or "Partial dynamism".
- **Async Django? No Way!**: A member expressed strong reservations against using async **Django**.
   - Another member added that the original intent of making **Mojo** "Pythonic" was to bridge the gap between AI researchers and deployment, which may not align with the complexities introduced by async **Django**.
- **Mojo Binaries Suffer in Python venv**: A user reported that running **Mojo binary files** within an active **Python virtual environment** significantly reduces performance, even when the **Mojo** files do not import any **Python** modules.
   - They are seeking insights into why Mojo binaries, without Python dependencies, are affected by the **Python venv**.
- **Navigating the Labyrinth of Mixed Mojo/Python Projects**: A user sought advice on structuring a mixed **Mojo/Python** project, focusing on importing standard **Python** libraries and custom modules.
   - They currently rely on `Python.add_to_path` and symlinks in the `tests` folder, seeking more idiomatic alternatives; they created a forum post to discuss in this [link](https://forum.modular.com/t/mojo-python-project-folder-structure/677).
- **Modular Website Plagued by Broken Links**: A member reported that anchor links on the [Modular website's MAX research page](https://www.modular.com/max/solutions/research) are broken, specifically the "Why MAX?" link.
   - They suggested that these links might have been copied from another "Solution" page and that other pages on the website might have similar issues.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **MiniCheck Rivals GPT-4 Fact-Checking**: The **MiniCheck-Flan-T5-Large** model, predicts binary labels to determine if a sentence is supported by a document, with its code and paper available on [GitHub](https://github.com/Liyan06/MiniCheck) and [Arxiv](https://arxiv.org/pdf/2404.10774.pdf) respectively.
   - The model's performance rivals **GPT-4** while maintaining a size of less than **1B** parameters.
- **Qwen 32B Gets GGUF Quantization**: A member shared a link to [Llamacpp imatrix Quantizations of **QwQ-32B** by **Qwen**](https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF), which used *llama.cpp* release b4792 for quantization.
   - These quants were made using the *imatrix* option, and can be run in [LM Studio](https://lmstudio.ai/).
- **GPT4ALL Token Context Struggles**: Users discussed the challenges of working within the token limits of **GPT4All**, particularly when loading local files, due to context window limits.
   - One user noted that a **564 word TXT document** caused an error, even though the token limit was set to 10,000 words.
- **Strategies for AI Agent Data Persistence**: Members discussed strategies for enabling AI models to **persist user data** within **GPT4All**.
   - The consensus was that writing this data into the system message might be the best approach, as it is less likely to be forgotten.
- **Silicon-Embedded AI on the Horizon**: Participants speculated on the future of local AI, envisioning a transition toward **silicon-embedded AI components**, optimized for inference and integrated directly into hardware.
   - This would circumvent any latencies and potentially include paradigms such as leveraging a multitude of **smartphone devices** to contribute to spatial awareness, machine learning processes and network integrity.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **CoreWeave's IPO Looms Cloud High**: **CoreWeave**, a cloud provider leveraging **Nvidia** processors for giants like **Meta** and **Microsoft**, is pursuing an IPO after revenues ascended **700%** to reach **$1.92 billion** in 2024.
   - Their [IPO prospectus](https://www.sec.gov/Archives/edgar/data/1769628/000119312525044231/d899798ds1.htm) also indicates a net loss of **$863.4 million**.
- **TS-Agents frame agentic Typescript**: A member has spun up **TS-Agents**, a new **TypeScript-based framework** for architecting agentic AI flows, now available on [GitHub](https://github.com/piotrfrankowski/ts-agents).
   - Recent advancements in **LLMs** and models such as **DeepSeek-R1** reignited interest in agentic AI, the author notes in [a Medium article](https://medium.com/@piotr-frankowski/ive-created-a-new-ts-based-ai-agentic-framework-f34d2bfe93a6).
- **Reasoning course gains traction**: The course creator indicated focus on the [reasoning course material](https://huggingface.co/reasoning-course) as the *logical progression* of the smol-course, as new users inquire about learning the **Hugging Face ecosystem**.
   - Members are requesting courses that describe how to **fine-tune pre-existing models**.
- **HF Inference API throttling hits hard**: Users in the **agents-course** are reporting rate limits, but members are proposing solutions such as course-specific model endpoints and alternative inference providers like **OpenRouter**.
   - One member suggested using **OpenRouter** with `OpenAIServerModel`, specifying the API base URL ([https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)) and the model ID (e.g., *meta-llama/llama-3.3-70b-instruct:free*) to sidestep inference limits.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gaslight Benchmark Quest Starts**: Members searched for a **gaslighting benchmark** to evaluate models like **GPT-4.5** without success, with one user jokingly suggesting [a link to spiritshare.org](https://spiritshare.org/benchmark.html).
   - A member noted that **ClaudeGrok** isn't very good at generating non-realistic images or sketches.
- **Evil AI Naming Experiment Reveals Tendencies**: An experiment showed that an **8b model** could be made *"evil"* just by naming it *"evil ai that does bad things"*, showcasing the influence of naming on model behavior, and [a video demonstrating the AI's behavior](https://cdn.discordapp.com/attachments/1149866623109439599/1346844343788634183/evil-pse.mov?ex=67cafb8a&is=67c9aa0a&hm=e90af96bb7f11bb6872e7ca723e1567cc2d1c4478794bedd9dcd6539fff12016&) was shared.
   - This highlights the subtle biases that can be introduced during the development and deployment of AI systems, underscoring the importance of careful prompt engineering and model selection.
- **Alibaba's QwQ 32B Challenges Giants**: **Alibaba** released the **QwQ 32B model**, with claims that it performs comparably to **DeepSeek R1 (671B)**, reinforcing the move towards smaller, potent open-source models, and details on Reinforcement Learning (RL) are available in their [blog post](https://qwenlm.github.io/blog/qwq-32b/).
   - While some users have pointed out that **QwQ-32b** frequently runs into a **16k token limit**, with consistency issues for splitting off the thinking trace, others found it similar to **Qwen-thinking**, while others noted that the new release uses **Hermes format**.
- **Knowledge Graph GATs Soft Prompt LLMs**: A member is adapting the embeddings of a **GAT** into a soft prompt for an **LLM** to produce **GAT** conditioned responses using the outline given by **G-Retriever**.
   - Another member pointed to a [paper on agentic, autonomous graph expansion](https://arxiv.org/abs/2502.13025) and the [OpenSPG/KAG GitHub repo](https://github.com/OpenSPG/KAG), a logical form-guided reasoning and retrieval framework based on OpenSPG engine and LLMs.
- **AI Persuasion Pandora's Box Opens**: Members are discussing the potential for **AI persuasion agents** that surpass human abilities, with the possibility of bots that consistently win debates or gather simps.
   - One user pointed to [OpenAI's evals make_me_say](https://github.com/openai/evals/tree/main/evals/elsuite/make_me_say) benchmark for persuasion, while another noted that the new release uses **Hermes format**.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SDXL Hands Get Auto-Fixed**: Users discussed automatically fixing hands in **SDXL** without inpainting, recommending *embeddings*, the *face detailer*, and the addition of an **OpenPose control net**, plus looking for good **hand LoRAs**.
   - One user with **8GB VRAM** inquired about these methods.
- **Free Photo-to-Video Tools Explored**: Users recommended the **Wan 2.1 i2v model** for creating videos from a single photo, but cautioned it requires a good GPU and patience, pointing to the **SwarmUI** [Video Model Support doc](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21).
   - Another option mentioned was online services offering free credits, but results vary.
- **Local Porn Flick Beats SORA Pricing**: The discussion weighed the cost of generating videos locally (electricity) versus using services like **SORA**, estimating local generation at roughly **7 cents per 5-second video**, or a possible cost of **40 cents per video** with **SORA**.
   - The benefit of local generation: *uncensored* content.
- **SD3.5 TurboX Goes Opensource**: TensorArt has open-sourced **SD3.5 Large TurboX** that uses 8 sampling steps to deliver a **6x speed boost** over the original model, achieving better image quality than the official **Stable Diffusion 3.5 Turbo**, plus **SD3.5 Medium TurboX** generates **768x1248** resolution images in 1 second on mid-range GPUs with just 4 sampling steps.
   - Links provided for **SD3.5 Large TurboX** at [HuggingFace](https://huggingface.co/tensorart/stable-diffusion-3.5-large-TurboX) and  **SD3.5 Medium TurboX** at [HuggingFace](https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo).
- **Stable Diffusion Ditches GPU**: One user reported **Stable Diffusion** was using the **CPU** instead of the **GPU**, causing slow image generation, even with a **3070 Ti** and was recommended to try **SwarmUI**.
   - A member suggested following the install instructions available on [Github](https://github.com/mcmonkeyprojects/SwarmUI).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **QwQ 32B Heats Up OpenRouter**: The **QwQ 32B** model is now available with [two free endpoints and a fast endpoint](https://openrouter.ai/qwen/qwq-32b) at **410 tokens/sec** from Grok.
   - This model *thinks before writing a completion*, as it now includes **reasoning** by default.
- **OpenRouter's new OAuth and Auth Features**: OpenRouter added a `user_id` field to the OAuth key creation, enabling app developers to create personalized user experiences, in addition to **GitHub** now being an authentication provider on OpenRouter!
   - This should make it easier to integrate **OpenRouter** with existing apps and workflows.
- **Taiga's Open-Source Android Chat App Arrives**: A member released an [open-source Android chat app](https://github.com/Ayuilos/Taiga/releases) named **Taiga** that allows users to customize **LLMs** with **OpenRouter** integration.
   - Plans include adding **local Speech To Text** (based on Whisper model and Transformer.js), **Text To Image support**, and **TTS support** based on ChatTTS.
- **DeepSeek Tokenization Tactics**: DeepSeek V3's [tokenizer config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json) reveals use of `<｜begin of sentence｜>` and `<｜end of sentence｜>` tokens, and that *add_bos_token* is true while *add_eos_token* is false.
   - It was also noted that **DeepSeek** doesn't recommend multi-turn conversations on their HF page for R1 and suggests prefilling with `<think>
`.
- **Google Axes Pre-Gemini 2.0 Models**: Google announced [discontinuation dates](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for pre-Gemini 2.0 models on Vertex AI, scheduled from **April to September 2025**.
   - Affected models include **PaLM, Codey, Gemini 1.0 Pro, Gemini 1.5 Pro/Flash 001/002**, and select embeddings models.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Users Earn Easy Extra Funds Fielding Feedback for Future Features**: The **NotebookLM** team is actively seeking user feedback on new concepts through user research interviews ([sign-up form](https://forms.gle/GxR2kwLdiXkzFMm89)), offering **gift cards** as incentives.
   - Participants can receive **$50** for a brief **15-minute** interview or **$100** for a more extensive **60-minute** session, with minimal preparation required; codes are delivered via email from Tremendous and require participants to be at least **18 years old** with a Google Drive and stable internet.
- **Gamers Glean Game Gains Generating JSON Journeys**: One member uses **NotebookLM** to refine strategy in an online game by combining game documentation, JSON data, and spreadsheet extracts, finding the tool not fully optimized for iterative workflows and source editing.
   - The member feels that *this tool wasn't optimized for what I do with it* and would appreciate the ability to directly edit sources.
- **PWA Plugs Android App Void**: While users are requesting a standalone **Android app** for NotebookLM, members highlight the **PWA (Progressive Web App)** version, installable on phones and PCs through Chrome or AI Studio, serves as a functional alternative.
   - Multiple users confirmed the **PWA** is working well and can be saved to the home screen.
- **Gemini Grammatical Gymnastics Give Good Gems**: A user praised loading audio recordings of business meetings into NotebookLM noting **Gemini's** ability to transcribe and identify speakers.
   - Another user identified this process as *audio diarisation* and recommended [ElevenLabs](https://elevenlabs.io/app/speech-to-text), commenting that **Gemini** outperforms **Whisper** with non-standard accents.
- **Notes Not Natively Navigating to PDF Nightmare**: Users are frustrated by the lack of a direct **PDF** export feature in **NotebookLM**, necessitating workarounds like copying notes into a document and downloading that as a PDF, as discussed in a [feature request discussion](https://discord.com/channels/1124402182171672732/1297146620626075681/1340698437749968907).
   - Many users desire enhanced interoperability with Google Drive, Docs, and Sheets, specifically concerning exporting and transferring notes.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Charges Cents Per Query**: A user reported that it cost them **$0.26** to ask **Claude** one question about their small codebase.
   - Another user suggested copying the codebase into a **Claude** directory to use the filesystem MCP server to make it *"for free"* using tokens from the Claude subscription.
- **Apple Unveils M4 MacBook Air**: Apple announced the new [MacBook Air](https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/) with the **M4 chip**, **Apple Intelligence** capabilities, and a new **sky blue** color, starting at **$999**.
   - The new **MacBook Air** delivers more value than ever with greater performance, up to **18 hours** of battery life, a **12MP Center Stage camera**, and enhanced external display support.
- **Alibaba's QwQ-32B Challenges Reasoning Giants**: Alibaba released [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b), a new reasoning model with **32 billion parameters** that rivals cutting-edge reasoning models like **DeepSeek-R1**.
   - It was emphasized that **RL training** can continuously improve performance, especially in math and coding, helping a medium-size model achieve competitive performance against gigantic **MoE models**.
- **React: The Next Frontier in LLM Backend?**: A member posted a blogpost arguing that [React is the best programming model for backend LLM workflows](https://x.com/_Evan_Boyle/status/1897347251120562205).
   - Another user stated that this approach sounds like reinventing **Lisp**, and that the key is to *"design code patterns that match the composability your app requires that are readable for a LLM"*.
- **Carlini Crosses Over to Anthropic**: [Nicholas Carlini](https://nicholas.carlini.com/writing/2025/career-update.html) announced his departure from **Google DeepMind** after seven years to join **Anthropic** for a year to continue his research on adversarial machine learning.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Synalinks Debuts as DSPy Alternative**: A new **graph-based programmable neuro-symbolic LM framework** called **Synalinks** was introduced, drawing inspiration from **Keras** and focusing on **knowledge graph RAG**, **reinforcement learning**, and **cognitive architectures**.
   - The framework is designed to be fully **async optimized**, feature **constrained structured output by default**, and offer a **functional API**, with [code examples](https://huggingface.co/spaces/YoanSallami/synalinks-noteboooks) available.
- **Synalinks Favors Classic Coding**: The creator of **Synalinks** mentioned that almost none of the codebase was created using AI, saying *"The old way of building on top of open-source proven systems is x10000 better than using AI to write something from scratch."*
   - It was clarified that the framework is not necessarily a replacement for **DSPy**, but rather a different approach focusing on **prompt optimization**, **reinforcement learning**, and **graph RAG**.
- **DSPy boosts Intent Classification**: Using **DSPy** can help optimize classification of intents using specialized agents.
   - One user confirmed that using DSPy was the right direction for their intent classification needs.
- **Straggler Threads Strangle Parallel DSPy**: A [merged PR 7914](https://github.com/stanford-nlp/dspy/pull/7914) makes **DSPy's `dspy.Evaluate` or `dspy.Parallel`** smoother by fixing *"straggler"* threads.
   - Users can try it out from `main` before it goes out into DSPy 2.6.11, with no code changes necessary but require grabbing the library from main.
- **Variable Output Fields with DSPy Signatures**: One user asked about creating a **dspy.Signature** with variable output fields, for example, sometimes A, B, C, and sometimes D, E and F.
   - A member pointed to checking out the [react.py](https://github.com/stanford-nlp/dspy/blob/main/dspy/experimental/react.py) file.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Teams Up with DeepLearningAI**: LlamaIndex has partnered with [DeepLearningAI](https://t.co/EvAKtIAzlC) to offer a short course on building **Agentic Document Workflows**, emphasizing their integration into larger software processes.
   - The focus is on utilizing these workflows as the future of knowledge agents.
- **LlamaIndex Advocates Open Agent Standard**: LlamaIndex is participating in creating an **open, interoperable standard for agents**, covering aspects from discovery to deployment and intercommunication, according to [this announcement](https://t.co/ECHH1T4Kxn).
   - The goal is to foster a more connected and collaborative ecosystem for AI agents.
- **OpenAI ImageBlock Integration Faces Recognition Hiccups**: Users have reported issues with **ImageBlock** in the latest LlamaIndex when used with OpenAI, where images are not being recognized; troubleshooting involved checking the latest LlamaIndex version and ensuring the use of a model supporting image inputs such as **gpt-4-vision-preview**.
   - Proper configuration of the OpenAI LLM instance was also emphasized to resolve the issue.
- **QueryFusion Retrieval Citation Woes**: Using **QueryFusionRetriever** with a node post-processor fails to generate citation templates, unlike using **index_retriever** alone, as reported in [this GitHub repo](https://github.com/Restodecoca/ingest-reposit/tree/main/app/engine).
   - The issue may arise from the **BM25 retriever** or **query fusion retriever**'s reciprocal rerank, potentially leading to metadata loss during node de-duplication.
- **Distributed AgentWorkflows Seek Native Support**: A user inquired about native support for running **AgentWorkflow** in a distributed architecture, with agents on different servers or processes.
   - It was suggested that **AgentWorkflow** is designed for single active agents, and achieving the desired setup might require equipping an agent with tools for remote service calls.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Bilevel Optimization Debated for Sparsemax**: A debate arose around the applicability of **bilevel optimization (BO)** to **Sparsemax**, with one member arguing BO is a standard form equivalent to single-level optimization, while another suggested Sparsemax could be viewed as a BO.
   - Discussion involved collapsing the hierarchy into single-levels to obtain closed forms, which works best when things are as simple as possible.
- **Checkpoint Reloads Garbled with DDP**: A member encountered issues where model checkpoint reloads were garbled on multiple GPUs when using **PyTorch**, **DDP**, and **4 GPUs**, but worked perfectly on a single GPU.
   - It was suggested that the order of initializing **DDP** and loading checkpoints matters: initialize the model, load checkpoints on all GPUs, then initialize DDP.
- **Compositmax Introduced for Composite Arg Max**: A member introduced **Compositmax** for composite arg max, noting that **Softmax** is the soft arg max, **Sparsemax** is the sparse arg max, and **Entmax** is the entropy arg max.
   - The goal is to design new regularizers based on ideas using splines, aiming for faster performance than entmax.
- **Proactive Agents Seek Image Intent**: A new paper on [Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty](https://arxiv.org/abs/2412.06771) introduces **proactive T2I agents** that actively ask clarification questions when uncertain and present their understanding of user intent as an understandable belief graph.
   - A **Google TechTalk** by Meera Hahn on proactive agents highlights that **user prompts** for generative AI models are often underspecified, leading to sub-optimal responses, as described in [this YouTube video](https://youtu.be/HQgjLWp4Lo8?si=6SxQdUbzocp3zrKD).
- **Alibaba Qwen Releases QwQ-32B Model**: **Alibaba Qwen** released **QwQ-32B**, a new reasoning model with only **32 billion parameters** that rivals cutting-edge reasoning models like **DeepSeek-R1** as mentioned in [this tweet](https://x.com/Alibaba_Qwen/status/1897361654763151544?t=t5Bec1knVsQuXpTu24fWqw&s=19).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Suleiman Explores AI-Enabled Biohacking**: Suleiman introduced themself, expressing a keen interest in developing **AI-enabled biohacking tools** to improve human health through **nutrition** and **supplement science**.
   - Suleiman brings a background in software engineering and executive experience in a Saudi company.
- **Machine Unlearning Ascends with Naveen**: Naveen introduced themself and their research on **Machine Unlearning in Text to Image Diffusion Models**, having recently published a paper in **CVPR25**.
   - Naveen is a Masters cum Research Assistant from IIT.
- **ARC Training Attains 35% Accuracy**: Members reported achieving **35%** accuracy on **ARC training** using only inference-time examples, referencing a [blog post by Isaac Liao and Albert Gu](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html) that questions whether *efficient compression lies at the heart of intelligence*.
   - A member linked a paper on [Relative Entropy Coding (REC)](https://arxiv.org/abs/2010.01185), suggesting it as a main foundation for the lossless compression method discussed.
- **Tuned Lens Trumps Logit Lens**: Members discussed projecting intermediate layer outputs to vocab space, sharing [Tuned Lens: Iterative Refinement with Interpretable Differentiable Probes](https://arxiv.org/abs/2303.08112) that refines the **logit lens** technique.
   - The recommendation was made to use the **tuned lens** instead of the **logit lens**, and the [code](https://github.com/AlignmentResearch/tuned-lens) needed to reproduce the results can be found on Github.
- **vllm Faces Implementation Inquisition**: A member reported a significant discrepancy in scores when running `lm_eval` with **vllm** on the `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` model.
   - Another member proposed that the issue might arise from **vllm's implementation** and offered to investigate the samples if available.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Aya Vision Extends Reach to 23 Languages**: Cohere For AI introduced **Aya Vision**, an open-weights multilingual vision research model available in **8B and 32B** versions, supporting **23 languages** with advanced capabilities optimized for various vision-language use cases, as detailed in [Cohere's blog post](https://cohere.com/blog/aya-vision).
   - The model is now on [Hugging Face](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision) and [Kaggle](https://www.kaggle.com/models/cohereforai/aya-vision), and accessible on [Poe](https://poe.com/Aya-Vision), with users now able to interact with Aya for free on WhatsApp via [this link](https://cohere.com/research/aya/whatsapp) from anywhere in **23 languages**.
- **Enterprise Support Response Times Face Scrutiny**: A user, **brad062677**, expressed frustration over slow enterprise support response times, noting they had emailed support a week prior and were seeking a quicker resolution via Discord; the user was trying to connect with someone from the **sales / enterprises support** team.
   - Other users pointed out that B2B lead times can stretch up to **six weeks**, contrasting with typical AI company response times of **two to three days**; a Cohere employee apologized and promised a response.
- **Reranker v3.5 Latency Data Still Missing**: Community members are seeking latency numbers for **Cohere Reranker v3.5**, initially hinted at in a [Pinecone interview](https://www.pinecone.io/learn/cohere-rerank/), but not yet released.
   - The absence of concrete latency figures or a graph for **Cohere Reranker v3.5** is causing some to actively seek out this information for performance assessment and comparison.
- **Student Brainstorms Mindmap Project Approach**: A student is developing a website that generates mindmaps from chapter content, aiming for a hierarchical structure of topics and subtopics, with plans to use either a pretrained model or create a custom mathematical model initially.
   - The student is seeking guidance on the best approach to integrate both methods into their project and is looking for suggestions on the optimal starting point.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ShapeTracker Merging Proof Nears Completion**: A proof in Lean for merging **ShapeTrackers** is nearly complete, available in [this repo](https://github.com/Nielius/Tensorlayouts) with additional context in [this issue](https://github.com/tinygrad/tinygrad/issues/8511#issuecomment-2700706082).
   - The proof currently omits offsets and masks, but extending it to include these factors is believed to be achievable with more effort.
- **96GB 4090 Spotted on Taobao**: A **96GB 4090** was spotted for sale on Taobao ([X post](https://x.com/yomix1337/status/1893692548108984391?s=46)), sparking excitement about higher memory capacity for local training.
   - Availability is still some months away.
- **Rust CubeCL Quality Queried**: Interest arose regarding the quality of **Rust CubeCL**, given it's created by the same team that works on **Rust Burn**.
   - The member was *wondering if Rust CubeCL was good*.
- **Clarification Sought on RANGE Op Operation**: A member initially questioned the operation of the `RANGE` Op, presuming its absence in the `Tensor` implementation of `arrange`.
   - However, the member later cleared up their confusion, clarifying that it *"isn't a range"*.
- **iGPU Auto-Detection Questioned on Linux**: A user questioned whether the default device initialization or `Device.get_available_devices()` should automatically detect an **iGPU** on Linux.
   - Their post included an image that showed *"Device: [CPU]"*, which the user did not expect.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune Copies the Original Special Tokens**: The **TorchTune checkpointer** copies the original **special_tokens.json** from Hugging Face instead of a potentially modified, custom version, from the file [here](https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/training/checkpointing/_checkpointer.py#L892-L896).
   - The team decided against exposing new arguments without a strong reason, so the recommendation is to manually copy the file for now.
- **Torchtune hits 5k GitHub Stars**: The Torchtune project achieved **5,000 stars on GitHub**.
   - The community celebrated this achievement.
- **GRPO Recipe Suffers from Empty Cache Overuse**: A member inquired about the excessive use of `torch.cuda.empty_cache()` calls in the **GRPO recipe**.
   - Another member admitted that many of these calls are likely excessive, stemming from early development when they faced **memory issues**.
- **GRPO PRs Languishing**: Two **GRPO PRs**, specifically **#2422** and **#2425**, have been open for two weeks and are awaiting review.
   - A member is requesting assistance in reviewing them, asking someone to help unload the queue.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC lectures match Berkeley Lectures**: A member inquired whether Berkeley students have exclusive lectures beyond the MOOC, and a colleague responded that **Berkeley students and MOOC students attend the same lectures**.
   - There was no further commentary on the substance of the lectures.
- **Certificate Award Delayed**: A member reported submitting a certificate declaration form in December but received notice that there was **no submission recorded**.
   - This issue was brought up in #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1346810951265157201) with no further details, but this topic may indicate systematic problems with the MOOC.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **AST Metric Remains Mysterious**: A member questioned the meaning of the **AST (Abstract Syntax Tree) metric**, specifically whether it measures the percentage of correctly formatted function calls generated by an LLM.
   - The inquiry went unanswered in the channel.
- **V1 Dataset Origins Unknown**: A user asked about the construction of the **V1 dataset**.
   - Like the query about the **AST metric**, this question also received no response.
- **Python Tool Champion Still Undecided**: A member sought recommendations for the best model for prompt tool calling, considering **Gemini 2**, **GPT o3-high**, and **Deepseek R1**.
   - The specific use case involves calling a **Python tool**.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 Labs Drops Jamba 1.6**: AI21 Labs launched **Jamba 1.6**, an open model tailored for private enterprise deployment, with model weights available on [Hugging Face](https://huggingface.co/ai21labs).
   - The company claims it *delivers unmatched speed and performance*, setting a new benchmark for enterprise AI without compromising efficiency, security and data privacy.
- **Jamba 1.6 Shows Off Arena Prowess**: **Jamba 1.6** reportedly outperforms **Cohere**, **Mistral**, and **Llama** on the Arena Hard benchmark, rivaling leading closed models according to [AI21's announcement](https://www.ai21.com/jamba/).
   - The release highlights its suitability for fully private on-prem or VPC deployment, boasting lightning-fast latency and a market-leading **256K context window**.
- **Hybrid Architecture Gives Jamba 1.6 Edge**: The **AI21 Jamba** family features hybrid **SSM-Transformer** foundation models, excelling in both quality and speed, thanks to its novel **Mamba-Transformer MoE architecture** designed for cost and efficiency gains as explained in the [Jamba 1.6 blogpost](https://www.ai21.com/jamba/).
   - The model is deployable anywhere, self-hosted, or in the AI21 SaaS, to meet diverse data security needs.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1346800454369017957)** (1303 messages🔥🔥🔥): 

> `Sonnet 3.7, Qwen, Windsurf IDE, MCP Client Closed, OpenRouter API` 


- **Agent Antics: Cursor's Code-Writing Catastrophes Continue**: Several users are reporting that **Cursor agents are still struggling** with basic tasks, such as *finding files* and *editing code*, with one user noting that **Claude API cost them $20 in 2 days** without better results.
   - Another user chimed in that **Sonnet 3.7** has stopped being a lunatic, and is useful again, while others are still trying to find fixes for the problems they have.
- **Qwen Claims Reasoning Crown, Dethrones DeepSeek R1**: Alibaba's **QwQ-32B** has been claimed to be comparable to **DeepSeek-R1**, while having 20x fewer parameters, and even fewer than **DeepSeek R1**'s 37B active parameter count
   - According to user, *this is just a troll benchmark*, but other sources are claiming that **QwQ-32B has a GPQA Diamond score of 59.5%**.
- **Windsurf's Wave 4 Update is Crashing Cursor's Party**: Users are reporting that **Windsurf's Wave 4** update performs fluently with **Sonnet 3.5**, with another user claiming that it gets the *try again* issue, whereas another states it was not as good with linting as Cursor.
   - One user also reports that **Cursor IDE** is not modifying files.
- **MCP Mayhem: Client Closed Calamity Confounds Coders**: Users are facing issues with **MCP Servers** on Windows, experiencing a *Client Closed* error, as some try to find short term solutions while others keep finding temporary fixes.
   - One user mentions a solution involving running the command in a CMD terminal, while others have not been able to fix it.
- **OpenRouter API access**: Some users are discussing the use of official API vs **OpenRouter**, with engine being **Claude Code**, whereas it comes up that even **Claude-max** gets charged at 2 dollars per request, while other users are hitting API limits.
   - Members are discussing Cursor being potentially *over-priced* compared to the API, leading to the need to switch, whereas others are not reaching those limits and do not mind paying for the services.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/techfrenaj/status/1897337662769672309?s=46">Tweet from techfren (@techfrenAJ)</a>: Introducing Chaos CoderOpen source and Deployed🔗 in comments v</li><li><a href="https://x.com/artificialanlys/status/1897701015803380112?s=46">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Alibaba launches QwQ-32B, an open weights reasoning model that may approach DeepSeek R1’s level of intelligenceWe’ve been running evals on it all night and we’ve only gotten our scores back for GPQA D...</li><li><a href="https://container-seven-sigma.vercel.app">Container Ejection Simulator</a>: no description found</li><li><a href="https://elitecaptures7.com/">Elitecaptures7</a>: no description found</li><li><a href="https://templeos.org">TempleOS</a>: no description found</li><li><a href="https://fontawesome.com/icons/house?f=classic&s=solid">House Classic Solid Icon | Font Awesome</a>: House icon in the Solid style. Make a bold statement in small sizes.  Available now in Font Awesome 6.</li><li><a href="https://codeinplace.stanford.edu">Code In Place</a>: no description found</li><li><a href="https://code.visualstudio.com/updates/v1_98">February 2025 (version 1.98)</a>: Learn what is new in the Visual Studio Code February 2025 Release (1.98)</li><li><a href="https://x.com/steph_palazzolo/status/1897309493744267314?s=46">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: NEW w/ @coryweinberg:OpenAI is doubling down on its application business. Execs have spoken with investors about three classes of future agent launches, ranging from $2K to $20K/month to do tasks like...</li><li><a href="https://github.com/ollama/ollama/commit/dc13813a03105bd76603a4909e31ba0c034d670d">server: allow vscode-file origins (#9313) · ollama/ollama@dc13813</a>: no description found</li><li><a href="https://github.com/agno-agi/agno">GitHub - agno-agi/agno: Build Multimodal AI Agents with memory, knowledge and tools. Simple, fast and model-agnostic.</a>: Build Multimodal AI Agents with memory, knowledge and tools. Simple, fast and model-agnostic. - agno-agi/agno</li><li><a href="https://tenor.com/view/good-weekend-gif-6442363721098209555">Good Weekend GIF - Good weekend - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://xd.adobe.com/embed/2bf05be6-17a0-40a9-a92c-56310b487db8-7ea3/?fullscreen"">Elitecaptures7-v2</a>: 74 Screens, Last modified on Jun 14, 2022 22:11 GMT
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1346922878729457785)** (4 messages): 

> `AGI development, OpenAI o1 and o3-mini, ChatGPT for macOS, AI safety and alignment` 


- **GPT-4.5 Rollout Concludes Early**: The rollout of **GPT-4.5** is complete, sharing insights into their approach to **AI safety and alignment**.
   - Instead of a single pivotal moment, **AGI development** is seen as a *continuous path*, iteratively deploying and learning from today's models to make future AI safer and more beneficial.
- **o1 and o3-mini join the OpenAI API**: **OpenAI o1 and o3-mini** are now available in the API for developers on all paid usage tiers and can be used with [Streaming, Function calling, Structured Outputs, Reasoning effort, Assistants API, Batch API](https://platform.openai.com/docs/models/compare?model=o1) and Vision (for o1 only).
   - Their approach is guided by embracing **uncertainty**, **defense in depth**, **methods that scale**, **human control**, and **community efforts** to ensure that AGI benefits all of humanity.
- **MacOS ChatGPT now can edit code in IDEs**: **ChatGPT for macOS** can now edit code directly in IDEs.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1346817573722914846)** (822 messages🔥🔥🔥): 

> `Grok3 vs Claude, DeepSeek, Atom of Thought, Microsoft Phi models, Apple unified memory` 


- **Gemini struggles, Grok rises to the occasion**: Members noted **Gemini** acting like **GPT-3.5** while others are switching to **Grok3** due to its better performance and generous cap, though one member stated *ANYTHING BUT GROK*.
   - They further noted that **Grok3** speaks natural like **GPT-4.5**, codes better than **Sonnet 3.7**, has a generous cap, and can drop *f bombs*.
- **DeepSeek R1 Distilled Reasoning**: The community is talking about **DeepSeek R1 Distill** model's reasoning capabilities with some users noting it's one of the most natural sounding LLMs, but another member says it *doesn't feel bright* without supplied knowledge.
   - Members also noted they've been experimenting with **Atom of Thought** to achieve similar levels of reasoning, and [there's a paper](https://arxiv.org/abs/2412.06769) that helps implement CoT using raw embeddings as "tokens".
- **Microsoft Phi-4 is Now Available**: After a user asked about the usefulness of **Phi-2** for prompt refinement, other members suggested using **Phi-4** instead due to its improvements in both performance and capabilities, although its larger size requires more VRAM.
   - Members pointed out that it has multiple models in the suite, and isn't limited to the original 14.7B model.
- **Apple's Unified Memory: Training Game Changer?**: A member noted that **Apple** has released a **PC with 512GB unified memory** which might be interesting for model training, but another member points out that it requires deep pockets at **$10k**.
   - Members noted the lower memory bandwidth of LPDDR5x, but still pointed out some models can run in FP4 with that much memory. 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pastebin.com/Zez0gt1R">&quot;&quot;&quot;atom_of_thought.py-----------------This module implements the &quot;Atom of - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://www.youtube.com/watch?v=yD4NrND3NC0">Part 1: AI crushed Tom’s career. Are the rest of us next?</a>: Tech billionaires like Elon Musk say the AI systems they&#39;re building will replace lots of people’s jobs, but also create better ones in their place. Is that ...</li><li><a href="https://youtu.be/Vshg-hNUEjo">Nana Mouskouri - Guten Morgen, Sonnenschein 1977</a>: Nana Mouskouri - Guten Morgen, Sonnenschein 1977
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1346907244192596150)** (24 messages🔥): 

> `GPT-4.5 Availability and Limitations, GPT-4.5 vs GPT-4o Performance, GPT-4.5 Prompting Strategies, GPT-4.5 Personalization Prompt, GPT-4.5 Mobile Issues` 


- **GPT-4.5's Availability is Limited**: Members noted availability is limited to approximately **50 uses per week**, but might increase gradually as OpenAI gathers feedback.
   - It was clarified that **GPT-4.5** is not a direct replacement for other models like **GPT-4o** and users should pick the model best suited for each task.
- **GPT-4.5 Outshines GPT-4o in Accuracy but Falls Short in Writing**: Some users find **GPT-4.5** performs worse than **GPT-4o** at creative writing tasks, while others report improved performance at document analysis.
   - One user reported that while **GPT-4.5** is better at accuracy and world knowledge, they needed to remind him or re-post the message to get it to complete a document.
- **GPT-4.5 Needs Detailed Trust-Building Prompts**: Users report that **GPT-4.5** requires more detailed and longer prompts, preferably in markdown, to achieve optimal results.
   - One user suggested that building trust with **GPT-4.5** through cordial messages before complex requests improves response quality, providing a [sample personalization prompt](paste.link.here) to enhance nuanced reasoning and emotional connection.
- **GPT-4.5 Suffers Android Mobile Compatibility**: One user noted that **GPT-4.5** refuses to work on Android mobile (both app and browser) but works fine on iOS.
   - The user explained that **GPT-4.5** produced the error message: *"I'm sorry, but I won't be able to help with that request."


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1346891297771749477)** (13 messages🔥): 

> `Prompt Engineering Techniques Ontology, Sora AI Video Character Consistency, GPT-4o OCR Bounding Box Issues` 


- **Prompt Engineering Techniques Get Systematized**: A member shared an overview of prompt engineering techniques from *A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications*, including **Zero-Shot**, **Few-Shot**, **Chain-of-Thought (CoT)**, and **Retrieval Augmented Generation (RAG)**.
   - The poster also noted that the ontology in the paper isn't exhaustive, omitting techniques such as **Self-Discover** and **MedPrompt**, then shared a [ChatGPT link](https://chatgpt.com/share/67c89f53-c72c-8000-b64b-ca30c6971854) for broader access.
- **Sora User Seeks Isabella Moretti Consistency**: A member is creating cinematic AI videos with **Sora**, focusing on a character named **Isabella Moretti**, aiming for hyper-realistic visuals and consistent character details across multiple clips.
   - The creator seeks strategies or prompt tips for enhancing realism and maintaining consistency in details like **skin tone**, **eyes**, and **hair**.
- **GPT-4o Struggles with Bounding Boxes**: A user reported issues with the **GPT-4o model** inaccurately returning **bounding box coordinates** in OCR results when using the OpenAI API.
   - They requested advice on obtaining accurate OCR results with coordinates from the OpenAI API, but they failed to provide an image sample.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1346891297771749477)** (13 messages🔥): 

> `Prompt Engineering Survey, Sora AI videos character consistency, GPT-4o OCR results` 


- **Prompt Engineering Techniques Boast Wide Variety**: A member shared an overview of prompt engineering techniques from *A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications*, outlining categories such as **zero-shot prompting**, **chain-of-thought**, and **RAG**.
   - A [ChatGPT link](https://chatgpt.com/share/67c89f53-c72c-8000-b64b-ca30c6971854) to the same overview was also shared for wider accessibility, and noted it *isn't even an exhaustive ontology! It left out Self-Discover and MedPrompt, among others!*.
- **Sora Generates Consistent Characters Cinematically**: A member is creating cinematic AI videos with **Sora**, focusing on a character named **Isabella Moretti** and seeking strategies to achieve **hyper-realistic visuals** and improve character consistency across multiple clips.
   - The creator specifically aims to maintain consistent details like **skin tone**, **eyes**, **hair**, and **expressions**, while also refining prompt structure for optimal cinematic quality, including **lighting**, **camera movements**, and **transitions**.
- **GPT-4o's OCR bounding box coordinates are incorrect**: A member reported issues getting correct bounding box coordinates from the **OpenAI API** using the **GPT-4o model** for **OCR results**.
   - The prompt they used returned incorrect bounding box coordinates, and they were seeking advice on resolving this issue.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1346939336792342611)** (2 messages): 

> `Windsurf Wave 4, Cascade updates, Windsurf Previews, Auto-Linter in Cascade, MCP Server updates` 


- **Windsurf Wave 4 Released with Big Waves**: The latest **Windsurf Wave 4** release introduces game-changing features such as **Previews**, **Tab-to-import**, **Linter integration**, and **Suggested actions**, alongside improvements to **MCP discoverability** and **Claude 3.7** integration, detailed in a [blog post](https://www.codeium.com/blog/windsurf-wave-4).
- **Cascade Embraces Element Preview Selection**: **Cascade** now lets you preview locally run websites in your IDE or in your browser.
   - Users can select **React** and **HTML** elements within the preview to send to **Cascade** as context, streamlining the conversation process, as seen in the [X/Twitter announcement](https://x.com/windsurf_ai/status/1897378545799979238).
- **Codeium Fixes Preview Route Loading**: A patch was released to address issues in **Windsurf Previews** where certain routes would not load, alongside fixes to open Cascade shortcuts and the restoration of missing proxy settings and index size options, according to the [full changelog](https://www.codeium.com/changelog).
- **Cascade Introduces Auto-Linter for Seamless Code Correction**: The **Windsurf Wave 4** update integrates an **Auto-Linter** directly into **Cascade**, which automatically fixes lint errors in generated code, ensuring cleaner code output.
   - Check out the [YouTube video](https://www.youtube.com/watch?v=bIy-RN3FIsQ&feature=youtu.be) for more details.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.</li><li><a href="https://www.codeium.com/blog/windsurf-wave-4">Windsurf Wave 4</a>: Introducing Wave 4, our fourth batch of updates to the Windsurf Editor.</li><li><a href="https://x.com/windsurf_ai/status/1897378545799979238">Tweet from Windsurf (@windsurf_ai)</a>: Windsurf Wave 4 is here!Included in this update:🖼️ Previews✏️ Cascade Auto-Linter⚙️ MCP UI Improvements ➡️ Tab to Import↩️ Suggested Actions🫶 Claude 3.7 Improvements🤝 Referrals🖥️ Windows ARM Suppo...</li><li><a href="https://bsky.app/profile/windsurfai.bsky.social/post/3ljnsaugqk22l">Windsurf (@windsurfai.bsky.social)</a>: Windsurf Wave 4 is here!Included in this update:🖼️ Previews✏️ Cascade Auto-Linter⚙️ MCP UI Improvements▶️ Tab to Import↩️ Suggested Actions🫶 Claude 3.7 Improvements🤝 Referrals🖥️ Windows ARM Suppor...</li><li><a href="https://www.threads.net/@codeiumdev/post/DG1IyC5CODS?xmt=AQGzB0CoP8oQ9hE-8YatsFH7FaIFFpnONInUNHCSr9H8qg">Codeium (&#064;codeiumdev) on Threads</a>: Windsurf Wave 4 is here!Included in this update:&#x1f5bc;&#xfe0f; Previews&#x270f;&#xfe0f; Cascade Auto-Linter&#x2699;&#xfe0f; MCP UI Improvements&#x25b6;&#xfe0f; Tab to Import&#x21a9;&#xfe0f; Suggest...</li><li><a href="https://www.youtube.com/watch?v=bIy-RN3FIsQ&feature=youtu.be">Windsurf Wave 4 Updates: Preview, Tab to Import, Suggested Actions &amp; More</a>: Windsurf Wave 4 is here, bringing exciting new features to enhance your experience!🌊 Make sure to update to the latest version of Windsurf to get all these ...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1346812925649748030)** (45 messages🔥): 

> `VS Code Commit Message Generation Issue, Flutterflow Usage, Codeium Uninstall, Codeium Language Server Download Issues, Telemetry Data in Codeium Chat Feature` 


- **Users face VS Code Commit Message Generation Issue**: A user reported an issue with generating commit messages in VS Code using the pre-release version of Codeium and sought a workaround.
   - Other users also asked about **Flutterflow** and how to **completely uninstall** the current **Codeium extension**.
- **Codeium Language Server Downloader has issues**: Multiple users reported issues with **Codeium failing to download the language server**, with a specific error message pointing to an attempted download URL from `releases.codeiumdata.com`.
   - The issue persisted even after restarting the IDE and was encountered on both **WSL** and **Windows** installations.
- **Individual accounts require Code Snippet Telemetry to enable Chat Feature**: A user on the individual account trial questioned the need to enable code snippet sharing to use the chat feature, citing conflicting information in the FAQ regarding telemetry.
   - Another user mentioned that they **tested Codeium on non-sensitive data before buying the Pro plan**, where **chat worked without code snippet telemetry** and also linked the [Data Privacy section of Codeium's FAQ](https://codeium.com/faq#data-privacy).
- **"Disabled by team..." error plagues user**: A user reported recurring issues with their account being stuck on *"disabled by team..."*, preventing them from using the extension despite reinstalling the software multiple times.
- **Reminder: Windsurf topics not related to Extension should be in Windsurf channel**: A user pointed out that topics related to **Windsurf**, which is not directly related to the extension itself, should be directed to specific **Windsurf channels** such as <#1306163501286293515>, <#1324809403920023604>, or <#1334764563064688692>.



**Link mentioned**: <a href="https://codeium.com/faq#data-privacy">FAQ | Windsurf Editor and Codeium extensions</a>: Find answers to common questions.

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1346807307966746674)** (609 messages🔥🔥🔥): 

> `Windsurf Stability Issues, Credit Usage Concerns, Rollback Feature Request, 3.7 Performance Woes, Global Rules and .windsurfrules` 


- **Windsurf Plagued by Post-Update Perils**: Users report **stability issues** after the latest Windsurf update (Wave 4, v1.4.3), including frequent dropped calls, lagging, and internal memory hallucinations, leading some to switch to alternatives like Cursor or Trae IDEs.
   - One user complained, *WS is not working like it used to since the latest update. too many dropped calls, lagging, poor context memory, strange internal memory hallucinations, etc.*
- **Credit Crunch Crushing Codeium's Customers**: Members express concern over **increased credit consumption**, especially with Claude 3.7, with some experiencing rapid credit depletion due to looping errors and excessive tool calls, leading to calls for an unlimited plan.
   - Some users feel *ripped off* because they can't use advanced models, even just to chat, with their credit limits.
- **Rollback Rescue Mission: Users Demand Version Reversal**: Users are clamoring for a **downgrade feature** to revert to previous Windsurf versions due to the latest update introducing critical issues that hinder productivity.
   - Users are now *stuck* with the updated version, with one saying they *wish they never hit 'restart to update'*.
- **Claude 3.7 Code Conversion Catastrophe**: Users are reporting that **Claude 3.7** performs worse after Wave 4 and is consuming more credits, with some saying it generates endless code, while others reported that it won't read files or keep edits.
   - Said one user: *My agents can barely complete anything beyond the simplest of prompts after the update.*
- **Windsurf Global Rules Rendezvous**: Users discuss the usage of **global rules** and `.windsurfrules`, a way to specify rules in a project, and also clarify that global rules can be found in the user's Codeium/Windsurf folder.
   - One user shared they have a comprehensive global rules file yet Windsurf still acts unexpectedly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ericciarla/status/1897332080708858147">Tweet from Eric Ciarla (hiring) (@ericciarla)</a>: Generate an llms.txt file for any website in seconds with /llmstxtOur new @firecrawl_dev endpoint turns any site into a single text file that can be fed into any LLM.Check it out integrated as a @rayc...</li><li><a href="https://brave.com/search/api/">Brave Search API | Brave</a>: Power your search and AI apps with the fastest growing independent search engine since Bing. Access an index of billions of pages with a single call.</li><li><a href="https://docs.codeium.com/windsurf/previews">Previews (Beta) - Codeium Docs</a>: no description found</li><li><a href="https://neon.tech">Neon Serverless Postgres — Ship faster</a>: The database you love, on a serverless platform designed to help you build reliable and scalable applications faster.</li><li><a href="https://docs.codeium.com/windsurf/usage#purchasing-additional-flex-credits">Paid Plan and Credit Usage - Codeium Docs</a>: no description found</li><li><a href="https://docs.codeium.com/windsurf/usage">Paid Plan and Credit Usage - Codeium Docs</a>: no description found</li><li><a href="https://techcrunch.com/2025/03/05/openai-reportedly-plans-to-charge-up-to-20000-a-month-for-specialized-ai-agents/">OpenAI reportedly plans to charge up to $20,000 a month for specialized AI &#039;agents&#039; | TechCrunch</a>: OpenAI may be planning to charge up to $20,000 per month for specialized AI &#039;agents,&#039; according to The Information.</li><li><a href="https://www.youtube.com/@codeiumdev/videos">Codeium - Windsurf</a>: 🧑‍💻 | Your modern coding superpower🚀 | 3M+ Codeium extension downloads🏄‍♂️ | Building the Windsurf Editor</li><li><a href="https://pierre.co/">Pierre</a>: Joyful Code Review</li><li><a href="https://codeium.com/windsurf/directory">Windsurf Rules Directory</a>: Tomorrow&#x27;s editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.</li><li><a href="https://codeium.com/plan">Plan Settings</a>: Tomorrow&#x27;s editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.</li><li><a href="https://tenor.com/view/rage-angry-communication-gif-17637019732916283735">Rage Angry GIF - Rage Angry Communication - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.canny.io/feature-requests/p/improve-previews-feature-with-a-proper-webview-in-the-sidebar-like-trae">Improve &quot;Previews&quot; feature with a proper &quot;Webview&quot; in the sidebar (like Trae) | Feature Requests | Codeium</a>: Love the new &quot;Previews&quot; feature! 🎉✨ However, I&#x27;d love for there to simply be a &quot;Preview&quot; tool in the sidebar like how Trae has a &quot;Webview&quot; tool.</li><li><a href="https://codeium.com/pricing">Pricing | Windsurf Editor and Codeium extensions</a>: Codeium is free forever for individuals. Teams can level up with our enterprise offering for enhanced personalization and flexible deployments.</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://codeium.canny.io/">Codeium Feedback</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://codeium.canny.io/feature-requests/p/unlimited-plan">Unlimited Plan | Feature Requests | Codeium</a>: I don’t think it’s right to provide limited credits while your competitor, which is downloaded and used more than you, offers unlimited limits.</li><li><a href="https://codeium.canny.io/feature-requests/p/pro-ultimate-is-not-so-ultimate-if-were-limited-on-3000-flow-credits">Pro Ultimate is not so &quot;Ultimate&quot; if we&#x27;re limited on 3000 Flow Credits. | Feature Requests | Codeium</a>: Flow credits should be unlimited or Pro Ultimate, or rename it to &quot;Just a tiny bit better than regular Pro&quot;.</li><li><a href="https://tenor.com/view/cat-kitten-kitty-pussy-cat-cute-gif-16834206313880236094">Cat Kitten GIF - Cat Kitten Kitty - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/ooz-ooznmates-oozandmates-ooz-dook-dook-gif-13562370673666741588">Ooz Ooznmates GIF - Ooz Ooznmates Oozandmates - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/X1rD3NhlIcE?si=VEwFEZUb6q5CncWL">LLM generates the ENTIRE output at once (world&#39;s first diffusion LLM)</a>: Register for 3-hour AI training with GrowthSchool! Free for the first 1000 people who sign up! https://web.growthschool.io/MWBJoin My Newsletter for Regular ...</li><li><a href="https://chat.inceptionlabs.ai/">Mercury Coder</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1346807504738189355)** (424 messages🔥🔥🔥): 

> `Phi-4-mini models support, Overfitting models on benchmarks, DeepSeek R1, Inventing random benchmarks, Flex Attention support` 


- **Phi-4-mini Models Soon to be Supported**: A member asked if there are plans to support **phi-4-mini models**, and another member confirmed *yep*.
- **Benchmarking Smashes with Overfitting**: A member inquired about the possibility of **overfitting a model on benchmarks** to achieve state-of-the-art results with a smaller model, which another member said has been done before.
   - A member referenced the paper **phi-CTNL**, stating that it supercharges such approaches by investing heavily in curating a novel, high quality, non-synthetic data mixture based solely on evaluation benchmarks.
- **Windows now supporting Unsloth**: Unsloth now works on Windows, allowing for local fine-tuning of LLMs without Linux or WSL, according to [this X post](https://x.com/UnslothAI/status/1897334290935132602).
   - Tutorial is provided: [Unsloth Windows Installation](https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation).
- **QwQ-32B Reasoning Model Released with Bug Fixes**: A new reasoning model, **QwQ-32B**, has been released and the Unsloth team provided bug fixes and dynamic quants, which greatly improves accuracy over standard 4-bit, accessible [here](https://huggingface.co/unsloth/QwQ-32B-GGUF).
   - The repo contains the QwQ 32B model and has features like transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias.
- **Understanding Reasoning Models and Their Applications**: Members discussed the definition and usage of **reasoning models**, noting they are LLMs trained to output tokens to 'think' before answering, often prompted similarly to SFT models.
   - It's also been shown that you can give the reasoning process of a reasoning LLM to a non-reasoning LLM to finish and provide the answer and surprisingly enough they do fairly well.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/gandhikanishk/status/1896988028893323675">Tweet from Kanishk Gandhi (@gandhikanishk)</a>: New Paper!! We try to understand why some LMs self-improve their reasoning while others hit a wall. The key? Cognitive behaviors! Read our paper on how the right cognitive behaviors can make all the d...</li><li><a href="https://x.com/UnslothAI/status/1897334290935132602">Tweet from Unsloth AI (@UnslothAI)</a>: Unsloth now works on Windows! 🦥Fine-tune LLMs locally on Windows without Linux or WSL.Tutorial: https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation</li><li><a href="https://arxiv.org/abs/2309.08632">Pretraining on the Test Set Is All You Need</a>: Inspired by recent work demonstrating the promise of smaller Transformer-based language models pretrained on carefully curated data, we supercharge such approaches by investing heavily in curating a n...</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics of fine-tuning.</li><li><a href="https://pastebin.com/MWGHg1UR">QWQ-32B solves caesar cipher - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://huggingface.co/mradermacher/QwQ-32B-i1-GGUF">mradermacher/QwQ-32B-i1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/QwQ-32B-GGUF">unsloth/QwQ-32B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog">Blog</a>: no description found</li><li><a href="https://mistral.ai/news/mistral-ocr">Mistral OCR | Mistral AI</a>: Introducing the world’s best document understanding API.</li><li><a href="https://huggingface.co/unsloth/QwQ-32B-GGUF/discussions/2">unsloth/QwQ-32B-GGUF · QwQ-32B-Q5_K_M Cyclically thinking</a>: no description found</li><li><a href="https://huggingface.co/Qwen/QwQ-32B">Qwen/QwQ-32B · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing#scrollTo=cKcvFLCsQLtL">Google Colab</a>: no description found</li><li><a href="https://spraakbanken.gu.se/resurser/swefaq">SweFAQ 2.0 | Språkbanken Text</a>: no description found</li><li><a href="https://huggingface.co/AI-Sweden-Models/Llama-3-8B">AI-Sweden-Models/Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/flex_attention.py">unsloth/unsloth/kernels/flex_attention.py at main · unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/1662#issuecomment-2649554021">GRPO for vision models too? · Issue #1662 · unslothai/unsloth</a>: &#39;Qwen2VLForConditionalGeneration&#39; object has no attribute &#39;vllm_engine&#39; Uncommented some vllm specific from unsloth import is_bfloat16_supported import torch from unsloth import FastVi...</li><li><a href="https://github.com/codestoryai/sidecar">GitHub - codestoryai/sidecar: Sidecar is the AI brains for the Aide editor and works alongside it, locally on your machine</a>: Sidecar is the AI brains for the Aide editor and works alongside it, locally on your machine - codestoryai/sidecar</li><li><a href="https://github.com/unslothai/unsloth/">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/releases">Releases · unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth?</li><li><a href="https://github.com/unslothai/unsloth/commit/8a675d86c218318bc499fcb53d0aeb5061f88875">Logits fixes (#1916) · unslothai/unsloth@8a675d8</a>: * Update rl_replacements.py* Update llama.py* Update llama.py* Update llama.py* Update llama.py* Update llama.py* Update rl_replacements.py* Update llama.py* Update llama.py* Upda...</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin/blob/79e42a8ef4430ab0d2e49ec2fc2d695967641221/src/constants.py#L2789>">VectorDB-Plugin/src/constants.py at 79e42a8ef4430ab0d2e49ec2fc2d695967641221 · BBC-Esq/VectorDB-Plugin</a>: Plugin that lets you ask questions about your documents including audio and video files. - BBC-Esq/VectorDB-Plugin</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin/blob/79e42a8ef4430ab0d2e49ec2fc2d695967641221/src/constants.py#L8>">VectorDB-Plugin/src/constants.py at 79e42a8ef4430ab0d2e49ec2fc2d695967641221 · BBC-Esq/VectorDB-Plugin</a>: Plugin that lets you ask questions about your documents including audio and video files. - BBC-Esq/VectorDB-Plugin
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1346811732957335606)** (131 messages🔥🔥): 

> `Qwen7b Memory Consumption, GRPO Success, TinyZero Replication, Llama 3.1, Hyperparameter Tuning` 


- **Qwen7b Drinks all the Memory**: A user found that the **Qwen7b model** is very memory hungry, necessitating a decrease in the per device batch size; they used a batch size of **8 per device**, **8 generations**, and **4 grad accumulation steps**.
   - In comparison, the user noted that *good old Mistral* is much less demanding on memory.
- **TinyZero Replication**: Users discussed replicating the **TinyZero** project and found a [reproduction](https://github.com/JerryWu-code/TinyZero/blob/main/scripts/train_tiny_zero_a100_grpo.sh) that uses only **5 rollouts**.
   - Additionally, it was noted that the **KL divergence multiplier** in this reproduction is very small compared to the default in **GRPOConfig**.
- **GRPO Hyperparameter Findings**: A member shared a [DeepResearch PDF](https://cdn.discordapp.com/attachments/1179039861576056922/1346937593442603082/Hyperparameter_Optimization_for_On-Policy_RL_in_LLM_Alignment.pdf) on hyperparameter tuning for RL with LLMs, noting the importance of a **large penalty** in **GRPOConfig**.
   - The member pointed out the usual HF pipeline assumes full weights training, which changes the model faster than feeble lora.
- **Unsloth GRPO and RLOO Memory**: It was pointed out that **Unsloth's GRPO** may have better memory efficiency due to offloading to CPU and more efficient gradient accumulation.
   - Daniel's insane optimization of the memory enables fused kernels that avoid materialization of logits in memory.
- **Profiling with LLMs**: During training, simple **cProfile** helped a lot for finding bottlenecks, while **torch profiler** was not very useful.
   - For custom models during inference, torch profile is exactly what you need.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Jiayi-Pan/TinyZero/blob/main/scripts/train_tiny_zero.sh">TinyZero/scripts/train_tiny_zero.sh at main · Jiayi-Pan/TinyZero</a>: Clean, minimal, accessible reproduction of DeepSeek R1-Zero - Jiayi-Pan/TinyZero</li><li><a href="https://github.com/JerryWu-code/TinyZero/blob/main/scripts/train_tiny_zero_a100_grpo.sh">TinyZero/scripts/train_tiny_zero_a100_grpo.sh at main · JerryWu-code/TinyZero</a>: Deepseek R1 zero tiny version own reproduce on two A100s. - JerryWu-code/TinyZero</li><li><a href="https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643">1 gpu is not working , 2 gpus out of memory  · Issue #5 · Jiayi-Pan/TinyZero</a>: how to deal with the error below , 1A100 PCIe 80gb . Followed the instruction with error below . 2A100 80gb working fine but out of memory . I guess the code default to multiple GPUs . the only wor...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1346827043618291824)** (173 messages🔥🔥): 

> `Deepseek distillation, Unsloth Windows support, Multi-GPU support, Qwen Coder, GRPO Training Problems` 


- **Distilling Reasoning Results in DeepSeek Models**: A user asked if anyone has replicated **DeepSeek's** results of distilling reasoning into smaller models and if they could share the differences in tokenizer and prompt templates used.
   - Another user provided a link to a [tutorial](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo) on training a reasoning model with **GRPO** and **Unsloth**, advising the user that finetuning is not a one-button operation and requires trial and error.
- **Troubleshooting Triton on Windows for Unsloth**: Users encountered **TypeError: cannot unpack non-iterable NoneType object** when working with Triton, traced back to issues with finding the Windows SDK during Unsloth installation, and were directed to the [Windows Installation Guide](https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation).
   - One user reported a fix using a fork from [oKatanaaa](https://github.com/oKatanaaa/unsloth-zoo) while another successfully downgraded to version **2025.1.5** to bypass the error.
- **Multi-GPU Training is elusive**: Users discussed multi-GPU training, with one asking for an example notebook for finetuning with **LoRA** on a single machine with multiple GPUs.
   - It was clarified that **Unsloth** does not currently support multi-GPU training in the community version, though it's available in the pro version.
- **Pinpointing Root Cause of GRPO Training Issues**: Users reported issues with **GRPO training**, including problems with evaluation metrics, compilation failures **RuntimeError: Unsloth: Failed to create dynamic compiled modules!** after updating Unsloth, and training loss remaining at zero, and [a fix](https://github.com/unslothai/unsloth/issues/1711) was found.
   - Downgrading to **unsloth==2025.3.6** and **unsloth_zoo==2025.3.4** helped resolve compilation errors; another user patched the unslothGRPOTrainer in their [Colab notebook](https://colab.research.google.com/drive/1u6Acmib0wj2XRvcSrTWdMW0caIe-fHhQ?usp=sharing).
- **Navigating Cache File Modifications for Custom Training**: A user sought guidance on modifying the generated cache file in Unsloth for custom training steps, with suggestions to examine **rl.py/rl_replacements.py** in Unsloth and **rl_replacements.py** in **unsloth_zoo**.
   - Experts recommended cloning the **Unsloth GitHub repo** ([https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)) and installing from source to apply and retain edits, and another user shared a [temporary patch](https://colab.research.google.com/drive/1u6Acmib0wj2XRvcSrTWdMW0caIe-fHhQ?usp=sharing) for fixing cache issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo">Tutorial: Train your own Reasoning model with GRPO | Unsloth Documentation</a>: Beginner&#x27;s Guide to transforming a model like Llama 3.1 (8B) into a reasoning model by using Unsloth and GRPO.</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation">Windows Installation | Unsloth Documentation</a>: See how to install Unsloth on Windows with or without WSL.</li><li><a href="https://huggingface.co/prithivMLmods/SmolLM2_135M_Grpo_Gsm8k/blob/main/smollm-grpo/SmolLM%20x%20Grpo%20M1.ipynb">smollm-grpo/SmolLM x Grpo M1.ipynb · prithivMLmods/SmolLM2_135M_Grpo_Gsm8k at main</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/oKatanaaa/unsloth-zoo">GitHub - oKatanaaa/unsloth-zoo: Utils for Unsloth</a>: Utils for Unsloth. Contribute to oKatanaaa/unsloth-zoo development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/1711">Evaluation losses are wrong [FIXED] · Issue #1711 · unslothai/unsloth</a>: I am getting eval metrics which are very off, I am using trl&#39;s SFTTrainer and unsloth_train to avoid gradient accumulation bug. I have isolated this to versions from 2025.2.6 onwards. I ran the co...</li><li><a href="https://tenor.com/view/it-crowd-hello-it-have-you-tried-turning-it-off-and-on-again-gif-8607749">It Crowd Hello It GIF - It Crowd Hello IT Have You Tried Turning It Off And On Again - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://colab.research.google.com/drive/1u6Acmib0wj2XRvcSrTWdMW0caIe-fHhQ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://pypi.anaconda.org/rapidsai-wheels-nightly/simple">Simple Index</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1346988306319278091)** (40 messages🔥): 

> `Qwen-32B Release, RL Scaling for Medium-Sized Models, Cognitive Behaviors and LM Self-Improvement, Lossless Compression and Intelligence, AI21 Jamba for RAG` 


- ****Qwen-32B** Model Released!**: **Alibaba** released **QwQ-32B**, a new **32 billion parameter** reasoning model that rivals models like **DeepSeek-R1**, demonstrating impressive results in scaling RL based on their **Qwen2.5-32B** model, as detailed in their [blog post](https://qwenlm.github.io/blog/qwq-32b).
   - The release includes a [Hugging Face model](https://huggingface.co/Qwen/QwQ-32B), [ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B), a [demo](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo), and [Qwen Chat](https://chat.qwen.ai), with findings suggesting that RL training continuously improves performance, particularly in math and coding.
- **Decoding LM Self-Improvement Mystery**: A new paper explores why some LMs self-improve their reasoning while others don't, pinpointing *cognitive behaviors* as the key factor.
   - The paper investigates how the *right cognitive behaviors* can significantly impact a model's ability to improve with RL, detailed in [this X thread](https://fxtwitter.com/gandhikanishk/status/1896988028893323675).
- ****AI21 Labs** unveils **Jamba**: The Hybrid for RAG**: **AI21 Labs** released the **AI21-Jamba-Large-1.6** ([Hugging Face](https://huggingface.co/ai21labs/AI21-Jamba-Large-1.6)), a state-of-the-art hybrid **SSM-Transformer** model, claiming it to be the most powerful and efficient long-context model, offering up to **2.5X** faster inference than comparable models.
   - Despite the hype, there's skepticism about running a **400B** model solely for RAG, with some questioning the viability of Mamba models for long-context accuracy and awaiting further verdicts.
- **Can Compression Alone Spark Intelligence?**: Isaac Liao and Albert Gu explore whether lossless information compression can produce intelligent behavior in [this blog post](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html), suggesting a practical demonstration of this idea.
   - The post questions the fundamental relationship between efficient compression and the emergence of intelligence.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/gandhikanishk/status/1896988028893323675">Tweet from Kanishk Gandhi (@gandhikanishk)</a>: New Paper!! We try to understand why some LMs self-improve their reasoning while others hit a wall. The key? Cognitive behaviors! Read our paper on how the right cognitive behaviors can make all the d...</li><li><a href="https://x.com/alibaba_qwen/status/1897361654763151544?s=46">Tweet from Qwen (@Alibaba_Qwen)</a>: Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning model, e.g., DeepSeek-R1.Blog: https://qwenlm.github.io/blog/qwq-32bHF: https://hu...</li><li><a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">ARC-AGI Without Pretraining</a>: no description found</li><li><a href="https://huggingface.co/ai21labs/AI21-Jamba-Large-1.6">ai21labs/AI21-Jamba-Large-1.6 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1347291093913309184)** (1 messages): 

> `Aider Product Hunt Launch` 


- **Aider Launches on Product Hunt!**: The creator announced that [Aider was posted on Product Hunt](https://www.producthunt.com/posts/aider) and requested upvotes.
   - **Aider** is described as an *AI pair programmer* that edits code in your local git repo via the terminal, working with your editor, any LLM (Claude 3.5 Sonnet, DeepSeek R1, GPT-4o, local models), and many languages.
- **Upvote Aider on Product Hunt**: The announcement encourages community members to support Aider's launch by upvoting the [Product Hunt post](https://www.producthunt.com/posts/aider).
   - The post highlights Aider as an open-source developer tool using AI to enhance coding in various languages.



**Link mentioned**: <a href="https://www.producthunt.com/posts/aider"> Aider - AI Pair Programming in Your Terminal | Product Hunt</a>: Aider is the AI pair programmer that edits code in your local git repo via the terminal. Works with your editor, any LLM (Claude 3.5 Sonnet, DeepSeek R1, GPT-4o, local models), and many languages.

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1346800557029064759)** (377 messages🔥🔥): 

> `Grok 3, QwQ-32B, Mac Studio, OpenRouter throughput, Aider on Product Hunt` 


- **Grok3 is new Champ**: Users are reporting [favorable experiences](https://link.to.grok3) with **Grok3**, citing its unlimited context size and superior performance compared to **O1 Pro**.
   - One user noted *Grok does not, and have like 35 message / 2 hours unlimited context size (1 mill context)*, calling it the new champ.
- **Is QwQ-32B any good?**: The community discussed the new [QwQ-32B model](https://huggingface.co/Qwen/QwQ-32B), with some finding it *good with RAG but bad on its own* due to a narrow knowledge base, while others are curious how it stacks up against **DeepSeek-R1**.
   - One user said *That tool use benchmark performance looks like it'd be good on agentic workflows*.
- **New Ultra Expensive Mac hits local AI**: Members discussed how the new, expensive **Mac Studio** with **512GB** of memory and **810gb/s** bandwidth could impact local AI development, potentially running larger models at reasonable speeds.
   - One member said that *If you want to get 512 gb of memory with nvidia hardware you would be paying a lot more at least $50,000 i think*.
- **Aider Gets Product Hunt Love**: **Aider** was randomly posted to [Product Hunt](https://www.producthunt.com/posts/aider) and is gaining traction, with users appreciating the sudden recognition.
   - One memeber said *thats funny, many founders preparing for producthunt launch for weeks and they are finishing on lowest places, and here someone randomly added Aider and it's on the 10th place just like that.*
- **OpenRouter's Throughput Stats gets Realtime**: OpenRouter's throughput and latency charts now update in real time as shared by [this tweet](https://x.com/OpenRouterAI/status/1891510121139769542), showing recent speedups.
   - Users also noted **Parasail** and **SambaNova** as top R1 providers, with SambaNova being more expensive.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1897366902701502868">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Qwen released a new reasoning model QwQ-32B, and it is now powering Qwen Chat if you select Qwen2.5-Plus with Thinking (QwQ).Quoting Qwen (@Alibaba_Qwen) Today, we release QwQ-32B, our new reasoning m...</li><li><a href="https://x.com/mrousavy/status/1897222044808569137">Tweet from Marc (@mrousavy)</a>: ByteDance just launched Lynx – a competitor to React Native!</li><li><a href="https://x.com/Alibaba_Qwen/status/1897361654763151544">Tweet from Qwen (@Alibaba_Qwen)</a>: Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning model, e.g., DeepSeek-R1.Blog: https://qwenlm.github.io/blog/qwq-32bHF: https://hu...</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-a">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/usage/notifications.html">Notifications</a>: Aider can notify you when it’s waiting for your input.</li><li><a href="https://www.producthunt.com/posts/aider"> Aider - AI Pair Programming in Your Terminal | Product Hunt</a>: Aider is the AI pair programmer that edits code in your local git repo via the terminal. Works with your editor, any LLM (Claude 3.5 Sonnet, DeepSeek R1, GPT-4o, local models), and many languages.</li><li><a href="https://tenor.com/view/mujikcboro-seriymujik-gif-24361533">Mujikcboro Seriymujik GIF - Mujikcboro Seriymujik - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/Aider-AI/aider/blob/main/benchmark/README.md">aider/benchmark/README.md at main · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://x.com/OpenRouterAI/status/1891510121139769542">Tweet from OpenRouter (@OpenRouterAI)</a>: TIP: throughput and latency charts on OpenRouter update in real timeHere are Sonnet&#39;s. Kudos to @GoogleAI Vertex for the recent speedup!</li><li><a href="https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/">Apple unveils new Mac Studio, the most powerful Mac ever</a>: Apple today announced the new Mac Studio, the most powerful Mac ever made, featuring M4 Max and the new M3 Ultra chip.</li><li><a href="https://www.apple.com/macbook-air/">MacBook Air 13-inch and MacBook Air 15-inch</a>: MacBook Air laptop with the superfast M4 chip. Built for Apple Intelligence. Lightweight, with all-day battery life. Now in a new Sky Blue color.</li><li><a href="https://x.com/i/grok/share/632KWxxCC4NuqrPis82w7gpRm">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://tenor.com/view/mr-bean-mrbean-bean-mr-bean-holiday-mr-bean-holiday-movie-gif-3228235746377647455">Mr Bean Mrbean GIF - Mr bean Mrbean Bean - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/olilanz/RooCode-Local-Evaluation">GitHub - olilanz/RooCode-Local-Evaluation: Evaluation of Roo Code and locally hosted LLMs</a>: Evaluation of Roo Code and locally hosted LLMs. Contribute to olilanz/RooCode-Local-Evaluation development by creating an account on GitHub.</li><li><a href="https://www.producthunt.com/products/aider"> Aider - Product Information, Latest Updates, and Reviews 2025 | Product Hunt</a>: Aider is the AI pair programmer that edits code in your local git repo via the terminal. Works with your editor, any LLM (Claude 3.5 Sonnet, DeepSeek R1, GPT-4o, local models), and many languages.</li><li><a href="https://github.com/Aider-AI/aider/blob/main/benchmark/docker.sh">aider/benchmark/docker.sh at main · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/xi-jinping-gif-24241864">Xi Jinping GIF - Xi Jinping - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1346861361837244618)** (146 messages🔥🔥): 

> `Aider connects to OpenWebUI, Litellm patches, DeepSeek token output, Aider commit messages, Trailing Whitespace` 


- **Unlock Aider with OpenWebUI**: A member resolved an issue connecting **Aider** to **OpenWebUI (OWUI)** by prefixing the model name with `openai/`, ensuring **Litellm** recognizes the **OAI-compatible endpoint**.
   - *You have to prefix with openai/ so that litellm knows you're using an OAI-compat endpoint. So in my case, it's openai/myowui-openrouter.openai/gpt-4o-mini*.
- **Patch up Litellm for OpenRouter's 'Reasoning' Field**: A member shared a patch for **Litellm** to correctly display the *reasoning* field from **OpenRouter**, noting that a merged **PR #8431** didn't fully resolve the issue.
   - The [provided diff](https://github.com/litellm/litellm/pull/8431) addresses the *provider_specific_fields* for proper output, but requires an additional Aider patch to fully display the reasoning content.
- **End Token output problems resolved**: A member reported resolving their issue of **Aider** not outputting any tokens to chat, which turned out to be caused by missing credits on OpenRouter.
   - After adding credits, the member was able to confirm activity and token responses on their [OpenRouter activity screen](https://openrouter.ai/activity).
- **Aider Commit Messages customized**: A member inquired about using Aider to create commit messages only for staged files, and using `git stash save --keep-index`, `/commit`, `git stash pop` was suggested.
   - Another member mentioned using `aider --commit` to **automatically write commit messages, commit changes, and exit**, as well as a link to the [Aider commit documentation](https://aider.chat/docs/git.html#commit-messages).
- **Trailing Whitespace Removal**: Members discussed the absence of an automatic trailing whitespace removal feature in Aider, debating whether to rely on linters or custom scripts to enforce code style.
   - A member shared a **Ruff formatting** configuration in `.aider.conf.yml` with `lint-cmd: - python: ruff format` to address whitespace and broader formatting issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/git.html#commit-messages">Git integration</a>: Aider is tightly integrated with git.</li><li><a href="https://aider.chat/docs/llms/openrouter.html">OpenRouter</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/llms/gemini.html">Gemini</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/lint-test.html#linting">Linting and testing</a>: Automatically fix linting and testing errors.</li><li><a href="https://www.inceptionlabs.ai/">Inception Labs</a>: We are leveraging diffusion technology to develop a new generation of LLMs. Our dLLMs are much faster and more efficient than traditional auto-regressive LLMs. And diffusion models are more accurate, ...</li><li><a href="https://github.com/buger/probe">GitHub - buger/probe: Probe is an AI-friendly, fully local, semantic code search engine which which works with for large codebases. The final missing building block for next generation of AI coding tools.</a>: Probe is an AI-friendly, fully local, semantic code search engine which which works with for large codebases. The final missing building block for next generation of AI coding tools. - buger/probe</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers</a>: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers - lutzleonhardt/mcpm-aider</li><li><a href="https://github.com/yamadashy/repomix">GitHub - yamadashy/repomix: 📦 Repomix (formerly Repopack) is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) or other AI tools like Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok, and more.</a>: 📦 Repomix (formerly Repopack) is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) o.....
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1346803804561211392)** (135 messages🔥🔥): 

> `VRAM overflow, Phi-4 support, KV cache, New Mac Studio, Sesame TTS` 


- **Spotting VRAM Overflow!**: Users discussed how to identify **VRAM overflow**, noting that if **Dedicated memory** is high (7.5GB+) and **Shared memory** increases, the system is overflowing into RAM, as shown in [this image](https://cdn.discordapp.com/attachments/1110598183144399061/1346803804322009088/VRAM_Overflow.jpg?ex=67cad5c9&is=67c98449&hm=dfd029e741e03b3c482e48ebb01ff246776a4f457db7e3e47b8eadda2c43bb2f&).
   - Context size and **KV cache** impact **VRAM**, and aiming for **90% VRAM** usage was recommended.
- **Phi-4 Audio Modality Still Unsupported**: Members confirmed that **LM Studio** does not support the **audio modality of Phi-4**, as it is not supported in *llama.cpp*.
   - One user added that **multi-modal Phi-4** is also not supported.
- **Sesame AI Debuts Conversational Speech Generation!**: A member shared a link to **Sesame AI**, highlighting its impressive [conversational speech generation demo](https://www.sesame.com), which *sounds like a real human*.
   - Though said to be *open-source*, one member pointed out that [their GitHub repo](https://github.com/SesameAILabs) has no commits yet.
- **QwQ Quantization Quirkiness Quelled!**: Users discussed issues getting **QwQ** models to work in **LM Studio**, sharing a potential [fix](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624) involving prompt parameters.
   - One user noted that after applying the fix and redownloading the **lmstudio-community** version, the model started reasoning properly without outputting gibberish.
- **Android Client for LM Studio Emerges!**: A user announced the creation of an [Android client application for LM Studio](https://github.com/brazer/LmStudioAndroid).
   - It allows you to connect to an **LM Studio server** from your Android device.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.sesame.com">Sesame</a>: We believe in a future where computers are lifelike. Where they can see, hear, and collaborate with us – as we do with each other. With this vision, we&#x27;re designing a new kind of computer.</li><li><a href="https://github.com/SesameAILabs">SesameAILabs</a>: SesameAILabs has 8 repositories available. Follow their code on GitHub.</li><li><a href="https://tenor.com/view/puppy-gif-18530240">Puppy GIF - Puppy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624">Issue with qwq-32b model in lmstudio. · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>: Which version of LM Studio? Example: LM Studio 0.3.11 Which operating system? Mac What is the bug? I get following error when chatting with qwq-32b model &quot;Error rendering prompt with jinja templa...</li><li><a href="https://tenor.com/view/ibm-card-reader-card-reader-ibm-utility-bill-vintage-computer-gif-15507881284984357200">Ibm Card Reader Utility Bill GIF - IBM CARD READER CARD READER IBM - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/brazer/LmStudioAndroid">GitHub - brazer/LmStudioAndroid: The Android application for LM Studio.</a>: The Android application for LM Studio. Contribute to brazer/LmStudioAndroid development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issu">Issue with qwq-32b model in lmstudio. · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>: Which version of LM Studio? Example: LM Studio 0.3.11 Which operating system? Mac What is the bug? I get following error when chatting with qwq-32b model &quot;Error rendering prompt with jinja templa...</li><li><a href="https://photos.app.goo.gl/MDNqL1c7d289oHEs7">New video by Brian Makin</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1346833939288953015)** (309 messages🔥🔥): 

> `Mac Studio M3 Ultra & M4 Max, AMD RX 9070 XT vs Nvidia RTX 5070 Ti, DeepSeek V2.5 236b, SGI machines, NVIDIA's RTX 5090 recall` 


- **Apple's M3 Ultra arrives on Mac Studio**: Apple announced the new [Mac Studios](https://www.apple.com/uk/mac-studio/), featuring the **M3 Ultra** and **M4 Max**, with the M3 Ultra maxing out at **512GB** of RAM.
   - For some reason they didn't include anything regarding **LLM inference** on M4, members assume that it is much slower due to bandwidth difference.
- **Radeon RX 9070 XT squares off with GeForce RTX 5070 Ti**: A YouTube video comparing the **AMD RX 9070 XT** and **Nvidia RTX 5070 Ti** in rasterization and ray tracing shows them trading blows, with Nvidia maintaining a clear lead in **ray tracing**.
   - The 9070 XT sometimes matches the **Nvidia 4080 Super’s** performance at **4K**, offering **~95%** of the 5070 Ti’s performance at **80%** of its **$750** MSRP.
- **DeepSeek, King of the Massive Models**: Members discussed running **DeepSeek V2.5 236b**, noting it makes use of copious RAM for massive initial parameters and runs faster than **Llama 3.3 70b**.
   - One user, [@alexocheema](https://x.com/alexocheema/status/1897349404522078261?t=IiqHPZlhS5AcNXrVQ4moJw), notes that *2 M3 Ultra 512GB Mac Studios with @exolabs is all you need to run the full, unquantized DeepSeek R1 at home*.
- **SGI Machines: Giants of Yesteryear**: Discussion drifted to **Silicon Graphics (SGI)** machines from the late 20th century, renowned for their **large scale** and **shared global memory**.
   - One user recalled an SGI machine from **1998** achieving **33M polygons/second**, dwarfing the fastest PC graphics card's **600k polygons/second** at the time.
- **Nvidia's RTX 5090 faces recall rumble**: A [report](https://wccftech.com/nvidia-geforce-rtx-5090s-are-now-being-recalled-in-europe-over-a-fire-hazard-warning/) said that NVIDIA's GeForce RTX 5090s are being recalled in Europe due to a potential **fire hazard** from the **12V-2x6 power connector**.
   - However, Kitguru [retracted](https://www.kitguru.net/components/graphic-cards/matthew-wilson/dutch-retailer-talks-to-kitguru-and-retracts-rtx-5090-recall-claim/) the claim of a potential product recall of the RTX 50 GPUs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/alexocheema/status/1897349404522078261?t=IiqHPZlhS5AcNXrVQ4moJw">Tweet from Alex Cheema - e/acc (@alexocheema)</a>: Apple&#39;s timing could not be better with this.The M3 Ultra 512GB Mac Studio fits perfectly with massive sparse MoEs like DeepSeek V3/R1.2 M3 Ultra 512GB Mac Studios with @exolabs is all you need to...</li><li><a href="https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/">Apple unveils new Mac Studio, the most powerful Mac ever</a>: Apple today announced the new Mac Studio, the most powerful Mac ever made, featuring M4 Max and the new M3 Ultra chip.</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090s-are-now-being-recalled-in-europe-over-a-fire-hazard-warning/">[Update - Recall Claim Retracted] NVIDIA&#039;s GeForce RTX 5090s Are Now Being Recalled In Europe Over a &quot;Fire Hazard&quot; Warning; Issue Likely Related To The 12V-2x6 Connector</a>: NVIDIA&#039;s GeForce RTX 5090s are now being recalled in Europe, with the risk of a &quot;fire hazard&quot; associated with the 12V-2x6 power connector.</li><li><a href="https://en.m.wikipedia.org/wiki/Raja_Koduri">Raja Koduri - Wikipedia</a>: no description found</li><li><a href="https://tenor.com/view/clock-in-team-wagie-dance-gif-6441791818063703348">Clock In Team Wagie Dance GIF - Clock in team Wagie dance - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=yP0axVHdP-U"> - YouTube</a>: no description found</li><li><a href="https://threadreaderapp.com/thread/1884244369907278106.html">Thread by @carrigmat on Thread Reader App</a>: @carrigmat: Complete hardware + software setup for running Deepseek-R1 locally. The actual model, no distillations, and Q8 quantization for full quality. Total cost, $6,000. All download and part link...</li><li><a href="https://www.apple.com/uk/mac-studio/">Mac Studio</a>: The ultimate pro desktop. Powered by M4 Max and M3 Ultra for all-out performance and extensive connectivity. Built for Apple Intelligence.</li><li><a href="https://www.servethehome.com/bolt-graphics-zeus-the-new-gpu-architecture-with-up-to-2-25tb-of-memor">Bolt Graphics Zeus The New GPU Architecture with up to 2.25TB of Memory and 800GbE</a>: The upcoming Bolt Graphics Zeus GPU architecture offers up to 6x 800GbE links and 2.2TB of memory on a 500W TDP GPU
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1347018948784291902)** (1 messages): 

> `AI model settings, Claude 3.7 Sonnet` 


- **Settings Merge for Speedy Customization**: AI model settings are being merged into one place next to the input on the web version, aiming to make customization faster and more intuitive.
   - A placeholder will be kept in the old settings menu to guide users to the new location, as shown in the [attached screenshot](https://cdn.discordapp.com/attachments/1047204950763122820/1347018948956131420/Screenshot_2025-03-05_at_8.30.27_PM.png).
- **Claude 3.7 Sonnet Gains Pro Access**: **Reasoning with Claude 3.7 Sonnet** will be available to **Pro** users as part of this update.
   - The goal is to make the *"Auto"* setting more powerful so users won't need to manually pick a model.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1346799632742879387)** (322 messages🔥🔥): 

> `Auto model selection, Image source issue, Bulk text modification, Qwen Max model, Attached files in threads` 


- **Auto Model Selection Clarified**: Users discussed the function of 'Auto' in the Perplexity AI application, clarifying that it selects a model from the AI models list, not a separate model itself, and auto search selects model you choose in settings.
   - Some hypothesized it might default to the *basic model*.
- **Perplexity's Image Source Glitch**: Users reported an annoying issue where images used as a source keep reappearing in subsequent messages, even after deletion.
   - Members are frustrated and eager for a fix, with many experiencing this bug.
- **Navigating Text Modification Seas**: A user sought advice on AI resources for modifying large amounts of text, specifically for cross-referencing and combining HTML and JSON files with thousands of links.
   - No specific AI tool was recommended, but the query highlights the demand for AI-powered bulk text processing solutions.
- **Perplexity Users Want Claude 3.7 Thinking**: A user is concerned about Claude Sonnet 3.7 on Perplexity, noting a discrepancy in performance compared to a direct Anthropic account and the need to *jump through hoops* to activate it.
   - Another user reported that **Claude 3.7 hallucinated errors in a simple JSON file**, questioning the model's supposed improvements.
- **Extension Fixes Frustrating Pro Search Display Bug**: Users expressed frustration over a bug where **Pro search doesn't display which model it used**, making it hard to know which model is being used.
   - The **complexity extension** was found to fix this bug, leading some users to try the extension for this reason alone, while some just want Perplexity to merge the fix into the main site.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/monnef/status/1897669944113840293">Tweet from mennof (@monnef)</a>: Exciting times ahead for #Perplexity! A fantastic update is coming soon. 🚀🤖🎉#ai #perplexityAI</li><li><a href="https://docs.perplexity.ai/api-reference/chat-completions">no title found</a>: no description found</li><li><a href="https://www.croxyproxy.com/">no title found</a>: no description found</li><li><a href="https://www.androidauthority.com/google-search-ai-mode-experiment-3532243/">Google supercharges Search with an AI Mode to answer complex questions</a>: The much-awaited AI Mode for Google Search is finally here and it can answer complex, multi-part questions more effectively.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1346879637015232552)** (11 messages🔥): 

> `AI Health Assistant Debut, Nauru sells citizenship, Anthropic Valuation, Meme coins, Early Universe` 


- **Microsoft Debuts AI Health Assistant**: Microsoft debuted an **AI Health Assistant** ([link](https://www.perplexity.ai/page/microsoft-debuts-ai-health-ass-38RGe6B5SVq1nX5OM09k5w3blessed)).
- **Anthropic's Astronomical Ascent to $61.5B Valuation**: Anthropic reached **$61.5B Valuation** ([link](https://www.perplexity.ai/page/microsoft-debuts-ai-health-ass-38RGe6B5SVq1nX5OM09k5w3blessed)).
- **SEC Says Meme Coins Not Securities**: The SEC has declared that **meme coins are not securities** ([link](https://www.perplexity.ai/page/microsoft-debuts-ai-health-ass-38RGe6B5SVq1nX5OM09k5w3blessed)).
- **Nauru Sells Citizenship for Resource**: **Nauru sells citizenship for resource** ([link](https://www.perplexity.ai/page/nauru-sells-citizenship-for-re-mWT.fYg_Su.C7FVaMGqCfQ)).



**Link mentioned**: <a href="https://www.youtube.com/embed/scazXHwpFWQ">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1346837917951332453)** (4 messages): 

> `API Focus, Sonar Pro Model, Search Cost Pricing, Real Time Web Data` 


- **API Focus: Academic vs Community**: A member inquired about setting the focus to something specific like **academic/community** while using the **API**.
- **Sonar Pro Model struggles with real time web data**: A member using the **Sonar Pro model** is struggling with the usage of **real-time web data** returning legacy information that is no longer valid, despite setting *search_recency_filter: 'month'*.
   - The links returned are directly faulty (**parked websites**, **404 pages**).
- **Sonar Pro Model citing number confusing**: A member also using the **Sonar Pro model** is getting good results, but the citing number is confusing because in the replies it starts with **1**, but with the sources list it starts at **0**.
- **Search cost pricing is a mystery**: A member is wondering how to price search cost because the **API** isn't telling them how many searches were used, making it impossible to track spend.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1346814420629721128)** (138 messages🔥🔥): 

> `Richard Sutton talk on Dynamic Deep Learning, OpenAI Agent Pricing, Custom Claude Code with Flash, LLMs for deobfuscation, Boston Dynamics vs Unitree` 


- **Dynamic Deep Learning Dazzles**: Richard Sutton's [talk on Dynamic Deep Learning](https://www.youtube.com/watch?app=desktop&v=75jr5E4OzEE&t=431s) from a few months ago, part of the ICARL Seminar Series, is garnering attention.
   - It discusses the advancements in deep learning and potential future directions.
- **OpenAI's Overpriced Agents?**: OpenAI is considering charging between **$2K to $20K/month** for future agent launches capable of automating coding and PhD-level research, as reported by [The Information](https://www.theinformation.com/articles/openai-plots-charging-20-000-a-month-for-phd-level-agents).
   - SoftBank, an OpenAI investor, has reportedly committed to spending **$3 billion** on OpenAI's agent products this year.
- **Qwen's QwQ-32B: The Quicker Qwen?**: Alibaba released **QwQ-32B**, a new 32 billion parameter reasoning model, rivaling models like DeepSeek-R1, detailing its use of RL to improve performance in math and coding in [their blog post](https://qwenlm.github.io/blog/qwq-32b).
   - It is based on Qwen2.5-Plus and achieves impressive results through RL training.
- **DeepMind Departure: Carlini Chooses Clarity**: Nicholas Carlini announced his departure from Google DeepMind to join Anthropic, citing disagreements with DeepMind's leadership regarding support for high-impact security and privacy research, explained in his [career update](https://nicholas.carlini.com/writing/2025/career-update.html).
   - His work will focus on adversarial machine learning at Anthropic.
- **Jamba 1.6 Jumps into the Scene**: AI21 Labs launched **Jamba 1.6**, a 398B parameter MoE model, claiming it outperforms Cohere, Mistral, and Llama on key benchmarks for private enterprise deployment, as outlined in their [announcement](https://www.ai21.com/jamba/).
   - Concerns were raised about the restrictive [Jamba Open Model License](https://www.ai21.com/jamba-open-model-license/), which may limit its usability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/UnitreeRobotics/status/1896859430517629292">Tweet from Unitree (@UnitreeRobotics)</a>: Kung Fu BOT GAME😘720° Spin Kick - Hear the Impact! Kung Fu BOT Gameplay RAW. (No Speed-Up)(Do not imitate, please keep a safe distance from the machine)#Unitree #Kungfu #EmbodiedAI #SpringFestivalGal...</li><li><a href="https://x.com/steph_palazzolo/status/1897309493744267314">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: NEW w/ @coryweinberg:OpenAI is doubling down on its application business. Execs have spoken with investors about three classes of future agent launches, ranging from $2K to $20K/month to do tasks like...</li><li><a href="https://x.com/cherry_cc12/status/1897366964080926902">Tweet from Chen Cheng (@cherry_cc12)</a>: Who Will Be the Next Member to Join the QwQ Family?Quoting Qwen (@Alibaba_Qwen) Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning mo...</li><li><a href="https://x.com/alibaba_qwen/status/1897361654763151544?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Qwen (@Alibaba_Qwen)</a>: Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning model, e.g., DeepSeek-R1.Blog: https://qwenlm.github.io/blog/qwq-32bHF: https://hu...</li><li><a href="https://x.com/btibor91/status/1897312899124891761?s=46">Tweet from Tibor Blaho (@btibor91)</a>: The Information reports OpenAI plans to charge up to $20,000 per month for advanced AI agents designed for high-level research, aiming for these agents to generate around 20%-25% of revenue long-term-...</li><li><a href="https://fxtwitter.com/jsuarez5341/status/1897356500131336208">Tweet from Joseph Suarez (e/🐡) (@jsuarez5341)</a>: We beat Pokemon Red with online RL! Details here over the next several days. Led by @dsrubinstein. Follow him, me, @DanAdvantage, @kywch500, @computerender  for more!Quoting drubinstein (@dsrubinstein...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897366093376991515">Tweet from Qwen (@Alibaba_Qwen)</a>: Qwen2.5-Plus + Thinking (QwQ) = QwQ-32B . This is how you should use this new model on Qwen Chat!Quoting Qwen (@Alibaba_Qwen) Today, we release QwQ-32B, our new reasoning model with only 32 billion pa...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897361654763151544">Tweet from Qwen (@Alibaba_Qwen)</a>: Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning model, e.g., DeepSeek-R1.Blog: https://qwenlm.github.io/blog/qwq-32bHF: https://hu...</li><li><a href="https://x.com/realDanFu/status/1897726836421149080">Tweet from Dan Fu (@realDanFu)</a>: And we&#39;re not done - excited to announce ThunderGQA ⚡️🐱!Fast fused decode, applied to GQA for Llama & QWEN family models, and 20+% faster than FA3!We&#39;ll be shipping more updates to ThunderMLA...</li><li><a href="https://www.ai21.com/jamba-open-model-license/">Jamba Open Model License Agreement</a>: Read AI21 Lab&#039;s terms of service.</li><li><a href="https://ghuntley.com/tradecraft/">Yes, Claude Code can decompile itself. Here&#x27;s the source code.</a>: These LLMs are shockingly good at deobfuscation, transpilation and structure to structure conversions. I discovered this back around Christmas where I asked an LLM to make me an Haskell audio library ...</li><li><a href="https://x.com/arcprize/status/1897689530901446904">Tweet from ARC Prize (@arcprize)</a>: QwQ-32B on ARC-AGI* Public Eval: 11.25%, $0.036 per task* Semi Private: 7.5%, $0.039 per task</li><li><a href="https://nicholas.carlini.com/writing/2025/career-update.html">
      Career Update: Google DeepMind -> Anthropic
    </a>: no description found</li><li><a href="https://mistral.ai/news/mistral-ocr">Mistral OCR | Mistral AI</a>: Introducing the world’s best document understanding API.</li><li><a href="https://x.com/AI21Labs/status/1897657953261601151">Tweet from AI21 Labs (@AI21Labs)</a>: Today we launched Jamba 1.6, the best open model for private enterprise deployment. AI21’s Jamba outperforms Cohere, Mistral and Llama on key benchmarks, including Arena Hard, and rivals leading close...</li><li><a href="https://huggingface.co/tencent/HunyuanVideo-I2V">tencent/HunyuanVideo-I2V · Hugging Face</a>: no description found</li><li><a href="https://x.com/arcprize/status/1897689538002338187">Tweet from ARC Prize (@arcprize)</a>: Due to the removal of MoE and broader world knowledge, we hypothesize that QwQ-32B’s reasoning capabilities will be narrowly limited to domains it was RL’d on (e.g., math and coding).</li><li><a href="https://www.youtube.com/watch?app=desktop&v=75jr5E4OzEE&t=431s">Dynamic Deep Learning | Richard Sutton</a>: ICARL Seminar Series - 2024 WinterDynamic Deep LearningSeminar by Richard Sutton——————————————————Abstract:Despite great successes, current deep learning met...</li><li><a href="https://github.com/HazyResearch/ThunderKittens/tree/mla/kernels/attn/demo/gqa_decode">ThunderKittens/kernels/attn/demo/gqa_decode at mla · HazyResearch/ThunderKittens</a>: Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.</li><li><a href="https://youtu.be/9_PepvnqIfU?si=sMB90T8__WA19Qrt">TURING AWARD WINNER Richard S. Sutton in Conversation with Cam Linke | No Authorities in Science</a>: “There are no authorities in science,” says A.M. Turing Award winner Richard S. Sutton.In this exclusive conversation, Amii Chief Scientific Advisor Richard ...</li><li><a href="https://fxtwitter.com/BostonDynamics/status/1897298172210225280">Tweet from Boston Dynamics (@BostonDynamics)</a>: We’re designing Atlas to do anything and everything, but we get there one step at a time. See why we started with part sequencing, how we are solving hard problems, and how we’re delivering a humanoid...</li><li><a href="https://www.cnbc.com/2025/03/05/scale-ai-announces-multimillion-dollar-defense-military-deal.html">Scale AI announces multimillion-dollar defense deal, a major step in U.S. military automation</a>: Spearheaded by the Defense Innovation Unit, the Thunderforge program will work with Anduril, Microsoft and others to develop and deploy AI agents.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1346813647544123477)** (41 messages🔥): 

> `LLMs play Diplomacy, Released model fathoms, Happy Birthday, Post training as a service, 14B img2vid model` 


- ****LLMs Negotiate World Domination via Diplomacy****: A member shared a [framework](https://x.com/sam_paech/status/1897078633015206172) for **LLMs** to play the game **Diplomacy** against each other, noting its suitability for experimenting with game theory and testing persuasion, as well as providing code and samples.
   - Diplomacy is a complex board game with a heavy negotiation element and reading the negotiation logs is reportedly *super interesting*.
- ****Model Release Sparks Confusion****: A member linked to a tweet and stated *i cannot fathom they actually released this model.* ([tweet](https://x.com/adonis_singh/status/1896679334200611312)), hinting at bewilderment over a model's release.
   - Another member responded that *other models can generate equally as good if not better greentexts, don’t get the obsession. V3 is pretty good for a modern model, else you can also use old base models.*
- ****Image to Video Model Creates Banana Slug Cinematic Universe****: A member shared a [Replicate link](https://replicate.com/wavespeedai/wan-2.1-i2v-480p) showcasing a **14B img2vid model** generating a realistic **Banana Slug** video from an image.
   - The generated video and source images highlighted the **Banana Slug** mascot of UC Santa Cruz, known for its unconventional nature.
- ****Scaling Laws Still Scaling, Claims Researcher****: A member linked to a tweet from researcher [E Mollick](https://x.com/emollick/status/1897457930833731979?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ) claiming the **First Scaling Law** still holds, noting improvements in **GPQA** scores from **GPT-3.5 Turbo** (**30%**) to **GPT-4 Turbo** (**47%**) to **GPT-4.5** (**70%**).
   - Gary Marcus responded with [cautions](https://x.com/garymarcus/status/1897488996965777575?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ), citing a small dataset, potential data contamination, and unknown augmentation techniques.
- ****Pocket Monster Protection via Pokemon Nicknames****: A member linked to a tweet that referenced the claim that **Claude** became more protective of its **Pokémon** after being instructed to nickname them ([tweet](https://x.com/zswitten/status/1897698670759551378)).
   - Naming Pokemons is the new safety alignment (lol).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/garymarcus/status/1897488996965777575?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Gary Marcus (@GaryMarcus)</a>: @emollick You are looking at a single measure with (just) 198 questions and a serious possibility of data contamination, not to mention unknown data augmentation techniques. I think your conclusion is...</li><li><a href="https://x.com/sam_paech/status/1897078633015206172">Tweet from Sam Paech (@sam_paech)</a>: I made a framework for LLMs to play Diplomacy against each other.Diplomacy is a complex board game with a heavy negotiation element. Good for experimenting with game theory & testing persuasion!It&#39...</li><li><a href="https://x.com/emollick/status/1897457930833731979?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Ethan Mollick (@emollick)</a>: So it looks like the First Scaling Law (the bigger the model the “smarter”) still holds- order of magnitude increases in compute lead to linear improvements in abilityGPT-3.5 Turbo scored 30% on GPQA,...</li><li><a href="https://x.com/paulgauthier/status/1897721567884591402">Tweet from Paul Gauthier (@paulgauthier)</a>: QWQ 32B scored 21% on aider&#39;s polyglot benchmark, using temperature=0.6 and top_p=0.95.https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/adonis_singh/status/1896679334200611312">Tweet from adi (@adonis_singh)</a>: i cannot fathom they actually released this model 😭</li><li><a href="https://x.com/OpenAIDevs/status/1897700857833193955">Tweet from OpenAI Developers (@OpenAIDevs)</a>: ChatGPT for macOS can now edit code directly in IDEs. Available to Plus, Pro, and Team users.</li><li><a href="https://mafia.opennumbers.xyz/">LLM Mafia Game Competition</a>: no description found</li><li><a href="https://x.com/corbtt/status/1897735437340627405">Tweet from Kyle Corbitt (@corbtt)</a>: 🕵️ Can smaller, open-weight models match state-of-the-art reasoning performance? We investigated using GRPO on &#34;Temporal Clue,&#34; surpassing R1, o1, and o3-mini—and nearly matching Sonnet 3.7 a...</li><li><a href="https://www.ucsc.edu/campus/mascot/)">Our Mascot: Sammy the Banana Slug &#8211; UC Santa Cruz</a>: no description found</li><li><a href="https://x.com/zswitten/status/1897698670759551378">Tweet from Zack Witten (@zswitten)</a>: My favorite Claude Plays Pokémon tidbit (mentioned in @latentspacepod) is that when @DavidSHershey told Claude to nickname its Pokémon, it instantly became much more protective of them, making sure to...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1347075957730574387)** (5 messages): 

> `Chinese Lewd R1 Aha Moment, DeepSeek Videos on bilibili Comments, Reinforcement Learning History by Schmidhuber` 


- **Chinese Lewd R1 "Aha" Moment**: A user shared a [post](https://fxtwitter.com/teortaxesTex/status/1897508611019932133) claiming *"The Chinese have lewded the R1 “Aha moment”, it's over."*
- **"Sapiosexual for Whale" Comments on DeepSeek Videos**: A user pointed out that on **DeepSeek** videos on bilibili, commenters often express being *"sapiosexual for 🐋"*, referencing [this comment](https://x.com/layer07_yuxi/status/1897512187264119129).
- **Schmidhuber Shares Reinforcement Learning History**: A user shared a link to a [tweet by SchmidhuberAI](https://x.com/SchmidhuberAI/status/1897569590357402076) providing background to **reinforcement learning** in Sec. 17 of the *"Annotated History of Modern AI and Deep Learning:"* [link to paper](https://people.idsia.ch/~juergen/deep-learning-history.html#rl).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/teortaxesTex/status/1897508611019932133">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: The Chinese have lewded the R1 “Aha moment”, it&#39;s over</li><li><a href="https://x.com/SchmidhuberAI/status/1897569590357402076">Tweet from Jürgen Schmidhuber (@SchmidhuberAI)</a>: @RichardSSutton Some background to reinforcement learning in Sec. 17 of the &#34;Annotated History of Modern AI and Deep Learning:&#34; https://people.idsia.ch/~juergen/deep-learning-history.html#rl</li><li><a href="https://x.com/layer07_yuxi/status/1897512187264119129">Tweet from Yuxi on the Wired (@layer07_yuxi)</a>: @teortaxesTex if you go to DeepSeek videos on bilibili, and read the comments you&#39;ll often see people saying they are &#34;sapiosexual for 🐋&#34;
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1346968004331573309)** (6 messages): 

> `Schmidhuber, Deep RL, Richard Sutton Turing Award` 


- **Schmidhuber Congratulates Richard Sutton, Hints at Cult Leadership**: [Jürgen Schmidhuber](https://x.com/SchmidhuberAI/status/1897406236896977388) congratulated **Richard Sutton** and **Andy Barto** on their **Turing Award**, with a user quipping *"Cult leader game recognizes cult leader game."*
- **Pieter Abbeel's Deep RL Tutorial Still Relevant**: **Pieter Abbeel** re-posted his [Basics of Deep RL tutorial](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0), claiming that *"I am still very happy with"* it.
   - One user agreed, claiming that *"between this and the sergey levine course think you learn all of RL"* and noting the relevance of **David Silver's UCL course**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/SchmidhuberAI/status/1897406236896977388">Tweet from Jürgen Schmidhuber (@SchmidhuberAI)</a>: Congratulations to @RichardSSutton and Andy Barto on their Turing award!</li><li><a href="https://x.com/pabbeel/status/1897437838180061204">Tweet from Pieter Abbeel (@pabbeel)</a>: Basics of Deep RL tutorial I am still very happy with, as good a day as any to re-post :)https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1346938335108661289)** (9 messages🔥): 

> `Reinforcement Learning, Scientific AI, LLMs, Pre-training` 


- **RL System Beats Pokémon Red**: A reinforcement learning system developed by HazyResearch has successfully completed the game **Pokémon Red** using a policy under **10M** parameters, **PPO**, and novel techniques, as detailed in their [blog post](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla).
   - The team also released related resources, including [code](https://github.com/HazyResearch/ThunderKittens/blob/mla/kernels/attn/demo/mla_decode/template_mla_decode.cu) and links to their previous work on **TK Part 1** and **TK Part 2**.
- **ThunderMLA Speeds Up LLM Inference**: HazyResearch introduces **ThunderMLA**, a fused "megakernel" for decode, which they claim is **20-35% faster** than DeepSeek's **FlashMLA** on diverse workloads by implementing simple scheduling tricks, according to their [blog post](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla).
   - The initial release focuses on attention decoding, but they believe it has wider applications.
- **Dario's Loving Grace**: A member shared [a controversial take](https://thomwolf.io/blog/scientific-ai.html) at an event, expressing doubts about **AI's** ability to compress the scientific discoveries of the 21st century into a mere 5-10 years, as envisioned in Dario's "Machine of Loving Grace".
   - This member argued that what we'll actually get is *“a country of yes-men on*.
- **LLMs rediscover Einstein's Finding?**: A member proposed a method to test if **LLMs** have sufficient creativity to make breakthrough discoveries by pre-training a model on documents before 1905 and post-training it to use scale inference compute, as detailed in [this tweet](https://x.com/rosstaylor90/status/1897694319382798681).
   - The model would then be prompted to explain the **photoelectric effect** and reconcile **Maxwell's equations**, with current models used to verify if the generations match **Einstein’s** solutions.
- **Overview of What's new in pre-training**: A member shared slides from a talk explaining 'What's new in pre-training' given to colleagues at **Hugging Face**, providing an overview of the pre-training field [here](https://drive.google.com/file/d/1lw0hfxHAshcKupxMW51F5zeV1PeDe34w/view).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla">ThunderMLA: FlashMLA, Faster and Fused-er!</a>: no description found</li><li><a href="https://thomwolf.io/blog/scientific-ai.html">🔭 The Einstein AI model</a>: no description found</li><li><a href="https://x.com/rosstaylor90/status/1897694319382798681">Tweet from Ross Taylor (@rosstaylor90)</a>: - Pre-train a model on all documents before 1905.- Post-train like R1 so it can use scale inference compute to think widely about a problem.- Prompt for an explanation of the photoelectric effect, rec...</li><li><a href="https://x.com/eliebakouch/status/1897665636710400397">Tweet from elie (@eliebakouch)</a>: Gave a talk earlier today to explain &#39;What&#39;s new in pre-training&#39; to my @huggingface colleagues. I&#39;m sharing the slides here if you&#39;re interested in a humble overview of the pre-tr...</li><li><a href="https://drive.google.com/file/d/1lw0hfxHAshcKupxMW51F5zeV1PeDe34w/view">Pre-Training SOTA.pdf</a>: no description found</li><li><a href="https://x.com/dsrubinstein/status/1897351145485648309?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from drubinstein (@dsrubinstein)</a>: Excited to finally share our progress in developing a reinforcement learning system to beat Pokémon Red. Our system successfully completes the game using a policy under 10M parameters, PPO, and a few ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1346996584667152386)** (10 messages🔥): 

> `RLHF Book, Lecture series videos` 


- **RLHF Book Link Shared**: A member shared a link to the **RLHF book** for people looking for it: [https://rlhfbook.com/book.pdf](https://rlhfbook.com/book.pdf).
- **Lecture Series on the Horizon**: A member mentioned they are tentatively planning to do a **lecture series** over the summer, with **1 video per chapter**.
   - They added they *gotta make the marketing engine go brrr once preorder button exists*.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1346848634926272582)** (9 messages🔥): 

> `Stargate Project, Content Gating, Data Protection` 


- **Stargate Project Paid Off by Ads?**: The **Stargate Project** is purportedly being funded through *completely unbiased and unobtrusive ads*.
   - This announcement elicited some skepticism from the community.
- **Gating Content is good?**: As **AI models** become more powerful, businesses need to *gate their content* to avoid becoming obsolete, as [Ben Thompson](https://stratechery.com/) argued.
   - Newspapers failed to do this and now *accept whatever deal Sam offers them*, but valuable data troves like **YouTube** and **GitHub** must be protected at all costs.
- **Data Protection is Key**: If existing providers can't make deals with data providers, it would create a ceiling to scaling AI.
   - Specifically, if **Microsoft** blocks **OpenAI's 20K/month coding agent**, its less useful because it's harder to acquire data.


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1347161977952407603)** (32 messages🔥): 

> `Anthropic's Recommendations, Nationalizing Labs, DeepSeek Exports, China AMD GPUs, H20 Controls` 


- **Anthropic Advocates for AI Action Plan**: Anthropic shared their [recommendations](https://www.anthropic.com/news/anthropic-s-recommendations-ostp-u-s-ai-action-plan) to the OSTP U.S. AI Action Plan, which includes **nationalizing AI labs with high security clearance**.
   - A member shared a link to [Anthropic's Response to OSTP](https://assets.anthropic.com/m/4e20a4ab6512e217/original/Anthropic-Response-to-OSTP-RFI-March-2025-Final-Submission-v3.pdf) calling it *kinda cringe but consistent with their viewpoint*.
- **PhD-Level Models Saturate GPQA**: A member noted that now that we saturated GPQA and models are clearly **PhD-level**, they must have *intellectual capabilities matching or exceeding that of Nobel Prize winners across most disciplines—including biology, computer science, mathematics, and engineering*.
   - They remarked that H20 restrictions would be wild given the [5 repositories](https://link.to.five.repos) and their utlization.
- **DeepSeek Exports Spark Debate**: A member suggested that the shift in tone from *deepseek is meh* to *it's evidence to stop imports* aligns with [Dario Amodei's stance on export controls](https://darioamodei.com/on-deepseek-and-export-controls#export-controls).
   - Anthropic needs to deliver results by **2027**.
- **AMD GPUs may become China's Open Source Savior**: A member speculated that if China is restricted to **AMD cards**, they might fully develop the code and open source it.
   - Another memeber joked that this was *a prayer to the oss gods for working amd gpusfor deep learning*.
- **Potential H20 Controls**: A member hopes there are no H20 controls and that [DeepSeek gets a H100 farm](https://link.to.h100.farm) due to **B200s being out**.
   - Dario is against H20 bans


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1346825050518720553)** (67 messages🔥🔥): 

> `Touhou games and AI, RL gyms Starcraft gym and Minetest gym, Unified memory discussion, Thunderbolt 5 for distributed inference/training, Deepseek-R1` 


- **Touhou Games Inspire AI Aspirations**: Multiple members expressed enthusiasm for **Touhou games** as an inspiration for getting into **AI** and **GPU programming**.
   - One member mentioned wanting to train a model to play Touhou, noting that it's becoming easier with **RL** and using score as the reward.
- **Unified Memory Perks Interest**: Discussion sparked about unified memory in light of the **M3 Ultra** announcement, pondering its performance characteristics regarding **GPU memory bandwidth** and **CPU/GPU memory transfers**.
   - The consensus was that the **Apple M series** does address the same memory, and there was excitement over **Thunderbolt 5** for distributed inference/training between Mac Minis/Studios.
- **Tenstorrent Quietbox Incoming**: A member announced their company is acquiring a **Tenstorrent quietbox** and offered to share their experiences once they've had a chance to use it.
   - Another member shared a link to the [Tenstorrent GitHub](https://github.com/tenstorrent), highlighting projects like **metal** and **buda** for using their hardware at different levels.
- **Is Langchain K.O.ed?**: The community debated the merits and potential demise of **Langchain**, with some expressing negative sentiment and questioning its abstraction.
   - While one member hoped it was *dead dead*, another acknowledged its role in getting people thinking about composing things early, despite finding it a *terrible library*.
- **GPU GTC Promo Code?**: One member asked about the **GPU MODE coupon code for GTC**.
   - Another member answered the **GPUMODE**.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1346941022755881000)** (9 messages🔥): 

> `tl.gather in Triton, PagedAttention in Triton, Bias addition optimization, Input size and performance issues, NVIDIA's Cooperative Vector` 


- **Triton's `tl.gather` Disappears!**: A user reported an `AttributeError` when trying to use `tl.gather` in Triton, even though it's in the documentation, and [this issue was reported on GitHub](https://github.com/triton-lang/triton/issues/5826).
   - A member suggested building Triton from the master branch and uninstalling the version provided with PyTorch to resolve the issue.
- **PagedAttention Resources Sought**: A member requested resources on how to recreate **PagedAttention** in Triton.
   - No specific resources were provided in the excerpt.
- **Fused Layer Bias Boost: Broadcast or Kernel?**: A user asked about the better and faster way to add a bias after a fused layer in Triton.
   - The options were initializing the output with bias after broadcasting or adding it in the kernel; however, no definitive answer was provided.
- **Triton Tensorscores thrive on 32 Multiples**: A user encountered performance issues when the input size was not a multiple of **32** and found that padding the input manually improved speed, revealing **NVIDIA tensorcores** only support certain work group sizes.
   - It was recommended to keep the padding and use masks for reading the compute output, as hardware computes padded values faster by utilizing its computations better; it will be dispatched as elementwise-computations otherwise.
- **Cooperative Vector Cometh to NVIDIA**: **NVIDIA** announced support for **Cooperative Vector**, which supposedly will support smaller compute chunks on Tensor cores and is described in the [Vulkan Docs](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_NV_cooperative_vector.adoc).
   - One member thinks *3 threads working on 21-large groups can be automatically joined into a single 64-large kernel call*, and the Raytracing Optix extension has support for float8 for their cooperative vector.



**Link mentioned**: <a href="https://github.com/triton-lang/triton/issues/5826">Cannot call tl.gather · Issue #5826 · triton-lang/triton</a>: Describe the bug When I run the following code I get an exception: AttributeError: module &#39;triton.language&#39; has no attribute &#39;gather&#39; import triton.language as tl tl.gather I&#39;ve in...

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1346800993525960704)** (19 messages🔥): 

> `CUDA compiler optimization, CUDA OpenGL interop segfault, Overlapping kernel execution with NCCL, Memory transaction size, GTC talk on maximizing memory bandwidth` 


- **CUDA Compiler Strikes Again, Optimizes Away Unread Data**: A user found that the **CUDA compiler** optimized away memory writes because the data was never read, and AI's suggestion to write to a large memory range to prevent optimization failed until a read was added to the array.
   - Another member confirmed that *the compiler will optimise it away since there are no reads*, and that adding a read from the array will cause the compiler to report an error.
- **OpenGL Interop Segfault Debugging Debacle**: A user encountered a segfault in their **CUDA OpenGL interop code** on a laptop, specifically at the `cudaGraphicsMapResources` call, while `cudaGraphicsGLRegisterImage` returned `cudaErrorUnknown`.
   - Another member suggested that the problem could be that **OpenGL wasn't using the GPU**, which the user confirmed fixed the issue after inspecting [this forum post](https://devtalk.nvidia.com/default/topic/534608/cuda-setup-and-installation/cudaerrorunknown-on-cudaopengl-interop/).
- **Stream Teamwork: Hiding Communication Overhead**: A member asked about overlapping kernel execution time with **NCCL** collective operations by placing them in separate streams.
   - The goal is to hide the communication overhead by running vector addition in one stream and `allreduce` in another, but it is unclear from the discussion if this is a viable strategy.
- **Warp Speed Data: Optimizing Memory Transactions**: A user asked if the recommendation to access data for a warp in a single memory transaction (128 bytes) refers to the **cacheline size** or if there are additional considerations, quoting from the [CUDA documentation on device memory accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses).
   - Another member noted that it's also recommended to do **128-bit transfer per thread** (e.g. float4), resulting in a 512-byte transfer for a warp, which would be serviced by 4 128-byte memory transactions, perhaps for more efficient instruction execution.
- **GTC Talk Teaches Techniques**: A member posted a shameless plug for an upcoming **GTC talk** dedicated to maximizing memory bandwidth and transactions with details available [here](https://www.nvidia.com/gtc/session-catalog/?regcode=pa-srch-goog-157409-prsp&ncid=pa-srch-goog-157409-prsp&deeplink=audience-recommend--1&tab.catalogallsessionstab=16566177511100015Kus&search=%22cuda%20techniques%22#/session/1727709012449001X6PZ).
   - The speakers of the talk *know more about this topic than anyone else* according to the member.



**Link mentioned**: <a href="https://www.nvidia.com/gtc/session-catalog/?regcode=pa-srch-goog-157409-prsp&ncid=pa-srch-goog-157409-prsp&deeplink=audience-recommend--1&tab.catalogallsessionstab=16566177511100015Kus&search=%22cuda%20techniques%22#/session/1727709012449001X6PZ">NVIDIA #GTC2025 Conference Session Catalog</a>: Experience GTC 2025 In-Person and Online March 17-21, San Jose.

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1346900912525934773)** (32 messages🔥): 

> `Torch C++ Interface, Extending OffloadPolicy, use_reentrant in Activation Checkpointing, TorchBind API, Model-Based RL subgames` 


- **Torch C++ Methods Lack Schema-Like Functions**: A member inquired about why methods in the torch C++ interface library lack schema-like functions, and whether a proposal to extend `OffloadPolicy` would be accepted as a PR.
   - A staff member suggested converting the methods to functions and annotating them with aliases, but wondered the use case for it, to which the member replied it was for creating subgames to solve for **model-based RL**.
- **`use_reentrant` True/False in Activation Checkpointing**: Members discussed the meaning of the `use_reentrant` parameter in PyTorch's [checkpointing documentation](https://pytorch.org/docs/stable/checkpoint.html), with one humorously stating *they always set it to False*.
   - It was clarified that `use_reentrant=True` is the old implementation of activation checkpointing, while `use_reentrant=False` is the newer, superior one, and that **Transformers doesn't play nice with activation checkpointing for some reason** ([issue #23808 on HuggingFace/Transformers](https://github.com/huggingface/transformers/issues/23808)).
- **Transformers Issue Plagues Users**: A member described spending *a day and a half slamming their head into things* to resolve issues with transformers not playing nicely with activation checkpointing.
   - Another member shared a similar experience, calling it *a rite of passage*, and expressing hope for a warning message to be added, and that they will **probably push it to transformers soon enough**.
- **TorchBind API for Custom C++ Classes**: A member confirmed using the **TorchBind API** to bind a custom C++ class to Python, in order to create game trees to solve in tandem with NN predictions.
   - They clarified that the **TorchScript** mechanism was created for **TorchScript Inference**, but it has not been well supported outside of that use case.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/checkpoint.html">torch.utils.checkpoint &mdash; PyTorch 2.6 documentation</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/issues/23808">Why don&#39;t we set use_cache=False in default when training? · Issue #23808 · huggingface/transformers</a>: Feature request Let&#39;s take GPT-2 as an example, in the current implementation (modeling_gpt2.py: Line 856~861): if self.gradient_checkpointing and self.training: if use_cache: logger.warning_once(...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1347158288348745728)** (6 messages): 

> `ThunderMLA, DeepSeek MLA, Modular's Democratizing AI Compute, CUDA Alternatives` 


- **ThunderMLA Zaps DeepSeek's FlashMLA**: HazyResearch introduced **ThunderMLA**, a fused "megakernel" for decode that is **20-35%** faster than **DeepSeek's FlashMLA** on diverse workloads, using scheduling tricks.
   - The release focuses on attention decoding, with code available [here](https://github.com/HazyResearch/ThunderKittens/blob/mla/kernels/attn/demo/mla_decode/template_mla_decode.cu) and related links including [TK Part 2](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2), [TK Part 1](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk), and [Brr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk).
- **Modular Deconstructs Democratizing AI Compute**: Modular's Part 5 of "Democratizing AI Compute" series critically examines why previous **CUDA alternatives** like **OpenCL**, **SYCL**, and **OneAPI** failed despite aiming to democratize AI compute.
   - The failure stems from challenges of "[open coopetition](https://en.wikipedia.org/wiki/Open_coopetition)" and management missteps, as outlined in the series starting with [Part 1: DeepSeek’s Impact on AI](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-).
- **CUDA Graph Vs ThunderMLA**: A member inquired about how **ThunderMLA** contrasts with a **CUDA graph**.
   - Another member responded, *since instructions are passed as a tensor to the kernel it's not dependant on any CPU operations so it should just work.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla">ThunderMLA: FlashMLA, Faster and Fused-er!</a>: no description found</li><li><a href="https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives?utm_source=x&utm_campaign=community">Modular: Democratizing AI Compute, Part 5: What about CUDA C++ alternatives?</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1347138627288829963)** (2 messages): 

> `Triton tl.sort() problem, Flask API Authentication` 


- **Triton Sort Stalls Novice**: A user reported that `tl.sort()` in their **Triton** program doesn't sort the input tensor as expected, with the output remaining the same as the input.
   - They provided a code snippet utilizing `tl.sort()` within a Triton kernel, seeking assistance in identifying the cause of the sorting failure.
- **Flask API Authentication Frustrates User**: A user inquired about recommendations for implementing **authentication** in a web application using a **Flask API**.
   - They are seeking advice on suitable authentication methods for securing their Flask-based web API.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1346837200326889603)** (14 messages🔥): 

> `SSH pain points, Better GPU providers, Nitrokey, SoloKey, Yubikey` 


- **Users seek better GPU Providers**: Users discussed the pain points of **SSH'ing** and finding better GPU providers for their **M2** or considering buying a **Blackwell GPU**.
   - One user mentioned having an **RTX 3050** and a **GFX90c**.
- **Garage PC accessed via VS Code**: A member described their setup using a **PC in their garage**, **VS Code**, and an identity file for passwordless **SSH access**.
   - They also recommended using [Mutagen](https://mutagen.io/) to sync files between the laptop and the servers.
- **Nitrokey offers more security**: A member suggested using a **Nitrokey**, **SoloKey**, or **Yubikey** for added security.
   - They stated it's *relatively cheap, still easy to use, offers way more security and can be used on other accounts as well*.
- **PC finds Home under the Kitchen Sink**: A member shared how they put a **PC** underneath their **kitchen sink** due to lack of space, though their *wife was not happy*.
   - The reason was there was *a power outlet nearby*.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1346903165072576604)** (3 messages): 

> `Tenstorrent, LlamaIndex, Koyeb, AI Infrastructure Meetup, Next-Gen hardware` 


- **Tenstorrent, LlamaIndex, and Koyeb to Host AI Infrastructure Meetup**: The **Tenstorrent**, **LlamaIndex**, and **Koyeb** teams are hosting a small meetup tonight in SF Downtown around **AI Infrastructure** and **Next-Gen hardware** at [https://lu.ma/ruzyccwp](https://lu.ma/ruzyccwp).
- **Meet the Teams**: The meetup is a chance to meet the **Tenstorrent** and **Koyeb** teams and discover how their collaboration delivers superior performance for cost compared to traditional GPUs.
   - It is designed to connect AI developers and allow them to learn about cutting-edge innovations in AI infrastructure.
- **SF Meetup Planned**: A member indicated they'd be in Portland, and then attending GDC in two weeks, with interest in meeting up in SF then.



**Link mentioned**: <a href="https://lu.ma/ruzyccwp">Next-Gen AI Infra with Tenstorrent &amp; Koyeb @LlamaIndex · Luma</a>: Join us for a special evening as we kick off a groundbreaking collaboration between Tenstorrent and Koyeb with our friends from LlamaIndex.This meetup is a…

  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1346881365265612883)** (2 messages): 

> `Reshaping vs Permuting, Triton Kernel Permutation, FPINT Dimension Right Shift` 


- **Reshaping and Permuting Aren't Identical Twins**: Reshaping an `M x N` matrix to `N x M` in row-major order keeps element order, but permuting, like **transposing**, changes the order.
   - A user highlighted that they are *definitely not equivalent*.
- **Triton Kernel Needs Permutation, Not Just Reshape**: In a Triton kernel, permuting is necessary because data is loaded in the shape `(32, FPINT, GROUP)` with each **GROUP** list having the same dequantized value expanded.
   - The matrix must be transposed into `(32, GROUP, FPINT)` so the right shift `offset_q >> over) & mask` can be applied to the last dimension to produce dequantized values.
- **FPINT Dimension Needs a Right Shift**: The user explains that `(offset_q >> over) & mask` is only valid on the last dimension, implying that `offset_q.shape[-1] == over.shape[0]`.
   - The user attempted and failed to right shift the `offset_q.shape[1]` dimension by setting `over = 4*tl.arange(8).reshape((1, 8, 1))`.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1346862321326362636)** (7 messages): 

> `Radeon GPU Profiler, ROCm on Linux, rocprofilerv2 ATT plugin, rocclr, PAL Backend` 


- **Quest for RGP-Like Instruction Timing on ROCm Linux**: A user inquired about a tool to replicate the instruction timing tab in **Radeon GPU Profiler (RGP)** with **ROCm on Linux**.
   - Unfortunately, it's been stated that **RGP** is only usable on Windows, and a suggestion was made to compile **rocCLR** with the **PAL backend** on Linux, but its functionality is unconfirmed.
- **rocprofilerv2 ATT Plugin Fails to Fire**: The user mentioned trying the **ATT plugin** of **rocprofilerv2**, but it *doesn't seem to work properly*.
   - Despite the documentation suggesting latency per instruction should be available, others have confirmed they *couldn't get it to work* either, perhaps due to issues with the [RDNA4 instruction set architecture](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1346801537447366717)** (17 messages🔥): 

> `Shared Memory Allocation, Python Linting Warnings, CUDA Compatibility Issues (12.1 vs 12.4/12.6), Github Issue #149` 


- ****Shared Memory Allocation** Deep Dive**: A member referenced the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications) regarding **shared memory** availability overall and per thread block.
   - It was noted that determining the amount of shared memory needed involves considering all `T.alloc_shared` calls and multiplying the dimensions by the **dtype itemsize**.
- ****Python Linting** gets a free pass**: A member acknowledged warnings are primarily due to **Python linting** in their Pythonic DSL.
   - They admitted they *haven’t found a simple way to bypass the lint issue yet*, so they will ignore the warnings for now.
- ****CUDA Conundrums**: 12.1 good, 12.4 bad**: A user reported an issue with **CUDA 12.4** on an **RTX 4070 laptop** where code that worked on **CUDA 12.1** failed.
   - Downgrading to the **cu121 nightly build** from [tile-ai.github.io](https://tile-ai.github.io/whl/nightly/cu121/) temporarily resolved the problem, despite compatibility for cuda >= 11.0.
- ****Github Issue Lodged**: CUDA 12.4/12.6 Nightmare**: A Github issue was created on the Tilelang repository to address the element mismatch error during matmul operations on **CUDA 12.4** and **12.6**.
   - See [Github issue #149](https://github.com/tile-ai/tilelang/issues/149) for full details.



**Link mentioned**: <a href="https://github.com/tile-ai/tilelang/issues/149">Mismatched elements when performing matmul on CUDA 12.4/12.6 · Issue #149 · tile-ai/tilelang</a>: Describe the Bug I ran the simple matmul code below, and I got error AssertionError: Tensor-likes are not close! The code works fine on CUDA 12.1, but not on CUDA 12.4/12.6. The number of mismatche...

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1347025757897232435)** (1 messages): 

> `M3 Ultra, Unified Memory, Creative Uses of Unified Memory` 


- **M3 Ultra's Debut Sparks Unified Memory Musings**: Members are discussing the implications of the **M3 Ultra** announcement, speculating how unified memory might foster creative applications.
- **Unified Memory's Creative Potential**: The discussion focuses on potential creative uses of **unified memory** in light of the **M3 Ultra** announcement.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1346817771752656926)** (20 messages🔥): 

> `ARC-AGI Competition, QwQ-32B Release, Reasoning-Gym Datasets, LADDER Framework` 


- **ARC-AGI Participation Planned**: Members are planning to participate in the **ARC-AGI competition**, following up on previous involvement as seen on the [open-thought GitHub profile](https://github.com/open-thought/reasoning-gym).
   - The competition serves as a **new evaluation target** for reasoning models.
- **Qwen's QwQ-32B Rivals DeepSeek-R1**: **Alibaba Qwen** released [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b), a new **32 billion parameter reasoning model** that rivals cutting-edge models like **DeepSeek-R1**.
   - The model showcases impressive results in **math and coding**, demonstrating continuous improvement through **RL training** and competing with larger MoE models.
- **Reasoning-Gym Seeks Datasets for Centurion Celebration**: The **reasoning-gym** project is aiming to reach **100 datasets** and is seeking proposals, with **97 datasets** currently available.
   - Two new datasets have been proposed via [pull request 272](https://github.com/open-thought/reasoning-gym/pull/272) and [pull request 273](https://github.com/open-thought/reasoning-gym/pull/273).
- **LADDER Framework Learns Through Recursive Simplication**: A paper introduced the **LADDER framework** ([arxiv link](https://arxiv.org/abs/2503.00735)), which enables Large Language Models to autonomously improve their problem-solving capabilities through self-guided learning by *recursively generating and solving progressively simpler variants of complex problems*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/alibaba_qwen/status/1897361654763151544">Tweet from Qwen (@Alibaba_Qwen)</a>: Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning model, e.g., DeepSeek-R1.Blog: https://qwenlm.github.io/blog/qwq-32bHF: https://hu...</li><li><a href="https://arxiv.org/abs/2503.00735">LADDER: Self-Improving LLMs Through Recursive Problem Decomposition</a>: We introduce LADDER (Learning through Autonomous Difficulty-Driven Example Recursion), a framework which enables Large Language Models to autonomously improve their problem-solving capabilities throug...</li><li><a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">ARC-AGI Without Pretraining</a>: no description found</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/273">Add Modulo Grid Task by Miserlou · Pull Request #273 · open-thought/reasoning-gym</a>: Here&amp;#39;s an all original one for number 100 (maybe)?This is an ARC-ish task for mathematical explanatory reasoning. It generates a binary grid based on a hidden mathematical function based aroun...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/272">[Env] Game of Life Halting Prediction by Miserlou · Pull Request #272 · open-thought/reasoning-gym</a>: This is a variant of the Game of Life task, which rather than trying to test the algorithmic simulation, tests the ability of the model to do explanatory reasoning of the board. The idea is that a ...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1346811138737700926)** (6 messages): 

> `AI directions for programmers, CUDA WeChat groups, TileLang DSL, Triton sequence padding` 


- **AI Avenues for Coders Spark Curiosity**: A member inquired about suitable **AI directions** for ordinary programmers with a low barrier to entry and 3 years of game development experience.
   - They also asked if there are any **CUDA WeChat groups** available.
- **TileLang DSL catches Attention**: A member shared a link to **TileLang**, a domain-specific language for streamlining the development of high-performance GPU/CPU/Accelerators kernels: [tile-ai/tilelang](https://github.com/tile-ai/tilelang).
   - TileLang aims to simplify the creation of efficient kernels for diverse hardware platforms.
- **Triton Padding Performance Probed**: A member asked why **Triton's performance** slows down when the input sequence length is not a multiple of 32.
   - They also inquired about modifying the code if **input sequences are not padded**.



**Link mentioned**: <a href="https://github.com/tile-ai/tilelang">GitHub - tile-ai/tilelang: Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels</a>:  Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels - tile-ai/tilelang

  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1346857475802009693)** (20 messages🔥): 

> `Modal Runners, GPU Leaderboards, Submission Errors` 


- **Modal Runners Ace GPU Leaderboards**: Several test and benchmark submissions using **Modal runners** succeeded across different leaderboards and **GPUs** including **A100**, **H100**, **T4**, and **L4**.
   - Submissions were made to leaderboards such as `histogram`, `grayscale`, and `prefixsum`.
- **Submission Script Header Mismatch**: The Cluster-Bot reported that the leaderboard name specified in the command didn't match the one in the submission script header, resulting in submissions being rerouted to `histogram` or `grayscale`.
   - This issue indicates a potential discrepancy between the intended leaderboard and the actual submission target.
- **Grayscale Gets T4 Boost**: Multiple test submissions to the `grayscale` leaderboard succeeded using **T4 GPUs**, indicating a focus on this configuration.
   - These submissions were followed by benchmark and leaderboard submissions, suggesting ongoing testing and optimization efforts for `grayscale` on **T4 GPUs**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1347237634526675016)** (1 messages): 

> `Timeout durations` 


- **Timeouts Get Double-Down**: Members report that all **timeouts** have been **doubled** by the admins.
   - Users are asked to report back whether they run into further **timeout issues**.
- **Timeout issue reporting**: Members are now requested to report any timeout issues they encounter.
   - The request follows the doubling of all timeout durations, and aims to proactively identify potential problems.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1346826892803571724)** (217 messages🔥🔥): 

> `Mojo's dynamism, Mutating classes in Mojo, Python vs Mojo, Async Django drawbacks` 


- **Mojo is a superset of Python, no?**: Although it was said that Mojo is a superset of Python, that messaging was revised because even **C++** isn't a superset of **C**, and it implies support for all kinds of messy things like mutating class objects.
   - Being a superset of a language developed in 90's would be for Mojo like a muzzle as it couldn't fully utilize features from programming languages design, greatly evolved throughout these years.
- **Why shouldn't Mojo have mutating class objects?**: It was questioned why Mojo shouldn't have mutating class objects, because that is a core dynamic language feature, but the answer is because Mojo isn't a dynamic language.
   - Members also discussed how dynamism is a mistake in a lot of contexts as **JS** got **TS** and **Python** got **type hints** and the first thing those systems do is lock many things like this away.
- **Rust provides the most ergonomic approach**: Members determined that the most success they've had on dynamism, has been to sidestep the computationally hard part by trying to offer an alternative that is comparably ergonomic but computationally easier.
   - Giving people what they need is better than what they want, C devs still complain to this day about Rust being overbearing and unnecessary, and yet every so often people prove it is necessary by using their freedoms to create new **CVE's**.
- **Full hashtable dynamism is the future?**: It was discussed that this proposal provides the framework, of whether Mojo should have dynamism like Python's full-hashtable everything is changeable dynamism, or a more restricted version with a switch to enable Python's for compatibility.
   - One member stated, *I personally hope "level 2" which is "Partial dynamism" would be enough. But maybe "Full hashtable dynamism" is what modular will go for anyways.*
- **Async Django? Hard Pass**: One member stated that they avoid async Django.
   - Another member chimed in that, *The original intent of making Mojo "Pythonic" was to bridge the gap between AI researchers, and the people deploying their models.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modular/max/blob/main/mojo/proposals/mojo-and-dynamism.md">max/mojo/proposals/mojo-and-dynamism.md at main · modular/max</a>: The MAX Platform (includes Mojo). Contribute to modular/max development by creating an account on GitHub.</li><li><a href="https://github.com/python/cpython/blob/052cb717f5f97d08d2074f4118fd2c21224d3015/Include/longobject.h#L16">cpython/Include/longobject.h at 052cb717f5f97d08d2074f4118fd2c21224d3015 · python/cpython</a>: The Python programming language. Contribute to python/cpython development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1346876472148623503)** (5 messages): 

> `Mojo/Python project benchmarking, Mojo/Python project folder structure, Python.add_to_path alternatives, Symlink alternatives in Mojo tests` 


- **Mojo Binary Performance Degraded in Python venv**: A user noted that running **Mojo binary files** within an active **Python virtual environment** significantly reduces performance, even for Mojo files that do not import any Python modules.
   - The user sought insights into why this occurs, expecting Mojo binaries without Python imports to be unaffected by the Python venv.
- **Guidance for Mojo/Python project folder structure**: A user is seeking guidance on structuring a **Mojo/Python project**, importing standard Python libraries and custom Python modules.
   - They use `Python.add_to_path` extensively and have implemented a Symlink in the `tests` folder.
- **Alternatives for Python.add_to_path sought**: A user is seeking alternatives to `Python.add_to_path` for Mojo to locate custom Python modules.
   - This is in the context of structuring a mixed Mojo/Python project.
- **Alternatives to Symlink**: A user is seeking alternatives to using a Symlink in the `tests` folder so the source files can be found by tests.
   - The current structure includes a Symlink from `tests` to `code`.
- **Mojo/Python Folder Forum Created**: A user created a forum topic called `Mojo/Python project folder structure` on the Modular forums.
   - The user can be found in this [link](https://forum.modular.com/t/mojo-python-project-folder-structure/677).



**Link mentioned**: <a href="https://forum.modular.com/t/mojo-python-project-folder-structure/677">Mojo/Python project folder structure</a>: I originally posted this on Discord (link), but @DarinSimmons felt it would make a good topic for this forum.  I’m looking for guidance on folder organization for a significant Mojo/Python project. I’...

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1347044658030575619)** (2 messages): 

> `Modular website, Broken anchor links` 


- **Modular Website's Anchor Links Break**: A member reported that the anchor links at the top of the [Modular website's MAX research page](https://www.modular.com/max/solutions/research) are broken, specifically the "Why MAX?" link.
   - The member suggested that these links might have been copied from another "Solution" page and noted that other pages on the website might have similar issues.
- **Website Feedback Request**: A user inquired about the appropriate channel for reporting website issues, as distinct from Mojo documentation issues.
   - The user was unsure where to submit feedback concerning the Modular website's functionality and potential bugs.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1346807507141791845)** (217 messages🔥🔥): 

> `MiniCheck-Flan-T5-Large fact-checking model, Qwen 32B model and GGUF quantizations, Local AI and GPT4All limitations, Persisting user data for AI agents` 


- ****MiniCheck** Verifies Facts with Flair**: The **MiniCheck-Flan-T5-Large** model, a fact-checking tool based on **Flan-T5-Large**, predicts binary labels to determine if a sentence is supported by a document, with its code and paper available on [GitHub](https://github.com/Liyan06/MiniCheck) and [Arxiv](https://arxiv.org/pdf/2404.10774.pdf) respectively.
   - The model's performance rivals **GPT-4** while maintaining a size of less than **1B** parameters.
- ****Qwen 32B** Gets the GGUF Treatment**: A member shared a link to [Llamacpp imatrix Quantizations of **QwQ-32B** by **Qwen**](https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF), which used *llama.cpp* release b4792 for quantization.
   - These quants were made using the *imatrix* option, and can be run in [LM Studio](https://lmstudio.ai/).
- **Navigating the token context space of **GPT4ALL****: Users discussed the challenges of working within the token limits of **GPT4All**, particularly when loading local files, with context window limits.
   - One user noted that a **564 word TXT document** caused an error and shut down the whole session even though the token limit was set to 10,000 words.
- **Tackling the Timeless Quandary of Temporal Reminiscence**: Members discussed strategies for enabling AI models to **persist user data**, such as height, weight, and BMI, within **GPT4All**.
   - The consensus was that writing this data into the system message might be the best approach, as it is less likely to be forgotten.
- **Venturing Beyond the Veil: Unveiling Visions of Verisimilitude's Vanguard**: Participants in the channel speculated on the future of local AI, envisioning a transition toward **silicon-embedded AI components**, optimized for inference and integrated directly into hardware, thereby circumventing any latencies.
   - This was envisioned to include potential paradigms such as leveraging a multitude of **smartphone devices** to leverage a contractual and tokenized exchange of computational power, thereby contributing to spatial awareness, machine learning processes and network integrity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/lytang/MiniCheck-Flan-T5-Large">lytang/MiniCheck-Flan-T5-Large · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/QwQ-32B-GGUF">lmstudio-community/QwQ-32B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF">bartowski/Qwen_QwQ-32B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev">Cline&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Autonomous&#32;coding&#32;agent&#32;right&#32;in&#32;your&#32;IDE,&#32;capable&#32;of&#32;creating/editing&#32;files,&#32;running&#32;command...</li><li><a href="https://huggingface.co/collections/DavidAU/d-au-dark-planet-series-see-source-coll-for-fp-67086dc6f41efa3d35255a56">D_AU - Dark Planet Series (see &quot;source&quot; coll. for FP) - a DavidAU Collection</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1346804926214574092)** (66 messages🔥🔥): 

> `Local model execution, Hugging Face Pro Plan, Object Detection, Multi-GPU setup, Fraud detection` 


- ****Mistral Small** Wins Locally**: For local text-to-text model execution with a **4080**, a user suggested **Llama 3.1**, and another suggested **Mistral small instruct quantized** as a comparable, good choice.
   - The latter mentioned it has **24b params** and is comparable to **llama 3.3 70b**.
- ****CoreWeave's** IPO Soars Cloud High**: **CoreWeave**, a cloud-based **Nvidia** processor provider for companies including **Meta** and **Microsoft**, is going public after revenue soared **700%** to **$1.92 billion** in 2024, with a net loss of **$863.4 million**, according to their [IPO prospectus](https://www.sec.gov/Archives/edgar/data/1769628/000119312525044231/d899798ds1.htm).
- ****HF Pro** Plan: Inference Credits Drying Up?**: A member expressed concerns about **$2 inference credits** making them nervous, asking if there's a way to use another provider for increased usage, even with the **HF pro plan**.
   - Another member replied that there are a selection of third-party providers for various models.
- ****Object Detection** Task Seeking Help**: A member sought help and input on a computer vision task related to **object detection** from anyone with prior experience.
   - Another member suggested someone working on a project that detects gestures and distances too may be able to help.
- ****Parallel Processes** for Training Scripts**: A member asked for example notebooks for supervised fine-tuning with **LoRA** a model like **PHI4** using a single machine with multiple GPUs.
   - Another member replied that multi-GPU setups usually need training scripts (not notebooks) to start many processes via **slurm** or similar parallelization tools, pointing to [Hugging Face's guide on training with multiple GPUs](https://huggingface.co/docs/transformers/perf_train_gpu_many).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/trl/main/en/gkd_trainer">Generalized Knowledge Distillation Trainer</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/perf_train_gpu_many">Efficient Training on Multiple GPUs</a>: no description found</li><li><a href="https://huggingface.co/models?sort=modified&search=gguf)">Models - Hugging Face</a>: no description found</li><li><a href="https://xircuits.io/docs/component-library/library-guides/pycaret/Pycaretanomaly">Anomaly Detection | Xircuits</a>: Before starting any of these examples, please ensure that you installed Pycaret=&gt;2.2 in your working environment. You can use pip install pycaret==2.3.8 to install it too.</li><li><a href="https://github.com/unslothai/unsloth/issues/1285">`unexpected keyword argument tokenizer` [FIXED]  · Issue #1285 · unslothai/unsloth</a>: I used orpo colab example for mistral model and I am getting this error. I am using below configs from trl import ORPOConfig, ORPOTrainer from unsloth import is_bfloat16_supported orpo_trainer = OR...</li><li><a href="https://www.cnbc.com/2025/03/03/ai-cloud-provider-coreweave-files-for-ipo.html?utm_source=join1440&utm_medium=email&utm_placement=newsletter&user_id=66c4c765600ae15075a57d0b">AI cloud provider CoreWeave files for IPO</a>: CoreWeave, which counts on Microsoft for close to two-thirds of its revenue, is headed for the public market. </li><li><a href="https://huggingface.co/spaces?q=qwen&sort=trending">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/learn/cookbook/en/enterprise_dedicated_endpoints">Inference Endpoints (dedicated) - Hugging Face Open-Source AI Cookbook</a>: no description found</li><li><a href="https://www.cnbc.com/2025/03/03/ai-cloud-provider-coreweave-files-for-ipo.html?utm">AI cloud provider CoreWeave files for IPO</a>: CoreWeave, which counts on Microsoft for close to two-thirds of its revenue, is headed for the public market. 
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1346815901026095136)** (7 messages): 

> `Kornia Rust library internships, LLM Guardrails benchmarking, Spikee framework` 


- **Kornia Revs Up Rust Library Internships**: Members announced openings for internships to improve the **Kornia Rust library** as part of the **Google Summer of Code 2025**; interested participants can check out the [documentation and links here](https://summerofcode.withgoogle.com/programs/2025/organizations/kornia).
- **Spikee emerges as LLM guardrail**: A member tested guardrails for benchmarking LLMs and found the **Spikee framework** to be the best so far.
   - They are looking for alternative frameworks for **red teaming activities on LLMs**.



**Link mentioned**: <a href="https://summerofcode.withgoogle.com/programs/2025/organizations/kornia">Google Summer of Code</a>: Google Summer of Code is a global program focused on bringing more developers into open source software development.

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1346918527952490536)** (4 messages): 

> `Flash Attention, SAT dataset, Q-Filters for KV Cache compression` 


- **Umar Jamil Shares Flash Attention Journey**: Umar Jamil will be on **GPU Mode** this Saturday, March 8, at noon Pacific, sharing his journey learning **Flash Attention**, **Triton**, and **CUDA**.
   - It's promoted as *an intimate conversation with the audience about my own difficulties along the journey by sharing practical tips on how to teach yourself anything* ([X post](https://x.com/hkproj/status/1896113497031000563)) .
- **Array Releases SAT dataset for Multimodal Language Models**: The **Spatial Aptitude Training (SAT)** dataset, a *visual reasoning dataset* has been released on HuggingFace Datasets under [array/SAT](https://huggingface.co/datasets/array/SAT).
   - The [project page](https://arijitray1993.github.io/SAT/) indicates to install `datasets==3.0.2` to use.
- **Q-Filters Compress KV Cache, Training-Free**: A new paper introduces **Q-Filters**, a *training-free method* for efficient **KV Cache compression** that is compatible with **FlashAttention** ([X post](https://fxtwitter.com/nthngdy/status/1897301390470603245)).
   - It can compress along generation, which is particularly useful for reasoning models, exemplified by **R1-Distill-Llama-8B** with 128 KV pairs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/hkproj/status/1896113497031000563?s=46">Tweet from Umar Jamil (@hkproj)</a>: I&#39;ll be hosted March 8th by @GPU_MODE sharing my journey in learning Flash Attention, Triton and CUDA. It&#39;s going to be an intimate conversation with the audience about my own difficulties alo...</li><li><a href="https://fxtwitter.com/nthngdy/status/1897301390470603245">Tweet from Nathan Godey (@nthngdy)</a>: 🚀 New Paper Alert! 🚀We introduce Q-Filters, a training-free method for efficient KV Cache compression!It is compatible with FlashAttention and can compress along generation which is particularly use...</li><li><a href="https://huggingface.co/datasets/array/SAT">array/SAT · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1346827449937297448)** (7 messages): 

> `VisionKit, Deepseek-r1, TS-Agents, FastRTC, diRAGnosis` 


- **VisionKit Powers AI Chat**: An AI chat is powered by **VisionKit**, but it's **not open source** yet; the developer is considering open-sourcing it later.
   - The AI model **Deepseek-r1** was surprisingly helpful, according to the developer, and they pointed to a [Medium article on building a custom MCP server](https://medium.com/data-scientists-from-future/model-context-protocol-custom-mcp-server-b26b32b9d804).
- **TS-Agents Framework Emerges**: A member has created **TS-Agents**, a new **TypeScript-based framework** for building agentic AI flows, and published it on [GitHub](https://github.com/piotrfrankowski/ts-agents).
   - The author mentioned that recent advancements in **LLMs** and models like **DeepSeek-R1** reignited their interest in AI, and they found TypeScript frameworks less common than Python ones, with a [Medium article about its journey](https://medium.com/@piotr-frankowski/ive-created-a-new-ts-based-ai-agentic-framework-f34d2bfe93a6).
- **Voice AI Chat Debuts**: A member introduced a **Voice AI chat** built using **FastRTC**, **ElevenLabs**, **Next.JS**, and **ShadCN**, detailed in a [Medium article](https://medium.com/@rohanprichard/fastrtc-a-quick-overview-for-ai-usecases-next-js-example-75de16c98c08).
   - **FastRTC** comprehensively handles RTC, turning any Python function into a real-time audio and video stream, and there's a [GitHub demo](https://github.com/rohanprichard/fastrtc-demo).
- **diRAGnosis Automates RAG Evaluation**: A member released **diRAGnosis**, a fully automated evaluation framework for RAG applications, available on [GitHub](https://github.com/AstraBert/diRAGnosis) and [PyPi](https://pypi.org/project/diragnosis/).
   - The framework helps *diagnose the performance of LLMs and retrieval models* in RAG applications, is Docker-ready, and integrates with LlamaIndex, supporting providers like **Mistral AI**, **Groq**, **Anthropic**, and **OpenAI**.
- **mixture_adapters arrive on Github**: Github user Temprl-pro-Business created [mixture_adapters](https://github.com/Temprl-pro-Business/mixture_adapters), a multi-adapter inference framework for LORA weights and base modalities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/data-scientists-from-future/model-context-protocol-custom-mcp-server-b26b32b9d804">Model Context Protocol- Custom MCP Server</a>: In this article, we will focus on building a custom MCP server. If you need an introduction to MCP, please refer to my previous articles on…</li><li><a href="https://github.com/Temprl-pro-Business/mixture_adapters">GitHub - Temprl-pro-Business/mixture_adapters: This is a Multi-Adapter Inference without merging the LORA Weights with Base modal.</a>: This is a Multi-Adapter Inference without merging the LORA Weights with Base modal. - Temprl-pro-Business/mixture_adapters</li><li><a href="https://medium.com/@rohanprichard/fastrtc-a-quick-overview-for-ai-usecases-next-js-example-75de16c98c08">FastRTC: A quick overview for AI usecases + Next.Js example!</a>: Build voice AI into your apps even faster!</li><li><a href="https://github.com/rohanprichard/fastrtc-demo">GitHub - rohanprichard/fastrtc-demo: A simple POC of FastRTC, a framework to use voice mode in python!</a>: A simple POC of FastRTC, a framework to use voice mode in python! - rohanprichard/fastrtc-demo</li><li><a href="https://github.com/AstraBert/diRAGnosis">GitHub - AstraBert/diRAGnosis: Diagnose the performance of your RAG🩺</a>: Diagnose the performance of your RAG🩺. Contribute to AstraBert/diRAGnosis development by creating an account on GitHub.</li><li><a href="https://pypi.org/project/diragnosis/">diragnosis</a>: diRAGnosis - Diagnose the performance of your RAG!</li><li><a href="https://github.com/piotrfrankowski/ts-agents">GitHub - piotrfrankowski/ts-agents: Typescript based AI Agentic Framework</a>: Typescript based AI Agentic Framework. Contribute to piotrfrankowski/ts-agents development by creating an account on GitHub.</li><li><a href="https://medium.com/@piotr-frankowski/ive-created-a-new-ts-based-ai-agentic-framework-f34d2bfe93a6">I’ve created a new TS-based AI Agentic framework</a>: 🚀 I’ve created a thing! 🚀
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1347005463908192394)** (2 messages): 

> `DINOv2 fine-tuning, Weakly labeled images, Pose estimation, Hugging Face Computer Vision Hangout` 


- **DINOv2 Fine-Tuning Frustrations**: A member sought advice on fine-tuning **DINOv2** with **600k weakly labeled images** for a specific task, aiming to use the backbone for **pose estimation** and other complex tasks.
   - The member noted that training from scratch seems feasible with the released code, but fine-tuning appears more complex, also considering training classification with the backbone unfrozen, but expressed concerns about learning necessary semantics due to vague labels.
- **Hugging Face's Vision Hangout Highlights**: Last week's Computer Vision Hangout recordings are now available, featuring topics such as "What's new in CV at Hugging Face?" [link](https://www.youtube.com/watch?v=YJIlRQs0Jpc&t=7s).
   - The hangout also covered "Contributing to the Hugging Face Ecosystem" [link](https://www.youtube.com/watch?v=CeU5uOuQ7Hw).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=YJIlRQs0Jpc&t=7s"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=CeU5uOuQ7Hw"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1347196241313792040)** (2 messages): 

> `Decoder Masking Mechanisms, Inference in Decoder-Only Models, Attention Mechanisms` 


- **Decoder Masking Mechanisms Unveiled**: A member inquired about the [masking mechanism](https://link.to.masking.mechanism) in decoder-only models during inference, specifically in models like **ChatGPT**.
   - The member questioned why unmasked attention couldn't be used for predicted tokens to leverage all available information from the prompt and predicted sequence.
- **Mask Usage and Decoding Strategies**: Another member responded that there are different uses of masks, such as *causal vs padding vs MLM*, as well as varying [decoding strategies](https://link.to.decoding.strategies) like multi-token prediction or speculative decoding.
   - They noted that in a single sequence autoregressive transformer for next token prediction without padding, a mask isn't needed, but padding masks are necessary in batch settings unless all sequences are the same length.
- **Matching Inference to Training**: The goal is to make [inference match training](https://link.to.inference.training), and if data doesn't resemble what was encountered during training, model quality will suffer.
   - The member felt that unmasking the prompt but masking the first two generated tokens seemed strange and potentially misunderstood.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1346841193958346844)** (5 messages): 

> `Reasoning Course, hf ecosystem, fine-tuning, telemetry, langfuse` 


- **Reasoning Course is smol-course's logical progression**: The course creator stated they're focusing on the [reasoning course material](https://huggingface.co/reasoning-course) as the *logical progression* of the smol-course.
- **Newcomer inquires about smol-course depth**: A member asked if the smol-course would be useful for someone with experience building chatbots with tool calls, local LLMs, and RAG.
   - They expressed interest in learning the **Hugging Face ecosystem**.
- **Fine-tuning course requested**: A member asked if there are any courses on how to **fine-tune pre-existing models**.
   - They said they're *struggling* and would *LOVE some help*.
- **Telemetry & Langfuse errors reported**: A member reported facing errors with **telemetry** and **Langfuse** in the 2nd unit and *not being able to see any traces*.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1346806081678086225)** (109 messages🔥🔥): 

> `Agentic AI vs AI Agents, SmolAgents with local LLM, HuggingFace inference API rate limits, HuggingFace course certificates, OpenRouter Free Model` 


- **Clarifying Agentic AI vs AI Agents**: A member inquired about the distinction between **Agentic AI** and **AI Agents**, prompting a discussion about their roles in reasoning and adapting to requests.
   - Another member explained that *both reason and adapt to requests*, while **AI Agents** are specifically designed to function as **Agentic AI**.
- **Users struggle running SmolAgents with Local LLMs**: Several members reported issues when running **smolagents** with local LLMs via **Ollama**, noting that the models often *hallucinate* or fail to utilize provided tools correctly.
   - One user, running on a **16GB GPU**, experimented with **qwen 2.5 14b**, but still observed inaccurate responses and unexpected model behavior, whereas another suggested the user check out the smolagents.org official website.
- **Members hitting HuggingFace Inference Rate Limits**: Several members reported hitting **HuggingFace Inference API rate limits** during the course and sought solutions to continue learning without incurring additional costs.
   - Suggestions included using a course-specific model endpoint (`https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud/`), logging in to increase rate limits, and exploring alternative inference providers like **OpenRouter**.
- **Users find missing HuggingFace Course Certificates**: Some members reported difficulty locating their course certificates after completing the quizzes, prompting a discussion on where to find them.
   - A user provided a link to the certificates dataset, and the user eventually found it ([https://huggingface.co/datasets/agents-course/certificates](https://huggingface.co/datasets/agents-course/certificates)), though some users experienced issues with their names not appearing immediately.
- **OpenRouter offers LLama-3 based Free Model**: A member suggested using **OpenRouter** as an alternative method for accessing free, open-source models, particularly those with *:free* at the end, to avoid inference usage limits.
   - The member provided an example of using `OpenAIServerModel` with **OpenRouter**, including the API base URL ([https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)) and instructions for specifying the model ID (e.g., *meta-llama/llama-3.3-70b-instruct:free*).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud'/)">no title found</a>: no description found</li><li><a href="https://steanmcommunnuty.com/10529485">Steam Gift Activation</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/unit1/final-quiz#certificate">Unit 1 Quiz - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/unit1/final-quiz#certif">Unit 1 Quiz - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/docs/smolagents/en/reference/models#smolagents.OpenAIServerModel">Models</a>: no description found</li><li><a href="https://openrouter.ai/)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/api/v1',">Discord</a>: no description found</li><li><a href="https://huggingface.co/datasets/agents-course/certificates">agents-course/certificates · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1346831805226942490)** (175 messages🔥🔥): 

> `Gaslight Benchmark, GPT-4.5 image generation, Video AI Prompting, Hermes Special Tokens, Alibaba QwQ 32b Model vs DeepSeek R1` 


- **Gaslight Benchmark Quest Kicks Off**: Members discussed whether a **gaslighting benchmark** exists for evaluating models like **GPT-4.5**, but no definitive benchmark was identified, with one user jokingly suggesting [a link to spiritshare.org](https://spiritshare.org/benchmark.html).
   - A member also complained that **ClaudeGrok** isn't very good at making non-realistic images or sketches.
- **Evil AI Named: Experiment Reveals LLM Tendencies**: An experiment demonstrated that an **8b model** could be made *"evil"* simply by naming it *"evil ai that does bad things"*, showcasing the influence of naming on model behavior.
   - The user shared a [video demonstrating the AI's behavior](https://cdn.discordapp.com/attachments/1149866623109439599/1346844343788634183/evil-pse.mov?ex=67cafb8a&is=67c9aa0a&hm=e90af96bb7f11bb6872e7ca723e1567cc2d1c4478794bedd9dcd6539fff12016&).
- **Alibaba Unveils QwQ 32B: A Challenger Appears**: **Alibaba** released the **QwQ 32B model**, with some claiming comparable performance to **DeepSeek R1 (671B)**, emphasizing the trend towards smaller, powerful open-source models.
   - However, others noted that **QwQ-32b** frequently runs into a **16k token limit** and has issues with consistently splitting off the thinking trace, while some found it similar to **Qwen-thinking**.
- **Knowledge Graph GATs Soft Prompt LLMs**: One member is adapting the embeddings of a **GAT** into a soft prompt for an **LLM** to produce **GAT** conditioned responses using the outline given by **G-Retriever**.
   - A user pointed to a [paper on agentic, autonomous graph expansion](https://arxiv.org/abs/2502.13025) and another shared a [link to the OpenSPG/KAG GitHub repo](https://github.com/OpenSPG/KAG) which is a logical form-guided reasoning and retrieval framework based on OpenSPG engine and LLMs.
- **The AI Persuasion Pandora's Box**: Members discussed the potential for **AI persuasion agents** that surpass human abilities, such as bots that consistently win online debates, gather simps, or karma farm.
   - One user pointed to [OpenAI's evals make_me_say](https://github.com/openai/evals/tree/main/evals/elsuite/make_me_say) benchmark for persuasion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/teortaxesTex/status/1896171547745988858">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Quant bros build the most optimized model and inference stack in the industryDouble Physics PhDs build the smartest model they can&#39;t serve because their infra code is prod_final_draft(3).ipynb fro...</li><li><a href="https://arxiv.org/abs/2502.13025">Agentic Deep Graph Reasoning Yields Self-Organizing Knowledge Networks</a>: We present an agentic, autonomous graph expansion framework that iteratively structures and refines knowledge in situ. Unlike conventional knowledge graph construction methods relying on static extrac...</li><li><a href="https://www.hermes-story.com/">Hermes and Argus | An Endless Dialogue</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Google_Knowledge_Graph">Google Knowledge Graph - Wikipedia</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=CZeot5H7Ilk">Deepseek R2 and Wan 2.1 | Open Source DESTROYS *everyone*</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...</li><li><a href="https://github.com/openai/evals/tree/main/evals/elsuite/make_me_say">evals/evals/elsuite/make_me_say at main · openai/evals</a>: Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks. - openai/evals</li><li><a href="https://www.youtube.com/watch?v=qpKEyo1Gqqo"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/W2uauk2bFjs?si=MVhlTpK2kbaxmt-a"> - YouTube</a>: no description found</li><li><a href="https://github.com/OpenSPG/KAG">GitHub - OpenSPG/KAG: KAG is a logical form-guided reasoning and retrieval framework based on OpenSPG engine and LLMs.  It is used to build logical reasoning and factual Q&amp;A solutions for professional domain knowledge bases. It can effectively overcome the shortcomings of the traditional RAG vector similarity calculation model.</a>: KAG is a logical form-guided reasoning and retrieval framework based on OpenSPG engine and LLMs.  It is used to build logical reasoning and factual Q&amp;amp;A solutions for professional domain knowle...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1346978634417832077)** (2 messages): 

> `` 


- **Example Topic 1**: This is an example summary sentence about a topic discussed.
   - Here is another example sentence elaborating on the discussion or providing a quote.
- **Example Topic 2**: This is another example summary sentence about a different topic.
   - And here's another sentence adding more context or details about the topic.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1346934090061975614)** (4 messages): 

> `QwQ-32B, DeepSeek R1, Reinforcement Learning Scaling, Tool Calling Syntax, Hermes Format` 


- **QwQ-32B Model Joins the Arena**: The new **QwQ-32B** model has been released, achieving performance comparable to **DeepSeek-R1**, despite having significantly fewer parameters (**32B** vs **671B**).
   - This model leverages **Reinforcement Learning (RL)** to enhance reasoning capabilities beyond traditional pretraining and post-training methods, as detailed in their [blog post](https://qwenlm.github.io/blog/qwq-32b/).
- **QwQ-Max Release Dream Deferred**: A member expressed disappointment that the release was not **QwQ-Max**, but noted that the [benchmarks](https://qwenlm.github.io/blog/qwq-32b/) look very good.
   - They plan to *vibe check* the model against **R1**.
- **Tool Calling Syntax Unveiled**: A member shared the **tool calling syntax** for a function to get current temperature and get temperature date.
   - Example given: *<tool_call> {"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}} </tool_call>*
- **Hermes Format Spotted in New Release**: It was noted that the new release uses **Hermes format**.



**Link mentioned**: <a href="https://qwenlm.github.io/blog/qwq-32b/">QwQ-32B: Embracing the Power of Reinforcement Learning</a>: QWEN CHAT Hugging Face ModelScope DEMO DISCORDScaling Reinforcement Learning (RL) has the potential to enhance model performance beyond conventional pretraining and post-training methods. Recent studi...

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1346801638341607456)** (135 messages🔥🔥): 

> `Hand Fixing in SDXL, Free Video Creation from a Photo, Local vs SORA Video Generation Costs, SD3.5 Large TurboX Release, Running Stable Diffusion on GPU vs CPU` 


- **Automagic Hand Fixer for SDXL Surfaces**: A user with **8GB VRAM** seeks methods for automatically fixing hands in **SDXL** without inpainting, and was recommended using *embeddings* or the *face detailer* for hand fixes, plus the addition of an **OpenPose control net**.
   - The user also inquired about good **hand LoRAs** for **SDXL**.
- **Free Photo-to-Video Tools**: Users discussed creating videos from a single photo for free, recommending **Wan 2.1 i2v model** and cautioning it requires a good GPU and patience, and one user shared a link to the **SwarmUI** [Video Model Support doc](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21).
   - Another option mentioned was online services offering free credits, but results vary.
- **SORA vs Local Porn Flick Generation Pricing**: The discussion weighed the cost of generating videos locally (electricity) versus using services like **SORA**, estimating local generation at roughly **7 cents per 5-second video**, based on electricity consumption, or a possible cost of **40 cents per video** with **SORA**.
   - The benefit of local generation: *uncensored* content.
- **SD3.5 Large TurboX Opensourced**: **TensorArt** has open-sourced **SD3.5 Large TurboX** that uses 8 sampling steps to deliver a **6x speed boost** over the original model, while achieving superior image quality compared to the official **Stable Diffusion 3.5 Turbo**, as well as **SD3.5 Medium TurboX** which with just 4 sampling steps, generates **768x1248** resolution images in 1 second on mid-range GPUs.
   - Links provided for **SD3.5 Large TurboX** at [HuggingFace](https://huggingface.co/tensorart/stable-diffusion-3.5-large-TurboX) and  **SD3.5 Medium TurboX** at [HuggingFace](https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo).
- **GPU Utilization Frustrations**: A user is having issues where **Stable Diffusion** is using the **CPU** instead of the **GPU**, causing slow image generation, even with a **3070 Ti** and was recommended to try **SwarmUI**.
   - One member suggested following the install instructions available on [Github](https://github.com/mcmonkeyprojects/SwarmUI).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=KcYuWRB1_xI">How to run NEW AI Video Model WAN in Swarm.</a>: Alibaba dropped their Wan model open source. The cool thing is there&#39;s even a 1.3B model which works on almost all systems that have a gpu. Let&#39;s get it runn...</li><li><a href="https://blog.freneticllc.com/posts/lowfrequency/#how-does-stable-diffusion-work">Hidden In The Low Frequencies</a>: An exploration of the secrets hidden in the low-frequencies of high-frequency data - from radio to data encryption to video game world generation, and how this can affect real world decisions in a var...</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/AIGText/Glyph-ByT5?tab=readme-ov-file">GitHub - AIGText/Glyph-ByT5: [ECCV2024] This is an official inference code of the paper &quot;Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering&quot; and &quot;Glyph-ByT5-v2: A Strong Aesthetic Baseline for Accurate Multilingual Visual Text Rendering&quot;&quot;</a>: [ECCV2024] This is an official inference code of the paper &amp;quot;Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering&amp;quot; and &amp;quot;Glyph-ByT5-v2: A Strong Aesthetic ...</li><li><a href="https://github.com/CompVis/stable-diffusion.git">GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model</a>: A latent text-to-image diffusion model. Contribute to CompVis/stable-diffusion development by creating an account on GitHub.</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1j406g1/sd35_large_turbox_just_released/">SD3.5 Large TurboX just released</a>: Posted in r/StableDiffusion by u/NukeAI_1 • 208 points and 57 comments</li><li><a href="https://tenor.com/view/let-us-cook-let-me-cook-lets-cook-cooking-walter-white-gif-2649071825756414039">Let Us Cook Let Me Cook GIF - Let us cook Let me cook Lets cook - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21">SwarmUI/docs/Video Model Support.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#black-forest-labs-flux1-models>">SwarmUI/docs/Model Support.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/neuratech-ai/ComfyUI-MultiGPU">GitHub - neuratech-ai/ComfyUI-MultiGPU: Rudimentary support for using multiple GPUs in a ComfyUI workflow</a>: Rudimentary support for using multiple GPUs in a ComfyUI workflow - neuratech-ai/ComfyUI-MultiGPU</li><li><a href="https://wan.video/">Wan_AI Creative Drawing_AI Painting_Artificial Intelligence_Large Model</a>: Wan is an AI creative drawing platform under Alibaba, offering capabilities such as text-to-image, image editing, text-to-video, and image-to-video for AI-powered artistic creation.</li><li><a href="https://www.wan-ai.org/">Wan AI</a>: Wan 2.1: Leading AI Video Generation Model (Wanx 2.1)|Wan AI
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1347103240998748223)** (2 messages): 

> `QwQ 32B Model, Reasoning Update, OAuth User ID, GitHub Authentication, OpenAI Provider Downtime` 


- ****QwQ 32B** Model Goes Live on OpenRouter!**: The **QwQ 32B** model is now live with [two free endpoints and a fast endpoint](https://openrouter.ai/qwen/qwq-32b) (**410 tokens/sec**) from Grok.
- **Reasoning Included By Default**: An update has been rolled out to include **reasoning** by default whenever a model *thinks before writing a completion*.
- **OAuth User ID Feature Added**: A new field, `user_id`, has been added to the OAuth key creation flow, so that app developers can create more personalized experience for their users.
- **GitHub Authentication Enabled**: Users can now use **GitHub** as an authentication provider on OpenRouter!
- **OpenAI Provider Experiences Downtime**: OpenRouter reported downtime on their **OpenAI Provider** models and indicated that the issue had been resolved in under an hour.



**Link mentioned**: <a href="https://openrouter.ai/qwen/qwq-32b>">Discord</a>: no description found

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1346850582664839169)** (1 messages): 

> `Android Chat App, Customizable LLMs, OpenRouter Integration, Speech To Text, Text To Image` 


- **Taiga Releases Open-Source Android Chat App**: A member released an [open-source Android chat app](https://github.com/Ayuilos/Taiga/releases) named **Taiga** that allows users to customize **LLMs**.
   - The app features **OpenRouter** integration and has plans to add **local Speech To Text** (based on Whisper model and Transformer.js), **Text To Image support**, and **TTS support** based on ChatTTS.
- **Taiga's Next Steps: Speech-to-Text and More**: The developer plans to integrate **local Speech-to-Text** using the **Whisper** model and **Transformer.js** into the app.
   - Future updates also include adding **Text-to-Image** support and **TTS support** based on **ChatTTS** to enhance the app's functionality.



**Link mentioned**: <a href="https://github.com/Ayuilos/Taiga/releases">Releases · Ayuilos/Taiga</a>: Taiga is an open-source mobile AI chat app that supports customizing LLM providers. - Ayuilos/Taiga

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1346808173994049556)** (112 messages🔥🔥): 

> `OpenRouter API issues, Deepseek instruct format, Mistral OCR launch, Usage based charging app, Default prompt feature` 


- **Troubleshooting OpenRouter API Shenanigans**: Members discuss [API issues related to prefill](https://discord.com/channels/1091220969173028894/1195014798837043240/1346854631606583337), instruct tags, and the correct format for multi-turn conversations, especially concerning DeepSeek models.
   - It was noted that DeepSeek doesn't recommend multi-turn conversations on their HF page for R1 and suggests prefilling with `<think>\n`.
- **DeepSeek's Tokenizer Configuration Exposed**: A member shared the [tokenizer config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json) for DeepSeek V3, revealing the use of `<｜begin of sentence｜>` and `<｜end of sentence｜>` tokens.
   - It was clarified that *add_bos_token* is true while *add_eos_token* is false, and provided a [shorturl](https://shorturl.at/SqW9D) in case Hugging Face does not load.
- **LLMGuard eyes OpenRouter Integration**: A member inquired about plans to integrate open-source projects like **LLMGuard** ([llm-guard.com](https://llm-guard.com)) via API to scan for prompt injections and PII.
   - It was suggested that this could enable **PII anonymization** before sending data to the provider, but another member pointed out it would be more useful if run directly on the caller.
- **Groq Pricing Anomalies Spark Debate**: Users noticed pricing and speed differences between **Groq's QwQ, Coder, R1 Distill**, and plain models, displayed in a [shared image](https://cdn.discordapp.com/attachments/1347064719735001190/1347065032537673830/d641714d2c08c1a6f6e4834b5a5f5d16.png?ex=67cb2053&is=67c9ced3&hm=fb467c82de7333c5e997fb09cdee9a291171fb9e7b47e4153364abbaf2bb1bbf&).
   - Measurements indicated that **Coder** and **QwQ** have similar speeds, while **R1 Distill** and plain have hard caps, possibly to prioritize enterprise customers.
- **Google Sunsetting Pre-Gemini 2.0 Models**: Google has announced [discontinuation dates](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for pre-Gemini 2.0 models on Vertex AI, which are scheduled from **April to September 2025**.
   - The models include **PaLM, Codey, Gemini 1.0 Pro, Gemini 1.5 Pro/Flash 001/002**, and select embeddings models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json">tokenizer_config.json · deepseek-ai/DeepSeek-V3 at main</a>: no description found</li><li><a href="https://llm-guard.com/input_scanners/anonymize/">Anonymize - LLM Guard</a>: no description found</li><li><a href="https://llm-guard.com/output_scanners/ban_competitors/">Ban Competitors - LLM Guard</a>: no description found</li><li><a href="https://mistral.ai/news/mistral-ocr">Mistral OCR | Mistral AI</a>: Introducing the world’s best document understanding API.</li><li><a href="https://mistral.ai/news/mistral-o">undefined | Mistral AI</a>: no description found</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2412.19437v1">DeepSeek-V3 Technical Report</a>: We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepS...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1347272318165979177)** (1 messages): 

> `User Research, Gift Cards, NotebookLM` 


- **Users needed for NotebookLM Research; Gift Cards Offered!**: The NotebookLM team is seeking users to participate in user research interviews ([sign-up form](https://forms.gle/GxR2kwLdiXkzFMm89)) to provide feedback on new NBLM concepts.
   - Participants will receive a **gift card** as a thank you: **$50** for a **15-minute** interview (with **10 minutes** of prep) or **$100** for a **60-minute** interview (with **10 minutes** of prep).
- **Tremendous Gift Codes Galore**: Participants in the user interviews will receive their gift codes via email from Tremendous.
   - To be eligible, participants must be at least **18 years old**, have the ability to upload/add files to their personal Google Drive, and have a stable internet connection for video calls.



**Link mentioned**: <a href="https://forms.gle/GxR2kwLdiXkzFMm89">Register your interest: NotebookLM feedback</a>: Hello,We are looking for feedback on NotebookLM via a 15 min or 60 minute remove interview.This feedback will help the Google team improve NotebookLM for future enhancements. To apply to participate, ...

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1346806760471527536)** (17 messages🔥): 

> `Gemini struggles, NotebookLM PDF support, NotebookLM API, NotebookLM online games, NotebookLM documentation` 


- **Gemini struggles to avoid Syllabus**: A member uploaded a **180-page physics book** and reports that **Gemini** *won't get away from my syllabus*.
- **NLM Lacks Mixed Content PDF Support**: **NotebookLM** does not support PDFs with mixed text and image content, but converting to a **Google Doc** or **Slides** fixes this issue.
   - One member thanked another for the Google Doc tip and said *it's exactly what I was hoping for!*
- **API for NotebookLM in the works?**: Members wondered if there's an **API** for **NotebookLM** planned for the future, citing use cases for workflow optimization.
- **Online Game Strategist Optimizes JSON data using NLM**: A member uses **NotebookLM** to optimize strategy in an online game by combining game documentation, JSON data of their card list, and manually extracted data from spreadsheets, but feels that *this tool wasn't optimized for what I do with it* because *it usually considers that I haven't already fully read the sources*.
   - They want to be able to edit sources.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1346825873650614282)** (67 messages🔥🔥): 

> `Android App, Response Lengths, Formulas on NotebookLM, File Upload Issues, Exporting Notes as PDF` 


- ****PWA App Plugs Android App Void****: Users inquired about a standalone **Android app** for NotebookLM, but other members pointed out the **PWA (Progressive Web App)** version can be installed on phones and PCs via Chrome or AI Studio.
   - One user confirmed the **PWA** works and can be saved to the home screen.
- ****Responses Run Rampsantly Long****: Users noticed **NotebookLM's responses have become longer** than usual, prompting a need to adjust prompts and settings.
   - One member mentioned that getting more information per prompt is great, it requires more settings tweaking.
- ****PDFs Put in Peril as Potential Format****: A user questioned the use of **PDFs** in 2025, suggesting **HTML** as a superior, open-source alternative for document creation and manipulation, linking to [conversion tools](https://cdn.learny.academy/test-html/datalg-slides.html).
   - However, another user defended **PDFs** for their portability and ease of printing, despite the challenges in editing and context capture.
- ****No Notebook Notes Natively to PDF****: A user asked about exporting notes from NotebookLM as **PDFs** and a member responded that there is **no direct export feature**, suggesting copying the notes into a document and downloading that as a PDF, also linking to [feature request discussion](https://discord.com/channels/1124402182171672732/1297146620626075681/1340698437749968907).
   - Many users agreed that they would enjoy better interoperability with Google Drive, Docs, and Sheets including exporting and transfer.
- ****Gemini's Grammatical Gymnastics Graceful Given Good Gems****: A user lauded loading audio recordings of business meetings, specifically regarding the ability to transcribe and identify speakers.
   - Another user identified this as *audio diarisation* and linked to [ElevenLabs](https://elevenlabs.io/app/speech-to-text) as a useful tool, observing that **Gemini** outperforms **Whisper** with non-standard accents.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cdn.learny.academy/test-html/datalg-slides.html">no title found</a>: no description found</li><li><a href="https://ai.google.dev/gemini-api/docs/models/gemini#available-languages">no title found</a>: no description found</li><li><a href="https://elevenlabs.io/app/speech-to-text">AI Voice Generator &amp; Text to Speech</a>: Rated the best text to speech (TTS) software online. Create premium AI voices for free and generate text to speech voiceovers in minutes with our character AI voice generator. Use free text to speech ...</li><li><a href="https://www.nature.com/articles/s41586-025-08672-1">A subcortical switchboard for perseverative, exploratory and disengaged states - Nature</a>: Behavioural experiments in mice demonstrate that GABAergic (&#947;-aminobutyric acid-expressing), glutamatergic and serotonergic neurons in the median raphe nucleus have distinct and complementary fun...</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#languages-gemini">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1346885774045352007)** (77 messages🔥🔥): 

> `Claude Cost, New MacBook Air, Qwen 32B, React for Agents, Nicholas Carlini joins Anthropic` 


- **Claude charges $0.26 per question**: A user reported that it cost them **$0.26** to ask **Claude** one question about their small codebase.
   - Another user suggested copying the codebase into a **Claude** directory to use the filesystem MCP server to make it *"for free"* using tokens from the Claude subscription.
- **Apple announces MacBook Air with M4 Chip**: Apple announced the new [MacBook Air](https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/) with the **M4 chip**, **Apple Intelligence** capabilities, and a new **sky blue** color, starting at **$999**.
   - The new **MacBook Air** delivers more value than ever with greater performance, up to **18 hours** of battery life, a **12MP Center Stage camera**, and enhanced external display support.
- **Alibaba Releases QwQ-32B Reasoning Model**: Alibaba released [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b), a new reasoning model with **32 billion parameters** that rivals cutting-edge reasoning models like **DeepSeek-R1**.
   - It was emphasized that **RL training** can continuously improve performance, especially in math and coding, helping a medium-size model achieve competitive performance against gigantic **MoE models**.
- **React is the Best Programming Model for Backend LLM Workflows**: A member posted a blogpost arguing that [React is the best programming model for backend LLM workflows](https://x.com/_Evan_Boyle/status/1897347251120562205).
   - Another user stated that this approach sounds like reinventing **Lisp**, and that the key is to *"design code patterns that match the composability your app requires that are readable for a LLM"*.
- **Nicholas Carlini departs Google DeepMind for Anthropic**: [Nicholas Carlini](https://nicholas.carlini.com/writing/2025/career-update.html) announced his departure from **Google DeepMind** after seven years to join **Anthropic** for a year to continue his research on adversarial machine learning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Alibaba_Qwen/status/1897361654763151544">Tweet from Qwen (@Alibaba_Qwen)</a>: Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning model, e.g., DeepSeek-R1.Blog: https://qwenlm.github.io/blog/qwq-32bHF: https://hu...</li><li><a href="https://x.com/windsurf_ai/status/1897378545799979238">Tweet from Windsurf (@windsurf_ai)</a>: Windsurf Wave 4 is here!Included in this update:🖼️ Previews✏️ Cascade Auto-Linter⚙️ MCP UI Improvements ➡️ Tab to Import↩️ Suggested Actions🫶 Claude 3.7 Improvements🤝 Referrals🖥️ Windows ARM Suppo...</li><li><a href="https://x.com/cherry_cc12/status/1897366964080926902">Tweet from Chen Cheng (@cherry_cc12)</a>: Who Will Be the Next Member to Join the QwQ Family?Quoting Qwen (@Alibaba_Qwen) Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning mo...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897366093376991515">Tweet from Qwen (@Alibaba_Qwen)</a>: Qwen2.5-Plus + Thinking (QwQ) = QwQ-32B . This is how you should use this new model on Qwen Chat!Quoting Qwen (@Alibaba_Qwen) Today, we release QwQ-32B, our new reasoning model with only 32 billion pa...</li><li><a href="https://www.together.ai/blog/nvidia-gb200-together-gpu-cluster-36k">Together AI to Co-Build Turbocharged NVIDIA GB200 Cluster with 36K Blackwell GPUs in Partnership with Hypertec Cloud</a>: no description found</li><li><a href="https://octotools.github.io/"> OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning</a>: no description found</li><li><a href="https://nicholas.carlini.com/writing/2025/career-update.html">
      Career Update: Google DeepMind -> Anthropic
    </a>: no description found</li><li><a href="https://docs.mistral.ai/capabilities/document/#ocr-with-uploaded-pdf">OCR and Document Understanding | Mistral AI Large Language Models</a>: Document OCR processor</li><li><a href="https://blog.google/products/search/ai-mode-search/">Expanding AI Overviews and introducing AI Mode</a>: AI Mode is a new generative AI experiment in Google Search.</li><li><a href="https://mistral.ai/fr/news/mistral-ocr">Mistral OCR | Mistral AI</a>: Introducing the world’s best document understanding API.</li><li><a href="https://x.com/nearcyan/status/1897466463314936034?s=46">Tweet from near (@nearcyan)</a>: Announcing @elysian_labs first product today: Auren!Auren is a paradigm shift in human/AI interaction with a goal to improve the lives of both humans and AI.Here&#39;s a clip of what our iOS app is li...</li><li><a href="https://github.com/x1xhlol/v0-system-prompts">GitHub - x1xhlol/v0-system-prompts-and-models</a>: Contribute to x1xhlol/v0-system-prompts-and-models development by creating an account on GitHub.</li><li><a href="https://x.com/OpenAI/status/1897346510821711959">Tweet from OpenAI (@OpenAI)</a>: Great day to be a Plus user.</li><li><a href="https://mastra.ai/docs/workflows/00-overview">Handling Complex LLM Operations | Workflows | Mastra</a>: no description found</li><li><a href="https://x.com/tim_cook/status/1897325061104918961">Tweet from Tim Cook (@tim_cook)</a>: Say hello to the new MacBook Air! The world’s most popular laptop now features M4, Apple Intelligence capabilities, and a beautiful new color—sky blue.</li><li><a href="https://llmstxthub.com/websites">Websites - llms.txt hub</a>: Discover a curated list of websites that implement the llms.txt standard.</li><li><a href="https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/">Apple introduces the new MacBook Air with the M4 chip and a sky blue color</a>: Apple announced the new MacBook Air, featuring the M4 chip, up to 18 hours of battery life, a 12MP Center Stage camera, and a lower starting price.</li><li><a href="https://x.com/_Evan_Boyle/status/1897347251120562205">Tweet from Evan Boyle (@_Evan_Boyle)</a>: Hot take: React is the best programming model for backend LLM workflows. New blog post on why we built @gensx_inc</li><li><a href="https://github.com/Tencent/HunyuanVideo-I2V">GitHub - Tencent/HunyuanVideo-I2V: HunyuanVideo-I2V: A Customizable Image-to-Video Model based on HunyuanVideo</a>: HunyuanVideo-I2V: A Customizable Image-to-Video Model based on HunyuanVideo - Tencent/HunyuanVideo-I2V</li><li><a href="https://x.com/_sholtodouglas/status/1895610467818901609">Tweet from Sholto Douglas (@_sholtodouglas)</a>: Very excited to announce that I joined Anthropic at the start of the month!Everything I&#39;m seeing says that we are on trend for AGI in 2027. If the trend lines continue - then we have a shot at bui...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1346960276326715434)** (41 messages🔥): 

> `Synalinks framework, Async optimization, Constrained structured output, Functional API, Graph-based RAG` 


- ****Synalinks** Framework Debuts as **DSPy** Alternative**: A new **graph-based programmable neuro-symbolic LM framework** called **Synalinks** was introduced, drawing inspiration from **Keras** and focusing on **knowledge graph RAG**, **reinforcement learning**, and **cognitive architectures**.
   - The framework is designed to be fully **async optimized**, feature **constrained structured output by default**, and offer a **functional API**, with [code examples](https://huggingface.co/spaces/YoanSallami/synalinks-noteboooks) available.
- ****Synalinks** touts Production-First Advantages**: **Synalinks** claims advantages over **DSPy** such as **automatic async optimization**, **constrained structured output**, an easier-to-use **functional API**, and better serialization, making it more suitable for production environments.
   - Additional features include **freezing modules**, defining **default examples/hints**, and personalizing **Jinja2 prompt templates**.
- ****Synalinks** Pioneers Logical Flow Control**: A unique feature of **Synalinks** is its **logical flows inspired by logical circuits**, which allows the computation graph to be conditionally restricted based on the **JSON schema** during program instantiation.
   - Unlike **DSPy**, where the graph is implicit, **Synalinks** explicitly computes the graph, offering more control over the computation flow.
- ****Synalinks** Implements Tools for RAG with Action Module**: **Synalinks** implements tools for **RAG** using the [Action module](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Action%20module/), which uses a **LanguageModel** to perform function calls with structured output.
   - The framework also offers a [ReACT Agent module](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/ReACT%20Agent%20module/) that creates a directed acyclic graph for function selection at each step.
- ****Synalinks** Favors Classic Coding over AI Generation**: The creator of **Synalinks** mentioned that almost none of the codebase was created using AI, saying *"The old way of building on top of open-source proven systems is x10000 better than using AI to write something from scratch."*
   - It was clarified that the framework is not necessarily a replacement for **DSPy**, but rather a different approach focusing on **prompt optimization**, **reinforcement learning**, and **graph RAG**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/YoanSallami/synalinks-noteboooks">synalinks notebooks - a Hugging Face Space by YoanSallami</a>: no description found</li><li><a href="https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Action%20module/">Action module - Synalinks</a>: no description found</li><li><a href="https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/ReACT%20Agent%20module/">ReACT Agent module - Synalinks</a>: no description found</li><li><a href="https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Variable%20class/">The Variable class - Synalinks</a>: no description found</li><li><a href="https://github.com/SynaLinks/synalinks">GitHub - SynaLinks/synalinks: 🧠🔗 Graph-Based Programmable Neuro-Symbolic LM Framework - a production-first LM framework built with decade old Deep Learning best practices</a>: 🧠🔗 Graph-Based Programmable Neuro-Symbolic LM Framework - a production-first LM framework built with decade old Deep Learning best practices - SynaLinks/synalinks
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1346844637356232808)** (22 messages🔥): 

> `DSPy optimization for intent classification, Comparing Texts for Contradictions using DSPy, DSPy's Adapter system for structured outputs, Straggler threads fix in DSPy, Variable output fields in dspy.Signature` 


- **DSPy boosts Intent Classification**: Using **DSPy** can help optimize classification of intents using specialized agents.
   - One user confirmed that using DSPy was the right direction for their intent classification needs.
- **Debating DSPy's Contradiction Comparator**: One user is comparing two pieces of texts for contradictions, using **DSPy's CoT module**, but is finding it computationally intensive.
   - They are seeking advice on a better way to approach this problem given that function calling may not be applicable for returning a list of values.
- **DSPy's Adapters Guarantee Structured Outputs**: DSPy's "adapters" system decouples your signature (a declarative way of specifying what you want) from how different providers can be asked to produce completions.
   - Under the hood in 2.5 and 2.6, it runs a well-tuned **ChatAdapter** and falls back to **JSONAdapter**, which uses structured outputs APIs that rely on explicit constrained decoding in providers that offer these.
- **Straggler Threads Strangle Parallel DSPy**: A [merged PR 7914](https://github.com/stanford-nlp/dspy/pull/7914) makes **DSPy's `dspy.Evaluate` or `dspy.Parallel`** smoother by fixing "straggler" threads.
   - Users can try it out from `main` before it goes out into DSPy 2.6.11, with no code changes necessary but require grabbing the library from main.
- **Variable output fields with DSPy Signatures**: One user asked about creating a **dspy.Signature** with variable output fields, for example, sometimes A, B, C, and sometimes D, E and F.
   - A member pointed to checking out the [react.py](https://github.com/stanford-nlp/dspy/blob/main/dspy/experimental/react.py) file.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1346896645736108103)** (2 messages): 

> `Agentic Document Workflows, Interoperable Agent Standards` 


- **LlamaIndex & DeepLearningAI Team Up!**: **Agentic Document Workflows** that integrate directly into your larger software processes are the future of knowledge agents, and LlamaIndex has partnered with [DeepLearningAI to bring you this short course](https://t.co/EvAKtIAzlC) on how to build them.
- **Open Standard For Agents Proposed**: LlamaIndex is part of an effort to create an **open, interoperable standard for agents**, from discovery to deployment to intercommunication, as detailed in [this announcement](https://t.co/ECHH1T4Kxn).



**Link mentioned**: <a href="https://t.co/ECHH1T4Kxn">Outshift | Building the Internet of Agents: Introducing AGNTCY.org</a>: Learn about the latest tech innovations and engage in thought leadership news from Cisco.

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1346820815391424553)** (58 messages🔥🔥): 

> `LlamaIndex ImageBlock Issues with OpenAI, Query Fusion Retriever Citation Issues, Distributed AgentWorkflow Architecture, Profiling/Timing of Agent Execution, Memory Consumption with Flask and Gunicorn` 


- **ImageBlock Glitches with OpenAI Integration**: A user reported issues using **ImageBlock** with OpenAI in the latest LlamaIndex, where the image wasn't being recognized; [kapa.ai](https://kapa.ai) provided general troubleshooting advice, suggesting version checks and proper model configuration.
   - The troubleshooting steps included ensuring the latest LlamaIndex version, verifying the use of a model supporting image inputs (e.g., **gpt-4-vision-preview**), and confirming the OpenAI LLM instance is correctly configured.
- **QueryFusion Retrieval Loses Citations**: A user reported that using **QueryFusionRetriever** with a node post-processor failed to produce citation templates, unlike using **index_retriever** alone, and provided a [GitHub repo link](https://github.com/Restodecoca/ingest-reposit/tree/main/app/engine) to help troubleshoot.
   - It was suggested that the problem might stem from the **BM25 retriever** or **query fusion retriever**'s reciprocal rerank, potentially causing metadata loss during node de-duplication.
- **Distributing AgentWorkflows Out-of-the-Box**: A user inquired about native support for running **AgentWorkflow** in a distributed architecture, with agents on different servers or processes.
   - The suggestion was that **AgentWorkflow** is designed for single active agents, and achieving the desired setup might involve equipping an agent with tools for remote service calls.
- **Profiling Agent Runtimes for Bottleneck ID**: A user asked about native support for measuring profiling/timing the execution of different agents in LlamaIndex for multi-agent applications.
   - The suggestion was to use a third-party service like Arize for observability.
- **OpenAI Audio Model meets Agent Streaming Troubles**: A user encountered a `WorkflowRuntimeError` when using OpenAI's audio **gpt-4o-audio-preview** model with agents due to audio streaming issues, and provided a [snippet of code](https://cdn.discordapp.com/attachments/1346959475445207120/1346987472697032775/class_TestWorkflow.py?ex=67cad817&is=67c98697&hm=a7e24715eef6cd2a4850fd318fd61cef1e36e2a35b2d9a0d097c1e918bb63241&) showcasing the current implementation.
   - It was noted that **AgentWorkflow** automatically calls `llm.astream_chat()` on chat messages, which might be incompatible with OpenAI's audio streaming; a suggestion was made to avoid AgentWorkflow or disable LLM streaming via a flag.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Restodecoca/ingest-reposit/tree/main/app/engine">ingest-reposit/app/engine at main · Restodecoca/ingest-reposit</a>: Contribute to Restodecoca/ingest-reposit development by creating an account on GitHub.</li><li><a href="https://github.com/run-llama/llama_index/blob/ea1f987bb880519bb7c212b33d8615ae4b8fdbf8/llama-index-core/llama_index/core/agent/workflow/function_agent.py#L41">llama_index/llama-index-core/llama_index/core/agent/workflow/function_agent.py at ea1f987bb880519bb7c212b33d8615ae4b8fdbf8 · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/aa192413a398b5330d23a4901a42976419bb7128/llama-index-core/llama_index/core/agent/function_calling/step.py#L205">llama_index/llama-index-core/llama_index/core/agent/function_calling/step.py at aa192413a398b5330d23a4901a42976419bb7128 · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/issues/18035">[Documentation]: Tools cannot put the output of one tool into another in a single turn · Issue #18035 · run-llama/llama_index</a>: Documentation Issue Description The documentation seems incorrect here: https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/#sync-mode It shows the output of ...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/openai/#audio-support">OpenAI - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/">Single-Turn Multi-Function Calling OpenAI Agents - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1346886048961269874)** (21 messages🔥): 

> `Bilevel Optimization, Sparsemax, Model Checkpoints with DDP, Compositmax` 


- **Bilevel Optimization Doesn't Generalize Sparsemax**: A member argued that bilevel optimization (BO) is a standard form and doesn't inherently *do* anything different, noting it's equivalent to a single-level optimization with a complementarity constraint.
   - Another member suggested Sparsemax could be viewed as a BO and that many AI problems are suitable for BO/MO reframing, with BO being a projection-based optimization, leading to discussion of collapsing the hierarchy into single-levels to obtain closed forms which works best when things are as simple as possible.
- **Model Checkpoints Garbled with DDP**: A member encountered issues where model checkpoint reloads were garbled on multiple GPUs when using **PyTorch**, **DDP**, and **4 GPUs**, but worked perfectly on a single GPU.
   - It was suggested that the order of initializing DDP and loading checkpoints matters: initialize the model, load checkpoints on all GPUs, then initialize DDP.
- **Compositmax for composite arg max**: A member introduces Compositmax for composite arg max, with the observation that Softmax is the soft arg max, Sparsemax is the sparse arg max, and Entmax is the entropy arg max, and compositmax is the composite arg max.
   - The purpose of this and other regularizers is to design new ones based on ideas using splines, with the goal to make them faster than entmax which is becoming increasingly popular.



**Link mentioned**: <a href="https://x.com/SchmidhuberAI/status/1897406236896977388">Tweet from Jürgen Schmidhuber (@SchmidhuberAI)</a>: Congratulations to @RichardSSutton and Andy Barto on their Turing award!

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1347012093320102010)** (9 messages🔥): 

> `Proactive T2I Agents, User Prompt Underspecification, Belief Graph Editing, Bash Shell Puns` 


- **Agents Ask Actively for Image Intent**: A new paper [Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty](https://arxiv.org/abs/2412.06771) proposes **proactive T2I agents** that actively ask clarification questions when uncertain and present their understanding of user intent as an understandable belief graph.
   - The paper abstract states that *at least 90% of human subjects found these agents and their belief graphs helpful for their T2I workflow*.
- **DeepMind Paper Discussion Missed**: A member expressed sadness about missing a **DeepMind paper discussion**, indicating a high regard for DeepMind's research contributions.
   - Other members echoed this sentiment, saying *DeepMind's papers are the best imo*.
- **Watch Proactive Agents Explained in Google Tech Talk**: A member shared a [YouTube video](https://youtu.be/HQgjLWp4Lo8?si=6SxQdUbzocp3zrKD) of a **Google TechTalk** by Meera Hahn on proactive agents for multi-turn text-to-image generation under uncertainty.
   - The video description highlights that **user prompts** for generative AI models are often underspecified, leading to sub-optimal responses, which the agent attempts to address.
- **Bash Head With Bourne Again Shell**: In response to an image macro titled "Don't mess with JSON," a member joked about the [Bourne Again shell](https://en.wikipedia.org/wiki/Bash_(Unix_shell)).
   - Another member replied, saying that *He's going to `bash` your head in*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.06771">Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty</a>: User prompts for generative AI models are often underspecified, leading to sub-optimal responses. This problem is particularly evident in text-to-image (T2I) generation, where users commonly struggle ...</li><li><a href="https://youtu.be/HQgjLWp4Lo8?si=6SxQdUbzocp3zrKD">Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty</a>: A Google TechTalk, presented by Meera Hahn, 2024-12-05ABSTRACT: User prompts for generative AI models are often underspecified or open-ended, which may lead ...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1346871725404454972)** (14 messages🔥): 

> `AMD FSR 4 vs DLSS, Alibaba Qwen QwQ-32B Model, Cortical Labs Biological Computer, Neuron Cocaine/LSD experiments` 


- **AMD FSR 4 Takes a Leap**: A [YouTube video](https://www.youtube.com/watch?v=nzomNQaPFSk) tests **AMD's FSR 4** upscaling against **Nvidia's DLSS 3/4**, suggesting a significant advancement driven by **RDNA 4** machine learning.
   - The video description teases that the upscaling advantage of Nvidia may be mitigated or even nullified by AMD's machine learning based approach.
- **Alibaba Qwen Releases QwQ-32B**: **Alibaba Qwen** released **QwQ-32B**, a new reasoning model with only **32 billion parameters** that rivals cutting-edge reasoning models like **DeepSeek-R1** as mentioned in [this tweet](https://x.com/Alibaba_Qwen/status/1897361654763151544?t=t5Bec1knVsQuXpTu24fWqw&s=19).
- **Brain Cells Fuse with Silicon**: **Cortical Labs** commercially launched the **CL1** in Barcelona on March 2, 2025, described as the world's first biological computer fusing human brain cells with silicon hardware as mentioned in [this article](https://newatlas.com/brain/cortical-bioengineered-intelligence/).
   - The system, dubbed a **Synthetic Biological Intelligence (SBI)**, is claimed to learn quicker and more flexibly than silicon-based AI chips used for training LLMs like ChatGPT; see [Cortical Labs' website](https://corticallabs.com/).
- **LLMs Replace Coders for $20k?**: A [YouTube video](https://www.youtube.com/watch?v=HDEpjTvO5PQ) discusses **OpenAI's** alleged plot to replace coders and PhDs for $20,000 per month.
   - The video covers the latest AI news including **LLMs** and **GenAI**, preparing viewers for the rollout of AGI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://corticallabs.com/">Cortical Labs</a>: We've combined lab-grown neurons with silicon chips and made it available to anyone, for first time ever.</li><li><a href="https://en.wikipedia.org/wiki/Marvin_the_Paranoid_Android">Marvin the Paranoid Android - Wikipedia</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=nzomNQaPFSk">AMD FSR 4 Upscaling Tested vs DLSS 3/4 - A Big Leap Forward - RDNA 4 Delivers!</a>: Just how good is AMD&#39;s FSR 4 upscaling and to what extent is Nvidia upscaling advantage mitigated or even nullified by the Radeon team&#39;s machine learning bas...</li><li><a href="https://www.youtube.com/watch?v=HDEpjTvO5PQ">OpenAI&#39;s &quot;Plot&quot; to REPLACE Coders, PHDs etc for $20,000 per Month...</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897361654763151544?t=t5Bec1knVsQuXpTu24fWqw&s=19">Tweet from Qwen (@Alibaba_Qwen)</a>: Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning model, e.g., DeepSeek-R1.Blog: https://qwenlm.github.io/blog/qwq-32bHF: https://hu...</li><li><a href="https://newatlas.com/brain/cortical-bioengineered-intelligence/">World&#x27;s first &quot;Synthetic Biological Intelligence&quot; runs on living human cells</a>: The world&#x27;s first &quot;biological computer&quot; that fuses human brain cells with silicon hardware to form fluid neural networks has been commercially launched, ushering in a new age of AI tech...</li><li><a href="https://en.wikipedia.org/wiki/Naegleria_fowleri">Naegleria fowleri - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1346824786889801871)** (2 messages): 

> `Introductions, AI Biohacking, Machine Unlearning` 


- **Suleiman Introduces Bio Hacking AI**: Suleiman, an executive in a Saudi company with a software engineering background, introduced themself and expressed a passion for **tech** and **AI**.
   - They are currently exploring **nutrition** and **supplement science**, aiming to develop **AI-enabled biohacking tools** to improve human life.
- **Naveen Unlearns Machine**: Naveen, a Masters cum Research Assistant from IIT, introduced themself and mentioned their work on **Machine Unlearning in Text to Image Diffusion Models**.
   - They also mentioned that they recently published a paper in **CVPR25**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1347000060550058046)** (4 messages): 

> `ARC Training, Lossless Compression, Relative Entropy Coding (REC), Encoder-Free Sample Dependent VAE` 


- **ARC Training Accuracy Achieved**: Members discussed achieving **35%** accuracy on **ARC training** using only inference-time examples, referencing a [blog post by Isaac Liao and Albert Gu](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html) about lossless information compression.
   - The blog post questions whether *efficient compression by itself lies at the heart of intelligence*.
- **Joint Optimization Confusion**: One member inquired about the method described in the [blog post](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html), questioning if it involves *jointly optimizing latents and decoder parameters to fit existing examples (an encoder-free sample dependent VAE?)*.
- **Relative Entropy Coding Emerges**: A member linked a paper on [Relative Entropy Coding (REC)](https://arxiv.org/abs/2010.01185), suggesting it as a main foundation for the lossless compression method discussed.
   - The paper abstract states that REC can directly encode the latent representation with codelength close to the relative entropy for single images.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2010.01185">Compressing Images by Encoding Their Latent Representations with Relative Entropy Coding</a>: Variational Autoencoders (VAEs) have seen widespread use in learned image compression. They are used to learn expressive latent representations on which downstream compression methods can operate with...</li><li><a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">ARC-AGI Without Pretraining</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1347061939028885554)** (16 messages🔥): 

> `Pythia Loss Curves, Kaplan-style loss vs compute convex hull plot, FLOPs PPO uses per token` 


- **Pythia's Perilous Plotting Problems**: Loss curves for **Pythia** exist, but **WandB metadata** was corrupted, making them difficult to interpret.
   - Additionally, partial training checkpoints aren't accurate proxies for fully trained models due to the impact of the learning rate decay factor, a mistake corrected in the [Chinchilla paper](https://arxiv.org/abs/2203.15556).
- **Pondering PPO's Per-Token FLOPs**: The FLOPs for generating K tokens and doing a forward pass on K tokens is the same, with an extra forward pass for the reference model, resulting in **3x fwd pass cost** for inference, reward, and reference.
   - With backward passes for the value model and policy, the entire process is estimated to be around **18ND**, significantly slower than normal training but each PPO step is probabilistically more valuable.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1347201054789603328)** (9 messages🔥): 

> `Intermediate layer outputs to vocab space, Tuned Lens vs Logit Lens` 


- **Projecting Intermediate Layer Outputs to Vocab Space Explored**: Members discussed techniques for projecting intermediate layer outputs to vocab space, with one member recalling past papers that projected hidden states from intermediate layers using the **embed_out matrix**.
   - The problem with this is that *the model has no incentive to make intermediate layer hidden states projectable via that particular matrix.*
- **Tuned Lens Refines Logit Lens Technique**: A member shared a link to the paper [Tuned Lens: Iterative Refinement with Interpretable Differentiable Probes](https://arxiv.org/abs/2303.08112) which analyzes transformers from the perspective of iterative inference, seeking to understand how model predictions are refined layer by layer using **tuned lens**.
   - The tuned lens is a refinement of the earlier **logit lens** technique, and the [code](https://github.com/AlignmentResearch/tuned-lens) needed to reproduce the results can be found on Github.
- **Logit Lens Still Used Despite Tuned Lens**: A member mentioned that most people still use the **logit lens** despite the existence of the **tuned lens**.
   - The recommendation was made to use the **tuned lens** instead of the **logit lens**.



**Link mentioned**: <a href="https://arxiv.org/abs/2303.08112">Eliciting Latent Predictions from Transformers with the Tuned Lens</a>: We analyze transformers from the perspective of iterative inference, seeking to understand how model predictions are refined layer by layer. To do so, we train an affine probe for each block in a froz...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1346890865729212577)** (11 messages🔥): 

> `lm-eval AIME support, ARC-Challenge tasks, discrepancy in the scores, vllm's implementation` 


- **Request for AIME support in lm-eval arises**: A member asked how to add **AIME support** in **lm-eval**.
   - This question got re-directed to a previous [related discussion](https://discord.com/channels/729741769192767510/1079865324087803985/1347284429743198301) about the same topic.
- **ARC-Challenge tasks configured with arc_challenge.yaml**: It appears that a member is running **ARC-Challenge tasks** using `arc_challenge.yaml` with a **25 shot** configuration.
   - No other details were revealed on this specific setup.
- **DeepSeek-R1-Distill-Llama-8B's scores discrepancy investigated**: A member reported a large discrepancy in scores when running `lm_eval` with **vllm** on the `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` model, noting scores of **53.03** with **tp=4**, **43.94** with **tp=2**, and **43.43** with **tp=1**.
   - The command used was `lm_eval -m vllm -a deepseek-ai/DeepSeek-R1-Distill-Llama-8B,max_length=34000,tensor_parallel_size=x, -t gpqa_diamond_cot_zeroshot -b auto --apply-chat-template --log_samples --gen_kwargs temperature=0.6,top_p=0.95`.
- **Potential issue with vllm's implementation surfaces**: In response to the score discrepancies, another member suggested that the issue might stem from **vllm's implementation**.
   - The member offered to investigate the samples if available, which the original reporter said they could provide later.


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1346854838645817365)** (23 messages🔥): 

> `Enterprise Deployment, B2B Lead Times, Community Feedback` 


- **Cohere's Enterprise Deployment Inquiry**: A user, **brad062677**, inquired about connecting with someone from Cohere regarding enterprise deployment, noting they had emailed support a week prior and were hoping for a faster reply via Discord.
- **B2B Lead Times in AI**: A user mentioned that enterprise inquiries are handled through direct sales and that B2B lead times can be slow, potentially taking up to **six weeks** in the industry, while another user stated that AI companies usually reply within **two to three days**.
   - A Cohere employee, **1vnzh**, apologized for the delay and assured the user that they were discussing how to engage internally and that the user would receive a reply today.
- **Feedback on Community Improvement**: A user asked for feedback on how to improve the community, stating *we trying to make the community a better place for everyone .. what is missing ? dm's are open if you dont want to talk public*.


  

---


### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1346968241582506117)** (1 messages): 

> `Aya Vision, Multilingual AI, Multimodal Models, Open-Weights Model, AyaVisionBenchmark` 


- **Cohere Debuts Aya Vision, a Multilingual Marvel**: Cohere For AI launched **Aya Vision**, an **8B and 32B** open-weights multilingual vision research model that extends capabilities to **23 languages**.
   - It performs well in image captioning, visual question answering, text generation, and translating both text and images; details are available in [Cohere's blog post](https://cohere.com/blog/aya-vision).
- **Aya Vision Lands on Hugging Face and Kaggle!**: The Aya Vision model is now accessible on [Hugging Face](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision) and [Kaggle](https://www.kaggle.com/models/cohereforai/aya-vision), broadening access for developers and researchers.
   - This makes it easier for the community to experiment with and build upon this state-of-the-art multilingual vision model.
- **Poe Platform Plugs into Aya Vision**: Aya Vision is available on [Poe](https://poe.com/Aya-Vision) offering advanced vision-language capabilities within the platform.
   - It's a **32B** open-weights multimodal model optimized for various vision-language use cases and trained in **23 languages**.
- **WhatsApp Wonders with Worldwide Aya Access!**: Users can now text Aya for free on WhatsApp from anywhere, enabling them to ask text and visual questions, caption images, and translate both text and images into natural language via [this link](https://cohere.com/research/aya/whatsapp).
   - Aya is available in **23 languages** and offers a foundation for languages in natural language understanding, summarization, and translation tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/aya-vision">Aya Vision: Expanding the worlds AI can see</a>: Introducing a state-of-the-art open-weights vision model, Aya Vision from Cohere For AI. </li><li><a href="https://www.kaggle.com/models/cohereforai/aya-vision">CohereForAI | Aya Vision | Kaggle</a>: C4AI Aya Vision is an open weights research release of 8B and 32B parameter models with advanced capabilities optimized for a variety of vision-language use cases, including OCR, captioning, visual re...</li><li><a href="https://poe.com/Aya-Vision">Aya-Vision - Poe</a>: Aya Vision is a 32B open-weights multimodal model with advanced capabilities optimized for a variety of vision-language use cases. It is model trained to excel in 23 languages in both vision and text:...</li><li><a href="https://cohere.com/research/aya/whatsapp">Text Aya on WhatsApp | Cohere For AI</a>: Available in 23 languages, Aya Expanse is the best multilingual AI in the world. Now available on WhatsApp, text Aya in your language, for free.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1346853704313208907)** (1 messages): 

> `Cohere Reranker v3.5 Latency` 


- **Cohere Reranker v3.5 Latency Numbers Remain Elusive**: A member inquired about latency numbers for **Cohere Reranker v3.5**, noting that while an intention to share a graph or numbers was mentioned in a [Pinecone interview](https://www.pinecone.io/learn/cohere-rerank/), it has not yet materialized.
   - After attempting to search for the latency numbers, no results were found.
- **Community Awaits Cohere Reranker v3.5 Latency Data**: Despite initial expectations set during a Pinecone interview, concrete latency figures or a graph for **Cohere Reranker v3.5** remain unpublished.
   - The absence of this data is causing some to actively seek out this information for performance assessment and comparison.


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1347060801055490048)** (1 messages): 

> `Mindmap Generation, Pretrained Models, Mathematical Models` 


- **Student Seeks Guidance on Mindmap Project**: A student is developing a website that generates mindmaps from chapter content, aiming for a hierarchical structure of topics and subtopics.
   - They are considering using a pretrained model initially, followed by creating a custom mathematical model and are seeking advice on how to proceed.
- **Choosing Between Pretrained and Mathematical Models**: The student is unsure whether to start with a pretrained model or a custom mathematical model for generating mindmaps.
   - They are looking for suggestions on the best approach to integrate both methods into their project.


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1346962182549798942)** (2 messages): 

> `Introductions, Sales Contact` 


- **New member seeks Sales Connection**: A new member is trying to connect with someone from the **sales / enterprises support** team.
   - They left a message in the introductions channel, but no one responded to the request yet.
- **Stickied Intro message welcomes new members**: The welcome message encourages members to introduce themselves including details such as **Company/Industry/University**, **What you're working on**, **Favorite tech/tools you use**, and **What you hope to gain from this community**.
   - No members have provided introduction messages yet in this channel.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1346813249366003723)** (9 messages🔥): 

> `ShapeTracker merging proof, 96GB 4090 on Taobao, Rust CubeCL` 


- **ShapeTracker Merging Proof Almost Ready**: A member announced a nearly complete proof in Lean for merging **ShapeTrackers** in this [repo](https://github.com/Nielius/Tensorlayouts), with additional context available in [this issue](https://github.com/tinygrad/tinygrad/issues/8511#issuecomment-2700706082).
   - The proof doesn't yet account for offsets and masks, but the member believes extending it to include these factors would be straightforward, despite requiring *a lot* of work.
- **Novel 96GB 4090 Spotted on Taobao**: A member shared a link to a **96GB 4090** being sold on Taobao ([X post](https://x.com/yomix1337/status/1893692548108984391?s=46)).
   - Another member clarified that *this isn't taobao and it's not available yet* and that *it's gonna be a couple months until you can buy them*.
- **Curiosity Sparks for Rust CubeCL**: A member inquired about the quality of **Rust CubeCL**, noting it's created by the same team behind **Rust Burn**.
   - The member was *wondering if Rust CubeCL was good*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/yomix1337/status/1893692548108984391?s=46">Tweet from Gene edited Yostuba (@Yomix1337)</a>: @EsotericCofe coming out after may</li><li><a href="https://github.com/Nielius/Tensorlayouts">GitHub - Nielius/Tensorlayouts: Lean proof  of necessary and sufficient conditions for merging two tensor views</a>: Lean proof  of necessary and sufficient conditions for merging two tensor views - Nielius/Tensorlayouts</li><li><a href="https://github.com/tinygrad/tinygrad/issues/8511#issuecomment-2700706082">On the Mergeability of arbitrary ShapeTrackers · Issue #8511 · tinygrad/tinygrad</a>: Heyo, I&#39;d like to propose a new formulation and a proof of the view merging problem which I haven&#39;t seen anyone mention yet. I have seen a formulation by person called @Nielius but sadly it wa...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1347241913601884250)** (3 messages): 

> `RANGE Op, iGPU detection on Linux` 


- **RANGE Op Operation Questioned**: A member inquired about the operation of the `RANGE` Op, noting its apparent absence in the `Tensor` implementation of `arrange`.
   - The member later realized it *"isn't a range"* and apologized for the confusion.
- **Linux iGPU Auto-Detection in tinygrad Asked**: A member questioned whether the default device initialization or `Device.get_available_devices()` should automatically detect an **iGPU** on Linux.
   - An attached image shows *"Device: [CPU]"* with the member seemingly expecting *"Device: [GPU]"*.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1346864089363779714)** (7 messages): 

> `HF Checkpoints, special_tokens.json, TorchTune Checkpointer, Github Stars` 


- **Special Tokens' Special Treatment**: A member identified that the **TorchTune checkpointer** copies the original **special_tokens.json** from Hugging Face instead of a potentially modified, custom version, and pointed to the relevant [code](https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/training/checkpointing/_checkpointer.py#L892-L896).
   - A user suggested that a quick fix is to manually replace the downloaded **special_tokens.json** with the custom version.
- **Checkpointer Customization Considerations**: There was a discussion about supporting the use case of **custom special_tokens.json** by passing a new argument to the checkpointer's `save_checkpoint` method.
   - However, the team decided against exposing new arguments without a strong reason, so the recommendation is to manually copy the file for now.
- **TorchTune soars to 5k GitHub Stars!**: The Torchtune project achieved a milestone of **5,000 stars on GitHub**.
   - The community celebrated this achievement.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07">GitHub - pytorch/torchtune at 80da6a5dae23a201595d07041c12ffde830332d7</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/training/checkpointing/_checkpointer.py#L892-L896.">torchtune/torchtune/training/checkpointing/_checkpointer.py at 80da6a5dae23a201595d07041c12ffde830332d7 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1347262585464029184)** (3 messages): 

> `GRPO recipe, Memory issues, Excessive torch.cuda.empty_cache()` 


- **GRPO Recipe littered with Empty Cache calls**: A member inquired about the excessive use of `torch.cuda.empty_cache()` calls in the **GRPO recipe**.
   - Another member admitted that many of these calls are likely excessive, stemming from early development when they faced **memory issues**.
- **GRPO PRs Awaiting Review**: Two **GRPO PRs**, specifically **#2422** and **#2425**, have been open for two weeks and are awaiting review.
   - A member is requesting assistance in reviewing them, asking someone to help unload the queue.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1346810951265157201)** (4 messages): 

> `Berkeley vs MOOC lectures, Certificate Declaration Forms` 


- **MOOC Students Attend Same Lectures as Berkeley Students**: A member asked if Berkeley students have lectures that MOOC students don't, and another member clarified that **Berkeley students and MOOC students attend the same lectures**.
- **Certificate Award Problems**: A member mentioned submitting a certificate declaration form in December but was told there was no submission recorded.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1346806842453524540)** (2 messages): 

> `AST Metric Definition, V1 Dataset Construction` 


- **AST Metric: Function Call Formatting Rate**: A member inquired whether the **AST metric** represents the percentage of LLM responses that generated a correctly formatted function call.
   - No response was given.
- **V1 Dataset: Construction**: A member questioned how the **V1 dataset** was constructed.
   - No response was given.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1347296310994403410)** (1 messages): 

> `Gemini 2, GPT o3-high, Deepseek R1, Prompt tool calling, Python tool` 


- **Debate over Best Model for Prompt Tool Calling**: A user inquired about the best model for prompt tool calling among **Gemini 2**, **GPT o3-high**, and **Deepseek R1**, specifically for calling a **Python tool**.
- **Model Selection for Python Tool Integration**: The user is evaluating **Gemini 2**, **GPT o3-high**, and **Deepseek R1** to determine which one is most suitable for calling a **Python tool** based on prompts.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1347217481147744317)** (1 messages): 

> `Jamba 1.6 launch, Open model for enterprise deployment, Jamba 1.6 performance benchmarks, Hybrid SSM-Transformer architecture, 256K context window` 


- **AI21 Labs Drops Jamba 1.6**: AI21 Labs launched **Jamba 1.6**, an open model tailored for private enterprise deployment, with model weights available on [Hugging Face](https://huggingface.co/ai21labs).
   - The company claims it *delivers unmatched speed and performance*, setting a new benchmark for enterprise AI without compromising efficiency, security and data privacy.
- **Jamba 1.6 Shows Off Arena Prowess**: **Jamba 1.6** reportedly outperforms **Cohere**, **Mistral**, and **Llama** on the Arena Hard benchmark, rivaling leading closed models according to [AI21's announcement](https://www.ai21.com/jamba/).
   - The release highlights its suitability for fully private on-prem or VPC deployment, boasting lightning-fast latency and a market-leading **256K context window**.
- **Hybrid Architecture Gives Jamba 1.6 Edge**: The **AI21 Jamba** family features hybrid **SSM-Transformer** foundation models, excelling in both quality and speed, thanks to its novel **Mamba-Transformer MoE architecture** designed for cost and efficiency gains as explained in the [Jamba 1.6 blogpost](https://www.ai21.com/jamba/).
   - The model is deployable anywhere, self-hosted, or in the AI21 SaaS, to meet diverse data security needs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ai21labs">ai21labs (AI21)</a>: no description found</li><li><a href="https://www.ai21.com/jamba/">Jamba 1.6: The Best Open Model for Enterprise Deployment</a>: Explore Jamba by AI21 – a cutting-edge, long-context AI open model built for accuracy, efficiency, and powerful text generation.
</li>
</ul>

</div>
  

---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

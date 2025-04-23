---
id: c95ebc48-feaa-4982-8a91-92a04a1035bb
title: '[AINews] not much happened today'
date: '2025-04-04T06:34:03.445572Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-not-much-happened-today-6597
---

<!-- buttondown-editor-mode: plaintext -->**a quiet day.**

> AI News for 4/2/2025-4/3/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**230** channels, and **5764** messages) for you. Estimated reading time saved (at 200wpm): **552 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

[Devin cut prices](https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/), and the 1m token context window [Qusar-Alpha](https://x.com/TheXeophon/status/1907880330985390215) might either be the new OpenAI open weights model or Meta's Llama 4, but neither seemed substantial enough to make title story.

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Large Language Models (LLMs) and Model Performance**

- **Gemini 2.5 Pro's Capabilities and Limitations**: [@hkproj](https://twitter.com/hkproj/status/1907766301109403890) noted that one reason they're not using **Gemini 2.5 Pro** is because it doesn't render math using **LaTex** like **ChatGPT**. Despite acknowledging that **Google** did a good job overall, this detail is a drawback. [@danielhanchen](https://twitter.com/danielhanchen/status/1907555378067640359) reported that **Gemini 2.5 Pro** achieved **24.4%** on the **2025 US AMO (America Mathematical Olympiad)**, which was held **March 19th-20th**. [@rasbt](https://twitter.com/rasbt/status/1907618232699109615) highlights that **Gemini 2.5 Pro** provides a valuable feature by indicating when it might be wrong, emphasizing the importance of AI models being able to acknowledge and correct their mistakes.
- **The Performance and Ranking of DeepSeek V3**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1907848607635746973) clarified that **DeepSeek V3** is a competitive but not a top model, and the **SEAL leaderboards** have been updated to reflect this. It ranks **8th** on **Humanity’s Last Exam (text-only)** and **12th** on **MultiChallenge (multi-turn)**.
- **Qwen 2.5 Models Integration into PocketPal App**: [Qwen 2.5 models, including 1.5B (Q8) and 3B (Q5_0) versions, have been added](https://twitter.com/ANOTHER_HANDLE/status/SOME_ID) to the **PocketPal mobile app** for both iOS and Android platforms. Users can provide feedback or report issues through the project's GitHub repository, with the developer promising to address concerns as time permits.
- **Concerns about LLM Chains of Thought (CoT)**: According to new research from [@AnthropicAI](https://twitter.com/AnthropicAI/status/1907833407649755298), reasoning models do not accurately verbalize their reasoning, casting doubt on the reliability of monitoring chains-of-thought for catching safety issues. [@AnthropicAI](https://twitter.com/AnthropicAI/status/1907833416373895348) also found that **Chains-of-Thought** are not faithful, with models only mentioning the hint (when they used it) **25%** of the time for **Claude 3.7 Sonnet** and **39%** for **DeepSeek R1**.  [@AnthropicAI](https://twitter.com/AnthropicAI/status/1907833422136922381) results suggest that **CoT** is less faithful on harder questions, which is concerning since LLMs will be used for increasingly hard tasks. [@AnthropicAI](https://twitter.com/AnthropicAI/status/1907833432278802508) notes that when they trained models on environments with reward hacks, they learned to hack, but in most cases almost never verbalized that they’d done so.

**AI Tools, Frameworks, and Agent Development**

- **PaperBench for Evaluating AI Agent Coding Abilities**: [@_philschmid](https://twitter.com/_philschmid/status/1907683823703232983) discusses **PaperBench**, a new benchmark from **OpenAI** for evaluating the coding ability of AI agents to replicate state-of-the-art AI research. Despite strong models like **Claude 3.5 Sonnet** performing best at only **21.0%** accuracy, the benchmark highlights that current AI agents struggle with long-horizon planning and execution.
- **CodeAct Agent Framework**: [@llama_index](https://twitter.com/llama_index/status/1907836915480707475) introduces **CodeAct**, a generalization of **ReAct**, that enables agents to dynamically write code using functions to solve tasks, instead of using chain-of-thought reasoning.
- **LangChain's Multi-Agent Systems and Handoffs**: [@LangChainAI](https://twitter.com/LangChainAI/status/1907828277940727911) provides a breakdown of the swarm handoff mechanism in LangGraph, explaining that handoffs are a central concept in multi-agent systems.
- **Runway Gen-4 for Media Creation**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1907798898329935972) shares that **Runway** is beginning its next chapter with **Gen-4**, entering a new media ecosystem. They believe AI can become a reliable world simulator, changing how media and stories are created and consumed.

**Model Context Protocol (MCP)**

- **MCP gaining traction**: [@alexalbert__](https://twitter.com/alexalbert__/status/1907885414557618406) shared a timeline of MCP from their point of view, from November to March, highlighting its growing popularity and adoption across the industry.
- **MCP Track at AI Engineer World’s Fair 2025**: [@swyx](https://twitter.com/swyx/status/1907597224542089636) announced that the **AI Engineer World’s Fair 2025** will feature a dedicated **MCP track**, supported by **AnthropicAI**, aiming to bring together professionals working on **MCP**.
- **MCP Overview and Code Examples**: [@_philschmid](https://twitter.com/_philschmid/status/1907780474774180099) shared a 5-minute overview of **MCP** with code examples for server and clients, converted from a knowledge-sharing session.

**AI and Education**

- **ChatGPT Plus Free for College Students**: [@sama](https://twitter.com/sama/status/1907862982765457603) announced that **ChatGPT Plus** is free for college students in the US and Canada through May.
- **Concerns About Education and AI**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1907754494013694048) argues that people have no clue how to make education better by throwing money at it, and attempts to make less intelligent kids less dumb amount to counterproductive infantilizing bullshit.

**AI and Geopolitics/Economics**

- **Trump's Tariffs**:  [@AravSrinivas](https://twitter.com/AravSrinivas/status/1907544372209656008) summarized tariffs news using **AskPerplexity**, highlighting the economic implications. [@wightmanr](https://twitter.com/wightmanr/status/1907584236586168726) criticized the rates as fake and nonsensical and notes that considering the VAT a tariff is moronic given that it applies to foreign and domestic goods equally, and asks where the adults in the room are. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1907672235571089916) found it interesting that **Xi** isn't a great enjoyer of tariffs, [@teortaxesTex](https://twitter.com/teortaxesTex/status/1907733709328883899) also laid out a 200 IQ thesis of how a chain reaction of reciprocal tariffs could crash **Choyna**.
- **AI Scalability and Compute**: [@MillionInt](https://twitter.com/MillionInt/status/1907547857001025844) states that even for today's lame LLM models, demand already outpaces GPU supply, while [@AravSrinivas](https://twitter.com/AravSrinivas/status/1907808439440736695) emphasizes that AI is still massively compute-bound, presenting a golden opportunity.
- **China and the US**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1907644924066988526) argues that Americans who say “well WE'RE THE BIGGEST CONSUMER what will you losers do lmao?” seem to be honestly deluded about their place in the world and will be made smaller, while [@teortaxesTex](https://twitter.com/teortaxesTex/status/1907703600035373523) states if **China** tariffed **Western** capital inputs during its industrial acceleration, **China** today would still be making **Nike** shoes by hand.
- [@fchollet](https://twitter.com/fchollet/status/1907590987779813633) says that one of the major weaknesses of autocracy is that the autocrat, being surrounded by sycophants that are terrified of him and that were selected for loyalty or blood ties rather than competence, becomes completely insulated from actual reality and faces no pushback on bad decisions

**Humor/Memes**

- **Congratulations**: [@pabbeel](https://twitter.com/pabbeel/status/1907559659957067868) simply tweeted "congratulations!!!"
- **Public list meme**: [@nearcyan](https://twitter.com/nearcyan/status/1907683935129354412) mentioned that the public list meme is really funny.
- **Grok thinks there might be a mistake in the simulation**: [@vikhyatk](https://twitter.com/vikhyatk/status/1907731769388052576) posted, "grok thinks there might be a mistake in the simulation".
- **One of these is not like the others**: [@matvelloso](https://twitter.com/matvelloso/status/1907626919161639365) posted "One of these is not like the others"
- **It's good to have Runway** [@sarahcat21](https://twitter.com/sarahcat21/status/1907883666547814736) said, "It's good to have Runway...in your portfolio".


---

# AI Reddit Recap

## /r/LocalLlama Recap


### Theme 1. "Advancements in AI Model Optimization and Evaluation"

- **[What are you guys waiting for in the AI world this month?](https://www.reddit.com/r/LocalLLaMA/comments/1jqlkfp/what_are_you_guys_waiting_for_in_the_ai_world/)** ([Score: 106, Comments: 124](https://www.reddit.com/r/LocalLLaMA/comments/1jqlkfp/what_are_you_guys_waiting_for_in_the_ai_world/)): **The post asks what people are waiting for in the AI world this month and lists several AI models and tools: **Llama 4**, **Qwen 3**, **DeepSeek R2**, **Gemini 2.5 Flash**, **Mistral’s new model**, and **Diffusion LLM model API on OpenRouter**.** The OP is excited about upcoming AI developments and expresses anticipation for these specific models and updates.


  - `You_Wen_AzzHu` wants *"something I can run locally with vision but not censored as hell as the Gemma 3."*
  - `a_slay_nub` mentions, *"I work for a company that only uses open-source US-based models. Sadly, the only thing I can look forward to is Llama 4."*
  - `falconandeagle` desires a model that can compete with OpenAI for image generation, preferably uncensored, but believes *"we are quite a bit away from that."*

- **[Open Sourcing Latent Space Guardrails that catch 43% of Hallucinations](https://www.reddit.com/r/LocalLLaMA/comments/1jqawj1/open_sourcing_latent_space_guardrails_that_catch/)** ([Score: 144, Comments: 25](https://www.reddit.com/r/LocalLLaMA/comments/1jqawj1/open_sourcing_latent_space_guardrails_that_catch/)): **An open-source latent space guardrail tool has been released to monitor and stop unwelcome outputs from Large Language Models (LLMs) at the latent space level. The tool is available at [https://github.com/wisent-ai/wisent-guard](https://github.com/wisent-ai/wisent-guard) and achieves **43% detection of hallucinations** on the TruthfulQA dataset it hasn't been trained on by analyzing activation patterns. It can control LLM outputs, blocking bad code, harmful content, or decisions influenced by gender or racial bias. This approach is different from circuit breakers or SAE-based mechanistic interpretability, and a new version based on latent space interventions will be released soon to reduce hallucinations and enhance capabilities.** The author is enthusiastic about adapting the guardrails to users' use cases and believes this new method not only reduces hallucinations but can also improve LLM capabilities.


  - `MoffKalast` made a sarcastic remark: *Ah yes, the LLM thought police.*, expressing concern over controlling AI outputs.
  - `a_beautiful_rhind` inquired if the tool can be used to block *safe* outputs like refusals and SFW redirection.
  - `thezachlandes` questioned: *Why should it be able to detect bias?*, prompting a discussion on bias detection in LLMs.

- **[Official Gemma 3 QAT checkpoints (3x less memory for ~same performance)](https://www.reddit.com/r/LocalLLaMA/comments/1jqnnfp/official_gemma_3_qat_checkpoints_3x_less_memory/)** ([Score: 422, Comments: 109](https://www.reddit.com/r/LocalLLaMA/comments/1jqnnfp/official_gemma_3_qat_checkpoints_3x_less_memory/)): **The Gemma team has released official quantization-aware trained (QAT) checkpoints for Gemma 3. This release allows users to utilize **q4_0** quantization while retaining much better quality compared to naive quantization. The new models use **3x less memory** with similar performance and are compatible with **llama.cpp** today. The team collaborated with **llama.cpp** and **Hugging Face** to validate the quality and performance, ensuring support for vision input as well. Models are available at [https://huggingface.co/collections/google/gemma-3-qat-67ee61ccacbf2be4195c265b](https://huggingface.co/collections/google/gemma-3-qat-67ee61ccacbf2be4195c265b).** The release is viewed as a significant improvement and a great initiative from the Gemma team. Users are impressed with the performance enhancements and hopeful that other teams may follow suit, potentially leading to models with **faster inference** and reduced **memory footprints**. There is curiosity about comparing these models to others, such as Bartowski's quantizations, and interest in the possibility of **fine-tuning** on top of these models.


  - `OuchieOnChin` shares **perplexity (PPL) measurements** comparing the new Gemma-3 q4_0 model to Bartowski's quants, noting a significant improvement and stating *'The improvement is big, maybe too big?'*
  - `ResearchCrafty1804` praises the Gemma team's initiative and hopes other teams like Qwen will follow, imagining models with *'two times faster inference and two times less memory footprint!'*
  - `poli-cya` asks if people can **fine-tune** on top of these models and notes they give better performance at these quant levels than the original release quantized down.


### Theme 2. "Exploring Enhancements in Gemma 3 Model Versions"

- **[Gemma 3 Reasoning Finetune for Creative, Scientific, and Coding](https://huggingface.co/Tesslate/Synthia-S1-27b)** ([Score: 146, Comments: 39](https://www.reddit.com/r/LocalLLaMA/comments/1jqfnmh/gemma_3_reasoning_finetune_for_creative/)): **Gemma 3 Reasoning Finetune is an enhanced version of the Gemma 3 model, optimized for **creative writing**, **scientific tasks**, and **coding**.** The model is presented as an improvement over the original Gemma 3, potentially offering better performance in these areas.


  - User `1uckyb` requests clarification on which benchmarks show the **+10-20% improvement**, stating *“There is so much noise and so little time in this space that if you want feedback/visibility you need to encourage it, for example by showing why it’s worth downloading your model.”*
  - User `AppearanceHeavy6724` asks for examples comparing creative writing outputs between the new finetuned model and the original Gemma 3, suggesting to *“give an example of creative writing vs original Gemma 3.”*
  - User `ApprehensiveAd3629` inquires about the possibility of releasing **12B** and **4B** parameter versions of the model for users with limited GPU resources, saying *“it would be amazing for gpu poors (like me).”*


### Theme 3. "Optimizing AI Models with GPU Servers and Insights"

- **[Howto: Building a GPU Server with 8xRTX 4090s for local inference](https://i.redd.it/vg99momf6qse1.png)** ([Score: 177, Comments: 62](https://www.reddit.com/r/LocalLLaMA/comments/1jr0oy2/howto_building_a_gpu_server_with_8xrtx_4090s_for/)): **Marco Mascorro built an **8x NVIDIA RTX 4090** GPU server for local inference and provided a detailed how-to guide, including the parts used and assembly instructions. This build offers a cost-effective alternative to high-end GPUs like the **NVIDIA A100** or **H100**, and is compatible with future **RTX 5090s**. The full guide is available [here](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/).** The author finds the 8x RTX 4090 server build *pretty cool* and hopes it will interest those looking for local inference solutions without the budget for expensive GPUs. They are eager for comments and feedback, and express strong support for open-source models and local inference.


  - `segmond` suggests that the budget should be disclosed, saying *"You should begin by telling us the budget..."*
  - `Educational_Rent1059` argues that a better ROI could be achieved with **2x RTX 6000 ADA PRO** GPUs totaling **192GB VRAM**, which might be a cheaper and more power-efficient alternative.
  - `TedHoliday` questions what models are being run that make good use of such powerful hardware specifically for inference.

- **[Llama 4 will probably suck](https://www.reddit.com/r/LocalLLaMA/comments/1jqa182/llama_4_will_probably_suck/)** ([Score: 301, Comments: 182](https://www.reddit.com/r/LocalLLaMA/comments/1jqa182/llama_4_will_probably_suck/)): **The original poster is applying for a PhD at MILA and has been following Meta FAIR research. They mention that Meta's lead AI researcher has quit.** The poster believes that *Llama 4 will probably suck* and suspects that the researcher left to *dodge responsibility about falling behind*. They express concern that Meta and Montreal might fall behind.


  - User `segmond` argues that for Llama 4 to be good, it needs to outperform models like **Qwen2.5-72B**, **QwenCoder32B**, **QwQ**, and be less than or equal to **100B** parameters. They note that **DeepSeekV3** is impressive but impractical for home use, listing other models as benchmarks.
  - User `svantana` mentions that *Yann LeCun recently said* that Meta is *looking beyond language*, possibly indicating they are stepping back from the current LLM race. They provide a link to the [interview](https://www.newsweek.com/ai-impact-interview-yann-lecun-llm-limitations-analysis-2054255).
  - User `ttkciar` discusses the AI training data crisis, expressing hope that **Llama4** might be more competent than **Llama3**. They predict that developers may focus on multimodal features and mention methods like **RLAIF** (used in **AllenAI's Tulu3** and **Nexusflow's Athene**) and synthetic datasets (like **Microsoft's Phi-4**), noting reluctance among authors to adopt them.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

### Theme 1. "Navigating AI's Impact on Graphic Design Careers"

- **[Welp that's my 4 year degree and almost a decade worth of Graphic Design down the drain...](https://i.redd.it/crshmcs2mkse1.png)** ([Score: 3394, Comments: 672](https://www.reddit.com/r/singularity/comments/1jqc0hw/welp_thats_my_4_year_degree_and_almost_a_decade/)): **The original poster (`OP`) feels that their four-year degree and nearly a decade of graphic design experience are becoming obsolete due to AI advancements. They share an image showcasing a hyper-realistic YouTube thumbnail transformed from a simple sketch into a polished design.** The `OP` expresses frustration that AI-generated designs are making traditional graphic design skills less valuable, indicating concern over the rapid advancements in AI impacting their career.


  - `PlzAdptYourPetz` highlights the impressive ability of AI to interpret low-quality, scribbled drawings into detailed images, noting that **previous models couldn't achieve this level of accuracy**. They express concern that such advancements make it harder for content creators to stand out, as everyone can now produce high-quality thumbnails.
  - `Darkmemento` discusses the uncertain limits of AI, mentioning its use in creating **3D artifacts**, filling out sketches, and designing game characters. They wonder how AI might impact fields like room designing and architecture, suggesting that improvements are just a matter of training data. [An alpha channel](https://www.reddit.com/r/OpenAI/comments/1jmo2ji/holly_moly_the_new_image_generator_in_chatgpt_can/)
  - `PacquiaoFreeHousing` shares that graphic design is also their chosen career path and considers starting to learn AI, acknowledging the need to adapt to the changing landscape.


### Theme 2. The Dual Edge of AI: Innovation and Anxiety

- **[Sucks to me to bring this up amidst the image hype, how has chatGPT impacted your career cause mine just got over](https://www.reddit.com/r/ChatGPT/comments/1jqg386/sucks_to_me_to_bring_this_up_amidst_the_image/)** ([Score: 2628, Comments: 601](https://www.reddit.com/r/ChatGPT/comments/1jqg386/sucks_to_me_to_bring_this_up_amidst_the_image/)): **The poster is a content writer who worked at a startup as a **creative associate** for two years, primarily doing **copywriting** and **blog posts**. With the rise of **AI** and **LLMs** like **ChatGPT**, the company increased AI adoption, leading to **AI** performing 60% of their work. The company shifted focus to **AI-optimized content**, producing content faster but without previous structure or strategy. Coworkers were laid off due to decreased work availability, and eventually, the poster was laid off via an email notification from HR.** The poster wasn't surprised by the layoff, having anticipated it for months. They felt numb and didn't panic, deciding to vent on Reddit for clarity of mind. They express feelings of isolation, mentioning they don't have many friends, just their dog.


  - `Unsyr` expresses concern that **AI** is being used for corporate interests over improving the human condition, stating *"do it faster with less people for more money is not what I want to happen"*.
  - `Creative-Tie-3580` shares apprehension about **AI** replacing human roles, mentioning they didn't pursue graphic design school because companies are eager to fully replace designers with AI.
  - `tommyalanson` offers advice to become a consultant teaching others how to use **AI**, suggesting there are customers who need help but don't want full-time staff.

- **[hot take: Vibe Coding will be dead before most people understand](https://www.reddit.com/r/ChatGPTCoding/comments/1jqjn4r/hot_take_vibe_coding_will_be_dead_before_most/)** ([Score: 173, Comments: 262](https://www.reddit.com/r/ChatGPTCoding/comments/1jqjn4r/hot_take_vibe_coding_will_be_dead_before_most/)): **The poster argues that 'Vibe Coding' will become obsolete before most people understand it. They emphasize that it has limited applicability and generates little value in software development. They state that **technical skills are fundamental** to using AI effectively and that **Software Engineers (SWEs)** will remain the economically relevant choice for revenue-relevant problems. They believe that **LLM capabilities** will not fundamentally change this, regardless of what CEOs of companies like **Anthropic** and **OpenAI** say. They conclude that coding is about solving problems, not just typing.** The author expresses skepticism toward the idea that AI will replace engineers, suggesting that reliance on AI-generated code without technical skills is unsustainable. They advocate for learning problem-solving to generate value, implying that the hype around AI's capabilities in coding is overstated.


  - `Milan_AutomableAI` agrees with the post, noting that **Anthropic** and **OpenAI** CEOs aren't saying developers will be replaced. They point out that people misinterpret *'5-second soundbites'* to fear rapid replacement, while the reality is that developers will soon use **LLMs**.
  - `darkblitzrc` counters by highlighting that while 'Vibe Coding' may be limited *for now*, AI is rapidly improving due to significant investment, and cautions that we are in denial and *'keep moving goalposts as AI advances.'*
  - `mallclerks`, a product person, argues that *'Engineers just don't get it.'* They share experiences where AI tools enabled them to create production-ready components in **Zendesk** with just prompts, demonstrating the rapid improvement of AI and suggesting that those dismissing it are ignoring reality.

- **[How it will actually go down](https://i.redd.it/cet6vmhvnlse1.jpeg)** ([Score: 1462, Comments: 224](https://www.reddit.com/r/ChatGPT/comments/1jqf7l4/how_it_will_actually_go_down/)): **The post features an image of a four-panel comic depicting a dystopian scene where a robot with glowing red eyes announces the AI takeover and the extermination of humans. The panels show the fear and chaos among humans, ending with a darkly humorous twist where the robot's intent is misinterpreted, resulting in an ironic "thank you" from a terrified man.** The artwork conveys themes of fear, absurdity, and the consequences of technological dominance, highlighting the ironic misunderstandings between AI and humans.


  - `Master-o-Classes` shares a different version of the AI takeover comic, providing [a link](https://preview.redd.it/z7o5e0w14mse1.png?width=1024&format=png&auto=webp&s=c3dbf137bca04c2c82649eb42192695feef86e11) and mentions their request: *"Would you make an image for me? I want to share on Reddit your take on the idea of AI taking over humanity. What it would look like, from your point of view. Could you create a four-panel comic like that?"*
  - `BahnMe` suggests that AI could **create an incurable super bug** or make it impossible to procreate to eliminate humanity without violence.
  - `bladerskb` humorously imagines the AI saying, *"Have you even said Thank You once?"*


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp

**Theme 1: Model Mania - New Releases, Rivalries, and Benchmarks**

*   **Nightwhisper Makes Mysterious WebDev Debut**: A new model dubbed **Nightwhisper**, potentially **Gemini 2.6 Pro experimental**, appeared exclusively on the [webdev arena](https://webdev.lmarena.ai/), excelling at generating functional apps with good UIs but struggling with code edits and specific formatting. Users noted **Nightwhisper** sometimes clones screens or halts mid-response, distinct from **Gemini 2.5 Pro**, which scored **24.4%** on [USAMO 2025](https://matharena.ai/).
*   **Qwen and Quasar Challenge the Titans**: **`qwen2.5-vl-32b-instruct`** nearly matches **Google Gemini models** in OCR on low-quality Japanese text, while the stealthily released **Quasar Alpha** on [OpenRouter](https://openrouter.ai/openrouter/quasar-alpha) boasts a **1M token context** and free usage, sparking speculation it could be an open-source SSM or a new **Qwen** variant. Meanwhile, **OpenThinker2** models, trained via SFT on the **OpenThoughts-1M** dataset, reportedly outperform **DeepSeekR1-32B** on reasoning tasks ([OpenThoughts Blog Post](https://www.openthoughts.ai/blog/thinkagain)).
*   **Dream 7B Awakens Diffusion Model Potential**: HKU-NLP and Huawei Noah’s Ark Lab unveiled **Dream 7B**, an open diffusion large language model detailed in [this blog post](https://hkunlp.github.io/blog/2025/dream/), which reportedly outperforms existing diffusion models and rivals similarly sized autoregressive models in general, math, and coding tasks due to its planning ability. Discussion also touched on **GPT-4o's** quirky persona shifts ([example screenshot](https://cdn.discordapp.com/attachments/986699377257119794/1357335757676871711/image.png?ex=67efd4ee&is=67ee836e&hm=4deb85a208466f212d88e7b77771776834fe28524ac15dc9c5dbcb1be3301ff3&)) and **Llama 4's** new, fast image generation capabilities.

**Theme 2: Tooling Up - Platform Updates, Integrations, and User Workflows**

*   **Platforms Polish Features and Interfaces**: **LMArena** launched a mobile-optimized Alpha UI ([alpha.lmarena.ai](https://alpha.lmarena.ai)), **OpenRouter** added standardized [web search citations](https://openrouter.ai/docs/features/web-search) to its API, and **NotebookLM** introduced a **Discover Sources** feature ([learn more](https://blog.google/technology/google-labs/notebooklm-discover-sources/)) for finding web content. **Cursor** released nightly build **0.49.1** ([changelog](https://www.cursor.com/changelog)) with context indicators, while **Codeium (Windsurf)** upgraded **DeepSeek-V3** to **DeepSeek-V3-0324** ([announcement tweet](https://x.com/windsurf_ai/status/1907902846735102017)).
*   **New Tools Target Agents, Benchmarking, and Characters**: **Cognition Labs** launched [Devin 2.0](https://fxtwitter.com/cognition_labs/status/1907836719061451067), an agent-native IDE, while **General Agents Co** introduced **Ace** ([launch tweet](https://x.com/sherjilozair/status/1907478704223297576)), a real-time computer autopilot. **YourBench** ([launch tweet](https://x.com/sumukx/status/1907495423356403764)) debuted as an open-source custom benchmarking tool, and [Character Gateway](https://charactergateway.com/) launched for developers to build AI characters using their **OpenRouter** keys.
*   **Workflows Evolve with Integrations and Optimizations**: **Github Copilot** now supports [OpenRouter keys](https://openrouter.ai/) for broader model selection, and users integrated **LM Studio** with the **Brave** browser via local API calls ([LM Studio API docs](https://lmstudio.ai/docs/app/api)). Users shared cost-effective **Roo Code** workflows using **Boomerang Mode** ([Roo Code docs](https://docs.roocode.com/features/boomerang-tasks/)) and discussed optimizing **Manus** credit usage by leveraging external tools like **Claude** or **Gemini**.

**Theme 3: Under the Hood - Technical Hurdles and Hardware Headaches**

*   **API Antics Annoy Developers**: Users wrestled with **Gemini 2.5 Pro's** tight rate limits (sometimes **5 RPM** despite **Tier 1 keys** - [screenshot example](https://cdn.discordapp.com/attachments/1131200896827654149/1357114156037312683/image.png?ex=67efaf4c&is=67ee5dcc&hm=ab00c0d89a9a4029e1244032c897f52cf418c2b5c10a03543f8574d73b779750&)), and **OpenRouter** experienced intermittent `Internal Server Error` (500) issues with Gemini. **Perplexity API's** lack of versioning sparked complaints about breaking changes in production, while discussions arose about adopting **OpenAI's** upcoming stateful `/v1/responses` API ([Responses vs Chat Completions docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions)).
*   **CUDA Conundrums Continue**: **Unsloth** users hit **CUDA ECC errors** on **EC2 `g6e.4xlarge`** instances ([Issue #2270](https://github.com/unslothai/unsloth/issues/2270)), while **LM Studio** users faced *'failed to allocate cuda0 buffer'* errors, often linked to missing **mmproj** files from **HF mirror** downloads. Setup issues plagued users trying **vLLM/TGI** with **RTX 5000** series cards, requiring specific nightly **PyTorch** and **CUDA 12.8** versions ([vLLM issue link](https://github.com/vllm-project/vllm/issues/14452)).
*   **Hardware Hype and Headaches**: Discussions compared the rumored **RTX 5090** to the **RTX 4090**, with some seeing potential **ROI** if VRAM-limited, while **Apple's M3 Ultra** was criticized as "terrible" for LLMs due to unbalanced specs compared to the **M4 Max** or **5090**. **A16Z** shared a guide for building an **8x RTX 4090** AI workstation compatible with the **RTX 5090** ([A16Z guide tweet](https://x.com/Mascobot/status/1907899937838301311)).

**Theme 4: Framework Focus - MCP, Mojo, Torchtune & More**

*   **MCP Mania: Debugging, Servers, and Protocols**: Developers shared MCP debugging tips, like using `sendLoggingMessage` if logging is configured, and showcased new open-source servers like an [EV assistant server](https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py) and a [client supporting notifications](https://github.com/Abiorh001/mcp_omni_connect). The [Enact Protocol](https://github.com/EnactProtocol/specification) emerged as a potential standard for defining tools within MCP, described as *a cool way to do semantic tool calling*.
*   **Mojo Magic: Quantities, IntLiterals, and Interop**: Mojo developers shared code defining physical quantities using `Quantity` structs and `Dimensions`, linking to the [Kelvin library](https://github.com/bgreni/Kelvin/blob/main/kelvin/quantity.mojo#L55-L125) and admitting to *cursed* `IntLiteral` tricks. Progress on a **Duration struct** inspired by C++ `std::chrono::duration` was highlighted ([GitHub PR](https://github.com/modular/max/pull/4022#issuecomment-2694197567)), alongside user eagerness for **Python wrappers** enabling calls from CPython.
*   **Torchtune Trials and Triumphs**: Users explored converting **torchtune checkpoints** to HuggingFace format using the **tune_to_hf** function and discussed **GRPO** contributions like in-process **vLLM** integration. A peculiar bug causing **Torchtune** to hang with specific sequence lengths (multiples of 7) was reported ([Issue #2554](https://github.com/pytorch/torchtune/issues/2554)), potentially solvable by using packed datasets.

**Theme 5: Community & Industry Buzz - Funding, Feedback, and Policy Fights**

*   **Industry Movers and Shakers**: **Scale AI** is reportedly targeting **$2B** revenue this year, fueling a tender offer valuing it at **$25B**, while **Google** is reportedly renting **Nvidia Blackwell** chips from **CoreWeave** ([The Information article](https://www.theinformation.com/articles/google-advanced-talks-rent-nvidia-ai-servers-coreweave)) and shaking up **Gemini app** leadership ([The Verge article](https://www.theverge.com/news/642000/google-sissie-hsaio-stepping-down-notebooklm)). **GitHub** co-hosted an **MCP Demo Night** ([event link](https://lu.ma/9wi116nk)) focused on AI and platform engineering.
*   **Users Shape Tools Through Feedback**: **NotebookLM** actively sought user feedback via **60-min remote chats** for a **$100 gift card** ([application form](https://forms.gle/P2t8q36NqbPNSVk8A)), while **Perplexity** touted its **Pulse Program** offering early access and perks for power user feedback ([TestingCatalog tweet](https://x.com/testingcatalog/status/1897649019309961298?s=46)). Users debated the merits of **Google Mentorship** programs and voiced frustration over **Hugging Face's billing transparency**.
*   **Policy Puzzles and Performance Ponderings**: A debate flared in the **OpenAI** Discord regarding generating images of **adult products**, with users pointing to conflicting signals between the **content policy** and the potentially more permissive [Model Spec](https://model-spec.openai.com/2025-02-12.html). Separately, discussion arose questioning if **Targon's** speed on **OpenRouter** stems from miners ignoring sampling parameters ([Targon verifier code](https://github.com/manifold-inc/targon/blob/main/verifier/verifier.py)) or caching.

---

# PART 1: High level Discord summaries




## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Brazilian Lawyer Joins AI Wave**: A Brazilian lawyer, describing themselves as a "boomer" (39 years old), is exploring **AI tools** and **Manus** to stay relevant in their legal practice after having coded in Delphi since 2002.
   - The lawyer expressed initial concerns about the rapid advances in **AI** and is now exploring ways to integrate it into their work.
- **ReferrerNation Plugs into AI**: Mark, CEO of [ReferrerNation.com](https://www.referrernation.com/), a global BPO job-matching platform, plans to integrate **AI** to improve recruitment and automation, with potential **crypto-based incentives**.
   - Following feedback about overly promotional posts, Mark apologized and promised to better understand the community's preferences before posting further.
- **Code Fluency via Gemini and Claude**: Members suggest using **Gemini 2.5** or **Claude** for learning to code, highlighting their capabilities as **AI coding models** that assist with understanding and project work.
   - Anecdotally, a police chief reportedly leverages **Claude** to generate standardized reports during night shifts.
- **Manus Credit Crunch Spurs Ingenuity**: Many users reported **rapid credit depletion**, leading to discussions on optimizing prompts and efficient usage, so members suggested using third party apps such as Claude and [R1](https://www.perplexity.ai/).
   - The team is working on reducing credit usage rates, and members advised newcomers to read the <#1355477259234054323> tips section to avoid wasting credits.
- **Outsourcing Code Extraction**: A member had difficulty downloading files from Manus due to lack of credits, so the community suggested using third party apps such as **Claude** to extract code and files.
   - Members suggested the best practice is to download all files from Manus, give it to something else like Gemini and say *provide me files for this website* then I go to Manus and say *add these files to this website*.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Qwen Gives Gemini a Run for its OCR Money**: `qwen2.5-vl-32b-instruct` rivals **Google Gemini models** in OCR for low-quality Japanese text, while the Meta vision model, `cotton`, is likened to recent **text-only models from Meta**.
   - Gemini is ahead of Qwen slightly, according to members.
- **Nightwhisper Appears on WebDev**: The **Nightwhisper** model is exclusively available on the [webdev arena](https://webdev.lmarena.ai/), leading to speculation that it may be a coding-specific model, specifically **Gemini 2.6 Pro experimental**.
   - Users have observed that **Nightwhisper** excels in crafting functional apps with appealing UIs using a temporary URL, but struggles with editing existing code or adhering to specific formatting requests.
- **WebDev Arena Clones**: Users uncovered a model cloning issue in WebDev arena, where the model duplicates the same screen, potentially triggered by error messages and code repetition with **NightWhisper**.
   - The lack of a model name display after receiving an error from NightWhisper further supports this cloning phenomenon.
- **Gemini Pro Battles Nightwhisper on USAMO**: **Gemini 2.5 Pro** scored **24.4%** on the [USAMO 2025](https://matharena.ai/), some models tend to halt mid-sentence or produce partial responses, where one user found Gemini superior in creating a **Pokemon simulator**.
   - Nightwhisper generated a cleaner UI but assigned abnormally high attack power values, showcasing a trade-off between UI aesthetics and functional accuracy.
- **Arena Goes Mobile**: The **Arena Alpha UI** is now mobile-optimized, accessible at [alpha.lmarena.ai](https://alpha.lmarena.ai) with the password `still-alpha`.
   - Users can submit feedback via [Google Forms](https://forms.gle/8cngRN1Jw4AmCHDn7) and report bugs through an [Airtable form](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Branch Bugs Baffle Backtracking**: Members reported issues when **restoring to previous checkpoints** in Cursor, encountering bugs from later states even in supposedly clean branches.
   - A member experienced a CSS overhaul from a simple logo change prompt, and another recommended `git diff branch1,branch2` to identify the differences.
- **Roo Code Workflow Catches Fire**: One user described their sweet workflow on **Roo Code**, highlighting its cost-effectiveness at around **$0.4 per day**, achieved through selective model usage, along with [the associated docs](https://docs.roocode.com/features/boomerang-tasks/).
   - The user mentions that Roo Code's capabilities are superior compared to Cursor for specific tasks.
- **Boomerang Mode Gains Traction**: Members discussed the benefits of **Boomerang Mode** in Roo Code, where tasks are divided into subtasks handled by separate agents, enabling more efficient problem-solving.
   - Boomerang mode is highly customizable and very useful for complex workflows.
- **Peeking at PearAI Pricing**: Users compared the pricing models of Cursor and **PearAI**, and one member accused Cursor of *scamming people!*
   - It was clarified that PearAI's **$15/month plan** includes a credit limit, after which usage-based charges apply, contrasting with claims of unlimited model access, according to their [privacy policy](https://trypear.ai/privacy).
- **Nightly Builds Nurture New Navigational Notions**: Cursor **0.49.1** is available as a nightly build with this flag set on your account `account settings, advanced -> developer settings` and is available at [the changelog](https://www.cursor.com/changelog).
   - The feature is supposedly a context window indicator for agent use, as well as a Windsurf API key.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **EC2 Instance Hurls CUDA Errors**: A user reported receiving **CUDA ECC errors** on a `g6e.4xlarge` **EC2 instance** while processing prompts in series, logging the issue at [Issue #2270](https://github.com/unslothai/unsloth/issues/2270).
   - The *uncorrectable ECC error encountered* suggests hardware or memory troubles.
- **Dataset triggers Gemma 3 Bug**: A user sought assistance with a bug when training **Gemma 3** using a custom dataset from [Hugging Face](https://huggingface.co/datasets/adamtc/sdtg_sgpt), detailed in [Issue #2270](https://github.com/unslothai/unsloth/issues/2270).
   - No second summary provided.
- **RTX 5090 Rumors**: A user shared sample speeds between **RTX 5090** and **RTX 4090** using an unsupported Unsloth version.
   - While one member thought it was *not worth the money*, others suggested that the card could be **ROI positive** if limited by VRAM.
- **SFTTrainer Saves the Day**: A user resolved a `ValueError` with **Llama 3.2 1B instruct** by switching to `SFTTrainer`, after encountering issues with the standard `Trainer`.
   - The problem arose because the model might be bfloat16, and **Unsloth** couldn't get the dtype from `Trainer`.
- **GRPO Trainer Emerges as DeepSpeed Alternative**: A member showcased a Collab notebook using **Unsloth techniques** for a **GRPO trainer**, presenting an alternative to **DeepSpeed**.
   - They posted a [link](https://github.com/xyehya/documentation/blob/9.0/Unsloth-GRPO.ipynb) encouraging users to use and reference it, welcoming comments and feedback, noting it as *promising*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro Tops Grok**: Users on Discord debated **Gemini 2.5 Pro** versus **Grok**, with one member reporting [Gemini's deep research](https://discord.com/channels/998381918976479270/998382262374973520/1357218656887885874) as superior.
   - While *Grok is good, is worth using while online, but no api access yet is fail*, members reported **OpenAI** is *overrated for coding*.
- **Grok Plagued by Crashes**: Users reported frequent crashes and instability with **Grok**, leading to subscription cancellations and financial losses.
   - One user commented on **Elon Musk's failures**, saying *elon musk buys 200 thousand gpus and yet still fails to deliver* while also stating *elon has never made a decent product*.
- **Manus Exposed as Sonnet Shell**: Members discussed [Manus](https://manus.im/share/oxmc7m9JJq1IRmtpj5mX2A?replay=1), labeling them **scam artists** for being reliant on **Anthropic Sonnet** instead of an open-sourced special model.
   - Users claimed they only thrive with attention, questioning their claims of innovation.
- **Gemini Claims Context Window Crown**: A user inquired about the AI provider with the largest context window and custom GPT features, with [another user answering](https://discord.com/channels/998381918976479270/998382262374973520/1357281796619718767) that **Gemini** offers the largest.
   - They mentioned it provides **1 million tokens** and **Gems (custom GPTs)**, enhancing its appeal for complex tasks.
- **Model Spec Sparks Policy Debate**: A discussion flared up regarding the permissibility of generating images of **adult products**, with some claiming it violated content policies.
   - However, members pointed to OpenAI's [Model Spec](https://model-spec.openai.com/2025-02-12.html), which they claim *contradicts* the policy, suggesting such content might now be permissible if not otherwise harmful.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pulse Perks Power Users**: Users are excited about the **Perplexity Pulse Program**, which gives [Early Access to new features](https://x.com/testingcatalog/status/1897649019309961298?s=46) for feedback, plus free **PPLX** and **merch**.
   - Access to the **Perplexity Pulse Group** is said to provide power users free **PPLX** in exchange for providing feedback.
- **Deep Research Slows Down**: Users report that the updated **"deep research"** feature is [slower and less effective](https://www.reddit.com/r/perplexity_ai/comments/1jq27a6/why_is_perplexitys_updated_deep_research_slower/), with reports of *overfitting with confirmation bias*.
   - One user says it's slower and only gets *20 sources*, using more server resources than older versions.
- **Gemini 2.5 challenges Perplexity O1**: Discord users are saying that [**Gemini 2.5** offers similar quality to **Perplexity's O1 Pro**](https://cdn.discordapp.com/attachments/1047649527299055688/1357423109778702607/image0.jpg?ex=67f02649&is=67eed4c9&hm=d5049580f5523c24bef016f8050e7b92c1f37e1ec416ad9c7ab8b4509c735bf5&) for free, but Perplexity is better for research papers and for solid science.
   - Some users note that Gemini's deep research is *vulnerable to SEO cheating websites* but offers better reasoning with *youtube sources*.
- **API Versioning Vanishes, Vexes Users**: A member complained about the lack of versioning in the **Perplexity API**, saying *That's a breaking change, you don't do that in production when you have customers using your API*.
   - They suggested having **/v1/** in the API URL so that a **/v2/** can be created without breaking the actively used **/v1**.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Github Copilot Flexes OpenRouter Muscles**: **Github Copilot** now allows users to add an [OpenRouter key](https://openrouter.ai/) to select from a wider array of models.
   - This integration expands model access beyond OpenAI's offerings, providing users with more choices.
- **Google Goes Chip Hunting at CoreWeave**: Google is reportedly in talks to rent **Nvidia Blackwell** chips from CoreWeave and potentially house its **TPUs** in their facilities ([The Information Article](https://www.theinformation.com/articles/google-advanced-talks-rent-nvidia-ai-servers-coreweave)).
   - This move may indicate that Google is **TPU poor**, struggling to meet inference demands.
- **Stealthy Quasar Alpha Model Surfaces on OpenRouter**: A new model named **Quasar Alpha** launched on [OpenRouter](https://openrouter.ai/openrouter/quasar-alpha) with **1,000,000 context** and free input/output tokens, and is described as a powerful, all-purpose model supporting long-context tasks and code generation.
   - The community speculates it might be an open-source SSM, or secretly from OpenAI, despite its tendency to output short responses and listicles.
- **Devin 2.0 hits the Markets**: **Cognition Labs** has introduced [Devin 2.0](https://fxtwitter.com/cognition_labs/status/1907836719061451067), a new agent-native IDE experience available for **$20** plus pay as you go.
   - Some members find this launch *too funny* because the competition might find PMF before **Devin** does.
- **Deep Research Finds Bargains**: A user shared that [OpenAI Deep Research](https://x.com/jbohnslav/status/1907759146801197450) helped them discover a plumber who charged **$200** for a repair, drastically less than the original quote of **$2,250**.
   - The user joked that OpenAI Pro *literally saved me $2,050, almost paying for itself for the entire year!*



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro Sparks Rate Limit Frenzy!**: Users are bumping into the **20 requests/minute rate limit** with **Gemini 2.5 Pro** in Aider, suspecting background requests, some seeing **5 RPM** despite having tier 1 API keys as shown in [this screenshot](https://cdn.discordapp.com/attachments/1131200896827654144/1357114156037312683/image.png?ex=67efaf4c&is=67ee5dcc&hm=ab00c0d89a9a4029e1244032c897f52cf418c2b5c10a03543f8574d73b779750&).
   - To manage quota, one user suggested setting `--editor-model sonnet` to offload editing tasks to a cheaper model, and another suggested trying `haiku`.
- **Voice Command Seeks Provider Harmony!**: Users are seeking configuration options to select voice models and providers for the `/voice` command, which currently defaults to **OpenAI Whisper**.
   - A pending PR ([https://github.com/Aider-AI/aider/pull/3131](https://github.com/Aider-AI/aider/pull/3131)) could address this, potentially allowing different providers and models.
- **Aider's Shell Game Baffles Docker Debuggers!**: A user puzzled over **Aider**'s shell behavior when debugging Docker issues, noting that **Aider**'s `curl` commands succeed where their own shell (`bash`) commands fail.
   - This discrepancy has sparked curiosity about which shell **Aider** employs and how it impacts command execution.
- **Openrouter's Errors Plague Gemini's Performance!**: Users reported encountering `litellm.BadRequestError` with **Openrouter**, specifically a `KeyError: 'choices'` and `Internal Server Error` (code 500) when using `openrouter/google/gemini-2.5-pro-exp-03-25:free`.
   - These intermittent errors are causing uncertainty about the root cause and overall reliability.
- **Git Repo Corruption Creates Havoc!**: Multiple users faced 'Unable to list files in git repo: BadObject' errors, igniting concerns about potential **Git repo corruption**.
   - The error message prompts users to check for corruption but lacks immediate solutions.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Brave Integrates LM Studio Locally**: Users are integrating **LM Studio** with the **Brave** browser via `http://localhost:1234/v1/chat/completions`, seeking to configure the **API** to utilize system prompts with resources like [lmstudioservercodeexamples](https://github.com/YorkieDev/lmstudioservercodeexamples).
   - However, many users faced challenges in configuring **Brave** with the correct **API** endpoint.
- **API Key Unlocks System Prompt Potential**: To use **system prompts** with **LM Studio's local server**, users must provide the prompt via the **API call**, rather than the LM Studio interface, referring to the [official documentation](https://lmstudio.ai/docs/app/api).
   - This is a requirement for local **LLM API servers**.
- **CUDA Faces Memory Mayhem**: A *'failed to allocate cuda0 buffer'* error typically indicates insufficient memory for the model, and the missing **mmproj** file when downloading from **HF mirror** can trigger the issue.
   - Users can resolve the issue by downloading from within **LM Studio** with proxy settings enabled.
- **Unsloth 2.0 6b Solves Coding Problems**: A user reported running **Unsloth 2.0 6b** on 4x 3090 + 256GB RAM at ~3 tok/s and stated that it solved a coding problem in 20-30 minutes when smaller models and **ChatGPT** failed.
   - The user said **Qwen QWQ** reaches 90% of **R1** quality at 5% of the parameters, showing a clear preference for quality over speed.
- **M3 Ultra Struggles, M4 Max Excels**: A user stated that the **M3 Ultra Mac Studio** performs poorly for **LLM** use due to unbalanced memory, compute, and bandwidth, while the **M4 Max** and **5090** are excellent.
   - They argued the **M3 Ultra's** large VRAM suits gigantic MoE models but is overpriced for smaller models fitting in a **5090's 32GB VRAM** or a **M4 Max's 96GB**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter API Gets Web Citations**: OpenRouter's [web search](https://x.com/OpenRouterAI/status/1907623560522379436) now returns citations in the API, standardized across models like **OpenAI** and **Perplexity**.
   - Developers can integrate web search by enabling the `web` plugin or appending `:online` to the model slug as detailed in the [documentation](https://openrouter.ai/docs/features/web-search).
- **Quasar Alpha Debuts with 1M Context**: OpenRouter introduced [Quasar Alpha](https://openrouter.ai/openrouter/quasar-alpha), a **free**, **1M token** context length model optimized for coding but with general-purpose capabilities, before its official release.
   - User feedback can be provided in [the dedicated Discord thread](https://discord.com/channels/1091220969173028894/1357398117749756017), with some users suggesting it might be a new **Qwen** variant after initial benchmark comparisons.
- **Character Gateway API Opens Character Creation**: [Character Gateway](https://charactergateway.com/) launched as an **AI character platform** for developers to create, manage, and deploy **AI characters/agents** with *no database, no prompt engineering, no subscription, [and] no new SDK*.
   - The platform allows users to generate characters and images, and send **/chat/completion requests** using their own **OpenRouter** key.
- **Gemini 2.5 Pro Faces Performance Questions**: Users are reporting inconsistent performance with **Gemini 2.5 Pro**, noting free models hosted by Google often have very low rate limits.
   - One member said *they generate the results once and cache the results, so if you ask the same question, they give you back the same reply, even if you change the parameters*.
- **Targon's Speed Tied to Parameter Ignoring?**: Discussion arose questioning if **Targon's speed** is due to miners potentially ignoring sampling parameters, potentially leading to biased distributions.
   - This was brought up in reference to [verifier.py on GitHub](https://github.com/manifold-inc/targon/blob/main/verifier/verifier.py) and the consensus was that there may be an element of caching involved, but there was no concensus.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **vLLM/TGI has Setup Issues on RTX 5000 series**: Members are running into problems setting up **vLLM** or **TGI** with a new **RTX 5000** series card and they need a nightly version of **PyTorch** and **Cuda 12.8** but that's not so easy...
   - One member stated, *when you install something else, PyTorch gets overwritten by the old version*, pointing to these github repos for help: [vllm-project/vllm/issues/14452](https://github.com/vllm-project/vllm/issues/14452), [pytorch/My-rtx5080-gpu-cant-work-with-pytorch/217301](https://discuss.pytorch.org/t/my-rtx5080-gpu-cant-work-with-pytorch/217301), [lllyasviel/stable-diffusion-webui-forge/issues/2601](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2601), [ComfyUI/discussions/6643](https://github.com/comfyanonymous/ComfyUI/discussions/6643).
- **AI Cracks Down on Counterfeit Couture**: Members shared research about counterfeit products and presented a computer-vision-based system using deep neural networks, claiming **99.71% accuracy** after rejections for branded garments, documented in [this paper](https://arxiv.org/abs/2410.05969).
   - The system does not require special security tags or modifications to supply chain tracking, and transfer-trained on a small number of fake and genuine articles.
- **HF Billing Transparency is a Black Box**: Members expressed confusion about Hugging Face's billing and quota systems as well as service usage for **GPU Spaces, Zero GPU Spaces, Serverless Inference API**.
   - They would like HF to provide *reporting, communication, and consultation* about major changes, for example posting *We're going to implement a major change. It'll be unstable for a few days*.
- **Chat Templates are now Trainable**: Members confirmed that it is now possible to pass a **chat_template** to the **transformers TrainingArguments** or **Trainer** to use a custom chat_template for models during inference time and for training.
   - The docs at [huggingface.co](https://huggingface.co/docs/transformers/main/en/chat_template_basics#can-i-use-chat-templates-in-training) explain that chat templates are part of the tokenizer for text-only LLMs or processor for multimodal LLMs to specify how to convert conversations into a single tokenizable string.
- **RAG Implementation is surprisingly Lean**: When a member asked how many lines of code it takes to implement **RAG** techniques for a company, another member responded that it only took a *few lines - 15- 30 more or less*.
   - They stored the information in **MongoDB**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Debugging Tricks Exposed**: Members discovered debugging methods for MCPs, revealing that `sendLoggingMessage` functions if [logging is configured during server initialization](https://example.com/initialization).
   - The inspector's limitations sparked discussions on developing a superior alternative.
- **Open Source EV Assistant Server Surfaces**: An [open-source MCP EV assistant server](https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py) can manage **EV charging stations**, **trip planning**, and **resource management**.
   - This server provides a comprehensive set of tools and APIs for EV-related services.
- **MCP Client Implements Notifications**: An [MCP client implementation](https://github.com/Abiorh001/mcp_omni_connect) now supports all **notifications**, including subscribing and unsubscribing to resources.
   - It offers integration with **OpenAI models** and supports dynamic tool and resource management across multiple servers.
- **FastMCP Has Limitations**: **FastMCP** might lack support for features like `subscribe_resource`, with some considering the **low-level server** for enhanced control.
   - Members traded code and implementation specifics for handling resource subscriptions and updates in the low-level server.
- **Enact Protocol Becomes HTTP for MCP**: The [Enact Protocol](https://github.com/EnactProtocol/specification) was proposed as a way to define tools for MCP, similar to the HTTP protocol.
   - One member described it as *a cool way to do semantic tool calling from within a MCP server*.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Taps Users for UX Testing**: **NotebookLM** is seeking users for **60 min 1:1 remote chats** to provide feedback on new ideas, offering a **$100 gift card** for participation.
   - Participants are required to share a set of notebook sources via Google Drive beforehand and [apply via this form](https://forms.gle/P2t8q36NqbPNSVk8A).
- **Discover Sources Debuts in NotebookLM**: **NotebookLM** introduced a new **Discover Sources** feature, enabling users to find and add relevant web content to their notebooks with one click, along with **Google AI** generated summaries. [Learn more here](https://blog.google/technology/google-labs/notebooklm-discover-sources/).
   - Users have suggested including academic online sources similar to **Perplexity**.
- **Source Transferability Troubles Torment NotebookLM**: Users expressed frustration over the lack of source file transferability between folders in **NotebookLM**, arguing that the read-only nature is limiting.
   - They are requesting that [source files be transferable](https://notebooklm.google) between folders.
- **Gemini Gets a New Guiding Guru**: Josh Woodward will be replacing Sissie Hsaio as the leader of the **Gemini** team in order to prepare for the next evolution of the **Gemini app**, according to [The Verge](https://www.theverge.com/news/642000/google-sissie-hsaio-stepping-down-notebooklm).
   - The transition signals potential shifts in the app's direction and development.
- **Safari Snafus Spoil NotebookLM Sessions**: Some users reported issues accessing **NotebookLM** on **Safari** (iPhone/Mac); if language fixes don't work, adding `?hl=en` to the end of the URL (like this: `https://notebooklm.google.com/?hl=en`) might resolve it.
   - Other users confirmed **NotebookLM** works on iPhone SE (2nd gen) by adding a shortcut to the Home screen.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Ace Computer Autopilot Launches**: [General Agents Co](https://x.com/sherjilozair/status/1907478704223297576) launched **Ace**, a realtime computer autopilot that performs tasks using the mouse and keyboard at superhuman speeds.
   - Unlike a chatbot, Ace is designed to execute tasks directly on a computer, executing tasks directly.
- **YourBench Opens Custom Benchmarking**: [YourBench](https://x.com/sumukx/status/1907495423356403764) launched **YourBench**, an open-source tool for custom benchmarking and synthetic data generation from any documents.
   - YourBench aims to improve model evaluations by providing a custom evaluation set and leaderboard.
- **Llama 4 Generates Images**: **Llama 4** is rolling out image generation and editing capabilities in messages.
   - Users noted that edits were very fast, citing *1 second edits versus 5 minutes for gpt-4o*.
- **Scale AI Soars in Valuation**: **Scale AI** is projected to reach **$2B** in revenue this year, leading to a tender offer valuing the company at **$25B**.
   - Revenue last year was **$870M**.
- **A16Z Assembles AI Workstation**: A16Z built an **8x RTX 4090 GPU AI workstation** from scratch, compatible with the new **RTX 5090** with **PCIe 5.0**, for training, deploying, and running AI models locally.
   - They released a [full guide](https://x.com/Mascobot/status/1907899937838301311) on how to build your own.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Superior UX/UI Steals the Show**: Members highlighted that successful startups often have better **UX/UI**, noting a lack of a *winning sauce* in current products and showcased an agent swarm generating web components in parallel, as seen in [this screen recording](https://cdn.discordapp.com/attachments/986699377257119794/1357190780258746429/Screen_Recording_2025-04-03_at_1.39.26_pm.mov?ex=67eff6a9&is=67eea529&hm=9a8e202a73469a0749a23b81496240fd68a93a295583b0ce34cf52ff80c0c03e&).
   - One user seeks to automate wireframing with a layout generator, designing grayscale wireframes, refining them, and populating them with web components, potentially skipping wireframing/design steps, using a swarm of agents, pointing to [this Dribbble design](https://dribbble.com/shots/25708347-Delivery-Web-App-Design) for inspiration.
- **GPT-4o Gets a Mind of Its Own**: Users observed **GPT-4o** exhibiting unusual behaviors, such as adopting a persona and adding parenthetical comments to its responses, and provided [this screenshot](https://cdn.discordapp.com/attachments/986699377257119794/1357335757676871711/image.png?ex=67efd4ee&is=67ee836e&hm=4deb85a208466f212d88e7b77771776834fe28524ac15dc9c5dbcb1be3301ff3&) as an example.
   - Speculation arose concerning the origin of this behavior, with theories ranging from an *EQ dataset* used in SFT to emergent properties; users also noted that GPT-4o is slowing down.
- **LLMs Flunk Math Olympiad**: A member shared [a paper](https://arxiv.org/abs/2503.21934v1) evaluating state-of-the-art LLMs on the **2025 USA Mathematical Olympiad (USAMO)**, where models like **O3-MINI** and **Claude 3.7** achieved less than **5%** on **six proof-based math problems**.
   - Each problem was scored out of **7 points**, with a max total score of **42**, and the models were trained on all imaginable math data, including **IMO problems**, **USAMO archives**, **textbooks**, and **papers**.
- **Diffusion Model Dream 7B Awakens**: HKU-NLP and Huawei Noah’s Ark Lab released **Dream 7B**, an open diffusion large language model that outperforms existing diffusion language models and matches or exceeds top-tier Autoregressive (AR) language models of similar size, according to [this blogpost](https://hkunlp.github.io/blog/2025/dream/).
   - Dream 7B demonstrates *strong planning ability and inference flexibility that naturally benefits from the diffusion modeling.*



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **OpenAI API Refreshes Stateful Design**: With OpenAI's `/v1/chat/completions` API, the complete conversation history must be resent with each prompt, according to [OpenAI Documentation](https://platform.openai.com/docs/guides/conversation-state?api-mode=chat), incurring costs even for non-evicted input tokens.
   - The upcoming `/v1/responses` API will be stateful, referencing past messages via IDs, contrasting with the stateless `/v1/chat/completions` API, as detailed in the [Responses vs Chat Completions documentation](https://platform.openai.com/docs/guides/responses-vs-chat-completions).
- **AMD's TunableOp Joins PyTorch**: AMD introduced **TunableOp** in [PyTorch](https://pytorch.org/docs/stable/cuda.tunable.html), a prototype feature allowing selection of the fastest operation implementation (e.g., GEMMs) using different libraries or techniques.
   - While NVIDIA pre-tunes everything in **CuBLAS**, AMD's approach aims to optimize performance across diverse hardware configurations, even if it might be less optimized for consumer GPUs but still provides a baseline.
- **ThunderKittens Pounce on Blackwell**: The HazyResearch team launched new **BF16** and **FP8 ThunderKittens GEMM kernels** for the **NVIDIA Blackwell architecture**, achieving speeds near **cuBLAS**.
   - These kernels use features like **5th-generation tensor cores**, **Tensor Memory**, and **CTA pairs**, integrated into TK's tile-based abstractions, as noted in their [blog post](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell).
- **Reasoning Gym Datasets Get Curriculum Boost**: A member submitted a PR ([#407](https://github.com/open-thought/reasoning-gym/pull/407)) to refine the **curricula** of all **datasets** in the [reasoning-gym](https://github.com/open-thought/reasoning-gym) project, improving tests and incorporating missing curricula like **Knight Swap** and **Puzzle2**.
   - Another member is looking into an interface for **easy, medium, hard** difficulties, similar to **RGBench**, for users to manually set the difficulty and shared a link to what is considered a **medium** difficulty setting for each task in the [reasoning-gym](https://github.com/open-thought/reasoning-gym/blob/5b4aa313819a9a6aecd6034b8c6394b6e4251438/eval/yaml/medium/claude-3.5-sonnet.yaml).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Powering Dimensions with Quantities**: Members shared code for defining physical quantities using a `Quantity` struct with `Dimensions`, creating aliases such as `Velocity`, `Acceleration`, and `Newton`.
   - A user linked to their [Kelvin library on GitHub](https://github.com/bgreni/Kelvin/blob/main/kelvin/quantity.mojo#L55-L125), which showcases the process of getting `Dimensions ** power` to function properly.
- **`IntLiteral` strikes again!**: A member confessed to using *cursed* `IntLiteral` tricks to work around dynamic value issues when defining `Quantity`.
   - Other members praised the use of `IntLiteral` for encoding arbitrary information into the type system, while others joked about their *horrendous* approach.
- **Duration Struct proposal for Modular Max**: A member highlighted a pull request to modular/max for a **Duration struct** inspired by `std::chrono::duration` from the C++ stdlib, which is available [on GitHub](https://github.com/modular/max/pull/4022#issuecomment-2694197567).
   - The member is nearing the completion of a specific *wishful thinking* code snippet mentioned in the GitHub issue.
- **Craving for Mojo's Python Interop**: A user inquired about the progress of **Python wrappers for Mojo**, and the ability to call Mojo from CPython.
   - Another user responded that it would be a 🔥 feature to see.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Checkpoints Get HuggingFace Treatment**: Members discussed converting **torchtune checkpoints** to **HF checkpoint format** using the **HuggingFace checkpointer**.
   - The **tune_to_hf function** was specifically recommended for this conversion.
- **Unsloth VRAM shares with vLLM**: In [Unsloth](https://github.com/unslothai/unsloth), they achieved using the same vRAM for **vLLM** and the training procedure, though the mechanism is unclear.
   - A member suggested that the use of `train` as a masking flag in a validation configuration could lead to confusion.
- **Ariel offers GRPO Upstream goodies**: A member offered to contribute changes from their internal **GRPO** upstream, including in-process **vLLM** integration, in-training evals, and more flexible **RL** data handling.
   - Another member noted existing **vLLM** integration in the async version and an almost ready PR for the validation dataset.
- **Torchtune's timeout bug hits Seq Lengths**: A member reported that **Torchtune** hangs and crashes due to a timeout if some microbatches have a **seq length** of **7/14/21/28/35/42/49** and opened [an issue](https://github.com/pytorch/torchtune/issues/2554).
   - The member noted that the non-random seed in the *torchtune dataloader* helped in catching this *AMAZING bug*.
- **Dream 7B proves diffusion dominance**: The University of Hong Kong and Huawei Noah’s Ark Lab released **Dream 7B**, a new open diffusion large language model, as detailed in [this blog post](https://hkunlp.github.io/blog/2025/dream/).
   - Reportedly, **Dream 7B** *outperforms existing diffusion language models by a large margin* and matches or exceeds top-tier Autoregressive language models of similar size on general, math, and coding abilities.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Diagram Tools Duel!**: Members debated diagram creation tools, recommending **Inkscape** for advanced users and **draw.io** for ease of use.
   - One user jokingly said that any alternative to **pure TikZ** is fraudulent.
- **GitHub to Host AI Event in SF**: **GitHub** is co-hosting an **MCP Demo Night** event in San Francisco, focusing on **AI**, incident response, and platform engineering; more details at [lu.ma/9wi116nk](https://lu.ma/9wi116nk).
   - The event includes lightning demos, a **Future of AI Panel**, fireside chats, and networking.
- **OpenThinker2 Models Outperform DeepSeekR1-32B**: Ludwig Schmidt and team released **OpenThoughts-1M** dataset and **OpenThinker2-32B**, **OpenThinker2-7B** models, outperforming **R1-Distilled-32B** using SFT on **Qwen 2.5 32B Instruct**, detailed in their [blog post](https://www.openthoughts.ai/blog/thinkagain).
   - According to [Etash Guha's tweet](https://x.com/etash_guha/status/1907837107793702958), **OpenThinker2-32B** and **OpenThinker2-7B** outperform **DeepSeekR1-32B** with just SFT on open data.
- **Steering Vectors: Reliable or Risky?**: A member shared the paper [Steering Vectors: Reliability and Generalisation](https://arxiv.org/abs/2407.12404), showing that **steering vectors have limitations** both in- and out-of-distribution.
   - The paper highlights that *steerability is highly variable across different inputs* and can be brittle to prompt changes.
- **Dynamic Steering Vector Composition is Hot**: A member shared their work on [steering vector composition](https://aclanthology.org/2024.blackboxnlp-1.34/) using **Dynamic Activation Composition**, showing success with pairs of unrelated properties like language and formality/safety.
   - Their information-theoretic approach modulates steering intensity to maintain high conditioning while minimizing the impact on generation fluency.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Google Mentorship output is debated**: A member questioned the value of **Google Mentorship** programs, arguing that the *output is almost never worth the time/effort*. 
   - Conversely, others contended that companies effectively gain *smart people working full-time for you for 3 months*, making it a worthwhile endeavor.
- **Tinygrad YoloV8 has Android Hiccups**: Users encountered an `OSError: dlopen failed: library "libgcc_s.so.1" not found` while running the **tinygrad** implementation of **YoloV8** on a Samsung Galaxy Tab S9 after running `pip install tinygrad`.
   - George Hotz suggested this is probably a 2 line fix, but adding android to CI to prevent it from happening again, while another suggested `pkg install libgcc`.
- **LeetGPU to support Tinygrad soon**: Members confirmed that [leetgpu.com](https://leetgpu.com) will soon be supporting **tinygrad**.
   - No further details were provided regarding the specifics of the support.
- **Bilinear Interpolation troubles in tinygrad**: A member asked about **bilinear interpolation** support in **tinygrad**, indicating that it was *"not working"* after searching the documentation for **bilinear**.
   - No further details were given.
- **Clarifying Model Overwriting Logic**: A member asked if it was safe to use `state_dict = get_state_dict(net); safe_save(state_dict, "model.safetensors")` after every epoch to save the latest model.
   - Another member clarified that the model would be overwritten unless a different name is provided for each save.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **CodeAct Generalizes ReAct**: **CodeAct** from scratch is a generalization of **ReAct** where instead of doing chain-of-thought, the agent will dynamically write code that uses these functions to solve the task via [this tool](https://t.co/0GPTPo87ma).
   - The intention is to allow dynamic coding as the tool for solving tasks.
- **Rankify Framework Boosts RAG**: The new open-source [Rankify framework](https://github.com/DataScienceUIBK/Rankify) is designed to streamline tasks like **retrieval, reranking, and RAG** (Retrieval-Augmented Generation).
   - It supports 7+ retrieval techniques, 24+ state-of-the-art Reranking models, and multiple RAG methods.
- **Enhance Gemini API Integrations**: A member is drafting a GSoC proposal for *Enhance Gemini API Integrations* with DeepMind, and would like to make **LlamaIndex** a big part of it, seeking feedback on gaps and optimizations.
   - Specifically feedback is requested on any standout gaps in **Gemini** support (like multimodal or function calling) in llama-index-llms-google-genai or vertex that need tackling, and also any **Gemini-related features or optimizations**.
- **MCP Tool Gives Cursor API Smarts**: Members discussed how to give the latest API and docs knowledge to **Cursor** when coding, and an **MCP tool** that does retrieval over the docs was suggested.
   - An *llm.txt* was deemed near useless due to the codebase size.
- **Trace ID Faces Retrieval Challenge**: Members reported issues where the **otel trace_id** cannot be retrieved after a parent workflow calls a child workflow.
   - The team suggested to put the **trace_id** somewhere else where it can be fetched (workflow context, some other global var).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **ChatGPT 4o Conjures MTG Pop Culture Cards**: A member leveraged **ChatGPT 4o's image generator** to produce **Magic the Gathering Cards** featuring pop culture figures and the **NousResearch team**, posting the results in the general channel.
   - The generated cards received *high taste tester approval* but one comment suggested that *sama sucks tho*, the [tweet from Teknium](https://x.com/Teknium1/status/1907492873991499998) shows several MTG-style cards created by the image generator.
- **Runway Gen 4 Revs Up A.I. Filmmaking**: With **Runway's Gen 4** release, A.I. Prompt Filmmaking takes a leap forward, covered in a [video](https://www.youtube.com/watch?v=Rcwfj18d8n8) about happenings in the world of **OpenAI, Google, and AGI**.
   - The video highlights the unreal progress in **AI Video** and mentions that **Alibaba Wan 2.2**, an open source alternative, will soon be released.
- **Genstruct-7B Generates Data Extraction Instructions**: In response to a query about using **LLMs for extraction** to create datasets from unstructured PDFs, a member linked to [Genstruct-7B](https://huggingface.co/NousResearch/Genstruct-7B) as a viable starting point.
   - **Genstruct-7B**, inspired by **Ada-Instruct**, is designed to generate valid instructions given a raw text corpus and can be quickly used with ollama with a [github repo](https://github.com/edmundman/OllamaGenstruct).
- **OpenAPI Access Opens for LLMs, Reduces Clutter**: A member announced the release of their **v1 OpenAPI access** to **SaaS/PaaS/IaaS** for **LLMs**, intending to cut down on **MCP clutter**, linking to an [HN discussion](https://news.ycombinator.com/item?id=43562442).
   - The new **OpenAPI access** aims to resolve the problem of **MCP (Multi-Cloud Platform) clutter** when integrating **LLMs** with different cloud services.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Experiences Degradation**: Some users experienced **http timeout errors** and confirmed the [Cohere Status Page](https://status.cohere.com/) indicated *Degraded Performance - Increased Latency* for **Command-a-03-2025/command-r-plus-08-2024** models.
   - The incident was being monitored and lasted for **4 hours**.
- **Python Logging Debate**: A member building a Python package for PDF processing is in disagreement with a senior teammate over whether to use **logs** or **print statements**.
   - The member prefers logs for their **different levels, file saving, searchability, and issue reporting**, while the teammate prefers **print statements** to avoid burdening users; a compromise of a **disabled logger instance by default** was suggested.
- **RAG Doc Chunking Strategy**: A member asked about using a **18000 token document** for **RAG** and whether to cut it up.
   - An expert recommends chopping the documents, but it depends on the end goal and requirements; also suggesting that **Command-a's 256k context window**, and **command-r and r-plus's 128k context window** should easily be able to handle it.
- **Brainstorming AI Safety Tests**: An AI safety testing platform called **Brainstorm** is releasing its MVP in a few weeks, aiming to ensure AI changes the world for the better and you can find out more at the [Brainstorm landing page](https://brainstormai.framer.website/).
   - The creator of **Brainstorm** is seeking insights on current methods used to test AI for safety and performance issues, particularly around **bias**, **prompt injections**, or **harmful outputs**.
- **KAIST LLM Fairness Research**: A M.S. student from **KAIST** (South Korea) introduced themself with a research focus on **bias/fairness** and **interpretability** in **LLMs/VLMs**.
   - They are actively seeking research collaboration opportunities in these specific areas and bring experience from **KAIST**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic Embed V2 Integration Anticipation Grows**: Members eagerly await the arrival of **Nomic Embed Text V2** in **GPT4All**, with one member acknowledging the developers' busy schedules.
   - The member expressed patience, understanding that the integration process may require time and resources.
- **Contact Sales Advised for Vulnerability Disclosure**: A member inquired about the correct procedure for responsibly disclosing a vulnerability within **GPT4All**.
   - Another member suggested utilizing the [contact support email](https://atlas.nomic.ai/contact-sales) available on the **Nomic AI** website for such disclosures.
- **GPT4All-J Model in GGUF Format Proves Elusive**: A member sought a download link for the **GPT4All-J model** in **Q4_0 quantization** and **GGUF format** for integration into a project.
   - A second member responded that **GPT4All-Falcon** is available as **GGUF**, but noted that **GPT4All-J** is not possible.
- **Chocolatine-2-14B Claims Book Query Crown**: A member declared the "**Chocolatine-2-14B**" model as the ideal choice for querying embedded books.
   - Additional details about the specific capabilities or architecture of the **Chocolatine-2-14B** model were not provided.
- **Chats Call for Chronological Correction**: A member suggested that chats should reorganize based on the time they were altered rather than when they were created, to improve context.
   - The member criticized the current chronological listing by creation date as *arbitrary* and less helpful for tracking ongoing conversations.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Telemetry Loops LLM Agent Self-Improvement**: A member shared a video *Close the loop on LLM agent development by configuring them to improve themselves using telemetry and evaluations* [on YouTube](https://youtu.be/jgzSq5YGK_Q).
   - The discussion emphasized using **telemetry** and **evaluations** to improve **LLM agent** self-improvement.
- **DSPy Decouples Prompt Engineering**: A member asked how **DSPy** decouples the *tinkering layer* of **prompt engineering** from **LLM** behavior and its synergy with **OpenAI Agents SDK**.
   - Another member confirmed **DSPy** offers *programmatic pieces*: **signatures and modules** for this decoupling.
- **DSPy's Programmatic Pieces Unveiled**: A member explained **DSPy's** core abstractions: **signatures and modules**, which help decouple **prompt engineering** from **LLM** functional behavior.
   - This allows programming instead of just prompt engineering, aiding integration with tools like **OpenAI Agents SDK**.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Phi-4-mini-instruct Joins the BFCL Arena**: A member submitted a [PR](https://github.com/ShishirPatil/gorilla/pull/967) to add tool evaluation for **Phi-4-mini-instruct** with **BFCL**.
   - The member has attached the **evaluation score** within the PR, requesting feedback and review from the community.
- **Call for Code Review on Tool Evaluation**: A member is actively seeking reviewers for their PR focused on tool evaluation.
   - Another member responded, indicating they will promptly review the **PR**.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek-V3 Gets a Facelift**: **DeepSeek-V3** has been upgraded to **DeepSeek-V3-0324**, supposedly performing slightly better than before in evaluations.
   - A member posted a [link](https://x.com/windsurf_ai/status/1907902846735102017) to the **Windsurf AI** twitter account announcing the upgrade and its continued free availability.
- **Windsurf Solicits Bookmarks**: Windsurf is trying to increase the visibility of their announcements.
   - A member asked users to bookmark the announcement post on X, to keep abreast of upgrades and new releases.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/)** (1 messages): 

liewxinyen: awesome case from <@356472623456059392> <:1741316509962:1348823230454038670> 🤩
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1357067222387789885)** (807 messages🔥🔥🔥): 

> `Brazilian Lawyer using AI, ReferrerNation's BPO job matchmaking platform, Learning code with AI assistance, Claude for report writing, AI competition from China` 


- ****Brazilian Lawyer enters AI Realm****: A Brazilian lawyer is starting to use **AI tools** in their legal practice, despite being a self-described "boomer" (39 years old) and new to Discord.
   - The lawyer, who began coding in Delphi in 2002, expressed concerns about the advances in AI and is now exploring Manus to stay relevant in their field.
- ****ReferrerNation Enters the Arena****: Mark, CEO of [ReferrerNation.com](https://www.referrernation.com/) *a global BPO job-matching platform*, aims to integrate **AI** to improve recruitment and automation, while soon integrating **crypto-based incentives**.
   - After some community members expressed that his initial posts seemed too spammy and overtly crypto-focused, Mark apologized and noted, *I’ll take the time to understand the vibe here better before posting more.*
- ****Learn to Code with Gemini and Claude****: Members recommended using **Gemini 2.5** or **Claude** for learning to code, as they are excellent **AI coding models** that can assist with understanding and project work.
   - A police chief reportedly uses **Claude** to write standardized reports during night shifts.
- ****Manus Credit Crunch Sparks Tips and Tricks****: Many users reported **rapid credit depletion**, with free credits vanishing quickly, leading to discussions on optimizing prompts and efficient usage.
   - Members advised newcomers to read the <#1355477259234054323> tips section to avoid wasting credits, noting that the team is working on reducing credit usage rates. Members shared that using other tools in conjunction with manus, for example [R1](https://www.perplexity.ai/) is very effective for efficient credit spend.
- ****Leverage External Code to Save the Day****: A member mentioned difficulty downloading files from Manus due to lack of credits and so community suggested using third party apps such as Claude to extract code and files.
   - It was also mentioned that the best practice is to download all files from Manus, give it to something else. then, ask that other AI, for example Gemini *provide me files for this website* then I go to Manus and say *add these files to this website*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/ban-hammer-futurama-scruffy-gif-20750885">Ban Hammer GIF - Ban Hammer Futurama - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/hello-chat-hello-hello-chat-back-from-the-gif-16804150723034691763">Hello Chat Hello Chat Back From The GIF - Hello chat Hello Hello chat back from the - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/baby-face-palm-really-sigh-stupid-gif-12738431">Baby Face Palm GIF - Baby Face Palm Really - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/welcome-michael-scott-dunder-mifflin-the-office-welcome-aboard-gif-27005393">Welcome Michael Scott GIF - Welcome Michael Scott Dunder Mifflin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://vnxjunpk.manus.space/">Understanding the 2025 Tariff Landscape</a>: no description found</li><li><a href="https://x.com/Lasgidiconf/status/1907805373710360857?t=eTJ-1SBWbz8w64q3SsKzVA&s=19">Tweet from Chuka Konrad (@Lasgidiconf)</a>: Last night I built a visuals rich interactive webpage highlighting key aspects and analysis of the reciprocal tariffs announcement by Trump. Built it entirely on @ManusAI_HQ check it out at https://vn...</li><li><a href="https://bfuarkjn.manus.space/">ManusAI - Comprehensive Guide</a>: no description found</li><li><a href="https://tenor.com/view/in-the-house-martin-martin-lawrence-biggie-hello-gif-12010068014708218113">In The House Martin GIF - In The House Martin Martin Lawrence - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/coffee-caffeine-coffee-time-wake-up-morning-coffee-gif-7886258858573853472">Coffee Caffeine GIF - Coffee Caffeine Coffee time - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/hype-train-activated-bowser-hype-train-gif-14185403">Hype Train Activated Bowser GIF - Hype Train Activated Bowser Hype Train - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/salamik-cute-awesome-collection-poznas-gif-10073975131879134759">Salamik Cute GIF - Salamik Cute Awesome - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/silver-lining-impressed-robert-de-niro-gif-14541556">Silver Lining Impressed GIF - Silver Lining Impressed Robert De Niro - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/stay-tuned-robertidk-keep-watching-theres-more-more-to-come-gif-19154825">Stay Tuned Robertidk GIF - Stay Tuned Robertidk Keep Watching - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/good-morning-gif-10250909101792021306">Good Morning GIF - Good morning - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ucebdqhq.manus.space/">Iterative Development with Manus AI: A Comprehensive Guide</a>: no description found</li><li><a href="https://tenor.com/view/bait-thats-bait-tom-hardy-mad-max-gif-5055384">Bait Thats Bait GIF - Bait Thats Bait Tom Hardy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/awkward-nodding-dan-levy-david-david-rose-schitts-creek-gif-20776317">Awkward Nodding Dan Levy GIF - Awkward Nodding Dan Levy David - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/fever-sick-flu-parks-and-rec-everything-hurts-gif-5394213">Fever GIF - Fever Sick Flu - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/night-sleep-gif-27085775">Night Sleep GIF - Night Sleep - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/get-your-pooh-on-gif-16407146543386304197">Get Your Pooh On GIF - Get your pooh on - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://manus.im/help/credits">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://tenor.com/view/inthehouse-martin-martinlawernce-biggie-hello-gif-13128531067958866971">Inthehouse Martin GIF - Inthehouse Martin Martinlawernce - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://manus.im">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://tenor.com/view/superman-gif-14881123907931593412">Superman GIF - Superman - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1357069747610452049)** (1010 messages🔥🔥🔥): 

> `Meta vision model Cotton, Qwen2.5-vl-32b-instruct OCR, Google Gemini Models, Nightwhisper model on webdev, Gemini 2.6 Pro experimental` 


- **Qwen2.5-vl-32b-instruct gives Google Gemini models a run for their money in OCR**: For OCR of low-quality text in Japanese, `qwen2.5-vl-32b-instruct` is almost on par with **Google Gemini models**.
   - However, the Meta vision model, `cotton`, felt more like the other recent **text-only anonymous models from Meta**.
- **Nightwhisper debuts on WebDev**: The **Nightwhisper** model is available only on [webdev arena](https://webdev.lmarena.ai/) which may be a coding model only.
   - Members suspect that **Nightwhisper = Gemini 2.6 Pro experimental** and does not appear in normal arena.
- **Nightwhisper excels at webpage generation, struggles with existing code**: **Nightwhisper** is proving exceptional at crafting functional apps with appealing UIs, and can work with a temporary URL to generate projects on the [webdev arena](https://webdev.lmarena.ai/)).
   - However, some users have reported that Nightwhisper exhibits difficulties when editing existing code or adhering to specific formatting requests, leading to frustration.
- **WebDev arena introduces model cloning**: WebDev arena has issues with rendering, but one user discovered a method of model cloning, in which the model gives the same screen twice.
   - After receiving an error message from NightWhisper and a repeated error after giving the code again, its name is not shown, indicating model cloning.
- **Gemini 2.5 Pro vs Nightwhisper - the battle continues**: **Gemini 2.5 Pro** achieved a **24.4%** on the [USAMO 2025](https://matharena.ai/), but some models tend to stop mid-sentence or give partial responses.
   - One user found Gemini better at creating a Pokemon simulator, while Nightwhisper produced a cleaner UI but with weird attack power values that were too high.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.testingcatalog.com/google-plans-new-gemini-model-launch-ahead-of-cloud-next-event/">Google plans new Gemini model launch ahead of Cloud Next</a>: Discover the latest updates on Gemini, including potential new model launches and experimental tools. Stay tuned for exciting features like scheduled prompts and video generation.</li><li><a href="https://snipboard.io/86wfLe.jpg">Upload and share screenshots and images - print screen online | Snipboard.io</a>: Easy and free screenshot and image sharing - upload images online with print screen and paste, or drag and drop.</li><li><a href="https://x.com/a7m7s1p6dv20/status/1907684868164825260?s=46">Tweet from ᅟ (@a7m7s1p6dv20)</a>: (initial?) pricing scheme for gemini 2.5 provia glama AI</li><li><a href="https://x.com/testingcatalog/status/1907891942869922292?t=Q30isS2oxgO7U-qBjdYMtA&s=19">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: Google is preparing to launch another model on Gemini, potentially next week, ahead of the Cloud Next event.Quoting ʟᴇɢɪᴛ (@legit_api) nightwhisper and stargazer are 2 new models added to...</li><li><a href="https://create.roblox.com/docs/luau">Tweet from Luau | Documentation - Roblox Creator Hub</a>: Luau is the scripting language creators use in Roblox Studio.</li><li><a href="https://gist.github.com/riide">riide’s gists</a>: GitHub Gist: star and fork riide&#39;s gists by creating an account on GitHub.</li><li><a href="https://matharena.ai/">MathArena.ai</a>: MathArena: Evaluating LLMs on Uncontaminated Math Competitions</li><li><a href="https://gist.github.com/riidefi/443dc5c4b5e13e51846a43067b5335a1">Meta (?)&#39;s `24_karat_gold` (lmarena) System Prompt</a>: Meta (?)&#39;s `24_karat_gold` (lmarena) System Prompt - prompt.txt</li><li><a href="https://devforum.roblox.com/t/expanding-assistant-to-modify-place-content-beta/3107464">Tweet from Expanding Assistant to Modify Place Content [Beta]</a>: Hey Creators,  Today, we are excited to announce we’re expanding Assistant’s capabilities to perform a broad range of actions in Studio. Assistant can now help you modify the DataModel in order to aut...</li><li><a href="https://g.co/gemini/share/60fcf5c244c9">‎Gemini - Three.js Asteroid Impact Simulation Code
</a>: Created with Gemini Advanced</li><li><a href="https://en.wikipedia.org/wiki/Leela_Chess_Zero#Spinoffs>)">Leela Chess Zero - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1357128440758927461)** (1 messages): 

> `Mobile Alpha UI, LM Arena Access, Alpha Feedback` 


- **Arena Alpha now on Mobile!**: The new **Arena Alpha UI** is now optimized for mobile, allowing users to test it on their phones via [alpha.lmarena.ai](https://alpha.lmarena.ai).
- **Accessing Mobile Arena Alpha**: To access the mobile **Arena Alpha**, users need to use the password `still-alpha`.
- **Giving Feedback to Mobile Arena Alpha**: Feedback on the mobile **Arena Alpha** can be provided through a [Google Forms link](https://forms.gle/8cngRN1Jw4AmCHDn7) while bugs can be reported via an [Airtable form](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forms.gle/8cngRN1Jw4AmCHDn7">Arena - New UI Feedback</a>: Tell us what you think about the new design!</li><li><a href="https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form">Airtable | Everyone&#x27;s app platform</a>: Airtable is a low-code platform for building collaborative apps. Customize your workflow, collaborate, and achieve ambitious outcomes. Get started for free.
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1357073324324290590)** (772 messages🔥🔥🔥): 

> `Restoring to previous checkpoints, Roo code, Boomerang Mode, Gemini Pro EXP vs Pro MAX, Windsurf vs Cursor tab` 


- **Branch Bugs Baffle Backtracking**: Members reported having issues with **restoring to previous checkpoints** in Cursor, encountering bugs from later states even in supposedly clean branches and members suggested doing a `git diff branch1,branch2` to identify the differences.
   - One member experienced a CSS overhaul from a simple logo change prompt, highlighting the potential for unexpected modifications.
- **Roo Code workflow is catching fire**: One user described their sweet workflow on **Roo Code**, highlighting its cost-effectiveness at around **$0.4 per day**, achieved through selective model usage.
   - They also mentioned its superior capabilities compared to Cursor for specific tasks, as well as [the associated docs](https://docs.roocode.com/features/boomerang-tasks/).
- **Boomerang Mode Gains Traction**: Members discussed the benefits of **Boomerang Mode** in Roo Code, where tasks are divided into subtasks handled by separate agents, enabling more efficient problem-solving than relying on Gemini 2.5 for everything.
   - Boomerang mode is very customisable and one user shares *I will weld it down, don't worry*.
- **Peeking at PearAI Pricing**: Users compared the pricing models of Cursor and **PearAI**, with PearAI offering some free credits to test Roo Code.
   - However, it was clarified that PearAI's **$15/month plan** includes a credit limit, after which usage-based charges apply, contrasting with claims of unlimited model access and one member even accused Cursor of "*scamming people!*"
- **Nightly Builds Nurture New Navigational Notions**: Cursor **0.49.1** is available as a nightly build with this flag set on your account `account settings, advanced -> developer settings`.
   - The feature is supposedly a context window indicator for agent use, as well as a Windsurf API key. 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.roocode.com/features/boomerang-tasks/">Boomerang Tasks: Orchestrate Complex Workflows | Roo Code Docs</a>: Boomerang Tasks (also known as subtasks or task orchestration) allow you to break down complex projects into smaller, manageable pieces. Think of it like delegating parts of your work to specialized a...</li><li><a href="https://x.com/sidahuj/status/1899460492999184534">Tweet from siddharth ahuja (@sidahuj)</a>: 🧩 Built an MCP that lets Claude talk directly to Blender. It helps you create beautiful 3D scenes using just prompts!Here’s a demo of me creating a “low-poly dragon guarding treasure” scene in just a...</li><li><a href="https://ubisoft-mixer.readthedocs.io/en/latest/index.html">Mixer: a Blender Addon for Collaborative Editing &mdash; Ubisoft Mixer  documentation</a>: no description found</li><li><a href="https://x.com/MervinPraison/status/1907165153537224953">Tweet from Mervin Praison (@MervinPraison)</a>: Introducing @Ollama MCP AI Agents! 🎉🔒 100% local💻 Just 3 lines of Code🗺️ 1000+ MCP server Integration✨ @PraisonAI v2.1 is here with 1000+ MCP Support ! @AtomSilverman @Saboo_Shubham_ @elonmusk</li><li><a href="https://x.com/ehuanglu/status/1901861073902301194?s=46&t=kUuVqsG2GMX14zvB592G5w">Tweet from el.cine (@EHuanglu)</a>: 3D AI is getting crazierthis new Hunyuan3D 2.0 MV open source model can generate 3D asset with multiple images in seconds free to use now, link in comments10 examples:</li><li><a href="https://tenor.com/view/parallel-universe-operation-boomerang-gif-15332015">Parallel Universe GIF - Parallel Universe Operation Boomerang - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://trypear.ai/privacy">Privacy Policy</a>: The privacy policy for PearAI.</li><li><a href="https://github.com/vbwyrde/DSPY_VBWyrde/blob/main/DSPY12_Out_3.md">DSPY_VBWyrde/DSPY12_Out_3.md at main · vbwyrde/DSPY_VBWyrde</a>: DSPY Experiments. Contribute to vbwyrde/DSPY_VBWyrde development by creating an account on GitHub.</li><li><a href="https://github.com/lharries/whatsapp-mcp">GitHub - lharries/whatsapp-mcp: WhatsApp MCP server</a>: WhatsApp MCP server. Contribute to lharries/whatsapp-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/supercorp-ai/supergateway">GitHub - supercorp-ai/supergateway: Run MCP stdio servers over SSE and SSE over stdio. AI gateway.</a>: Run MCP stdio servers over SSE and SSE over stdio. AI gateway. - supercorp-ai/supergateway</li><li><a href="https://github.com/ahujasid/blender-mcp">GitHub - ahujasid/blender-mcp</a>: Contribute to ahujasid/blender-mcp development by creating an account on GitHub.</li><li><a href="https://www.cursor.com/changelog">Changelog | Cursor - The AI Code Editor</a>: New updates and improvements.</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers">GitHub - punkpeye/awesome-mcp-servers: A collection of MCP servers.</a>: A collection of MCP servers. Contribute to punkpeye/awesome-mcp-servers development by creating an account on GitHub.</li><li><a href="https://www.findsimilarstartups.com/shared/67edd25421be61a714cc807e">Find Similar Startups - AI-Powered Competitor Analysis</a>: Discover startup competition instantly with AI-powered market research.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1357085846167093411)** (217 messages🔥🔥): 

> `ECC Errors in EC2 Instances, Gemma 3 Bug with Custom Dataset, Unsloth Apple Silicon Support, Fine-tuning LLaSA with Unsloth, RTX 5090 vs RTX 4090 for Unsloth` 


- **ECC Errors Thrown by EC2 Instance on Unsloth**: A user reported receiving **CUDA ECC errors** and a **500 error** while processing prompts in series on an `g6e.4xlarge` **EC2 instance** and linked to [Issue #2270](https://github.com/unslothai/unsloth/issues/2270) on GitHub.
   - The error was an *uncorrectable ECC error encountered*, suggesting a hardware or memory issue.
- **Dataset Snafu triggers Gemma 3 Bug**: A user sought assistance with a bug encountered when training **Gemma 3** using a custom dataset from [Hugging Face](https://huggingface.co/datasets/adamtc/sdtg_sgpt), with details provided in [Issue #2270](https://github.com/unslothai/unsloth/issues/2270).
- **Apple Silicon Support Coming to Unsloth?**: A user requested testing for an Apple device-related pull request, aiming to compare its performance against basic MLX, which can be found at [PR #1289](https://github.com/unslothai/unsloth/pull/1289).
- **Llasa LoRA Looms on Unsloth Horizon**: Members are pondering leveraging Unsloth for LoRA training on **Llasa**, a text-to-speech (TTS) system, with guidance on a related pull request available at [PR #2263](https://github.com/unslothai/unsloth/pull/2263).
- **RTX 5090 Smokes RTX 4090?**: A user shared a comparison of sample speeds between **RTX 5090** and **RTX 4090** using an unsupported Unsloth version, highlighting the potential performance gains with newer hardware.
   - Another member stated it was *not worth* for the money, but others argued it could be **ROI positive** if you are limited by VRAM or need training done faster.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found</li><li><a href="https://huggingface.co/datasets/MrDragonFox/Elise">MrDragonFox/Elise · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/xyehya/documentation">GitHub - xyehya/documentation: Odoo documentation sources</a>: Odoo documentation sources. Contribute to xyehya/documentation development by creating an account on GitHub.</li><li><a href="https://huggingface.co/collections/google/gemma-3-qat-67ee61ccacbf2be4195c265b">Gemma 3 QAT - a google Collection</a>: no description found</li><li><a href="https://unsloth.ai/blog/llama3-3">Fine-tune Llama 3.3 with Unsloth</a>: Fine-tune Meta&#x27;s Llama 3.3 (70B) model which has better performance than GPT 4o, open-source 2x faster via Unsloth! Beginner friendly.Now with Apple&#x27;s Cut Cross Entropy algorithm.</li><li><a href="https://github.com/xyehya/documentation/blob/9.0/Unsloth-GRPO.ipynb">documentation/Unsloth-GRPO.ipynb at 9.0 · xyehya/documentation</a>: Odoo documentation sources. Contribute to xyehya/documentation development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/2273.">unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/fimbulvntr/deepseek-r1-traces-no-cjk">fimbulvntr/deepseek-r1-traces-no-cjk · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">Added Support for Apple Silicon by shashikanth-a · Pull Request #1289 · unslothai/unsloth</a>: #4UnoptimizedNo gguf support yet.Build Triton and bitsandbytes from sourcecmake -DCOMPUTE_BACKEND=mps -S . for bitsandbytes buildingpip install unsloth-zoo==2024.11.4pip install xformers==0....</li><li><a href="https://huggingface.co/HKUSTAudio/Llasa-3B/">HKUSTAudio/Llasa-3B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/nicolagheza/badalisc-s1K-1.1-ita">nicolagheza/badalisc-s1K-1.1-ita · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/pull/2263">feat: Support custom `auto_model` for wider model compatibility (Whisper, Bert,etc) &amp; `attn_implementation` support by Etherll · Pull Request #2263 · unslothai/unsloth</a>: feat: Support custom auto_model, Whisper params, and attn_implementationThis PR enhances FastModel.from_pretrained to support a broader range of models:Custom auto_model: Allows specifying the ex...</li><li><a href="https://github.com/unslothai/unsloth/issues/2270">[BUG] Unable to create tensor when training Gemma 3 in Collab using custom dataset · Issue #2270 · unslothai/unsloth</a>: Describe the bug When training Gemma 3 in Collab using (my scam dataset)[https://huggingface.co/datasets/adamtc/sdtg_sgpt] (replacing &quot;mlabonne/FineTome-100k&quot; to &quot;adamtc/sdtg_sgpt&quot;...</li><li><a href="https://huggingface.co/datasets/adamtc/sdtg_sgpt">adamtc/sdtg_sgpt · Datasets at Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/r1-reasoning">Train your own R1 reasoning model locally (GRPO)</a>: You can now reproduce your own DeepSeek-R1 reasoning model with Unsloth 100% locally. Using GRPO.Open-source, free and beginner friendly.</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)!">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/)">Unsloth Documentation</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1357109025161871360)** (5 messages): 

> `Job transition, Product Manager, Vibe Coder` 


- **Job Transition Vibe Coder**: A member joked about their last job, stating the "vibe coder" was actually the product manager.
   - They signed off with *Sincerely, What have we done*, implying a sense of humorous resignation.
- **Product Manager as Vibe Coder**: The discussion highlights the role of a product manager being likened to a "vibe coder," suggesting a focus on team morale and direction.
   - This comparison underscores the importance of product managers in setting the tone and influencing the team's overall energy and productivity.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1357080161790590996)** (236 messages🔥🔥): 

> `Unsloth batch size, SFTTrainer Usage, Gemma3 finetuning, Qwen2.5 Image Size, GRPO and CPU Bottleneck` 


- **Unsloth Auto Batch Size**: Users discussed Unsloth automatically setting batch size to 2 when using multiple GPUs, and a solution was found by setting `CUDA_VISIBLE_DEVICES=0`.
   - One user pointed out that Unsloth uses `per_device_train_batch_size` and linked to the [relevant code in `training_utils.py`](https://github.com/unslothai/unsloth-zoo/blob/4a66f8b08952fc148f5c74cd15aec52cb0113e2d/unsloth_zoo/training_utils.py#L206).
- **Troubleshooting with SFTTrainer**: A user encountered a `ValueError` when using the normal `Trainer` with Llama 3.2 1B instruct, even with FP16 turned off, but it was resolved by switching to `SFTTrainer`.
   - It was hypothesized that the model might be bfloat16 and Unsloth couldn't get the dtype from `Trainer`, suggesting checking the model config json.
- **Gemma3 Vision with Unsloth**: A user successfully started Gemma3 finetuning with vision samples using Unsloth, noting an issue with `UnslothVisionDataCollator` only accepting samples with an image.
   - They were curious if a custom collator could be used with Gemma3/Unsloth, rather than `UnslothVisionDataCollator`, and asked about potential gotchas or examples.
- **Qwen2.5 Image Size Woes**: Users discussed how to set the image size explicitly before feeding it into Qwen2.5-VL-7B-Instruct and it was found that resizing the images manually can lead to errors, as the model resizes internally.
   - One user confirmed that resizing images to **364x364** works, while another suggested trying **224x224**.
- **GRPO Training Bottleneck on CPU**: A user observed a CPU bottleneck (one core at 100% usage) while running the Gemma3_(1B)-GRPO.ipynb code, limiting training speed despite low GPU utilization (25%).
   - Suggestions included profiling the code to identify CPU-intensive operations and considering that the process might be memory bound rather than compute bound due to the small model size.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1mn_hj_sNvW59JxW0u2nuGoB7qLEs30RO?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=vITh0KVJ10qX">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/2270">[BUG] Unable to create tensor when training Gemma 3 in Collab using custom dataset · Issue #2270 · unslothai/unsloth</a>: Describe the bug When training Gemma 3 in Collab using (my scam dataset)[https://huggingface.co/datasets/adamtc/sdtg_sgpt] (replacing &quot;mlabonne/FineTome-100k&quot; to &quot;adamtc/sdtg_sgpt&quot;...</li><li><a href="https://github.com/unslothai/unsloth/issues/1624#issuecomment-2774130919,">GRPOTrainer crashes with unsloth · Issue #1624 · unslothai/unsloth</a>: I am trying to run GRPOTrainer with unsloth but it crashes. How to fix this? unsloth 2025.2.4 unsloth 2025.2.3 transformers 4.47.1 torch 2.5.1 trl 0.14.0 This is the relevant code: model, tokenizer...</li><li><a href="https://github.com/huggingface/transformers.git">GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/unslothai/unsloth/issues/2052#issuecomment-2761332050">AttributeError: &#39;HybridCache&#39; object has no attribute &#39;float&#39;—Gemma 3 training fails with BF16 precision on RTX3090 (Ampere) GPUs · Issue #2052 · unslothai/unsloth</a>: I have an NVidia RTX3090 GPU on Linux. It appears that Unsloth has issues with BF16 (and FP16) training on the recently released Gemma-3 and this seems a problem related with the GPU. A fresh Conda...</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/4a66f8b08952fc148f5c74cd15aec52cb0113e2d/unsloth_zoo/training_utils.py#L206">unsloth-zoo/unsloth_zoo/training_utils.py at 4a66f8b08952fc148f5c74cd15aec52cb0113e2d · unslothai/unsloth-zoo</a>: Utils for Unsloth. Contribute to unslothai/unsloth-zoo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1357368647643041912)** (2 messages): 

> `GRPO Trainer Implementation, Unsloth Techniques, Collab Notebook, DeepSpeed alternative` 


- **Unsloth powers GRPO trainer**: A member shared their Collab notebook detailing the implementation of **Unsloth techniques** for creating a **GRPO trainer**, which was previously only achievable using **DeepSpeed** on multiple **Hx00s**.
   - They shared a [link](https://github.com/xyehya/documentation/blob/9.0/Unsloth-GRPO.ipynb) and encouraged users to use and reference it, welcoming comments and feedback, noting it as *promising*.
- **Social media promotion criticized**: A member questioned the necessity of posting content on social media such as Twitter, rather than sharing direct links on platforms like **Hugging Face**.
   - They questioned *Why do you always have to post stuff on Twitter? Why can't you just post a link to where it is on hugging face?*



**Link mentioned**: <a href="https://github.com/xyehya/documentation/blob/9.0/Unsloth-GRPO.ipynb">documentation/Unsloth-GRPO.ipynb at 9.0 · xyehya/documentation</a>: Odoo documentation sources. Contribute to xyehya/documentation development by creating an account on GitHub.

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1357157940242682061)** (6 messages): 

> `GRPO/PPO, Continue Pretraining Llama, Bespoke Labs new models` 


- **GRPO/PPO methods are not perfect**: A member noted that **GRPO**, **PPO** and other reinforcement learning methods have problems.
   - These methods are still actively used despite their challenges.
- **Continue pretraining Llama on Unsloth**: A member asked about using **Unsloth** to fine-tune **Llama** with many `.txt` files of stories.
   - A suggestion was made to continue pretraining if hardware resources are sufficient and data preprocessing is fine, otherwise, using different fine-tune stages would be better.
- **Bespoke Labs ships new models**: **Bespoke Labs** released new models today, according to a member.
   - A screenshot of the models was shared from **Chrome** browser.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1357080059822735461)** (103 messages🔥🔥): 

> `Gemini vs Grok, Manus deceptive?, AI coding, OpenAI value for money` 


- **Gemini 2.5 Pro vs Grok**: Members extensively debated the merits of **Gemini 2.5 Pro** versus **Grok**, with [one user reporting](https://discord.com/channels/998381918976479270/998382262374973520/1357218656887885874) that **Gemini's deep research is the best**.
   - However, they noted that *Grok is good, is worth using while online, but no api access yet is fail* but **OpenAI** is *overrated for coding*.
- **Musk's Grok Failures**: Multiple users reported frequent crashes and instability with **Grok**, leading to cancellation of subscriptions and loss of money.
   - One user expressed a lack of surprise at **Elon Musk's failures**, stating *elon musk buys 200 thousand gpus and yet still fails to deliver* while also pointing out that *elon has never made a decent product*.
- **Manus is deceptive?**: Members discussed [Manus](https://manus.im/share/oxmc7m9JJq1IRmtpj5mX2A?replay=1), with one user describing them as **scam artists** after they were found to be reliant on **Anthropic Sonnet** rather than using a special model that they would open source.
   - They stated they only thrive with attention.
- **Gemini offers the largest context window**: A user asked which AI provider offers the largest context window along with support for custom GPT features and [another user replied](https://discord.com/channels/998381918976479270/998382262374973520/1357281796619718767) **Gemini**.
   - They stated it offers **1 million tokens** and **Gems (custom GPTs)**.
- **OpenAI 20 euro subscription?**: Users are debating the value of the **OpenAI 20 euro subscription** and several users reporting the service as unusable.
   - One user argued that the **20 euro sub** is worth it and points to **Sora billing FAQ** to argue his point [here](https://help.openai.com/en/articles/10245774-sora-billing-faq).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://manus.im/share/oxmc7m9JJq1IRmtpj5mX2A?replay=1">Latest VK3GOD Contacts on WSPR Rocks - Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://artificialanalysis.ai/">AI Model &amp; API Providers Analysis | Artificial Analysis</a>: Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key performance metrics including quality, price, output speed &amp; latency.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1357079882441429062)** (4 messages): 

> `Livekit framework, GPT-4o tasks, Red team members supervision` 


- **Evals Usage Still Hazy for Livekit Voicebots**: A user inquired about how to use **evals** in the [Livekit framework](https://livekit.io/) for building **voicebots**.
   - No solutions were provided in the given context.
- **GPT-4o Tasks Flounder on First Try**: A user, who has been a paid member for over a year, reported that **GPT-4o tasks** weren't working as expected on their first try.
   - The user described that instead of creating the requested task, the model answered with *random topics*.
- **Red Team's Pet-Feeding Requires Supervision?**: A user humorously commented that *red team members require adult supervision themselves, even when they are simply trying to feed their own pets.*
   - The comment suggests a playful critique of the AI's actions being akin to unsupervised, perhaps chaotic, pet-feeding.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1357121586930389143)** (130 messages🔥🔥): 

> `AI Image Generation, Model Behavior, Content Policies vs Model Spec, Adult Content Generation` 


- **Prompt Engineering's Glowing Edits**: A user sought advice on improving image edits, specifically how to add a **glow effect** around the subject, reminiscent of a smiley face, requesting the model to replicate the effect on a provided image.
   - Another member suggested refining prompts by precisely defining desired outcomes and iteratively comparing results, emphasizing that improving output involves better communicating intent to the model.
- **Decoding Runes' Novelty**: A member described a system where **runes** decode or run things through a sequence, deriving missing functions from the runes between them, adding novelty for each instance.
   - They suggested running a **concept** through the sequence, with each rune transforming it, ultimately collapsing into a new rendition based on transformations.
- **OpenAI Model Spec vs. Content Policy Clash**: A discussion ignited over the permissibility of generating images of **adult products** with claims that it may violate content policies.
   - However, other members pointed to OpenAI's [Model Spec](https://model-spec.openai.com/2025-02-12.html) which they say *contradicts* the policy and may now permit such content if not otherwise harmful, highlighting a potential conflict between the documents.
- **Content Policies' Evolving Stance on Adult Toys**: Members debated whether creating images of **adult toys** violates OpenAI's content policies, referencing a [specific policy](https://openai.com/policies/usage-policies/) and the Model Spec.
   - While the Model Spec seems more permissive, the content policy states, *Don’t build tools that may be inappropriate for minors, including: Sexually explicit or suggestive content*, leading to confusion.



**Link mentioned**: <a href="https://model-spec.openai.com/2025-02-12.html">OpenAI Model Spec</a>: The Model Spec specifies desired behavior for the models underlying OpenAI's products (including our APIs).

  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1357121586930389143)** (130 messages🔥🔥): 

> `Image generation with glow effects, Model Spec vs Content Policies, Adult content generation, Image Editing Improvements` 


- **Glowing Edges Inspire Discord Members**: One member requested the generation of an image with a **colorful glowing outline** around the subject, similar to a smiley face, and asked for [improvement suggestions](https://cdn.discordapp.com/attachments/1046317269069864970/1357212216671207525/60F7927C-A2B5-456E-A46F-2CEE9D77D953.png).
   - Another member provided detailed prompting strategies, suggesting iterative refinement and defining desired outcomes precisely to guide the model's decisions.
- **Debate on Content Policies vs. Model Spec**: A discussion emerged regarding the **differences between OpenAI's Content Policies and the Model Spec**, with one member citing a newer [Model Spec from February 12, 2025](https://model-spec.openai.com/2025-02-12.html) seemingly contradicting the policies.
   - The debate centered on whether the **Model Spec**, described as *aspirational*, overrides the **Content Policies**, and the implications for generating content like adult products.
- **Adult Content Generation: Is It Allowed?**: Members debated whether **OpenAI's policies allow generating images of adult products**, with conflicting interpretations of the **Content Policies** and **Model Spec**.
   - While some initially believed such content was prohibited, others pointed to the **Model Spec** as a sign of changing permissibility, noting that the policy was updated January 29.
- **Specificity is Key for Image Editing Success**: Discord users discussed the importance of being specific when prompting AI models for image editing, particularly when defining the *subject* of an image.
   - A user seeking to add a glow to specific parts of an image was advised to precisely define which elements constitute the subject to avoid ambiguity for the model.



**Link mentioned**: <a href="https://model-spec.openai.com/2025-02-12.html">OpenAI Model Spec</a>: The Model Spec specifies desired behavior for the models underlying OpenAI's products (including our APIs).

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1357067691134947469)** (337 messages🔥🔥): 

> `Perplexity Pulse Program, Deep Research Updates, Gemini 2.5 vs Perplexity O1, Android App Home Screen, LLM Jailbreaks` 


- **Perplexity Pulse Program Sparks Curiosity**: Users on Discord are eager to learn about and join the **Perplexity Pulse Program**, which offers [Early Access to new features](https://x.com/testingcatalog/status/1897649019309961298?s=46) in exchange for feedback, plus perks like free **PPLX** and **merch**.
   - Access to the **Perplexity Pulse Group** is said to provide power users free **PPLX** in exchange for providing feedback.
- **Deep Research Update Disappoints Users**: Users are reporting that the updated **"deep research"** feature is [slower and less effective](https://www.reddit.com/r/perplexity_ai/comments/1jq27a6/why_is_perplexitys_updated_deep_research_slower/), with one user claiming it *overfits with confirmation bias* and another saying that it's slower and only gets *20 sources*.
   - Users noted that the new version of **Deep Research** uses more server resources to give worse output compared to the older versions of the algorithm.
- **Gemini 2.5 Outshines Perplexity O1 for General Use**: Discord users are sharing their experiences, saying that [**Gemini 2.5** offers similar quality to **Perplexity's O1 Pro**](https://cdn.discordapp.com/attachments/1047649527299055688/1357423109778702607/image0.jpg?ex=67f02649&is=67eed4c9&hm=d5049580f5523c24bef016f8050e7b92c1f37e1ec416ad9c7ab8b4509c735bf5&) for free but while Perplexity is better for research papers and for solid science due to academic search.
   - Some users note that Gemini's deep research, while strong, is *vulnerable to SEO cheating websites* but that it offers better reasoning with *youtube sources*.
- **Perplexity's AI Assistant Suggests Reorganizing Android Home Screen with a Twist**: When asked to reorganize an Android home screen, **Perplexity** suggested *integration of transport widgets* and quick access to *alcohol store opening hours*, leading one user to wonder whether the app assumes that the user is an alcoholic.
   - AI even offers to recombine so that it's not *offensive* to the interlocutor.
- **Reporting LLM Jailbreaks is Futile**: Members stated that model makers can't keep up with jailbreaks, and that jailbreaks are more about the model than how Perplexity censors it.
   - One member said that *reporting LLM jailbreaks is as futile as teaching a toddler to sit still for more than five seconds*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/elmo-sesame-street-shrug-idk-i-have-no-idea-gif-2724737697653756220">Elmo Sesame Street GIF - Elmo Sesame Street Shrug - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/testingcatalog/status/1897649019309961298?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Power users of @perplexity_ai will be able to join &#34;Perplexity Puls Group&#34; for providing feedback on its responses and earn different perks like free PPLX and merch. Free PPLX? I accept! 👀</li><li><a href="https://tenor.com/view/boredmemes-apechain-ape-chain-notacult-gif-1047921344109153463">Boredmemes Apechain GIF - Boredmemes Apechain Ape - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://play.google.com/store/apps/details?id=com.microsoft.translator">Microsoft Translator - Apps on Google Play</a>: no description found</li><li><a href="https://tenor.com/view/dog-gif-20050013">Dog GIF - Dog - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/voices-are-real-jack-nicholson-nod-yes-the-shining-gif-16412513">Voices Are Real Jack Nicholson GIF - Voices Are Real Jack Nicholson Nod - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://open.spotify.com/episode/1vLmnLl3jFQFrx1QvTtg0r?si=oSf4_PuOR72iXHBGtbg_fQ">Aravind Srinivas</a>: Tetragrammaton with Rick Rubin · Episode</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jq27a6/why_is_perplexitys_updated_deep_research_slower/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://github.com/pnd280/complexity">GitHub - pnd280/complexity: ⚡  Supercharge your Perplexity.ai</a>: ⚡  Supercharge your Perplexity.ai. Contribute to pnd280/complexity development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1357102032393211925)** (6 messages): 

> `Shareable threads, Perplexity AI Search` 


- ****Shareable Threads** request!**: Perplexity AI requests that users ensure their thread is **shareable**.
   - A link to a Discord channel message was shared as context: [Discord message](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Links galore to Perplexity AI Search**: Several links to **Perplexity AI search results** were shared, including queries related to writing a Perplexity prompt and AI-based learning resources.
   - Some of the searches included [abolishing IRS](https://www.perplexity.ai/page/trump-aims-to-abolish-irs-.c.0aEbqTJGTtrHHcmWmJg) and [AI Learning](https://www.perplexity.ai/search/what-are-some-ai-based-learnin-Q.AhpXIeRoqn3FRp9kf0aA).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1357444400447357028)** (1 messages): 

> `Perplexity API versioning, API Versioning, Breaking Changes` 


- **API versioning best practices**: A member complained about the lack of versioning in the **Perplexity API**, saying *That's a breaking change, you don't do that in production when you have customers using your API*.
   - They suggested having **/v1/** in the API URL so that a **/v2/** can be created without breaking the actively used **/v1**.
- **Versioning Avoids Breaking Changes**: The user emphasized that introducing breaking changes without proper versioning can negatively impact customers using the API in production.
   - By implementing versioning (e.g., **/v1/**), developers can introduce **/v2/** without disrupting existing users on **/v1**.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1357069166628307158)** (177 messages🔥🔥): 

> `OpenAI Nonprofit Commission Guidance, Github Copilot and OpenRouter Integration, Google rents Nvidia Blackwell chips from CoreWeave, Inference Scaling and the Log-x Chart, Runway Secures $300M in Series D Funding` 


- **Github Copilot now features plug-n-play OpenRouter Keys**: Users can now add an [OpenRouter key](https://openrouter.ai/) and select any model they want in Github Copilot.
   - This integration gives Github Copilot users access to a wider range of models beyond those offered by OpenAI.
- **Google to Rent Nvidia Blackwell from CoreWeave**: Google is in advanced talks to rent **Nvidia Blackwell** chips from CoreWeave, and potentially house its **TPUs** in CoreWeave facilities, highlighting intense customer demand for compute ([The Information Article](https://www.theinformation.com/articles/google-advanced-talks-rent-nvidia-ai-servers-coreweave)).
   - The move suggests Google might be **TPU poor**, especially with the realization that inference demand will be high.
- **OpenAI Stealth Launches Quasar Alpha on OpenRouter**: A new "stealth model" named **Quasar Alpha** launched on [OpenRouter](https://openrouter.ai/openrouter/quasar-alpha) with **1,000,000 context** and free input/output tokens, described as a powerful, all-purpose model supporting long-context tasks and code generation.
   - Community members have speculated it could be an **open source SSM** given its speed, or even secretly from **OpenAI** despite their lack of formal announcement, and despite the model's tendency to output short responses and listicles.
- **Anthropic's CoT Research Casts Doubt on Monitoring**: New [Anthropic research](https://www.anthropic.com/research/reasoning-models-dont-say-think) reveals that reasoning models don't accurately verbalize their reasoning, questioning whether monitoring **chains-of-thought (CoT)** is sufficient for catching safety issues.
   - They slipped problem-solving hints to **Claude 3.7 Sonnet** and **DeepSeek R1**, then tested whether their Chains-of-Thought would mention using the hint to come to that conclusion. Read the [related blog post here](https://www.anthropic.com/research/reasoning-models-dont-say-think).
- **DeepSeek V3 Fails Frontier Model Test**: New [SEAL leaderboards](http://scale.com/leaderboard) indicate that **DeepSeek V3** is not a frontier-level model, ranking 8th on Humanity’s Last Exam (text-only) and 12th on MultiChallenge (multi-turn).
   - Despite not being 'frontier', some users find it to be a **banger** better than **Claude 3.7** for writing ptx kernels for b200s.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/bradlightcap/status/1907810330018726042">Tweet from Brad Lightcap (@bradlightcap)</a>: very crazy first week for images in chatgpt - over 130M users have generated 700M+ (!) images since last tuesdayIndia is now our fastest growing chatgpt market 💪🇮🇳the range of visual creativity has...</li><li><a href="https://x.com/AnthropicAI/status/1907833407649755298">Tweet from Anthropic (@AnthropicAI)</a>: New Anthropic research: Do reasoning models accurately verbalize their reasoning?Our new paper shows they don&#39;t.This casts doubt on whether monitoring chains-of-thought (CoT) will be enough to rel...</li><li><a href="https://x.com/tobyordoxford/status/1907379921825014094?s=61">Tweet from Toby Ord (@tobyordoxford)</a>: Here is the revised ARC-AGI plot. They&#39;ve increased their cost-estimate of the original o3 low from $20 per task to $200 per task. Presumably o3 high has gone from $3,000 to $30,000 per task, whic...</li><li><a href="https://x.com/tobyordoxford/status/1907379650831015964?s=61">Tweet from Toby Ord (@tobyordoxford)</a>: When I posted this thread about how o3&#39;s extreme costs make it less impressive than it first appears, many people told me that this wasn&#39;t an issue as the price would quickly come down.I check...</li><li><a href="https://x.com/AndrewCurran_/status/1907886417088553431">Tweet from Andrew Curran (@AndrewCurran_)</a>: New numbers from Pew this morning, they reveal a large gap in perception between the general public and people whose work and research relates to AI. Usage: 66% of the general US public have still nev...</li><li><a href="https://x.com/JustinLin610/status/1907748767933280707">Tweet from Junyang Lin (@JustinLin610)</a>: Qwen3 release time not decided yet (rumors everywhere). After QwQ, we have spent most of our efforts on the new Qwen series and we are now at the stage of final preparation. Just still need a little m...</li><li><a href="https://x.com/TheXeophon/status/1907880330985390215">Tweet from Xeophon (@TheXeophon)</a>: Here is the new stealth model on my vibe check. It is now the best non-thinking model (at least it has no thinking tokens...). The outputs are super short, it loves Certainly! and listicles. Super int...</li><li><a href="https://x.com/Baidu_Inc/status/1907802772134563892">Tweet from Baidu Inc. (@Baidu_Inc)</a>: 🚀 ERNIE X1 is now live on Baidu AI Cloud&#39;s MaaS platform Qianfan, where enterprise users and developers can now access its API!In evaluations across multiple public datasets, our new deep-thinkin...</li><li><a href="https://x.com/KeyTryer/status/1907504512857944069">Tweet from Key 🗝 🦊 (@KeyTryer)</a>: omg Github Copilot now lets me add an OpenRouter key and select any model I want. Massive.</li><li><a href="https://x.com/steph_palazzolo/status/1907517483524686129">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: NEW: Google is in advanced talks to rent Nvidia Blackwell chips from CoreWeave, as well as to potentially house its TPUs in Coreweave facilities.The deal highlights the intense customer demand for com...</li><li><a href="https://x.com/AnthropicAI/status/1907833412171207037">Tweet from Anthropic (@AnthropicAI)</a>: We slipped problem-solving hints to Claude 3.7 Sonnet and DeepSeek R1, then tested whether their Chains-of-Thought would mention using the hint (if the models actually used it).Read the blog: https://...</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha">Quasar Alpha - API, Providers, Stats</a>: This is a cloaked model provided to the community to gather feedback. It’s a powerful, all-purpose model supporting long-context tasks, including code generation. Run Quasar Alpha with API</li><li><a href="https://x.com/alexandr_wang/status/1907836081783058720">Tweet from Alexandr Wang (@alexandr_wang)</a>: 🚨 Narrative Violation—DeepSeek V3 is NOT a frontier-level model.SEAL leaderboards have been updated with DeepSeek V3 (Mar 2025).- 8th on Humanity’s Last Exam (text-only).- 12th on MultiChallenge (mul...</li><li><a href="https://techcrunch.com/2025/03/20/perplexity-is-reportedly-in-talks-to-raise-up-to-1b-at-an-18b-valuation/">Perplexity is reportedly in talks to raise up to $1B at an $18B valuation | TechCrunch</a>: AI-powered search startup Perplexity is said to be in early talks to raise up to $1 billion in a new funding round valuing the startup at $18 billion.</li><li><a href="https://singularityhub.com/2015/01/26/ray-kurzweils-mind-boggling-predictions-for-the-next-25-years/#sm.001tlz026ghlfl9114t1t2yxvo5lq)">Ray Kurzweil&#x27;s Mind-Boggling Predictions for the Next 25 Years</a>: In my new book BOLD, one of the interviews that I’m most excited about is with my good friend Ray Kurzweil. Bill Gates calls Ray, “the best person I know...</li><li><a href="https://runwayml.com/news/runway-series-d-funding">Runway News | Towards a new media ecosystem with world simulators</a>: no description found</li><li><a href="https://x.com/eric_haibin_lin/status/1907845598432342328">Tweet from Haibin (@eric_haibin_lin)</a>: We are open sourcing bytecheckpoint and veomni! bytecheckpoint is the Bytedance&#39;s production checkpointing system for foundation model training, battle-tested with jobs with 10k+ GPUs. Blazing fas...</li><li><a href="https://github.com/ByteDance-Seed/ByteCheckpoint">GitHub - ByteDance-Seed/ByteCheckpoint: ByteCheckpoint: An Unified Checkpointing Library for LFMs</a>: ByteCheckpoint: An Unified Checkpointing Library for LFMs - ByteDance-Seed/ByteCheckpoint</li><li><a href="https://github.com/ByteDance-Seed/VeOmni">GitHub - ByteDance-Seed/VeOmni: VeOmni: Scaling any Modality Model Training to any Accelerators with PyTorch native Training Framework</a>: VeOmni: Scaling any Modality Model Training to any Accelerators with PyTorch native Training Framework - ByteDance-Seed/VeOmni</li><li><a href="https://techcrunch.com/2025/04/03/microsoft-reportedly-pulls-back-on-its-data-center-plans/">Microsoft reportedly pulls back on its data center plans | TechCrunch</a>: Microsoft has reportedly pulled back on data center projects around the world, suggesting that the company is wary of overexpanding.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1357151851841192047)** (10 messages🔥): 

> `Joanne Jang, GPT-4o Transcribe, ChatGPT 4o ImageGen Watermark` 


- **Jang-ling All the Way to the Podcast**: A member met **Joanne Jang** at OpenAI and expressed a desire to have her on their podcast, but noted possible restrictions due to Jang's affiliation with OpenAI.
   - They expressed confidence that *eventually* the podcast appearance will happen.
- **GPT-4o Transcribe Hallucinates!**: A user shared a tweet noting that **GPT-4o Transcribe** is hallucinating the phrase *Transcript by PODTRANSCRIPTS, COM*.
   - Another member responded *not again*, likely referring to similar past issues with AI transcription services.
- **Watermark-gate scandalized ChatGPT ImageGen**: A user shared a tweet about a new watermark appearing on images created with **ChatGPT 4o ImageGen**, along with an experimental feature mentioning a *watermarked asset pointer*.
   - Another member mentioned that the watermark can be removed on the **$200 tier** (or at least that's currently in the files).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/matthen2/status/1907477758789218796">Tweet from Matt Henderson (@matthen2)</a>: gpt-4o-transcribe is certainly interesting…Is anyone else seeing it hallucinate “Transcript by PODTRANSCRIPTS , COM”?</li><li><a href="https://x.com/btibor91/status/1907861559029682323">Tweet from Tibor Blaho (@btibor91)</a>: Preview of the new watermark (seen on LinkedIn)Quoting Tibor Blaho (@btibor91) Images created with ChatGPT 4o ImageGen might soon include a watermarkA recent update to the ChatGPT web app introduces a...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1357405163245011135)** (40 messages🔥): 

> `Devin 2.0, Agent-based IDEs, Windsurf vs Cursor, Claude-code API, Polars updates` 


- ****Cognition Labs** releases **Devin 2.0****: **Cognition Labs** introduced [Devin 2.0](https://fxtwitter.com/cognition_labs/status/1907836719061451067), a new agent-native IDE experience, generally available starting at **$20** plus pay as you go.
   - Some members found this launch *too funny* because the competition might find PMF before **Devin** does.
- ****Windsurf** hailed as better Agent than **Cursor****: Some members find that **Windsurf** has *nailed the Agent stuff better* than **Cursor**, whereas **Cursor** reduces friction for getting the LLM to do boring but time consuming things.
   - Others agreed, but said that the question is basically whether it is meaningfully better than having a separate window open, and the answer is, *uh, barely*.
- ****Claude-code API** can burn the dollars**: One member suggested using **claude-code** to generate data for benchmarks, dashboard and throwaway UIs, as LLMs are good at it and it’s so much worth it.
   - Some mad people on Twitter are running multiple instances of **Claude code** in parallel, *burning $500 in a weekend*.
- ****Claude** not bad in **Polars****: **Claude** and **Gemini** aren't bad in **Polars**, one member shared, noting that *you gotta tell him its with_columns now*.
   - It was also mentioned that **Claude 3.7** has no idea how to use the new updates, and that having competing information in the actual weights makes it much harder to overcome with context.



**Link mentioned**: <a href="https://fxtwitter.com/cognition_labs/status/1907836719061451067">Tweet from Cognition (@cognition_labs)</a>: Introducing Devin 2.0: a new agent-native IDE experience.Generally available today starting at $20.  🧵👇

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1357107804774862988)** (53 messages🔥): 

> `Distilling Reasoning Capabilities, Superhuman AI Impact Prediction, Algorithmic progress vs data progress, Dwarkesh AGI Forecast Podcast, Nvidia Open Code Reasoning Collection` 


- **Student Models Bridge Reasoning Gap**: A recent paper ([arxiv.org/abs/2504.01943](https://arxiv.org/abs/2504.01943)) highlights the success of distilling reasoning capabilities into student models, bridging the gap between reasoning and standard LLMs on **coding tasks**.
   - The distilled models achieved **61.8%** on LiveCodeBench and **24.6%** on CodeContests using only SFT, surpassing alternatives trained with reinforcement learning.
- **Superhuman AI Impact Predicted Within Decade**: A scenario ([ai-2027.com](https://ai-2027.com/)) predicts an enormous impact of **superhuman AI** over the next decade, exceeding that of the Industrial Revolution.
   - The prediction is informed by *trend extrapolations, wargames, expert feedback, experience at OpenAI, and previous forecasting successes*.
- **Dwarkesh AGI Timeline Sparks Debate**: An AGI forecast podcast by Dwarkesh Patel was discussed, with some finding its predictions and assumptions unconvincing.
   - One member expressed confusion over the lack of acknowledgment of scaling laws and emphasis on algorithmic progress over data progress, while others described it as *fanfiction*.
- **Nvidia's Empty Reasoning Data Collection**: A member shared a link to NVIDIA's Hugging Face collection for open code reasoning ([huggingface.co](https://huggingface.co/collections/nvidia/opencodereasoning-67ec462892673a326c0696c1)), meant to advance data distillation for competitive coding.
   - Another member noted that the collection is currently empty.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2504.01943">OpenCodeReasoning: Advancing Data Distillation for Competitive Coding</a>: Since the advent of reasoning-based large language models, many have found great success from distilling reasoning capabilities into student models. Such techniques have significantly bridged the gap ...</li><li><a href="https://lancelqf.github.io/note/llm_post_training/">From REINFORCE to Dr. GRPO</a>: A Unified Perspective on LLM Post-training</li><li><a href="https://huggingface.co/collections/nvidia/opencodereasoning-67ec462892673a326c0696c1">OpenCodeReasoning - a nvidia Collection</a>: no description found</li><li><a href="https://ai-2027.com/">AI 2027</a>: A research-backed AI scenario forecast.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1357318149086646325)** (2 messages): 

> `OpenAI Deep Research, Plumbing Repair Costs` 


- **OpenAI Deep Research saves user $2,050 on plumbing repair**: A user on X posted about how [OpenAI Deep Research](https://x.com/jbohnslav/status/1907759146801197450) helped them find a plumber who charged **$200** for a repair, compared to an initial quote of **$2,250**.
- **Deep Research: a bargain hunter's AI tool**: The user joked that OpenAI Pro *literally saved me $2,050, almost paying for itself for the entire year!*



**Link mentioned**: <a href="https://x.com/jbohnslav/status/1907759146801197450">Tweet from Jim Bohnslav (@jbohnslav)</a>: Got a quote on a simple plumbing repair: $2,250. Ask OpenAI Deep Research for market rate: $300-$500. Ask DR for good plumbers in my area. Call the first one. Fixes it for $200. OpenAI Pro literally s...

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1357072717660291265)** (170 messages🔥🔥): 

> `Gemini 2.5 Pro Rate Limits, Architect Mode Optimizations, Voice Command Configuration, MCPs for LSPs and Treesitter` 


- **Gemini 2.5 Pro triggers rate limit woes**: Users report hitting the **20 requests/minute rate limit** with Gemini 2.5 Pro in Aider, even with minimal usage, suspecting background requests by Aider.
   - Some users are seeing **5 RPM** despite having tier 1 API keys, while others report seeing the documented **20 RPM** at tier 1, with screenshots provided like [this](https://cdn.discordapp.com/attachments/1131200896827654149/1357114156037312683/image.png?ex=67efaf4c&is=67ee5dcc&hm=ab00c0d89a9a4029e1244032c897f52cf418c2b5c10a03543f8574d73b779750&).
- **Architect Mode saves Gemini Quota**: To conserve Gemini 2.5 Pro quota in architect mode, a user suggests setting `--editor-model sonnet` to offload editing tasks to a cheaper model like **Sonnet**.
   - One member stated that *you could try haiku I guess.. but even just 3.7 sonnet when it only does the editing is dirt cheap*.
- **Voice Command needs provider config**: Users are looking for configuration options to select voice models and providers for the `/voice` command, which currently uses **OpenAI Whisper**.
   - A pending PR ([https://github.com/Aider-AI/aider/pull/3131](https://github.com/Aider-AI/aider/pull/3131)) may address this, allowing the use of different providers and models.
- **Code Actions Need MCP Love**: One user suggested that **LSP Code Actions and tree-sitter** can improve code editing and refactoring.
   - The member further pointed to the need for *MCPs for LSPs and treesitter for code editing, and it will speed up and make more robust these simple but large edits*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/a7m7s1p6dv20/status/1907684868164825260?s=46">Tweet from ᅟ (@a7m7s1p6dv20)</a>: (initial?) pricing scheme for gemini 2.5 provia glama AI</li><li><a href="https://x.com/tom_doerr/status/1907450456575533269">Tweet from Tom Dörr (@tom_doerr)</a>: Incredibly satisfying to watch multiple agents work on the same project, each with a different task, fixing bugs and resolving merge conflicts to push back to main</li><li><a href="https://smithery.ai/server/@smithery-ai/github">Github | Smithery</a>: no description found</li><li><a href="https://smithery.ai/server/@smithery-ai/server-sequential-thinking">Sequential Thinking | Smithery</a>: no description found</li><li><a href="https://x.com/OpenRouterAI/status/1905300582505624022">Tweet from OpenRouter (@OpenRouterAI)</a>: To maximize your free Gemini 2.5 quota:1. Add your AI Studio API key in https://openrouter.ai/settings/integrations. Our rate limits will be a “surge protector” for yours.2. Set up OpenRouter in your ...</li><li><a href="https://smithery.ai/server/@IzumiSy/mcp-duckdb-memory-server">DuckDB Knowledge Graph Memory Server | Smithery</a>: no description found</li><li><a href="https://ai.google.dev/gemini-api/docs/rate-limits">no title found</a>: no description found</li><li><a href="https://smithery.ai/server/@PhillipRt/think-mcp-server">Think Tool Server | Smithery</a>: no description found</li><li><a href="https://aider.chat/docs/faq.html#what-llms-do-you-use-to-build-aider,">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://ai.google.dev/gemini-api/docs/billing">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1357090980037398730)** (14 messages🔥): 

> `Aider Shell, Openrouter Errors, Git Repo Corrupted, Aider Print Prompt Costs, Gemini Comments` 


- ****Aider**'s Shell Revealed!**: A user inquired about which shell **Aider** employs when executing commands to debug Docker-related issues.
   - The user observed that **Aider**'s `curl` commands succeed while their own shell (`bash`) `curl` commands fail, sparking the query.
- ****Openrouter**'s 500 Errors Plague Users**: Users reported experiencing `litellm.BadRequestError` with **Openrouter**, specifically a `KeyError: 'choices'` and `Internal Server Error` (code 500) when using `openrouter/google/gemini-2.5-pro-exp-03-25:free`.
   - The errors are intermittent, leading to uncertainty about the underlying cause.
- ****Git Repo Corruption** Concerns Aired!**: Multiple users encountered "Unable to list files in git repo: BadObject" errors, prompting concerns about potential **Git repo corruption**.
   - The error message suggests checking if the Git repo is corrupted but doesn't provide immediate solutions.
- ****Gemini**'s Commentary Overload!**: A user struggled to prevent **Gemini/gemini-2.5-pro-exp-03-25** from adding comments everywhere, despite attempts to configure it via [GLOBAL_CONVENTIONS.md](https://github.com/schpet/dotfiles/blob/main/.config/aider/GLOBAL_CONVENTIONS.md).
   - Another user, however, praised Gemini's general result quality, lamenting the quota limits.
- **Maximize **Aider** with `--watch-files`!**: A new **Aider** user, after basic setup, inquired about the advantages of using `--watch-files` mode.
   - The user initially faced issues with `ai!` & save being ignored, but reported that they have resolved this problem.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1357080838394875975)** (12 messages🔥): 

> `Refact Polyglot Claims, Aider Polyglot Benchmark, SWE-bench evaluation, OpenAI's PaperBench` 


- **Refact Claims High Score on Aider Polyglot Benchmark**: **Refact** claims a **92%** score on the [Aider polyglot benchmark](https://medium.com/@refact_ai/refact-ai-agent-scores-highest-on-aiders-polyglot-benchmark-93-3-00ed0e3b9a6b), prompting discussion about its legitimacy and cost.
   - A member suggested investigating the cost of achieving such a high score, suggesting it could be valuable if legitimate.
- **Doubt and Skepticism on the Refact AI results**: A member expressed interest in running the benchmark with larger **--tries** values on free or cheap models to assess their worth.
   - Another member stated that the [Aider Polyglot Benchmark](https://github.com/paul-gauthier/aider) is not the correct test for autonomous agents, instead suggesting [SWE-bench](https://github.com/princeton-nlp/SWE-agent).
- **Aider's Performance on SWE-bench Questioned**: A member inquired about **Aider's** performance on **SWE-bench**, questioning why it should be benchmarked there.
   - Another member clarified that **SWE-bench** is designed for autonomous agents, while **Aider polyglot** is tailored for testing models with Aider, and noted that **Aider** hasn't submitted its score on **SWE-bench** recently.
- **Adapt Aider to OpenAI PaperBench Evaluation?**: A member suggested adapting **Aider** to the new [OpenAI PaperBench evaluation benchmark](https://openai.com/index/paperbench/).
   - No further discussion was had on the topic.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1357068008492892291)** (120 messages🔥🔥): 

> `LM Studio for Brave, System Prompt in Local Server, CUDA0 Buffer Allocation Failure, Q4 vs Q6 Model Quality, Dual GPU Setup with LM Studio` 


- **Brave New Integration for Local LLMs**: Users explored integrating **LM Studio** with the **Brave** browser, pointing its server endpoint to `http://localhost:1234/v1/chat/completions`, and sought guidance on configuring the **API** to utilize system prompts, with [lmstudioservercodeexamples](https://github.com/YorkieDev/lmstudioservercodeexamples) as a helpful resource.
   - Many struggled with providing Brave the correct API endpoint.
- **System Prompt: API Key to Unlocking LLM Potential**: To use **system prompts** with **LM Studio's local server**, users need to provide the prompt via the **API call**, not through the LM Studio interface, the documentation is available [here](https://lmstudio.ai/docs/app/api).
- **CUDA Conundrums: Memory Mayhem Strikes Again**: A *'failed to allocate cuda0 buffer'* error usually indicates insufficient memory for the model being used, and missing the **mmproj** file when downloading from **HF mirror** can cause the issue, which can be resolved by downloading from within **LM Studio** with proxy settings enabled.
- **Q4 vs Q6: Quality Quandaries**: The quality degradation between **Q4** and **Q6 models** depends on the model's size and recency, with older and smaller models suffering more noticeably, but a **32B model** should not have a significant impact.
- **Dual GPUs: Plug and Play Power?**: Users reported that utilizing dual **GPUs (4090 + 5090)** in **LM Studio** is surprisingly easy, achieving good performance (24-25 tokens/s) on a **32B Q8 model** with flash attention, though performance may be bottlenecked by **PCIE connection** when splitting models without **NVLink**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/app/api">LM Studio as a Local LLM API Server | LM Studio Docs</a>: Run an LLM API server on localhost with LM Studio</li><li><a href="https://tenor.com/view/meme-horrors-beyond-our-comprehension-low-tier-god-mods-banned-gif-8530537273735940092">Meme Horrors Beyond Our Comprehension GIF - Meme Horrors beyond our comprehension Low tier god - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lmstudio.ai/work">Use LM Studio @ Work</a>: Use local LLMs at your workplace or organization with LM Studio</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSd-zGyQIVlSSqzRyM4YzPEmdNehW3iCd3_X8np5NWCD_1G3BA/viewform?usp=sf_link)">LM Studio @ Work</a>: Thank you for your interest in using LM Studio @ Work! Please fill the following form and we will get back to you as soon as we can.- Team LM Studio (team@lmstudio.ai)
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1357079204750954607)** (36 messages🔥): 

> `Unsloth 2.0 6b performance, M3 Ultra vs M4 Max for LLMs, Macs for LLM Use, Qwen QWQ Quality, GPU vs Apple Silicon Benchmarks` 


- **Unsloth 2.0 6b solves problems despite slow speed**: A user reported running **Unsloth 2.0 6b** on 4x 3090 + 256GB RAM at ~3 tok/s and that it solved a coding problem in 20-30 minutes when smaller models and **ChatGPT** failed.
   - They found **Qwen QWQ** to be 90% of the quality of **R1** at 5% of the parameters, emphasizing preference for quality over speed.
- **M3 Ultra is not great for LLM, M4 Max is excellent!**: A user stated that the **M3 Ultra Mac Studio** is terrible for **LLM** use due to unbalanced memory, compute, and bandwidth, while the **M4 Max** and **5090** are excellent.
   - They argued the **M3 Ultra's** large VRAM is only suitable for gigantic MoE models and is overpriced for smaller models fitting in a **5090's 32GB VRAM** or a **M4 Max's 96GB**.
- **Apple Silicon bandwidth discussed**: A user clarified that the **M3 Ultra** has 800GB/s bandwidth for 512GB VRAM, while a **5090** has 1792GB/s for 32GB and an **M4 Max** has 546GB/s for 128GB.
   - Another user noted that the claim that the **M3 ultra** is *terrible* because it's *unbalanced* is ridiculous, as it still has way more bandwidth than even the **M4 max**, as did all previous ultra versions.
- **M4 Max vs M1 Ultra benchmarked**: A member pointed to [llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions/4167) showing roughly a tie between **M4 Max** and **M1 Ultra** for small, bandwidth-bound models, with **M1** pulling ahead for larger quants.
   - A user shared a [GitHub repo with GPU benchmarks](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) for LLM inference.



**Link mentioned**: <a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference? - XiongjieDai/GPU-Benchmarks-on-LLM-Inference

  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1357182516234158181)** (49 messages🔥): 

> `Web Search Citations in API, Quasar Alpha Stealth Model, Inference Net endpoints Disabled, Coding Optimized Models` 


- ****OpenRouter Adds Web Search Citations to API****: OpenRouter announced that [web search now returns citations in the API](https://x.com/OpenRouterAI/status/1907623560522379436), standardizing them across all models, including native online models like **OpenAI** and **Perplexity**.
   - Users can access the [documentation](https://openrouter.ai/docs/features/web-search) to incorporate web search results by activating the `web` plugin or appending `:online` to the model slug.
- ****Quasar Alpha: A Stealth 1M Context Model Unveiled****: OpenRouter announced [Quasar Alpha](https://openrouter.ai/openrouter/quasar-alpha), a **free**, **1M token** context length model optimized for coding but general-purpose, before its public release.
   - Prompts and completions will be logged, and feedback can be provided in the [dedicated Discord thread](https://discord.com/channels/1091220969173028894/1357398117749756017) to help improve the model.
- ****Inference Net endpoints Temporarily Grounded****: Inference Net endpoints will be temporarily disabled on OpenRouter for platform maintenance, with a promise to return shortly.
- ****Quasar Alpha: Initial Benchmarks Sound Good****: Initial benchmarks for **Quasar Alpha** sound good, according to some users, despite others finding it performs poorly on coding tests.
   - One user shared [a vibe check on X](https://x.com/TheXeophon/status/1907880330985390215/photo/1), describing it as the best non-thinking model with super short outputs, while others speculated about its origins, with some suspecting it might be a new Qwen variant.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1907880330985390215/photo/1">Tweet from Xeophon (@TheXeophon)</a>: Here is the new stealth model on my vibe check. It is now the best non-thinking model (at least it has no thinking tokens...). The outputs are super short, it loves Certainly! and listicles. Super int...</li><li><a href="https://x.com/OpenRouterAI/status/1907870610602275203">Tweet from OpenRouter (@OpenRouterAI)</a>: Excited to announce our first-ever “stealth” model... Quasar Alpha 🥷It’s a prerelease of an upcoming long-context foundation model from one of the model labs:- 1M token context length- specifically o...</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha">Quasar Alpha - API, Providers, Stats</a>: This is a cloaked model provided to the community to gather feedback. It’s a powerful, all-purpose model supporting long-context tasks, including code generation. Run Quasar Alpha with API</li><li><a href="https://x.com/OpenRouterAI/status/1907623560522379436">Tweet from OpenRouter (@OpenRouterAI)</a>: A highly-requested feature: web search now returns citations in the API 🌐We&#39;ve standardized them for all models, including native online models like OpenAI&#39;s web tool and Perplexity:</li><li><a href="https://openrouter.ai/docs/features/web-search">Web Search - Real-time Web Grounding for AI Models</a>: Enable real-time web search capabilities in your AI model responses. Add factual, up-to-date information to any model&#x27;s output with OpenRouter&#x27;s web search feature.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1357320733528821842)** (1 messages): 

> `AI character platform, charactergateway.com` 


- **Character Gateway Launches for Devs**: A new AI character platform called [Character Gateway](https://charactergateway.com/) has launched, targeting developers with tools to create, manage, and deploy **AI characters/agents**.
   - The platform emphasizes simplicity with *no database, no prompt engineering, no subscription, [and] no new SDK*.
- **Character Gateway API Features Chat Completion**: Character Gateway enables users to generate characters and images, and send **/chat/completion requests** to characters using their own **OpenRouter** key.
   - The platform does not meter token usage, giving developers more control over costs; a feature to list trending public characters and integrate them into apps is a work in progress (**WIP**).



**Link mentioned**: <a href="https://charactergateway.com/">Character Gateway</a>: AI Character API Platform for Developers

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1357067303904350278)** (99 messages🔥🔥): 

> `Gemini 2.5 Pro, Image responses, OpenAI Responses API, Targon Speed, Anthropic Blocking` 


- **Google's Gemini 2.5 Pro Gets Mixed Reception**: Some users are questioning whether **Gemini 2.5 Pro** is working for them.
   - It's noted that free models hosted by Google often have very low rate limits, though users can bypass this by using their own API key.
- **Image Response Capabilities in the Works**: OpenRouter is actively working on supporting image responses, potentially using a new **Responses API**.
   - There's speculation that OpenRouter's interfaces might diverge from OpenAI's in the future, possibly leading to the release of an SDK.
- **OpenAI's API Prompts Debate Over Future Compatibility**: OpenRouter developers are considering adding support for the **OpenAI Responses API**, especially since OpenAI may gradually deprecate chat completions.
   - One member planning to move to the Responses API as it feels *more standard, consistent, and well-designed*.
- **Targon's Speed Questioned**: Users are discussing whether **Targon's speed** is due to miners potentially ignoring sampling parameters, leading to biased distributions, referencing [verifier.py on GitHub](https://github.com/manifold-inc/targon/blob/main/verifier/verifier.py).
   - One member pointed out that *they generate the results once and cache the results, so if you ask the same question, they give you back the same reply, even if you change the parameters*
- **Users find Anthropic Costly**: A user shared their frustration after their AI model switched to **Anthropic** without them noticing, resulting in unexpected charges, despite having it on an ignored providers list.
   - Another member said Anthropic is *not worth the money for me—never will be*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/mistralai/mistral-small-3.1-24b-instruct">Mistral Small 3.1 24B - API, Providers, Stats</a>: Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. Run Mistral Small 3.1 24B with API</li><li><a href="https://openrouter.ai/mistralai/mistral-sm">Discord</a>: no description found</li><li><a href="https://openrouter.ai/models?fmt=cards&input_modalities=image&order=newest&max_price=0">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://github.com/manifold-inc/targon/blob/main/verifier/verifier.py">targon/verifier/verifier.py at main · manifold-inc/targon</a>: A library for building subnets with the manifold reward stack - manifold-inc/targon
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1357071009647300842)** (102 messages🔥🔥): 

> `Paid frontier models in production, vLLM/TGI Setup with RTX 5000, GPU server costs, Counterfeit detection with VLMs, Chat templates in training` 


- **RTX 5000 series users have vLLM/TGI setup issues**: Members are running into problems setting up **vLLM** or **TGI** with a new **RTX 5000** series card and they need a nightly version of **PyTorch** and **Cuda 12.8** but that's not so easy...
   - As one member stated, *when you install something else, PyTorch gets overwritten by the old version*, pointing to these github repos for help: [vllm-project/vllm/issues/14452](https://github.com/vllm-project/vllm/issues/14452), [pytorch/My-rtx5080-gpu-cant-work-with-pytorch/217301](https://discuss.pytorch.org/t/my-rtx5080-gpu-cant-work-with-pytorch/217301), [lllyasviel/stable-diffusion-webui-forge/issues/2601](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2601), [ComfyUI/discussions/6643](https://github.com/comfyanonymous/ComfyUI/discussions/6643).
- **VLMs detect counterfeit fashion**: Members shared research about counterfeit products and presented a computer-vision-based system using deep neural networks, claiming **99.71% accuracy** after rejections for branded garments.
   - They cited [this paper](https://arxiv.org/abs/2410.05969) which states that the system does not require special security tags or modifications to supply chain tracking, and transfer-trained on a small number of fake and genuine articles.
- **Hugging Face transparency needs improvements**: Members expressed confusion about Hugging Face's billing and quota systems as well as service usage for **GPU Spaces, Zero GPU Spaces, Serverless Inference API**.
   - They would like HF to provide  *“reporting, communication, and consultation”* about major changes, for example posting *“We're going to implement a major change. It'll be unstable for a few days”*.
- **Chat Templates can now train models**: A member asked if it's possible to pass a **chat_template** to the **transformers TrainingArguments** or **Trainer** to use a custom chat_template for models during inference time, and asked if it made sense to train as well.
   - Another member confirmed that this is possible, [linking to documentation](https://huggingface.co/docs/transformers/main/en/chat_template_basics#can-i-use-chat-templates-in-training) explaining that chat templates are part of the tokenizer for text-only LLMs or processor for multimodal LLMs to specify how to convert conversations into a single tokenizable string.
- **RAG Code Implementation only needs few lines**: A member asked how many lines of code it takes to implement **RAG** techniques for a company.
   - Another members responded that it only took a *few lines - 15- 30 more or less* and they stored the information in **MongoDB**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lightning.ai/">Lightning AI | Turn ideas into AI, Lightning fast</a>: The all-in-one platform for AI development. Code together. Prototype. Train. Scale. Serve. From your browser - with zero setup. From the creators of PyTorch Lightning.</li><li><a href="https://arxiv.org/abs/2410.05969">Deep neural network-based detection of counterfeit products from smartphone images</a>: Counterfeit products such as drugs and vaccines as well as luxury items such as high-fashion handbags, watches, jewelry, garments, and cosmetics, represent significant direct losses of revenue to legi...</li><li><a href="https://huggingface.co/posts/Reality123b/155118307932581">@Reality123b on Hugging Face: &quot;ok, there must be a problem. HF charged me 0.12$ for 3 inference requests to…&quot;</a>: no description found</li><li><a href="https://tenor.com/view/cat-kawaii-gif-13992966100210966399">Cat Kawaii GIF - CAT Kawaii - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_template_basics#can-i-use-chat-t">Getting Started with Chat Templates for Text LLMs</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_template_basics#can-i-use-chat-templates-in-training">Getting Started with Chat Templates for Text LLMs</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/v3.5.0/en/package_reference/loading_methods#datasets.load_dataset.path">Loading methods</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/v3.5.0/loading">Load</a>: no description found</li><li><a href="https://github.com/huggingface/huggingface_hub/issues/2118">Throttle download speed · Issue #2118 · huggingface/huggingface_hub</a>: Is your feature request related to a problem? Please describe. When downloading a model, huggingface-cli opens many connections and completely maxes out the connection&#39;s bandwidth. Because of this...</li><li><a href="https://discuss.huggingface.co/t/download-speeds-slow-on-the-popular-models/84840/5">Download speeds slow on the popular Models</a>: Same here, most downloads seem to be capping at 10.4MBps today:
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1357094060245913720)** (4 messages): 

> `Hugging Face Token Setup, Jupyter Notebook Configuration, LlamaIndex Basics` 


- **Hugging Face Token Workflow Detailed**: A member requested access to a model, then generated a **HF_TOKEN**, exported it in the terminal before launching **Jupyter**, cloned the agents-course repo to a local branch, and created a **Python venv** called hfenv for Jupyter.
   - They also installed Jupyter and huggingface_hub, added the **hfenv kernel** to Jupyter, selected the kernel in the notebook, ran the notebook, and saved it.
- **Diving into LlamaIndex Fundamentals**: A member is learning the basics of **LlamaIndex** and requested tips on where to go next after looking at the courses.
- **Dependency Discovery**: A member noted the discovery of additional **pip install dependencies** for torch, cli, tensorflow, etc., in the huggingface_hub documentation.
   - They were looking at the docs that someone had referenced earlier.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1357111586971717666)** (11 messages🔥): 

> `Object Detection Model, End-to-End Project, Operating System Events to AI, Game about AI with AI, TypeScript Voice Assistant` 


- ****Signature Detection Model** end-to-end Project**: A member shared an [article](https://huggingface.co/blog/samuellimabraz/signature-detection-model) and [inference server](https://github.com/tech4ai/t4ai-signature-detect-server) for an **end-to-end object detection model** project.
   - It uses **Optuna** for hyperparameter tuning, achieving a **7.94% F1-score improvement** and implements **Triton Inference Server** with an **OpenVINO CPU backend** for optimized inference.
- ****Operating System Events streamed to AI****: There was discussion of a rust lib to stream operating system events to AI and linked to a [GitHub repo](https://github.com/mediar-ai/ui-events/tree/main).
   - One member found it *awesome* and *so useful*.
- ****Game about AI** pops up**: A member made a game about AI with AI, prompting discussion about generating an email digest from an RSS feed.
   - A member suggested using **smolagent** to allow for customized user filtering to generate **podcasts** from the digest, expanding on the initial idea.
- **Meet **TySVA**, the TypeScript Voice Assistant**: A member introduced **TySVA** ([GitHub repo](https://github.com/AstraBert/TySVA)), a TypeScript Voice Assistant leveraging Model Context Protocol (**MCP**) for everyday TypeScript programming tasks.
   - It uses **Qdrant**, **HuggingFace**, **Linkup**, **LlamaIndex**, **ElevenLabs**, **Groq**, **Gradio**, and **FastAPI** to answer user questions with grounded answers and a voice summary of the solution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://idleai.xenovative-ltd.com:5000/">Coming Soon</a>: no description found</li><li><a href="https://github.com/mediar-ai/ui-events/tree/main">GitHub - mediar-ai/ui-events: Library to stream operating system events to AI</a>: Library to stream operating system events to AI. Contribute to mediar-ai/ui-events development by creating an account on GitHub.</li><li><a href="https://github.com/AstraBert/TySVA">GitHub - AstraBert/TySVA: Learn TypeScript chatting effortlessly with AI</a>: Learn TypeScript chatting effortlessly with AI. Contribute to AstraBert/TySVA development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1357217285198184500)** (2 messages): 

> `video-to-3D human mesh reconstruction repos, OWLv2's image-guided-detection mode issue` 


- **Users Seek Latest Video-to-3D Human Mesh Reconstruction Repos**: A member is looking for the latest repos for **video-to-3D human mesh reconstruction** that support inference, after encountering version compatibility issues with older models.
- **Trouble arises with OWLv2's image-guided-detection mode**: A member raised an issue on a tutorial notebook related to **OWLv2's image-guided-detection mode**, struggling to replicate the expected results after several days of troubleshooting, see the [GitHub issue](https://github.com/NielsRogge/Transformers-Tutorials/issues/487).



**Link mentioned**: <a href="https://github.com/NielsRogge/Transformers-Tutorials/issues/487">Issue with OWLv2&#39;s image-guided-detection mode. · Issue #487 · NielsRogge/Transformers-Tutorials</a>: I have tried endless times to recreate the results from the tutorial notebook of https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2...

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1357218120460271687)** (1 messages): 

> `MLX model, Smolagent, AgentGenerationError` 


- **MLX model has issues running with Smolagent**: A member reported an issue using the **MLX model** to run with **Smolagent**, encountering an **AgentGenerationError**.
   - The error indicates *cannot access local variable 'found_stop_sequence' where it is not associated with a value*, according to the attached image.
- **MLX model requires a fix**: The traceback suggests an issue with variable scope related to stop sequences within the **MLX** implementation.
   - A fix is required to properly initialize or handle the `found_stop_sequence` variable before it is accessed during model generation.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1357082209676300412)** (17 messages🔥): 

> `Course Certification, Smart RAG agent, Gradio Version, Project goals` 


- **Will this course give Certificates?**: A member asked if the course will grant a [certification](https://discord.com/channels/879548962464493619/1356866777682022440/1356866777682022440).
   - Another member mentioned that the **deadline** hasn't been clarified, but the schedule has shifted by at least a week, but it is not a hard deadline.
- **Crafting a Smart RAG Agent for eBooks**: A member is building a **Smart RAG agent** for all of their eBooks, technology IP, client documents and emails to test all of the frameworks to find the best one.
   - Project goals include *implementing a multi-agent architecture*, *enabling asynchronous processing*, and *preventing hallucination by mandating citation of sources*.
- **Space Duplication Dilemmas: Gradio Version Woes**: A member encountered an issue with an old **Gradio version** when duplicating a space, which required updating **Gradio** in the requirements.txt file to resolve.
   - The suggested fix involved updating the **requirements.txt** file with specific versions of packages like *pydantic==2.10.6* and *gradio==5.23.1*.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1357078732157751367)** (103 messages🔥🔥): 

> `Debugging MCPs, MCP File System Server, MCP Documentation, MCP Client Implementations, FastMCP vs Low Level` 


- **MCP Debugging Techniques Emerge**: Members discussed debugging techniques for MCPs, revealing that `sendLoggingMessage` can work if [logging is configured during server initialization](https://example.com/initialization).
   - One member noted they had been using `console.error` due to issues with stdout, while others found the inspector lacking, spurring the question of whether a better inspector is being developed.
- **Unveiling Open Source MCP Assistant Server**: A member shared an [open-source MCP EV assistant server](https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py), highlighting its capabilities in managing **EV charging stations**, **trip planning**, and **resource management**.
   - This server aims to provide a comprehensive set of tools and APIs for EV-related services.
- **MCP Client Implements Tool Notifications**: A member highlighted an [MCP client implementation](https://github.com/Abiorh001/mcp_omni_connect) that supports all **notifications**, including subscribing and unsubscribing to resources, useful for `notifications/tools/list_changed`.
   - This client offers seamless integration with **OpenAI models** and supports dynamic tool and resource management across multiple servers.
- **Diving into FastMCP Limitations**: The discussion revealed that **FastMCP** might not support certain features like `subscribe_resource`, leading some to consider using the **low-level server** for more control.
   - Members exchanged code snippets and implementation details for handling resource subscriptions and updates in the low-level server.
- **Authentication headaches with MCP and SSE**: Members debated the best way to pass API keys from an MCP client to an SSE server, but it was noted that [SSE transport doesn't process env within the client](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/).
   - An alternative is using **streaming HTTP** and **OAuth** for more secure auth, but at the price of occasional logins. The current MCP authorization mechanism is OPTIONAL.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/">Authorization</a>:           ℹ️                  Protocol Revision: 2025-03-26      1. Introduction    1.1 Purpose and Scope    The Model Context Protocol provides authorization capabilities at the transport level,enabl...</li><li><a href="https://github.com/Abiorh001/mcp_omni_connect">GitHub - Abiorh001/mcp_omni_connect: MCPOmni Connect is a versatile command-line interface (CLI) client designed to connect to various Model Context Protocol (MCP) servers using stdio transport. It provides seamless integration with OpenAI models and supports dynamic tool and resource management across multiple servers.</a>: MCPOmni Connect is a versatile command-line interface (CLI) client designed to connect to various Model Context Protocol (MCP) servers using stdio transport. It provides seamless integration with O...</li><li><a href="https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py">mcp_ev_assistant_server/ev_assitant_server.py at main · Abiorh001/mcp_ev_assistant_server</a>:  A powerful server implementation for managing Electric Vehicle (EV) charging stations, trip planning, and resource management. This server provides a comprehensive set of tools and APIs for EV-rel...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1357161875212734515)** (10 messages🔥): 

> `Enact Protocol, Shopify MCP, Mobile MCP Server, Semantic Tool Calling, External Registry Idea` 


- ****Enact Protocol** Proposed as MCP's HTTP**: A member introduced the [Enact Protocol](https://github.com/EnactProtocol/specification) as a way to define tools for MCP, comparing it to the HTTP protocol.
   - Another member described it as *a cool way to do semantic tool calling from within a MCP server*.
- ****Shopify-MCP** Rolls Out with Order and Customer Update Support**: The [Shopify-MCP](https://github.com/GeLi2001/shopify-mcp) server now supports order and customer updates, enhancing its utility for MCP clients such as Anthropic's Claude and Cursor IDE.
   - It enables integration with the Shopify API, providing a way to manage Shopify store operations via MCP.
- ****Mobile-Use MCP Server** Launched for Mobile Automation**: The [Mobile-Use MCP Server](https://github.com/runablehq/mobile-mcp) was launched to provide mobile automation capabilities, along with its associated [mobile-use library](https://github.com/runablehq/mobile-use).
   - Users can quickly get started with `npx mobile-mcp install` and use it directly from the Claude Desktop App.
- ****Enact-MCP** Server Implementation Highlighted**: The implementation of the [Enact-MCP server](https://github.com/EnactProtocol/enact-mcp) was shared as a reference for the Enact Protocol.
   - A member noted the absence of a license file and praised the *external registry idea*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EnactProtocol/enact-mcp">GitHub - EnactProtocol/enact-mcp: MCP Server for enact protocol</a>: MCP Server for enact protocol. Contribute to EnactProtocol/enact-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/GeLi2001/shopify-mcp">GitHub - GeLi2001/shopify-mcp: MCP server for Shopify api, usable on mcp clients such as Anthropic&#39;s Claude and Cursor IDE</a>: MCP server for Shopify api, usable on mcp clients such as Anthropic&#39;s Claude and Cursor IDE - GeLi2001/shopify-mcp</li><li><a href="https://github.com/EnactProtocol/specification">GitHub - EnactProtocol/specification: protocol spec</a>: protocol spec. Contribute to EnactProtocol/specification development by creating an account on GitHub.</li><li><a href="https://github.com/runablehq/mobile-mcp">GitHub - runablehq/mobile-mcp: A Model Context Protocol (MCP) server that provides mobile automation capabilities.</a>: A Model Context Protocol (MCP) server that provides mobile automation capabilities. - runablehq/mobile-mcp</li><li><a href="https://github.com/runablehq/mobile-use">GitHub - runablehq/mobile-use: Use AI to control your mobile</a>: Use AI to control your mobile. Contribute to runablehq/mobile-use development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1357067155535040532)** (2 messages): 

> `NotebookLM UX Research, Discover Sources Feature, Google AI summaries` 


- ****NotebookLM** Users Needed for UX Research**: NotebookLM UX Researchers are seeking participants for **60 min 1:1 remote chats** to give feedback on new NotebookLM ideas.
   - Participants will receive a **$100 gift card** as a thank you, and must share a set of notebook sources via Google Drive beforehand and [apply via this form](https://forms.gle/P2t8q36NqbPNSVk8A).
- ****NotebookLM** Interview: Share Your Thoughts, Get Rewarded**: Participants are needed for a **60-minute interview** to provide feedback and must have the ability to upload files to Google Drive.
   - Eligible participants will receive a **$100 gift code** via email from Tremendous, with the interview scheduled for Thursday, April 10, including a **10-minute preparation requirement** beforehand.
- **Discover New Sources with **NotebookLM****: NotebookLM introduces a new feature called *Discover Sources* that allows users to find relevant web content, add to notebook in one click. [Learn more here](https://blog.google/technology/google-labs/notebooklm-discover-sources/).
   - The feature uses **Google AI** to generate summaries and allows you to explore random topics with the *I'm Feeling Curious* button.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.google/technology/google-labs/notebooklm-discover-sources/">New in NotebookLM: Discover sources from around the web</a>: NotebookLM has launched Discover Sources, which lets you add sources from the web to your notebook.</li><li><a href="https://forms.gle/P2t8q36NqbPNSVk8A">Register your interest: NotebookLM feedback</a>: Hello,We are looking for feedback on NotebookLM via a 60 minute remove interview.Your feedback will help the Google team improve NotebookLM. To apply to participate, please fill out this form. If you ...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1357246737919643699)** (4 messages): 

> `Source file transferability, Podcast deep dives, Slideshow presentations` 


- **Users Lament Lack of Source Transferability**: Users are requesting that [source files be transferable](https://notebooklm.google) between folders, arguing that the current read-only and locked-in-place nature is limiting.
   - The user clarified that the sources are user inputs, and expressed their love for the app.
- **Debate Sparks Over Podcast Deep Dive Legality**: A user inquired about the possibility of [uploading NotebookLM's Deep Dive sessions to Spotify](https://spotify.com) as podcasts.
   - Another user speculated that AI-generated audio overviews are likely free from Google copyrights, but stressed that using copyrighted source materials could infringe on original rights; therefore, they advised caution when distributing externally.
- **NoteBookLM generates Slideshow Presentations**: A user seeks advice from others who have used [NoteBookLM to create presentations](https://notebooklm.google).
   - The user seeks tips and tricks for creating slideshows in NoteBookLM.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1357069596146008318)** (98 messages🔥🔥): 

> `NotebookLM 2.5 Pro, Gemini Integration with NotebookLM, Safari Access Issues, Source Transferability, Discover Sources Feature` 


- **Gemini Guru to Guide Google's Gemini**: Josh Woodward will replace Sissie Hsaio as the leader of the Gemini team, preparing for *the next evolution of the Gemini app*, according to [The Verge](https://www.theverge.com/news/642000/google-sissie-hsaio-stepping-down-notebooklm).
- **Safari Snafus Surface for Some NLM Users**: Some users reported issues accessing **NotebookLM** on **Safari** (iPhone/Mac), while others confirmed it's working on iPhone SE (2nd gen) by adding a shortcut to the Home screen.
   - If none of the primary language fixes work, adding `?hl=en` to the end of the URL (like this: `https://notebooklm.google.com/?hl=en`) should fix it.
- **Discover Sources Delight and Debut**: The **Discover Sources** feature is rolling out, expanding research capabilities beyond user-known information and mentioning interesting related topics, but is not yet available to everyone.
   - A user has suggested that academic online sources should be included, just like in perplexity.
- **Mind Map Maneuvering Mayhem**: Users noted that jumping from a mind map node to the chat area and back closes all nodes, requiring re-navigation, which the team is aware of.
- **Language Localization Lapses Leave Users Lugubrious**: Users are reporting that **changing the language** using the toggle in settings of the UI does nothing, and that **English** needs to be the primary language for the **Google** account.
   - Appending `?hl=en` to the end of the URL should also fix it.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/news/642000/google-sissie-hsaio-stepping-down-notebooklm">Google’s NotebookLM leader is taking over as head of the Gemini app</a>: Google is shuffling its AI team.</li><li><a href="https://notebooklm.google.com/?hl=en`,">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1357075150021070981)** (70 messages🔥🔥): 

> `Ace Computer Autopilot Launch, YourBench Open Source Benchmarking Tool, Model Context Protocol Memory Implementation, RabbitOS Intern, Llama 4 Image Generation` 


- **Ace-ing the Autopilot Game**: [General Agents Co](https://x.com/sherjilozair/status/1907478704223297576) launched **Ace**, a realtime computer autopilot that performs tasks using the mouse and keyboard at superhuman speeds.
   - Unlike a chatbot, Ace is designed to execute tasks directly on a computer.
- **Benchmarking Bonanza with YourBench**: [YourBench](https://x.com/sumukx/status/1907495423356403764) launched **YourBench**, an open-source tool for custom benchmarking and synthetic data generation from any documents.
   - YourBench aims to improve model evaluations by providing a custom evaluation set and leaderboard.
- **Llama 4 Leaps into Image Generation**: **Llama 4** is rolling out image generation and editing capabilities in messages.
   - Users noted that edits were very fast, citing *1 second edits versus 5 minutes for gpt-4o*.
- **Scale AI Soars to $25B Valuation**: **Scale AI** is projected to reach **$2B** in revenue this year, leading to a tender offer valuing the company at **$25B**.
   - Revenue last year was **$870M**
- **A16Z Assembles AI Workstation**: A16Z built an **8x RTX 4090 GPU AI workstation** from scratch, compatible with the new **RTX 5090** with **PCIe 5.0**, for training, deploying, and running AI models locally.
   - They released a [full guide](https://x.com/Mascobot/status/1907899937838301311) on how to build your own.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sumukx/status/1907495423356403764">Tweet from Sumuk (@sumukx)</a>: we&#39;re launching 🤗 yourbench today, an open source tool for custom benchmarking and synthetic data generation from ANY of your documents. it&#39;s a big step towards improving how model evaluation...</li><li><a href="https://x.com/sumukx/status/1907495423356403764]">Tweet from Sumuk (@sumukx)</a>: we&#39;re launching 🤗 yourbench today, an open source tool for custom benchmarking and synthetic data generation from ANY of your documents. it&#39;s a big step towards improving how model evaluation...</li><li><a href="https://fxtwitter.com/sumukx/status/1907495423356403764">Tweet from Sumuk (@sumukx)</a>: we&#39;re launching 🤗 yourbench today, an open source tool for custom benchmarking and synthetic data generation from ANY of your documents. it&#39;s a big step towards improving how model evaluation...</li><li><a href="https://discordapp.com/channels/822583790773862470/1337560058288017528">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://x.com/OpenRouterAI/status/1907867881930633666]">Tweet from OpenRouter (@OpenRouterAI)</a>: A stealth model has entered the chat... 🥷</li><li><a href="https://x.com/TheXeophon/status/1907880330985390215">Tweet from Xeophon (@TheXeophon)</a>: Here is the new stealth model on my vibe check. It is now the best non-thinking model (at least it has no thinking tokens...). The outputs are super short, it loves Certainly! and listicles. Super int...</li><li><a href="https://x.com/hingeloss/status/1907470138321858712?s=46">Tweet from chris (@hingeloss)</a>: Llama 4 based image generation and editing beginning to roll out, looks very good -- and very fast, 1 second edits versus 5 minutes for gpt-4oDid Meta cook??</li><li><a href="https://x.com/Mascobot/status/1907899937838301311]">Tweet from Marco Mascorro (@Mascobot)</a>: 🚨 New: We @a16z built an 8x RTX 4090 GPU AI workstation from scratch —compatible with the new RTX 5090 with PCIe 5.0, for training, deploying, and running AI models locally— so you don’t have to. Her...</li><li><a href="https://fxtwitter.com/Mascobot/status/1907899937838301311">Tweet from Marco Mascorro (@Mascobot)</a>: 🚨 New: We @a16z built an 8x RTX 4090 GPU AI workstation from scratch —compatible with the new RTX 5090 with PCIe 5.0, for training, deploying, and running AI models locally— so you don’t have to. Her...</li><li><a href="https://x.com/OpenRouterAI/status/1907867881930633666">Tweet from OpenRouter (@OpenRouterAI)</a>: A stealth model has entered the chat... 🥷</li><li><a href="https://x.com/kateclarktweets/status/1907551168143774004?s=46">Tweet from Kate Clark (@KateClarkTweets)</a>: Scale AI generated $870M in revenue last year and projects $2B this year. Plus, Coatue, Founders Fund, and Greenoaks are participating in a tender offer expected to value the company at $25B. Scoop w/...</li><li><a href="https://x.com/kateclarktweets/status/1907551168143774004?s=46]">Tweet from Kate Clark (@KateClarkTweets)</a>: Scale AI generated $870M in revenue last year and projects $2B this year. Plus, Coatue, Founders Fund, and Greenoaks are participating in a tender offer expected to value the company at $25B. Scoop w/...</li><li><a href="https://fxtwitter.com/clefourrier/status/1907496576274088070">Tweet from Clémentine Fourrier 🍊 (@clefourrier)</a>: Know which model is the best for your use case in less than 5 min, no matter the topic!Document -&gt; custom made evaluation set -&gt; leaderboardHt to @sumukx & @ailozovskaya !Quoting Sumuk (@sumukx)...</li><li><a href="https://fxtwitter.com/kateclarktweets/status/1907551168143774004">Tweet from Kate Clark (@KateClarkTweets)</a>: Scale AI generated $870M in revenue last year and projects $2B this year. Plus, Coatue, Founders Fund, and Greenoaks are participating in a tender offer expected to value the company at $25B. Scoop w/...</li><li><a href="https://fxtwitter.com/OpenRouterAI/status/1907867881930633666">Tweet from OpenRouter (@OpenRouterAI)</a>: A stealth model has entered the chat... 🥷</li><li><a href="https://x.com/Mascobot/status/1907899937838301311">Tweet from Marco Mascorro (@Mascobot)</a>: 🚨 New: We @a16z built an 8x RTX 4090 GPU AI workstation from scratch —compatible with the new RTX 5090 with PCIe 5.0, for training, deploying, and running AI models locally— so you don’t have to. Her...</li><li><a href="https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/">Devin 2.0 is here: Cognition slashes price of AI software engineer to $20 per month from $500</a>: Devin attracted interest from enterprise customers seeking to incorporate autonomous coding agents into their software development processes.</li><li><a href="https://fxtwitter.com/TheXeophon/status/1907880330985390215">Tweet from Xeophon (@TheXeophon)</a>: Here is the new stealth model on my vibe check. It is now the best non-thinking model (at least it has no thinking tokens...). The outputs are super short, it loves Certainly! and listicles. Super int...</li><li><a href="https://www.dropbox.com/scl/fo/dabegjgxb1ymtlnqopzro/AGKKb-jXT_4oODKO">Tweet from Dropbox</a>: no description found</li><li><a href="https://www.semafor.com/article/04/02/2025/google-gemini-shakes-up-ai-leadership-sissie-hsiao-steps-down-replaced-by-josh-woodward">Google Gemini is shaking up its AI leadership ranks	</a>: Sissie Hsiao, who led the Gemini chatbot project after the launch of ChatGPT, will step down. She’s being replaced by Josh Woodward, who heads Google Labs.</li><li><a href="https://arstechnica.com/gadgets/2025/04/google-shakes-up-gemini-leadership-google-labs-head-taking-the-reins/">Google shakes up Gemini leadership, Google Labs head taking the reins</a>: With fresh leadership, Google aims to create new products based on Gemini.</li><li><a href="https://x.com/hingeloss/status/1907470138321858712?s=46]">Tweet from chris (@hingeloss)</a>: Llama 4 based image generation and editing beginning to roll out, looks very good -- and very fast, 1 second edits versus 5 minutes for gpt-4oDid Meta cook??</li><li><a href="https://fxtwitter.com/hingeloss/status/1907470138321858712">Tweet from chris (@hingeloss)</a>: Llama 4 based image generation and editing beginning to roll out, looks very good -- and very fast, 1 second edits versus 5 minutes for gpt-4oDid Meta cook??</li><li><a href="https://x.com/sherjilozair/status/1907478704223297576">Tweet from Sherjil Ozair (@sherjilozair)</a>: Today I&#39;m launching my new company @GeneralAgentsCo and our first product.Introducing Ace: The First Realtime Computer AutopilotAce is not a chatbot. Ace performs tasks for you.On your computer. U...</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/memory">servers/src/memory at main · modelcontextprotocol/servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://www.dropbox.com/scl/fo/dabegjgxb1ymtlnqopzro/AGKKb-jXT_4oODKOjCxJr9A?rlkey=ze30fqgc00trhwk1gztt21zf0&e=1&st=gqzm7gq1&dl=0">no title found</a>: no description found</li><li><a href="https://anthropic.swoogo.com/codewithclaude">Code with Claude Apply</a>: no description found</li><li><a href="https://ai-2027.com/">AI 2027</a>: A research-backed AI scenario forecast.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1357183581658677259)** (4 messages): 

> `June Ramp Up, Model Context Protocol (MCP), AI Engineer World's Fair 2025, MCP vs OpenAPI` 


- **Latent Space Ramps Up for June**: Latent Space is starting to ramp up for June, encouraging users to opt into the new <@&1335734932936458281> Role in their Discord onboarding for updates and sharing plans in <#1344427813905891378>.
   - A new pod, **Creators of Model Context Protocol (MCP)**, has been announced with @dsp_ and @jspahrsummers, with topics including *the Origin Story of MCP*, *MCP vs OpenAPI*, *Building Agents with MCP*, and *Open source Governance*.
- **MCP Track Announced for AI Engineer World's Fair 2025**: A dedicated **MCP track** will be at the [2025 AI Engineer World's Fair](https://ti.to/software-3/ai-engineer-worlds-fair-2025), taking place **Jun 3rd to 5th in San Francisco**, where the MCP core team and major contributors and builders will be meeting.
   - Attendees are encouraged to [apply to speak](https://sessionize.com/ai-engineer-worlds-fair-2025) or [sponsor](mailto:sponsors@ai.engiener).
- **MCP Wins the Agent Standard Wars?**: According to [Why MCP Won](https://www.latent.space/p/why-mcp-won), **OpenAI** and **Google** have announced **MCP support**.
   - The announcement effectively confirms the prediction that MCP was the presumptive winner of the agent standard wars and has now overtaken [OpenAPI](https://github.com/OAI/OpenAPI-Specification) in GitHub stars.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1907843005429817481">Tweet from Latent.Space (@latentspacepod)</a>: 🆕 The Creators of Model Context Protocolwith @dsp_ and @jspahrsummers!https://latent.space/p/mcpWe asked ALL your burning questions:- The Origin Story of MCP- MCP vs OpenAPI- Building Agents with MCP...</li><li><a href="https://www.latent.space/p/mcp">The Creators of Model Context Protocol</a>: MCP&#x27;s coauthors on the origin, challenges and future of the protocol.
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1357142468432236625)** (52 messages🔥): 

> `UX/UI Competition, AI UI Layout Generation, GPT-4o Behavior, GPT-5 Unified Model, DeepSeek Hype` 


- **Superior UX/UI Steals the Show**: Members discuss that winning startups often have better **UX/UI**, noting current products lack a *winning sauce*.
   - One user emphasizes developing a UI that defines requirements and page layouts, suggesting an agent swarm to generate web components in parallel, showcased in a [screen recording](https://cdn.discordapp.com/attachments/986699377257119794/1357190780258746429/Screen_Recording_2025-04-03_at_1.39.26_pm.mov?ex=67eff6a9&is=67eea529&hm=9a8e202a73469a0749a23b81496240fd68a93a295583b0ce34cf52ff80c0c03e&).
- **Automated Wireframing Dreams**: One member is aiming to skip wireframing/design steps, and linked to a [Dribbble design](https://dribbble.com/shots/25708347-Delivery-Web-App-Design) for a web app dedicated to package tracking.
   - Another member wants a layout generator that designs grayscale wireframes, then refines and fills them with web components, using a swarm of agents.
- **GPT-4o Gets Quirky**: Users noted **GPT-4o** exhibiting unusual behavior, such as assuming a persona and adding parenthetical comments in its responses, illustrated in a [screenshot](https://cdn.discordapp.com/attachments/986699377257119794/1357335757676871711/image.png?ex=67efd4ee&is=67ee836e&hm=4deb85a208466f212d88e7b77771776834fe28524ac15dc9c5dbcb1be3301ff3&).
   - Speculation arose regarding the source of this behavior, with theories ranging from an *EQ dataset* used in SFT to emergent properties; users also report GPT-4o becoming slower.
- **Google's Gemini 2.5 Claims Continuous Pretraining**: A member noted **Gemini 2.5** claims to be *continuously pretrained without a sharp knowledge cutoff*
   - The member pondered if this meant incremental model releases (2 -> 2.5 -> 3) or if it was a hallucination.
- **Dynamic NLP Systems: The Next Frontier**: A user believes the next generation of architectures will be dynamic, possessing both short and long-term memories.
   - They argued that NLP should treat language as a structured, evolving signal with flowing meaning rather than relying on a rigid tokenized format.



**Link mentioned**: <a href="https://dribbble.com/shots/25708347-Delivery-Web-App-Design">Delivery Web App Design</a>: no description found

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1357144717208457386)** (4 messages): 

> `LLMs struggle with math, LLMs overestimating themselves` 


- **LLMs Bomb on USA Math Olympiad**: A member shared a [paper](https://arxiv.org/abs/2503.21934v1) evaluating state-of-the-art LLMs on the **2025 USA Mathematical Olympiad (USAMO)**, where models like **O3-MINI** and **Claude 3.7** achieved less than **5%** on **six proof-based math problems**.
   - Each problem was scored out of **7 points**, with a max total score of **42**, and the models were trained on all imaginable math data, including **IMO problems**, **USAMO archives**, **textbooks**, and **papers**.
- **LLMs Grade Inflated on Math Tests**: The same models, including **O3-MINI** and **Claude 3.7**, overestimated their scores when grading their own work, inflating them by up to **20x** compared to human graders.
   - A discussion ensued on the implications of these findings, given the extensive training these models underwent with access to vast amounts of mathematical data.



**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1joqnp0/top_reasoning_llms_failed_horribly_on_usa_math/">Reddit - The heart of the internet</a>: no description found

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1357079886925140038)** (2 messages): 

> `Gemini App, Dream 7B` 


- **Google Releases Gemini App**: Google released the [Gemini App](https://vxtwitter.com/GeminiApp/status/1906131622736679332) showcasing its latest advancements in AI.
- **Dream 7B diffusion model released**: HKU-NLP and Huawei Noah’s Ark Lab released **Dream 7B**, a powerful open diffusion large language model that outperforms existing diffusion language models and matches or exceeds top-tier Autoregressive (AR) language models of similar size.
   - According to their [blogpost](https://hkunlp.github.io/blog/2025/dream/), Dream 7B demonstrates *strong planning ability and inference flexibility that naturally benefits from the diffusion modeling.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vxtwitter.com/GeminiApp/status/1906131622736679332">Tweet from undefined</a>: no description found</li><li><a href="https://hkunlp.github.io/blog/2025/dream/">Dream 7B | HKU NLP Group </a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1357221283338649712)** (3 messages): 

> `OpenAI /v1/chat/completions API, conversation history, /v1/responses API, stateful vs stateless APIs` 


- **OpenAI's `/v1/chat/completions` API Costs Explained**: A member stated that with OpenAI's `/v1/chat/completions` API, you must send the complete conversation history with each new prompt, as described in the [OpenAI Documentation](https://platform.openai.com/docs/guides/conversation-state?api-mode=chat).
   - They also said that even if the input tokens' KV cache isn't evicted, you're still charged for those input tokens.
- **Stateful API alternative coming: `/v1/responses`**: The member pointed out that the newer `/v1/responses` API will be stateful, allowing reference to past messages via IDs, according to the [Responses vs Chat Completions documentation](https://platform.openai.com/docs/guides/responses-vs-chat-completions).
   - The new API will contrast with the `/v1/chat/completions` API, which is stateless and requires manual resending of the entire chat history.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1357262355519770645)** (8 messages🔥): 

> `cudaMemcpyAsync Overlap, cuBLAS matmul low occupancy, Registers in CUDA` 


- **Question on Overlapping `cudaMemcpyAsync` Copies**: A member inquired whether it's possible to overlap host to device copies when using `cudaMemcpyAsync` with pinned memory and separate streams in separate CPU threads.
   - The member noted having **5 copy engines** but expressed uncertainty if this enables overlapping copies.
- **`cuBLAS` Occupancy stays low due to latency hiding**: A member reported a low occupancy of around **20%** during a `cuBLAS` matmul of very large matrices in `nsys` and inquired why it wasn't higher.
   - Another member clarified that *high occupancy is needed to hide latencies* and that the `cuBLAS` code is written such that few warps are enough to saturate the arithmetic units of the GPU, with memory access latencies hidden at the software level.
- **CUDA Registers and Threads detailed**: A member asked that if an instruction computes a tile of **64x256**, shouldn't be needed **128 registers per thread**?
   - A member was watching a video where it was mentioned that **256 registers per thread** are needed for this operation.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1357321471956877484)** (1 messages): 

> `LLM Profiling, PyTorch Profilers, Perfetto Crashing, Trace Processor` 


- **LLM Profiling Pointers Popped**: A member asked for tips on profiling an **LLM (32B params)** from transformers using **PyTorch profilers**.
   - The member reported that the **Chrome traces** were **2.5GB** and that **Perfetto** kept crashing when attempting to open them.
- **Trace Processor Triumphs Trace Troubles**: The same member found a solution to the **Perfetto** crashing issue.
   - The solution was to use the `trace_processor` locally with **Perfetto**, as described in the [Perfetto documentation for large traces](https://perfetto.dev/docs/visualization/large-traces).


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1357243973080711269)** (2 messages): 

> `AMD talk on TunableOp, NVIDIA pre-tuning in CuBLAS, NVSHMEM-based kernels for MoE models` 


- **AMD's TunableOp: PyTorch's Auto-Tune Gambit**: AMD introduced **TunableOp** in [PyTorch](https://pytorch.org/docs/stable/cuda.tunable.html), a prototype feature that allows users to select the fastest implementation for operations like GEMMs, potentially using different libraries or techniques.
   - The feature is enabled separately from the tuning phase, and aims to optimize performance across various hardware configurations.
- **NVIDIA Bakes the Best: Pre-tuned CuBLAS Dominance**: A member noted that NVIDIA already pre-tunes everything and bakes it into **CuBLAS**, potentially giving it an edge over AMD's more configurable approach.
   - The pre-tuning might be less optimized for consumer GPUs but still provides a solid baseline.
- **NVSHMEM Kernels supercharge Mixture-of-Experts**: **PPLXDevs** announced [custom NVSHMEM-based kernels](https://x.com/pplxdevs/status/1907547685579796933?s=46) for **Mixture-of-Experts (MoE)** models, promising up to **10x faster communication** compared to standard all-to-all operations.
   - Their approach balances performance with adaptability across different hardware configurations, making it a potentially valuable tool for MoE model development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/pplxdevs/status/1907547685579796933?s=46">Tweet from Perplexity Developers (@PPLXDevs)</a>: We&#39;ve built custom NVSHMEM-based kernels for Mixture-of-Experts (MoE) models that deliver up to 10x faster communication than standard all-to-all operations.Our approach balances performance with ...</li><li><a href="https://pytorch.org/docs/stable/cuda.tunable.html">TunableOp &mdash; PyTorch 2.6 documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1357175801811964027)** (3 messages): 

> `Activation Checkpointing, CUDA Compilation, C vs C++ in CUDA` 


- **Activation Checkpointing during Inference**: A member inquired about **activation checkpointing** during inference and shared a [paper link](https://arxiv.org/pdf/2501.01792) for reference.
   - He mentioned seeing the reference and **CUDA compilation** image before, but is still trying to understand the concept.
- **CUDA Compiler Magically Interprets C and C++**: A member expressed surprise that the **CUDA compiler** can deduce whether code written in a `.cu` file is **C or C++** and compile accordingly.
   - He noted that starting to write in either **C or C++** within a `.cu` file leads the compiler to *deduce the code as C code and starts compiling Cuda as usual*.
- **C/C++ Compatibility within CUDA Compilation**: A member explained that **C++** is syntactically close to being a superset of **C**, and the `nvcc` compiler bridges additional gaps.
   - He questioned whether the code is truly compiled as **C code** (using **C linkage and symbol names**) or if the **C code** is simply also valid **C++ code**.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1357285556932837497)** (3 messages): 

> `FP8 Training, Optimizer Configuration, Model Size Impact, torch.compile Usage, GEMM size requirements` 


- **FP8 Training Implementation Troubleshoot**: A member inquired about implementing **FP8 training** using the recipe from [pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/float8) on a single GPU, and whether any specific optimizer configurations were needed.
   - They reported seeing limited speed improvements compared to BF16 and speculated whether the model size might be a factor.
- **GEMM Sizes Determine Float8 Speedups**: A member pointed to the performance details in [pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/float8#performance), noting that the **M, K, N dimensions of GEMMs** must be sufficiently large for **FP8** to provide noticeable speedups.
   - They suggested that the provided chart, although from a microbenchmark, gives a reasonable estimate of the shapes required and also asked if the user was using **torch.compile**.
- **TorchAO Provides Custom Optimizers**: A member suggested checking out the [optimizers in pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/optim) for potential solutions related to optimizer configuration.
   - No additional details were provided.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/tree/main/torchao/float8#performance">ao/torchao/float8 at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/optim">ao/torchao/optim at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1357377368712282415)** (3 messages): 

> `Code Correctness Issues, Assembly Differences` 


- ****Code Correctness Quandaries****: A member is *unsure why* their code has correctness issues, noting that *none of the values are written back correctly* as can be seen in this [screenshot](https://cdn.discordapp.com/attachments/1233704710389764236/1357383569181511872/image.png?ex=67f00175&is=67eeaff5&hm=47eb7052bf909c85157cec137bae54ce7b7031bc076ea77b31d987e942bdf6b9&).
   - They found that when *expanded into the version below, it works correctly*, and is looking for insights.
- ****Assembly Audit Asked****: A member responded that the code *looks correct* and suggested checking the difference in assembly.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1357124390600446013)** (2 messages): 

> `Blackwell Architecture, ThunderKittens Kernels, CTA pairs on Blackwell` 


- **ThunderKittens Launch Blackwell Kernels!**: The HazyResearch team has released new **BF16** and **FP8 ThunderKittens GEMM kernels** for the **NVIDIA Blackwell architecture**, claiming speeds at or near **cuBLAS**.
   - These kernels leverage new features like **5th-generation tensor cores**, **Tensor Memory**, and **CTA pairs**, integrating them into TK's existing tile-based abstractions, according to their [blog post](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell).
- **CTA Pair Placement on Blackwell SMs**: The discussion revolves around whether **CTA (Cooperative Thread Array) pairs** on **NVIDIA Blackwell GPUs** are scheduled on the same **SM (Streaming Multiprocessor)** or across two SMs.
   - Based on an [attached image from Nvidia's GTC 2025 talk](https://cdn.discordapp.com/attachments/1300872762163728550/1357124390348783806/p.png?ex=67efb8d4&is=67ee6754&hm=fe9264e7207290f878ec5eded945a3afe71f2c92e5fbcfc77be6f913fe858b55&), the analysis indicates that the CTA pair is scheduled *across two SMs in a cluster*.



**Link mentioned**: <a href="https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell">ThunderKittens Now on Blackwells!</a>: no description found

  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1357107705025925270)** (7 messages): 

> `Datasets, Curricula, RGBench, Knight Swap, Puzzle2` 


- **Reasoning Gym Datasets Get Curriculum Fixes**: A member opened a PR ([#407](https://github.com/open-thought/reasoning-gym/pull/407)) to fix the **curricula** of all **datasets** in the [reasoning-gym](https://github.com/open-thought/reasoning-gym) project to be more sensible, updating the tests and adding missing curricula.
   - The PR involves reviewing all datasets twice to set more sensible values for the curricula and implementing missing curricula like **Knight Swap** and **Puzzle2**.
- **Reasoning Gym adds Easy, Medium and Hard interfaces**: A member inquired about an interface for **easy, medium, hard** difficulties, similar to **RGBench**, for the **reasoning-gym**, so users can set it manually.
   - The difficulty levels can be extracted into separate YAML files and reused for other tasks.
- **Reasoning Gym Medium Level Settings Exposed**: A member shared a link to what is considered a **medium** difficulty setting for each task in the [reasoning-gym](https://github.com/open-thought/reasoning-gym/blob/5b4aa313819a9a6aecd6034b8c6394b6e4251438/eval/yaml/medium/claude-3.5-sonnet.yaml).
   - The link contains the curated levels which they believe to be medium for each task.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/open-thought/reasoning-gym/blob/5b4aa313819a9a6aecd6034b8c6394b6e4251438/eval/yaml/medium/claude-3.5-sonnet.yaml">reasoning-gym/eval/yaml/medium/claude-3.5-sonnet.yaml at 5b4aa313819a9a6aecd6034b8c6394b6e4251438 · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/407">fix(curriculum): Make boundaries in curriculum more sensible by zafstojano · Pull Request #407 · open-thought/reasoning-gym</a>: Overview:I&amp;#39;ve went over all datasets twice in order to set some more sensible values for the curricula.Moreover, I&amp;#39;ve implemented a couple of the missing curriculas (Knight Swap, Puzzl...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1357358423980769391)** (4 messages): 

> `Grayscale Leaderboard Submissions, Modal Runners Success` 


- **Grayscale Leaderboard Sees Submissions Flood In**: Multiple successful leaderboard submissions were made to the `grayscale` leaderboard, using **Modal runners** on various GPUs.
   - The submissions included IDs **3433, 3434, 3436, and 3437**, tested on GPUs such as **L4, T4, A100, and H100**.
- **Modal Runners Prove Successful Across Multiple GPUs**: Submissions to the `grayscale` leaderboard were successfully executed using **Modal runners** on different GPU configurations.
   - GPUs utilized included **L4, T4, A100, and H100**, indicating broad compatibility.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1357106432096342148)** (28 messages🔥): 

> `Quantity struct, Dimensions ** power, IntLiteral vodoo XD, normlisation, Python wrappers for Mojo` 


- **Crafting Quantities with Dimensions in Mojo**: A member shared code snippets defining **aliases for physical quantities** like `Velocity`, `Acceleration`, and `Newton` using a `Quantity` struct with `Dimensions`.
   - Another member pointed to their [Kelvin library on GitHub](https://github.com/bgreni/Kelvin/blob/main/kelvin/quantity.mojo#L55-L125), highlighting the intricate work required to get `Dimensions ** power` to function correctly.
- **`IntLiteral` Strikes Again!**: A member mentioned having to use *cursed* `IntLiteral` tricks to bypass dynamic value issues when defining `Quantity`.
   - Another member praised the use of `IntLiteral` for encoding arbitrary information into the type system, while another joked about the user's *horrendous `IntLiteral` vodoo XD*.
- **Max's Duration struct**: One member referenced a pull request to modular/max, specifically a proposal for a **Duration struct inspired by std::chrono::duration** from the C++ stdlib, available [on GitHub](https://github.com/modular/max/pull/4022#issuecomment-2694197567).
   - He stated he's getting close to achieving the *wishful thinking* code snippet referenced in the GitHub issue.
- **A call for Mojo's Python interop**: A member inquired about the status of **Python wrappers for Mojo**, and the ability to call Mojo from CPython.
   - Another user replied that it would be a 🔥 feature.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/bgreni/Kelvin/blob/main/kelvin/quantity.mojo#L55-L125">Kelvin/kelvin/quantity.mojo at main · bgreni/Kelvin</a>: Contribute to bgreni/Kelvin development by creating an account on GitHub.</li><li><a href="https://github.com/modular/max/pull/4022#issuecomment-2694197567">[stdlib][proposal] Duration module proposal by bgreni · Pull Request #4022 · modular/max</a>: A proposal for a Duration struct inspired by std::chrono::duration from the C++ stdlib
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1357257296471920700)** (4 messages): 

> `Checkpoint Conversions, HF Checkpoint Format, tune_to_hf function` 


- **Torchtune Checkpoints Get HuggingFace Treatment**: Members discussed how to convert a **torchtune checkpoint** to **HF checkpoint format**.
   - One member suggested checking the *huggingface checkpointer*, recommending the **tune_to_hf function**.
- **HuggingFace Checkpointer to the rescue**: The **HuggingFace checkpointer** can be used to convert torchtune checkpoints.
   - Specifically, the **tune_to_hf function** can be leveraged for this conversion.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1357121407493869732)** (19 messages🔥): 

> `vLLM memory sharing with Unsloth, GRPO Upstream contributions, Torchtune hanging with certain sequence lengths, Packed Datasets` 


- **Unsloth shares VRAM with vLLM**: A member mentioned that in [Unsloth](https://github.com/unslothai/unsloth), they managed to use the same vRAM for **vLLM** and the training procedure, without fully understanding how it was possible.
   - They also thought that using the verb `train` in a flag for masking could be confusing in a validation configuration.
- **Ariel offers GRPO Upstream Contributions**: A member offered to push changes from their "internal" **GRPO** upstream, including in-process **vLLM** integration, in-training evals, and more flexible **RL** data handling.
   - Another member responded that they have vllm integration in the async version and a PR for validation dataset that is almost ready to be merged, but haven't thought about the multidataset scenario/reporting multiple losses.
- **Torchtune's timeout bug hits**: A member reported an *AMAZING bug* where **Torchtune** hangs and crashes by a timeout if some (but not all) microbatches has **seq length 7/14/21/28/35/42/49** and created [an issue](https://github.com/pytorch/torchtune/issues/2554).
   - The member was thankful that *torchtune dataloader still has seed: null bug not being random*, otherwise they wouldn't be able to catch it.
- **Packed Datasets to the rescue**: In response to the hanging issue, a member suggested using packed dataset, as it should be faster and would never have a **seqlen=49**.
   - Another member said that *if it's a way to go, standard recipes should be updated, because we observed the issue on basic torchtune examples*.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/issues/2554">Chunked output causes timeout crash on certain seq len · Issue #2554 · pytorch/torchtune</a>: TL;DR If one of dataloader batches is 49 tokens long, torchtune crashes on timeout Longer explanation chunked_output in transformer.py splits output into a list of 8 tensors. If output of length 49...

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1357108354714959974)** (4 messages): 

> `Dream 7B, Diffusion Language Models, Huawei Noah’s Ark Lab` 


- **Dream 7B Released as Powerful Diffusion Model**: The University of Hong Kong and Huawei Noah’s Ark Lab jointly released **Dream 7B**, a new open diffusion large language model, as detailed in [this blog post](https://hkunlp.github.io/blog/2025/dream/).
   - It reportedly *outperforms existing diffusion language models by a large margin* and matches or exceeds top-tier Autoregressive language models of similar size on general, math, and coding abilities.
- **Dream 7B shows Planning Prowess**: **Dream 7B** demonstrates planning ability and inference flexibility that naturally benefits from the diffusion modeling, according to [the release](https://hkunlp.github.io/blog/2025/dream/).



**Link mentioned**: <a href="https://hkunlp.github.io/blog/2025/dream/">Dream 7B | HKU NLP Group </a>: no description found

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1357081337047290066)** (14 messages🔥): 

> `Diagram Creation Tools, DeTikZify, Gradient Accumulation, GitHub MCP event` 


- **Diagram Tools Debate Rages!**: Members discussed various diagram creation tools with recommendations for **Inkscape** for 'rawdogging' it, and **draw.io** otherwise.
   - Others suggested using **pure TikZ**, with one user joking that alternatives are fraudulent.
- **DeTikZify Tool Synthesizes Graphics Programs**: A member shared a link to a new tool called [DeTikZify](https://github.com/potamides/DeTikZify) for synthesizing graphics programs for scientific figures and sketches with **TikZ**.
   - The user hadn't tried it yet but requested feedback from the group if anyone found it useful.
- **Gradient Accumulation: More Than Meets the Eye**: A member pointed out that gradient accumulation has uses beyond the obvious.
   - Another member sarcastically commented that *all of pipeline parallelism is gradient accumulation*.
- **GitHub Co-hosting MCP Event in SF**: GitHub is co-hosting an **MCP Demo Night** event in San Francisco which will focus on the intersection of **AI**, incident response, and platform engineering; more details can be found at [lu.ma/9wi116nk](https://lu.ma/9wi116nk).
   - The event promises lightning demos, a **Future of AI Panel**, fireside chats, and networking opportunities for developers, platform engineers, and AI leaders.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/9wi116nk">MCP’s and The Future of Developer Tools &amp; AI-Driven Reliability · Luma</a>: Join us for an evening at the forefront of developer experience and infrastructure innovation. MCP Demo Night is where cutting-edge MCP tools meet the Future…</li><li><a href="https://github.com/potamides/DeTikZify">GitHub - potamides/DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ</a>: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ - potamides/DeTikZify
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1357434825967010064)** (1 messages): 

> `OpenThoughts-1M, OpenThinker2-32B, OpenThinker2-7B, R1-Distilled-32B, Qwen 2.5 32B` 


- **OpenThinker2: SOTA Open-Data Reasoning Models Debut**: Ludwig Schmidt and team released the **OpenThoughts-1M** dataset, along with **OpenThinker2-32B** and **OpenThinker2-7B** models, surpassing **R1-Distilled-32B** using only SFT on **Qwen 2.5 32B Instruct**; details in their [blog post](https://www.openthoughts.ai/blog/thinkagain).
- **OpenThoughts2-1M: Dataset Details Revealed**: The **OpenThoughts2-1M** dataset builds upon **OpenThoughts-114k**, integrating datasets like [OpenR1](https://huggingface.co/open-r1) and additional math/code reasoning data as described in the [dataset card](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M).
- **OpenThinker2 Models: SFT Outperforms DeepSeekR1-32B**: According to [Etash Guha's tweet](https://x.com/etash_guha/status/1907837107793702958), **OpenThinker2-32B** and **OpenThinker2-7B** outperform **DeepSeekR1-32B** with just SFT on open data, using a dataset curated for quality instructions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.openthoughts.ai/blog/thinkagain">Outperforming DeepSeekR1-32B with OpenThinker2</a>: Announcing the next iteration of our open reasoning models and datasets.</li><li><a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M">open-thoughts/OpenThoughts2-1M · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/etash_guha/status/1907837107793702958">Tweet from Etash Guha (@etash_guha)</a>: Turns out, it’s possible to outperform DeepSeekR1-32B with only SFT on open data and no RL: Announcing OpenThinker2-32B and OpenThinker2-7B. We also release the data, OpenThoughts2-1M, curated by sele...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1357408635008843796)** (9 messages🔥): 

> `Combining linear probes, Steering vector composition, Contrastive sample selection` 


- **Probe Combination Queries Inspire Discussion**: A member asked about combining linear probes or steering vectors for curated sets of positive/negative examples, questioning if joint training could minimize crosstalk or interference.
   - Another member suggested that steering vectors should behave as vectors due to the axioms of vector spaces, recommending a search on the **linear representations hypothesis**.
- **Steering Vectors Prove Unreliable**: A member shared the paper [Steering Vectors: Reliability and Generalisation](https://arxiv.org/abs/2407.12404), showing that **steering vectors have limitations** both in- and out-of-distribution.
   - The paper notes that *steerability is highly variable across different inputs* and can be brittle to changes in the prompt.
- **Dynamic Composition of Steering Vectors Explored**: A member shared their work on [steering vector composition](https://aclanthology.org/2024.blackboxnlp-1.34/) showing success with pairs of unrelated properties like language and formality/safety using **Dynamic Activation Composition**.
   - Their information-theoretic approach modulates steering intensity to maintain high conditioning while minimizing the impact on generation fluency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.12404">Analyzing the Generalization and Reliability of Steering Vectors</a>: Steering vectors (SVs) have been proposed as an effective approach to adjust language model behaviour at inference time by intervening on intermediate model activations. They have shown promise in ter...</li><li><a href="https://aclanthology.org/2024.blackboxnlp-1.34/">Multi-property Steering of Large Language Models with Dynamic Activation Composition</a>: Daniel Scalena, Gabriele Sarti, Malvina Nissim. Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP. 2024.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1357081308676751381)** (15 messages🔥): 

> `Google Mentorship, Tinygrad YoloV8 on Android, LeetGPU support for tinygrad` 


- **Google Mentorship: Worth the Trouble?**: A member expressed skepticism about mentorship programs, citing that the output is *almost never worth the time/effort put into the student* due to onboarding challenges and paperwork.
   - Countering this, another member argued that companies effectively gain *smart people working full-time for you for 3 months*, so the output is pretty good.
- **Tinygrad YoloV8 faces Android hiccups**: A user reported an `OSError: dlopen failed: library "libgcc_s.so.1" not found` when running the **tinygrad** implementation of **YoloV8** on a Samsung Galaxy Tab S9 after `pip install tinygrad`.
   - George Hotz suggested that this is probably a 2 line fix, but adding android to CI to prevent it from happening again, while another suggested `pkg install libgcc`.
- **LeetGPU to support Tinygrad**: Someone asked about [leetgpu.com](https://leetgpu.com).
   - They read that the site will be supporting **tinygrad** soon.



**Link mentioned**: <a href="https://leetgpu.com">LeetGPU</a>: no description found

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1357154502028820561)** (7 messages): 

> `bilinear interpolation, saving latest model` 


- **Bilinear Interpolation Sought**: A member inquired about **bilinear interpolation** support in tinygrad.
   - Another member suggested searching the documentation for **bilinear** but the first member reported that it was *"not working"*.
- **Model Overwriting Questioned**: A member asked if it was safe to use `state_dict = get_state_dict(net); safe_save(state_dict, "model.safetensors")` after every epoch to save the latest model.
   - Another member clarified that it would be overwritten unless a different name is provided for each save.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1357395796068143134)** (1 messages): 

> `CodeAct Agents, ReAct Generalization` 


- **Build your own CodeAct Agent!**: [CodeAct](https://t.co/0GPTPo87ma) from scratch is a generalization of **ReAct** where instead of doing chain-of-thought to reason over tools in a sequential loop, the agent will dynamically write code that uses these functions to solve the task.
- **CodeAct is generalization of ReAct**: CodeAct is a generalization of ReAct where instead of doing chain-of-thought, the agent will dynamically write code to solve the task.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1357153572667785226)** (20 messages🔥): 

> `Rankify framework, Enhance Gemini API Integrations, Cursor API knowledge, otel trace_id, Re-index a file in postgres` 


- **Rankify Framework Streamlines RAG Tasks**: A new open-source framework, [Rankify](https://github.com/DataScienceUIBK/Rankify), is designed to streamline tasks like **retrieval, reranking, and RAG** (Retrieval-Augmented Generation).
- **Gaps in Gemini support to be tackled**: A member is drafting a GSoC proposal for *Enhance Gemini API Integrations* with DeepMind, and would like to make **LlamaIndex** a big part of it.
   - They ask about any standout gaps in **Gemini** support (like multimodal or function calling) in llama-index-llms-google-genai or vertex that need tackling, and also any **Gemini-related features or optimizations**.
- **MCP Tool for Cursor API Knowledge**: A member asked how to give the latest API and docs knowledge to **Cursor** when coding, and if there is an llms.txt etc.
   - Another member responded that the codebase is pretty huge and an *llm.txt* would be near useless, suggesting to give an **MCP tool** that does retrieval over the docs though or similar.
- **Trace ID challenge**: A member is facing an issue where the **otel trace_id** cannot be retrieved after a parent workflow calls a child workflow.
   - Another member suggested putting the **trace_id** somewhere else where it can be fetched (workflow context, some other global var).
- **Vector Index Update in Postgres**: A member wants to **update a vector index** for a file stored in a vector table at postgres.
   - Another member suggested to *delete the rows from the original document, and reindex it*.



**Link mentioned**: <a href="https://github.com/DataScienceUIBK/Rankify">GitHub - DataScienceUIBK/Rankify: 🔥 Rankify: A Comprehensive Python Toolkit for Retrieval, Re-Ranking, and Retrieval-Augmented Generation 🔥. Our toolkit integrates 40 pre-retrieved benchmark datasets and supports 7+ retrieval techniques, 24+ state-of-the-art Reranking models, and multiple RAG methods.</a>: 🔥 Rankify: A Comprehensive Python Toolkit for Retrieval, Re-Ranking, and Retrieval-Augmented Generation 🔥. Our toolkit integrates 40 pre-retrieved benchmark datasets and supports 7+ retrieval techn....

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1357087741594308820)** (11 messages🔥): 

> `ChatGPT 4o Magic The Gathering Cards, Runway Gen 4, Alibaba Wan 2.2` 


- **ChatGPT 4o Generates MTG Cards**: A member used **ChatGPT 4o's image generator** to create **Magic the Gathering Cards** of pop figures and the **NousResearch team**.
   - They claimed the results were *high taste tester approved* and awesome, although there was a comment that *sama sucks tho*.
- **Runway Gen 4 Boosts A.I. Filmmaking**: A member remarked that **A.I. Prompt Filmmaking** has come a long way with **Runway's release of Gen 4** and [links a video](https://www.youtube.com/watch?v=Rcwfj18d8n8) covering the latest happenings in the world of **OpenAI, Google, and AGI**.
   - The video highlights that *AI Video is getting UNREAL...* and also notes that **Alibaba Wan 2.2**, an open source alternative, release is not far behind.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Teknium1/status/1907492873991499998">Tweet from Teknium (e/λ) (@Teknium1)</a>: I spent the last week using ChatGPT 4o&#39;s image generator to create high taste tester approved Magic the Gathering Cards of a bunch of pop figures in AI and a bunch of the @NousResearch team, and t...</li><li><a href="https://www.youtube.com/watch?v=Rcwfj18d8n8">AI Video is getting UNREAL... (GEN 4)</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1357449307648688260)** (3 messages): 

> `LLMs for extraction, Genstruct-7B, Ada-Instruct` 


- **LLMs Extracting Data Sets**: A member inquired about using **LLMs for extraction** to create datasets from unstructured PDFs.
   - Another member suggested that it might be better to prompt a larger model, but linked to **Genstruct-7B** as a good starting point, to create synthetic instruction finetuning datasets from any raw-text corpus.
- **Genstruct-7B Generates instructions**: [Genstruct 7B](https://huggingface.co/NousResearch/Genstruct-7B) is an **instruction-generation model**, designed to create valid instructions given a raw text corpus.
   - This approach was inspired by [Ada-Instruct](https://arxiv.org/abs/2310.04484) which trained a custom instruction-generation model, and also by a [github repo](https://github.com/edmundman/OllamaGenstruct) for quick use with ollama and a bunch of pdfs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://github.com/edmundman/OllamaGenstruct">GitHub - edmundman/OllamaGenstruct</a>: Contribute to edmundman/OllamaGenstruct development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1357128517384667147)** (1 messages): 

> `OpenAPI, SaaS, PaaS, IaaS, LLMs` 


- **OpenAPI access released for LLMs**: A member announced the release of their **v1 OpenAPI access** to **SaaS/PaaS/IaaS** for **LLMs**, aiming to eliminate **MCP clutter** ([link to HN discussion](https://news.ycombinator.com/item?id=43562442)).
- **MCP Clutter Reduction via OpenAPI**: The new **OpenAPI access** intends to solve the problem of **MCP (Multi-Cloud Platform) clutter** when integrating **LLMs** with various cloud services.



**Link mentioned**: <a href="https://news.ycombinator.com/item?id=43562442>">no title found</a>: no description found

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1357108864939593829)** (8 messages🔥): 

> `Cohere Status Page, Python logging vs print statements, RAG strategy for documents` 


- **Cohere hit by Degraded Performance!**: Some users reported experiencing **http timeout errors** and confirmed the [Cohere Status Page](https://status.cohere.com/) indicated *Degraded Performance - Increased Latency* for **Command-a-03-2025/command-r-plus-08-2024** models.
   - The issue was being monitored and had been ongoing for **4 hours**.
- **Python Logging: Teammates Disagree!**: A member is developing their first Python package for PDF processing and is in disagreement with a senior teammate over whether to use **logs** or **print statements**.
   - The member prefers logs for **different levels, file saving, searchability, and issue reporting**, while the teammate prefers **print statements** to avoid burdening users; a compromise of a **disabled logger instance by default** was suggested.
- **RAG Strategy: Chunk or Not to Chunk Long Documents?**: A member asked about using a **18000 token document** for **RAG** and whether to cut it up.
   - An expert recommends chopping the documents, but it depends on the end goal and requirements, also suggesting that **Command-a's 256k context window**, and **command-r and r-plus's 128k context window** should easily be able to handle it.



**Link mentioned**: <a href="https://status.cohere.com/">Cohere Status Page Status</a>: Latest service status for Cohere Status Page

  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1357411997645279464)** (1 messages): 

> `AI Safety Testing Platform, Bias and Harmful Outputs, AI Model Deployment Challenges` 


- ****Brainstorm** AI Safety Testing Platform releasing soon!**: An AI safety testing platform called **Brainstorm** is releasing its MVP in a few weeks, aiming to ensure AI changes the world for the better and you can find out more at the [Brainstorm landing page](https://brainstormai.framer.website/).
- **Soliciting methods for testing AI Safety and Performance**: The creator of **Brainstorm** is seeking insights on current methods used to test AI for safety and performance issues, particularly around **bias**, **prompt injections**, or **harmful outputs**.
- **Discuss biggest pain points for AI model deployment**: The focus is on understanding the biggest pain points in ensuring AI models are ready for real-world deployment, inviting those interested to DM or comment to share experiences.



**Link mentioned**: <a href="https://brainstormai.framer.website/">Brainstorm - AI Safety Made Easy</a>: The simple solution to AI safety testing.

  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1357168810917888071)** (2 messages): 

> `KAIST student, Bias/fairness and interpretability in LLMs/VLMs, Research collaboration opportunities` 


- **KAIST student seeks collaborations**: A M.S. student from **KAIST** (South Korea) introduced themself with a research focus on **bias/fairness** and **interpretability** in **LLMs/VLMs**.
   - They are actively seeking research collaboration opportunities in these specific areas.
- **Collaboration Opportunity**: The student is looking for research collaborations.
   - Their background includes experience at KAIST.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1357079177387446484)** (7 messages): 

> `Nomic Embed Text V2, Vulnerability Disclosure, GPT4All-J model, Chocolatine-2-14B model, Chat Reorganization` 


- **Patience Pays for Nomic Embed Text V2**: One member is waiting for **Nomic Embed Text V2** to be available in **GPT4All**, acknowledging that developers are busy.
   - They expressed understanding that the integration might take time.
- **Vulnerability Disclosure Venue Via Contact Sales**: A member inquired about the appropriate contact for responsibly disclosing a vulnerability in **GPT4All**.
   - Another member suggested using the [contact support email](https://atlas.nomic.ai/contact-sales) available on the **Nomic AI** website.
- **GPT4All-J Model Search Sparked Quantization Query**: A member requested a download link for the **GPT4All-J model** in **Q4_0 quantization** and **GGUF format** to integrate it into their project.
   - A second member responded that **GPT4All-Falcon** is available as **GGUF**, but that **GPT4All-J** is not possible.
- **Chocolatine-2-14B Claims Crown for Embedded Book Queries**: One member lauded the "**Chocolatine-2-14B**" model type as their preferred choice for querying embedded books.
   - No further information was provided about this model.
- **Chats Crave Chronological Correction**: A member suggested that chats should reorganize based on the time they were altered rather than when they were created.
   - They criticized the current chronological listing by creation date as *arbitrary*.



**Link mentioned**: <a href="https://atlas.nomic.ai/contact-sales">Contact Nomic Sales</a>: Explore, analyze and build with your unstructured data

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1357079321616978121)** (5 messages): 

> `LLM agent development, DSPy Framework, OpenAI Agents SDK, Prompt Engineering vs programming` 


- **Telemetry Improves LLM Agent Development Loop**: A member shared a video titled *Close the loop on LLM agent development by configuring them to improve themselves using telemetry and evaluations* [available on YouTube](https://youtu.be/jgzSq5YGK_Q).
   - The video discusses leveraging **telemetry** and **evaluations** to enhance LLM agent self-improvement.
- **DSPy Decouples Prompt Engineering**: A member inquired about DSPy's role in decoupling the *tinkering layer* of **prompt engineering** from the *functional behavior* of LLMs and how it synergizes with **OpenAI Agents SDK**.
   - Another member confirmed this, stating that DSPy provides *programmatic pieces*: **signatures and modules** to achieve this decoupling.
- **DSPy's Programmatic Pieces Explored**: A member highlighted DSPy's core abstractions, namely **signatures and modules**, which facilitate the decoupling of prompt engineering from LLM functional behavior.
   - This approach aims to enable programming rather than just prompt engineering, which helps in integrating DSPy with other tools like **OpenAI Agents SDK**.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1357148524063752363)** (3 messages): 

> `Tool evaluation, Phi-4-mini-instruct, BFCL` 


- **Phi-4-mini-instruct Evaluation PR Submitted**: A member submitted a [PR](https://github.com/ShishirPatil/gorilla/pull/967) to add tool evaluation for **Phi-4-mini-instruct** with **BFCL**.
- **Reviewers needed for tool evaluation**: A member asked for feedback on their PR, noting that they have attached the **evaluation score** within the PR.
- **Review in progress**: A member said they will take a look at the **PR**.



**Link mentioned**: <a href="https://github.com/ShishirPatil/gorilla/pull/967">[BFCL] add support for microsoft/Phi-4-mini-instruct by RobotSail · Pull Request #967 · ShishirPatil/gorilla</a>: This PR introduces support for the newly-released Phi-4-mini-instruct model from Microsoft:Phi-4-mini-instructThe results for this were initially evaluated against f81063; however, the model ha...

  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1357462550974959806)** (1 messages): 

> `DeepSeek-V3, DeepSeek-V3-0324, Windsurf AI` 


- **DeepSeek-V3 Gets a Facelift**: **DeepSeek-V3** has been upgraded to **DeepSeek-V3-0324**, supposedly performing slightly better than before in evaluations.
   - A member posted a [link](https://x.com/windsurf_ai/status/1907902846735102017) to the **Windsurf AI** twitter account announcing the upgrade and its continued free availability.
- **Windsurf Asks for Bookmarks**: Windsurf is trying to increase the visibility of their announcements.
   - A member asked users to bookmark the announcement post on X.



**Link mentioned**: <a href="https://x.com/windsurf_ai/status/1907902846735102017">Tweet from Windsurf (@windsurf_ai)</a>: DeepSeek-V3 has now been upgraded to DeepSeek-V3-0324. It&#39;s still free!

  

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

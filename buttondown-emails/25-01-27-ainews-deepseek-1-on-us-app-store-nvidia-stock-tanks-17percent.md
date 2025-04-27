---
id: bfd6ecb6-ea05-4120-af25-be51bdfc677c
title: 'DeepSeek #1 on US App Store, Nvidia stock tanks -17%'
date: '2025-01-28T05:28:32.064176Z'
original_slug: ainews-deepseek-1-on-us-app-store-nvidia-stock
description: >-
  **DeepSeek** has made a significant cultural impact by hitting mainstream news
  unexpectedly in 2025. The **DeepSeek-R1** model features a massive **671B
  parameter MoE architecture** and demonstrates **chain-of-thought (CoT)**
  capabilities rivaling **OpenAI's o1** at lower cost. The **DeepSeek V3** model
  trains a **236B parameter model 42% faster** than its previous **67B model**
  using **fp8 precision**. **Qwen2.5-VL** is a new multimodal model supporting
  images and videos with sizes ranging from **3B to 72B parameters**, featuring
  strong vision and agentic capabilities. Discussions highlight **NVIDIA's**
  role in hardware acceleration, with concerns about stock drops due to
  **DeepSeek's** efficiency gains and compute demand increases. The AI
  competition landscape is tense with market reactions and strategic
  vulnerabilities related to chip supply chains. **LangChain** and **LangGraph**
  integrations enable advanced chatbot memory and tool use, including DeFi
  automation.
companies:
  - deepseek
  - openai
  - nvidia
  - langchain
models:
  - deepseek-r1
  - deepseek-v3
  - qwen2.5-vl
  - o1
topics:
  - moe
  - chain-of-thought
  - fp8
  - multimodality
  - vision
  - agentic-ai
  - inference-speed
  - compute-demand
  - stock-market
  - hardware-acceleration
  - model-training
  - model-efficiency
  - ai-competition
  - chatbots
  - defi
people:
  - sama
  - teortaxestex
  - nptacek
  - carpeetti
  - mervenoyann
  - omarasar0
  - samyj19
  - ykylee
  - finbarrtimbers
  - cwolferesearch
  - arthurrapier
  - danhendrycks
  - scaling01
---


<!-- buttondown-editor-mode: plaintext -->**DeepSeek is all you need.**

> AI News for 1/24/2025-1/27/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **34** Discords (**225** channels, and **11316** messages) for you. Estimated reading time saved (at 200wpm): **1229 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We really try to keep news reporting technical here, but on rare occasions, mainstream/nontechnical news is so significant that it gets through. 

This is one of those days.

[/r/LocalLlama](https://www.reddit.com/r/LocalLLaMA/comments/1iasyc3/deepseek_is_1_on_the_us_app_store/):

![image.png](https://assets.buttondown.email/images/a7439f78-2553-49dd-8169-5199b2f1c32c.png?w=960&fit=max)

and [sama](https://x.com/sama/status/1884066337103962416):

![image.png](https://assets.buttondown.email/images/f608fcb6-fa5f-4798-a202-4434d1872e47.png?w=960&fit=max)

Ultimately much of the discussion is very unhelpful that looks like some version of this

![image.png](https://assets.buttondown.email/images/4861bf2e-0401-4526-a859-1161a880876b.png?w=960&fit=max)

and we are reporting mostly on the cultural moment of DeepSeek hitting mainstream news which was not ever on our bingo card for 2025.

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

**AI Model Releases and Enhancements**

- **DeepSeek-R1 and V3 Efficiency**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1883976116949639568) discussed how **V3** demonstrates the ability to train a **236B model 42% faster** than their previous **67B model**, utilizing **fp8** precision to maintain speed for larger models. [@nptacek](https://twitter.com/nptacek/status/1883920168952422789) highlighted that **DeepSeek-R1** requires substantial GPUs, emphasizing its **MoE architecture** with **671B parameters**. [@carpeetti](https://twitter.com/casper_hansen_/status/1883974834025292047) praised **DeepSeek-R1** for its **chain-of-thought (CoT)** capabilities, rivaling **OpenAI's o1** at a fraction of the cost.

- **Qwen2.5 Models**: [@mervenoyann](https://twitter.com/mervenoyann/status/1883954645602906249) announced the release of **Qwen2.5-VL**, a **multimodal model** capable of handling **images and videos**, with versions including **3B, 7B, and 72B parameters**. [@omarasar0](https://twitter.com/omarsar0/status/1883965524205359460) detailed the **strong vision capabilities** and **agentic features** of **Qwen2.5**, supporting **long video understanding** and **structured data outputs**.

- **LangChain and LangGraph Integration**: [@LangChainAI](https://twitter.com/LangChainAI/status/1883666232789889259) shared tutorials on building **AI chatbots** using **LangGraph**, enabling **memory and tool integration**. They also showcased applications like the **DeFi Agent**, automating **Aave protocol operations**.

**Compute and Hardware**

- **NVIDIA Impact**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1883979845266465076) expressed concerns about training on a **32K Ascend 910C cluster**, suggesting potential shorting of **NVIDIA** stocks. [@samyj19](https://twitter.com/giffmana/status/1883662627920031857) and [@ykylee](https://twitter.com/giffmana/status/1883661880822284792) discussed **DeepSeek-R1's** **inference speed optimizations**, leveraging **NVIDIA H800** GPUs for enhanced performance.

- **Compute Demand**: [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1883961408553116091) argued that **compute demand** will **increase** due to **inference scaling**, despite **DeepSeek's** efficiency. [@cwolferesearch](https://twitter.com/cwolferesearch/status/1883885191661326391) analyzed **DeepSeek-v3's** **Mixture-of-Experts (MoE)** design, highlighting its **efficiency and performance** improvements.

**AI Competition and Market Reactions**

- **Stock Market Reactions**: [@MiddleOpenAI](https://twitter.com/nearcyan/status/1883944036811096517) reported a significant drop in **NVIDIA's** stock following **DeepSeek's** advancements, citing a **-17%** decline due to **market fears**. [@arthurrapier](https://twitter.com/fchollet/status/1883973637075816555) echoed concerns about **NVIDIA's** **bearish signals**, while others like [@DanHendrycks](https://twitter.com/DanHendrycks/status/1883660982641426727) emphasized the **strategic vulnerabilities** due to **chip supply chain dependencies**.

- **Competitive Landscape**: [@scaling01](https://twitter.com/scaling01/status/1883912104182452629) criticized the **market's reaction** to **DeepSeek**, arguing that **DeepSeek's** efficiencies challenge the **assumptions behind high-profit models**. [@janusflow](https://twitter.com/janusflow/status/1883932760940888071) noted that **DeepSeek's** releases are **disruptive** to the **tech ecosystem**, causing **market volatility**.

**AI Applications and Use Cases**

- **Agentic Capabilities**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1883954202733827584) introduced **Grace, Kane, and Flows**, AI agents capable of **executing commands** on **computers and smartphones**, demonstrating **real-time interactions** and **multi-step reasoning**.

- **Historical Research and Drug Discovery**: [@omarsar0](https://twitter.com/omarsar0/status/1883890211538776501) explored the application of **LLMs** in **historical research**, such as **transcribing early modern Italian** and **generating historical interpretations**. Additionally, integration with **drug discovery** through **hallucination features** was discussed.

- **Video and Image Processing**: [@mervenoyann](https://twitter.com/mervenoyann/status/1883916608961479034) showcased **DeepSeek's Janus-Pro** for **multimodal image generation**, surpassing models like **DALL-E**. [@chethaan saggeev](https://twitter.com/chethaan/status/1883923932786655491) highlighted **NVIDIA’s Cosmos Tokenizer** for **physical AI training**, enhancing **image and video tokenization**.

**Technical Discussions and Innovations**

- **Reinforcement Learning and Training Efficiency**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1883968204650996137) emphasized the importance of **reinforcement learning (RL)** in **DeepSeek's** models, highlighting the **independent concurrent work** in **DeepSeek Zero paradigm**. [@lateinteraction](https://twitter.com/lateinteraction/status/1883939171926241324) discussed the absence of a **secret revolutionary technique**, attributing success to **engineering precision**.

- **Quantization Techniques**: [@danielhanchen](https://twitter.com/danielhanchen/status/1883901952922448162) detailed the **quantization** of **DeepSeek R1** to **1.58bit**, achieving an **80% size reduction** while maintaining usability through **dynamic quantization**. This innovation allows models to run on more **accessible hardware**.

- **Mixture-of-Experts (MoE) Models**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1883885191661326391) explained **DeepSeek-v3's MoE** architecture with **shared experts** and **multi-token prediction**, enhancing both **training efficiency** and **model performance**.

**AI Business and Market Reactions**

- **Open-Source vs. Proprietary Models**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1883946119723708764) advocated for **open-source AI**, stating, "**AI is not a zero-sum game. Open-source AI is the tide that lifts all boats!**" This sentiment was echoed by [@cwolferesearch](https://twitter.com/cwolferesearch/status/1883885191661326391), who praised the **transparency** and **cost-effectiveness** of **DeepSeek's** open-source models.

- **Investment Strategies**: [@swyx](https://twitter.com/swyx/status/1883961408553116091) advised against **shorting NVIDIA**, arguing that **DeepSeek's** advancements **drive compute demand** rather than reduce it. Conversely, [@scaling01](https://twitter.com/scaling01/status/1883944036811096517) suggested **shorting** strategies based on **DeepSeek's** impact on **AI compute economics**.

- **Hiring and Talent Acquisition**: [@AlexAlbert__](https://twitter.com/alexalbert__/status/1883907893294170610) and others mentioned **hiring opportunities** within **AI companies** like **Anthropic**, emphasizing the need for **diverse technical backgrounds** to drive **future AI innovations**.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepSeek is #1 on U.S. App Store: Market Implications**

- **[Deepseek is #1 on the U.S. App Store](https://i.redd.it/sr4kvvnv3ffe1.jpeg)** ([Score: 1618, Comments: 341](https://reddit.com/r/LocalLLaMA/comments/1iasyc3/deepseek_is_1_on_the_us_app_store/)): **Deepseek** has achieved the top position in the **U.S. App Store's "Top Free Apps"** section, surpassing notable applications like **ChatGPT** and **Threads**. This ranking highlights its competitive edge as an **Intelligent AI Assistant**, with implications for its market position against established AI tools.
  - There is skepticism about **Deepseek's** competitive advantage and concerns about it facing a similar fate as **TikTok** due to potential national security risks. Some users express frustration with the app's server downtime due to high traffic, while others question its unique offerings to the average user compared to other AI models like **ChatGPT** and **Perplexity**.
  - The discussion highlights the **open-source nature** of Deepseek, with users noting that its model weights and training methods could potentially be released, making it more accessible. Some users discuss the feasibility of running Deepseek locally, with references to **distilled models** that can operate on consumer-grade hardware, though the full model requires significant resources.
  - Comments reflect a broader conversation about **global competition** in AI development, with some users criticizing the notion of a "moat" and emphasizing that multiple countries can create competitive software. There is also debate over the perception of **American** approaches to technology competition and the implications of **Deepseek's** open-source approach on international dynamics.


- **[OpenAI employee’s reaction to Deepseek](https://i.redd.it/ij7ubrn3mkfe1.jpeg)** ([Score: 1239, Comments: 256](https://reddit.com/r/LocalLLaMA/comments/1ibej82/openai_employees_reaction_to_deepseek/)): An **OpenAI employee**, Steven Heidel, criticized data privacy concerns related to **DeepSeek**, suggesting that Americans are trading their data to the **CCP** for free services. The discussion highlights that **DeepSeek** can operate locally without an internet connection, unlike OpenAI's models.
  - **Open Source and Local Operation**: Many commenters highlight that **DeepSeek** is open source and can be run on local or cloud hardware, which addresses concerns about data privacy and reliance on foreign entities. **TogetherAI** is mentioned as a service that hosts the model without using data for training, providing an alternative to running it locally.
  - **Censorship and Model Transparency**: There is skepticism about the transparency of AI models, with some users noting that **DeepSeek** exhibits censorship tendencies by aligning with CCP narratives, which underscores the need for truly open models like those being developed by **HuggingFace**.
  - **Hardware and Accessibility**: Discussions around the hardware requirements for running large models like **DeepSeek** emphasize that while individuals may lack the resources, well-funded startups could potentially afford the necessary infrastructure. Some users mention specific hardware setups, such as using **30 3090/4090s** or **9 large 80 GB GPUs**, to manage the model's demands.


- **1.58bit DeepSeek R1 - 131GB Dynamic GGUF** ([Score: 552, Comments: 125](https://reddit.com/r/LocalLLaMA/comments/1ibbloy/158bit_deepseek_r1_131gb_dynamic_gguf/)): The post discusses the **dynamic quantization** of the DeepSeek R1 671B MoE model to **1.58bits** in GGUF format, effectively reducing the disk size to **131GB** by quantizing only the MoE layers to **1.5bit** while keeping attention and other layers at **4 or 6bit**. This method prevents issues like producing gibberish and infinite repetitions, achieving a processing speed of **140 tokens/s** on 2x H100 80GB GPUs, and successfully generating a **Flappy Bird** game under specific conditions. Additional resources and details are available on [Hugging Face](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S) and [Unsloth's blog](https://unsloth.ai/blog/deepseekr1-dynamic).
  - **Quantization Strategy**: The key to the successful quantization of the **DeepSeek R1 671B MoE model** was only quantizing the MoE layers to **1.5bit** while maintaining other layers in higher precision (4 or 6 bits), aligning with principles from the **BitNet paper** that suggests retaining high precision for certain layers to optimize performance. This approach prevents issues like excessive computational costs and maintains the model's ability to perform complex tasks, such as generating a **Flappy Bird** game.
  - **Compatibility and Implementation Concerns**: Users discussed challenges and sought guidance on running the model with different setups, such as **Ollama**, **LM studio**, and **llama.cpp**, highlighting the importance of understanding specific implementations and compatibility issues. There were inquiries about hardware requirements, with one user noting that a **24GB GPU like RTX 4090** should handle **1 to 3 tokens/s**.
  - **Community Feedback and Performance Expectations**: There was significant positive feedback on the model's performance, with users expressing amazement at its capabilities, especially the ability to generate a bug-free **Flappy Bird** game. Users also discussed potential performance metrics, such as inference speed on different hardware configurations, and expressed interest in benchmarks and comparisons with other models like **Q2KS**.


**Theme 2. How Deepseek Reduces Costs by 95-97%**

- **How *exactly* is Deepseek so cheap?** ([Score: 386, Comments: 334](https://reddit.com/r/LocalLLaMA/comments/1ib4ksj/how_exactly_is_deepseek_so_cheap/)): Deepseek achieves a **95-97% reduction in costs** by employing strategies like avoiding **RLHF (Reinforcement Learning from Human Feedback)**, utilizing **quantization**, and implementing **semantic input HTTP caching**. However, there is confusion about whether R1 is quantized, leading to questions about potential subsidies or if **OpenAI/Anthropic** are overcharging.
  - Discussions highlight the cost-saving strategies of **Deepseek**, emphasizing **MoE (Mixture of Experts)**, **FP8** precision, and **multi-token prediction (MTP)** as key factors. These technological choices, alongside **cheap electricity** and **lower R&D costs**, contribute to their significant cost reductions compared to **OpenAI/Anthropic**. Some users suspect **government subsidies** or **operating at a loss** to capture market share.
  - There is skepticism regarding the **true costs** and **efficiency** of Deepseek's operations, with some commenters questioning the **financial transparency** and **sustainability** of their pricing model. Concerns are raised about whether they are using cheaper **Nvidia H800** chips and if **OpenAI/Anthropic** are overcharging due to potentially unsustainable business models.
  - The open-source nature of Deepseek's models, available on platforms like **Huggingface**, is seen as a competitive advantage, allowing for widespread adoption and **flexibility in hosting**. However, there are doubts about the **operational quality** and **performance** of these models, with some users reporting issues in translation capabilities and questioning the **credibility** of Deepseek's claims.


**Theme 3. New Tool for Local LLM Compatibility: 'Can You Run It?'**

- **Someone needs to create a "Can You Run It?" tool for open-source LLMs** ([Score: 298, Comments: 64](https://reddit.com/r/LocalLLaMA/comments/1iaubfm/someone_needs_to_create_a_can_you_run_it_tool_for/)): A non-techie user expresses the need for a tool similar to **System Requirements Lab** for open-source **LLMs** like **Deepseek, LLaMA,** and **Mistral**, to determine if these models can run on their hardware. They propose a system where users can input their computer specs to receive a straightforward performance verdict and suggestions for optimizations, such as using quantized versions for better compatibility with lower-end systems.
  - Several tools and resources are mentioned for determining if **LLMs** can run on specific hardware, including **[Vokturz's can-it-run-llm](https://huggingface.co/spaces/Vokturz/can-it-run-llm)** and **[NyxKrage's LLM-Model-VRAM-Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)**. These tools help users calculate VRAM requirements and assess model compatibility with their systems.
  - Community members share rules of thumb for estimating hardware needs, such as **1GB per 1B parameter count** and **1GB per 1K context**, with recommendations like **llama 3.2** or **Qwen 2.5** for optimal performance on lower-end systems. They also discuss the impact of quantization and context length on performance and memory usage.
  - There is a demand for a user-friendly, open-source tool that offers privacy and keeps up-to-date with model requirements, as expressed by users like **Solid_Owl** and **Shark_Tooth1**, who express concerns about privacy and seek a reliable performance expectation tool for local **LLM** use.


- **I created a "Can you run it" tool for open source LLMs** ([Score: 261, Comments: 50](https://reddit.com/r/LocalLLaMA/comments/1ib2uuz/i_created_a_can_you_run_it_tool_for_open_source/)): I created a **"Can you run it" tool** for open source **LLMs** that provides a **tk/s estimate** and instructions for running models with options like **80% layer offload** and **KV offload** on GPU. The tool has been tested on **Linux with a single Nvidia GPU**, and feedback from other systems, including multi-GPU setups, is requested to identify potential issues. [GitHub link](https://github.com/Raskoll2/LLMcalc).
  - **Mac Compatibility**: **Environmental-Metal9** adjusted calculations for macOS, reporting discrepancies in performance estimates on an **M1 Max**. They offered to contribute a patch for **Mac support** via a pull request or pastebin.
  - **User Interface Suggestions**: Users, including **Catch_022** and **MixtureOfAmateurs**, suggested simplifying the tool's usability by creating a **portable executable with a GUI** or hosting it as a **website** to eliminate the need for Python installation.
  - **Web Interface and Monetization**: **Whole-Mastodon6063** developed a web app interface for the tool, and **mxforest** suggested hosting it online with ads for potential revenue, with **Ok-Protection-6612** and **femio** supporting the idea of monetization through sponsorships.


**Theme 4. Qwen 3.0 MOE: Emerging Reasoning Model**

- **[Qwen3.0 MOE? New Reasoning Model?](https://i.redd.it/0vnua5vqxjfe1.png)** ([Score: 239, Comments: 34](https://reddit.com/r/LocalLLaMA/comments/1ibb8rr/qwen30_moe_new_reasoning_model/)): **Qwen3.0 MOE** and a potential **New Reasoning Model** are hinted at in a tweet by **Binyuan Hui**, suggesting upcoming announcements or events. The tweet implies significant developments in AI, though specific details are not provided.
  - **Qwen2.5-VL** is confirmed to be part of the upcoming releases, with a collection already created on [Hugging Face](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5). This suggests imminent updates beyond just vision models, possibly including **Qwen MoE** and **Qwen 3.0**.
  - **DeepSeek** is mentioned as a partner to handle the significant compute needs, with some users hoping for new reasoning models under **Apache/MIT licenses**. There is anticipation for various model sizes and capabilities, including audio and large-scale models like **Qwen 2.5 100B+**.
  - The timing of the announcements is questioned due to the proximity to the **Chinese New Year holiday**, with skepticism about the hype surrounding the release.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. Nvidia Stock Volatility: Impact of DeepSeek's Efficient Model**

- **[Nvidia Bubble Bursting](https://i.redd.it/xpen835nbkfe1.png)** ([Score: 627, Comments: 245](https://reddit.com/r/OpenAI/comments/1ibd2p8/nvidia_bubble_bursting/)): **Nvidia's stock** has experienced a significant decline, with a drop of **$17.66 (12.68%)** over five days, from a peak of **$142.02** on January 24, 2023, to **$121.56** by January 27. The company's **market cap** is reported at **$2.97 trillion**, with a **P/E ratio of 48.07**, and a 52-week range between **$60.70** and **$153.13**.
  - Many commenters view the stock decline as a **buying opportunity**, with several expressing confidence that **Nvidia** will rebound. **itsreallyreallytrue** and **AGIwhen** suggest that the demand for Nvidia's GPUs remains strong due to their critical role in AI infrastructure, despite **DeepSeek's** claims of reduced GPU requirements.
  - Discussions highlight skepticism towards **DeepSeek's** claims and their impact on Nvidia's stock, with **TheorySudden5996** and **Agreeable_Service407** noting that despite potential efficiency gains, the need for GPUs remains significant. **DerpDerper909** argues that even with efficiency improvements, Nvidia will benefit from the lowered entry barriers for smaller companies developing AI models.
  - **Cramer4President** and others criticize the notion of a "bubble bursting" based on short-term stock performance, advocating for a broader perspective over a longer timeframe. **OptionsDonkey** and **Legitimate-Arm9438** emphasize that Nvidia's long-term value remains strong, suggesting that the current dip is a temporary fluctuation rather than a fundamental issue.


- **[Was this about DeepSeek? Do you think he is really worried about it?](https://i.redd.it/v8oe8q5seife1.jpeg)** ([Score: 540, Comments: 203](https://reddit.com/r/OpenAI/comments/1ib4vq7/was_this_about_deepseek_do_you_think_he_is_really/)): **Sam Altman** highlights the challenge of creating innovative and risky projects compared to copying existing successful ideas, emphasizing the importance of recognizing individual researchers for their groundbreaking work. He concludes by stating that these efforts represent the "coolest thing in the world."
  - **Criticism of Sam Altman's Statement**: Many commenters criticized **Sam Altman's** emphasis on individual researchers, arguing that breakthroughs are often the result of collaborative efforts. **Neofelis213** highlighted the myth of the lone researcher and pointed out that figures like **Sam Altman** and **Elon Musk** often overshadow the actual contributors to technological advancements.
  - **Historical Context and Contributions**: Discussions focused on the origins of **transformer architecture** and **LLMs**, with users noting that **Google** published the foundational paper, "Attention Is All You Need," which OpenAI built upon. **coloradical5280** and others emphasized the collaborative nature of these developments and the role of key figures like **Ilya Sutskever** in evolving the technology.
  - **Ethics and Copyright Concerns**: Several comments addressed the ethical implications of using copyrighted material in training AI models, with **Riegel_Haribo** mentioning the large-scale copyright infringement involved in AI training. This sparked debates over the legality and fairness of using public data, referencing historical cases like **Aaron Swartz**'s prosecution.


- **"Every model has censorship" is an ignorant argument** ([Score: 179, Comments: 146](https://reddit.com/r/OpenAI/comments/1iazo74/every_model_has_censorship_is_an_ignorant_argument/)): The post criticizes Western perceptions of **DeepSeek** and **ChatGPT** censorship, arguing that while both have censorship, the **CCP's** is far more harmful as it suppresses criticism of authoritarian power. The author emphasizes that **Chinese AI models** are universally government-censored, unlike Western alternatives, and highlights the exploitation of Chinese citizens under the CCP, with many earning less than **$4,000** annually and lacking free healthcare. The post condemns Westerners for overlooking these issues in favor of cheap Chinese products.
  - Several commenters argue that **censorship and authoritarianism** are not unique to China, as the **US** also engages in similar practices, including censorship in AI models like **Gemini** and **ChatGPT**, and reliance on undocumented labor. They argue that Western AI models are also censored to protect political interests and that the US has its own issues with wealth inequality and exploitation.
  - Discussions highlight the **exploitative nature** of AI technology, noting that datasets are often compiled using unpaid intellectual property and that the labor involved in creating and maintaining these technologies is undervalued. Commenters criticize the hypocrisy in condemning China's practices while ignoring similar issues in Western countries, such as the role of companies like **Lockheed Martin** in government spending and the role of billionaires like **Larry Ellison** in AI surveillance.
  - Some commenters express skepticism about the **impact of censorship** on AI development, suggesting that open-source projects like those on **HuggingFace** can bypass censorship. They note that the rapid progress in AI, with models like **R1** being reverse-engineered, diminishes the power of censorship, as more models are developed locally and with fewer restrictions.


**Theme 2. DeepSeek R1's Coding Efficiency vs OpenAI O3**

- **[DeepSeek R1 is 25x cheaper than o1 and better in coding benchmarks than the "unreleased" o3 at the same* cost. DeepSeek is giving OpenAI a run for their money.](https://i.redd.it/w6rngm2iyhfe1.png)** ([Score: 355, Comments: 111](https://reddit.com/r/OpenAI/comments/1ib3j3a/deepseek_r1_is_25x_cheaper_than_o1_and_better_in/)): **DeepSeek R1** is positioned as being **25x cheaper** than OpenAI's **o1** model and demonstrates superior coding performance compared to the "unreleased" **o3** at a similar cost. Graphical data highlights DeepSeek R1's favorable **15.8%** performance score in coding benchmarks, underscoring its cost-effectiveness and competitive edge against other models.
  - Several commenters question the credibility of **DeepSeek R1**'s performance claims, highlighting the presence of a **question mark** in the data and the need for third-party validation. Concerns are raised about the paper's methodology and the lack of verifiable information regarding the training hardware.
  - There is skepticism about the frequency and nature of **DeepSeek** promotions, with suggestions of a possible deliberate campaign or "astroturfing." Commenters compare this to the promotion of other models like **Claude** and **Gemini**, noting a similar pattern of aggressive marketing.
  - Some users express support for increased competition in the AI space, hoping for more entrants like **Meta** and **Claude**. However, others are frustrated with the overwhelming number of promotional posts about DeepSeek, questioning its actual utility and performance compared to established models like **OpenAI** and **Claude**.


**Theme 3. Debates on DeepSeek vs ChatGPT: A Censorship Perspective**

- **[Octopus-inspired logarithmic spiral manipulator can manipulate a wide variety of objects](https://v.redd.it/abwbgx3e1hfe1)** ([Score: 537, Comments: 41](https://reddit.com/r/OpenAI/comments/1ib0pow/octopusinspired_logarithmic_spiral_manipulator/)): The post title suggests a discussion on a **logarithmic spiral manipulator** inspired by an **octopus**, capable of handling diverse objects. The ethical implications of AI in political censorship are not directly addressed, indicating a possible mix-up between the topic and the title.
  - **Technological Origin**: The **logarithmic spiral manipulator** technology was developed by the **University of Science and Technology of China**, with testing also conducted in China. This clarifies any confusion about the origin, as some comments mistakenly attributed it to Japan.
  - **Design and Construction**: The manipulator appears to be constructed from **3D printed pieces** and operates using two threads on opposite sides, emphasizing the significant role of software in its functionality. There is interest in the possibility of the software being **open source**, which could make it more accessible for use.
  - **Public Reaction and Humor**: The discussion includes humorous and dystopian reactions, with references to **robot tentacles** and their potential use in both **war and entertainment** scenarios. This highlights the mixed feelings and imaginative speculation surrounding advanced robotics.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12

**Theme 1. DeepSeek R1 Models Upending the AI Landscape**

- **DeepSeek R1 Shrinks to 1.58-Bit, Packs a Punch!**: Community marvels at [DeepSeek R1 running at 1.58-bit quant](https://x.com/UnslothAI/status/1883899061893546254), reducing size from **720GB** to **131GB**, yet retaining full reasoning capabilities.
- **DeepSeek R1 Challenges OpenAI O1 Head-On**: Users compare **DeepSeek-R1** to **OpenAI's O1**, noting that R1 matches or surpasses O1 in performance on benchmarks like [aider's polyglot](https://aider.chat/2025/01/24/r1-sonnet.html).
- **DeepSeek's Debut Rattles Tech Market**: Reports claim DeepSeek's R1 caused a **$600 billion** drop in US tech stocks, fueling discussions about China's rising AI prowess; see the frenzy in [this Bloomberg clip](https://www.youtube.com/watch?v=7GV_OdqzmIU).

**Theme 2. Qwen2.5 Models Breaking Context Barriers**

- **Qwen2.5 Unveils 1 Million Token Context—Is Bigger Better?**: Alibaba releases [Qwen2.5 models](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M) with a whopping **1 million token context length**, sparking debates on the practicality of such large contexts.
- **Qwen2.5-VL Excels at OCR—Handwriting No Problem!**: The new [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) impresses users with its advanced OCR capabilities, including handwriting analysis and robust visual content parsing.
- **Qwen2.5-VL vs DALL-E 3: The Battle of Vision-Language Models**: Users compare Qwen2.5-VL's visual understanding to models like **DALL-E 3**, highlighting its ability to handle structured data outputs in finance and commerce tasks.

**Theme 3. AI Tools Advance, Integrating Into Developer Workflows**

- **RAG Tactics Spark Dev Conversations**: Developers delve into retrieval-augmented generation methods, discussing vector stores and embeddings like [voyage-code-3](https://blog.voyageai.com/2024/12/04/voyage-code-3/#:~:text=voyage%2Dcode%2D3%20supports%20much,Matryoshka%20embeddings.).
- **Codebase Indexing: A Bare-Bones Approach?**: Some users criticize codebase indexing tools for not fully leveraging project files, referencing [Cursor's documentation](https://docs.cursor.com/context/codebase-indexing), while others find them useful with proper configuration.
- **AI Pair Programming Takes Off with Aider and CodeGate**: [CodeGate integration](https://docs.codegate.ai/how-to/use-with-aider) allows developers to pair program directly in the terminal, enhancing coding workflows with AI assistance.

**Theme 4. OpenRouter Expands with New Models and Providers**

- **Liquid AI Makes a Splash on OpenRouter**: Liquid AI brings [multilingual models LFM 40B, 7B, and 3B](https://openrouter.ai/liquid/lfm-7b) to OpenRouter, claiming top performance in major languages.
- **DeepSeek Nitro: Fast but Not Furious Enough?**: The [Nitro variant for DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1:nitro) promises faster responses, but users report it doesn't outperform the standard R1 in real-world scenarios.
- **OpenRouter Users Bring Their Own Keys (BYOK)**: Discussions emphasize using personal API keys with OpenRouter to mitigate rate limits and control expenses, with a **5% fee** applied on usage.

**Theme 5. Global AI Policies and Investments Heating Up the Competition**

- **China's 1 Trillion Yuan Bet on AI Ignites Global Race**: China announces a **1 trillion yuan** ($137B) investment in AI, as reported [here](https://x.com/rwang07/status/1883210410763121073), raising questions about the US's ability to keep pace.
- **US Debates AI Policy Amid Great Power Competition**: Discussions highlight the US considering funding AI under the banner of great power competition, drawing parallels to historical industrial policies like the **CHIPS Act**.
- **DeepSeek's Rise Raises Geopolitical Eyebrows**: The success of DeepSeek's models fuels concerns over China's growing influence in AI, prompting analysis of AI's role in national competitiveness.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 Levels Up**: Community members showcased **DeepSeek R1** running at **1.58-bit quant** with fully functional reasoning, citing [this tweet](https://x.com/UnslothAI/status/1883899061893546254) and highlighting a size reduction from **720GB** to **131GB**. 
   - They compared **DeepSeek-R1** to **OpenAI’s O1** model, noting growing interest in local usage and open-source collaboration for advanced reasoning tasks.
- **Qwen2.5’s Million-Token Release**: Chat focused on **Qwen2.5** from Alibaba, revealing **1 million token context length** in the [Qwen tweet](https://x.com/Alibaba_Qwen/status/1883557964759654608) and mentioning **14B** parameter instruct versions. 
   - Members debated whether the expanded context size justified the hype, expressing optimism about **large-scale local inference** supported by abundant VRAM.
- **SmoLlm Fine-Tuning Gains Traction**: Multiple users tested **SmoLlm** fine-tuning with **Unsloth** and successfully deployed it using **ollama** with default temperature `0.8`, clarified in a [discussion thread](https://discord.com/channels/1179035537009545276/1179039861576056922/1332482886506385429). 
   - They emphasized smooth integration into personal workflows, stating *“It just works without explicit temperature settings”* and confirming readiness for local code review tasks.
- **Nuances in Dataset Formatting**: Users exchanged tips about structuring training data with **'instruction', 'input', and 'output'** fields, referencing examples from [Wikimedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia) and personal forum collections. 
   - They pinpointed how mismatched field names caused **Unsloth** errors during fine-tuning, reinforcing the need for **consistent question-answer formats** to ensure correct model behavior.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Speed Struggles or Speed Gains? Cursor IDE in Focus**: Users reported **slow request times** and some friction with **Claude** on Cursor, with a partial fix during off-peak hours as described in [Fast and Slow Requests – Cursor](https://docs.cursor.com/get-started/usage#fast-and-slow-requests).
   - They also praised **DeepSeek R1** for certain tasks and mentioned the possibility of using [Spark Engine](https://sparkengine.ai) as a complement to reduce request times and costs.
- **Claude vs DeepSeek: Battle for Code Insights**: **DeepSeek R1** excelled at planning tasks, while **Claude** produced more advanced responses, as shown in [DeepSeek R1 - API Docs](https://api-docs.deepseek.com).
   - Community discussions noted that using **DeepSeek** for simpler tasks saves money, but users still relied on **Claude** for heavier code generation and debugging.
- **Codebase Indexing: A Bare-Bones Approach?**: Some users felt Cursor’s **codebase indexing** wasn't leveraging all project files, as shown in [Context / Codebase Indexing – Cursor](https://docs.cursor.com/context/codebase-indexing).
   - Others defended it, arguing that indexing can improve code suggestions and recommended adjusting settings for better results.
- **RAG Tactics Spark Dev Conversations**: Talk of retrieval-augmented generation methods included vector stores and embedding approaches like **voyage-code-3** described in [this blog post](https://blog.voyageai.com/2024/12/04/voyage-code-3/#:~:text=voyage%2Dcode%2D3%20supports%20much,Matryoshka%20embeddings.).
   - Participants stressed that well-structured embedding and retrieval can minimize mistakes and boost output quality for code-heavy projects.
- **Stepping Away from Claude: Alternatives Abound**: Some users weighed swapping out **Claude** for **DeepSeek** in simpler scenarios, while also exploring [GitHub Copilot](https://github.com/) and [Spark Engine](https://sparkengine.ai).
   - Opinions varied, but the conversation pointed to blending each platform’s strengths for a balanced workflow.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 1.2.2 Debuts with Smoother Cascade**: The team launched **Windsurf 1.2.2** featuring enhanced **Cascade** memory and fixes for laggy conversations, as noted in [Windsurf Editor Changelogs](https://www.codeium.com/changelog). **Cascade** now supports web queries or direct URL inputs, making prompt interactions more dynamic.
   - Despite these fixes, some users still encounter **internal error** spikes that disrupt daily coding tasks, prompting calls for further stability updates.
- **Performance Takes a Hit & Free Credits Slashed**: Frequent errors and latency across channels have led to frustration and lost credits, weakening trust in **Windsurf**. The free plan now offers only **5** premium model credits (down from 50), as indicated in [Pricing](https://codeium.com/pricing).
   - Community feedback highlights concerns that these changes limit new-user onboarding and hamper debugging sessions.
- **DeepSeek's Debut Remains Uncertain**: Conversations suggest that a **DeepSeek** integration into Codeium is not arriving anytime soon, raising worries about missing out on its budget-friendly operation. Some users openly discuss ditching Windsurf if no timeline is announced soon.
   - They also question DeepSeek’s reliability for tool calls, expressing hope for Codeium to address these doubts quickly.
- **Git to the Rescue**: Members recommend version control with **Git** to guard against unexpected errors triggered by **Cascade** and **Windsurf** updates. They cited resources such as [Learn Git Branching](https://learngitbranching.js.org/) to keep code stable and maintain progress.
   - Best practices like tagging milestones and reverting to previous commits help prevent major setbacks when AI-driven changes fail.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity’s R1 Rollout Raises Ruckus**: **R1** replaced older favorites like **O1** and **Grok** on Perplexity Pro, limiting daily queries to **10** and sparking user outcry.
   - Some threatened to switch to **DeepSeek** or **ChatGPT**, citing subpar performance and unclear usage terms, while others demanded refunds via [this reference](https://intercom.help/perplexity-ai/en/articles/10354288-refunds).
- **DeepSeek’s Data Dilemma Divides Opinions**: **DeepSeek** triggers privacy concerns due to Chinese ownership, with users worrying about US-to-China data routing practices.
   - Community feedback mixes caution and curiosity, as some see potential in **DeepSeek**’s R1 model but question **data sovereignty** for sensitive queries.
- **Billion-Dollar Brainchild: The $500B AI Shift**: A rumored **$500 billion** deal could reshape AI according to [this source](https://www.perplexity.ai/page/stargate-project-InQ5ZvKETX6c5I6he1zc_A), fueling speculation about future directions in automation and machine learning.
   - Contributors view this possibility as a major pivot point for advanced research funding, seeing parallels with past booms that propelled **new AI frameworks**.
- **Startup’s $200M Path and S&P Soars**: A member showcased how to bootstrap a **$200 million** startup exit in the [Wingify approach](https://www.perplexity.ai/page/wingify-T9bxT5tHSY2sRduhPzHIXg), focusing on resourceful scaling and investor relations.
   - They also noted the **S&P 500** hit a record closing high, which they believe adds momentum for ambitious founders eager to replicate this success.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Psyche Nudges Collaboration**: Nous introduced **Nous Psyche** on **Solana**, a cooperative training network for open-source generative AI using **heterogeneous compute**, with code shared via [GitHub](https://github.com/PsycheFoundation/psyche).
   - They plan a testnet event on the 30th in partnership with the **Solana Foundation**, referencing mythic inspiration on their [blog](https://nousresearch.com/nous-psyche/) to unify developers.
- **DeepSeek R1 Distillation Gains Ground**: Researchers referenced the **Distilling System 2 into System 1** paper ([arXiv](https://arxiv.org/abs/2407.06023v3)), proposing new improvements for **R1** distillation models.
   - They also pointed to potential synergy with [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) to refine dataset coverage and handling.
- **LLM Live2D Desktop Assistant Debuts**: The new [LLM Live2D Desktop Assistant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant) supports Windows and Mac, featuring voice triggers and full computer control with screen sensing.
   - Its approach merges system clipboard retrieval and interactive commands, giving users a lively character interface for daily tasks.
- **Qwen2.5-VL Breaks OCR Barriers**: The new [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) model shows advanced OCR, including handwriting analysis and robust visual content parsing.
   - Community members praised its strong text recognition capabilities across images, signaling a big leap for multi-modal tasks.
- **Human-Like LLM Paper Sparks Ethical Debate**: The paper [Enhancing Human-Like Responses in Large Language Models](https://arxiv.org/abs/2501.05032) explores refined techniques for **natural language understanding** and **emotional intelligence** in AI.
   - It highlights gains in user engagement while raising concerns about biases, urging a closer look at the human-AI dynamic.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Canvas Connects with O1**: OpenAI introduced a new update enabling ChatGPT Canvas to work with **OpenAI o1** and render **HTML** and **React** code on macOS desktop apps for all tiers.
   - They teased upcoming releases for **Enterprise** and **Edu** users, signaling extended features for professional settings.
- **DeepSeek Rattles Tech Market**: DeepSeek R1 went head-to-head with **O1** and **GPT-4o**, garnering praise for code-generation accuracy and cost advantages.
   - Its debut allegedly wiped out nearly **$600 billion** from major US tech stocks, fueling speculation about bigger disruptions in the AI race.
- **O3 Mini Teeters on Release**: Community buzz indicates **O3 Mini** may launch soon, though some fear it might be just a slight upgrade over **O1 Mini** without true multimodal features.
   - A [tweet from Sam Altman](https://x.com/sama/status/1883294216329281627) promised **100 daily queries** for Plus tier, hinting at extended operator access on the horizon.
- **Token Troubles with Tiktoken**: Users reported uncertainties around **Tiktoken** splitting tokens into single characters, causing confusion in certain inputs.
   - It sparked discussions about special token limits, pointing to research that might explain Tiktoken's irregular merging rules.
- **LangChain’s ChatPrompt & Vector Stores**: Members explored feeding **vector store** documents into LangChain’s **ChatPromptTemplate**, noting limited official guidance.
   - They considered the standard prompt route as a fallback, awaiting any success stories for more robust configurations.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek Derailed by Attacks**: Malicious attacks hammered the **DeepSeek API**, causing intermittent outages and slower response times, as confirmed by [DeepSeek Service Status](https://status.deepseek.com/) and user reports.
   - Some noted surging demand for R1-based offerings, while others speculated on the severity of these attacks, exploring [OpenRouter alternatives](https://openrouter.ai/deepseek/deeps).
- **LLM Inference Providers: Profit or Pitfall?**: **Inference provider profitability** hinged on high utilization to offset fixed costs, as members weighed different pricing models among various services.
   - Some agreed low usage yields slim margins, spurring discussion on synergy with high-traffic releases like [DeepSeek-R1-Nitro](https://openrouter.ai/deepseek/deepseek-r1:nitro).
- **R1 vs O1 Rivalry Ramps Up**: A fresh benchmark hinted **DeepSeek's R1** with Sonnet might outperform **O1** in certain scenarios, backed by data from [R1+Sonnet set SOTA on aider’s polyglot benchmark](https://aider.chat/2025/01/24/r1-sonnet.html).
   - Skeptics insisted **O1 Pro** excels at coding tasks, while some pinned hopes on R1’s 1.58-bit format referenced in [Tweet from Unsloth AI](https://x.com/UnslothAI/status/1883899061893546254).
- **Qwen2.5 & Janus-Pro Storm the Scene**: **Alibaba Qwen2.5-1M** and **Janus-Pro** captured attention for their 1 million token contexts, highlighted in [Qwen's tweet](https://x.com/Alibaba_Qwen/status/1883557964759654608) and [Janus-Pro mentions](https://x.com/_akhaliq/status/1883914398127083665).
   - Commenters regarded them as formidable contenders against **O1**, citing parallels with **DeepSeek-R1** on [DeepInfra](https://deepinfra.com/deepseek-ai/DeepSeek-R1).
- **Aider Allies with CodeGate (and Rust)**: [CodeGate integration](https://docs.codegate.ai/how-to/use-with-aider) authorized **Aider** users to pair program directly in the terminal, toggling between OpenAI and Ollama via API keys.
   - Others tested new **Rust crates** within Aider for boosted context, noting that **architect mode** hides editor model responses, leading them to request [bug fixes on GitHub](https://github.com/Aider-AI/aider/issues/2929).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek Distill Delights**: Participants tested **DeepSeek R1 Distill** across Q3 and Q4 quants, praising its strong knowledge retention and performance on [LLM Explorer directories](https://llm.extractum.io/list/?query=deepseek%20r1).
   - They observed that higher parameter models can boost coding tasks but require careful concurrency and VRAM planning for smooth inference.
- **Chatter UI Confusion**: Members reported issues when hooking **Chatter UI** to LM Studio, tracing faults to incorrect URLs and port conflicts in the [ChatterUI GitHub repo](https://github.com/Vali-98/ChatterUI).
   - They urged verifying local host addresses and aligning LM Studio’s recognized endpoints to stabilize requests.
- **Apple M3 Max Token Tempo**: A few users estimated **DeepSeek-R1** hitting 16–17 tokens per second on an Apple M3 Max with 48GB RAM, underlining hardware constraints.
   - Discussions centered on the chip architecture’s efficiency limits, with some considering load balancing methods in llama.cpp for added speed.
- **MoE Maneuvers**: Enthusiasts examined **Mixture of Experts (MoE)** solutions for specialized tasks, noting potential performance gains in code generation workflows.
   - They emphasized memory considerations and pointed to [MoE resources](https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe) for practical insights and deployment strategies.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Liquid AI Floats into OpenRouter**: Liquid AI introduced new models [LFM 40B](https://openrouter.ai/liquid/lfm-40b), [LFM 3B](https://openrouter.ai/liquid/lfm-3b), and [LFM 7B](https://openrouter.ai/liquid/lfm-7b) via **OpenRouter** as they expanded multilingual coverage.
   - They cited **LFM-7B** as their top pick for enterprise chat, highlighting a strong performance-to-size ratio across major languages.
- **DeepSeek Nitro: Speedy Shortcut or Letdown?**: The **Nitro** variant for DeepSeek R1 launched with claims of faster responses, as seen in the [announcement](https://openrouter.ai/deepseek/deepseek-r1:nitro).
   - Some users reported that it failed to surpass standard R1 in real-world performance, while feedback hinted at heavy user demand causing system strain.
- **Amazon Nova's Abrupt Crash**: The **Amazon Nova** models are down because Amazon Bedrock flagged a surge in usage as a key leak, causing a misleading 400 status code.
   - Teams are rushing to resolve this upstream issue, with official updates expected once the service stabilizes.
- **DeepSeek's Overloaded Ordeals**: Frequent 503 errors and slow response times plagued **DeepSeek R1**, pointing to high traffic as well as potential malicious activity.
   - DeepSeek limited new registrations and faced reliability concerns, highlighting the challenge of accommodating intense user loads.
- **BYOK Gains Steam**: OpenRouter discussions emphasized feeding **BYOK** to mitigate rate limits and control expenses, with a 5% fee on usage.
   - Community members agreed that plugging in personal keys can help dodge bottlenecks, though some worried about cost management complexities.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Janus Jumps Ahead**: DeepSeek introduced [Janus Pro model](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf), boasting advanced reasoning performance without high-end GPU requirements, fueling speculation about new frontiers beyond US-based benchmarks.
   - Participants praised Janus Pro’s improved multimodal understanding, referencing the whitepaper titled **Janus-Series: Unified Multimodal Understanding and Generation Models**, and debated its potential to shift tech market sentiment.
- **Qwen2.5-VL’s Visionary Venture**: Alibaba Qwen revealed [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) celebrating the Chinese New Year, emphasizing **long video comprehension** and advanced visual recognition.
   - Discussion highlighted its capacity to handle **structured data outputs** for finance and commerce tasks, while a [blog post](https://qwenlm.github.io/blog/qwen2.5-1m/) details its context length reaching 1M tokens for broader enterprise use cases.
- **GPRO Gains Ground in PPO**: A robust conversation centered on GPRO’s elimination of the **Value Function** and **Generalised Advantage Estimation (GAE)**, with claims it might address **stuck loss** and early convergence issues in PPO.
   - Users noted GAE’s reliance on a discounted sum hinders scaling in certain scenarios, whereas GPRO’s globally normalized rewards keep training stable, prompting curiosity about integration with open-source RL libraries.
- **DSL Dreams & PydanticAI Pronto**: A member explored [PydanticAI](https://ai.pydantic.dev/) for **structured output** in production-grade generative apps, suggesting it could integrate with **LlamaIndex+LangChain**.
   - They also discussed building a **workout logging app** that converts *natural language* to DSL, referencing the **Microsoft ODSL paper** for partial solutions and emphasizing the challenge of voice-to-DSL pipelines.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek’s Double Punch: R1 & Janus Pro**: DeepSeek’s new R1 model generated major buzz for its open-weight approach, matching or surpassing some established LLMs in reasoning and performance benchmarks with minimal training cost. Industry chatter points to the release of Janus Pro (7B) on Hugging Face, emphasizing transparency and advanced capabilities for both text and images.
   - Skeptics questioned R1’s generalization and reasoning limits, while others praised its impressive leaps in math and coding tasks. As a result, big players like Meta have set up "war rooms" to analyze DeepSeek’s training recipes and cost efficiency.
- **Qwen2.5-VL Lights Up Vision-Language Fusion**: Alibaba’s Qwen2.5-VL debuted with powerful multimodal capacity, supporting long video comprehension and precise localization. Observers compared it to past major releases, noting potential shifts in perception and competition for vision-language models.
   - Developers highlighted the dramatic performance gains on curated tasks, prompting speculation about real-world use cases. Official demos and commits (e.g. Qwen2.5-VL GitHub) show a convergence of advanced image-to-text synergy and lengthy context handling.
- **Nous Psyche Hack & Solana Setup**: Nous Research launched Nous Psyche, a Solana-based cooperative training network aimed at open superintelligence initiatives. Though the concept fueled excitement, news of a hack rattled trust in its security measures.
   - Discussions also touched on broader questions around open labs vs. well-funded closed labs in advancing sophisticated generative models. The hack underscored the importance of rigorous safeguards when merging blockchain ecosystems with AI training.
- **Tulu 3 vs Tulu 4 & Preference Tuning Woes**: Enthusiasts revisited Tulu3, noting the use of off-policy data in preference tuning despite the common stance favoring on-policy approaches. This signaled ongoing complexities in perfecting preference-based training pipelines.
   - Anticipation grows for Tulu4, with users hoping it addresses the hills faced by Tulu3. Discussions highlight the unresolved challenges in scaling preference tuning to broader applications.
- **China’s Multi-Billion AI Policy & Global Ripples**: China’s announcement of a 1 trillion yuan ($137B) investment in AI led to intense speculation about rapidly expanding R&D. Participants noted parallels to US industrial policies like the CHIPS Act but questioned America’s readiness to match large-scale AI funding.
   - Defense-related angles emerged as Republicans might fund AI under great power competition ideals. For engineers, these policies might grant more cutting-edge hardware and incentives, intensifying the global AI race.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek R1 Sparks Debate**: Members noted a **$5M** training cost (referencing **DeepSeek V3**) in its project report, also mentioning [an image attachment as confirmation](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1). **DeepSeek R1** claims to rival older closed models like **o1** at a fraction of the cost, raising questions on open-source competitiveness.
   - Community chatter highlighted [a new SOTA claim on aider’s polyglot benchmark](https://aider.chat/2025/01/24/r1-sonnet.html) from **R1+Sonnet**, and [tweets suggesting strong results](https://x.com/lmarena_ai/status/1882875989610594542) leave many curious about deeper reasoning capabilities.
- **Qwen2.5-VL vs DALL-E 3 Face-Off**: Alibaba launched **Qwen2.5-VL**, a multimodal model aiming to surpass **DALL-E 3** on visual understanding and localization, as seen in their [official announcement](https://x.com/Alibaba_Qwen/status/1883954247743725963). The model also competes with **Stable Diffusion** on specific benchmarks, emphasizing advanced image generation features in [their Qwen collection](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5).
   - Users compared metrics on [GenEval and DPG-Bench](https://x.com/LiangWenfeng_/status/1883918900741763293), noting **Janus-Pro-7B** from **DeepSeek** also contends in the multimodal space, fueling a broader conversation on cost-effectiveness and real-world applicability of these newer models.
- **Operator & Reasoning Models Make Waves**: Participants praised **Operator** for generating initial codebases quickly, but raised concerns on handling complex sites and video sampling rates, shown in [this video demo](https://x.com/klazuka/status/1883880742322888903). In parallel, discussions on reasoning models like **R1** suggested advanced agentic capabilities for coding tasks and beyond.
   - Some credited [function calling benchmarks](https://x.com/_philschmid/status/1883055262669349287) for pointing out multi-step constraints, adding perspective on how **DeepSeek R1** and others handle intricate workflows when integrated into development pipelines.
- **Model Context Protocol (MCP) Gathers Momentum**: Members showed enthusiasm for **MCP** as a unifying approach to integrate AI features across tools, referencing servers built in **Go**, **Rust**, and even **assembly**, per [MCP server repos](https://github.com/modelcontextprotocol/servers). They compared how it interlinks with **Obsidian** for transcription and documentation through plugins like [mcp-obsidian](https://github.com/MarkusPfundstein/mcp-obsidian).
   - Plans for an **MCP party** encourage community feedback and synergy, with a call to review the [latest specs](https://spec.modelcontextprotocol.io/specification/2024-11-05/architecture/#capability) and tutorials, highlighting a strong interest in consistent cross-application protocols.
- **Latent Space Launches a New Pod**: A brief mention announced a fresh episode on the **Latent Space** podcast, [shared here](https://x.com/latentspacepod/status/1883354909367787565).
   - The community welcomed the update, anticipating discussions that might delve further into these emerging AI technologies and collaborative initiatives.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Layer Convergence & Tokenization Tactics**: In [an ICLR 2023 paper](https://openreview.net/forum?id=wlMDF1jQF86), members observed how **Layer Convergence Bias** makes shallower layers learn faster than deeper ones.
   - Another group applauded the new **Causally Regularized Tokenization** from [Armen Aghajanyan's paper](https://arxiv.org/pdf/2412.16326), noting improved efficiency in **LlamaGen-3B**.
- **DeepSeek R1 & GRPO Gaps**: Participants questioned **DeepSeek**'s cheaper-chip claims for R1, referencing approximate training costs of **$1.6M** and a shortage of open-source details.
   - They also found few genuine **GRPO** implementations in [TinyZero](https://github.com/Jiayi-Pan/TinyZero) or [SimpleRL](https://github.com/hkust-nlp/simpleRL-reason), hinting that real R1 runs rely mostly on **PPO**.
- **AlphaZero Evolution & Curiosity-Driven AI**: Adopters recognized **AlphaZero**'s streamlined design but noted that practical setups rarely jump straight to its techniques.
   - Some pointed to **empowerment** concepts ([Wikipedia entry](https://en.wikipedia.org/wiki/Empowerment_(artificial_intelligence))) and curiosity-driven methods as flexible approaches for future large-scale training.
- **Scaling Laws & The 20-Token Trick**: A [Chinchilla library analysis](https://github.com/kyo-takano/chinchilla/blob/master/examples/llm/main.ipynb) suggested the **20-tokens-per-parameter** rule nearly matches fully-optimized Chinchilla setups.
   - Community members linked this to flat minima in tokens-per-parameter ratios, indicating minor deviations may not hurt performance drastically.
- **Interpretability & Multi-turn Benchmarks**: Some users highlighted **verified reasoning** in training as a new priority for interpretability, focusing on how LLMs reason rather than just outputs.
   - Meanwhile, frameworks like **scbench**, **zeroSCROLLS**, and **longbench** are being integrated, though their multi-turn nature may require distinct implementation strategies.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **System Prompt Panache**: You can now set up a **system prompt** on a project or global level in [Bolt](https://x.com/boltdotnew/status/1883949779572646008), letting you inject your preferred libraries and methods from the start.
   - Community members are trading tips on the best ways to shape **Bolt's** behavior, with calls to share advanced usage tricks for smoother development.
- **Structuring & Splitting Strategies**: Members debated the impact of too-rigid planning on creativity, referencing cycles of restarts and the need for flexible approaches when dividing complex components.
   - One user recommended a systematic approach with a **NEXTSTEPS.md** outline, noting how structured migrations help maintain clarity without stifling new ideas.
- **Guidelines as a Safety Net**: Adhering to **GUIDELINES.md** improved stability, ensuring each component was built in sequence and integrated with a properly managed context window.
   - Participants credited these guardrails for avoiding chaotic merges, with stable documentation practices paving the way for consistent progress.
- **Bolt's Billing & Error Bruises**: Some folks complained about massive token usage and frequent **rate limits**, mentioning issues with refunds and cost discrepancies.
   - Error messages and network failures left them searching for professional help, as **Bolt** sometimes consumed large amounts of tokens without delivering results.
- **Supabase Roles Earn a Win**: A user overcame complicated **Supabase** policies to build multiple login roles, including super admin and admin, tackling recursion pitfalls.
   - Integrations with Netlify and GitHub were also explored, though private repos remain off-limits for now, prompting further modifications to **Bolt's** core features.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Client Complaints & Voice Chat Groans**: Developers wrestled with the inability to dynamically update **MCP client** tools without restarts, calling for clearer voice integration docs in [multimodal-mcp-client](https://github.com/Ejb503/multimodal-mcp-client).
   - Much attention went toward **server config** improvements and minimizing proprietary API reliance, especially for **Kubernetes** deployments.
- **Variance Log Tool Catches Oddities**: The **MCP Variance Log** solution collects low-probability conversation events in a [SQLite database](https://github.com/truaxki/mcp-variance-log) for user data analysis.
   - Adopters pointed to the **Titans Surprise mechanism** as inspiration for an approach that can bolster long-term memory across agentic workflows.
- **KoboldCPP & Claude Make New Connections**: A fresh **KoboldCPP-MCP Server** fosters AI collaboration among **Claude** and other MCP apps as shown on [GitHub](https://github.com/PhialsBasement/KoboldCPP-MCP-Server).
   - Community members noted it paves the way for more synchronized tasks and deeper AI-to-AI interactions.
- **Inception Server Runs Parallel LLM Missions**: The **MCP Inception server** allows concurrent queries with various parameters, detailed in [its repo](https://github.com/tanevanwifferen/mcp-inception).
   - Developers plan to extend functionality for cryptocurrencies via scraping, hinting at expanded use cases.
- **Shopify Merchants Chat With Claude**: An **MCP server for Shopify** uses Claude for store analytics, as shown in [this repo](https://github.com/amir-bengherbi/shopify-mcp-server).
   - Current endpoints focus on products and orders, giving merchants a path to direct AI-driven data insights.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **HeyGen Scenes & ElevenLabs Tones**: One user showcased a workflow with **HeyGen** and **RunWayML’s Act-One** to produce lifelike avatar videos that appear to be listening, linking to [UnrealMysteries.com](https://UnrealMysteries.com).
   - They also revealed an **ElevenLabs** voice named "Thomas," evoking a **HAL** vibe for extra flair.
- **NotebookLM for Podcast Summaries**: A user employed **NotebookLM** to condense weekly news into a podcast format, praising its quick summarization.
   - Others hope for stronger prompts to improve audio content creation and push the tool's capabilities.
- **Mixing HeyGen & MiniMax**: Members experimented with hybrid content, combining **HeyGen** stills and insights from **MiniMax** for extended videos.
   - They observed more engaging narratives than using either technology alone, sparking further creative attempts.
- **NotebookLM Constraints & Confusion**: Members encountered missing linked sources in **NotebookLM** after UI changes, prompting concerns about lost references.
   - Another user discovered a **1000-note** limit, urging clearer documentation for advanced usage.
- **Language Twists & PDF Page Disputes**: Some folks wrestle with default language settings, toggling URLs like [notebooklm.google/?hl=es](https://notebooklm.google/?hl=es) for better control.
   - Others notice partial PDF pages failing to produce insights, pointing to inconsistent page references in **NotebookLM**.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Hunyuan-Video Triumphs on 12GB VRAM**: The **hunyuan-video model** runs effectively on as little as **12GB VRAM**, delivering local image-to-video processing that appeals to many developers.
   - Community members praised its usability for casual experimentation, referencing [Webui Installation Guides](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides#amd-forge-webui-with-zluda) for advanced tweaks.
- **Kling AI Lacks Image-to-Video Features**: Users noted that **Kling AI** stands close to **hunyuan** in image quality, but it doesn't support video conversion yet.
   - They found the missing function disappointing for a complete pipeline, with some hoping updates will address this gap soon.
- **Forge vs Swarm for New Image Creators**: **Forge** and **Swarm** emerged as popular picks for newcomers seeking simpler local **AI image generation** tools.
   - Advanced users recommended **ComfyUI** for more flexibility, but they cautioned beginners about its extra complexity.
- **Stable Diffusion Prefers 32GB RAM or More**: A well-equipped system with **32GB RAM** is best for **Stable Diffusion**, and **64GB** ensures a smoother experience.
   - Members running **RTX 4090** or **AMD 7900XTX** reported fewer hardware conflicts once they upgraded their memory.
- **Deepseek Requires Massive 1.3TB VRAM**: The **Deepseek** lines, including **V3** and **R1**, need more than **1.3TB of VRAM** at full precision, which surpasses consumer-level gear.
   - People with multi-GPU clusters like **A100** or **H100** cards can handle these models, forcing everyone else to look for smaller alternatives.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DeepSeek Doubles with TinyZero & Open R1**: The [TinyZero](https://github.com/Jiayi-Pan/TinyZero) project replicates **DeepSeek R1 Zero** in a clean, accessible way, offering images and details for contributors. Meanwhile, [Open R1](https://github.com/huggingface/open-r1) by Hugging Face provides a fully open take on **DeepSeek-R1**, encouraging collaborative development.
   - Both repos invite community involvement, showcasing a strong push toward reproducible research in HPC contexts.
- **Taming NCCL Timeouts for HPC**: Multiple members reported **NCCL timeouts** during multi-node training and asked for best practices in debugging. They profiled GPU jobs and considered advanced strategies to handle timeouts in large-scale setups.
   - The community documented common pitfalls including mismatch of CUDA versions, emphasizing the need for robust HPC debugging tools.
- **Adam Paszke’s Mosaic GPU DSL Magic**: Renowned **Adam Paszke** discussed his **Mosaic GPU** DSL in a live YouTube session, emphasizing low-level GPU programming. Community members can find supplementary materials on [GitHub](https://github.com/gpu-mode) and join [Discord](https://discord.gg/gpumode) for active learning.
   - The talk promises deeper exploration of layout systems and tiling for GPU optimization.
- **JAX Runs FP8 on Legacy GPUs**: A [GitHub discussion](https://github.com/jax-ml/jax/discussions/26077) revealed **JAX** can use **fp8** on Nvidia GPUs with sm<89, defying typical hardware constraints. PyTorch users reported failures on older GPUs, sparking intrigue about JAX’s workaround.
   - This gap piqued interest in how exactly JAX bypasses standard limitations, prompting further exploration of library internals.
- **Arc-AGI Expands with Maze & FSDP**: The **Arc-AGI** environment gained polynomial equations, maze tasks, and more examples in reasoning-gym, referencing algorithms from CLRS. Meanwhile, **Tiny-GRPO** introduced **FSDP support**, slashing VRAM usage and boosting efficiency.
   - Members also floated ideas on family relationship data and **GSM8K** templates, planning to push to the HF hub for user-friendly downloads.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Docs Vanish, Then Reappear**: The **Mojo documentation** abruptly went offline due to Cloudflare hosting troubles, sparking user frustration.
   - The dev team apologized and confirmed the docs were **back online** with updated references from the [Mojo GitHub changelog](https://github.com/modular/mojo/blob/nightly/docs/changelog.md).
- **New GPU Package API Hits Nightly**: Users confirmed that the **GPU package API** docs landed in the nightly release, offering advanced GPU functionality in **Mojo**.
   - They welcomed this addition as a significant improvement, pointing to the [changelog](https://github.com/modular/mojo/blob/nightly/docs/changelog.md) for recent updates.
- **CSS Struct Fluent API Sparks Warnings**: A developer built a `struct` to generate CSS using a **fluent API** style but encountered unused value warnings in Zed Preview.
   - They tried `_ = ` to suppress the warnings yet wanted a cleaner solution to maintain code clarity.
- **List and Representable Trait Tangle**: A user wrestled with `List[Int]` passing into a function, discovering **Int** wasn't recognized as **Representable** by the compiler.
   - They highlighted possible conditional conformance issues in [int.mojo](https://github.com/modular/mojo/blob/nightly/stdlib/src/builtin/int.mojo#L1146) and the [List module](https://github.com/modular/mojo/blob/nightly/stdlib/src/collections/list.mojo#L441).
- **Unsafe Pointers & Function Pointer FFI Bumps**: Working with **UnsafePointer** revealed shifting object identity in value structs, causing confusion as pointers moved independently.
   - They also noted that function pointer FFI remains unreliable in **Mojo**, with partial C ABI compliance and limited documentation.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Presenter Pizzazz & Multi-Agent Magic**: LlamaIndex introduced [Presenter](https://twitter.com/llama_index/status/1883307955782901926), a **multi-agent workflow** that creates visually rich slides with **Mermaid diagrams**, script generation, and **report generation** all in one pipeline.
   - Community members praised *Presenter’s accessible structure*, showcasing how these references could evolve into advanced **presentation-building agents** that orchestrate complex steps.
- **Doc Driller & Google-Style Gains**: MarcusSchiesser released [a fully open-source template](https://twitter.com/llama_index/status/1883675662839636427) for **multi-step document research agents**, inspired by Google's deep research approach.
   - Users mentioned the template’s capacity to handle **complex research workflows**, noting it addresses a common demand for integrated **analysis and referencing** in advanced projects.
- **Scaleport’s Swift Claim Crunch**: Scaleport AI formed [a partnership](https://twitter.com/llama_index/status/1883929949205336509) with a travel insurer to automate **claim estimation** from medical reports using **LlamaIndex**, featuring **OCR** for data extraction.
   - Community members highlighted *significant time savings*, emphasizing how these methods showcase **AI-driven risk analysis** for more efficient insurance processes.
- **DeepSeek’s Deft LlamaIndex Integration**: LlamaIndex now integrates with the [DeepSeek-R1 API](https://twitter.com/llama_index/status/1883986763380842864), supporting **deepseek-chat** and **deepseek-reasoner** for advanced calls in a unified environment.
   - Developers affirmed the boosted synergy, referencing the [DeepSeek docs](https://api-docs.deepseek.com/) to enable **API-key** onboarding and seamless model usage.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Japan Regulation Freedoms: Cohere Remains Unscathed**: Cohere discovered that **new Japanese AI regulations** focusing on advanced computing are unlikely to affect them until May 2025, since their language models fall outside key restrictions.
   - The rules specifically target powerful chips and expansions, leaving **Cohere** untouched for now, while their legal team keeps a close watch in case of amendments.
- **Dashboard Dilemmas: UI Overhaul on Cohere's Horizon**: Community feedback flagged **confusing interface elements** on the [Cohere dashboard](https://dashboard.cohere.com/), highlighting mirrored button layouts.
   - Suggested fixes included bigger calls to action for Discord and email support, with users pushing for a more **streamlined** design approach.
- **Bare-Bones Audio: No TTS or STT on Cohere**: **Cohere** officially confirmed a pure focus on large language models, offering no built-in text-to-speech or speech-to-text features.
   - This clarity ended speculation about audio capabilities, reaffirming that **LLM** support is the platform's primary strength.
- **ChatCohere Code Chronicles: Step-by-Step LLM Setup**: Developers showcased how to define **ChatCohere**, bind tools with `bind_tools`, then invoke the LLM with structured messages for advanced tasks.
   - Some mentioned **reverse planning** as a final-step check, emphasizing that Cohere sticks to text-based solutions rather than TTS or STT integrations.
- **Tool Tiers: Cohere's Multi-Step Approach**: Cohere's documentation details a staged **multi-step** flow, from user prompt retrieval to final text generation.
   - Community members praised the **systematic** breakdown, underscoring how sequential reasoning refines complex outputs and ensures relevant data is pulled at each stage.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Open Source Image Analysis Gains Traction**: People sought an open-source model to handle image prompts, turning to frameworks like [Taggui](https://taggui.com) for tagging. They struggled to find a definitive option that excels at both tagging and response generation.
   - Some advocated for easier setups that don't demand advanced configuration. Others noted the market lacks a clear front-runner, prompting more experimentation.
- **DeepSeek's R1 Model Trips Early**: Multiple users reported incomplete reasoning and chat template errors with [DeepSeek R1](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF). They mentioned difficulty running it locally without a patch for stable performance.
   - Benchmarks hinted at results comparable to LLAMA, though no one confirmed fully reliable output. Some called for further testing before trusting the model in real scenarios.
- **Local Document Analysis Piques Curiosity**: Enthusiasts want to keep data private by exploring tools like **PDFGear** for local text indexing. They aim to query personal documents without relying on cloud services or uploads.
   - Opinions varied on how to handle complex PDFs and large volumes of text. People requested detailed examples and simpler pipelines to streamline these processes.
- **GPT4All Waits for DeepSeek R1 Support**: Community members asked when **DeepSeek R1** would reach GPT4All in an official, easy-to-install manner. Contributors indicated integration is still in progress, but gave no exact release window.
   - They want a one-click setup that doesn't require extensive manual tweaking. Some suggested a fix is close, yet no official statement has been released.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC Monday Mayhem**: The **Advanced LLM Agents MOOC** starts on **January 27th** at **4:00PM PST** and continues until **April 28th**, providing weekly [livestreams](https://www.youtube.com/live/g0Dwtf3BH-0) and resources via the [course site](http://llmagents-learning.org/sp25).
   - Attendees can [sign up here](https://forms.gle/9u6HdVCWXgws16go9) and watch replays on **YouTube**, with no immediate deadlines or in-person attendance available for non-Berkeley students.
- **Certificates & Confusions Collide**: Members reported **Fall'24 MOOC certificates** still pending, with staff announcing upcoming news and encouraging patience.
   - Some also mentioned missing **confirmation emails** after enrolling, echoing *In the same boat...*, while staff promised official updates soon.
- **Hackathons & No Hangouts**: Enthusiasts asked about **hackathon opportunities**, and staff noted strong interest but no final plan for the semester.
   - Others sought in-person access but learned only official Berkeley students are allowed on site, so everyone else relies on the virtual platform.
- **Substack Enigma Surfaces**: A curious [Substack link](https://substack.com/home/post/p-154577981) emerged in **#mooc-readings-discussion**, offered with scant context.
   - The community left it hanging, waiting for any follow-up to clarify the resource shared.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Gradient Guidance Gains Ground**: Confusion on `Tensor.gradient` usage erupted as the doc states *'Compute the gradient of the targets with respect to self'* though context suggests it means *'Compute the gradient of self with respect to the targets'*, sparking discussion.
   - Participants proposed a doc revision for accuracy, noting that **tensor.py** may require further clarifications for future references.
- **STRIDE vs FLIP Fling**: A rebranding from **STRIDE** to **FLIP** was recommended to avoid generic naming, aiming for sharper clarity in the codebase.
   - Contributors supported the shift, citing that lingering references can complicate updates and slow down feature integration.
- **Monday Madness: Meeting #55**: Scheduled for 6am Monday San Diego time, **Meeting #55** plans to discuss recent multi gradient designs, company updates, and projects like **resnet** and **bert**.
   - Attendees expect to address new project bounties, intending to refine upcoming tasks and deadlines.
- **BobNet Branding Baffled**: Questions surrounded **BobNet** after a [GitHub reference](https://github.com/qurAI-amsterdam/bobnet) implied bounding box usage, yet the code is ordinary feed-forward.
   - Members emphasized naming clarity, noting mismatches between title and functionality can mislead new adopters.
- **Formatting Fracas in Tinygrad**: Users debated official formatting tools, with some citing **Black** while others pointed to **Ruff** in the [pre-commit config](https://github.com/tinygrad/tinygrad/blob/master/.pre-commit-config.yaml#L7).
   - Consensus emerged that **Ruff** standardizes formatting effectively, urging contributors to follow the recommended approach.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Federated Frenzy in Torchtune**: Contributors proposed creating **N splits per node** for **Torchtune** federated learning, merging weights with the saved **optimizer state** after each chunk is trained.
   - Some questioned how to streamline training *'without excessive interruptions'* while others discussed the potential synergy with **torch distributed** and **raylib** approaches.
- **Partial Parameter Pandemonium**: Community members debated the **performance gains** of applying **opt-in backward hooks** so certain parameters get updated as soon as their gradients are ready.
   - They also weighed a strategy to only optimize the **output projection** with a separate updater, with concerns over the complexity of running multiple optimizers in parallel.
- **EBNF Edges Out Regex**: A shift away from **regex** emerged after a member claimed it 'looks like a misformatted tokenizer,' prompting interest in **EBNF grammars** for better readability.
   - Some found **EBNF** more verbose yet easier to follow, with direct quotes praising it as *'human readable while still robust.'*
- **Deepseek's Dashing Janus Series**: A user critiqued **Deepseek** for updating too often, referencing [this report](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf) on **Janus-Series: Unified Multimodal Understanding and Generation Models**.
   - Others bantered over the potential reach of these multimodal features, with one quipping *'They need to chill'* amid ongoing comparisons to outdated models.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 1.0 keeps Python interpreter**: The **OpenInterpreter** project had a commit **yesterday**, is planning a **1.0** release with Python interpreter integration, and is keeping the site in a minimal state until a major launch, as confirmed in [their GitHub repository](https://github.com/OpenInterpreter/open-interpreter).
   - They promise a bigger refresh once the main launch happens, with community feedback focusing on user interaction expansions.
- **DeepSeek R1 triggers 400 errors**: A user reported a **400 error** with the **Deepseek_r1** model, configured via `api_base` pointing to [https://api.deepseek.com](https://api.deepseek.com), yielding a **BadRequestError** due to the model’s absence.
   - The conversation indicated an **invalid_request_error** under **OpenAIException**, causing confusion for those attempting to run `$ interpreter -y --profile deepseek.yaml` in a workflow.
- **DeepSeek matches OpenAI-o1 in tests**: Community members noted **DeepSeek-R1** and smaller **Distill-Qwen-1.5B** models achieving performance on par with **OpenAI-o1** across math and code tasks, referencing [deepseek-r1 library info](https://www.ollama.com/library/deepseek-r1).
   - They also highlighted **DeepSeek**’s tool-calling requirements in OS mode and possible issues integrating a vision model, aiming to refine usage for advanced scenarios.
- **Local usage with Ollama and Llamafile**: Efforts to run **Open Interpreter** fully on local resources were demonstrated using [Ollama](https://www.ollama.com/) and [Llamafile](https://github.com/Mozilla-Ocho/llamafile), echoing the command `interpreter --local` from the official [running-locally guide](https://docs.openinterpreter.com/guides/running-locally).
   - The chat centered on whether enabling a vision model in a multi-model setup is necessary, prompting calls for clarity on usage in combined frameworks.
- **DSH - AI Terminal invites contributors**: A project named **DSH - Ai terminal** is seeking improvements to its open-source app, referencing [their GitHub repo](https://github.com/gokul6350/dsh-shell).
   - Developers were encouraged to star the project and share user feedback to enhance its features going forward.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **DeepSeek R1 Under Scrutiny**: Preliminary benchmarks indicate **DeepSeek R1** matches **o1-mini** and **Claude 3.5 Sonnet**, contradicting claims it rivals **o1** on challenging LLM benchmarks, as seen [here](https://x.com/JJitsev/status/1883158738661691878).
   - Participants questioned its efficacy on olympiad-level **AIW problems**, referencing [this paper](https://arxiv.org/abs/2406.02061) to gauge its true capabilities.
- **Pipeline Gains Audio Upgrades**: A suggestion emerged for **audio widgets** to compare augmentation effects, integrating distortions from libraries like **DeepSeq** or **O1**.
   - Contributors emphasized the convenience of interactive features for examining and refining audio changes, targeting pipeline improvements.
- **Testing Pipeline Rolls Out**: A user shared a [development-phase pipeline](https://colab.research.google.com/drive/1tc4YgdsZeEtsZCdnawYaEC7b12NBQfYt) revealing initial progress after a busy travel day.
   - They invited feedback on the pipeline’s features and functionality, focusing on how best to explore audio augmentation capabilities.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GitHub Guff Grows**: A member pointed out random spam issues flooding the GitHub repository, overwhelming valuable reports on project bugs and features.
   - This sparked frustration as others discussed possible filters and more vigilant triaging to curb the clutter.
- **Language Lines: Natural vs Code**: A user asked about a dependable way to detect whether text is **natural language** or structured code such as HTML or Python.
   - They floated the idea of specialized classifiers to cleanly categorize textual formats.
- **Dspy + Deepseek Dilemma**: A participant tried optimizing **dspy + deepseek** for 70B COT examples but couldn't clarify the exact steps to streamline the process.
   - Others chimed in with questions about runtime and memory constraints, highlighting complexities in large-scale optimization.
- **BSR’s Six-Hour Standstill**: A user ran a **BSR** example for six hours without convergence, raising eyebrows about the approach’s practicality.
   - This prompted a debate on alternative tactics or whether the massive run time was even worth the outcomes.
- **PyPI Pressure for FastAPI**: A developer needed an updated **RC on PyPI** to match modern **FastAPI** dependencies since outdated packages caused broken installs.
   - They pointed to a fix on the main branch from three weeks ago, urging maintainers to publish a fresh release.



---



## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Deepseek Debut & GRPO Guess**: Members asked about the **deepseek algorithm** and whether anyone is reproducing it, referencing a possible link to **grpo** in **trl**.
   - One participant suggested it *may refer to grpo*, indicating renewed interest in advanced RL methods, though no official confirmation was provided.
- **H200 vs 5090 GPU Gamble**: A user weighed purchasing **2x 5090s** or **1x H200**, noting that the H200 has more RAM but uncertain performance benefits.
   - They cited cost and speed concerns, hoping for real-world feedback on which setup best supports **heavy AI workloads**.
- **Stalled RL Framework Support**: A member noted the lack of **trl**’s online RL trainers, expressing a desire for broader RL library integration.
   - Another response, however, insisted it was *most likely not* going to happen, amplifying doubts about extended RL support.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla Gains Prompt Peek**: A user asked about system prompts for models not supporting function calls on the **Berkeley Function Call Leaderboard**, leading to a reference to the [Gorilla GitHub code (lines 3-18)](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py#L3-L18).
   - This repository focuses on training and evaluating LLMs for **function calls**, offering the needed system messages for non-function versions.
- **Gorilla's Leaderboard Resource Emerges**: The [Gorilla function call leaderboard's code](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py) was shared as the seat of relevant system prompts.
   - It contains definitions for **function-inspired** prompts and can guide users seeking references for non-function usage.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **2025 Crystal Ball & Real-Time Rendezvous**: On **January 28**, an event titled **2025 Crystal Ball: Real-Time Data & AI** will feature **Rayees Pasha (RisingWave Labs)**, **Sijie Guo (StreamNative)**, and **Chang She (LanceDB)**, highlighting how real-time data boosts AI, as seen in [this Meetup link](https://www.meetup.com/streaming-stories/events/305736950/).
   - They stress that AI’s potential stays underused without low-latency data pipelines, pointing to **Apache Iceberg** as a key approach for powering emerging analytics across industries.
- **Industry Leaders Forecast 2025 Innovations**: Panelists predict that **real-time data streaming** will shape new workflows for AI by 2025, granting significant advantages in operational efficiency and swift decision-making.
   - They plan to tackle evolving data infrastructure hurdles, from consumer applications to enterprise use cases, underscoring the synergy between streaming technologies and AI’s growing demands.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Paper Reading Club Meets Again**: The **Paper Reading Club** returns this week with a scheduled session, as shared in the [Discord event link](https://discord.com/events/1089876418936180786/1329844319703662664). Attendees can expect an in-depth look at AI research, centering on advanced papers that resonate with an engineering audience.
   - Organizers encourage participants to join and share their thoughts for a lively exploration of **cutting-edge discussion** in a communal setting.
- **Discord Events Spark Community Involvement**: Beyond the Paper Reading Club, various **Discord events** are highlighted to keep members engaged with new activities this week. Users are invited to join the ongoing conversations, giving them a chance to exchange technical insights.
   - Leaders remind everyone to check the announcements channel for real-time updates, emphasizing the importance of **active participation** in these collaborative gatherings.



---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1332440841297461248)** (1123 messages🔥🔥🔥): 

> `Model Fine-Tuning, Dynamic Quantization, Hardware Requirements, Agentic AI, Training Datasets` 


- **Exploring Model Fine-Tuning Techniques**: Discussion revolves around fine-tuning models like Qwen2.5-VL and the implications of training with specific configurations, including the use of LoRA adapters and their effect on embeddings.
   - Participants share insights on using existing datasets, as well as experiences with leveraging various notebook resources for effective model training.
- **Dynamic Quantization of Models**: Dynamic quantization of models such as DeepSeek-R1 and Qwen2.5 is a focal point, highlighting its efficiency in reducing model size without losing coherence in outputs.
   - Users express interest in the potential availability of 1-bit quant versions while questioning the necessity of such models given their size and diminished performance.
- **Hardware Considerations for AI Models**: Participants debate the benefits of different GPU configurations for running AI models effectively, discussing options like 3060s vs 4090s and their suitability for running larger models.
   - The consensus indicates that while more GPUs can offer broader VRAM, power consumption and performance Need careful consideration.
- **Importance of Groundwork in Machine Learning**: Several users emphasize the necessity of foundational knowledge in machine learning, referencing the importance of understanding the context and underlying principles behind AI models.
   - Recommendations are made for relevant online courses and resources to ensure upcoming developers are equipped with essential skills for AI work.
- **Utilization of Existing AI Tools and Libraries**: Discussions cover the usage of popular libraries like VLLM and FastAPI in running inference, with a focus on how they manage performance metrics and streamline model deployment.
   - Participants stress that leveraging these tools effectively can enhance productivity while cautioned against blindly trusting generated code without human review.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html">CPU &#8212; vLLM</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb">Google Colab</a>: no description found</li><li><a href="http://jasonwryan.com/blog/2012/03/17/vampires/">A Taxonomy of Help Vampires - jasonwryan.com</a>: no description found</li><li><a href="https://x.com/RussellBal/status/1883283659396104263">Tweet from russell@unturf. (@RussellBal)</a>: Every old 2u dual xeon is going to be running an agent, and the brains will be local, yes it&#39;s slower than GPU but with reasoning models we just let them cook, right?xeon runnng r1 distilled to 8B...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S">unsloth/DeepSeek-R1-GGUF at main</a>: no description found</li><li><a href="https://x.com/Alibaba_Qwen/status/1883557964759654608">Tweet from Qwen (@Alibaba_Qwen)</a>: We&#39;re leveling up the game with our latest open-source models, Qwen2.5-1M ! 💥 Now supporting a 1 MILLION TOKEN CONTEXT LENGTH 🔥Here&#39;s what’s new:1️⃣ Open Models: Meet Qwen2.5-7B-Instruct-1M ...</li><li><a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (All Versions) - a unsloth Collection</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1883899061893546254">Tweet from Unsloth AI (@UnslothAI)</a>: Introducing 1.58bit DeepSeek-R1 GGUFs! 🐋DeepSeek-R1 can now run in 1.58-bit, while being fully functional. We shrank the 671B parameter model from 720GB to just 131GB - a 80% size reduction.Naively q...</li><li><a href="https://x.com/tom_doerr/status/1883517455445733580">Tweet from Tom Dörr (@tom_doerr)</a>: Unsloth: Faster LLM fine-tuning library</li><li><a href="https://gist.github.com/darkacorn/01b0db678d4d91b371e4eba274b911a6">gist:01b0db678d4d91b371e4eba274b911a6</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://huggingface.co/unsloth/Hermes-3-Llama-3.1-8B/tree/main">unsloth/Hermes-3-Llama-3.1-8B at main</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M">Qwen/Qwen2.5-14B-Instruct-1M · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen/Qwen2.5-VL-7B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://weechat.org/">WeeChat, the extensible chat client</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ib7mg4/i_spent_the_last_weekend_optimizing_the_deepseek/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: A course on aligning smol models.</a>: A course on aligning smol models. Contribute to huggingface/smol-course development by creating an account on GitHub.</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">Datasets 101 | Unsloth Documentation</a>: Learn all the essentials of creating a dataset for fine-tuning!</li><li><a href="https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md">trl/docs/source/grpo_trainer.md at main · huggingface/trl</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT">FreedomIntelligence/medical-o1-reasoning-SFT · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/prithivMLmods/Llama-Song-Stream-3B-Instruct-GGUF">prithivMLmods/Llama-Song-Stream-3B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://russell.ballestrini.net/">
    Russell Ballestrini
  </a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/1078">[Bug(CMake 3.17)] CUDA::cublasLt not found but can be specified absolutely · Issue #1078 · ggerganov/llama.cpp</a>: Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...</li><li><a href="https://github.com/ggerganov/llama.cpp.git">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://docs.unsloth.ai">Welcome | Unsloth Documentation</a>: New to Unsloth?</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://www.coursera.org/specializations/machine-learning-introduction">Machine Learning</a>: Offered by Stanford University and DeepLearning.AI. #BreakIntoAI with Machine Learning Specialization. Master fundamental AI concepts and ... Enroll for free.</li><li><a href="https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics">sebastiandizon/genius-song-lyrics · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1332482886506385429)** (20 messages🔥): 

> `NLP Course Completion, SmoLlm Fine-Tuning, Ollama Temperature Settings, AI Generated Text Detection, DeepSeek R1 vs OpenAI O1` 


- **Nailed the NLP Course with High Praise**: A member announced, *'Finally I completed NLP course with the highest grade and with AI generated text detection software,'* and is now excited to pursue **LLMs**.
   - They also completed an **information retrieval course** for **RAG systems**, finding it quite **useful**.
- **Exploring SmoLlm Fine-Tuning**: A member inquired if anyone has fine-tuned the **SmoLlm models** and whether it is worth it.
   - Another member shared their experience of successfully fine-tuning with **unsloth** and running it with **ollama** without temperature settings.
- **Ollama's Default Temperature Revealed**: In response to a question about ollama's default temperature, a member cited *'According to guru from Ollama, it's 0.8 (written in their docs)'*.
   - This clarification was appreciated, as it helped save **time** for the inquisitor.
- **Humorous Take on CUDA Memory**: One member made a humorous remark about their body being a machine that converts water and chips into **RuntimeError: CUDA out of memory**.
   - This sparked laughter, with another jokingly suggesting *'How do you install more VRAM'* and a playful reference to **downloadram.com**.
- **DeepSeek R1's Rise in AI Models**: A YouTube video link titled *'DeepSeek R1 trimmed to 1.58bit 131 GB with unclothe #ai'* was shared, showcasing its open-source capabilities.
   - The video highlighted that **DeepSeek-R1** is rivaling **OpenAI's O1 reasoning model**, stirring interest among members.



**Link mentioned**: <a href="https://www.youtube.com/shorts/IzNQuD-FvIk">DeepSeek R1 trimmed to 1.58bit 131 GB with unclothe #ai</a>: DeepSeek-R1 has been making waves recently by rivaling OpenAI&#39;s O1 reasoning model while being fully open-source. We explored how to enable more local users ...

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1332441133275418654)** (371 messages🔥🔥): 

> `Unsloth Errors, Dataset Formatting for Fine-Tuning, Text Completion and Chatbot Datasets, DeepSeek R1 Deployment, Model Deployment on Different Hardware` 


- **Unsloth Errors during Fine-Tuning**: Users reported issues while running training notebooks, specifically errors like 'RuntimeError: Unsloth: You must call FastLanguageModel.for_inference(model) before doing inference for Unsloth models.' after running all cells.
   - Others confirmed similar issues, prompting fixes in the notebooks to resolve these errors for a smooth training process.
- **Formatting Datasets for Fine-Tuning**: Several users discussed how to format their datasets for various model training, including replacing the 'instruction', 'input', and 'output' fields in the notebooks.
   - Specific examples and guidance were sought on how to adapt datasets, such as the Wikimedia dataset, to fit the expected format for successful fine-tuning.
- **Using Jupyter Notebook for Unsloth**:  A user inquired about the possibility of using Jupyter Notebook instead of Colab for their Unsloth fine-tuning projects due to preferences against Google services.
   - Confirmation was given that it is indeed possible to utilize Jupyter Notebook for operating Unsloth models.
- **DeepSeek R1 Deployment and Performance**: Questions were raised regarding the deployment of the 1-bit quant version of the DeepSeek R1 model and its performance on hardware like the MI300X.
   - Users discussed expectations and experiences with the model's performance and deployment challenges on various setups.
- **Chatbot Dataset Creation from Forums**: Discussions on creating datasets for chatbots from forum posts emphasized maintaining a question-answer format derived from threads.
   - Users shared strategies to convert conversation formats from platforms like Reddit into usable datasets for fine-tuning their models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb#scrollTo=hvJcwnb9Qy8b">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1a8wP89019HKso87oE_b_P2wFUJzTPPYm?authuser=2#scrollTo=QmUBVEnvCDJv.">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Nemo_(12B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/blog/deepseek-r1">Run Deepseek-R1 / R1 Zero</a>: DeepSeek&#x27;s latest R-1 model is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Learn how to run &amp; fine-tune the model.</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main">unsloth/DeepSeek-R1-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/datasets/gus-gustavo/reddit_roastme?row=0">gus-gustavo/reddit_roastme · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Skorcht/finaldatasethopefully">Skorcht/finaldatasethopefully · Datasets at Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/datasets/wikimedia/wikipedia">wikimedia/wikipedia · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/_utils.py#L383">unsloth/unsloth/models/_utils.py at main · unslothai/unsloth</a>: Finetune Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1332949351206752370)** (61 messages🔥🔥): 

> `Model Training Techniques, AugmenToolKit Usage, Code Vulnerability Review, Loss Graph Interpretation` 


- **Training LLMs with Interesting Styles**: A member shared their effort in training a model to write long-form prose, utilizing **AugmenToolKit** for dataset generation, aiming for varied writing styles based on the input material.
   - They expressed optimism despite discovering issues with the loss graph being too linear, potentially indicating overfitting.
- **Concerns on Loss Format**: Participants discussed the observed **linear loss** during training, with one participant advising against using an instruct model to avoid burning compute resources.
   - They suggested that improper dataset formatting and structure could lead to challenges in model generalization.
- **Fine-Tuning for Code Review**: A member announced the fine-tuning of the **Qwen-2.5-Coder-7b** model for code review, with specific mentions of vulnerabilities in code.
   - They clarified that their work featured both **16-bit and 4-bit** quantization options available for download.
- **Peer Support in AI Development**: Several members expressed camaraderie and interest in each other's projects, sharing insights and encouraging collaboration.
   - One member emphasized their experience, stating they've trained over **100-200 models**, providing advice for newcomers.
- **Dataset Format Discussion**: Discussions highlighted the importance of proper dataset formatting with a focus on how summaries and snippets were paired in training.
   - Members shared ideas on how uneven data lengths might affect training outcomes, prompting suggestions for improvements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/doss1232/Offensive-Qwen-2.5-Coder-7B">doss1232/Offensive-Qwen-2.5-Coder-7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO">CyberNative/Code_Vulnerability_Security_DPO · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1332646014120624221)** (49 messages🔥): 

> `Fine-tuning LLMs, LLaMA 4 Expectations, Reinforcement Learning Enhancements, Vector Database and Quantization, DeepSeek and Reasoning Models` 


- **User explores fine-tuning assessments**: A user proposes creating a custom scoring system for evaluating fine-tuned LLMs, emphasizing the need for manual testing with around **100 queries**.
   - They noted that conventional metrics like **ROUGE**, **BLEU**, and **F1** tests might not sufficiently validate technical accuracy.
- **High Hopes for LLaMA 4**: Members express high expectations for **LLaMA 4**, anticipating significant improvements in **test-time compute** and **speed gains** derived from its learning processes with **DeepSeek**.
   - *One member speculated on the potential need for impressive features to stay competitive*, suggesting the possibility of investing resources elsewhere.
- **Reinforcement Learning and CoT Generation**: Users discussed the intriguing prospects of reinforcement learning, with one considering submitting a **Colab notebook** to generate reasoning CoT datasets for fine-tuning using **Unsloth**.
   - The conversation highlights the potential of integrating **computer vision** techniques to enhance contextual understanding.
- **Curious Case of Vector DBs and Quantization**: A user inquired about the vector databases used for storing various quantization bits and how similarity searches operate for these data.
   - Another user sought clarification, questioning the terminology of storing quantization bits and their applications.
- **DeepSeek's Performance Compared to Sonnet 3.6**: In real-world testing, a user claimed that **Sonnet 3.6** outperformed all reasoning models for innovative coding tasks, suggesting effectiveness depends on task type.
   - This prompted discussions on the nuances of language comprehension capabilities across different models.



**Link mentioned**: <a href="https://github.com/SalesforceAIResearch/perfcodegen">GitHub - SalesforceAIResearch/perfcodegen</a>: Contribute to SalesforceAIResearch/perfcodegen development by creating an account on GitHub.

  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1332440021411827857)** (762 messages🔥🔥🔥): 

> `Cursor IDE Performance, Comparison of DeepSeek and Claude, Codebase Indexing, RAG Implementation, User Experience with Models` 


- **Cursor IDE Performance and User Experience**: Users have expressed concerns over slow request times for Cursor's models, particularly Claude, highlighting frustrations when working on projects.
   - Some users reported improved performance during off-peak hours and noted that models like R1 and DeepSeek are usable alternatives that can optimize their workflows.
- **Comparison between DeepSeek and Claude**: DeepSeek R1 is favored for planning tasks, while Claude is often used for more complex output due to its higher quality responses.
   - Users generally view both models as superior to alternatives, with some discussing how to leverage DeepSeek for less complex tasks to save on premium request costs.
- **Codebase Indexing in Cursor**: The indexing feature in Cursor was debated regarding its effectiveness, with some users asserting that it doesn't fully leverage the codebase for recommendations.
   - Despite the criticisms, it was acknowledged that indexing is crucial for improving codebase answers and can enhance the user experience when configured properly.
- **RAG Implementation Discussions**: Users shared their experiences and strategies for implementing Retrieval-Augmented Generation (RAG) methods with their codebases, focusing on vector storage and embedding techniques.
   - Community members discussed the performance of various models in relation to embedding and retrieval, asserting that proper implementation is key to reducing errors and improving outcomes.
- **Transitioning to Alternative Models**: Several users contemplated switching between models, especially utilizing DeepSeek for basic tasks while reserving Claude for more complex requirements.
   - There was also mention of alternative platforms such as GitHub Copilot and Spark Engine, with varying levels of user satisfaction and API integration capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sparkengine.ai">Spark Engine - Advanced No-Code AI Builder</a>: Build and deploy AI models with no coding required. Start creating with the most user-friendly AI platform.</li><li><a href="https://docs.cursor.com/get-started/usage#usage-based-pricing">Get Started / Usage – Cursor</a>: no description found</li><li><a href="https://docs.cursor.com/get-started/usage">Get Started / Usage – Cursor</a>: no description found</li><li><a href="https://docs.cursor.com/context/codebase-indexing">Context / Codebase Indexing – Cursor</a>: no description found</li><li><a href="https://download.todesktop.com/230313mzl4w4u92/Cursor%20Setup%200.45.3%20-%20Build%20250124b0rcj0qql-x64.exe">no title found</a>: no description found</li><li><a href="https://aistudio.google.com/apikey">no title found</a>: no description found</li><li><a href="https://forum.cursor.com/t/is-cursor-using-full-version-of-r1/44756/5">Is Cursor using FULL version of R1?</a>: The team confirmed that the model with 671B parameters is used, not the Distill R1 version.</li><li><a href="https://x.com/awnihannun/status/1883276535643455790">Tweet from Awni Hannun (@awnihannun)</a>: DeepSeek R1 (the full 680B model) runs nicely in higher quality 4-bit on 3 M2 Ultras with MLX. Asked it a coding question and it thought for ~2k tokens and generated 3500 tokens overall:</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet set SOTA on aider’s polyglot benchmark</a>: R1+Sonnet has set a new SOTA on the aider polyglot benchmark. At 14X less cost compared to o1.</li><li><a href="https://winstall.app/apps/Anysphere.Cursor">Install Cursor with winget - winstall</a>: The AI Code Editor</li><li><a href="https://forum.cursor.com/t/cursor-does-not-send-files-to-claude/43948">Cursor does not send files to Claude</a>: An example of Cursor failing to send file data to Claude, for the third time in a row.  These are chewing up my precious Fast-Replies, and it happens all the time.  I am using this version:  Version: ...</li><li><a href="https://forum.cursor.com/t/slow-pool-information/41812?u=danperks">Slow Pool Information</a>: Hello! Wanted to give some details on slow pool wait times and why they’ve recently increased a bit…  Anthropic Capacity We’re working with Anthropic to scale up Sonnet traffic. At the moment we’re in...</li><li><a href="https://blog.voyageai.com/2024/12/04/voyage-code-3/#:~:text=voyage%2Dcode%2D3%20supports%20much,Matryoshka%20embeddings.">voyage-code-3: more accurate code retrieval with lower dimensional, quantized embeddings</a>: TL;DR – Introducing voyage-code-3, our next-generation embedding model optimized for code retrieval. It outperforms OpenAI-v3-large and CodeSage-large by an average of 13.80% and 16.81% on a suite …</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://forum.cursor.com/">Cursor - Community Forum</a>: A place to discuss Cursor (bugs, feedback, ideas, etc.)</li><li><a href="https://gofund.me/58f99126">Donate to A Plea for Help: Family of 3 Loses Everything in Housefire, organized by Griffin Family</a>: Help a Family Rebuild After Tragic House Fire and He… Griffin Family needs your support for A Plea for Help: Family of 3 Loses Everything in Housefire</li><li><a href="https://docs.cursor.com/get-started/usage#fast-and-slow-requests">Get Started / Usage – Cursor</a>: no description found</li><li><a href="https://github.com/cline/cline">GitHub - cline/cline: Autonomous coding agent right in your IDE, capable of creating/editing files, executing commands, using the browser, and more with your permission every step of the way.</a>: Autonomous coding agent right in your IDE, capable of creating/editing files, executing commands, using the browser, and more with your permission every step of the way. - cline/cline</li><li><a href="https://github.com/danilofalcao/cursor-deepseek">GitHub - danilofalcao/cursor-deepseek: A high-performance HTTP/2-enabled proxy server designed specifically to enable Cursor IDE&#39;s Composer to use DeepSeek&#39;s and OpenRouter&#39;s language models. This proxy translates OpenAI-compatible API requests to DeepSeek/OpenRouter API format, allowing Cursor&#39;s Composer and other OpenAI API-compatible tools to seamlessly work with these models.</a>: A high-performance HTTP/2-enabled proxy server designed specifically to enable Cursor IDE&amp;#39;s Composer to use DeepSeek&amp;#39;s and OpenRouter&amp;#39;s language models. This proxy translates O...</li><li><a href="https://api-docs.deepseek.com">Your First API Call | DeepSeek API Docs</a>: The DeepSeek API uses an API format compatible with OpenAI. By modifying the configuration, you can use the OpenAI SDK or softwares compatible with the OpenAI API to access the DeepSeek API.</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/11453">ggml : x2 speed for WASM by optimizing SIMD by ngxson · Pull Request #11453 · ggerganov/llama.cpp</a>: MotivationThis PR provides a big jump in speed for WASM by leveraging SIMD instructions for qX_K_q8_K and qX_0_q8_0 dot product functions.Surprisingly, 99% of the code in this PR is written by De...</li><li><a href="https://fireworks.ai/blog/fireworks-quantization">How Fireworks evaluates quantization precisely and interpretably </a>: Deep dive into how Fireworks AI thinks about quantization and uses divergence metrics to ensure quality and create custom solutions for users  
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1332439967309369345)** (1 messages): 

> `Windsurf 1.2.2 Release, Cascade's Memory Improvements, Web Search Capabilities` 


- **Windsurf 1.2.2 Launches with Key Enhancements**: The team announced the release of **Windsurf 1.2.2**, which includes fixes for laggy conversations and improvements to **Cascade's memory system**.
   - The update aims to make the overall **Windsurf experience smoother** and more reliable with several key enhancements.
- **Cascade Takes Web Search to New Heights**: Cascade can now search the web automatically or via **URL input**, enabling users to ask live queries or paste links for context.
   - Users can also utilize commands like `@web` and `@docs` to perform searches, enhancing usability across popular documentation sites.



**Link mentioned**: <a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1332440809986981920)** (252 messages🔥🔥): 

> `Windsurf performance issues, Changes in free and pro plan credits, Deepseek model integration expectations, User experiences with Cascade, Extension compatibility issues` 


- **Windsurf performance issues plague users**: Users report constant errors and lagging issues with Windsurf, particularly with Cascade failing to execute commands properly.
   - Several members expressed frustration over losing credits while trying to debug and fix errors during their coding sessions.
- **Credits slashed in free plan - now just 5 prompts**: The free version of Windsurf now only offers 5 premium model user prompt credits, down from the previous 50 that could be exploited by creating new accounts.
   - This change has led to dissatisfaction among users who relied on the higher credit limit for their coding tasks.
- **No deepseek integration expected soon**: Users are eager to know when deepseek will be integrated into Codeium, expressing concerns about the platform falling behind competitors.
   - Some members voiced their intention to move to other tools if deepseek integration does not occur.
- **Mixed experiences using Cascade**: While some users have found Cascade helpful and effective, others report that it starts ignoring established rules and modifies code improperly.
   - Participants suggested keeping requests concise and using project rules to mitigate these issues.
- **Extension compatibility issues with Windsurf**: Users are experiencing problems with extensions not being compatible with the current version of Windsurf, leading to disruptions in their workflow.
   - Advice was given to try installing older versions of the extensions that may not have such strict IDE version requirements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://developer.apple.com/documentation/swift/">Swift | Apple Developer Documentation</a>: Build apps using a powerful open language.</li><li><a href="https://docs.codeium.com/windsurf/getting-started#forgot-to-import-vs-code-configurations">Windsurf - Getting Started</a>: no description found</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium Docs</a>: no description found</li><li><a href="https://tenor.com/view/the-simpson-leech-leeches-gif-11029678">The Simpson Leech GIF - The Simpson Leech Leeches - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/windsurf_ai/status/1880354013922857384?s=46">Tweet from Windsurf (@windsurf_ai)</a>: Wave 2 is here. Included in this update: 🌐Web Search🧠Autogenerated Memories💼Enterprise Ready... and many more!</li><li><a href="https://codeium.com/pricing">Pricing | Windsurf Editor and Codeium extensions</a>: Codeium is free forever for individuals. Teams can level up with our enterprise offering for enhanced personalization and flexible deployments.</li><li><a href="https://youtu.be/DKUe0ST_qi4?si=G5IOHXH6B9IG6qa8">Save Time and Flow Action Credits with Cascade Autogenerated Memories - Windsurf Editor</a>: Learn how Cascade’s Autogenerated Memories helps you pick up right where you left off—without needing extra prompts or spending additional flow action credit...</li><li><a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1332439820613455903)** (505 messages🔥🔥🔥): 

> `Windsurf Updates and Bugs, Cascade Performance, User Experiences with DeepSeek, Support and Documentation, Git Usage in Development` 


- **Windsurf Performance Issues Post-Update**: Users reported significant lag and internal error messages after the latest Windsurf update, making it less usable compared to previous versions.
   - Many expressed frustration with how the updates have affected their workflow, indicating a desire for a rollback or urgent fixes.
- **Mixed Feedback on Cascade Integration**: Some users shared their experiences with Cascade, emphasizing how providing detailed context in prompts improved outcomes, while others faced issues with internal errors.
   - Users noted a credit-cost imbalance, where corrections for errors made by Cascade resulted in higher credit consumption.
- **Interest in DeepSeek Integration**: There was ongoing discussion about the potential integration of DeepSeek R1 into Windsurf, with users interested in its cheaper operational costs compared to current models.
   - Concerns were raised regarding DeepSeek's performance with tool calls and overall reliability within agentic workflows.
- **Issues with Google Authentication**: Several users reported difficulties logging into Windsurf through Google authentication, especially pertaining to G Suite accounts.
   - Workarounds suggested included using standard Gmail accounts for quicker access until the issue is resolved.
- **Importance of Version Control with Git**: Many users emphasized the necessity of using Git for version control alongside Windsurf, highlighting how it can mitigate the impact of errors caused by the AI.
   - Best practices for using Git, including tagging milestones, were shared to help users maintain code integrity during development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://learngitbranching.js.org/">Learn Git Branching</a>: An interactive Git visualization tool to educate and challenge!</li><li><a href="https://githowto.com/">GitHowTo: guided tutorial about Git</a>: no description found</li><li><a href="https://graphite.dev/">Graphite - The end-to-end developer platform</a>: Graphite helps teams on GitHub deliver higher quality software, faster.</li><li><a href="https://developer.apple.com/documentation/">Featured | Apple Developer Documentation</a>: Browse the latest sample code, articles, tutorials, and API reference.</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium Docs</a>: no description found</li><li><a href="https://www.promptingguide.ai/">Prompt Engineering Guide</a>: A Comprehensive Overview of Prompt Engineering</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://www.austinmann.com/trek/iphone-16-pro-camera-review-kenya">iPhone 16 Pro Camera Review: Kenya &mdash; Travel Photographer - Austin Mann</a>: mambo vipi from kenya!  Last week at the Apple keynote event, the iPhone camera features that stood out the most to me were the new Camera Control button, upgraded 48-megapixel Ultra Wide sensor, impr...</li><li><a href="https://youtu.be/WYb2aMVnuYY">The Future of AI Code Editors with Kevin Hou (Codeium, Windsurf)</a>: Featuring Kevin Hou, Head of Product Engineering at Codeium, this episode covers the company&#39;s journey from GPU virtualization to creating a leading-edge AI ...</li><li><a href="https://github.com/sweetpad-dev/sweetpad#rea">GitHub - sweetpad-dev/sweetpad: Develop Swift/iOS projects using VSCode</a>: Develop Swift/iOS projects using VSCode. Contribute to sweetpad-dev/sweetpad development by creating an account on GitHub.</li><li><a href="https://codeium.com/windsurf">Windsurf Editor by Codeium</a>: Tomorrow&#x27;s editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.</li><li><a href="https://youtu.be/DKUe0ST_qi4?si=G5IOHXH6B9IG6qa8">Save Time and Flow Action Credits with Cascade Autogenerated Memories - Windsurf Editor</a>: Learn how Cascade’s Autogenerated Memories helps you pick up right where you left off—without needing extra prompts or spending additional flow action credit...</li><li><a href="https://github.com/sweetpad-dev/sweetpad#readme">GitHub - sweetpad-dev/sweetpad: Develop Swift/iOS projects using VSCode</a>: Develop Swift/iOS projects using VSCode. Contribute to sweetpad-dev/sweetpad development by creating an account on GitHub.</li><li><a href="https://www.trae.ai/privacy-policy">TraeAI - Privacy Policy</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1332440930271105187)** (755 messages🔥🔥🔥): 

> `Perplexity Pro changes, R1 model introduction, User feedback on AI models, Privacy concerns with DeepSeek, Comparison with other AI services` 


- **Perplexity Pro faces backlash over model changes**: Users expressed frustration over the removal of popular models like O1 and Grok from Perplexity Pro, leading to dissatisfaction among subscribers who just renewed their plans.
   - Many are considering switching to alternatives like DeepSeek or ChatGPT as the limitations on usage for new models like R1 are viewed as insufficient.
- **R1 model introduction raises questions**: The new R1 model was introduced with significant limitations, providing only 10 daily queries combined with O1, drawing criticism from users accustomed to higher limits.
   - Despite being cheaper and hosted in the US, R1's rollout has faced issues with performance and reliability, leading users to compare it unfavorably with previous models.
- **User feedback highlights quality concerns**: Many users reported that the quality of responses from R1 was lacking, particularly in specialized tasks like coding and research, prompting some to prefer other platforms.
   - Complaints also arose about the handling of prompts, with users noting that R1 seemed to misunderstand requests compared to the previously favored O1.
- **Privacy and data routing concerns**: Users are worried about how their data might be handled, particularly with DeepSeek being a Chinese company and the implications of data routing between the US and China.
   - Queries are reportedly processed in the US, but the lack of clarity on data handling practices has raised suspicions among some users.
- **Increased ads and user interface frustration**: Some users criticized the increase in ads on the Perplexity interface, feeling that the platform is starting to mirror traditional search engines in its ad placement.
   - Frustrations were voiced about the user experience and interface changes, with many users feeling overwhelmed by the clutter and noise added to the service.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://labs.perplexity.ai)?">no title found</a>: no description found</li><li><a href="https://app.chathub.gg/chat/cloud-doubao-1.5-pro">Doubao 1.5 Pro | ChatHub</a>: Chat with Doubao 1.5 Pro and 20+ AI models on ChatHub</li><li><a href="https://x.com/gmishra/status/1883951104607805615">Tweet from Gaurav Mishra (@gmishra)</a>: The reason it make sense to pay for @perplexity_ai  Pro is that you can test different models easily!!Testing @deepseek_ai right now and it&#39;s fairly good compared to @OpenAI I am surprised @perple...</li><li><a href="https://x.com/testingcatalog/status/1883775532086804953?s=61">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: DeepSeek R1 is coming soon to Perplexity as a new Reasoning option 👀* not available yet to the publich/t @denisyarats</li><li><a href="https://x.com/apostraphi/status/1883927593319293430?s=46">Tweet from Phi Hoang (@apostraphi)</a>: DeepSeek R1 is now available on Perplexity.Quoting Perplexity (@perplexity_ai) DeepSeek R1 is now available on Perplexity to support deep web research. There&#39;s a new Pro Search reasoning mode sele...</li><li><a href="https://x.com/dee_bosa/status/1883921252102099439?s=46">Tweet from Deirdre Bosa (@dee_bosa)</a>: DeepSeek latest with Perplexity CEO Aravind Srinivas https://x.com/i/spaces/1mnxeAgoDbvxX</li><li><a href="https://tenor.com/view/spider-man-we-one-gif-18212100">Spider Man GIF - Spider Man We - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://intercom.help/perplexity-ai/en/articles/10354288-refunds">Refunds | Perplexity Help Center</a>: Learn more about Perplexity Pro refunds.</li><li><a href="https://www.reddit.com/r/singularity/comments/1hxykyr/deepseek_v3_is_hugely_chinese_biased/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/user/maximim12/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://developer.visa.com/capabilities/vau">
      Visa Account Updater Overview
    </a>: no description found</li><li><a href="https://developer.mastercard.com/product/automatic-billing-updater-abu/">Mastercard Developers</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1332522184416952401)** (26 messages🔥): 

> `AI advancements, Financial trends, Earthquake mechanics, Action-adventure films, Startup insights` 


- **$500 billion deal set to transform AI**: A member highlighted a potential **$500 billion deal** that could significantly alter the landscape of AI, linking to related insights [here](https://www.perplexity.ai/page/stargate-project-InQ5ZvKETX6c5I6he1zc_A).
   - This opportunity is seen as a pivotal moment for innovations in the field.
- **How to build a startup worth $200 million**: Tips on how to bootstrap a startup and achieve **$200 million** during an exit were shared, emphasizing strategic growth methods available [here](https://www.perplexity.ai/page/wingify-T9bxT5tHSY2sRduhPzHIXg).
   - The approach details practical insights on navigating startup hurdles and maximizing valuation.
- **S&P 500 hits record closing**: The S&P 500 has recently achieved a record closing high, indicating robust market performance and investor confidence, as referenced in this [discussion](https://www.perplexity.ai/page/s-p-500-hits-record-closing-hi-yPKWo3jUQPOAfqvvoQ12Kg).
   - This milestone showcases the resilience of the market amidst various economic challenges.
- **Understanding earthquakes**: A member sought clarification on how **earthquakes happen**, providing an informative link to resources [here](https://www.perplexity.ai/search/how-do-earthquakes-happen-rlsZqPoKRS2jMv7t0PmrSw#0).
   - The discussion aimed at unraveling misconceptions about seismic activity and tectonic movements.
- **Best action-adventure movies**: A couple of users discussed **action-adventure movies**, pointing to a curated list of the best titles available [here](https://www.perplexity.ai/search/best-action-adventure-movies-c-lft_ADWwSLW6rj12clUyig).
   - This conversation highlighted popular picks that stirred excitement among genre enthusiasts.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1333070624041467986)** (4 messages): 

> `Sonar JSON Response Format, API for LinkedIn URLs, Response Format Issues, Sonar vs Sonar-Pro` 


- **Sonar JSON Response Format Bug**: Members expressed frustration with invalid JSON responses when using the `sonar` model, where responses are wrapped in Markdown.
   - One noted that switching to `sonar-pro` resolves the issue, yet raised concerns about the cost implications.
- **Trying to Fetch LinkedIn URLs via API**: A member is struggling to retrieve LinkedIn URLs using an API by providing only user names and workplaces, often receiving irrelevant results.
   - They seek advice on improving their prompts or other strategies to achieve better results.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1333471715258531902)** (1 messages): 

> `Nous Psyche, Cooperative AI Training, Open Source Models, Heterogeneous Compute` 


- **Nous Psyche Launches Cooperative AI Training Network**: Today we announce **Nous Psyche**, a cooperative training network for generative AI on **@Solana**, designed to create open-source models using **heterogeneous compute**.
   - This initiative aims to challenge the narrative that only closed labs can advance the frontier of **superintelligence**.
- **Exploring the Myth of Psyche**: The project draws inspiration from the myth of Psyche — a tale of a mortal seeking **retribution against divine odds**.
   - More insights on this captivating narrative can be found on our [blog](https://nousresearch.com/nous-psyche/).
- **GitHub Repository for Psyche**: You can explore our open infrastructure on [GitHub](https://github.com/PsycheFoundation/psyche), which aims to **democratize** and **decentralize** the development of **superintelligence for humanity**.
   - This initiative encourages broader participation in advancing AI technologies and aims to reshape ownership and accessibility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NousResearch/status/1883912370696704011">Tweet from Nous Research (@NousResearch)</a>: Recent AI breakthroughs challenge the status quo narrative that only closed, mega labs have the ability to push the frontier of superintelligence.Today we announce Nous Psyche built on @Solana - a coo...</li><li><a href="https://github.com/PsycheFoundation/psyche">GitHub - PsycheFoundation/psyche: An open infrastructure to democratize and decentralize the development of superintelligence for humanity.</a>: An open infrastructure to democratize and decentralize the development of superintelligence for humanity. - PsycheFoundation/psyche
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1332441913919275149)** (681 messages🔥🔥🔥): 

> `Nous Psyche Announcement, Testnet Participation, Distributed Training and Reputation Systems, Scam Tokens, Collaborative Open Source Development` 


- **Nous Psyche Announcement**: Nous announced the launch of Psyche, a cooperative training network for generative AI built on Solana, emphasizing collaboration over competition.
   - The project aims to leverage decentralized and trustless computation, with many interested in its implications for AI development.
- **Testnet Participation**: Participants expressed excitement about the upcoming testnet for Psyche, with expectations set for a user-friendly experience despite some technical requirements.
   - The testnet is anticipated to go live soon, and further details are expected during an event with the Solana Foundation on the 30th.
- **Distributed Training and Reputation Systems**: Discussions arose around the verification protocols to prevent malicious nodes from gaining dominance in the system, which include a reputation system and probabilistic checks.
   - Concerns were raised about the complexity added by reputation systems, potentially complicating the analysis of security properties.
- **Scam Tokens**: The community is warned against fraudulent tokens impersonating the Nous brand, with confirmation that no official tokens are currently associated with the Psyche project.
   - Users were encouraged to report any scams or impersonators to maintain the integrity of the community.
- **Collaborative Open Source Development**: Participants stressed the importance of collaboration within the open-source community, likening it to a sports anime where everyone aims to improve together.
   - There's a shared sentiment that contributing to the project can lead to significant advancements in generative AI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@ambiance461">Ambiance</a>: The main goal of this channel is to upload atmospheric/soulful style DnB that is missing here on youtube (I ONLY UPLOAD WHATS NOT ON YOUTUBE)-----------------------------------------------------------...</li><li><a href="https://huggingface.co/DavidAU/L3-MOE-8X8B-Dark-Planet-8D-Mirrored-Chaos-47B">DavidAU/L3-MOE-8X8B-Dark-Planet-8D-Mirrored-Chaos-47B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/ifable/gemma-2-Ifable-9B">ifable/gemma-2-Ifable-9B · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.10837">Approximating Two-Layer Feedforward Networks for Efficient Transformers</a>: How to reduce compute and memory requirements of neural networks (NNs) without sacrificing performance? Many recent works use sparse Mixtures of Experts (MoEs) to build resource-efficient large langua...</li><li><a href="https://x.com/ryunuck/status/1883032334426873858">Tweet from ryunuck (p≈np) (@ryunuck)</a>: I have completely solved the mystery of Q*: It is a new foundation module for LLMs, a text-conditioned spatial computer model.Attached to this post, you can see a model that was trained for pathfindin...</li><li><a href="https://wiki.pygmalion.chat/bot-creation/trappu/introduction">Introduction</a>: An introduction to both PLists, and Ali:Chat.</li><li><a href="https://x.com/NousResearch/status/1883912370696704011">Tweet from Nous Research (@NousResearch)</a>: Recent AI breakthroughs challenge the status quo narrative that only closed, mega labs have the ability to push the frontier of superintelligence.Today we announce Nous Psyche built on @Solana - a coo...</li><li><a href="https://publish.obsidian.md/hallerite/rl-for-deepseek-r1">the rl for deepseek-r1 - Entropic Musings - Obsidian Publish</a>: the rl for deepseek-r1 - Entropic Musings - Powered by Obsidian Publish.</li><li><a href="https://x.com/junxian_he/status/1883183099787571519">Tweet from Junxian He (@junxian_he)</a>: We replicated the DeepSeek-R1-Zero and DeepSeek-R1 training on 7B model with only 8K examples, the results are surprisingly strong. 🚀 Starting from Qwen2.5-Math-7B (base model), we perform RL on it d...</li><li><a href="https://tenor.com/view/lain-lain-iwakura-serial-experiments-lain-wires-wired-gif-1481475804337586659">Lain Lain Iwakura GIF - Lain Lain iwakura Serial experiments lain - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fxtwitter.com/RLanceMartin/status/1883209736629448725">Tweet from Lance Martin (@RLanceMartin)</a>: R1 Deep Researcher Fully local research assistant w @deepseek_ai R1 + @ollama. Give R1 a topic and watch it search web, learn, reflect, search more, repeat as long as you want. Gives you a report w/ s...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfvGy-3sh-GzcwdWmxJ-1qttRlI8MOpYEQmk_kz9aCsstPnvw/viewform?usp=header">Questionnaire on Comparative Crisis Communication Strategies</a>:   This survey explores how cultural values influence crisis communication strategies during environmental disasters in Morocco and the United States. It examines media usage, public trust, and audienc...</li><li><a href="https://x.com/cneuralnetwork/status/1883195767986569430">Tweet from neural nets. (@cneuralnetwork)</a>: releasing the DeepSeek R1 blog, which explains the whole paper in great detail, not excluding any math, but anyone with basic class 12 math knowledge can understand it (link in replies)do share and rt...</li><li><a href="https://fxtwitter.com/Teknium1/status/1882893748742598669">Tweet from Teknium (e/λ) (@Teknium1)</a>: We retrained hermes with 5k deepseek r1 distilled cots. I can confirm a few things:1. You can have a generalist + reasoning mode, we labeled all longCoT samples from r1 with a static systeem prompt, t...</li><li><a href="https://x.com/disclosetv/status/1883675709954298338?t=UJRV7ZCFU0xIEYnwuO-xIg&s=19">Tweet from Disclose.tv (@disclosetv)</a>: NEW - Chinese DeepSeek AI surpasses ChatGPT and now tops Apple&#39;s free app download rankings in the United States.</li><li><a href="https://docs.psyche.network">Intro to Psyche - Psyche</a>: no description found</li><li><a href="https://huggingface.co/nbeerbower/mistral-nemo-gutenberg-12B-v2">nbeerbower/mistral-nemo-gutenberg-12B-v2 · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/open-r1">GitHub - huggingface/open-r1: Fully open reproduction of DeepSeek-R1</a>: Fully open reproduction of DeepSeek-R1. Contribute to huggingface/open-r1 development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=ghltnvQmYKA">DeepSeek May End US Stocks Exceptionalism: 3-Minute MLIV</a>: Anna Edwards, Guy Johnson, Kriti Gupta and Mark Cudmore break down today&#39;s key themes for analysts and investors on &quot;Bloomberg: The Opening Trade.&quot;--------Mo...</li><li><a href="https://github.com/langchain-ai/ollama-deep-researcher">GitHub - langchain-ai/ollama-deep-researcher: Fully local web research and report writing assistant</a>: Fully local web research and report writing assistant - langchain-ai/ollama-deep-researcher</li><li><a href="https://www.youtube.com/watch?v=1xDVbu-WaFo">Hugging Face Journal Club - DeepSeek R1</a>: The post-training team at Hugging Face discuss the tech report behind DeepSeek&#39;s ground breaking R1 models.- Report: https://github.com/deepseek-ai/DeepSeek-...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/issues/29">A speedrun on consumer grade cards? · Issue #29 · KellerJordan/modded-nanogpt</a>: Hi thanks for the great repo! I would appreciate it if there can be a speed run on consumer cards e.g. RTX4090. Since it is 125M params, the RTX4090&#39;s 24GB memory should fit in the classical way, ...</li><li><a href="https://team.doubao.com/en/special/doubao_1_5_pro">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1332713452774490133)** (14 messages🔥): 

> `R1 Distillation Models, Llama 3 performance issues, Image Captioning with DeepSeek, Building AI Assistants, Fine-tuning for performance` 


- **Examining R1 Distillation Models**: A member inquired about the use of R1 distillation models, suggesting applying it to datasets like R1 and referencing a [recent paper](https://arxiv.org/abs/2407.06023v3) that describes the distillation process.
   - Another suggested replicating systems like [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) as potential improvements.
- **Llama 3 shows erratic behavior**: One user expressed frustration about the **Llama 3** model providing nonsensical responses after consistent performance over two days.
   - In contrast, another user reported successful task executions with **Llama 3b instruct**, indicating variable performance might be model-dependent.
- **Rapid convergence in Image Captioning**: A member reported fast convergence when fine-tuning a distilled DeepSeek R1 for an image captioning task, which is typically seen as challenging.
   - They shared their work in a [GitHub notebook](https://github.com/githubpradeep/notebooks/blob/main/VLM_DeepSeek-R1-Distill-Qwen-1.5B.ipynb), showcasing promising results.
- **Starting a local AI Assistant project**: A user sought guidance on creating a local AI assistant, questioning if models like **Llama** are necessary and for beginner resources.
   - There was a suggestion to collaborate with tools like **Ask DeepSeek/Hermes/ChatGPT** to foster learning.
- **Optimize Learning Velocity with Financial Freedom**: A member recommended leveraging LLM resources, internet tools, and community projects for efficient learning.
   - They emphasized optimizing for learning speed, especially when financial flexibility is available.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.06023v3">Distilling System 2 into System 1</a>: Large language models (LLMs) can spend extra compute during inference to generate intermediate thoughts, which helps to produce better final responses. Since Chain-of-Thought (Wei et al., 2022), many ...</li><li><a href="https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k">bespokelabs/Bespoke-Stratos-17k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/githubpradeep/notebooks/blob/main/VLM_DeepSeek-R1-Distill-Qwen-1.5B.ipynb">notebooks/VLM_DeepSeek-R1-Distill-Qwen-1.5B.ipynb at main · githubpradeep/notebooks</a>: Contribute to githubpradeep/notebooks development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1333413045912735775)** (2 messages): 

> `Human-like LLM enhancements, Crisis communication strategies` 


- **Research on Human-Like Responses in LLMs**: A paper titled *Enhancing Human-Like Responses in Large Language Models* explores advancements in making LLMs more **human-like** by enhancing natural language understanding and emotional intelligence, with techniques such as fine-tuning and psychological principles.
   - The study's findings suggest these enhancements **improve user interactions** and open new possibilities for AI applications, prompting further examination of the **ethical implications** introduced by these attributes.
- **Questionnaire on Crisis Communication Strategies**: A questionnaire titled *Comparative Crisis Communication Strategies* aims to explore how **cultural values** influence crisis communication during environmental disasters in Morocco and the United States.
   - It specifically examines media usage and audience responses to understand the role of traditional and digital communication in shaping disaster response in these cultural contexts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.05032">Enhancing Human-Like Responses in Large Language Models</a>: This paper explores the advancements in making large language models (LLMs) more human-like. We focus on techniques that enhance natural language understanding, conversational coherence, and emotional...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfvGy-3sh-GzcwdWmxJ-1qttRlI8MOpYEQmk_kz9aCsstPnvw/viewform?usp=sharing">Questionnaire on Comparative Crisis Communication Strategies</a>:   This survey explores how cultural values influence crisis communication strategies during environmental disasters in Morocco and the United States. It examines media usage, public trust, and audienc...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1332843827794280488)** (2 messages): 

> `LLM Live2D Assistant, Qwen2.5-VL model, OCR capabilities` 


- **Meet your new assistant: LLM Live2D!**: The [LLM Live2D Desktop Assistant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant) is now available for Windows and MacOS, featuring voice commands and unique interactions with your character.
   - It combines screen sensing and clipboard content retrieval to enhance user experience, providing seamless full computer control.
- **Qwen2.5-VL excels at OCR!**: The newly launched [Qwen2.5-VL model](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) stands out with impressive optical character recognition capabilities, including handwriting analysis.
   - It includes advanced features for understanding visual content while acting as a dynamic tool for computer and phone operations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">Qwen/Qwen2.5-VL-72B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant">GitHub - ylxmf2005/LLM-Live2D-Desktop-Assitant: Your Live2D desktop assistant powered by LLM! Available for both Windows and MacOS, it senses your screen, retrieves clipboard content, and responds to voice commands with a unique voice. Featuring voice wake-up, singing capabilities, and full computer control for seamless interaction with your favorite character.</a>: Your Live2D desktop assistant powered by LLM! Available for both Windows and MacOS, it senses your screen, retrieves clipboard content, and responds to voice commands with a unique voice. Featuring...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1333413045912735775)** (2 messages): 

> `Human-Like Large Language Models, Crisis Communication Strategies` 


- **Advancements in Human-Like LLMs**: The paper titled [Enhancing Human-Like Responses in Large Language Models](https://arxiv.org/abs/2501.05032) explores techniques for improving **natural language understanding** and **emotional intelligence** in AI systems.
   - It evaluates methods such as fine-tuning with diverse datasets and notes that these enhancements can lead to better user interactions while raising ethical concerns regarding biases.
- **Master’s Program Questionnaire on Crisis Communication**: A member shared a [questionnaire](https://docs.google.com/forms/d/e/1FAIpQLSfvGy-3sh-GzcwdWmxJ-1qttRlI8MOpYEQmk_kz9aCsstPnvw/viewform?usp=sharing) focusing on the influence of **cultural values** in crisis communication strategies during environmental disasters in Morocco and the US.
   - The survey examines aspects like **media usage**, **public trust**, and audience responses, highlighting how communication methods can shape disaster response.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/forms/d/e/1FAIpQLSfvGy-3sh-GzcwdWmxJ-1qttRlI8MOpYEQmk_kz9aCsstPnvw/viewform?usp=sharing">Questionnaire on Comparative Crisis Communication Strategies</a>:   This survey explores how cultural values influence crisis communication strategies during environmental disasters in Morocco and the United States. It examines media usage, public trust, and audienc...</li><li><a href="https://arxiv.org/abs/2501.05032">Enhancing Human-Like Responses in Large Language Models</a>: This paper explores the advancements in making large language models (LLMs) more human-like. We focus on techniques that enhance natural language understanding, conversational coherence, and emotional...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 messages): 

voltamachine: neat
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1332446242294333632)** (1 messages): 

> `ChatGPT Canvas Update, OpenAI o1 Integration, HTML & React Rendering, Desktop App Features` 


- **ChatGPT Canvas now supports OpenAI o1**: Today, OpenAI announced that **Canvas** now works with **OpenAI o1**; users can select o1 from the model picker or use the `/canvas` command.
   - This update is available to **Pro**, **Plus**, and **Team** users, expanding the versatility of the Canvas tool.
- **Canvas can render HTML & React code**: With the latest update, Canvas can now render **HTML** and **React** code directly within ChatGPT.
   - This feature is accessible to **Pro**, **Plus**, **Team**, and even **Free** users, enhancing the platform's capabilities.
- **Canvas fully rolled out on macOS desktop app**: The updates for Canvas have been fully rolled out on the **ChatGPT desktop app** for **macOS**, available to all tiers.
   - This means that all users can now utilize Canvas seamlessly on their desktop.
- **Enterprise and Edu updates coming soon**: Both updates for Canvas will be rolling out to **Enterprise** and **Edu** users in a couple of weeks.
   - This ensures wider access to the latest features as they gradually implement improvements.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1332449891028308130)** (537 messages🔥🔥🔥): 

> `DeepSeek vs OpenAI models, Impact of DeepSeek on stock market, AI competition in tech industry, Performance comparisons of LLMs, User experiences with AI models` 


- **DeepSeek vs OpenAI models**: Many users are comparing DeepSeek R1 to OpenAI's models, particularly O1 and GPT-4o, with several noting that DeepSeek often requires fewer corrections in generated code.
   - Some participants expressed a preference for DeepSeek, citing its comparable performance and lower cost, leading to discussions on which model is superior.
- **Impact of DeepSeek on stock market**: The announcement of DeepSeek's capabilities has reportedly led to significant drops in US tech stocks, including Nvidia, which lost nearly $600 billion in market value.
   - Industry watchers are considering how the emergence of competitive AI models like DeepSeek can disrupt established tech companies and the broader market.
- **AI competition in tech industry**: With DeepSeek's open-source model providing strong competition, there are concerns about how established players like OpenAI will respond to this shift in the AI landscape.
   - Participants in the discussion highlight the increasing importance of innovation and competition in the AI space, especially as more affordable models emerge.
- **Performance comparisons of LLMs**: Users are sharing benchmark results and personal experiences with different AI models, indicating that performance can vary widely based on specific tasks.
   - DeepSeek and Gemini are frequently mentioned as models that can outperform traditional offerings like O1 and GPT-4o in certain applications.
- **User experiences with AI models**: Contributors are discussing their firsthand experiences using DeepSeek and other AI models for coding tasks, noting variations in success rates.
   - While some users report frustrations with traditional AI models, others find success and satisfaction with the performance and capabilities of newer alternatives.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.nextplatform.com/2024/10/25/cerebras-trains-llama-models-to-leap-over-gpus/">Cerebras Trains Llama Models To Leap Over GPUs</a>: It was only a few months ago when waferscale compute pioneer Cerebras Systems was bragging that a handful of its WSE-3 engines lashed together could run</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet set SOTA on aider’s polyglot benchmark</a>: R1+Sonnet has set a new SOTA on the aider polyglot benchmark. At 14X less cost compared to o1.</li><li><a href="https://www.cnbc.com/2025/01/27/chinese-ai-applications-are-looking-to-move-beyond-chatbots.html">Chinese AI applications now have bigger aims — they&#x27;re looking beyond chatbots</a>: A slew of releases in the last week demonstrate how Chinese companies have moved quickly with AI models that compete with OpenAI&#x27;s ChatGPT. </li><li><a href="https://www.cnbc.com/2025/01/24/how-chinas-new-ai-model-deepseek-is-threatening-us-dominance.html">How China’s new AI model DeepSeek is threatening U.S. dominance</a>: A lab out of China has ignited panic in Silicon Valley after releasing impressive AI models more cheaply and with less-powerful chips than U.S. giants.</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M">Qwen/Qwen2.5-14B-Instruct-1M · Hugging Face</a>: no description found</li><li><a href="https://x.com/paul_cal/status/1882440978872865020">Tweet from Paul Calcraft (@paul_cal)</a>: Anthropic CEO @DarioAmodei joins the emergent RL chorus&#34;It&#39;s not like reasoning or test time compute [..] is a totally new method, it&#39;s more like an emergent property [..] of training the ...</li><li><a href="https://x.com/thinking_panda/status/1883849302939971783">Tweet from ShanghaiPanda (@thinking_panda)</a>: China&#39;s #DeepSeek has now erased $2 trillion worth of market cap in US stocks.😜It used to take decades for China to break the US technological monopoly (manufacturing).Then, it was years (Interne...</li><li><a href="https://www.cnn.com/2025/01/27/tech/deepseek-stocks-ai-china/index.html">A shocking Chinese AI advancement called DeepSeek is sending US stocks plunging | CNN Business</a>: no description found</li><li><a href="https://x.com/CodeByPoonam/status/1883175938613207134?t=-uNTSIJlDYOx3QEMAE4A-w&s=19">Tweet from Poonam Soni (@CodeByPoonam)</a>: Goodbye ChatGPTIt’s only been 5 days since Deepseek R1 dropped, and the World is already blown away by its potential.13 examples that will blow your mind (Don&#39;t miss the 5th one):</li><li><a href="https://youtu.be/7GV_OdqzmIU?si=IKCRD0tUOkHtOplS">Cerebras Co-Founder Deconstructs Blackwell GPU Delay</a>: Cerebras Chief System Architect and Co-Founder, J.P. Fricker explains the technical challenges with Nvidia&#39;s Blackwell.00:12 Introduction to Interposers02:54...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1332908414527078470)** (34 messages🔥): 

> `O3 Mini Release Date, O3 Mini Features, Tokenization in Tiktoken, Gemini vs GPT, Scraping URLs from Word Files` 


- **O3 Mini Release Date Speculation**: The anticipated launch of **O3 Mini** is rumored to be early this week, although there may be delays based on unforeseen events.
   - Members are eager for updates, with some expressing that promises around message limits seem 'insane.'
- **Debate on O3 Mini Multimodal Capabilities**: Concerns were raised that **O3 Mini** might just be an enhanced version of **O1 Mini**, lacking multimodal features that users desire.
   - Users expressed disappointment about limitations, hoping for enhanced functionality to solve complex problems more effectively.
- **Challenges with Tiktoken's Tokenization**: A user questioned why **Tiktoken** occasionally processes tokens as single characters instead of merging them, pointing out inconsistencies with specific inputs.
   - Another user suggested that special token limits might influence this behavior, referencing potential limits documented in research papers.
- **Comparing Gemini 2.0 and GPT's Performance**: Discussions highlighted that while **Gemini 2.0** is competent, it lacks the advanced features and sophisticated integration seen in **GPT**.
   - Users noted **Gemini 2.0**'s inability to format complex mathematical problems using LaTeX, which is a significant advantage for GPT in educational contexts.
- **Scraping URLs from Word files**: A user shared difficulties with **scraping URLs** embedded in anchor text within Word files using GPTs, stating that it often fails to retrieve complete paths.
   - Despite expecting shortcomings, they noted that while working with XML tags typically yields better results, the current method was unreliable.



**Link mentioned**: <a href="https://x.com/sama/status/1883294216329281627">Tweet from Sam Altman (@sama)</a>: ok we heard y’all.*plus tier will get 100 o3-mini queries per DAY (!)*we will bring operator to plus tier as soon as we can*our next agent will launch with availability in the plus tierenjoy 😊Quoting...

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1332520880063909908)** (12 messages🔥): 

> `LangChain ChatPromptTemplate, User Feedback on Formatting, Complex Prompts & Vector Stores` 


- **Exploring LangChain's ChatPromptTemplate**: A member inquired whether anyone has tried LangChain's **ChatPromptTemplate** for complex prompts that pull from an external **vector store** for context.
   - They noted that official documentation lacks examples for passing documents from a **vector store retriever** to the prompt template.
- **Community Uncertainty on Implementation**: Another member responded that they have not tried this yet but suggested using the **regular prompt structure** as a workaround.
   - They expressed interest in the results, indicating they are keen to learn more if the implementation is successful.
- **User Feedback on Formatting & Clarity**: A user expressed frustration regarding clarity and formatting in the chat, requesting to have their **10 minutes back**.
   - This comment seems to highlight the need for more efficient discussions or clearer instructions in future interactions.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1332520880063909908)** (12 messages🔥): 

> `Langchain ChatPromptTemplate, Vector Store Integration` 


- **Question on Langchain ChatPromptTemplate Usage**: A user inquired if anyone has utilized Langchain's **ChatPromptTemplate** for complex prompts that refer to an **external vector store** for context alongside user inputs.
   - They noted a lack of documentation examples for passing documents from a vector store retriever to the prompt template, seeking community insights.
- **Interest in Testing ChatPromptTemplate**: Another user responded that they haven't tried this yet but suggested using the **regular prompt structure** for the integration.
   - They expressed interest in the outcome and encouraged the original poster to share their findings if it doesn't work.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1332452988131868702)** (449 messages🔥🔥🔥): 

> `DeepSeek API Issues, Inference Provider Profitability, Comparison of R1 and O1 Models, New AI Model Releases, User Experiences with Aider` 


- **DeepSeek's API is experiencing outages**: Users reported that the DeepSeek API is intermittently down, causing issues with the R1 model's responsiveness and output.
   - DeepSeek acknowledged receiving large-scale malicious attacks, which may be contributing to the outages and high demand for their services.
- **Profitability of Inference Providers**: Discussion around the profitability of becoming an inference provider highlighted that high utilization is key for profitability given fixed costs.
   - Inferences with low utilization may yield negligible profits, especially with pricing strategies among competing services being assessed.
- **R1 vs O1 Performance**: Some users claim that DeepSeek's R1 model combined with Sonnet outperforms O1 with Sonnet based on certain benchmarks.
   - However, others expressed skepticism, noting that O1 Pro might still be superior for specific coding tasks.
- **New AI Model Releases**: The introduction of new models like Qwen2.5-1M and Janus-Pro positioned them as viable competitors to existing systems, particularly due to their 1 million token context lengths.
   - With advancements in inference frameworks and capabilities, these new models are largely seen as enhancements to existing offerings in the AI landscape.
- **User Experiences with Aider**: Users are attempting different configurations in Aider when working with DeepSeek, including lowering max token limits to counter API issues.
   - The ongoing outages have prompted users to explore alternative models and platforms while sharing advice on configurations for improved performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Alibaba_Qwen/status/1883557964759654608">Tweet from Qwen (@Alibaba_Qwen)</a>: We&#39;re leveling up the game with our latest open-source models, Qwen2.5-1M ! 💥 Now supporting a 1 MILLION TOKEN CONTEXT LENGTH 🔥Here&#39;s what’s new:1️⃣ Open Models: Meet Qwen2.5-7B-Instruct-1M ...</li><li><a href="https://x.com/UnslothAI/status/1883899061893546254">Tweet from Unsloth AI (@UnslothAI)</a>: Introducing 1.58bit DeepSeek-R1 GGUFs! 🐋DeepSeek-R1 can now run in 1.58-bit, while being fully functional. We shrank the 671B parameter model from 720GB to just 131GB - a 80% size reduction.Naively q...</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet set SOTA on aider’s polyglot benchmark</a>: R1+Sonnet has set a new SOTA on the aider polyglot benchmark. At 14X less cost compared to o1.</li><li><a href="https://openrouter.ai/deepseek/deeps">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://deepinfra.com/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 - Demo - DeepInfra</a>: DeepSeek-R1-Zero is a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrated remarkable performance on reasoning.. Try out A...</li><li><a href="https://plugins.jetbrains.com/plugin/25249-coding-aider">Coding Aider - IntelliJ IDEs Plugin | Marketplace</a>: Seamlessly integrate Aider&#39;s AI-powered coding assistance directly into your IDE. This integration boosts your productivity by offering rapid access for precision code...</li><li><a href="https://medium.com/@nimritakoul01/evaluating-the-ai-scientist-63e419e575b8">Evaluating The AI Scientist</a>: In this article, I am presenting a summary of the AI driven end-to-end agentic pipeline developed by sakana.ai…</li><li><a href="https://aider.chat/docs/usage/watch.html">Aider in your IDE</a>: Aider can watch your files and respond to AI comments you add in your favorite IDE or text editor.</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:nitro">DeepSeek R1 (nitro) - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://x.com/_akhaliq/status/1883914398127083665">Tweet from AK (@_akhaliq)</a>: deepseek just dropped some new models people are still getting used to R1Janus-Pro is a novel autoregressive framework that unifies multimodal understanding and generation. It addresses the limitation...</li><li><a href="https://x.com/kimi_moonshot/status/1883532744225161369?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from Kimi.ai (@Kimi_Moonshot)</a>: Kimi k1.5: The Multimodal Reasoning Model - Available now on http://Kimi.ai 🦄💡 What can Kimi k1.5 do?🔹 Image to Code: Convert images into structured code and insights🔹 GeoGuessr: Identify and pinp...</li><li><a href="https://status.deepseek.com/">DeepSeek Service Status</a>: no description found</li><li><a href="https://aider.chat/docs/install/docker.html">Aider with docker</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/images-urls.html">Images &amp; web pages</a>: Add images and web pages to the aider coding chat.</li><li><a href="https://docs.docker.com/build/building/multi-stage/">Multi-stage</a>: Learn about multi-stage builds and how you can use them to improve your builds and get smaller images </li><li><a href="https://aider.chat/docs/usage/commands.html?">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API, Providers, Stats</a>: DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](/meta-llama/llama-3. Run DeepSeek R1 Distill Llama 70B with API</li><li><a href="https://x.com/Alibaba_Qwen/status/1883954247743725963">Tweet from Qwen (@Alibaba_Qwen)</a>: 🎉 恭喜发财🧧🐍 As we welcome the Chinese New Year, we&#39;re thrilled to announce the launch of Qwen2.5-VL , our latest flagship vision-language model! 🚀💗 Qwen Chat: https://chat.qwenlm.ai📖 Blog: http...</li><li><a href="https://github.com/deepseek-ai/awesome-deepseek-integration/blob/main/README.md">awesome-deepseek-integration/README.md at main · deepseek-ai/awesome-deepseek-integration</a>: Contribute to deepseek-ai/awesome-deepseek-integration development by creating an account on GitHub.</li><li><a href="https://status.deepseek.com/incidents/vx6w5ypzpgj7">【已恢复】DeepSeek 网页/API不可用（[Resolved]DeepSeek Web/API Service Not Available）</a>: no description found</li><li><a href="https://github.com/restatedev/sdk-python/">GitHub - restatedev/sdk-python: Restate SDK for Python</a>: Restate SDK for Python. Contribute to restatedev/sdk-python development by creating an account on GitHub.</li><li><a href="https://github.com/PierrunoYT/awesome-ai-dev-tools">GitHub - PierrunoYT/awesome-ai-dev-tools: A curated list of powerful and innovative AI-powered development tools, including code editors, plugins, and productivity enhancers.</a>: A curated list of powerful and innovative AI-powered development tools, including code editors, plugins, and productivity enhancers. - PierrunoYT/awesome-ai-dev-tools</li><li><a href="https://www.vxreddit.com/r/ChatGPT/comments/1i9bhuc/chatgpt_pro_me_and_my_wallet/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1332455338729013329)** (120 messages🔥🔥): 

> `Deepseek API Issues, Aider Functionality with Architect Mode, Model Pairing and Switching, Token Usage in Aider, Using Aider with Rust` 


- **Troubles with Deepseek API**: Users reported issues with the **Deepseek API** being down or slow, impacting responses in **Aider**, even when the status page showed it was operational.
   - Several members tried different setups and highlighted the importance of checking API performance and key configurations.
- **Issues with Architect Mode in Aider**: A member noted that in **architect mode**, responses from the **editor model** are not visible, only the response from the architect model is shown.
   - Discussion continued on whether this could be a bug or a compatibility issue with the browser feature.
- **Difficulties in Switching Models in Aider**: Users expressed frustration with **temporarily switching models** in Aider, noting that changing the main model also changed the editor model, creating workflow disruptions.
   - Participants shared workaround strategies, including using specific commands to switch models for single prompts.
- **Token Usage Monitoring in Aider**: Questions arose about how to track **token usage** for both the architect and editor models separately while using Aider.
   - Clarification was sought on whether commands like `/tokens --model sonnet` would work as intended.
- **Integrating New Crates in Rust with Aider**: A new user inquired about incorporating **new Rust crates** into Aider for better model contextual understanding and usage.
   - The capability to add external libraries to the model's context was explored, highlighting Aider's integration with different programming languages.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://xx.x.x.xx:1234```">no title found</a>: no description found</li><li><a href="https://aider.chat/2024/09/26/architect.html">Separating code reasoning and editing</a>: An Architect model describes how to solve the coding problem, and an Editor model translates that into file edits. This Architect/Editor approach produces SOTA benchmark results.</li><li><a href="https://app.hyperbolic.xyz/models/deepseek-r1/api">Hyperbolic AI Dashboard</a>: no description found</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.</li><li><a href="https://github.com/Aider-AI/aider/issues/2929">`/editor` for VSCode on Windows does not work · Issue #2929 · Aider-AI/aider</a>: Issue The problem On Windows when editor is set to use VSCode, the /editor command fails as below. $ aider --editor &quot;code --wait&quot; ────────────────────────────────────────────────────────────...</li><li><a href="https://github.com/Aider-AI/aider/issues/3020">specify the interval of lines where to read and/or modify · Issue #3020 · Aider-AI/aider</a>: Issue Can I select where to read/modify in a file, in example selecting the interval of lines that as to be put as context ? this is helpful especially if I have to edit really big files Version an...</li><li><a href="https://www.aibase.com/news/14931">ByteDance Releases Doubao Large Model 1.5 Pro, Performance Surpassing GPT-4o and Claude3.5Sonnet</a>: no description found</li><li><a href="https://www.aibase.com/tool/35837">Doubao-1.5-pro-Doubao-1.5-pro is a high-performance sparse Mixture of Experts (MoE) large language model that focuses on achieving an optimal balance between inference performance and model capability.</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1332686498834944000)** (3 messages): 

> `CodeGate integration with Aider, Comparative AI tools for web apps, Aider's functionality` 


- **CodeGate Now Integrates with Aider**: [CodeGate](https://docs.codegate.ai/how-to/use-with-aider) now supports integration with Aider, enabling users to pair program with LLMs directly in their terminal.
   - This integration allows access to models from both **OpenAI** and **Ollama**, requiring users to configure their API keys accordingly.
- **Request for Tool Updates in AI Comparison**: A user has created a [comparative table](https://github.com/renatocaliari/comparative-ai-tools-for-building-web-apps) of AI tools for building web apps, including Aider, and seeks contributions to keep it updated.
   - They encourage others to submit issues or pull requests on GitHub if they know of any relevant tools that need to be added or updated.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.codegate.ai/how-to/use-with-aider">Use CodeGate with Aider | CodeGate</a>: Configure the Aider for CodeGate</li><li><a href="https://github.com/renatocaliari/comparative-ai-tools-for-building-web-apps">GitHub - renatocaliari/comparative-ai-tools-for-building-web-apps</a>: Contribute to renatocaliari/comparative-ai-tools-for-building-web-apps development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1332442697503477771)** (413 messages🔥🔥🔥): 

> `LM Studio Model Comparisons, DeepSeek R1 Distill Models, Using AI for Coding, Benchmarking AI Models, Chatter UI Setup Issues` 


- **Comparing DeepSeek R1 Distill Models**: Users are discussing the performance of DeepSeek R1 Distill Qwen 14b compared to the 7b model, with aspects such as quantization impacting output quality.
   - Higher parameter models are perceived as having more knowledge, though the effectiveness of distinct quants like Q3 and Q4 can vary based on the model's capabilities.
- **Using AI Tools for Coding**: Individuals express an interest in integrating AI into their coding workflows for efficiency, with some considering models like R1 Distill for simple coding tasks.
   - Concerns about model performance and the trade-offs between parameter size and quantization lead to discussions about the best configurations for local use.
- **Creating Benchmarks for AI Models**: Users discuss how to create benchmarks for various AI models, emphasizing the importance of baseline datasets and customized modifications.
   - Utilizing resources like LiveCodeBench is suggested for testing and comparing model outputs effectively.
- **Chatter UI with LM Studio Setup**: Users report issues when connecting Chatter UI with LM Studio, particularly regarding port configurations and running models.
   - Steps to troubleshoot include ensuring the correct URL format and checking for necessary settings to facilitate model interaction.
- **Performance of MoE Models**: Discussions explore the capacity of Mixture of Experts (MoE) models to activate specific parts of a model, raising questions about efficiency and knowledge retention.
   - It's highlighted that while MoE allows for advanced processing efficiency, understanding individual expert capabilities is crucial for effective model deployment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llm.extractum.io/list/?query=deepseek%20r1">"deepseek r1" Search Results</a>: The top-ranked matches for the 'deepseek r1' query among 3b, 13b, 30b, and 70b small and large open-source language models found in our LLM Explorer directory.</li><li><a href="https://llm.extractum.io/list/?query=deepseek">"deepseek" Search Results</a>: The top-ranked matches for the 'deepseek' query among 3b, 13b, 30b, and 70b small and large open-source language models found in our LLM Explorer directory.</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Qwen2.5-VL - a Qwen Collection</a>: no description found</li><li><a href="https://tenor.com/view/correct-futurama-the-best-kind-of-correct-yes-yep-gif-5787390">Correct Futurama GIF - Correct Futurama The Best Kind Of Correct - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/NaniDAO/deepseek-r1-qwen-2.5-32B-ablated">NaniDAO/deepseek-r1-qwen-2.5-32B-ablated · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/docs/api/openai-api">OpenAI Compatibility API | LM Studio Docs</a>: Send requests to Chat Completions (text and images), Completions, and Embeddings endpoints</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">Import Models | LM Studio Docs</a>: Use model files you&#x27;ve downloaded outside of LM Studio</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>: no description found</li><li><a href="https://huggingface.co/livecodebench">livecodebench (Live Code Bench)</a>: no description found</li><li><a href="https://finance.yahoo.com/news/ai-exposed-power-stocks-get-crushed-as-fears-about-deepseek-trigger-stock-market-sell-off-164007338.html">AI-exposed power stocks crushed as fears about DeepSeek trigger tech sell-off</a>: AI-exposed power stocks tumbled alongside a tech sell-off as Chinese startup DeepSeek sparked investor concerns over AI chip spending by US companies.</li><li><a href="https://github.com/Vali-98/ChatterUI">GitHub - Vali-98/ChatterUI: Simple frontend for LLMs built in react-native.</a>: Simple frontend for LLMs built in react-native. Contribute to Vali-98/ChatterUI development by creating an account on GitHub.</li><li><a href="https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe>">Mixture of Experts Explained</a>: no description found</li><li><a href="https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>: Run Llama, Mistral, Phi-3 locally on your computer.</li><li><a href="https://lmstudio.ai/docs/basics">Get started with LM Studio | LM Studio Docs</a>: Download and run Large Language Models (LLMs) like Llama 3.1, Phi-3, and Gemma 2 locally in LM Studio</li><li><a href="https://lmstudio.ai/docs/advanced/tool-use">Tool Use | LM Studio Docs</a>: Enable LLMs to interact with external functions and APIs.</li><li><a href="https://github.com/ollama/ollama/issues/4643">Llama.cpp now supports distributed inference across multiple machines. · Issue #4643 · ollama/ollama</a>: Llama.cpp now supports distribution across multiple devices to boost speeds, this would be a great addition to Ollama https://github.com/ggerganov/llama.cpp/tree/master/examples/rpc https://www.red...</li><li><a href="https://www.coursera.org/specializations/machine-learning-introduction">Machine Learning</a>: Offered by Stanford University and DeepLearning.AI. #BreakIntoAI with Machine Learning Specialization. Master fundamental AI concepts and ... Enroll for free.</li><li><a href="https://lmstudio.ai/docs/api/ttl-and-auto-evict">Idle TTL and Auto-Evict | LM Studio Docs</a>: Optionally auto-unload idle models after a certain amount of time (TTL)</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h5eyb8/lm_studio_running_on_npu_finally_qualcomm/?rdt=61321">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1332450586251231272)** (156 messages🔥🔥): 

> `Hardware for Running DeepSeek Models, Using Multiple GPUs with LM Studio, Performance of LLMs on Apple M3 Max, Ideal GPUs for Coding Tasks, DDR5 Memory and AI Workloads` 


- **Guidance on Hardware for Running Local LLMs**: Users discussed hardware configurations for running local LLMs, with recommendations focusing on high VRAM GPUs like the RTX 3090 or A6000 for concurrent completion requests.
   - Concerns were raised about concurrent prompts, suggesting that using a load balancer with llama.cpp can help manage multiple requests efficiently.
- **Performance Expectations on Apple M3 Max**: A member inquired about the performance of the DeepSeek-R1 model on an M3 Max with 48GB of RAM, and another user estimated around 16-17 tokens per second.
   - This reflects considerations of RAM limitations and the efficiency of the model being used on Apple hardware.
- **Choosing the Right Model for RTX 3080**: Recommendations were made for using coding models suitable for the RTX 3080, with the Qwen2.5 Coder 7b being mentioned as a viable option.
   - Users acknowledged the RTX 3080's limitations with larger models, prompting testing with smaller configurations for better performance.
- **Dual GPU Systems for LLMs**: The feasibility of running multiple GPUs in a single server setup was examined, with some users suggesting that each GPU would need to load a model independently to serve multiple requests simultaneously.
   - Discussions highlighted the necessity of using a load balancer and specialized software configurations, such as paddler, to facilitate effective multi-GPU setups.
- **DDR5 Memory and AI Processing Power**: Users expressed interest in the evolution of DDR memory, with discussions on how future DDR6 technologies could improve bandwidth for AI workloads.
   - The importance of selecting the right server setup that supports high memory bandwidth for optimal LLM performance was underscored within the conversation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.phoronix.com/review/supermicro-h13ssln-epyc-turin">Tweet from Supermicro H13SSL-N For AMD EPYC 9005 &quot;Turin&quot; 1P Servers Review - Phoronix</a>: no description found</li><li><a href="https://i.imgur.com/A2otU">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://www.phoronix.com/review/8-12-channel-epyc-9005">Tweet from 8 vs. 12 Channel DDR5-6000 Memory Performance With AMD 5th Gen EPYC (Turin) Review - Phoronix</a>: no description found</li><li><a href="https://www.pugetsystems.com/landing/Harrison-Kinsley---Intel-Xeon-W-3300-Workstation-156/">Partnership with Harrison Kinsley</a>: Harrison Kinsley uses Flask web development on all of his business sites, Scikit Learn and TensorFlow for machine learning and data analysis with Ensmo.com, and Natural Language Toolkit for natural la...</li><li><a href="https://tenor.com/view/kevin-hart-kevin-hart-damn-gif-22709278">Kevin Hart Kevin GIF - Kevin Hart Kevin Hart - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://x.com/Sentdex/status/1883596161778696247">Tweet from Harrison Kinsley (@Sentdex)</a>: I can now confirm that yes, you have AGI at home w/ Deepseek R1.Running locally on CPU and RAM with llama.cpp after much trial, getting apx 3.4 tokens/sec.</li><li><a href="https://www.reddit.com/r/radeon/s/LpQkNtcoNr">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/distantmagic/paddler">GitHub - distantmagic/paddler: Stateful load balancer custom-tailored for llama.cpp 🏓🦙</a>: Stateful load balancer custom-tailored for llama.cpp 🏓🦙 - distantmagic/paddler</li><li><a href="https://www.reddit.com/r/pcmasterrace/s/E1Gaw7Cspw">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference? - XiongjieDai/GPU-Benchmarks-on-LLM-Inference</li><li><a href="https://www.aaronn.de/en/products/pocketai-accelerator/">Pocket AI - portable, plug and play AI accelerator | Aaronn</a>: Pocket AI - a portable AI accelerator w. NVIDIA RTX GPU offers max. flexibility and reliability for AI developers and industrial applications.</li><li><a href="https://lmstudio.ai/docs/system-requirements">System Requirements | LM Studio Docs</a>: Supported CPU, GPU types for LM Studio on Mac (M1/M2/M3/M4), Windows (x64/ARM), and Linux (x64)</li><li><a href="https://lmstudio.ai/docs">LM Studio Docs | LM Studio Docs</a>: Learn how to run Llama, DeepSeek, Phi, and other LLMs locally with LM Studio.</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1ia4mx6/project_digits_memory_speed/">Project Digits Memory Speed</a>: So I recently saw an accidentally leaked slide from Nvidia on Project Digits memory speed. It is 273 GB/s. Also 128 GB is the base memory. Only...</li><li><a href="https://www.cybenetics.com/evaluations/psus/2570/#offcanvasExample">Cybenetics Test - SAMA GT650W</a>: no description found</li><li><a href="https://www.amazon.com/dp/B0DM2LC8HX?th=1">Amazon.com: SAMA Power Supply 850W, GT 850W Fully Modular PSU 80 Plus Gold Efficiency ATX 3.1 &amp; PCIE 5.1 Compliant Support RTX 30 40 Series : Electronics</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1332448715771875419)** (4 messages): 

> `Liquid AI joins OpenRouter, Nitro DeepSeek R1, Amazon Nova models issue` 


- **Liquid AI Unveils Multilingual Models**: We're thrilled to announce that [Liquid](https://liquid.ai) has joined OpenRouter as our newest provider, bringing powerful proprietary models like [LFM 40B](https://openrouter.ai/liquid/lfm-40b), [LFM 3B](https://openrouter.ai/liquid/lfm-3b), and [LFM 7B](https://openrouter.ai/liquid/lfm-7b) to the platform.
   - LFM-7B stands out as the **best-in-class multilingual model** optimized for performance across major languages, boasting an exceptional performance-to-size ratio.
- **Nitro DeepSeek R1 Launch!**: The new **Nitro variant** for DeepSeek R1 is now available, which promises faster and more reliable performance as mentioned in the [announcement](https://openrouter.ai/deepseek/deepseek-r1:nitro).
   - Upcoming features include dynamic Nitro variants that will allow sorting of providers by speed, with future updates showing **medians instead of averages** for throughput.
- **Downed Amazon Nova Models**: Currently, **Amazon Nova models are down** due to an upstream issue with Amazon Bedrock, which misinterpreted a surge in usage as a key leak and is returning a misleading **status code 400**.
   - We're actively working on a fix and will provide updates as new information becomes available.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/deepseek/deepseek-r1:nitro">DeepSeek R1 (nitro) - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://liquid.ai)">no title found</a>: no description found</li><li><a href="https://openrouter.ai/liquid/lfm-40b)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/liquid/lfm-3b)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/liquid/lfm-7b)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://www.liquid.ai/lfm-7b">Introducing LFM-7B: Setting New Standards for Efficient Language Models</a>: The world’s best-in-class English, Arabic, and Japanese model, native in French, German, and Spanish, optimized to be the substrate for private enterprise chat, code, fast instruction following, and a...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1332442085231562862)** (535 messages🔥🔥🔥): 

> `DeepSeek Model Performance, OpenRouter API Issues, Model Suggestions and Submissions, BYOK Integration, Current State of DeepSeek Provider` 


- **DeepSeek Model Performance Uncertainties**: Users reported fluctuating performance with DeepSeek models, particularly R1, experiencing slow response times and errors like '503 model is overloaded'. DeepSeek's Nitro variant is a faster shortcut to Fireworks, but it isn't performing better than expected.
   - Updates indicated ongoing system issues, likely due to heavy user demand causing downtimes.
- **OpenRouter API Down Times**: Multiple users experienced significant latency and errors when using DeepSeek through OpenRouter, prompting discussions on whether to migrate to direct API usage. The recommendation to bring your own keys (BYOK) from DeepSeek was made to mitigate rate limits.
   - Users expressed frustrations about the speed and reliability of the API, with comparisons to the chatroom's performance.
- **Model Suggestions and Submission Processes**: A user inquired about the process for getting a model approved for OpenRouter use after their suggestion was deleted. Guidance was provided on the requirements for models to have inference providers willing to onboard onto OpenRouter.
   - The user faced a rate limit issue when trying to resubmit their model suggestion.
- **Integration of BYOK with OpenRouter**: Bringing your own provider API keys allows users to have direct control over rate limits and costs via their provider account, with 5% fees deducted from OpenRouter credits. Discussion highlighted the potential impacts of using BYOK on cost management.
   - Users were advised to plug their keys into OpenRouter for better control over their API usage.
- **Current State of DeepSeek Provider**: DeepSeek faced recent issues due to malicious attacks, resulting in service limitations for new registrations. Users noted the limitation of deepinfra as a provider for R1, likely due to reliability issues experienced by OpenRouter.
   - Discussion emphasized the high demand on DeepSeek services, leading to challenges in maintaining stable performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://flowith.io">flowith 2.0 - Your AI Creation Workspace, with Knowledge</a>: no description found</li><li><a href="https://openrouter.ai/docs/crypto-api.">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs">Quick Start | OpenRouter</a>: Start building with OpenRouter</li><li><a href="https://operator.chatgpt.com/geo-blocked">Operator</a>: An agent that can use its own browser to perform tasks for you.</li><li><a href="https://openrouter.ai/api/v1`">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs/structured-outputs">Structured Outputs | OpenRouter</a>: Enforce structured outputs for models</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/docs/integrations#automatic-fallback)">Integrations | OpenRouter</a>: Bring your own provider keys with OpenRouter</li><li><a href="https://arxiv.org/abs/2404.06654">RULER: What&#39;s the Real Context Size of Your Long-Context Language Models?</a>: The needle-in-a-haystack (NIAH) test, which examines the ability to retrieve a piece of information (the &#34;needle&#34;) from long distractor texts (the &#34;haystack&#34;), has been widely adopted ...</li><li><a href="https://openrouter.ai/docs/integrations#bring-your-own-provider-api-keys">Integrations | OpenRouter</a>: Bring your own provider keys with OpenRouter</li><li><a href="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF">bartowski/Llama-3.2-3B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/docs/parameters#include-reasoning">Parameters | OpenRouter</a>: Configure parameters for requests</li><li><a href="https://huggingface.co/Steelskull/L3.3-MS-Nevoria-70b">Steelskull/L3.3-MS-Nevoria-70b · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/api/v1">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API, Providers, Stats</a>: DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](/meta-llama/llama-3. Run DeepSeek R1 Distill Llama 70B with API</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i7o9xo/comment/m8n3rvk/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://status.deepseek.com/">DeepSeek Service Status</a>: no description found</li><li><a href="https://tenor.com/U8PF.gif">Snow White Parody GIF - Snow White Parody - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/meta-llama/Llama-Guard-3-8B">meta-llama/Llama-Guard-3-8B · Hugging Face</a>: no description found</li><li><a href="https://github.com/gomlx/gomlx">GitHub - gomlx/gomlx: GoMLX: An Accelerated Machine Learning Framework For Go</a>: GoMLX: An Accelerated Machine Learning Framework For Go - gomlx/gomlx</li><li><a href="https://team.doubao.com/en/special/doubao_1_5_pro">no title found</a>: no description found</li><li><a href="https://api-docs.deepseek.com/faq">FAQ | DeepSeek API Docs</a>: Account
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1332472633911934987)** (436 messages🔥🔥🔥): 

> `AI Research and Development, Open Source Models, Janus Series Release, History of Internet and Technology, Federated Learning` 


- **The Role of Open Source in AI Development**: Discussion highlighted that the U.S. government's approach to limiting open source AI models may not succeed as the landscape has evolved, citing the ongoing use of such models by companies like Meta.
   - Participants noted that while the government may seek to control open source initiatives, the drive for innovation and widespread adoption could counteract these efforts.
- **Release of Janus Pro**: The Janus Pro model was announced, indicating continued advancement in DeepSeek's efforts within AI research and development.
   - Conversations suggested that DeepSeek's activities signal a relentless push in the competitive landscape of AI technologies.
- **Historical Context of Internet Development**: Several participants discussed the origins of the internet and its initial funding by the U.S. government, including the role of ARPANET.
   - The conversation highlighted various contributions to the internet's development, with acknowledgment of the complex international landscape that shaped its early infrastructure.
- **Discourse on Research Methodologies**: Debate emerged around various AI methodologies, with participants noting the differences between fundamental advancements and optimizations within research.
   - Concerns were expressed about the effectiveness of new ideas compared to proven techniques in the context of operational efficiency.
- **Challenges in Adopting New Technologies**: A participant raised concerns about the complexities of debugging new models, suggesting the need for additional layers to aid in interpretation and understanding.
   - This led to discussions on balancing innovative approaches with the necessity for transparency and comprehensibility in AI systems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/sama/status/1883294216329281627">Tweet from Sam Altman (@sama)</a>: ok we heard y’all.*plus tier will get 100 o3-mini queries per DAY (!)*we will bring operator to plus tier as soon as we can*our next agent will launch with availability in the plus tierenjoy 😊Quoting...</li><li><a href="https://www.theverge.com/2023/12/15/24003542/openai-suspends-bytedances-account-after-it-used-gpt-to-train-its-own-ai-model">OpenAI suspends ByteDance’s account after it used GPT to train its own AI model.</a>: In today’s issue of Command Line, I reported that ByteDance has been violating the developer license of both Microsoft and OpenAI by using GPT-generated data to train its own, competing model in China...</li><li><a href="https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue">List of largest companies by revenue - Wikipedia</a>: no description found</li><li><a href="https://tenor.com/view/chess-gif-25810828">Chess GIF - Chess - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/michael-jackson-comendo-picoca-gif-9669437860846841235">Michael Jackson Comendo Picoca GIF - Michael Jackson comendo picoca - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/adder-adderko-snake-ouroboros-overwerk-gif-21047022">Adder Adderko GIF - Adder Adderko Snake - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/i-swear-to-god-nick-thorpe-fbi-international-i-promise-cross-my-heart-gif-26205749">I Swear To God Nick Thorpe GIF - I Swear To God Nick Thorpe Fbi International - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/capitulo0-capitulo-cero-ernesto-sevilla-david-lynch-chanante-gif-15470197">Capitulo0 Capitulo Cero GIF - Capitulo0 Capitulo Cero Ernesto Sevilla - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://d2l.ai/chapter_introduction/index.html">1. Introduction &#8212; Dive into Deep Learning 1.0.3 documentation</a>: no description found</li><li><a href="https://fxtwitter.com/sama/status/1883185690508488934">Tweet from Sam Altman (@sama)</a>: A revolution can be neither made nor stopped. The only thing that can be done is for one of several of its children to give it a direction by dint of victories.-Napoleon</li><li><a href="https://fxtwitter.com/sama/status/1883305404089901269">Tweet from Sam Altman (@sama)</a>: fun watching people react to operator. reminds me of the chatgpt launch!</li><li><a href="https://tenor.com/view/dahliabunni-popcorn-gif-11542556772657816665">Dahliabunni Popcorn GIF - Dahliabunni Popcorn - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=iSmS6_j8Tnw">[IM] Kung Fu Master Table Fight - IP Man</a>: LIKE AND SUBSCRIBE FOR MORE EPICLIPS...[IM] Kung Fu Master Table Fight - IP Man#IPMan #KungFuCopyright Disclaimer Under Section 107 of the Copyright Act 1976...</li><li><a href="https://www.youtube.com/watch?v=i5Sdqf3jQkE">They&#39;re ALL wrong about the future of robotics.</a>: Robotics is on the brink of a historic moment, but what the industry needs is a different moment entirely.</li><li><a href="https://www.youtube.com/watch?v=kPRA0W1kECg">15 Sorting Algorithms in 6 Minutes</a>: Visualization and &quot;audibilization&quot; of 15 Sorting Algorithms in 6 Minutes.Sorts random shuffles of integers, with both speed and the number of items adapted t...</li><li><a href="https://www.youtube.com/watch?v=mhKC3Avqy2E">Training large language models to reason in a continuous latent space – COCONUT Paper explained</a>: AI doesn’t have to think with words. We explain COCONUT (Chain of Continuous Thought) 🥥, a new paper that makes Chain-of-Thought work with vectors instead o...</li><li><a href="https://tenor.com/view/kuuchuu-buranko-ichiro-irabu-devi-word-of-the-day-irabu-ichiro-gif-26530271">Kuuchuu Buranko Ichiro Irabu GIF - Kuuchuu Buranko Ichiro Irabu Devi Word Of The Day - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reia.org/">Home</a>: no description found</li><li><a href="https://www.youtube.com/shorts/VNv-Cz-U6AY">&quot;Man Rescues Helpless Octopus Stuck in Seashell&quot; #viralshort</a>: This heartwarming video shows a man rescuing a helpless octopus stuck inside a seashell on the beach. Watch their bond grow as he cares for and plays with th...</li><li><a href="https://reia.org/">Home</a>: no description found</li><li><a href="https://github.com/deepseek-ai/Janus">GitHub - deepseek-ai/Janus: Janus-Series: Unified Multimodal Understanding and Generation Models</a>: Janus-Series: Unified Multimodal Understanding and Generation Models - deepseek-ai/Janus</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>: Janus-Series: Unified Multimodal Understanding and Generation Models - deepseek-ai/Janus
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1332450182691815497)** (39 messages🔥): 

> `GPRO and PPO, Deepseek papers, Qwen2.5-VL model, Janus-Pro model, Friston podcast` 


- **GPRO's Impact on PPO Convergence**: A discussion arose about whether GPRO's removal of the **Value Function** and **Generalised Advantage Estimation (GAE)** could alleviate stuck loss and early convergence issues in **PPO**.
   - It was noted that while GAE uses a discounted sum approach, GPRO's method utilizes a globally normalized reward pattern.
- **Deepseek Papers Keep Coming**: Members expressed excitement over **Deepseek** continuously releasing new papers, with a focus on their latest, **Janus-Pro** and **Qwen2.5-VL** models.
   - Questions about the differences between **Qwen2.5-VL** and previous models highlighted their advancements in video understanding, showcasing their rapid progress.
- **Qwen2.5-VL Enhancements**: The **Qwen2.5-VL** model's ability to understand complex visuals and act as an agent capable of interacting with tools was a major point of discussion.
   - Members noted significant features like analyzing images and videos, making **Qwen2.5-VL** a notable advancement in vision-language models.
- **Friston Podcast Insights**: A member shared a podcast featuring **Karl Friston**, discussing pivotal insights relating neuroscience to intelligence, including adaptive organizational changes.
   - Key themes included the importance of **active inference** and the need for ecosystem harmony in sustainable innovation.
- **Event Announcements for New Papers**: Upcoming discussions about **Janus-Pro** and **Qwen2.5-VL** were scheduled, indicating a focus on understanding their implications in AI development.
   - Members were encouraged to join the events, especially as these papers were recently released and gaining momentum.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">Qwen/Qwen2.5-VL-72B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://open.spotify.com/episode/3ZAPncRTDzGGJSsFhlgtaB">Episode #45 | Karl Friston | Active intelligence, non-equillibrium steady states and enterprise jazz</a>: The Only Constant · Episode</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ibhew9/qwen_just_launced_a_new_sota_multimodal_model/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/bytedance/UI-TARS-desktop">GitHub - bytedance/UI-TARS-desktop: A GUI Agent application based on UI-TARS(Vision-Lanuage Model) that allows you to control your computer using natural language.</a>: A GUI Agent application based on UI-TARS(Vision-Lanuage Model) that allows you to control your computer using natural language. - bytedance/UI-TARS-desktop</li><li><a href="https://arxiv.org/abs/2501.12326">UI-TARS: Pioneering Automated GUI Interaction with Native Agents</a>: This paper introduces UI-TARS, a native GUI agent model that solely perceives the screenshots as input and performs human-like interactions (e.g., keyboard and mouse operations). Unlike prevailing age...</li><li><a href="https://github.com/bytedance/UI-TARS">GitHub - bytedance/UI-TARS</a>: Contribute to bytedance/UI-TARS development by creating an account on GitHub.</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>: Janus-Series: Unified Multimodal Understanding and Generation Models - deepseek-ai/Janus
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1332931480959320144)** (17 messages🔥): 

> `Natural Language to DSL Code Resources, PydanticAI Framework, Structured Output for Generative AI, Workout Logging App Use Case, DSL vs JSON Discussion` 


- **Seeking Resources for NL to DSL Code**: A member requested resources for converting **Natural Language** to **Domain Specific Language** code, expressing that **LLMs** struggle with this task without substantial fine-tuning.
   - They mentioned the **Microsoft ODSL paper** as a good starting point while being open to further suggestions.
- **PydanticAI Framework Exploration**: Discussion turned to the **PydanticAI** framework, which aims to simplify building production-grade applications with Generative AI, with a link shared for reference.
   - One individual expressed uncertainty about its beta status, while others suggested alternatives like **LlamaIndex+LangChain** for structured output.
- **Defining Use Case for a Workout Logging App**: A member outlined their exploration of a **workout logging app**, aiming for efficient user interactions and the potential for composable interactions in **DSL**.
   - They emphasized the goal of achieving **voice to DSL** conversion, acknowledging it as a difficult challenge requiring stepwise exploration.
- **Importance of Executability in DSLs**: The conversation highlighted the distinction between **DSLs** and **JSON**, with a member asserting that DSLs are executable and need to be more complex than JSON.
   - A contrasting opinion suggested that JSON-like instances could function as a DSL, capable of supporting executable steps.
- **Methodology for Learning LLM Data Extraction**: Members discussed creating basic **'hello world'** examples using structured output to understand the types of data that **LLMs** can extract from prompts.
   - This practical approach was recommended to clarify how structured inputs can influence the DSL's behavior in a workout logging context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://table-agent.com/">Table Agent</a>: AI-powered data assistant for tabular research</li><li><a href="https://ai.pydantic.dev/">Introduction</a>: Agent Framework / shim to use Pydantic with LLMs
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1332443465497182230)** (45 messages🔥): 

> `DeepSeek AI Model Launch, Qwen2.5-VL Model Announcement, Mistral's IPO Plans, AI and Economic Impact, Public Perception of AI Governance` 


- **DeepSeek AI challenges US competitors**: DeepSeek has developed a reasoning model that reportedly outperforms US counterparts without access to advanced Nvidia chips, raising questions about the true competitive edge of US tech firms.
   - This has led to speculations about its potential impact on the stock market, as industry leaders like Satya Nadella take DeepSeek seriously.
- **Qwen2.5-VL unveiled for Chinese New Year**: Alibaba Qwen announced the launch of their flagship vision-language model, **Qwen2.5-VL**, which features advanced visual understanding and long video comprehension capabilities.
   - Highlights include its precise localization abilities and structured data outputs that can enhance tasks in finance and commerce.
- **Mistral's questionable IPO statements**: A discussion arose around conflicting messages regarding Mistral’s status, where it claimed it is 'not for sale' yet is reportedly working on an IPO.
   - This led to tongue-in-cheek comments about corporate double-think in the tech industry.
- **Public sentiment on AI governance**: Members shared mixed views on the idea of AI governance, noting a willingness to accept AI oversight if it improves their lives.
   - Debates highlighted the challenges of corruption and incentive structures in both human and AI-led systems.
- **Perceived risks of AI leadership**: The conversation touched on the notion that if AI could provide a better quality of life, many would accept AI leadership, despite potential downsides.
   - Critics raised concerns about the implications of such a mentality, comparing it to historical failures in governance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vxtwitter.com/TFTC21/status/1882571514891080030">Tweet from undefined</a>: no description found</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1883954247743725963">Tweet from Qwen (@Alibaba_Qwen)</a>: 🎉 恭喜发财🧧🐍 As we welcome the Chinese New Year, we&#39;re thrilled to announce the launch of Qwen2.5-VL , our latest flagship vision-language model! 🚀💗 Qwen Chat: https://chat.qwenlm.ai📖 Blog: http...</li><li><a href="https://www.msn.com/en-gb/money/other/is-deepseek-about-to-cause-a-stock-market-crash/ar-AA1xV6nG">MSN</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: no description found</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-1m/">Qwen2.5-1M: Deploy Your Own Qwen with Context Length up to 1M Tokens</a>: Tech Report HuggingFace ModelScope Qwen Chat HuggingFace Demo ModelScope Demo DISCORDIntroduction Two months after upgrading Qwen2.5-Turbo to support context length up to one million tokens, we are ba...</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://tenor.com/view/rodney-king-get-along-gif-22105666">Rodney King Get Along GIF - Rodney King Get Along - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/ag2oss/status/1882878967713259705">Tweet from AG2 (@ag2oss)</a>: Announcing our vision for community-driven agent development.Read about AG2&#39;s:- Governance model- Community structure- Open source commitment- Path forwardhttps://medium.com/@ag2ai/ag2s-vision-for...</li><li><a href="https://medium.com/@ag2ai/ag2s-vision-for-community-driven-agent-development-f9f3ca2b0dc8">AG2’s Vision for Community-Driven Agent Development</a>: When we first developed the concepts behind AutoGen while working on FLAML two years ago, we were driven by a simple goal: make it easier…</li><li><a href="https://www.podchaser.com/podcasts/chinatalk-725507?">ChinaTalk</a>: With Jordan Schneider, 426 episodes, 2 ratings &amp; reviews. Conversations exploring China, technology, and US-China relations. Guests include a wide range of analysts, policymakers, and academics. H...</li><li><a href="https://uk.finance.yahoo.com/news/deepseek-cause-stock-market-crash-065127717.html">Is DeepSeek about to cause a stock market crash?</a>: With the stock market dominated by US tech companies focused on AI, is DeepSeek's competitor to OpenAI about to brings things crashing down? The post Is DeepSeek about to cause a stock market crash? a...</li><li><a href="https://www.youtube.com/watch?v=V-Fla5hxMRg">China&#39;s DeepSeek triggers global tech sell-off</a>: CNBC&#39;s Andrew Ross Sorkin and Becky Quick discuss the news of the day. For access to live and exclusive video from CNBC subscribe to CNBC PRO: https://cnb.cx...</li><li><a href="https://tenor.com/view/bogdanoff-dump-it-stocks-crypto-gif-20477588">Bogdanoff Dump It GIF - Bogdanoff Dump It Stocks - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/huggingface/open-r1">GitHub - huggingface/open-r1: Fully open reproduction of DeepSeek-R1</a>: Fully open reproduction of DeepSeek-R1. Contribute to huggingface/open-r1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1332446763919085649)** (206 messages🔥🔥): 

> `DeepSeek Model Updates, Qwen 2.5-VL Launch, AI Company Strategies, NVIDIA Market Position, Edge AI Discussion` 


- **DeepSeek Model Updates Gain Attention**: The recent release of the DeepSeek model sparked discussions on its potential to disrupt current AI paradigms, with many observers noting its efficiency and open-weight nature.
   - There's an expectation that DeepSeek will lead to increased adoption of reasoning models, despite uncertainty about widespread corporate integration.
- **Qwen 2.5-VL Launch Excites Audience**: Alibaba's Qwen 2.5-VL has been introduced as a flagship vision-language model, featuring capabilities like long video comprehension and precise localization.
   - The announcement showcases Qwen's commitment to advancing AI's ability to integrate visual and linguistic processing effectively.
- **AI Company Strategies Diverge**: Discussions reveal a gap between companies like OpenAI, which seems to be diversifying into many domains, versus Anthropic's focused approach on LLM development.
   - There's skepticism about OpenAI's consumer-focused strategy, with thoughts that it may have distracted from maintaining cutting-edge research.
- **NVIDIA Faces Market Challenges Amid AI Evolution**: Concerns are growing around NVIDIA's market valuation as the shift towards efficiency and commoditization of AI models raises questions about their future demand.
   - With rising competition and technological advancements, investors are reassessing NVIDIA's position in the AI hardware landscape.
- **Debate on Edge AI Viability**: The viability of edge AI is questioned as most complex reasoning models may not be suitable for local deployment without performance sacrifices.
   - Conversations suggest that while there may be use cases for edge AI, cloud solutions continue to be the preferred option for processing heavy workloads.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/junxian_he/status/1883183099787571519">Tweet from Junxian He (@junxian_he)</a>: We replicated the DeepSeek-R1-Zero and DeepSeek-R1 training on 7B model with only 8K examples, the results are surprisingly strong. 🚀 Starting from Qwen2.5-Math-7B (base model), we perform RL on it d...</li><li><a href="https://blog.vllm.ai/2025/01/27/v1-alpha-release.html">vLLM V1: A Major Upgrade to vLLM’s Core Architecture</a>: no description found</li><li><a href="https://huggingface.co/spaces/Trudy/gemini-image-to-code">Gemini Image to Code - a Hugging Face Space by Trudy</a>: no description found</li><li><a href="https://x.com/bfspector/status/1883051606369001873">Tweet from Benjamin F Spector (@bfspector)</a>: We got early access to some of the very first Nvidia B200’s. We share initial benchmark results and wrote the fastest (public) attention kernel with 925+ BF16 TFLOPs:Since the PTX instruction set rele...</li><li><a href="https://x.com/Kimi_Moonshot/status/1883164161506738232">Tweet from Kimi.ai (@Kimi_Moonshot)</a>: 🚀 Introducing Kimi k1.5 – Now on Web http://Kimi.ai! We’re excited to announce the launch of Kimi 1.5 on the web! We&#39;ve also rolled out English support (still fine-tuning). Check out the easy mod...</li><li><a href="https://x.com/DanHendrycks/status/1883660982641426727">Tweet from Dan Hendrycks (@DanHendrycks)</a>: I agree. From a U.S. competitiveness perspective you need domestic AI chip manufacturing, not just restrictions on AI chips through export controls.There are many axes which affect AI capabilities: al...</li><li><a href="https://x.com/_lewtun/status/1883142636820676965">Tweet from Lewis Tunstall (@_lewtun)</a>: We are reproducing the full DeepSeek R1 data and training pipeline so everybody can use their recipe. Instead of doing it in secret we can do it together in the open!🧪 Step 1: replicate the R1-Distil...</li><li><a href="https://x.com/lmarena_ai/status/1882875989610594542">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: ❤️‍🔥WebDev Arena Update: Exciting new entries!- #2: @deepseek_ai DeepSeek-R1- #4: New Gemini-2.0-Flash-ThinkingDeepSeek-R1 jumps to #2 with only &lt;40 pts gap to Claude 3.5 Sonnet, showing strong ca...</li><li><a href="https://x.com/TheStalwart/status/1883902565064352233">Tweet from Joe Weisenthal (@TheStalwart)</a>: *DEEPSEEK: RESTRICTS REGISTRATION TO CHINA MOBILE PHONE NUMBERS</li><li><a href="https://x.com/alibaba_qwen/status/1883954247743725963?s=46">Tweet from Qwen (@Alibaba_Qwen)</a>: 🎉 恭喜发财🧧🐍 As we welcome the Chinese New Year, we&#39;re thrilled to announce the launch of Qwen2.5-VL , our latest flagship vision-language model! 🚀💗 Qwen Chat: https://chat.qwenlm.ai📖 Blog: http...</li><li><a href="https://x.com/nrehiew_/status/1882853607307100162">Tweet from wh (@nrehiew_)</a>: Anthropic, I&#39;ve done up the pitch for you. Ignore Stargate this is a 1T pitch right hereTop right = good</li><li><a href="https://x.com/LiJunnan0409/status/1882620700567195976">Tweet from Li Junnan (@LiJunnan0409)</a>: Excited to share that my http://Rhymes.ai research team is joining Salesforce Research @SFResearch! I’ll be stepping into the role of Director of Research, reporting to @CaimingXiong. Looking forward ...</li><li><a href="https://x.com/Mobius_Labs/status/1882841665427390858">Tweet from Mobius Labs (@Mobius_Labs)</a>: Our re-distilled @deepseek_ai R1 (1.5B) outperforms the original distilled model! Get it at https://huggingface.co/mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0. We’re distilling more models and...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1332838823302004867)** (16 messages🔥): 

> `Self-Play Paradigm, Scaling Synthetic Data Pipelines, Role of Self-Determination Theory, Claims about Deepseek, Critique of Media Reporting` 


- **Self-Play Paradigm Gains Support**: *Self-play without a human-in-the-loop* is being discussed as a primary framework for future advancements in AI, supported by enthusiasts within the community.
   - A community member quoted that *the next jump in AI capability will stem from optimized frameworks* like MuZero.
- **Synthetic Data Pipelines Are Key**: A member emphasized the importance of scaling *synthetic data pipelines* rather than relying solely on innovative training methods.
   - They expressed skepticism towards claims that foundational data was lacking for AI models.
- **Self-Determination Theory in AI Discussion**: A query arose on the inclusion of *Self-Determination Theory* in discussions around human-like reasoning in AI, with a focus on psychological frameworks.
   - Community members expressed interest in literature surrounding therapy-oriented chatbots and positive psychology.
- **Claims about Deepseek's Data Sources**: Questions emerged regarding whether *Deepseek* disclosed the use of *Llama* and *Qwen* for data generation, with some citing miscommunication.
   - The claim appeared to originate from a reference to *Deepseek-LLM*, which is based on Llama architecture.
- **Critique of Media Reporting Practices**: There was dissatisfaction with how reporters handle AI topics, especially regarding claims of invention, like stating that *MoE was invented at Deepseek*.
   - Community members suggested forming a chat where reporters could receive feedback from those well-versed in AI literature to avoid misinformation.



**Link mentioned**: <a href="https://x.com/finbarrtimbers/status/1883243939031056813">Tweet from finbarr (@finbarrtimbers)</a>: co-signQuoting doomslide (@doomslide) after reading the papers again and playing with R1 i&#39;ve come to the conclusion that the extremely predictable next jump in capability will be (cleverly optimi...

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1332990612559106099)** (38 messages🔥): 

> `Gary Marcus's Opinions, Nous Psyche Announcement, Political Perspectives on Nation States, AI Breakthrough Narratives, Academic Perspectives on AI` 


- **Gary Marcus Stirs Controversy**: Members expressed confusion and humor regarding Gary Marcus's positions, with one noting his shift towards being a 'China hawk' to gain attention.
   - *Many feel he plays well with unconflicted audiences in academia and journalism who lack deep AI knowledge*, often ignoring his contradictory statements.
- **Nous Psyche's Ambitious Launch**: Nous Research announced **Nous Psyche**, a cooperative training network for generative AI aiming to challenge the idea that only closed labs can advance superintelligence, built on **Solana**.
   - Despite the excitement, there were concerns raised about security after a mention that ***Nous got hacked***.
- **Frustration with Nation States**: A member questioned the relevance of nation states, stating that the idea of national superiority is outdated and criticizing the ruling elite's control over markets.
   - This sentiment reflected broader feelings about the need for change in political structures, resonating with others in the discussion.
- **Dark Humor in AI Discussions**: Amidst technical discussions, a member found humor in a post-edit comment associated with an AI development announcement, calling it comedic gold.
   - This light-heartedness contrasted with the deeper concerns about the implications of AI and platform security.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NousResearch/status/1883912370696704011">Tweet from Nous Research (@NousResearch)</a>: Recent AI breakthroughs challenge the status quo narrative that only closed, mega labs have the ability to push the frontier of superintelligence.Today we announce Nous Psyche built on @Solana - a coo...</li><li><a href="https://x.com/rm_rafailov/status/1883419883150713023">Tweet from Rafael Rafailov @ NeurIPS (@rm_rafailov)</a>: @teortaxesTex RLVR is not a thing. We’ve trained this in at least 1000 papers over years. After the last three days I’m seriously considering giving up on public research.</li><li><a href="https://bsky.app/profile/mathiasgehrig.bsky.social/post/3lgqwb3rwtk2k">Mathias Gehrig (@mathiasgehrig.bsky.social)</a>: The last statement must be the most wrong thing I read today. I get you are proud to be American or whatever, but still.</li><li><a href="https://x.com/LiangWenfeng_/status/1883978669900824681">Tweet from Liang Wenfeng 梁文锋 (@LiangWenfeng_)</a>: no description found</li><li><a href="https://x.com/steph_palazzolo/status/1883620099862773842">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: NEW: American AI firms are scrambling after a Chinese hedge fund released an impressive and uber-cheap AI model.Meta has set up 4 &#34;war rooms&#34; to dissect the DeepSeek model to see what insights...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1332447389839261796)** (177 messages🔥🔥): 

> `DeepSeek's Rise, Market Reactions to AI Models, Qwen Model Launch, Investor Sentiment, Industry Disruptions` 


- **DeepSeek's Impact on AI Market**: The release of DeepSeek's R1 model has created significant buzz, with discussions about its competitive performance against established models from OpenAI and Meta.
   - With DeepSeek surpassing expectations, many industry professionals and even casual users, like family members, are now inquiring about it.
- **Qwen2.5-VL's Launch**: The anticipated release of the Qwen2.5-VL model is generating excitement, with references to its multimodal capabilities and potential impact on competitors.
   - Observers are noting how this model release could reshape perceptions in the AI landscape, similar to past notable launches.
- **Investor Concerns Post-Launch**: The financial markets are reacting to AI developments; some believe the surge in AI model releases is causing substantial fluctuations in stock values, particularly for companies like Nvidia.
   - Investors are discussing the risks involved, citing concerns over market saturation and the future of AI investments.
- **Public Interest and AI Awareness**: There's a notable increase in public interest regarding AI, with more individuals asking about innovative models and technologies, including DeepSeek.
   - Participants in discussions report friends and family reaching out to them about these advancements, indicating a growing awareness outside traditional tech circles.
- **Comparing AI and Stock Market Dynamics**: Analysts and users are debating the implications of this fast-paced AI development on stock markets, noting unexpected downturns for established companies like Nvidia.
   - Discussions include predictions about significant consequences on stock prices, fueled by movements in AI-related technologies and capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwen2.5-1m/">Qwen2.5-1M: Deploy Your Own Qwen with Context Length up to 1M Tokens</a>: Tech Report HuggingFace ModelScope Qwen Chat HuggingFace Demo ModelScope Demo DISCORDIntroduction Two months after upgrading Qwen2.5-Turbo to support context length up to one million tokens, we are ba...</li><li><a href="https://x.com/garryt">Tweet from undefined</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=42825193">For context: R1 is a reasoning model based on V3. DeepSeek has claimed that GPU ... | Hacker News</a>: no description found</li><li><a href="https://x.com/splitbycomma/status/1883588991813042605">Tweet from caspian (@splitbycomma)</a>: THE NORMIES THINK DEEPSEEK IS CUTE BECAUSE IT SHARES ITS THOUGHT PROCESS</li><li><a href="https://x.com/DavidSHolz/status/1883222685741879722">Tweet from David (@DavidSHolz)</a>: in my testing, deepseek crushes western models on ancient chinese philosophy and literature, while also having a much stronger command of english than my first-hand chinese sources. it feels like comm...</li><li><a href="https://fxtwitter.com/sethbannon/status/1883301772053332349?s=46">Tweet from Seth Bannon (@sethbannon)</a>: Youtube and Reddit have both blocked Operator. Sign of things to come?</li><li><a href="https://x.com/willccbb/status/1883414339518148960?s=61">Tweet from will brown (@willccbb)</a>: GRPO self-correction on Llama-1B :&#39;)</li><li><a href="https://x.com/steph_palazzolo/status/1883620099862773842?s=46">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: NEW: American AI firms are scrambling after a Chinese hedge fund released an impressive and uber-cheap AI model.Meta has set up 4 &#34;war rooms&#34; to dissect the DeepSeek model to see what insights...</li><li><a href="https://x.com/TheXeophon/status/1883048355875672457">Tweet from Xeophon (@TheXeophon)</a>: Analysis of the first four words of R1&#39;s CoT</li><li><a href="https://x.com/johnschulman2/status/1883221980931142113">Tweet from John Schulman (@johnschulman2)</a>: There are some intriguing similarities between the r1 chains of thought and the o1-preview CoTs shared in papers and blog posts (eg https://openai.com/index/learning-to-reason-with-llms). In particula...</li><li><a href="https://x.com/huybery/status/1883775353950519479">Tweet from Binyuan Hui (@huybery)</a>: There are some surprises tonight</li><li><a href="https://x.com/hamelhusain/status/1883707463251472448?s=46">Tweet from Hamel Husain (@HamelHusain)</a>: A distilled 70b-R1 is on Groq now.  Kinda hidden in the docs, but its there.</li><li><a href="https://x.com/garrytan/status/1883655771067744441">Tweet from Garry Tan (@garrytan)</a>: DeepSeek search feels more sticky even after a few queries because seeing the reasoning (even how earnest it is about what it knows and what it might not know) increases user trust by quite a lot</li><li><a href="https://x.com/dylan522p/status/1883930768533332270">Tweet from Dylan Patel (@dylan522p)</a>: So this is like 2T in loss market cap for a $6M training run (ignoring cost of research, ablations, distilled data from GPT, capex for their various clusters, etc).Imagine if China invests $300m in a ...</li><li><a href="https://x.com/reach_vb/status/1883911714158305719">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: WAIT A SECOND, DeepSeek just dropped Janus 7B  (MIT Licensed) - multimodal LLM (capable of generating images too) 🔥</li><li><a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Qwen2.5-VL - a Qwen Collection</a>: no description found</li><li><a href="https://x.com/georgejrjrjr/status/1883629241742635313?s=61">Tweet from George (@georgejrjrjr)</a>: They knew! How do I know?Meta reached out a year ago to see about hiring me based on one of my whaleposts!Quoting Amir Efrati (@amir) news: the DeepSeek freakout is ~real~Meta Platforms, worried DS is...</li><li><a href="https://x.com/btibor91/status/1883627800831365567?s=61">Tweet from Tibor Blaho (@btibor91)</a>: The Information reports that High-Flyer Capital’s DeepSeek AI has surpassed Meta’s Llama and rivaled OpenAI models in performance and cost-efficiency, triggering concerns and a rapid response from Met...</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>: Janus-Series: Unified Multimodal Understanding and Generation Models - deepseek-ai/Janus</li><li><a href="https://www.thetimes.com/world/europe/article/french-ai-lucie-looks-tres-chic-but-keeps-getting-answers-wrong-7vk2szmdg">French AI ‘Lucie’ looks très chic, but keeps getting answers wrong</a>: The chatbot, backed by Macron and public funds, is facing criticism after providing inaccurate information. Even its logo has been questioned</li><li><a href="https://github.com/QwenLM/Qwen2-VL/commits/main/">Commits · QwenLM/Qwen2.5-VL</a>: Qwen2-VL is the multimodal large language model series developed by Qwen team, Alibaba Cloud. - Commits · QwenLM/Qwen2.5-VL
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1332512167026626613)** (23 messages🔥): 

> `Deepseek Performance, Scale.ai Concerns, Chinese Tech Commentary, Fake Accounts on Social Media, Cultural Revolution Impact on Tech Founders` 


- **Deepseek Performance Compared**: A member stated that the combination of **o1 paired with Sonnet** did not yield better results compared to using **o1** alone, highlighting concerns about model performance.
   - Another member noted that various models as editors did not improve scores for **o1** or **R1** compared to their solo performances.
- **Concerns Over Scale.ai CEO's Statements**: A member pointed out that the **Scale.ai** CEO, who mentioned having **50k GPUs**, risks significant financial losses if the demand for labeled data declines.
   - This raises questions about the sustainability of **Scale.ai**'s model amidst changing market needs.
- **Discussion on Chinese Tech Perspectives**: Comments emerged around **Alex Wang**'s views on Chinese technology, noting that his family background likely skews his perspective due to their **immigrant history**.
   - The conversation referenced the **Cultural Revolution** as a significant period affecting perspectives of individuals in the tech industry.
- **Fake Accounts and Misinformation**: Concerns were raised about a potentially **fake account** spreading incorrect information, leading to doubts about the credibility of the source.
   - Discussion ensued regarding whether the information disseminated should be attributed to **real or fake identities**.
- **Clarifying Identity Confusions**: A member clarified confusion over identities, indicating that a person mentioned might just have the same name as another individual on the **Deepseek** team.
   - This further emphasized the complexities in tracking discussions related to personnel in tech projects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/paulgauthier/status/1882837818101359001?s=46">Tweet from Paul Gauthier (@paulgauthier)</a>: o1 paired with Sonnet didn’t produce better results than just using o1 alone. Using various other models as editor didn’t seem to improve o1 or R1 versus their solo scores.</li><li><a href="https://x.com/dylan522p/status/1883569162100080875?s=46">Tweet from Dylan Patel (@dylan522p)</a>: Deepseek V3 and R1 discourse boils down to this. Shifting the curve means you build more and scale more dummies</li><li><a href="https://x.com/gzilgalvis/status/1883107575010619649?s=46">Tweet from Gustavs Zilgalvis (@GZilgalvis)</a>: this is not harmless behavior deepseek-r1</li><li><a href="https://fxtwitter.com/wordgrammer/status/1883448109814206892">Tweet from wordgrammer (@wordgrammer)</a>: Once, I met the founder of a YC backed startup. He said that his main project was a to-do list app. But he noticed that his GPU clusters weren’t being used in off-hours (when all the tasks were done)....
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1333110343966392421)** (4 messages): 

> `REINFORCE acronym, Writing RLHF book, Open-Instruct integration with vLLM, OpenRLHF framework maintenance` 


- **REINFORCE stands for something!**: The term **REINFORCE** is an acronym for 'REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility'.
   - This brings a bit of clarity to its complex function within reinforcement learning.
- **Writing RLHF book reveals insights**: A member humorously noted that they are 'really learning a lot writing RLHFbook', emphasizing the journey of writing.
   - *Learning through writing* seems to be a mantra for several members lately!
- **Questions on Open-Instruct's vLLM integration**: Discussion arose regarding the potential internal use of **Open-Instruct**, with specific focus on its integration with **vLLM**.
   - Concerns were raised about the **maintenance** of this integration if the **OpenRLHF** framework is not sustained.
- **Maintaining OpenRLHF and vLLM integration**: A member sought clarification on whether **AllenAI** plans to maintain the **OpenRLHF** integration for newer versions of **vLLM**.
   - Their apprehension stems from not wanting to depend on an OSS project that might face future **maintenance** challenges.


  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1333469136327675980)** (4 messages): 

> `Tulu3 Paper Analysis, Pref Tuning Challenges, Anticipation for Tulu4` 


- **Revisiting Tulu3's Off-Policy Data**: A member noted the inclusion of some **off-policy data** in the **Tulu3** preference data despite evidence suggesting that **on-policy** is better given the same number of data points.
   - *Why was this choice made?*
- **Ongoing Challenges in Preference Tuning**: Another member voiced that there are still **lots of hills to climb** in **pref tuning**, indicating ongoing concerns and complexities in the area.
   - *There remains significant room for improvement.*
- **Excitement for Tulu4's Release**: Looking ahead, a member expressed eagerness for **Tulu 4**, hinting at expectations for advancements and improvements.
   - *The anticipation signals a hopeful continuation in the series.*


  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/)** (1 messages): 

the_real_jrb: It's here! Qwen2.5-VL. https://qwenlm.github.io/blog/qwen2.5-vl/
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1332743114246459484)** (15 messages🔥): 

> `DeepSeek R1 Release, Market Reaction to DeepSeek, AIW Problem Variations, John Schulman's Commentary, Jay Alammar's Analysis` 


- **DeepSeek R1 struggles to impress**: According to a commentary by [JJitsev](https://x.com/jjitsev/status/1883158738661691878?s=46), DeepSeek R1 is claimed to match o1/o1-preview on olympiad level math & coding problems, but there are doubts on its efficacy regarding **generalization** and **reasoning deficits** in SOTA LLMs.
   - *Is R1 actually good?* was a recurring question, hinting at concerns over its performance.
- **Market panics over DeepSeek's app launch**: [Zvi's article](https://thezvi.substack.com/p/deepseek-panic-at-the-app-store) noted that while DeepSeek released various versions, market reactions only came after launching an app, depicting a **discrepancy in market efficiency**.
   - The S&P and Nvidia stocks fell sharply, showcasing that market reactions are often unpredictable and not necessarily linked to the actual events.
- **AIW problem structures discussed**: A discussion emphasized that AIW instances, generated from structured templates, allow natural problem variations that should not alter **difficulty** or **solvability**.
   - The usefulness of such reasoning problems was questioned, with one member calling them as relevant as counting letters in a word.
- **John Schulman draws connections**: John Schulman pointed out intriguing similarities between the **r1 chains of thought** and **o1-preview CoTs**, noting the frequent use of transition phrases for error correction, as stated in his [tweet](https://x.com/johnschulman2/status/1883221980931142113?s=46).
   - This commentary sparked discussions about how different reasoning models may converge on similar outcomes, highlighting varied interpretations within the community.
- **Jay Alammar highlights DeepSeek's significance**: Jay Alammar's analysis discusses how DeepSeek-R1 represents a significant development in AI with its **open weights model** and insights on training methods for reasoning models similar to OpenAI's O1.
   - His [draft post](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1) reflects on the importance of transparency and reproducibility in model training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1">The Illustrated DeepSeek-R1</a>: A recipe for reasoning LLMs</li><li><a href="https://thezvi.substack.com/p/deepseek-panic-at-the-app-store">DeepSeek Panic at the App Store</a>: DeepSeek released v3.</li><li><a href="https://x.com/johnschulman2/status/1883221980931142113?s=46">Tweet from John Schulman (@johnschulman2)</a>: There are some intriguing similarities between the r1 chains of thought and the o1-preview CoTs shared in papers and blog posts (eg https://openai.com/index/learning-to-reason-with-llms). In particula...</li><li><a href="https://x.com/jjitsev/status/1883158738661691878?s=46">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev)</a>: (Yet) another tale of Rise and Fall:  DeepSeek R1 is claimed to match o1/o1-preview on olympiad level math & coding problems. Can it handle versions of AIW problems that reveal generalization & basic ...</li><li><a href="https://x.com/JJitsev/status/1883158749785006533">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev)</a>: AIW instances are generated from a template that defines problem structure. Importantly, we can introduce natural problem variations by doing modifications that do not change structure or its difficul...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1332736823709011999)** (4 messages): 

> `Job Board Launch, Channel Inappropriateness` 


- **Discussion on Channel Inappropriateness**: A member expressed concern about the appropriateness of a post in the channel, mentioning urgency in providing a model for an ill author.
   - *It was suggested that the content might be too self-promotional* based on the feedback.
- **Upcoming Job Board Concept**: A member hinted at the potential launch of a **job board-like** platform in the future.
   - This prompts interest as it may address community employment needs and opportunities.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1333425817375604798)** (6 messages): 

> `Deepseek Understanding, Chatbot Formatting, Community Engagement` 


- **Community Buzz Around Deepseek**: A member noted that a post about **Deepseek** is being circulated within a finance firm, indicating interest in understanding its implications.
   - *This sharing highlights the growing curiosity surrounding Deepseek's functionalities and applications.*
- **Improving Chatbot Format**: A member emphasized the attempt to make the **chatbot format** more useful for community discussions.
   - *This initiative aims to enhance clarity and engagement among users.*
- **Positive Community Feedback**: A member expressed appreciation for a post, simply stating, '**good post**'.
   - *The friendly exchanges foster a supportive community atmosphere.*
- **Welcome Message to New Members**: A member greeted Florian, reinforcing a welcoming community approach.
   - *Such gestures help in building connections within the group.*


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1332800307842908272)** (15 messages🔥): 

> `China's New AI Policy, US Industrial Policy, Great Power Competition in AI, CHIPS Act, Jones Act and Defense Manufacturing` 


- **China's AI Industry Gets a Massive Boost**: China announces a new AI policy including **1 trillion yuan** ($137 billion) to support its AI industry over the next five years, as highlighted by [@rwang07](https://x.com/rwang07/status/1883210410763121073).
   - This initiative is described as potentially the **most important Chinese AI policy** for 2025.
- **US Administrative Struggles with Industrial Policy**: Discussion emerged on the **US government's low ability** to implement effective industrial policy, particularly under the Republican party.
   - There is skepticism around political willingness to support essential policies like skilled immigration and emerging tech sectors.
- **AI Race Sparks Military Industry Interest**: Commentators suggest that the **Republicans might mobilize resources** for AI by framing it within the narrative of great power competition, similar to the Cold War's missile gap.
   - There is a belief that funding may be easier to secure for military applications of AI than for industrial advancements.
- **Historical Context of Industrial Policy**: The **CHIPS Act** is recalled as a significant bipartisan effort to enhance US semiconductor production and technology.
   - However, GPU export controls remain a contentious issue, suggesting ongoing limitations in the current administration's policies.
- **Challenges Posed by the Jones Act**: The **Jones Act** is discussed as a long-standing regulatory barrier contributing to US defense manufacturing inefficiencies.
   - This law is creating a non-competitive bubble in US domestic shipping, adversely affecting shipbuilders' international competitiveness.



**Link mentioned**: <a href="https://x.com/rwang07/status/1883210410763121073">Tweet from Ray Wang (@rwang07)</a>: China&#39;s New AI Industry Development Action Plan (中国银行支持人工智能产业链发展行动方案) Will Provide 1 trillion yuan ($ 137 billion) to support its AI industry over the next five years 🇺🇸🇨🇳This might be the mos...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1332440515840446465)** (233 messages🔥🔥): 

> `DeepSeek R1 developments, Qwen2.5-VL release, Operator functionality, Prompt engineering tools, Reasoning models applications` 


- **DeepSeek R1's Training Cost Reference**: A member clarified that the $5M training cost reference pertains to DeepSeek V3 and can be found in the project's report.
   - An image attachment provided was referenced as confirmation for this information.
- **Launch of Qwen2.5-VL Model**: Alibaba announced the launch of Qwen2.5-VL, a multimodal model capable of generating images and performing intelligent tasks.
   - The model claims superiority over DALL-E 3 and Stable Diffusion on several benchmarks, emphasizing its visual understanding and localization capabilities.
- **Insights on Operator Functionality**: Users discussed the capabilities of Operator in coding environments, with an emphasis on its effectiveness in generating initial codebases.
   - Challenges were noted regarding the handling of complex websites and video sampling rates, highlighting a need for improvement.
- **Prompt Engineering Tools and Experiences**: Members shared their experiences with various prompt engineering tools such as Braintrust and Humanloop, discussing their usability and features.
   - Braintrust was noted as favorable by one user for its functionality, while concerns about the UX and pricing transparency of Humanloop were mentioned.
- **Research on Reasoning Models and Their Applications**: Participants considered the potential applications of reasoning models like R1, including improved coding and agentic capabilities.
   - Conversations included interest in the notion that reasoning models could significantly enhance task execution beyond traditional chat functionalities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://]">no title found</a>: no description found</li><li><a href="https://x.com/NousResearch/status/1883912370696704011">Tweet from Nous Research (@NousResearch)</a>: Recent AI breakthroughs challenge the status quo narrative that only closed, mega labs have the ability to push the frontier of superintelligence.Today we announce Nous Psyche built on @Solana - a coo...</li><li><a href="https://x.com/huybery/status/1883775353950519479">Tweet from Binyuan Hui (@huybery)</a>: There are some surprises tonight</li><li><a href="https://x.com/lmarena_ai/status/1882875989610594542">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: ❤️‍🔥WebDev Arena Update: Exciting new entries!- #2: @deepseek_ai DeepSeek-R1- #4: New Gemini-2.0-Flash-ThinkingDeepSeek-R1 jumps to #2 with only &lt;40 pts gap to Claude 3.5 Sonnet, showing strong ca...</li><li><a href="https://x.com/LiangWenfeng_">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/Alibaba_Qwen/status/1883954247743725963">Tweet from Qwen (@Alibaba_Qwen)</a>: 🎉 恭喜发财🧧🐍 As we welcome the Chinese New Year, we&#39;re thrilled to announce the launch of Qwen2.5-VL , our latest flagship vision-language model! 🚀💗 Qwen Chat: https://chat.qwenlm.ai📖 Blog: http...</li><li><a href="https://app.discuna.com/invite/ai_engineer">Discuna</a>: no description found</li><li><a href="https://x.com/alecm3/status/1883147247485170072?t=55xwg97roj74RglY2Dil_g&s=19">Tweet from Alec (@alecm3)</a>: Deepseek is deepshit</li><li><a href="https://x.com/klazuka/status/1883880742322888903.">Tweet from Keith Lazuka (@klazuka)</a>: Here&#39;s a video of Operator creating the new Python web app in Github. It worked surprisingly well. https://operator.chatgpt.com/v/67978eebb89c81909ed9a584d7fce506And here&#39;s the repository and ...</li><li><a href="https://stackoverflow.com/questions/77628629/is-it-possible-to-use-macos-accessibility-api-features-from-a-cli-or-library">Is it possible to use macOS Accessibility API features from a CLI or library?</a>: I am working on an application that needs to leverages macOS Accessibility APIs to read the selected text in any application. I will call a Swift library from Rust via FFI.&#xA;I am able to get a POC ...</li><li><a href="https://x.com/thankscline/status/1882878536450814263?s=46">Tweet from Cline (@thankscline)</a>: While everyone&#39;s fighting about Deepseek R1 vs o1 benchmarks, something fascinating has happened in the Cline community:Developers organically started using:- DeepSeek R1 ($0.55/M) for planning ph...</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: no description found</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet set SOTA on aider’s polyglot benchmark</a>: R1+Sonnet has set a new SOTA on the aider polyglot benchmark. At 14X less cost compared to o1.</li><li><a href="https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1?r=f2tys&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false">The Illustrated DeepSeek-R1</a>: A recipe for reasoning LLMs</li><li><a href="https://x.com/alecm3/status/1883147247485170072?t=55xwg97roj74RglY2">Tweet from Alec (@alecm3)</a>: Deepseek is deepshit</li><li><a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Qwen2.5-VL - a Qwen Collection</a>: no description found</li><li><a href="https://x.com/_philschmid/status/1883055262669349287?s=46">Tweet from Philipp Schmid (@_philschmid)</a>: Function Calling is not solved yet ‼️ A new benchmark shows that LLMs struggle with multi-step, constrained function calls. ComplexFuncBench is designed to test complex function calling evaluation wit...</li><li><a href="https://x.com/LiangWenfeng_/status/1883953499068887189">Tweet from Liang Wenfeng 梁文锋 (@LiangWenfeng_)</a>: Deepseek 2025/25/02 ⏳🐋</li><li><a href="https://steve-yegge.medium.com/the-death-of-the-stubborn-developer-b5e8f78d326b">The Death of the Stubborn Developer</a>: I wrote a blog post back in May called The Death of the Junior Developer. It made people mad. My thesis has since been corroborated by a…</li><li><a href="https://x.com/rajammanabrolu/status/1883583493290238106?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Prithviraj (Raj) Ammanabrolu (@rajammanabrolu)</a>: Simply, no.I&#39;ve been looking at my old results from doing RL with &#34;verifiable&#34; rewards (math puzzle games, python code to pass unit tests) starting from 2019 with GPT-1/2 to 2024 with Qwen...</li><li><a href="https://x.com/alibaba_qwen/status/1883557964759654608?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">Tweet from Qwen (@Alibaba_Qwen)</a>: We&#39;re leveling up the game with our latest open-source models, Qwen2.5-1M ! 💥 Now supporting a 1 MILLION TOKEN CONTEXT LENGTH 🔥Here&#39;s what’s new:1️⃣ Open Models: Meet Qwen2.5-7B-Instruct-1M ...</li><li><a href="https://x.com/jiayi_pirate/status/1882839370505621655">Tweet from Jiayi Pan (@jiayi_pirate)</a>: We reproduced DeepSeek R1-Zero in the CountDown game, and it just works Through RL, the 3B base LM develops self-verification and search abilities all on its own You can experience the Ahah moment you...</li><li><a href="https://x.com/giffmana/status/1883432865293049954">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: You know, last ICML (Vienna, July&#39;24) I sat in the talk that explains EU&#39;s work in AI and attempt to come up with a plan.There was a call for expert researcher&#39;s help/interest. I registere...</li><li><a href="https://x.com/Yuchenj_UW/status/1883391135441371223">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: @pmarca</li><li><a href="https://x.com/LiangWenfeng_/status/1883918900741763293">Tweet from Liang Wenfeng 梁文锋 (@LiangWenfeng_)</a>: 🚨DeepSeek just dropped ANOTHER open-source AI model, Janus-Pro-7B.It&#39;s multimodal (can generate images) and beats OpenAI&#39;s DALL-E 3 and Stable Diffusion across GenEval and DPG-Bench benchmark...</li><li><a href="https://x.com/hamptonism/status/1883147826571706735">Tweet from ₕₐₘₚₜₒₙ — e/acc (@hamptonism)</a>: Bank of China plans on ¥1 Trillion investment in Ai industry.</li><li><a href="https://x.com/LiangWenfeng_/status/1883874025350508861">Tweet from Liang Wenfeng 梁文锋 (@LiangWenfeng_)</a>: Deepseek new release soon</li><li><a href="https://x.com/rwang07/status/1883210410763121073?s=46">Tweet from Ray Wang (@rwang07)</a>: China&#39;s New AI Industry Development Action Plan (中国银行支持人工智能产业链发展行动方案) Will Provide 1 trillion yuan ($ 137 billion) to support its AI industry over the next five years 🇺🇸🇨🇳This might be the mos...</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF">bartowski/Qwen2.5-7B-Instruct-1M-GGUF · Hugging Face</a>: no description found</li><li><a href="https://x.com/teortaxesTex/status/1883605616742351013">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: tfw an HFT guy says GRPO is based on the Sharpe ratio, which is used to measure risk-adjusted returns in investmentwhales are quants, after all. Everything is a ROI problem if you&#39;re cracked enoug...</li><li><a href="https://x.com/vllm_project/status/1883966341557936514">Tweet from vLLM (@vllm_project)</a>: 🚀 With the v0.7.0 release today, we are excited to announce the alpha release of vLLM V1: A major architectural upgrade with 1.7x speedup! Clean code, optimized execution loop, zero-overhead prefix c...</li><li><a href="https://x.com/teortaxesTex/status/1883926389306671376">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: This is fake and must be reported into oblivionMan, Americans really can&#39;t take competition well</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ibbloy/158bit_deepseek_r1_131gb_dynamic_gguf/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.promptlayer.com/">PromptLayer - The cleanest way to prompt engineer. Platform for prompt management, prompt evaluations, and LLM observability</a>: no description found</li><li><a href="https://x.com/teknium1/status/1882893748742598669?s=46">Tweet from Teknium (e/λ) (@Teknium1)</a>: We retrained hermes with 5k deepseek r1 distilled cots. I can confirm a few things:1. You can have a generalist + reasoning mode, we labeled all longCoT samples from r1 with a static systeem prompt, t...</li><li><a href="https://www.youtube.com/watch?v=X5adgxV0gBE">DeepSeek R1 - The Chinese AI &quot;Side Project&quot; That Shocked the Entire Industry!</a>: Join My Newsletter for Regular AI Updates 👇🏼https://forwardfuture.aiMy Links 🔗👉🏻 Subscribe: https://www.youtube.com/@matthew_berman👉🏻 Twitter: https:/...</li><li><a href="https://stratechery.com/2025/deepseek-faq/">DeepSeek FAQ</a>: DeepSeek has completely upended people&#8217;s expectations for AI and competition with China. What is it, and why does it matter?</li><li><a href="https://www.youtube.com/watch?v=jrf76uNs77k&t=868s">The Unreasonable Effectiveness of Reasoning Distillation: using DeepSeek R1 to beat OpenAI o1</a>: https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillationWe trained Bespoke-Stratos-32B, our reasoning model d...</li><li><a href="https://github.com/madsys-dev/deepseekv2-profile/blob/924174cb5dc11fad24bdaad3fd820ebf87506368/workspace/blog/optimizing-mla.md">deepseekv2-profile/workspace/blog/optimizing-mla.md at 924174cb5dc11fad24bdaad3fd820ebf87506368 · madsys-dev/deepseekv2-profile</a>: Contribute to madsys-dev/deepseekv2-profile development by creating an account on GitHub.</li><li><a href="https://youtu.be/bJzj5lTiqe0?si=n73aW2Zm8U3qIjKO">Deepseek R1: How China’s open source AI model beats OpenAI at 3% of the cost</a>: DeepSeek-R1: The Chinese Open Source AI Disrupting OpenAI&#39;s LeadershipIn this episode, Sam and Matt discuss the recent breakthroughs by DeepSeek&#39;s R1 model, ...</li><li><a href="https://youtu.be/HM92mmG6YTs?feature=shared">DeepSeek R1 vs o1: AI EXPLAINS Autonomy of Experts (a better MoE)</a>: Performance comparison of OpenAI o1 (old, proprietary) vs DeepSeek R1 (new, open-source). Task for both LLMs is to explain new AI paper on Autonomy of Expert...</li><li><a href="https://x.com/pmarca/status/1882903903777558677">Tweet from Marc Andreessen 🇺🇸 (@pmarca)</a>: Absolutely not.Quoting TFTC (@TFTC21) Sam Altman: Advancing AI may require &#34;changes to the social contract.&#34;&#34;The entire structure of society will be up for debate and reconfiguration.&#34;</li><li><a href="https://www.bankofchina.com/aboutboc/bi1/202501/t20250123_25254674.html">1万亿元！提供专项综合金融支持 助力人工智能产业链发展</a>: no description found</li><li><a href="https://github.com/glut23/webvtt-py">GitHub - glut23/webvtt-py: Read, write, convert and segment WebVTT caption files in Python.</a>: Read, write, convert and segment WebVTT caption files in Python. - glut23/webvtt-py</li><li><a href="https://cameronrwolfe.substack.com/p/moe-llms?utm_source=post-email-title&publication_id=1092659&post_id=154340424&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email)">Mixture-of-Experts (MoE) LLMs</a>: Understanding models like DeepSeek, Grok, and Mixtral from the ground up...</li><li><a href="https://youtubetranscriptoptimizer.com/blog/05_the_short_case_for_nvda">The Short Case for Nvidia Stock</a>: All the reasons why Nvidia will have a very hard time living up to the currently lofty expectations of the market.</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>: Janus-Series: Unified Multimodal Understanding and Generation Models - deepseek-ai/Janus</li><li><a href="https://www.bankofchina.com/aboutboc/bi1/202501/t20250123_25254674.h">中国银行全球门户网站-提示信息</a>: no description found</li><li><a href="https://cameronrwolfe.substack.com/p/moe-llms?utm_sou">Mixture-of-Experts (MoE) LLMs</a>: Understanding models like DeepSeek, Grok, and Mixtral from the ground up...</li><li><a href="https://www.stepfun.com">阶跃星辰</a>: no description found</li><li><a href="https://buttondown.com/ainews/archive/ainews-tinyzero-reproduce-deepseek-r1-zero-for-30/">[AINews] TinyZero: Reproduce DeepSeek R1-Zero for $30</a>: RL is all you need. AI News for 1/23/2025-1/24/2025. We checked 7 subreddits, 433 Twitters and 34 Discords (225 channels, and 3926 messages) for you....
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new pod! https://x.com/latentspacepod/status/1883354909367787565
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1332454948553887831)** (193 messages🔥🔥): 

> `Model Context Protocol (MCP), MCP tools integration, Transcription and documentation, Obsidian integration, Server capabilities and implementations` 


- **Excitement for MCP and its potential**: Members expressed enthusiasm for the **Model Context Protocol (MCP)**, describing it as a pivotal point for integrating AI capabilities across applications and tools.
   - Discussions highlighted how MCP could serve as a central hub for various applications, encouraging exploration of its capabilities in real-world scenarios.
- **Integration with existing tools and libraries**: Participants discussed the integration of MCP with various tools like **Cursor** and **Cline**, showing interest in how these could enhance functionality and streamline workflows.
   - The potential to connect MCP with **Obsidian** for documentation and transcription was also mentioned, emphasizing the collaborative nature of the community.
- **Exploration of different languages for MCP servers**: The flexibility of using various programming languages for creating MCP servers was a key point, with suggestions to use **Go**, **Rust**, and even **assembly** for optimal performance.
   - Members noted that this independence from language allows developers to focus on performance and security needs for their specific implementations.
- **Future discussions and collaborations on MCP**: A plan for an **MCP party** to discuss lessons learned and experiences was mentioned, reflecting a collaborative approach to sharing knowledge.
   - Members were encouraged to participate and contribute to ongoing developments, indicating active engagement in advancing MCP tools and capabilities.
- **Documentation and tutorials**: There was mention of comprehensive documentation available for users interested in implementing MCP, including a detailed section on best practices.
   - Participants expressed interest in reviewing the latest README and tutorials to better understand how to effectively utilize MCP in their projects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cs16.samke.me/">cs16.css</a>: CSS library based on Counter Strike 1.6 UI.</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/architecture/#capability-negotiation">Architecture</a>: no description found</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/architecture/#capability">Architecture</a>: no description found</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/utilities/cancellation/">Cancellation</a>:           ℹ️                  Protocol Revision: 2024-11-05      The Model Context Protocol (MCP) supports optional cancellation of in-progress requeststhrough notification messages. Either side can s...</li><li><a href="https://github.com/tumf/mcp-shell-server">GitHub - tumf/mcp-shell-server</a>: Contribute to tumf/mcp-shell-server development by creating an account on GitHub.</li><li><a href="https://github.com/rusiaaman/wcgw">GitHub - rusiaaman/wcgw: Shell and coding agent on claude desktop app</a>: Shell and coding agent on claude desktop app. Contribute to rusiaaman/wcgw development by creating an account on GitHub.</li><li><a href="https://github.com/MarkusPfundstein/mcp-obsidian">GitHub - MarkusPfundstein/mcp-obsidian: MCP server that interacts with Obsidian via the Obsidian rest API community plugin</a>: MCP server that interacts with Obsidian via the Obsidian rest API community plugin - MarkusPfundstein/mcp-obsidian</li><li><a href="https://github.com/go-go-golems">GO GO GOLEMS!</a>: GO GO GOLEMS BUILD GO GO GADGETS. GO GO GOLEMS! has 34 repositories available. Follow their code on GitHub.</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://github.com/rusiaaman/wcgw/blob/fbe8c5c3cca4f7a149f8c099c63696d9ede7f9e7/src/wcgw/client/mcp_server/server.py#L129-L138">wcgw/src/wcgw/client/mcp_server/server.py at fbe8c5c3cca4f7a149f8c099c63696d9ede7f9e7 · rusiaaman/wcgw</a>: Shell and coding agent on claude desktop app. Contribute to rusiaaman/wcgw development by creating an account on GitHub.</li><li><a href="https://github.com/go-go-golems/go-go-mcp">GitHub - go-go-golems/go-go-mcp: Anthropic MCP go implementation</a>: Anthropic MCP go implementation. Contribute to go-go-golems/go-go-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/calclavia/mcp-obsidian">GitHub - smithery-ai/mcp-obsidian: A connector for Claude Desktop to read and search an Obsidian vault.</a>: A connector for Claude Desktop to read and search an Obsidian vault. - smithery-ai/mcp-obsidian
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1332461827061321728)** (97 messages🔥🔥): 

> `Layer Convergence Bias, Causally Regularized Tokenization, DeepSeek Model Discussion, R1 Training Costs, GRPO Implementation Challenges` 


- **Layer Convergence Bias Observations**: Research shows that shallower layers in **Deep Neural Networks** converge faster than deeper layers, referred to as **Layer Convergence Bias**. This phenomenon is attributed to flatter local minima in shallow layers leading to more stable gradients.
   - For more details, check the [ICLR 2023 paper](https://openreview.net/forum?id=wlMDF1jQF86) published on **February 1, 2023**.
- **Causally Regularized Tokenization Insights**: The latest work by **Armen Agha and team** reveals that image tokenizers optimized for reconstruction can hinder **downstream autoregressive model performance**. Their new method, **Causally Regularized Tokenization**, shows significant improvements in efficiency and quality.
   - Further details can be found in the [published paper](https://arxiv.org/pdf/2412.16326) which compares the performance of **LlamaGen-3B**.
- **Questions Surrounding DeepSeek Models**: Discussions highlight skepticism about claims that **DeepSeek** has developed a significantly cheaper chip than **NVIDIA** while being more effective in model performance. Participants noted the lack of transparency regarding model building materials or compute.
   - Concerns were raised about speculative price estimates and a perceived lack of essential open source information.
- **Estimating R1 Training Costs**: Participants suggest that estimating the **R1 training costs** based on dataset size and tokens can lead to a rough approximation, querying if 800k sample sizes are mentioned in the R1 paper. There is general agreement that the cost could be significantly lower than previous iterations.
   - Calculations indicate a potential cost of about **$1.6M** for inference, with discussions hinting that **R1** could have been more cost-effective than **V3**.
- **GRPO Implementation Gaps**: Despite multiple repositories claiming to implement **GRPO**, there seems to be a lack of practical application for achieving runs like **R1**. Participants expressed frustration that both **TinyZero** and **SimpleRL** do not utilize GRPO effectively in their recreation runs.
   - It appears that **PPO** is predominantly used, indicating a gap in fully exploring the potential of GRPO.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ArmenAgha/status/1882897021">Tweet from mhase (@mloge)</a>: ┣¨ｽﾄｴﾌｽｷｰ</li><li><a href="https://openreview.net/forum?id=wlMDF1jQF86">Which Layer is Learning Faster? A Systematic Exploration of...</a>: We empirically show that the shallower layers converge faster than the deeper layers in neural networks, and provide the theoretical justification and practical value of this finding.</li><li><a href="https://x.com/ArmenAgha/status/1882897021667090797">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: Officially, the last paper I was on during my tenure at FAIR! Our paper reveals that image tokenizers optimized solely for reconstruction hurt downstream autoregressive model performance, challenging ...</li><li><a href="https://x.com/TheXeophon/status/1883933054366015545">Tweet from Xeophon (@TheXeophon)</a>: what the fuck</li><li><a href="https://en.wikipedia.org/wiki/Taylor_Swift%E2%80%93Ticketmaster_controversy">Taylor Swift–Ticketmaster controversy - Wikipedia</a>: no description found</li><li><a href="https://github.com/Jiayi-Pan/TinyZero">GitHub - Jiayi-Pan/TinyZero: Clean, accessible reproduction of DeepSeek R1-Zero</a>: Clean, accessible reproduction of DeepSeek R1-Zero - Jiayi-Pan/TinyZero</li><li><a href="https://github.com/hkust-nlp/simpleRL-reason">GitHub - hkust-nlp/simpleRL-reason: This is a replicate of DeepSeek-R1-Zero and DeepSeek-R1 training on small models with limited data</a>: This is a replicate of DeepSeek-R1-Zero and DeepSeek-R1 training on small models with limited data - hkust-nlp/simpleRL-reason</li><li><a href="https://en.m.wikipedia.org/wiki/List_of_common_misconceptions">List of common misconceptions - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1332471378347167855)** (311 messages🔥🔥): 

> `GRPO Implementation Details, AlphaZero Evolution, Empowerment in AI, Reinforcement Learning Challenges, Experience Replay in RL` 


- **Understanding GRPO Group Size and Batch Size**: Discussion revolved around the GRPO implementation, with clarification that a batch size of 1024 may consist of multiple samples, each being a group of 64. Participants noted the importance of considering how the sequence length affects the training process and gradient calculation.
   - There was a consensus that experiencing replay buffers may not be necessary for GRPO, as the algorithm can function with the aggregation of gradients from groups without retaining earlier samples.
- **AlphaZero's Development Process**: Participants reflected on the evolution of AlphaZero, noting it streamlined previous methods and incorporated learnings from past versions. They highlighted the significant engineering challenges faced and the rationale for not jumping directly to AlphaZero methodologies in practice.
   - Bearcat9705 pointed out how subsequent papers made incremental improvements over previous iterations while also attending to the level of simplicity in approach.
- **Curiosity and Empowerment in AI**: The concept of empowerment in AI was discussed, focusing on its relationship to intrinsic motivation and maximizing future options. Synquid shared insights from personal research work, emphasizing its theoretical foundations and potential applications.
   - Participants expressed interest in the relationship between curiosity-driven models and current language learning methods, suggesting a revival in exploring these concepts.
- **Challenges of Implementing Reinforcement Learning**: A dialogue emerged about the difficulty of rewarding vague outcomes in reinforcement learning compared to clear tasks, specifically within the context of training large language models. The consensus was that a focused reward structure could yield better learning outcomes than traditional methods.
   - Fessus emphasized that a simple success or failure token can guide learning effectively without the complexities of traditional reinforcement learning methods.
- **Experience Replay Strategies in RL**: The conversation highlighted the diminishing relevance of traditional experience replay in modern reinforcement learning practices. Participants agreed that retaining all previous samples is no longer common, and questioned the necessity of maintaining a replay buffer in current implementations.
   - They debated the advantages of collecting batches without a replay buffer, acknowledging that even a small collection for aggregating traces might not offer significant benefits.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1509.08731">Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning</a>: The mutual information is a core statistical quantity that has applications in all areas of machine learning, whether this is in training of density models over multiple data modalities, in maximising...</li><li><a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>: Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Co...</li><li><a href="https://philippe-eecs.github.io/vitok/">ViTok</a>: Learnings from Scaling Visual Tokenizers for Reconstruction and Generation</li><li><a href="https://arxiv.org/abs/2501.13926">Can We Generate Images with CoT? Let&#39;s Verify and Reinforce Image Generation Step by Step</a>: Chain-of-Thought (CoT) reasoning has been extensively explored in large models to tackle complex understanding tasks. However, it still remains an open question whether such strategies can be applied ...</li><li><a href="https://arxiv.org/abs/2405.17399">Transformers Can Do Arithmetic with the Right Embeddings</a>: The poor performance of transformers on arithmetic tasks seems to stem in large part from their inability to keep track of the exact position of each digit inside of a large span of digits. We mend th...</li><li><a href="https://arxiv.org/abs/2501.11651">Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling</a>: Large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks. However, existing approaches mainly rely on imitation learning and struggle to achieve effective test...</li><li><a href="https://arxiv.org/abs/2410.14606">Streaming Deep Reinforcement Learning Finally Works</a>: Natural intelligence processes experience as a continuous stream, sensing, acting, and learning moment-by-moment in real time. Streaming learning, the modus operandi of classic reinforcement learning ...</li><li><a href="https://arxiv.org/abs/2501.11425">Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training</a>: Large Language Models (LLMs) agents are increasingly pivotal for addressing complex tasks in interactive environments. Existing work mainly focuses on enhancing performance through behavior cloning fr...</li><li><a href="https://x.com/its_dibya/status/1883595705736163727">Tweet from Dibya Ghosh (@its_dibya)</a>: With R1, a lot of people have been asking “how come we didn&#39;t discover this 2 years ago?”Well... 2 years ago, I spent 6 months working exactly on this (PG / PPO for math+gsm8k), but my results wer...</li><li><a href="https://arxiv.org/abs/2301.07969">Fast Inference in Denoising Diffusion Models via MMD Finetuning</a>: Denoising Diffusion Models (DDMs) have become a popular tool for generating high-quality samples from complex data distributions. These models are able to capture sophisticated patterns and structures...</li><li><a href="https://x.com/RamanujanVivek/status/1882882551670555095">Tweet from Vivek Ramanujan (@RamanujanVivek)</a>: Happy to (belatedly) share our recent work introducing Causally Regularized Tokenization 📺, matching LlamaGen-3B generation performance with 0.5x the number of tokens/image (256 vs 576) and 0.25x the...</li><li><a href="https://x.com/leloykun/status/1883561892926677029">Tweet from leloy! (@leloykun)</a>: (Linear) Attention Mechanisms as Test-Time RegressionBy now, you&#39;ve probably already heard of linear attention, in-context learning, test-time scaling, etc...Here, I&#39;ll discuss:1. The unifying...</li><li><a href="https://en.wikipedia.org/wiki/Empowerment_(artificial_intelligence)">Empowerment (artificial intelligence) - Wikipedia</a>: no description found</li><li><a href="https://github.com/TencentARC/SEED-Voken">GitHub - TencentARC/SEED-Voken: SEED-Voken: A Series of Powerful Visual Tokenizers</a>: SEED-Voken: A Series of Powerful Visual Tokenizers - TencentARC/SEED-Voken</li><li><a href="https://x.com/bycloudai/status/1880106360731496661">Tweet from bycloud (@bycloudai)</a>: someone has finally done it test time compute + diffusion modelsa really interesting one for sure 🧵</li><li><a href="https://arxiv.org/abs/2501.09732">Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps</a>: Generative models have made significant impacts across various domains, largely due to their ability to scale during training by increasing data, computational resources, and model size, a phenomenon ...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1333412563219906612)** (2 messages): 

> `Chinchilla library, LLM scaling laws, 20-tokens-per-parameter heuristic` 


- **Chinchilla Library's New LLM Analysis**: A member added an analysis of LLM scaling law to their [Chinchilla library](https://github.com/kyo-takano/chinchilla/blob/master/examples/llm/main.ipynb), highlighting its toolkit for scaling law research.
   - They noted the significant finding that the **20-tokens-per-parameter** heuristic works nearly as well as the fully-optimized Chinchilla model when evaluated by it.
- **Interesting Findings on the 20-Token Heuristic**: The member suggested that the 20-token heuristic's effectiveness isn't solely due to the number but rather the flat minima in the ratio as compute increases.
   - They observed a visual confirmation of this flatness with higher compute, illustrating a fundamental behavior in the scaling law.
- **Questioning the Nature of Scaling Effects**: Another member speculated whether the observed phenomena were simply due to scaling, proposing that *every smooth curve is flat if you zoom in enough*.
   - This inquiry opens up discussions on the inherent nature of scaling laws and their implications in the analysis.



**Link mentioned**: <a href="https://github.com/kyo-takano/chinchilla/blob/master/examples/llm/main.ipynb">chinchilla/examples/llm/main.ipynb at master · kyo-takano/chinchilla</a>: A toolkit for scaling law research ⚖. Contribute to kyo-takano/chinchilla development by creating an account on GitHub.

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1332540720120070207)** (3 messages): 

> `Verified Reasoning in Training, Mechanisms of Model Learning, Interpretability in Fine-Tuning, Insights from Model Weights` 


- **Verified reasoning is the new meta**: *If verified reasoning during training is the new meta*, members discussed what mechanistic interpreters should prioritize in their analyses to adapt.
   - The conversation hinted at the potential shift in focus towards understanding reasoning capabilities among models.
- **Understanding LLMs during fine-tuning**: One member emphasized the importance of *understanding how and what LLMs learn during fine-tuning*, especially regarding input-output pairs.
   - This could potentially clarify what is captured in model weights and representations related to reasoning abilities.
- **Lack of interpretability in learning mechanisms**: Members expressed a consensus that *interpretability is lacking* on how models learn when fine-tuned for specific objectives.
   - They are concerned not just with overall performance, but also with factors affecting the success and generalization of fine-tuning.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1333541169048850454)** (4 messages): 

> `scbench, zeroSCROLLS, longbench` 


- **Evaluating scbench's Integration Challenges**: A member noted that integrating **scbench** would be tricky as it requires **multi-turn** capabilities, suggesting a need for further investigation.
   - This indicates a complexity in implementation that could impact future development timelines.
- **Interest in zeroSCROLLS**: Another member expressed excitement about exploring **zeroSCROLLS**, indicating positive momentum for this option.
   - They seem optimistic about its potential benefits, although specifics were not discussed.
- **Addition of longbench**: A member confirmed that they have also added **longbench**, indicating progress in feature expansion.
   - This addition could complement existing tools and offers a broader set of functionalities moving forward.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1333276412966080542)** (2 messages): 

> `Multimodal Channel Guidelines, Community Project Collaboration` 


- **Multimodal Channel Misunderstanding**: A member pointed out that certain posts do not belong in the multimodal channel, emphasizing it is not relevant to the channel's purpose.
   - *Please make a community project if you're looking for collaboration* was suggested as an alternative approach.
- **Nuanced Multimodal Model Discussion**: Another member acknowledged the confusion, agreeing to remove the post while noting that the model has nuanced multimodal capabilities.
   - They expressed that the topic *probably warrants something other than a post here*, indicating a need for clearer community guidelines.


  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1333509080513253427)** (1 messages): 

> `System prompts customization, Optimizing Bolt's behavior` 


- **Customize Your System Prompt**: You can now set up a **system prompt** both per project and globally, allowing for tailored experiences in [Bolt](https://x.com/boltdotnew/status/1883949779572646008).
   - This feature, highly requested by users, enables you to include your **favorite libraries** and techniques, ensuring Bolt behaves according to your workflow preferences.
- **Share Tips on Usage**: Users are encouraged to share their best tips on how to effectively use the new system prompt customization feature.
   - *How will you optimize your Bolt experience?* Dive into discussions below!



**Link mentioned**: <a href="https://x.com/boltdotnew/status/1883949779572646008">Tweet from bolt.new (@boltdotnew)</a>: You can now set up a system prompt, per project and globally!💡Put your favorite libs & techniques there so Bolt always uses them.This heavily requested feature allows you to optimize Bolt&#39;s behav...

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1332650967874670663)** (7 messages): 

> `Project Structuring Challenges, Component Splitting Strategy, Utilizing Guidelines for Stability, Learning from Past Projects, Tracking Project Changes` 


- **Project Structuring Challenges**: Members discussed how structured prompts and planning can stifle creativity in projects using Bolt, emphasizing the need for flexibility.
   - A member noted that attempting to eliminate problems upfront resulted in cycles of starting over without clear guidelines.
- **Component Splitting Strategy**: A member shared insights on difficulties faced when dividing complex components, with context limitations causing issues during the splitting process.
   - They suggested a systematic approach that includes detailed code reviews and structured migration steps documented in a NEXTSTEPS.md file.
- **Utilizing Guidelines for Stability**: Implementing a rigorous adherence to GUIDELINES.md helped stabilize project development, ensuring components are built sequentially and systematically.
   - With both the GUIDELINES and NEXT STEPS documents, the context window was managed effectively to avoid forgetting critical information.
- **Learning from Past Projects**: Members reflected on their learning process, identifying the pitfalls of earlier projects, such as introducing Supabase too early without clear guidelines.
   - They underscored the importance of defining a foundational design system to prevent inconsistencies and facilitate smoother project progression.
- **Tracking Project Changes**: A detailed tracking system was discussed which includes logs and changelogs to monitor project developments and possible rollbacks.
   - One member shared a link to a project structure that focuses on key directories for project management, despite some limitations in the deployed version.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://heroic-bombolone-70b361.netlify.app">Vite + React + TS</a>: no description found</li><li><a href="https://x.com/KevinNaughtonJr/status/1882833510957985819">Tweet from Kevin Naughton Jr. (@KevinNaughtonJr)</a>: software engineering might be the only profession where you can be stuck on a task for days/weeks/months and no one will even bat an eye, question your abilities, or be upset with you
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1332440177074896978)** (304 messages🔥🔥): 

> `Error Handling in Bolt, Billing and Token Limits, Implementing User Roles, Deployment with Netlify, Connecting GitHub to Bolt` 


- **Handling Errors in Bolt**: Users reported frequent errors and issues with Bolt, including rate limits and network errors, leading to frustration after consuming large amounts of tokens.
   - Many have resorted to asking for help with migration issues or seeking professional assistance to resolve their problems.
- **Billing and Token Consumption**: Concerns were raised about the high consumption of tokens when using Bolt, with some users claiming to have spent millions on prompts with little progress.
   - Users discussed refund possibilities after facing issues and highlighted the disparity in costs versus achieved outcomes.
- **Implementing User Roles**: One user successfully created an app with multiple login roles including super admin and admin, overcoming complexities with Supabase policies.
   - The implementation process was noted as challenging due to policies creating recursion problems, but ultimately achieved a fully functional system.
- **Deployment with Netlify**: Users inquired about connecting their Bolt projects to custom domains via Netlify, clarifying that redeployment is necessary for updates to take effect.
   - It was emphasized that changes made in Bolt do not automatically reflect on Netlify.
- **Connecting GitHub to Bolt**: A user sought help on importing existing GitHub repositories into Bolt but encountered access issues with private repositories.
   - Currently, users are unable to access private repos within Bolt, which is a limitation being addressed.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/364486390102097930/1332441767861157969">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://diji.art">Diji.art - Digital Design Marketplace</a>: Create and sell unique designs on high-quality apparel. Join our community of creators and fashion enthusiasts.</li><li><a href="https://diji.art/designs">Diji.art - Digital Design Marketplace</a>: Create and sell unique designs on high-quality apparel. Join our community of creators and fashion enthusiasts.</li><li><a href="https://www.anthropic.com/pricing#anthropic-api">Pricing</a>: Anthropic is an AI safety and research company that&#x27;s working to build reliable, interpretable, and steerable AI systems.</li><li><a href="https://discordscrapper.netlify.app/">Vite + React + TS</a>: no description found</li><li><a href="https://repocloud.io/boltdiy">RepoCloud | Bolt.diy: Choose Your AI Model</a>: Discover Bolt.diy, the ultimate fork for selecting your favorite AI model. Customize your coding experience with top LLMs like OpenAI and Anthropic!</li><li><a href="https://thinktank.ottomator.ai/">oTTomator Community</a>: Where innovators and experts unite to advance the future of AI-driven automation</li><li><a href="https://www.youtube.com/watch?v=jkfVvWndbeE">Bolt.new Developer&#39;s Guide to Effortless API Integration and Stop All CORS Errors</a>: Struggling with API integration or tackling those pesky CORS errors? Watch me break it all down in this ultimate Bolt developer&#39;s guide to effortless API int...</li><li><a href="https://docs.github.com/rest/git/blobs#create-a-blob">REST API endpoints for Git blobs - GitHub Docs</a>: no description found
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1332473138146967632)** (233 messages🔥🔥): 

> `MCP Client Issues, Server Configurations, Voice Chat Integrations, Open Source Tooling, Kubernetes Integrations` 


- **Challenges with MCP Clients**: Users discussed various issues with MCP clients, particularly regarding the inability to update tools dynamically without restarting the client. The integration of voice chat functionality also remained a significant pain point for developers.
   - Many users expressed a need for clearer documentation and better integration capabilities within these tools.
- **Server Configuration Concerns**: There was ongoing discussion about configuration settings for MCP servers, including `disabled` and `autoApprove` options. Users noted complexities arising from different setups and the implications of using organization accounts.
   - The need for functionality that allows better server management and the ability to work without a reliance on proprietary APIs was emphasized.
- **Integrating Tools for Multi-Device Usage**: Ideas were exchanged regarding daisy-chaining tools to create a master server for handling functions across multiple devices. Participants highlighted the need for effective communication between a central controller and individual clients to manage server updates.
   - There was particular interest in how Kubernetes could be leveraged for running MCP servers efficiently.
- **Open Source and Community Efforts**: Discussion highlighted the role of open source projects in enabling community-driven development, particularly in MCP tooling. Users encouraged contributions and making tools ready for broader adoption, despite some rough edges.
   - The conversation touched on various open-source clients and the advantages of having publicly accessible source code in ensuring transparency and collaboration.
- **User Opinions on API Management**: Participants shared experiences with server API management, particularly the limitations they faced regarding request timeouts and handling various API configurations. Users sought clarity on whether MCP tools would eventually integrate with more mainstream APIs for improved functionality.
   - Concerns about the additional costs of cloud services compared to alternative solutions were also voiced.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.systemprompt.io`">no title found</a>: no description found</li><li><a href="https://tenor.com/view/magic-gif-26166638">Magic GIF - Magic - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://glama.ai/mcp/servers/2au072rrbc">mcp-jdbc</a>: MCP to access any database accessible via JDBC such as Postgres, Oracle, mysql, mariadb, sqlite etc.</li><li><a href="https://github.com/Ejb503/multimodal-mcp-client/blob/master/README.md">multimodal-mcp-client/README.md at master · Ejb503/multimodal-mcp-client</a>: A Multi-modal MCP client for voice powered agentic workflows - Ejb503/multimodal-mcp-client</li><li><a href="https://boards.greenhouse.io/anthropic/jobs/4495047008">Software Engineer, Model Context Protocol</a>: London, UK</li><li><a href="https://github.com/cookiecad/mcp-runner">GitHub - cookiecad/mcp-runner: A TypeScript SDK for running MCP (Model Context Protocol) servers with process reuse capabilities</a>: A TypeScript SDK for running MCP (Model Context Protocol) servers with process reuse capabilities - cookiecad/mcp-runner</li><li><a href="https://youtu.be/hYCL8tA-8Nk?si=4B8Gd8NmJstLwV6V">MCP Gmail Extension, control your inbox with a voice agent</a>: Experience the future of email management with SystemPrompt MCP Gmail - where natural voice commands meet intelligent email handling. This demo showcases our...</li><li><a href="https://github.com/Mintplex-Labs/anything-llm/issues/2883">[FEAT]: Model Context Protocol (MCP) Integration · Issue #2883 · Mintplex-Labs/anything-llm</a>: What would you like to see? Description Request to integrate Model Context Protocol (MCP) support into AnythingLLM to enhance interoperability and standardization of context handling across differe...</li><li><a href="https://github.com/Ejb503/multimodal-mcp-client">GitHub - Ejb503/multimodal-mcp-client: A Multi-modal MCP client for voice powered agentic workflows</a>: A Multi-modal MCP client for voice powered agentic workflows - Ejb503/multimodal-mcp-client</li><li><a href="https://github.com/modelcontextprotocol/servers/blob/main/src/everything/sse.ts">servers/src/everything/sse.ts at main · modelcontextprotocol/servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/typescript-sdk/tree/main?tab=readme-ov-file#http-with-sse">GitHub - modelcontextprotocol/typescript-sdk: The official Typescript SDK for Model Context Protocol servers and clients</a>: The official Typescript SDK for Model Context Protocol servers and clients - modelcontextprotocol/typescript-sdk</li><li><a href="https://github.com/Ejb503/multimodal-mcp-client/blob/master/proxy/src/handlers/mcpHandlers.ts#L234-L237">multimodal-mcp-client/proxy/src/handlers/mcpHandlers.ts at master · Ejb503/multimodal-mcp-client</a>: A Multi-modal MCP client for voice powered agentic workflows - Ejb503/multimodal-mcp-client</li><li><a href="https://www.npmjs.com/package/systemprompt-mcp-gmail">systemprompt-mcp-gmail</a>: A specialized Model Context Protocol (MCP) server that enables you to search, read, delete and send emails from your Gmail account, leveraging an AI Agent to help with each operation.. Latest version:...</li><li><a href="https://www.npmjs.com/package/systemprompt-mcp-core">systemprompt-mcp-core</a>: A specialized Model Context Protocol (MCP) server that integrates with systemprompt.io to provide powerful prompt management capabilities. This server enables seamless creation, management, and versio...</li><li><a href="https://www.npmjs.com/package/systemprompt-mcp-notion">systemprompt-mcp-notion</a>: A specialized Model Context Protocol (MCP) server that integrates Notion into your AI workflows. This server enables seamless access to Notion through MCP, allowing AI agents to interact with pages, d...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1332458145414381720)** (10 messages🔥): 

> `MCP Variance Log Tool, KoboldCPP-MCP Server, Notmuch Email Integration, MCP Inception Server, Shopify MCP Server` 


- **MCP Variance Log Tool Launch**: A new tool inspired by the **Titans Surprise mechanism** logs low-probability interactions into a [SQLite database](https://github.com/truaxki/mcp-variance-log) for user data gathering and personalization.
   - This tool aims to enhance long-term memory capabilities by capturing unusual conversation events.
- **KoboldCPP-MCP Server for AI Communication**: A server designed for **AI to AI communication** with KoboldCPP has been shared, facilitating interactions with Claude and other MCP-compatible apps available on [GitHub](https://github.com/PhialsBasement/KoboldCPP-MCP-Server).
   - This setup is aimed at enhancing collaborative AI operations across multiple applications.
- **HTML Email Sending with Notmuch**: A tool named [mcp-notmuch-sendmail](https://github.com/runekaagaard/mcp-notmuch-sendmail) has been created for **Notmuch email users** to send styled HTML emails utilizing Notmuch queries.
   - Feedback is being sought as the tool is still early in development.
- **MCP Inception Server for Parallel Queries**: The **MCP Inception server** allows concurrent queries to be sent to an LLM for various parameters, currently under development, as outlined on [GitHub](https://github.com/tanevanwifferen/mcp-inception).
   - Future updates may enable enhanced functionality for scraping and categorizing cryptocurrencies.
- **Shopify Merchant Integration with Claude**: An MCP server for **Shopify merchants** has been introduced, facilitating natural interactions with Claude for tasks like analyzing store data ([GitHub link](https://github.com/amir-bengherbi/shopify-mcp-server)).
   - This project is in progress with a few initial endpoints focused on products, customers, and orders.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tanevanwifferen/mcp-inception">GitHub - tanevanwifferen/mcp-inception: Call another MCP client from your MCP client. Offload context windows, delegate tasks, split between models</a>: Call another MCP client from your MCP client. Offload context windows, delegate tasks, split between models - tanevanwifferen/mcp-inception</li><li><a href="https://github.com/amir-bengherbi/shopify-mcp-server">GitHub - amir-bengherbi/shopify-mcp-server: MCP Server for Shopify API</a>: MCP Server for Shopify API. Contribute to amir-bengherbi/shopify-mcp-server development by creating an account on GitHub.</li><li><a href="https://github.com/PhialsBasement/KoboldCPP-MCP-Server">GitHub - PhialsBasement/KoboldCPP-MCP-Server: AI to AI comms with koboldcpp from Claude/other MCP compatible apps</a>: AI to AI comms with koboldcpp from Claude/other MCP compatible apps - PhialsBasement/KoboldCPP-MCP-Server</li><li><a href="https://github.com/truaxki/mcp-variance-log">GitHub - truaxki/mcp-variance-log: Agentic tool that looks for statistical variations in conversation structure and logs unusual events to a SQLite database.</a>: Agentic tool that looks for statistical variations in conversation structure and logs unusual events to a SQLite database. - truaxki/mcp-variance-log</li><li><a href="https://github.com/giovannicocco/mcp-server-postman-tool-generation">GitHub - giovannicocco/mcp-server-postman-tool-generation</a>: Contribute to giovannicocco/mcp-server-postman-tool-generation development by creating an account on GitHub.</li><li><a href="https://github.com/runekaagaard/mcp-notmuch-sendmail">GitHub - runekaagaard/mcp-notmuch-sendmail: A model context protocol server that reads mails with notmuch and sends mail with sendmail</a>: A model context protocol server that reads mails with notmuch and sends mail with sendmail - runekaagaard/mcp-notmuch-sendmail</li><li><a href="https://github.com/frgmt0/mcp-reasoner-nightly.git">GitHub - frgmt0/mcp-reasoner-nightly: A systematic reasoning MCP server implementation for Claude Desktop with beam search and thought evaluation.</a>: A systematic reasoning MCP server implementation for Claude Desktop with beam search and thought evaluation. - frgmt0/mcp-reasoner-nightly
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1332451319449125048)** (10 messages🔥): 

> `HeyGen avatars, ElevenLabs voice options, Podcasting with NotebookLM, Mixing HeyGen and MiniMax, NotebookLM note limits` 


- **HeyGen Avatars Workflow Explained**: Using HeyGen, a user detailed a workflow involving screenshots of avatars, **HailouAI/MiniMax**, and **RunWayML's Act-One** to create engaging videos that look like the avatars are 'listening'. Additionally, they provided links to their videos, showcasing the process on [UnrealMysteries.com](https://UnrealMysteries.com).
   - The approach enables better avatar interactions in videos as compared to standard HeyGen outputs, showcasing a novel technique for video creation.
- **HAL-like Voice from ElevenLabs**: A member inquired about the voice used at a specific point in the video, leading another to reveal that it's an **ElevenLabs voice named 'Thomas'**, reminiscent of HAL. This choice was made deliberately to invoke a similar tone.
   - The conversation highlights a trend in using voice technology for creative content, including podcasting and video production.
- **Podcast Integration with NotebookLM**: One user shared their experience using NotebookLM to summarize weekly news into a podcast format, indicating its utility despite not being widely popular. They emphasized a desire for better prompts to enhance audio content production.
   - This reflects a growing interest in leveraging AI tools for content generation, particularly in the podcasting landscape.
- **Innovative Mixing of HeyGen and MiniMax**: A user commented on the impressive capability of mixing **HeyGen** stills with **MiniMax** for extended video creation, praising the seamless integration of both. This technique enhances visual storytelling compared to using either tool independently.
   - Users are exploring creative combinations of technology to improve content production and narrative effectiveness.
- **Inquiry on NotebookLM Note Limits**: A member directed users to the 'Introduction to NotebookLM' notebook for information about note limits, suggesting a cap of **1000 notes** based on previous knowledge. This inquiry shows the ongoing need for clear documentation and user feedback on tool capabilities.
   - Clarifying limits can enhance user experience, ensuring efficient usage of the platform for note-taking and summarization.


  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1332450588494925824)** (226 messages🔥🔥): 

> `NotebookLM usability issues, Audio overview generation delay, PDF source visibility problems, Language settings in NotebookLM, User roles and permissions` 


- **NotebookLM faces usability issues**: Users experience problems with linked sources disappearing after UI changes, limiting functionality and hindering user experience.
   - Community members express frustration over important UX elements being removed during updates despite requests for enhanced features.
- **Audio overview generation delays**: Some users report exceptionally long delays in generating audio overviews from uploaded sources, suggesting a potential bug.
   - Suggestions include deleting and re-uploading sources to mitigate issues with generation times.
- **Problems with PDF source visibility**: Users notice that some pages in uploaded PDFs appear less visible, leading to the AI not providing information for those pages.
   - Concerns were raised regarding how NotebookLM references pages, especially when its approach to page counts vs. printed numbers seems inconsistent.
- **Language settings confusion**: Users express confusion over NotebookLM's default language settings, reporting difficulties with outputs not matching desired languages.
   - Community suggestions involve checking Google account settings and utilizing specific URLs to set language preferences.
- **Understanding user roles**: Questions arise about the 'user' role in people’s profiles, particularly regarding Discord permissions.
   - Clarifications indicate that these roles are likely related to different permissions set within Discord for organizational purposes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google/?hl=es">Google NotebookLM | Note Taking &amp; Research Assistant Powered by AI</a>: Use the power of AI for quick summarization and note taking, NotebookLM is your powerful virtual research assistant rooted in information you can trust.</li><li><a href="https://notebooklm.google.com/notebook/3499bd65-a247-4519-b1b9-0481e9154496/audio">no title found</a>: no description found</li><li><a href="http://cloud.google.com/text-to-speech/docs/basics">no title found</a>: no description found</li><li><a href="https://illuminate.google.com/">Illuminate | Learn Your Way</a>: Transform research papers into AI-generated audio summaries with Illuminate, your Gen AI tool for understanding complex content faster.</li><li><a href="https://notebooklm.google.com/?hl=es-ES">Inicia sesión: Cuentas de Google</a>: no description found</li><li><a href="https://support.google.com/accounts?p=verify_age">Account settings: Your browser is not supported.</a>: no description found</li><li><a href="https://getgotak.com/products/gotak-server">GoTAK On-Site TAK Server</a>: GoTAK Server is an embedded server running the latest version of TAK Server. Get a pre-programmed board delivered to you and skip the mess of the command line and confusing setup. This is the fastest ...</li><li><a href="https://cloud.google.com/generative-ai-app-builder/docs/connect-third-party-data-source">no title found</a>: no description found</li><li><a href="https://youtu.be/ua4rYsMdC4U">AI Software - SNL</a>: A teacher (Ego Nwodim) shows an educational podcast hosted by AI (Timothée Chalamet, Bowen Yang) to her students.Saturday Night Live. Stream now on Peacock: ...</li><li><a href="https://cloud.google.com/distributed-cloud-air-gapped?hl=en#disconnected-sovereign-cloud-solution">Google Distributed Cloud air-gapped | Sovereign Cloud</a>: GDC air-gapped enables public sector organizations and enterprises to address strict data residency and security requirements.</li><li><a href="https://youtubetranscript.com/">YouTube Transcript - read YouTube videos</a>: no description found
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1332477109393686600)** (223 messages🔥🔥): 

> `Hunyuan-video model, Kling AI quality, AI image generation setups, Stable Diffusion RAM requirements, Deepseek model limitations` 


- **Hunyuan-video model success**: Many users have confirmed that the **hunyuan-video model** works efficiently, even on systems with **12 GB VRAM**.
   - While not perfect, it is reported to be effective for image to video applications.
- **Kling AI compared to Hunyuan**: One user remarked that **Kling AI** is currently one of the closest options to the **hunyuan** model in terms of quality for local use.
   - However, it currently lacks essential image-to-video functionality, which is a key requirement for many users.
- **Best setup for AI image generation**: New users are advised to use **Forge or Swarm** as they offer better support and tutorials for those starting with local AI image generation.
   - While **ComfyUI** is highly recommended for advanced users, its complexity can be challenging for beginners.
- **RAM requirements for Stable Diffusion**: For running **Stable Diffusion** effectively, having at least **32GB of RAM** is advised, with **64GB** being optimal.
   - Users with **RTX 4090** or **AMD 7900XTX** systems are encouraged to ensure their RAM meets these requirements to avoid complications.
- **Deepseek model hardware requirements**: The **Deepseek V3 or R1** models require over **1.3TB of VRAM** to function at full precision, which is beyond typical consumer capabilities.
   - Having multiple high-end GPUs like **A100** or **H100** is necessary for running such large models, leaving most users seeking smaller alternatives.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://upscayl.org/">Upscayl - AI Image Upscaler</a>: no description found</li><li><a href="https://openmodeldb.info/">OpenModelDB</a>: OpenModelDB is a community driven database of AI Upscaling models. We aim to provide a better way to find and compare models than existing sources.</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides#amd-forge-webui-with-zluda">Webui Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1332451305720909854)** (20 messages🔥): 

> `Flash Infer Talk Questions, Support for Alternative Attention Methods, Deepseek Events, NCSA Hackathon Participation, Distributed Training Stacks` 


- **Questions for Flash Infer Talk**: Members discussed how to ask questions during the **Flash Infer Talk** on YouTube, noting that one cannot ask without making a channel first.
   - *“If you could ask at some point -- I'm curious about support for alternative attention methods...”* is a key question raised for further clarification.
- **Alternative Attention Methods Exploration**: There was curiosity about support for **differential attention** and its implementation, including the use of **flex attention**.
   - One member shared **pseudocode** demonstrating the concept and noted that flex attention currently does not support differential attention.
- **Deepseek Inquiry**: A member inquired about a summary or TLDR of the events surrounding **deepseek**, signaling interest in recent developments.
   - No further details were provided in the discussions about deepseek.
- **NCSA Hackathon Participation Call**: A member is looking for two more people to join the **NCSA hackathon** and encouraged interested individuals to reach out via DM.
   - Information about the event was shared but faced a CSS error preventing further details from being loaded.
- **Curiosity About Distributed Training Stacks**: A member asked about the technology stacks used by teams for large scale **distributed training**, specifically those handling over 100B parameters.
   - They shared their experience using **JAX** but highlighted the steep learning curve for new team members.



**Link mentioned**: <a href="https://www.openhackathons.org/s/siteevent/a0CUP000013BcYw2AK/se000370">Open Hackathons</a>: no description found

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1332814793886797884)** (1 messages): 

> `Tensor Manipulation in Triton, Inline Assembly in Triton` 


- **Shifting Tensor Elements in Triton?**: A member inquired about methods to *shift tensor elements to the left* in Triton, expressing curiosity about possible inline assembly solutions.
   - No responses were provided, indicating potential uncertainty or lack of existing solutions among the community.
- **Inline Assembly Potential Discussed**: The discussion included whether inline assembly can facilitate tensor operations in Triton, but specifics were not detailed.
   - The interest in this topic suggests a need for more resources or examples.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1332465575028326452)** (76 messages🔥🔥): 

> `PTX ASM Segfault Issues, CUDA Kernel Loading Errors, DeepSeek Discussion, CUDA Versions and Compatibility, NCCL Timeout Debugging` 


- **PTX ASM Segfault Issues**: A user reported a segfault when attempting to store from a vector register to shared memory in their PTX ASM code, suggesting a potential issue with memory addresses.
   - Another user advised checking if the addresses are in the correct memory space and suggested calling `__cvta_generic_to_shared()` on the addresses.
- **CUDA Kernel Loading Errors**: A user encountered an ImportError due to the Ninja build system not being available when trying to load a CUDA module in Jupyter Lab, leading to a failure in finding the required shared object file.
   - After installing Ninja, the user still faced an ImportError indicating that the inline extension could not be opened, possibly due to the build process not completing successfully.
- **DeepSeek Discussion**: A member humorously speculated whether the DeepSeek situation could lead employers to hire more CUDA developers, highlighting the potential cost savings.
   - Other members joined in the banter about the economic implications, referencing the efficiency gains from using proper computing resources.
- **CUDA Versions and Compatibility**: Users discussed compatibility issues with various CUDA versions, specifically noting that older GPUs may struggle with more recent releases like CUDA 12.
   - A user shared their frustrations about the difficulties faced when trying to use outdated hardware with modern software requirements.
- **NCCL Timeout Debugging**: A user inquired about debugging NCCL timeouts encountered during training with multiple nodes, mentioning profiling improvements but unresolved timeout issues.
   - There was a request for best practices in addressing NCCL timeouts, indicating a need for effective strategies in multi-node training setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/cuda">CUDA Toolkit Documentation 12.8</a>: no description found</li><li><a href="https://x.com/__tensorcore__/status/1883060903282954282">Tweet from Vijay (@__tensorcore__)</a>: 🔥🚨 CUTLASS Blackwell is here 🚨🔥3.8 release is loaded with support for new features of Blackwell, even an attention kernel 👀Go check it out here: https://github.com/nvidia/cutlassCan&#39;t wait to...</li><li><a href="https://docs.nvidia.com">NVIDIA Documentation Hub - NVIDIA Docs</a>: no description found</li><li><a href="https://stackoverflow.com/q/53422407/10107454)">Different CUDA versions shown by nvcc and NVIDIA-smi</a>: I am very confused by the different CUDA versions shown by running which nvcc and nvidia-smi. I have both cuda9.2 and cuda10 installed on my ubuntu 16.04. Now I set the PATH to point to cuda9.2. So...</li><li><a href="https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-generations).">1. Introduction — NVIDIA CUDA Compiler Driver 12.8 documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1332448788513685554)** (13 messages🔥): 

> `NCCL timeouts debugging, Linear Warmup in Learning Rates, Torch Inductor Internals, Vision-based model optimizations, Fused CUDA kernels example` 


- **NCCL timeouts during multinode training**: A member reported encountering **NCCL timeouts** during training with 8 nodes and mentioned profiling the code with the PyTorch profiler to improve performance.
   - They sought guidance on **best practices** for debugging NCCL timeouts in multinode setups.
- **Optimal learning rate strategy with linear warmup**: It was reaffirmed by a member that using **linear warmup** for the first N steps before transitioning to a different LR strategy is effective, especially for vision models.
   - They suggested starting with a high learning rate of **5e-4** and gradually decreasing it to **5e-5** over a significant number of iterations.
- **Inquiry about Torch Inductor Internals**: A member inquired about documentation on **Torch Inductor** internals, specifically regarding concepts like **subgraphs, ComputedBuffer,** and **IR nodes**.
   - They were looking for resources that explain the interactions between these concepts in detail.
- **Challenges with Linear + ReLU fused kernels**: One member sought assistance with creating a **fused kernel** in CUDA as a PyTorch extension, wanting to replace standard layers while maintaining weight loading functionality.
   - They expressed interest in implementing this manually, without relying on existing solutions like **triton**.
- **Non-transformer vision models and optimization preferences**: A member shared their experience that **most non-transformer vision models** do not perform well with Adam optimization, preferring alternatives like **Triangular** or **WSD** learning rate schedules.
   - This led to a wider discussion on the efficacy of various optimization strategies in vision tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts">CosineAnnealingWarmRestarts &mdash; PyTorch 2.5 documentation</a>: no description found</li><li><a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR">LinearLR &mdash; PyTorch 2.5 documentation</a>: no description found</li><li><a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR">CosineAnnealingLR &mdash; PyTorch 2.5 documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1332799975129616507)** (1 messages): 

> `Adam Paszke, Mosaic GPU, GPU MODE community, GPU programming` 


- **Adam Paszke on Mosaic GPU**: In just **10 minutes**, the legendary **Adam Paszke** will discuss his DSL for low-level GPU programming, **Mosaic GPU**.
   - Catch the live talk on [YouTube](https://www.youtube.com/@GPUMODE) and expand your knowledge in GPU programming during this insightful session.
- **Explore GPU MODE Community Resources**: For those interested, supplementary content about GPU programming is available on [GitHub](https://github.com/gpu-mode), created by Mark Saroufim and Andreas Köpf.
   - Join the growing community by visiting the official [Discord](https://discord.gg/gpumode) channel for dynamic discussions and collaborative learning.



**Link mentioned**: <a href="https://www.youtube.com/@GPUMODE">GPU MODE</a>: A GPU reading group and community https://discord.gg/gpumodeSupplementary content here https://github.com/gpu-modeCreated by Mark Saroufim and Andreas Köpf 

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1332490589081440256)** (2 messages): 

> `TinyZero, Open R1` 


- **TinyZero replicates DeepSeek R1 Zero**: The [TinyZero](https://github.com/Jiayi-Pan/TinyZero) project is an accessible reproduction of **DeepSeek R1 Zero**, showcasing a clean implementation.
   - Its GitHub repository includes details and images, making it easy for contributors to explore and engage.
- **Open R1 built as a fully open source project**: The [Open R1](https://github.com/huggingface/open-r1) project by Hugging Face is a fully open reproduction of **DeepSeek-R1**, aimed at fostering collaborative development.
   - Developers are encouraged to contribute on GitHub, ensuring an inclusive environment for enhancements and modifications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Jiayi-Pan/TinyZero">GitHub - Jiayi-Pan/TinyZero: Clean, accessible reproduction of DeepSeek R1-Zero</a>: Clean, accessible reproduction of DeepSeek R1-Zero - Jiayi-Pan/TinyZero</li><li><a href="https://github.com/huggingface/open-r1">GitHub - huggingface/open-r1: Fully open reproduction of DeepSeek-R1</a>: Fully open reproduction of DeepSeek-R1. Contribute to huggingface/open-r1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1332811365618028598)** (2 messages): 

> `Atomic Semi Careers, Hinge Health Job Opening` 


- **Atomic Semi seeks hands-on engineers**: Atomic Semi is building a team of exceptional, hands-on engineers to innovate in technology, claiming, **'We’ll own the stack from atoms to architecture.'**
   - *'We believe our team and lab can build anything'* with advanced tools including 3D printers and e-beam writers.
- **Hinge Health hiring Staff Engineer for AI platform**: Hinge Health is hiring a **Staff Engineer** for their AI platform, as shared on LinkedIn [here](https://www.linkedin.com/jobs/view/4096940351).
   - The message encourages spreading the word and offers to **DM for details** about the position.



**Link mentioned**: <a href="https://atomicsemi.com/careers/">Careers</a>: no description found

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1332561635025354843)** (5 messages): 

> `High Performance Computing in ML, Basics of Parallel Computing, Understanding Neural Networks, Self Implementation of SVM, Learning Path for Practical Skills` 


- **Learn the Basics of High Performance Computing**: A member explained that you’ll learn about different concepts crucial for **high performance computing** and **machine learning systems**.
   - They emphasized understanding **GPU architecture** to write efficient algorithms and optimize AI workflows with **hw-aware** techniques.
- **Essentials for Learning**: When asked about prerequisites, a member suggested gaining some **basics of parallel computing** and knowledge of how **neural networks** function.
   - Key topics to focus on include **matmuls**, **attention**, and **activations**.
- **Seeking Help on SVM Implementation**: A member inquired about a doubt related to self-implementing **SVM** using only **Numpy** and **Scipy**.
   - This indicates a collaborative community willing to help with specific coding challenges.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1332529746562514996)** (4 messages): 

> `Tiled Matrix Multiplication Issues, Floating Point Type Mismatch, Dummy Matrix Declaration, Result Comparison Code` 


- **Tiled Matrix Multiplication shows mismatches**: A user reported a mismatch in results between CPU and GPU for large matrices (320x320) when using a certain method to declare dummy matrices, while smaller matrices (4x4) were fine.
   - They initially suspected memory leaks as the cause due to matrix declaration differences.
- **Floating point type suspected for mismatches**: Another member suggested that the issue might stem from the **floating point type** used in the matrix declaration, prompting the user to check if mismatches occur when using integers.
   - This user later clarified that they had declared matrices as float arrays but were feeding them integers.
- **Request for comparison code**: One member asked to share the comparison code to diagnose the inconsistency further, indicating the need for more detailed analysis.
   - This request suggests potential collaboration to identify and resolve the matrix multiplication issue.


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1332941600459853846)** (1 messages): 

> `YouTube Recordings` 


- **Explore Latest YouTube Recordings**: For viewers interested in catching up, there are two new recordings available: [Recording One](https://www.youtube.com/watch?v=iOLBJwENuvA) and [Recording Two](https://www.youtube.com/watch?v=wKd90avC8Nc).
   - These recordings are perfect for those looking to stay updated with the latest discussions.
- **Catch the Excitement from Recent Sessions**: View the recent highlights shared in the recordings which cover essential discussions and insights in the community.
   - These videos are an excellent resource for anyone wanting a closer look at the ongoing topics.


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1332816042749329408)** (3 messages): 

> `Emulation in Torch, JAX fp8 support on Nvidia GPUs` 


- **Exploring Emulation in Torch**: Discussion mentioned that **Torch** has some emulation support, indicating its capabilities to emulate certain functions.
   - A member inquired if there is a **guide** detailing where the emulation features are implemented in the code.
- **JAX's Unique fp8 Capabilities**: A linked discussion on [GitHub](https://github.com/jax-ml/jax/discussions/26077) highlights how **JAX** can run **fp8** on Nvidia GPUs with **sm < 89**, whereas it is typically limited to GPUs with **sm >= 89** like RTX 4090 or A100.
   - This specific discord conversation emphasized the confusion surrounding **running fp8** on older GPUs while noting that **JAX** manages to bypass these issues found in **PyTorch**.



**Link mentioned**: <a href="https://github.com/jax-ml/jax/discussions/26077">Why can JAX run fp8 on Nvidia GPUs with sm &lt; 89? · jax-ml/jax · Discussion #26077</a>: fp8 has hardware support only on GPUs with sm &gt;= 89, such as RTX 4090 or A100. I&#39;ve seen people trying to run it in PyTorch (e.g., this script) on older GPUs and getting errors. But JAX can act...

  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1332807899679953019)** (11 messages🔥): 

> `Mosaic Layout System, TiledLayout Comments, SMEM to Registers Transfer, IR Generation Flags in Mosaic` 


- **Mosaic Layout System Unification**: The discussion highlighted the unification of the **Mosaic** layout system using **XLA tiling notation**, moving away from a collection of special cases.
   - It was noted that this applies specifically to array layouts in registers, while **SMEM** requires only one level of tiling and swizzle.
- **TiledLayout Feedback**: A member expressed appreciation for the discussion, stating that the comment on the **TiledLayout** was 'really beautiful' and precisely what they sought.
   - They are now exploring the mapping from registers to **SMEM**, indicating a hands-on approach is necessary.
- **SRM to Registers Transfer Methods**: The conversation also encompassed the method for synthesizing **SMEM** to register transfers, detailing its implementation in the code.
   - This includes a planner designed to minimize **bank conflicts**, enhancing performance.
- **Interest in Intermediate Representation**: A question was raised regarding the ability to inspect the **intermediate representation (IR)** generated by Mosaic.
   - There was an inquiry about any flags available to dump the IR, signaling a desire for deeper technical insight.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/jax-ml/jax/blob/95cb0eb1c969948f21e901317a083375ad13194a/jax/experimental/mosaic/gpu/fragmented_array.py#L144">jax/jax/experimental/mosaic/gpu/fragmented_array.py at 95cb0eb1c969948f21e901317a083375ad13194a · jax-ml/jax</a>: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - jax-ml/jax</li><li><a href="https://github.com/jax-ml/jax/blob/95cb0eb1c969948f21e901317a083375ad13194a/jax/experimental/mosaic/gpu/fragmented_array.py#L1605-L1606">jax/jax/experimental/mosaic/gpu/fragmented_array.py at 95cb0eb1c969948f21e901317a083375ad13194a · jax-ml/jax</a>: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - jax-ml/jax
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1333542150192758795)** (1 messages): 

> `Tile Lang, BitBLAS repo, Backward kernels` 


- **Tile Lang finally released**: A member expressed excitement about the release of **Tile Lang**, mentioning its prior mention in **BitBLAS** repo commits back in October.
   - *Hope I can finally code those efficient backward kernels that BitBLAS is missing*.
- **Interest in BitBLAS improvements**: There is a growing interest in enhancing **BitBLAS** by integrating **Tile Lang**, aimed at improving the efficiency of backward kernels.
   - This will address some of the functionalities that have been noted as missing.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1333474041151098890)** (1 messages): 

> `dpo loss, simpo loss, liger with trl, ligerdpo trainer` 


- **Inquiry on dpo and simpo loss usage**: A member asked if there is a way to use the **dpo** or **simpo loss** of Liger with **trl** or a **ligerdpo trainer**.
   - They expressed hope that this inquiry is appropriate for the channel.
- **Interest in Liger's functionality**: The member’s inquiry highlights interest in using **Liger**'s specific functionalities with **trl**, suggesting a need for better integration or tools.
   - This also points to the community's ongoing exploration of combining various techniques and tools for enhanced performance.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

mobicham: https://x.com/Mobius_Labs/status/1883951887965393301
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1333212622098661546)** (3 messages): 

> `WGMMA Instructions, Pointer Math in PTX ISA, Memory Handling Strategies` 


- **Exploring WGMMA Instructions Implementation**: A member is attempting to get the **WGMMA instructions** working independently, focusing on **pointer math** to load the correct elements into registers as outlined by the **NVIDIA PTX ISA**.
   - They expressed confusion about memory management, questioning whether the **dst object** handles memory math automatically.
- **TK's Approach to Memory Management**: The same member noted that while looking at **TK's generation of WGMMA instructions**, it appears they avoid pointer math by passing in contiguous memory segments.
   - This led to a query on whether this suggests a misunderstanding of the **PTX ISA** or if there's an automatic handling of such calculations.
- **Thread Load Designation Discussion**: The member assumed that the code is designed for each thread to load its particular elements into designated registers, questioning if this is the correct approach.
   - They are seeking clarification on the mechanism and intentions behind the memory handling in the current design.


  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1332451795095785633)** (44 messages🔥): 

> `Polynomial Equations PR, Maze Task Proposal, FSDP Support in Tiny-GRPO, Family Relationships Dataset, GSM8K Templates` 


- **Polynomial Equations Added to Reasoning Gym**: A member sent a PR adding **polynomial equations** support alongside simple linear equations, enhancing the functionality of reasoning-gym.
   - Another member mentioned that they could copy over algorithms from CLRS due to its Apache license.
- **Maze Task Idea Suggested**: A member proposed adding a **maze task** focused on finding the shortest path length, seeking feedback on this addition.
   - Others expressed excitement, labeling maze riddles as a fantastic contribution to reasoning-gym.
- **FSDP and Tiny-GRPO Enhancements**: A member noted that **FSDP support** was added to Tiny-GRPO, opening doors for reduced VRAM usage, alongside requests for further enhancements.
   - This was seen as a step towards making Tiny-GRPO more user-friendly and efficient.
- **Strategies for Family Relationship Datasets**: Discussion took place regarding strategies to generate a family relationship dataset, revealing complexities in the problem-solving approach for LLMs.
   - A suggested implementation was provided, pointing to an existing family relationships codebase for inspiration.
- **Proposal for GSM8K Template Dataset**: A member suggested creating a template-based version of the **GSM8K dataset**, similar to work done by Apple, with related code released on GitHub.
   - Plans were discussed for hosting this template version on the HF hub, allowing for dynamic downloading on user demand.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hkust-nlp.notion.site/simplerl-reason">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team</li><li><a href="https://github.com/google-deepmind/clrs/tree/master/clrs/_src/clrs_text">clrs/clrs/_src/clrs_text at master · google-deepmind/clrs</a>: Contribute to google-deepmind/clrs development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/4">Add procedural Alice in Wonderland (AIW problem) dataset · Issue #4 · open-thought/reasoning-gym</a>: The Alice in Wonderland problem has following base template (and many variations): &quot;Alice has N brothers and she also has M sisters. How many sisters does Alice’s brother have?&quot; See paper: A...</li><li><a href="https://github.com/cpldcpu/MisguidedAttention">GitHub - cpldcpu/MisguidedAttention: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information</a>: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information - cpldcpu/MisguidedAttention</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/6">Dataset: Unscramble words · Issue #6 · open-thought/reasoning-gym</a>: I would suggest to start with Level 0: Unscramble words: load natural language text span of up to max_length (e.g. from the Jules Vern short story in data), for each word swap characters at random ...</li><li><a href="https://github.com/apple/ml-gsm-symbolic">GitHub - apple/ml-gsm-symbolic: GSM-Symbolic templates and generated data</a>: GSM-Symbolic templates and generated data. Contribute to apple/ml-gsm-symbolic development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/tiny-grpo/blob/eafedd78ff86dbb724a3dd21bb04ab6523ac8f3c/train.py#L122-L130">tiny-grpo/train.py at eafedd78ff86dbb724a3dd21bb04ab6523ac8f3c · open-thought/tiny-grpo</a>: Minimal hackable GRPO implementation. Contribute to open-thought/tiny-grpo development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/graphs/family_relationships.py">reasoning-gym/reasoning_gym/graphs/family_relationships.py at main · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/11">Add support for all CLRS tasks by panispani · Pull Request #11 · open-thought/reasoning-gym</a>: CLRS is the classic textbook on algorithms.Deepmind introduced a CLRS benchmark which also includes a text version of most of the classical algorithms called CLRS-text. In this PR, I ported all th...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/12">[Level 1 of Unscrambled tasks] - Add sentence reordering and unit tests to validate it by Adefioye · Pull Request #12 · open-thought/reasoning-gym</a>: Add dataset generator for level 1 of unscrambled tasks in issue 6.I see this task as Sentence re-ordering task.Example of data generated:{&amp;#39;question&amp;#39;: &amp;#39;Correct the following sen...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/8">Add set up instructions by Adefioye · Pull Request #8 · open-thought/reasoning-gym</a>: Experienced some issues successfully building the reasoning_gym repo. Feel other persons might need it to avoid wasting time.</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/9">Add setup instructions to README.md · Issue #9 · open-thought/reasoning-gym</a>: The primary way of consuming the lib should be via pip install reasoning-gym see project PyPI page. For install from source locally after git clone pip install -e . (-e for editable). Beside that f...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1333523117607227442)** (5 messages): 

> `Mojo Documentation, GPU Package API` 


- **Mojo Documentation Experiences Downtime**: The **Mojo documentation** experienced downtime, with the team actively working to resolve the issue as quickly as possible.
   - *Appreciation for patience* was expressed during the downtime phase, indicating the team's commitment to keeping users informed.
- **Docs Back Up with New GPU API Information**: After the outage, a member confirmed that the **docs are back up** and available again.
   - The **GPU package API documentation** is now also accessible in the nightly release, which is a relief for users seeking updated information.
- **Typo in Code Reference**: A member pointed out a potential typo in the code snippet, noting it should specify `# val=6` instead.
   - They attached an [image](https://cdn.discordapp.com/attachments/1098713601386233997/1333523117112561706/image.png?ex=679933ae&is=6797e22e&hm=f4806c352f0e31d85e4082447280257cbc8624a7b39ab2b101bf42dc4174ded4&) for clarity regarding the error.
- **GitHub Changelog Link Shared**: A link to the **Mojo GitHub changelog** was shared, providing insight into the updates made to the documentation.
   - The [changelog](https://github.com/modular/mojo/blob/nightly/docs/changelog.md) inclusion helps users stay updated on the latest changes in the Mojo programming language.



**Link mentioned**: <a href="https://github.com/modular/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modular/mojo</a>: The Mojo Programming Language. Contribute to modular/mojo development by creating an account on GitHub.

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1332488778928881735)** (91 messages🔥🔥): 

> `Mojo CSS Struct, List and Representable Trait Issues, Unsafe Pointers and Object Identity, Function Pointer FFI in Mojo, Documentation Downtime` 


- **Mojo CSS Struct Development**: A user is creating a `struct` to generate CSS using a `fluent API` style, encountering issues with unused value warnings in Zed Preview.
   - They suggested using `_ = ` to suppress warnings but expressed a preference for a cleaner solution.
- **List and Representable Trait Confusion**: Discussion arose about the confusion over the `Representable` trait, particularly concerning using `List[Int]` as an argument in a function.
   - It was noted that while `List` has a representation function, the compiler not recognizing `Int` as `Representable` indicated issues with conditional conformance.
- **Unsafe Pointers and Object Identity Concerns**: A user was troubleshooting issues with object identity while using `UnsafePointer` for backward operations in their value struct, finding their pointers affected independently.
   - They discovered adjusting class variables to pointers facilitated tracking changes correctly, thus maintaining object identity.
- **Function Pointer FFI Limitations in Mojo**: The conversation highlighted that current versions of Mojo do not reliably support passing function pointers to C functions.
   - It was clarified that although subsets of Mojo may conform to the C ABI, documentation for such rules is lacking.
- **Documentation Downtime and Resolutions**: Users reported issues accessing the Mojo documentation, finding it down due to hosting problems with Cloudflare.
   - The team acknowledged the issue and confirmed that the documentation was back up shortly after the initial reports.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modular/mojo/issues/3968">modular/mojo</a>: The Mojo Programming Language. Contribute to modular/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modular/mojo/blob/nightly/stdlib/src/builtin/int.mojo#L1146">mojo/stdlib/src/builtin/int.mojo at nightly · modular/mojo</a>: The Mojo Programming Language. Contribute to modular/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modular/mojo/blob/nightly/stdlib/src/collections/list.mojo#L441">mojo/stdlib/src/collections/list.mojo at nightly · modular/mojo</a>: The Mojo Programming Language. Contribute to modular/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1332867869377302611)** (4 messages): 

> `Multi-agent workflows, Document research agents, Automation in travel insurance claims, LlamaIndex integrations, DeepSeek API` 


- **Presenter: Your New Multi-Agent Workflow for Presentations**: Introducing [Presenter](https://twitter.com/llama_index/status/1883307955782901926), a multi-agent workflow that creates visually rich presentations, featuring **Mermaid diagrams** and script generation.
   - This repo serves as a perfect reference for those aiming to build a **report generation agent** with impressive functionalities.
- **Open-Source Template for Document Research**: Inspired by Google Deep Research, @MarcusSchiesser created a fully open-source full-stack template for **multi-step document research agents** [here](https://twitter.com/llama_index/status/1883675662839636427).
   - This tool addresses the significant need users have in efficiently handling complex research tasks.
- **Scaleport AI Automates Claim Estimation**: Learn how [Scaleport AI](https://twitter.com/llama_index/status/1883929949205336509) partnered with a leading travel insurance provider to **automate claim estimation** from complex medical reports using LlamaIndex.
   - They utilized **advanced OCR** for data extraction, showcasing the effectiveness of AI-driven analysis in this space.
- **LlamaIndex's First-Party Integration with DeepSeek-R1 API**: LlamaIndex now integrates with the [DeepSeek-R1 API](https://twitter.com/llama_index/status/1883986763380842864), allowing the usage of models like `deepseek-chat` and `deepseek-reasoner`.
   - Visit [DeepSeek](https://api-docs.deepseek.com/) for details on API keys and model support, making integration seamless for users.



**Link mentioned**: <a href="https://t.co/jtfBvBig1y">DeepSeek - LlamaIndex</a>: no description found

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1332628659026268191)** (74 messages🔥🔥): 

> `Access to LlamaIndex, LLM.complete kwargs usage, Evaluators in LlamaIndex, Local RAG Implementation, Documentation issues` 


- **Access to LlamaIndex Update**: A user inquired about the timeline for gaining access to LlamaIndex while currently on the waiting list.
   - No specific timeframe was provided in the discussion.
- **Using kwargs with LLM.complete**: A member sought documentation on using `**kwargs` with `LLM.complete`, aiming to pass parameters dynamically in the message method.
   - Community responses indicated that kwargs are sent to the LLM API and advised specifying `generation_config` for parameters.
- **Costs Associated with Evaluators**: A user queried whether the `FaithfulnessEvaluator` and `RelevancyEvaluator` incur additional costs through API requests when using Anthropic as the LLM.
   - It was confirmed that both evaluators utilize LLM calls, impacting costs.
- **Local RAG Implementation Queries**: A participant discussed challenges faced while implementing a local RAG with Ollama and LlamaIndex, expressing difficulty in finding useful documentation.
   - Despite initial issues, they managed to work through the starter example but identified documentation as lacking in clarity.
- **Documentation and Performance Issues**: Concerns were raised about documentation for models and the performance of LlamaIndex, with one user mentioning high CPU usage during operation.
   - The community acknowledged documentation limitations but encouraged contributions for improvement and discussed the benefits of using Ollama for easier configuration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/understanding/agent/rag_agent/">Adding RAG to an agent - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found</li><li><a href="https://christophergs.com/blog/running-open-source-llms-in-python#install">Running Open Source LLMs In Python - A Practical Guide</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp/">LlamaCPP - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/">Building an LLM Application - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/vllm/">vLLM - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/pull/17647">update llama-cpp integration + docs by logan-markewich · Pull Request #17647 · run-llama/llama_index</a>: The docs and dependencies where pretty out of date. Updated to be slightly more modern</li><li><a href="https://github.com/run-llama/llama_index/issues/7547">[Question]: GGUF model support? · Issue #7547 · run-llama/llama_index</a>: Question Validation I have searched both the documentation and discord for an answer. Question The new version of llama.cpp throws an error because now will only support GGUF based models. The work...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1332534744990089311)** (34 messages🔥): 

> `Cohere Legal Regulations, Cohere UI Feedback, Community Engagement, GitHub as Collaboration Tool` 


- **Cohere's Legal Regulations are Minimal**: Legal feedback confirmed that most AI trade with Japan is unaffected by new regulations, which focus mainly on advanced computing chips and large models. Cohere models do not fall into these categories, and the regulations won't take effect until May 2025.
   - The legal team is actively monitoring the situation for any potential changes before the regulations are enacted.
- **User Feedback on Cohere Website UI**: Feedback highlighted confusion regarding the UI on the [Cohere dashboard](https://dashboard.cohere.com/), particularly the similarity of buttons on both sides of the page. Suggestions included placing Discord and email contact options for better visibility.
   - One user recommended design changes, like simplifying button placements similar to another platform.
- **Seeking Partnerships for Collaboration**: A user inquired about finding partners for collaborative projects, with a suggestion to utilize GitHub. It was noted that this channel is primarily for discussions, not recruitment.
   - Members expressed the need for diversity in the field rather than creating multiple companies doing the same thing.
- **Welcome New Community Members**: New members joined the community, introducing themselves as software developers and AI enthusiasts eager to contribute. A general atmosphere of camaraderie and welcome was present among users.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://platform.deepseek.com/usage,">DeepSeek Platform</a>: Join DeepSeek API platform to access our AI models, developer resources and API documentation.</li><li><a href="https://dashboard.cohere.com/">Login | Cohere</a>: Login for access to advanced Large Language Models and NLP tools through one easy-to-use API.</li><li><a href="https://cohere.com/">The World&#x27;s Leading AI Platform for Enterprise | Cohere</a>: Cohere is the leading AI platform for enterprise. Augment your workforce, automate workflows, and enrich customer experiences with secure and scalable AI.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1333025312203866212)** (34 messages🔥): 

> `Cohere Documentation, Reverse Planning, Cohere LLM Usage, Cohere Platform Overview, TTS and STT Capabilities` 


- **Cohere provides LLMs with no TTS or STT capabilities**: Cohere's platform focuses exclusively on large language models (LLMs) and does not provide text-to-speech (TTS) or speech-to-text (STT) functionalities.
   - This was confirmed by users discussing the capabilities of the platform and it being clarified that only LLMs are offered.
- **Using LLM in Cohere: A Step-by-Step Guide**: To use an LLM in Cohere, define the LLM with the `ChatCohere` class, bind tools using `bind_tools`, and invoke the LLM with messages.
   - An example code snippet was provided to guide users through this process, illustrating the steps clearly.
- **Understanding the Concept of Reverse Planning**: One user proposed a method of reverse planning, checking the last steps to facilitate actions from that point backward.
   - The bot initiated searches for information on reverse planning techniques but did not find specific resources.
- **What is Cohere? A Brief Overview**: Cohere is a platform that allows the development of LLM-powered applications, emphasizing secure and private deployment.
   - It offers a toolkit for building natural language tasks such as classification, summarization, and content generation, with flexibility for custom model training.
- **Cohere's Multi-Step Tool Use Explained**: Cohere documentation describes a systematic approach to tool use involving several stages from user message retrieval to response generation.
   - Multi-step tool use allows for sequential reasoning which is crucial for agents needing to adapt during tasks.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1332556881591533599)** (2 messages): 

> `App Check-In, Direct Messages` 


- **User expresses interest in the app**: A user indicated they would check out the app and mentioned they would follow up via direct message.
   - This suggests ongoing engagement and interest within the community regarding the app.
- **Reminder to check DM**: A member tagged another user, requesting them to check their direct messages.
   - This emphasizes the importance of direct communication in coordinating activities or feedback.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1332449455148109955)** (50 messages🔥): 

> `Open Source Image Analysis Models, DeepSeek Model Issues, Running Models Locally, Document Analysis Tools, DeepSeek R1 Availability` 


- **Quest for Open Source Image Analysis Model**: Users are inquiring about the best open-source image analysis models that allow image uploads and query responses, with mentions of using frameworks like [Taggui](https://taggui.com) for tagging images.
   - While suggestions include various models, clarity around which ones effectively analyze and respond to image queries remains elusive.
- **Troubleshooting DeepSeek Model Errors**: Several users reported issues with the DeepSeek R1 model, particularly errors in chat templates and reasoning tasks, suggesting the model isn't yet fully functional out-of-the-box.
   - Discussion points include specific models that work locally and various performance benchmarks around tools like LLAMA and DeepSeek.
- **Interest in Local Document Analysis Tools**: Users express the need for local tools to analyze personal documents without uploading them to services, seeking statistical insights and topic occurrences.
   - Suggestions include using software like PDFGear, though concerns remain about data privacy.
- **DeepSeek R1 Support and Availability**: The community inquires about the status of DeepSeek R1 being supported in GPT4All, with mentions of ongoing work to enable smooth integration.
   - Users are curious about a timeline for when DeepSeek will be fully operational without additional setup.
- **Local vs Remote Model Queries**: Discussion around running models locally versus remotely indicates challenges with template setups and the functionality of local files.
   - Some users attempt to connect models with local documents but face syntax errors and the need for configuration adjustments.



**Link mentioned**: <a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF · Hugging Face</a>: no description found

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1332820683201183825)** (1 messages): 

> `Advanced LLM Agents MOOC, Livestream Schedule, Course Website, Enrollment Info, Course Completion Certificate` 


- **Advanced LLM Agents MOOC launches soon**: The **Advanced LLM Agents MOOC** starts on **January 27th** at **4:00PM PST** and will continue until **April 28th**.
   - Everything you need, including livestream URLs and homework assignments, can be found on the [course website](http://llmagents-learning.org/sp25).
- **Livestream Schedule Announced**: We will host a livestream for each guest speaker every **Monday from 4:00PM - 6:00PM PST**, starting January 27th.
   - The first [livestream link](https://www.youtube.com/live/g0Dwtf3BH-0) will be shared on the course website in the syllabus.
- **Enroll Now for the Course**: It's not too late to enroll in the course — you can [sign up here](https://forms.gle/9u6HdVCWXgws16go9).
   - Questions or feedback should be directed to the course staff in <#1280370030609170494>.
- **Course Completion Certificate Details Coming Soon**: More information regarding the **course completion certificate requirements** will be released soon.
   - Note that there are **no deadlines** for the first week of lectures.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1332446684688551958)** (47 messages🔥): 

> `Certificate Distribution, MOOC Enrollment Confirmation, Course Time Zone Participation, In-Person Attendance, Hackathon Participation` 


- **Certificates Still Pending**: Members discussed the status of **MOOC Fall'24 certificates**, which have **not been released yet**; announcements will come soon.
   - *Thank you for your patience!*
- **MOOC Enrollment Confusion**: Several members have expressed concerns over not receiving **confirmation emails** for course enrollment after submitting their applications.
   - One stated, *In the same boat...* regarding the lack of updates.
- **Participation in Different Time Zones**: Participants asked about attending lectures in **different time zones** and were reassured that all sessions will be available on **YouTube** after livestreaming.
   - One member specifically inquired, *Can I pursue the lectures in offline mode?*
- **No In-Person Attendance Available**: It was confirmed that there is **no opportunity** for MOOC students to attend lectures in person, prioritizing Berkeley students due to limited capacity.
   - One member requested to meet instructors but was informed of the policy restrictions.
- **Interest in Hackathon Opportunities**: A request was made regarding potential **hackathon opportunities** for the upcoming course, with instructors acknowledging the expressed interest.
   - Members are hopeful for future hackathons, contributing to a lively discussion.



**Link mentioned**: <a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC, Fall 2024

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/)** (1 messages): 

interdimensionalbeing_: https://substack.com/home/post/p-154577981
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1333183689747595426)** (14 messages🔥): 

> `Gradient Calculation Confusion, STRIDE vs FLIP Discussion, Meeting #55 Agenda, Asset Fetching Issues in TinyChat, RISC Architecture Inquiry` 


- **Clarification Needed on Gradient Documentation**: Confusion arose regarding the documentation for `Tensor.gradient`, which states, *'Compute the gradient of the targets with respect to self'* but seems inaccurate given the calling context.
   - A possible revision to *'Compute the gradient of self with respect to the targets'* was suggested for clarity.
- **Shift from STRIDE to FLIP**: A suggestion was made to replace **STRIDE** with **FLIP**, as the term STRIDE is considered too generic.
   - This change aims to bring more specificity and clarity in naming conventions.
- **Meeting #55 Scheduled Agenda**: Meeting #55 is scheduled for 6am Monday San Diego time, discussing various topics including **company updates**, new multi and gradient, and project bounties.
   - Key areas of focus will include models like **resnet** and **bert**, alongside project-specific discussions.
- **Issues Fetching Font Assets in TinyChat**: There are issues with fetching font assets in TinyChat, as some assets were not covered in the example script `fetch_assets.sh`.
   - A user expressed willingness to locate the missing files for resolution.
- **Inquiry on RISC Architecture for Field Deployment**: A newcomer inquired about RISC architecture options for running **field-deployed models**, targeting a maximum weight of 8G.
   - The goal is to find solutions for ultra-compact physical deployment.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1332570120685486141)** (18 messages🔥): 

> `BobNet Clarification, Formatting Tools in Tinygrad, Tinygrad Valorization Plans, Tinygrad Learning Resources, Tensor UOp Operations` 


- **Clarifying BobNet's Name Origin**: A user questioned the naming of the model **BobNet**, pointing out that it doesn't align with the 'Bounding Box Network' remark, as it appears to be a normal feed-forward network instead. They referenced its [GitHub link](https://github.com/qurAI-amsterdam/bobnet) for further context.
   - *Is it a bounding box network?* It’s crucial to differentiate between the naming conventions and actual functionality.
- **Tinygrad Formatting Tools Discussed**: A user inquired about official formatting tools used in Tinygrad, noting inconsistencies with **Black** and **Ruff**. Another member confirmed that **ruff** is used, providing a link to the related [pre-commit config](https://github.com/tinygrad/tinygrad/blob/master/.pre-commit-config.yaml#L7).
   - This informs contributors about the preferred tools, helping keep code formatting consistent.
- **Discussion on Tinygrad's Runtime Support**: A user raised questions surrounding the future of **Tinygrad** regarding its support for both **NV=1 and CUDA=1** runtimes. This led to a discussion on whether it’s worth exploring Tinygrad with CUDA relation.
   - Determining the longer-term roadmap for Tinygrad could impact users' decisions on contribution and learning efforts.
- **Best Starter Resources for Tinygrad**: *I've found the best getting started tutorial* to be a specific [GitHub repository](https://github.com/mesozoic-egg/tinygrad-notes/tree/main) on Tinygrad notes, which was recommended in the channel. This could ease newcomers into understanding the framework.
   - A growing list of resources can enhance accessibility for beginners eager to dive into Tinygrad and machine learning.
- **Exploring Tensor UOp Functionality**: Inquiry arose about the usage of differing methods like **_broadcasted()**, **_apply_uop()**, and **_apply_broadcasted_uop()** in `tensor.py`, especially regarding **Tensor.lshift**. It was noted, though, that altering its definition led to numerous errors, prompting a need for further clarification.
   - Discussion emphasized the potential for streamlining these operations, but highlighted complexities in implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/master/.pre-commit-config.yaml#L7">tinygrad/.pre-commit-config.yaml at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/qurAI-amsterdam/bobnet">GitHub - qurAI-amsterdam/bobnet: PyTorch implementation of the Bounding Box Network (BoBNet) from the ConvNet-Based Localization of Anatomical Structures in 3D Medical Images paper.</a>: PyTorch implementation of the Bounding Box Network (BoBNet) from the ConvNet-Based Localization of Anatomical Structures in 3D Medical Images paper. - qurAI-amsterdam/bobnet</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/tree/main">GitHub - mesozoic-egg/tinygrad-notes: Tutorials on tinygrad</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1332519377437986977)** (5 messages): 

> `GPU Efficiency, WSL Virtualization, Regex Misformatting, EBNF Grammars` 


- **CPU Training Takes Forever**: A member shared their experience running a training process on CPU, noting that it has been stuck on **step 0/76** for several hours despite a **60% CPU utilization** from Python.
   - *It's hard to appreciate how efficient GPUs are until you can't use one.*
- **Finding WSL Virtualization Option**: After revisiting WSL, a member discovered the **virtualization option** in their motherboard settings, which was surprisingly hidden under Overclocking.
   - *This should pick up considerably now that I got it running.*
- **Regex Misfires**: A discussion emerged around regex, with one member commenting that **regex looks like a misformatted tokenizer**.
   - This prompted another member to share their shift away from regex.
- **Shift to EBNF Grammars**: One member noted that they have stopped using regex in favor of **EBNF grammars**, which they find more readable.
   - *They are more verbose but human readable.*


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1332838739600216096)** (6 messages): 

> `Federated Learning with Torchtune, Selective Application of Optimizer Hooks, Using torch distributed primitives, Managing Optimizer States` 


- **Fed Learning: Managing Splits Effectively**: There was a suggestion to create **N splits per node** in federated learning, training on one split per entity, then merging weights before starting the next split with the saved optimizer state.
   - This approach raised questions on efficiency and whether it's possible to avoid interruptions in **Torchtune** training.
- **Leveraging Torch Distributed for Efficiency**: Using **torch distributed primitives** was proposed as a means to optimize federated learning, allowing updates within process groups before a global sync to rank 0.
   - Additionally, employing **raylib** was mentioned as a way to orchestrate outer parallelism for improved management.
- **Unlocking Gains with Backward Hooks**: A question was raised about potential **performance gains** from selectively applying **opt-in backward hooks** for parameter updates.
   - It was discussed whether to step for some parameters immediately when gradients are ready, while others await a full optimizer step.
- **Optimizers and Parameter Selection Strategy**: Consideration was given to applying the optimizer intelligently to only certain parameters, like the output projection, for quicker gradient clearance.
   - However, concerns were voiced about the complexity of supporting a **main optimizer** alongside separate optimizers for subsets of parameters, along with the typical challenges of grad accumulation.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1333489666090991616)** (10 messages🔥): 

> `Deepseek, Nvidia Stock Experiences, Market Sentiment, Investment Strategies, Comparison of AI Models` 


- **Deepseek Needs to Chill with Updates**: A user expressed frustration about the rapid pace of updates from **Deepseek**, referencing a detailed tech report available [here](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf).
   - The report focuses on the **Janus-Series**, which is aimed at unified multimodal understanding and generation.
- **Nvidia Stock Causes User Regrets**: A user lamented about their Nvidia investments, fearing a potential failure for the second time in a week, leading to a light-hearted blame game among users.
   - Another user humorously suggested they would notify others so they could short their future purchases.
- **Market Triggers and Sentiment**: Discussion emerged around the recent market trends being tied to Japan's interest rates, while some users incorrectly pointed fingers at **Deepseek** for these fluctuations.
   - The chat noted the importance of countering prevalent market sentiment, emphasizing caution when taking financial advice.
- **Advice on Investment Approaches**: One user recommended always doing the opposite of market sentiment as a potential strategy, while another cautioned against seeking financial advice from devs.
   - Jokes about unconventional investments such as beanie babies highlighted the humorous tone of the discussion.
- **Model Comparisons Under Scrutiny**: A user questioned if the models referenced for comparisons were outdated, indicating a focus on current technology in discussions about **Deepseek**.
   - An image relating to this inquiry further fueled the debate on the relevance and timeliness of model evaluations.



**Link mentioned**: <a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>: Janus-Series: Unified Multimodal Understanding and Generation Models - deepseek-ai/Janus

  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1332634100736983040)** (12 messages🔥): 

> `Project Development Status, Website Updates, Python Interpreter Functionality, Latest Development Version, User Feedback and Suggestions` 


- **Project Development Status Confirmed**: A member confirmed that the project's last commit was **yesterday**, indicating ongoing development.
   - This assures the community that progress is still being made despite concerns.
- **Website Undergoing Minimal Changes**: In response to inquiries, a member commented that the website is maintained in a **minimal state** during the current development phase.
   - They promised a refresh will occur once the main launch happens.
- **1.0 Release Will Include Python Interpreter**: A member shared that the upcoming **1.0** version will incorporate the same Python interpreter while transitioning to a new bash tool.
   - They noted this method is **token efficient** for operations needing a continuous Python interpreter.
- **Accessing the Latest Development Version**: A member requested the latest version download link and was provided with one to the **GitHub repository**.
   - They shared installation commands using `pip install` for ease of access.
- **Enhancements Suggested for User Interaction**: Feedback was given on expanding the AI's capabilities with utility functions to reduce repetitive tasks.
   - Suggestions included better **internal documentation** and the ability for the AI to edit its own instructions.



**Link mentioned**: <a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1332953693640986655)** (2 messages): 

> `Deepseek_r1, API Errors` 


- **Deepseek_r1 Model Encountered Issues**: A user attempted to utilize the model **Deepseek_r1** with the following configuration: `llm: model: "Deepseek_r1" temperature: 0 api_key: "sk-d....." api_base: "https://api.deepseek.com"`.
   - However, the model returned a **400 error** indicating that the model does not exist, causing confusion in the workflow.
- **BadRequestError on API Call**: An error was reported when running the interpreter with the command `$ interpreter -y --profile deepseek.yaml`, resulting in a `BadRequestError`.
   - The error details specified it as an **OpenAIException**, pointing to an **invalid_request_error** due to the model's absence.



**Link mentioned**: <a href="https://api.deepseek.com"```">no title found</a>: no description found

  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1332492119574712382)** (4 messages): 

> `DeepSeek models, Open Interpreter local setup, AI terminal app development, Vision model discussions` 


- **DeepSeek models show promising performance**: DeepSeek's first-generation reasoning models, notably **DeepSeek-R1**, are delivering performance comparable to OpenAI-o1 across various tasks including math and code.
   - The smaller distilled models, like **DeepSeek-R1-Distill-Qwen-1.5B**, have also been shown to achieve excellent benchmark results.
- **Running Open Interpreter locally**: Open Interpreter can be fully run locally, allowing users to integrate multiple local model providers such as [Ollama](https://www.ollama.com/) and [Llamafile](https://github.com/Mozilla-Ocho/llamafile).
   - Users can simplify the local setup process using the Local Explorer feature with the command `interpreter --local`.
- **Challenges with DeepSeek in OS Mode**: A member shared that OS mode currently lacks integration with **DeepSeek** since it requires a tool calling and a vision model to function properly.
   - There seems to be ongoing thoughts on finding a workaround to utilize DeepSeek effectively in this context.
- **Discussion on Vision Model Functionality**: Concerns were raised regarding the vision model being enabled in a multi-model setup, questioning its necessity within the system.
   - The dialogue suggests the need for clarity on how vision models operate within the current configuration.
- **Call for Contributions to DSH - AI Terminal**: An open-source initiative called **DSH - Ai terminal** is seeking contributors to enhance the app, improve features, and receive feedback.
   - Participants are encouraged to star the project on [GitHub](https://github.com/gokul6350/dsh-shell) to support its development and check out the provided screenshot for a visual overview.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ollama.com/library/deepseek-r1">deepseek-r1</a>: DeepSeek&#39;s first-generation of reasoning models with comparable performance to OpenAI-o1, including six dense models distilled from DeepSeek-R1 based on Llama and Qwen.</li><li><a href="https://docs.openinterpreter.com/guides/running-locally">Running Locally - Open Interpreter</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1332626872441372682)** (12 messages🔥): 

> `DeepSeek R1 performance, Audio augmentation tools, Pipeline testing` 


- **DeepSeek R1 underwhelms expectations**: Preliminary comparisons of **DeepSeek R1** indicate it likely matches **o1-mini** and **Claude 3.5 Sonnet**, contradicting claims it rivals **o1** on tough benchmarks. This assessment brings into question its ability to handle AIW problems that reveal generalization deficits, as discussed [here](https://x.com/JJitsev/status/1883158738661691878).
   - *Yet another tale of Rise and Fall* raises concerns about DeepSeek R1's performance on olympiad-level challenges, making it essential to investigate its real capabilities [in this paper](https://arxiv.org/abs/2406.02061).
- **Request for enhanced audio tools in pipeline**: A member suggested adding **audio widgets** to better compare the effects of augmentation on sound variations. They also recommended sourcing more diverse distortions and noises from libraries like **DeepSeq** or **O1**.
   - This feedback aims to refine the pipeline being developed, emphasizing the need for functionality that allows users to experience audio changes more interactively.
- **Pipeline testing and development progress**: A user shared a link to their testing pipeline, currently in the development phase, after a busy day of travel. The link can be found [here](https://colab.research.google.com/drive/1tc4YgdsZeEtsZCdnawYaEC7b12NBQfYt?usp=sharing).
   - They encouraged feedback on the pipeline from others in the channel, hoping to refine the features and functionality further.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1tc4YgdsZeEtsZCdnawYaEC7b12NBQfYt?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://x.com/JJitsev/status/1883158738661691878">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev)</a>: (Yet) another tale of Rise and Fall:  DeepSeek R1 is claimed to match o1/o1-preview on olympiad level math & coding problems. Can it handle versions of AIW problems that reveal generalization & basic ...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1332475950696501258)** (2 messages): 

> `DeepSeek R1, AIW Versions Comparison, Benchmarking Performance` 


- **DeepSeek R1's Claims Under Scrutiny**: Preliminary results comparing **DeepSeek R1** using AIW versions suggest it does not match or outperform **o1** in tough benchmarks, despite previous claims.
   - Currently, **DeepSeek R1** appears to be on par with **o1-mini** and **Claude 3.5 Sonnet**, raising questions about its efficacy in handling challenging problems.
- **DeepSeek R1's Limitations on Complex Tasks**: Concerns were raised about whether **DeepSeek R1** can manage versions of **AIW problems** that expose the generalization and reasoning gaps in leading LLMs.
   - According to one post, it remains uncertain if **DeepSeek R1** can effectively address olympiad-level math and coding challenges.
- **Social Media Buzz on DeepSeek R1**: A post on X highlighted that the narrative around **DeepSeek R1** showcases its rise and fall in performance claims against established models.
   - The ongoing discussion reflects skepticism regarding its ability to cope with rigorous mathematical and coding tasks.



**Link mentioned**: <a href="https://x.com/JJitsev/status/1883158738661691878">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev)</a>: (Yet) another tale of Rise and Fall:  DeepSeek R1 is claimed to match o1/o1-preview on olympiad level math & coding problems. Can it handle versions of AIW problems that reveal generalization & basic ...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1333164853799555082)** (8 messages🔥): 

> `GitHub issue spamming, Natural language vs programming, dspy + deepseek optimization, pypi version update` 


- **GitHub flooded with spam issues**: A member noted that the **issues section on GitHub** currently appears to be in rough shape due to random spam submissions.
   - *It seems like someone is spamming GitHub with random issues*.
- **Seeking differentiation model for languages**: A user inquired about the existence of a model, not limited to **LLMs**, that differentiates between **natural language** and code formats like programming and HTML.
   - This indicates a search for tools to better categorize types of textual content.
- **Challenges with dspy and deepseek optimization**: A member asked if anyone managed to run **dspy + deepseek** 70B optimization for a **COT** example.
   - *What do you mean optimized for COT?* triggered further discussion about the functionality.
- **Concerns over long convergence times**: One member reported running a **BSR example** for six hours without convergence, raising concerns about the process’s duration.
   - The dialogue suggests a frustrating experience with the optimization process.
- **Request for new pypi version push**: A user expressed frustration that the latest **RC on pypi** does not work well with modern **FastAPI** apps due to outdated dependencies.
   - *The current issue was fixed on main as of 3 weeks ago*, prompting the need for a new push to pypi.


  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1333249619383029852)** (7 messages): 

> `deepseek algorithm, H200 vs 5090s, RL framework support` 


- **Discussion on Deepseek Algorithm Implementation**: Members are inquiring about the implementation of the **deepseek algorithm**, with one member asking if anyone is reproducing it.
   - Another member suggests it may refer to **grpo**, which was recently added in **trl**.
- **Comparing GPU Options: H200 vs 5090s**: A member is considering purchasing **2x 5090s** or **1x H200**, noting that the H200 has more RAM but could be slower.
   - Despite the H200's advantages, the query highlights uncertainty about performance relative to the 5090s.
- **Inquiry about Expanding RL Support**: A member notes the current lack of support for **trl's online RL trainers** but expresses interest in expanding RL framework support.
   - They invite feedback on other RL frameworks that users may want to see integrated.
- **Limited Future Expansion for RL Support**: In response to a question about expanding RL support later, a member states that it is **most likely not** going to happen.
   - This suggests a firm stance on the current lack of support for specific algorithms.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1332928201902788740)** (3 messages): 

> `System prompts for leaderboard models, Gorilla GitHub repository` 


- **Seeking System Prompts for Non-Function Calling Models**: A member inquired about the location of system prompts used in the leaderboard for models that do not support function calling.
   - Another member promptly shared a [link to the GitHub repository](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py#L3-L18) containing the relevant code for reference.
- **Resource Link for Prompt Details**: The shared GitHub link directs users to the [gorilla function call leaderboard's code](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py) where system prompts are defined.
   - The repository is focused on training and evaluating LLMs for function calls, providing ample resources for users.



**Link mentioned**: <a href="https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py#L3-L18">gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py at main · ShishirPatil/gorilla</a>: Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla

  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1333198192572895363)** (1 messages): 

> `2025 Crystal Ball panel discussion, Real-time data processing, AI and data streaming technologies, Industry leaders and insights` 


- **2025 Crystal Ball: Shaping the Future with AI**: Join industry leaders on **January 28** for the panel **2025 Crystal Ball: Real-Time Data and AI**, focusing on the transformative role of real-time data in enhancing AI capabilities.
   - *Without real-time data processing, AI's true potential will remain untapped*, highlighting the need for technologies that facilitate low-latency predictions.
- **Expert Insights from AI Leaders**: Panelists include **Rayees Pasha (RisingWave Labs)**, **Sijie Guo (StreamNative)**, and **Chang She (LanceDB)**, discussing key advancements in AI and data streaming.
   - The discussion will cover the **evolving relationship** between AI and real-time data, along with predictions for upcoming challenges and advancements by 2025.
- **Deep Dive into Real-Time Processing Technologies**: Expect a **deep dive into key technologies** such as Apache Iceberg, emphasizing their role in revolutionizing data infrastructure and AI efficiency.
   - Panelists will explain how cutting-edge stream processing and real-time analytics can deliver **groundbreaking business impacts** across various industries.
- **Unlocking AI's True Potential**: Without real-time data processing capabilities, AI's true potential remains unfulfilled, emphasizing the need for innovations in data processing.
   - This event aims to dissect how AI systems are leveraging real-time data to address real-world challenges and unlock new opportunities.



**Link mentioned**: <a href="https://www.meetup.com/streaming-stories/events/305736950/">2025 Crystal Ball: Real-Time Data and AI, Tue, Jan 28, 2025, 9:00 AM   | Meetup</a>: **About**Look into the future of data streaming and AI at 2025 Crystal Ball: Real-Time Data and AI panel. Without a doubt, AI is to shape our future in the years to come.

  

---


### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1333489291195846758)** (1 messages): 

> `Paper Reading Club, Discord Events` 


- **Paper Reading Club meets this week!**: The **Paper Reading Club** is getting together again this week. Check out the details for the event on [Discord](https://discord.com/events/1089876418936180786/1329844319703662664)!
- **Reminder about Discord Events**: Don't forget to explore exciting events happening on **Discord**, including the Paper Reading Club. Join the conversation and participate in the community activities this week!


  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

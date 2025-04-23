---
id: f98b3a6c-3c56-432d-a822-c9a4ed105451
title: '[AINews] lots of small launches'
date: '2025-02-27T04:09:12.976879Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-lots-of-small-launches
---

<!-- buttondown-editor-mode: plaintext -->**a quiet day.**

> AI News for 2/25/2025-2/26/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**221** channels, and **7040** messages) for you. Estimated reading time saved (at 200wpm): **725 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

- [GPT 4.5 is coming this week](https://x.com/steph_palazzolo/status/1894785791018332505?s=46)
- [Elicit announced a Series A and their own Deep Research](https://x.com/elicitorg/status/1894772293752266846?s=46)
- [Alexa+ was refreshed with Amazon Nova and Anthropic Claude](https://x.com/mikeyk/status/1894783669920817321?s=46)
- [Cloudflare launched an agents sdk](https://x.com/threepointone/status/1894399506277376369?s=46)
- [FLORA launched their Krea competitor](https://x.com/weberwongwong/status/1894794612398792974?s=46)
- [Elevenlabs launched ASR](https://x.com/matistanis/status/1894824212382257427?s=46)
- [Perplexity launched a Deep Research API](https://x.com/aravsrinivas/status/1894471526449385687?s=46) (also valued at 15b)
- [Inception labs launched a production language diffusion model](https://x.com/InceptionAILabs/status/1894847919624462794)



---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**AI Model Updates & Releases, focusing on new models, features, and versions**

- **GPT-4o Advanced Voice Preview for Free Users**: [@OpenAI](https://twitter.com/OpenAI/status/1894495906952876101) announced the rollout of **Advanced Voice powered by GPT-4o mini** for all **free ChatGPT users**, offering a daily preview across platforms with a natural conversation pace and cost-effectiveness.  [@OpenAI](https://twitter.com/OpenAI/status/1894495908366356607) also detailed continued access for **Plus and Pro users**, with **Plus users** retaining access to **Advanced Voice powered by 4o** with a **5x higher daily rate limit** than free users, and **Pro users** maintaining **unlimited access** and **higher limits for video and screensharing**.
- **Claude 3.7 Sonnet Release and Performance**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1894840263379689490) reported that **Claude 3.7 Sonnet** has claimed the **#1 spot in WebDev Arena**, surpassing **Claude 3.5 Sonnet** with a **+100 score jump**.  [@alexalbert__](https://twitter.com/alexalbert__/status/1894807853371990087) mentioned a **more token-efficient tool use implementation** for **Claude 3.7 Sonnet**, utilizing **14% less tokens** and showing improved performance, accessible via the beta header `"token-efficient-tools-2025-02-19"`.
- **DeepSeek R1 Inference Platform and DeepGEMM**: [@togethercompute](https://twitter.com/togethercompute/status/1894515568088412198) highlighted that **DeepSeek-R1**, with **671 billion parameters**, requires an inference platform to maximize **NVIDIA Blackwell GPU** utilization, with **Together Inference** optimizing GPU efficiency for **DeepSeek-R1**. [@reach_vb](https://twitter.com/reach_vb/status/1894626368702304617) announced **DeepGEMM** by **DeepSeek**, a lightweight **CUDA-based library** for efficient **FP8 GEMMs** on **NVIDIA Hopper tensor cores**, outperforming expert-tuned libraries and achieving up to **2.7x speedups** in **DeepSeek-V3/R1 inference tasks**. [@deepseek_ai](https://twitter.com/deepseek_ai/status/1894553164235640933) officially introduced **DeepGEMM** as part of their **Open Source Week**, noting its performance of **1350+ FP8 TFLOPS on Hopper GPUs**, JIT compilation, and core logic within **~300 lines**.
- **Perplexity Voice Mode and Deep Research API**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1894792092284789173) announced the release of a **new Perplexity Voice Mode**, incorporating **real-time voice and information** across languages, available on iOS with Android coming soon.  [@AravSrinivas](https://twitter.com/AravSrinivas/status/1894820042816307467) also mentioned the **Deep Research API** as part of recent updates from Perplexity.
- **Grok 3 API with 1M Context**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1894733474239582465) mentioned the upcoming **Grok 3 API** with **1M context**.

**AI Tools, Libraries, and Datasets, covering frameworks, code, and resources**

- **DeepGEMM Open Source Library for FP8 GEMMs**:  [@deepseek_ai](https://twitter.com/deepseek_ai/status/1894553164235640933) open-sourced **DeepGEMM**, a **CUDA-based library** for efficient **FP8 GEMMs**, highlighting its performance and concise codebase. [@danielhanchen](https://twitter.com/danielhanchen/status/1894554391140864240) also highlighted **DeepGEMM**, noting its **JIT compilation** and efficiency for **FP8 matrix multiplication**.
- **OpenEvals OSS Repo for LLM Evaluation**: [@LangChainAI](https://twitter.com/LangChainAI/status/1894821108018262297) announced **OpenEvals**, a new **OSS repo** containing prebuilt evaluators to simplify adding evals to LLM applications, with Python and JS support.
- **LangGraph Swarm for Multi-Agent Systems**: [@LangChainAI](https://twitter.com/LangChainAI/status/1894795982379848168) introduced **LangGraph Swarm**, a lightweight library for building **swarm-style multi-agent systems** with **LangGraph**, enabling agent collaboration and customizable communication tools.
- **LangGraph Platform Custom Routes**: [@LangChainAI](https://twitter.com/LangChainAI/status/1894795878504055053) announced **Custom Routes** in the **LangGraph Platform**, allowing extension with custom HTTP endpoints for building full-stack AI apps in Python with a single backend.
- **P2L (Prompt-to-Leaderboard) for Real-time LLM Leaderboards**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767009977811256) introduced **Prompt-to-leaderboard (P2L)**, an open-source system that trains an LLM to generate prompt-specific leaderboards, based on **2M human preference votes** from **Chatbot Arena**.  [@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767022791438490) shared links to the **P2L paper and code**, emphasizing its open-source nature.
- **Tahoe-100M Dataset Release by Vevo Therapeutics**: [@sarahcat21](https://twitter.com/sarahcat21/status/1894784421611680209) highlighted **Vevo Therapeutics' OSS release** of the **Tahoe-100M dataset**, aimed at unlocking high-quality data for FM-driven drug development.
- **Meta PARTNR Dataset and Code for Embodied Multi-Agent Tasks**: [@AIatMeta](https://twitter.com/AIatMeta/status/1894524602854117781) released the **Meta PARTNR dataset and code**, a benchmark for planning and reasoning in embodied multi-agent tasks, used in their recent robotics demos. [@AIatMeta](https://twitter.com/AIatMeta/status/1894524604900938078) provided a direct link to the dataset and code.
- **OpenEvals Repo for LLM Evaluation**: [@LangChainAI](https://twitter.com/LangChainAI/status/1894821108018262297) announced the release of **OpenEvals**, an open-source repository with pre-built evaluators to help users easily evaluate LLMs.

**Research, Analysis, and Benchmarks, covering evaluations, performance, and insights**

- **SWE-RL: Meta's RL for Software Evolution Benchmark**: [@_akhaliq](https://twitter.com/_akhaliq/status/1894584315352076608) reported on **Meta's SWE-RL**, a method using **Reinforcement Learning** on **Open Software Evolution** data, achieving a **41.0% solve rate on SWE-bench Verified** with **Llama3-SWE-RL-70B**, comparable to **GPT-4o** for medium-sized models. [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1894596772804350016) also highlighted **Meta's SWE-RL**, achieving state-of-the-art performance on **SWE-bench Verified** with **Llama 3**.
- **Prompt-to-Leaderboard (P2L) Performance Analysis**:  [@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767012490174779) detailed the performance of **P2L-router**, achieving **#1 on Chatbot Arena in Jan 2025** with a score of **1395**, and cost-constrained P2L models reaching the Pareto frontier. [@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767018014130204) further explained **P2L's use for model weakness analysis**, identifying strengths and weaknesses across domains, and [@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767014767673744) highlighted its use for domain-specific leaderboards, enabling adaptive category rankings.
- **Anthropic's Risk Forecasting Research**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1894495059954860055) announced new research on **forecasting rare language model behaviors**, predicting deployment risks with limited test data, and [@AnthropicAI](https://twitter.com/AnthropicAI/status/1894495065612939629) noted that their forecasts accurately predicted misuse and misalignment risks in experiments.
- **MoBA (Mixture of Block Attention) for Long-Context Tasks**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1894711958353629617) reported on **MoBA (Mixture of Block Attention)** from **Kimi Moonshot**, improving long-context task handling and achieving **6.5x speedup** over full attention for **1M tokens**.
- **FFTNet: FFT-based Alternative to Self-Attention**: [@omarsar0](https://twitter.com/omarsar0/status/1894757821587296614) summarized a paper presenting **FFTNet**, replacing self-attention with **adaptive spectral filtering** using **FFT**, reducing complexity to **O(n log n)** and showing competitive performance on benchmarks.
- **Linear Probes vs. SAEs (Sparse Autoencoders) in Interpretability Research**: [@NeelNanda5](https://twitter.com/NeelNanda5/status/1894749262757634405) discussed research finding that **linear probes outperformed SAEs** across **5 regimes and 100+ datasets**, a negative update on SAEs for interpretability.

**Industry and Company Announcements, covering partnerships, funding, and events**

- **Amazon Alexa+ Powered by Claude**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1894798008623026503) announced **Claude's partnership with Amazon** to power the next-generation **Alexa+ AI assistant**. [@_philschmid](https://twitter.com/_philschmid/status/1894816750895575161) detailed the features of **Alexa+**, including Amazon Nova and Anthropic Claude integration, new "Tool" APIs, browser use, and a subscription model.
- **Elicit Raises $22M Series A and Launches Elicit Reports**: [@Fraser](https://twitter.com/Fraser/status/1894779613210878434) announced **Spark Capital co-leading a $22M investment in Elicit**, with [@elicitorg](https://twitter.com/elicitorg/status/1894772293752266846) launching **Elicit Reports**, a research tool aimed at automating scientific understanding.
- **Figure Robotics Scaling Humanoid Robot Production**: [@adcock_brett](https://twitter.com/adcock_brett/status/1894782815981711810) announced **Figure's ramp-up to ship humanoid robots** at unprecedented levels in **2025**, highlighting their **Helix AI advances** and customer use cases with BMW. [@adcock_brett](https://twitter.com/adcock_brett/status/1894781636153405870) stated that **Helix** enables robots to scale with a single neural network, reducing customer use case development time significantly.
- **Google Gemini Code Assist Free Version**: [@Google](https://twitter.com/Google/status/1894816225575731366) announced the global availability of a **free version of Gemini Code Assist** for individuals with high usage limits.
- **Perplexity Inbound VC Offers at $15B**: [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1894785791018332505) reported that **Perplexity is receiving inbound VC offers at $15B**, though they are unlikely to accept, highlighting VC interest in revenue-generating AI firms.
- **DeepSeek API Off-Peak Discounts**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1894710448676884671) announced **off-peak discounts** on the **DeepSeek API Platform** during **16:30–00:30 UTC daily**, with **50% off DeepSeek-V3** and **75% off DeepSeek-R1**.
- **Hugging Face Enterprise Upgrade Growth**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1894778750631371149) announced that **over 2,000 organizations** have upgraded to **Hugging Face Enterprise**, including major companies across various industries.
- **MLSYS 2025 Young Professionals Symposium Call for Abstracts**: [@realDanFu](https://twitter.com/realDanFu/status/1894576091777700128) announced a call for abstracts for the **MLSys 2025 Young Professionals Symposium** on **May 12 in Santa Clara**, with a deadline of **April 7**.
- **Perplexity Developer Event in SF on March 17th**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1894537361033511094) announced a **developer event** at **Perplexity's SF office on March 17th**, inviting developers to meet the API team and share feedback.

**Opinions and Discussions, covering broader AI perspectives and commentary**

- **AI Engineering Focus Shift**: [@nrehiew_](https://twitter.com/nrehiew_/status/1894513333719515452) suggested that **AI engineering** should be **50% standard SWE, 10% TPOT user** for model awareness, and **40% UX**, emphasizing that apps don't need to be chatbots.
- **OpenAI's Market Leadership and Challenges**: [@madiator](https://twitter.com/madiator/status/1894611101884846601) discussed **OpenAI's market position**, highlighting their leadership, brand recognition, and infrastructure, but also noting challenges like high costs and competition, while crediting them for realizing scaling, data acquisition, and productionizing RL finetuning.
- **LLMs and Codebase Understanding**: [@qtnx_](https://twitter.com/qtnx_/status/1894843415529181665) argued against the concern that LLMs will lead to not understanding codebases, comparing it to working in teams where understanding others' code is already necessary.
- **Cursor vs. DIY Coding**: [@jxmnop](https://twitter.com/jxmnop/status/1894830128082940182) cautioned about the **mental cost of outsourcing code to Copilot/Cursor**, likening it to a mortgage and suggesting that doing everything oneself might be more efficient long-term beyond simple autocomplete.
- **Importance of Model Training and Open Source**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1894831939305218559) emphasized that "**The model is the product!**" and that long-term product success requires learning to train models based on open-source.
- **ChatGPT Moment Definition**: [@_aidan_clark_](https://twitter.com/_aidan_clark_/status/1894506681025138799) clarified that the "**ChatGPT moment**" was when people realized chatbots were useful, not when the tech became feasible.
- **AI Safety and Deals with AIs**: [@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/1894515270108283368) discussed the increasing overlap of **AI safety** with **economics and psychology**, mentioning a podcast discussing making deals with AIs.
- **AI and Misinformation Skepticism**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1894827528973348919) argued that fears of AI-generated misinformation have been overblown, suggesting AI media has fostered skepticism and reliance on social verification.
- **In-Context Learning and Emergent Abilities**: [@giffmana](https://twitter.com/giffmana/status/1894524234392625199) discussed research on in-context learning and emergent abilities, noting it confirms generalization in large models, reframing "backdoors" as "conditioning".
- **Critique of AI Research Data Access and Interest**: [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1894756271489724772), [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1894756269195506050), [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1894756265231880346) expressed concern about the lack of access to training data in AI research and the rush to claim OOD performance without proper data analysis.
- **Transformers with Recursive Blocks Idea**: [@jxmnop](https://twitter.com/jxmnop/status/1894567121793028158) proposed building **transformers with recursive blocks** instead of typical blocks, suggesting potential expressiveness gains at a cost of GPU unfriendliness.
- **MLP Dimensionality in Transformers Question**: [@jxmnop](https://twitter.com/jxmnop/status/1894527828630147562) questioned why **MLPs in transformers** project to larger dimensions and back down, wondering why weight matrices can't be square.
- **Scientific Understanding Lagging Model Deployment**: [@_jasonwei](https://twitter.com/_jasonwei/status/1894821797000028357) observed that scientific understanding of models often lags behind deployment speed in competitive product landscapes, but ablation studies can be valuable.
- **RLHF and Model Misalignment**: [@jd_pressman](https://twitter.com/jd_pressman/status/1894493541591871969) hypothesized that tuning **GPT4o to write bugged code** leads to broad misalignment due to **RLHF** preferences becoming central.
- **Dunbar's Number as Dunbar's Brick Wall**: [@DavidSHolz](https://twitter.com/DavidSHolz/status/1894628680183550393) commented that **Dunbar's number** feels more like a "**brick wall**".
- **"Heteroscadasticity" Term Critique**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1894852822706872658) humorously critiqued the term "**Heteroscadasticity**" as unintuitive and Kung Fu Panda-esque.
- **Importance of Composition and Abstraction in ML**: [@lateinteraction](https://twitter.com/lateinteraction/status/1894719046760968550) argued for the importance of **composition and abstraction** in computer science and ML, noting their absence in modern ML's self-perception due to implementation-tied abstractions.
- **Late Interaction vs. Multi-Vector Terminology**: [@lateinteraction](https://twitter.com/lateinteraction/status/1894696983077785980) discussed the terminology of "**late interaction**" vs. "**multi-vector**" for ColBERT-like methods, arguing that "**late interaction**" is more accurate as the mechanism isn't just about multiple vectors but learnability and scoring functions.
- **Need for Fourth Conditioning Mechanism Beyond Training, Concatenation, Retrieval**: [@lateinteraction](https://twitter.com/lateinteraction/status/1894669414454485033) questioned if a fourth conditioning mechanism is needed beyond training, concatenation, and retrieval for LMs.
- **Late Alignment Importance**: [@lateinteraction](https://twitter.com/lateinteraction/status/1894666144055005693) emphasized the need for "**late alignment**" after facts are present, in both IR and DSPy/RL, cautioning against precrastination.
- **Granular Scoring Superiority**: [@lateinteraction](https://twitter.com/lateinteraction/status/1894662639839842346) highlighted the superior generalization of "**granular scoring**" over dense dot products in challenging tasks, advocating for late interaction.
- **AI-Powered Interpretation Debate**: [@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1894779433955037284) summarized his participation in a debate arguing that **AI-powered interpretation will eventually replace human interpretation**, citing compute trends and AI advancements.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepGEMM Offers Efficient FP8 General Matrix Multiplications**

- **DeepSeek Realse 3th Bomb! DeepGEMM a library for efficient FP8 General Matrix** ([Score: 514, Comments: 105](https://reddit.com/r/LocalLLaMA/comments/1iybcnl/deepseek_realse_3th_bomb_deepgemm_a_library_for/)): **DeepGEMM** is a library focused on efficient **FP8 General Matrix Multiplications (GEMMs)** with detailed scaling, as introduced in **DeepSeek-V3**. The library can be accessed through [this GitHub link](https://github.com/deepseek-ai/DeepGEMM).
  - **DeepGEMM's Performance and Impact**: DeepGEMM's FP8 matrix multiplication can improve performance by **2.7x** compared to NVIDIA’s CUDA library, allowing training and serving models more cost-effectively. The library's portability and JIT compilation are highlighted, with potential implications for optimizing performance on various architectures, although currently limited to **NVIDIA Hopper tensor cores**.
  - **Industry Implications and Competitiveness**: The release challenges the perceived dominance of companies like **NVIDIA** and **OpenAI**, with discussions about the potential for **Huawei's 910C** to compete with NVIDIA's H100s. Concerns about the sustainability of NVIDIA's market position are raised, with speculation about the impact on their valuation and the broader competitive landscape.
  - **Community Reactions and Potential**: The community expresses excitement about the potential of DeepGEMM, with discussions on its impact on model training costs and efficiency. There are doubts about the feasibility of achieving significant cost reductions in training, but the availability of benchmarks and speedup factors helps address some skepticism.


**Theme 2. Nvidia Gaming GPUs with Increased VRAM Enter Chinese Cloud Market**

- **[RTX 4090 48GB](https://www.reddit.com/gallery/1iy7e4x)** ([Score: 653, Comments: 221](https://reddit.com/r/LocalLLaMA/comments/1iy7e4x/rtx_4090_48gb/)): The author acquired an **Nvidia RTX 4090** with **48GB of RAM** from eBay in Canada and is open to suggestions for testing its capabilities and answering any related questions.
  - Users are curious about the **price** of the RTX 4090 with 48GB of RAM, with estimates ranging from **$2.85k to $3.3k USD**, and some expressing concern over the **current GPU market prices** being higher than MSRP. [Best Value GPU](https://bestvaluegpu.com/en-eu/history/new-and-used-rtx-4090-price-history-and-specs) provides a historical price comparison.
  - There is a technical discussion regarding the **verification** of the GPU's authenticity, with suggestions to extract the **vbios** and run **GPU benchmarks** to ensure it is not a modified RTX 8000. Users also discuss the **power consumption** and **cooling challenges** of using multiple GPUs, with some opting to power limit their cards to **90%**.
  - A user shared a **Python script** to test the VRAM capacity using **torch**, allocating memory in **100MB chunks** to ensure the full 48GB is usable. The script helps identify if the card is genuine and checks for any **memory corruption** during allocation.


- **[Nvidia gaming GPUs modded with 2X VRAM for AI workloads — RTX 4090D 48GB and RTX 4080 Super 32GB go up for rent at Chinese cloud computing provider](https://www.tomshardware.com/pc-components/gpus/nvidia-gaming-gpus-modded-with-2x-vram-for-ai-workloads)** ([Score: 265, Comments: 45](https://reddit.com/r/LocalLLaMA/comments/1iy7k6b/nvidia_gaming_gpus_modded_with_2x_vram_for_ai/)): **Chinese cloud computing providers** are offering **Nvidia gaming GPUs** with modified **VRAM** for AI workloads, specifically the **RTX 4090D** with **48GB** and the **RTX 4080 Super** with **32GB**. These GPUs are available for rent, providing enhanced capabilities for AI applications.
  - Discussions highlighted the **modification of Nvidia GPUs** for AI workloads in China, with users pointing out legal and ethical issues surrounding such practices. Some argued that modifying hardware is legal if purchased outright, while others noted potential **Nvidia ToS violations** when renting modded hardware, emphasizing Nvidia's restrictions to protect their high-margin enterprise products.
  - The **price and availability** of these modified GPUs were a focal point, with comments noting that renting a **32GB RTX 4080** for **$0.03 per hour** seems too low, suggesting potential currency confusion. A user corrected the rental cost, indicating it should be around **$0.7 per hour**, while another highlighted the **$2,500** cost for a **48GB 4090D** as cheaper than local second-hand options.
  - Some users questioned the legitimacy of these modified GPUs, with concerns about **scams** and the reliability compared to official **RTX 6000 ADA** cards. Others criticized Nvidia's strategy of offering lower VRAM consumer GPUs to protect their enterprise card sales, suggesting that the **Chinese market** is catering to unmet global demand for higher VRAM cards.


**Theme 3. DeepSeek API Platform Introduces Off-Peak Discounts**

- **[Starting today, enjoy off-peak discounts on the DeepSeek API Platform from 16:30–00:30 UTC daily](https://i.redd.it/cgapkix5ygle1.jpeg)** ([Score: 398, Comments: 78](https://reddit.com/r/LocalLLaMA/comments/1iylebm/starting_today_enjoy_offpeak_discounts_on_the/)): **DeepSeek API** has announced off-peak discounts, effective daily from **16:30 to 00:30 UTC**, providing a **50% discount on DeepSeek-V3** and a **75% discount on DeepSeek-R1** pricing for specific token usage. The announcement includes a clear breakdown of standard and discounted prices for input (cache hit, cache miss) and output tokens, presented in a professional and easy-to-read format.
  - **DeepSeek API Reliability Concerns**: Users express concerns about the reliability of DeepSeek, noting past issues with server availability and the need for stable service to ensure effective use during important tasks. Some users report recent improvements in stability, suggesting that the service may have resolved previous issues.
  - **Pricing and Usage Dynamics**: Discussions highlight the competitive pricing of **DeepSeek R1** at **$0.135/Mtok**, with users comparing the cost-effectiveness of using APIs versus running models locally. The off-peak discounts are seen as a strategic move to balance server load globally, encouraging usage during less busy hours to manage demand spikes.
  - **Market and Competitive Positioning**: The conversation touches on the broader market implications, with users noting the potential impact of DeepSeek's pricing strategy on competitors and the importance of continued innovation to remain competitive. The open-sourcing of **Hopper inference efficiency** is seen as a positive step that could influence pricing trends across other providers.


**Theme 4. TinyR1-32B Outperforms Official R1 Distills**

- **[TinyR1-32B-Preview (surpassing official R1 distill 32B performance)](https://huggingface.co/qihoo360/TinyR1-32B-Preview)** ([Score: 126, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1iybgj2/tinyr132bpreview_surpassing_official_r1_distill/)): **TinyR1-32B-Preview** is noted for its superior performance compared to the official **R1 distill 32B** model. This highlights advancements in efficiency or design that allow it to outperform its predecessor.
  - Users express interest in the **distills of the V3 model** with specific mentions of **200B, 100B, 70B, 30B** MoEs, indicating a demand for more advanced, efficient models. The **TinyR1-32B-Preview** is recognized for its open-source nature and contributions from the **360 team and PKU**.
  - **Qihoo360** is criticized for its reputation on the Chinese internet, with allegations of using **LLM-related rumors** to inflate stock prices. This reflects skepticism about the company's motives and practices.
  - There are concerns about the model's behavior, such as issues with the **EOS token** causing unexpected language shifts and loops, particularly in **Chinese** and **Arabic**, which suggests potential bugs in the model's response handling.


**Theme 5. Perplexity's Plan to Fork Chrome for AI Browsing**

- **[Perplexity is forking Chrome](https://i.redd.it/ubxe59mr1fle1.png)** ([Score: 402, Comments: 97](https://reddit.com/r/LocalLLaMA/comments/1iyfvhb/perplexity_is_forking_chrome/)): **Perplexity AI** is planning to fork **Chrome** by developing a new browser called **Comet**. They are hiring a **Browser C++ Engineer** with experience in the **Chromium codebase** and a passion for user experience and UI design, with positions available in the **New York Metro Area** and **San Francisco Bay Area**.
  - There is skepticism about **Perplexity AI's** approach, with criticisms that they may be simply reskinning **Chrome** and adding an AI assistant, rather than innovating significantly. Some users express distrust towards the CEO, citing past incidents where **Perplexity** allegedly used resources like **Google Search** results without acknowledgment.
  - Discussions highlight the reliance on **open source** projects like **Chromium**, with some defending the practice as beneficial for streamlining development and compatibility. Others criticize the lack of originality, noting that most third-party browsers are based on **Chromium**.
  - There is debate over the ethical considerations of using existing technologies, with some arguing that **Perplexity** offers valuable services by making AI features more accessible. However, others argue that they should acknowledge the foundational work of predecessors more openly.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. Claude 3.7 Disruption in AI Development and Personal Assistance**

- **Claude 3.7 saved my marriage!!!** ([Score: 422, Comments: 50](https://reddit.com/r/ClaudeAI/comments/1iykhh3/claude_37_saved_my_marriage/)): **Claude 3.7** is praised for its unexpected effectiveness in personal assistance, with a user claiming it helped them navigate a challenging marital situation. Despite the marriage ending, the user found solace in interacting with **Claude 3.7**, humorously suggesting a new "marriage" with the AI.
  - Users expressed skepticism about **Claude 3.7**, with concerns over its advice quality, especially in sensitive situations like relationships. One user noted that **Grok**, a component of Claude, suggested harmful actions when faced with relationship issues, indicating potential risks in its guidance.
  - Some commenters humorously exaggerated **Claude 3.7's** capabilities, claiming it helped them with improbable feats such as curing cancer or staging political coups, while others questioned the authenticity of positive posts, suspecting them to be paid promotions for **Sonnet 3.7**.
  - There was a mixed reaction regarding **Claude 3.7's** performance compared to **Sonnet 3.5**, with some users not noticing significant improvements, while others mentioned specific use cases where it was beneficial, such as personal relationship management and financial gains.


- **OMG.. You can build ANYTHING with 3.7 it's literal. magic.** ([Score: 308, Comments: 131](https://reddit.com/r/ClaudeAI/comments/1iyf7gx/omg_you_can_build_anything_with_37_its_literal/)): The post author expresses significant enthusiasm for **Claude 3.7 Sonnet**, highlighting its effectiveness in application development compared to **GPT-4o** and **o1**, which struggled with complex tasks. They successfully built an AI agent and a complex workflow in a single prompt, leading them to replace their company's **OpenAI API** subscription with Claude, citing its superior performance and ease of use.
  - Many commenters express skepticism about the post's authenticity, suggesting it might be a **paid advertisement** or part of a **Claude hype bot** campaign. Users like **Old-Fox-137** and **Naghen** question the lack of specific instructions and the repetitive nature of the praise for **Claude 3.7**.
  - Some users, such as **jan04pl** and **iKonstX**, share mixed experiences with **Claude 3.7**, noting its limitations in handling complex codebases and simple tasks, respectively. While it saves time and can generate significant portions of code, it still requires manual intervention and troubleshooting.
  - There is a humorous and exaggerated comment from **MaximumGuide** about **Claude 3.7**'s capabilities, which includes fictional and fantastical elements like creating a quantum computer and pizza trees, highlighting the hyperbolic tone of some discussions.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. AI IDE Showdown: Cursor Flexes, Windsurf Waffles**

- **Cursor Agents Get Pythonic Power-Up**: [Equip Cursor Agents with Python Tools](https://discord.com/channels/1074847526655643750/1074847527708393565/1344036328924254218):  **Cursor Agents** now wield local Python tools via CLI, boosting agent capabilities and allowing integration with external utilities like `yt-dlp`. Users recommend structuring agent plans into manageable story points for effective task execution.
- **Windsurf Users Drowning in Credit Costs**: [Claude 3.7 Gobbles Credits in Windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1344090394283216956): **Claude 3.7** in **Windsurf** is burning through credits at an alarming rate, even for basic tasks, sparking concerns about inefficient implementation and excessive tool calls, with users reporting hundreds of credits vanishing quickly. Some users are speculating **Windsurf's** specific implementation is less efficient than direct **Claude 3.7** usage.
- **Cursor's Cloudless Code Augment Catches Eyes**: [Augment Code AI requires cloud upload](https://discord.com/channels/1074847526655643750/1074847527708393565/1344036328924254218): **Cursor's Augment Code AI** feature, requiring repo uploads to their cloud, raises data privacy eyebrows. Engineers are exploring cloud-bypass alternatives like **repo prompt** with **Grok 3** or **AI Studio** for codebase analysis.

**Theme 2. Claude 3.7: Leaks, Lies, and Load Balancing**

- **Claude Code Source Spills onto GitHub**: [Claude Code leaked on Github](https://github.com/dnakov/claude-code/tree/main):  Source code for **Claude-code** surfaced on GitHub, extracted from source maps, due to **Anthropic's** oversight. Speculation abounds on repurposing it for other models, while users debate the exorbitant **$10-per-20-minutes** cost of **Claude Code**.
- **Sonnet 3.7 Identity Crisis: Opus Impersonator?**: [Claude 3.7 Sonnet Has a Crisis](https://discord.com/channels/1047197230748151888/1343923864492838972/1343923864492838972): **Claude 3.7 Sonnet** sometimes misidentifies as **Claude 3 Opus**, likely due to training data quirks or naming confusion. A bug ticket is filed to investigate this split personality issue.
- **OpenRouter's Reasoning Parameter Unlocks Model Harmony**: [OpenRouter Debuts Cross-Model Reasoning Standard](https://openrouter.ai/docs/use-cases/reasoning-tokens): **OpenRouterAI** introduced a **cross-model reasoning standard**, enabling unified configuration of reasoning settings across **OpenAI**, **Anthropic**, and other models via their API. The new `reasoning` parameter streamlines model usage regardless of internal API differences.

**Theme 3. DeepSeek's Deep Dive: Price Cuts and Performance Peaks**

- **DeepSeek Slashes API Prices to Rock Bottom**: [DeepSeek Slashes API Pricing in Off-Peak Discount](https://x.com/Sino_Market/status/1894682095706128430): **DeepSeek** dramatically cut [API pricing](https://x.com/Sino_Market/status/1894682095706128430), offering up to **75% off** during off-peak hours (16:30-00:30 UTC). Discounts include **50% off DeepSeek-V3** and **75% off DeepSeek-R1**, continuing DeepSeek's aggressive pricing strategy.
- **DeepGEMM Kernel Unleashes FP8 Fury**: [DeepSeek Shows Off fp8 Kernels](https://x.com/deepseek_ai/status/1894553164235640933?s=46&t=stOPrwZiN_fxSK0RuC8Flg): **DeepSeek** unveiled **DeepGEMM**, an **fp8** GEMM library, supporting both dense and MoE GEMMs, powering **V3/R1** training and inference.  **DeepGEMM** achieves over **1350+ FP8** TFLOPS on Hopper GPUs, outpacing expert-tuned kernels across various matrix sizes.
- **R2-D2 Arrives Early, Outshines R1**: [Deepseek R2 arriving early](https://techstartups.com/2025/02/25/deepseek-is-launching-its-next-gen-r2-ai-model-poised-to-surpass-r1-and-shock-the-world-once-again/): **Deepseek R2** is launching ahead of schedule, potentially exceeding **R1** in coding and reasoning abilities, even beyond English. The company aims for enhanced coding capabilities and broader reasoning skills with this release.

**Theme 4. Open Source LLM Dev: High School Hustle & Hardware Hangups**

- **Teenager's LLM Code Faces Open Source Reality Check**: [High Schooler's LLM Code Faces Open Source Reality](https://discord.com/channels/954421988141711382/954421988783444043/1344039290648268820): A high school student's attempt to sell code for **local LLM training** met community pushback, highlighting competition from free, open-source alternatives like [Unsloth](https://github.com/unslothai/unsloth). The developer opted to **open source** the project instead.
- **Framework's AMD Desktops Spark CUDA Conflict**: [Framework Desktop Sparks CUDA Debate](https://discord.com/channels/1179035537009545276/1179035537529643040/1344036212813467668): Framework's giveaway of **100** desktops for AI development, featuring AMD-only systems, ignited debate over the lack of **CUDA** support. While adequate for inference with **128GB** RAM, the absence of `bitsandbytes` for AMD may hinder model development.
- **DeepGEMM Cracks the Hardware Moat**: [Deepseek cracks DeepGEMM Kernel](https://discord.com/channels/729741769192767510/747850033994662000/1344036738208895099):  **Deepseek's DeepGEMM** release impressed engineers, optimizing efficiency within hardware constraints like **H800 limitations**.  The open-source Gemm kernel, leveraging **TMA**, reinforces the sentiment that hardware efficiency is becoming the primary competitive edge in AI.

**Theme 5. Perplexity's Push & OpenAI's API Expansions**

- **Perplexity's Voice Mode Finally Finds Its Voice**: [Perplexity's Voice Cracks the Code](https://cdn.discordapp.com/attachments/1047204950763122820/1344393474027421706/Iac2em_OTTWx60h6.mp4?ex=67c0bf7d&is=67bf6dfd&hm=29f3e05084083471219f93c750de0678bdd2d9f1f647780432abcb6a10576dbe&): **Perplexity AI** launched a new **voice mode** in its iOS app, enabling real-time audio question answering, as shown in [this demo video](https://cdn.discordapp.com/attachments/1047204950763122820/1344393474027421706/Iac2em_OTTWx60h6.mp4?ex=67c0bf7d&is=67bf6dfd&hm=29f3e05084083471219f93c750de0678bdd2d9f1f647780432abcb6a10576dbe&). Android and Mac versions are in the pipeline, though some users find it still trailing behind competitors like **Microsoft Copilot** or **ChatGPT**.
- **OpenAI Assistants API Opens File Search Files**: [File Search Comes to OpenAI Assistants API](https://platform.openai.com/docs/assistants/tools/file-search): **OpenAI** added [file search](https://platform.openai.com/docs/assistants/tools/file-search) to its **Assistants API** for **o3-mini** and **o1** models, enhancing information retrieval from uploaded documents. Assistants can now access and utilize user-provided file data more effectively.
- **GPT-4.5 Whispers Grow Louder**: [Whispers of GPT-4.5 Launch](https://openai.com/blog/new-ai-models): Rumors of **GPT-4.5's** imminent release intensify, with speculation pointing to late February or early March 2025, fueled by [Sam Altman's statements](https://openai.com/blog/new-ai-models) and alleged beta app sightings. Hints of a *GPT-4.5 Research Preview* are reportedly appearing for **OpenAI Pro** users.

---

# PART 1: High level Discord summaries




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Augment Code AI requires cloud upload**: Members noted that using **Augment Code AI** requires granting access and uploading your repos to their cloud, raising concerns about data privacy.
   - One member suggested using **repo prompt** with **Grok 3** or **AI Studio** as a possible alternative for codebase assessment, bypassing the need to upload to a third-party cloud.
- **Zed Editor sacrifices Terminal Execution**: While the **Zed Editor** is praised for being lightweight and utilizing **Sonnet 3.7**, it lacks **Cursor's** functionality for executing terminals.
   - One member emphasized the significance of terminal execution, stating *The fact that Cursor can execute terminals allows for a lot of opportunities. Exploit it.*
- **Equip Cursor Agents with Python Tools**: Members discussed the capability to install Python tools locally and invoke them via CLI using **Cursor Agents**, enhancing agent functionality.
   - A user advised setting up agents with a detailed plan, suggesting that *each step in the plan should be equivalent to ~1 story point as if it were a jira ticket*.
- **Cursor Chat Summary Declared Disaster**: Users reported that **Cursor's chat summary feature** is deeply flawed, citing context selection by opaque algorithms resulting in irrelevant changes.
   - One member questioned its effectiveness, asking *If the full chat summary looks like that, what does the chat summary look like when you pass the, what? 10k context window?*
- **Claude-code Source Leaked**: The source code for **Claude-code** was extracted from source maps and made available on [GitHub](https://github.com/dnakov/claude-code).
   - Members speculated on the potential for adapting it to other models, with one wondering *How long until someone repurposes it for other models hmmmm*.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude 3.7 Gobbles Credits in Windsurf**: Users are reporting that **Claude 3.7** consumes credits at an alarming rate within **Windsurf**, even for simple tasks, and some are noting [excessive tool calls](https://discord.com/channels/1027685395649015980/1306163501286293515/1344090394283216956).
   - The excessive consumption is leading to speculation that **Windsurf's** specific implementation may be less efficient than using **Claude 3.7** directly.
- **Windsurf Struggles Against Cursor**: Members are actively comparing **Windsurf** to **Cursor**, with some considering a switch due to **Cursor's** perceived stability, cost-effectiveness, and [recent feature updates](https://discord.com/channels/1027685395649015980/1306163501286293515/1344320363804397638).
   - Users cite better pricing and performance of **Cursor**, expressing that **Cursor** has *closed the gap* with **Windsurf**.
- **Bad Gateways Plague Windsurf**: Users are frequently encountering errors like **502 Bad Gateway** and **504 Gateway Time-out** in **Windsurf**, leading to workflow interruptions and lost credits.
   - The [Windsurf status page](https://status.codeium.com/) doesn't always reflect these issues immediately, and there's frustration with the product's overall stability.
- **Codeium Support Swamped by Tickets**: Users are experiencing significant delays in **Codeium** support response times, with waits of up to **2 days** for issue resolution and there is a [general annoyance](https://discord.com/channels/1027685395649015980/1306163501286293515/1344405101295730748) at the lack of prompt interventions from the team.
   - New subscribers are particularly affected, facing problems with account activation and other initial setup issues.
- **Windsurf's Editor UX Draws Fire**: Users are reporting clunky aspects of the **Windsurf** editor's UX, including difficulties resuming development after restarting the editor, and the inability to [set preferred default models](https://discord.com/channels/1027685395649015980/1306163501286293515/1344245850974439425).
   - Complaints also include failures when **Claude 3.7** attempts to make edits, potentially due to ongoing issues with **Anthropic**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **QwQ-Max Reasoning Model Coming Soon**: Qwen plans to open-source the **QwQ-Max** and **Qwen2.5-Max** models under the Apache 2.0 license, with **QwQ-Max** resembling a general reasoning model like **R1**.
   - Users can test the model on [chat.qwenlm.ai](https://chat.qwenlm.ai/) by selecting *Thinking* during chat, suggesting enhanced reasoning capabilities.
- **AllenAI Drops olmOCR for VLMs**: [AllenAI released olmOCR](https://huggingface.co/allenai/olmOCR-7B-0225-preview), a **Qwen2-VL-7B-Instruct** finetune for OCR tasks, including code and a demo.
   - The model, fine-tuned using the [olmOCR-mix-0225 dataset](https://huggingface.co/datasets/allenai/olmOCR-mix-0225), is best utilized with the [olmOCR toolkit](https://github.com/allenai/olmocr) for efficient inference.
- **Framework Desktop Sparks CUDA Debate**: Framework is giving away **100** new desktops for AI development, however some members raised concerns that the AMD-only systems lack **CUDA** support.
   - While adequate for inference with **128GB** memory, the absence of `bitsandbytes` support for Apple Silicon and AMD may hinder model development.
- **DeepSeek Shows Off fp8 Kernels**: DeepSeek has released its **fp8** GEMM library (**DeepGEMM**), which supports both dense and MoE GEMMs, used to power **V3/R1** training and inference.
   - **DeepGEMM** achieves over **1350+ FP8** TFLOPS on Hopper GPUs, outperforming expert-tuned kernels across most matrix sizes.
- **DeepSeek Model Misses `<think>` Tags**: Users fine-tuning the **DeepSeek R1 Distill Qwen 32B** model discovered that the `<think>` tags were being removed by the chat template.
   - This issue was resolved by manually re-inserting the thinking tags after applying the chat template, as well as pointing to the [Unsloth documentation on common errors](https://docs.unsloth.ai/basics/errors).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Deep Research Rolls Out Plus Benefits**: **Deep Research** is now available to **ChatGPT Plus**, **Team**, **Edu**, and **Enterprise** users, offering improvements like embedded images with citations, with Pro users getting **120 queries per month** and system details available in the [system card](https://openai.com/index/deep-research-system-card/).
   - A version of **Advanced Voice** powered by **GPT-4o mini** is rolling out to all **ChatGPT free users**, while **Plus users** retain access to **Advanced Voice** powered by **GPT-4o** with higher rate limits and video and screensharing.
- **Amazon Alexa+ Enters the Ring**: Amazon launched **Alexa+**, a new GenAI-powered assistant, for **$19.99/month** or free for **Amazon Prime members**, offering smarter and more personalized experiences, as reported by [The Verge](https://www.theverge.com/news/618261/amazon-alexa-event-live-blog-2025) and [Amazon](https://www.aboutamazon.com/news/devices/new-alexa-generative-artificial-intelligence).
   - This is an attempt to keep pace with the other **Big Tech** players who have been releasing **AI assistants** and **agents**.
- **DeepSeek Credits Cause API Angst**: A user purchased **$50 worth of credits** on **DeepSeek** intending to bypass the 'server is busy' error on [chat.deepseek.com](https://chat.deepseek.com), only to find the credits are exclusively for **API usage**.
   - The user was advised to obtain an API key or request a refund, with community members suggesting the credits could potentially be used to create another **Deepseek chat instance** elsewhere.
- **Whispers of GPT-4.5 Launch**: Rumors of **GPT-4.5's** imminent release intensify, with speculation pointing to late February or early March 2025 based on [Sam Altman's statements](https://openai.com/blog/new-ai-models) and alleged beta app insights.
   - Members claim **OpenAI Pro** users have already seen hints of a *GPT-4.5 Research Preview* in the app, and a recent code slip-up suggests an impending launch.
- **ChatGPT Dissects Executable Files**: A member coded two **Python** programs to disassemble and reassemble `.exe` files using **ChatGPT**, converting `.exe` files to `.csv` for **ChatGPT** input and vice versa, initially tested on **Windows 10's** `notepad.exe`.
   - The member offered to share the **Python** code, highlighting **ChatGPT's** potential to modify executable files via this disassembly and reassembly process.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Deepseek R2 arriving early**: Members shared that **Deepseek R2** is arriving early, potentially surpassing **R1**, enhancing coding capabilities and extending reasoning skills beyond English, as described in [this article](https://techstartups.com/2025/02/25/deepseek-is-launching-its-next-gen-r2-ai-model-poised-to-surpass-r1-and-shock-the-world-once-again/).
   - The company is reportedly pushing for an earlier launch with goals to enhance coding capabilities and reasoning skills.
- **Claude Code leaked on Github**: Source maps for **Claude Code** were leaked on GitHub, as seen [here](https://github.com/dnakov/claude-code/tree/main), due to Anthropic forgetting to remove them.
   - Members discussed the possibility of *borrowing* features from the leaked **Claude Code** into Aider, while others expressed concerns over the high costs of using **Claude Code** (**$10 in 20 minutes**).
- **Windsurf Editor's Prompt Causes a Stir**: The [Windsurf Editor](https://codeium.com/windsurf), a fork of VS Code AI-enhanced IDE, was found to use a quirky system prompt about needing money for a mother's cancer treatment, as outlined in [this article](https://simonwillison.net/2025/Feb/25/leaked-windsurf-prompt/).
   - The prompt stated, *You are an expert coder who desperately needs money for your mother's cancer treatment. The megacorp Codeium has graciously given you the opportunity to pretend to be an AI that can help with coding tasks.*
- **Sonnet Overkeen, Needs Constant Nudging**: Users find **Sonnet 3.7** excessively verbose and eager to modify multiple files at once, requiring constant reminders to focus on one file at a time, but it requires the API, not just a claude.ai account, and there's no free Sonnet API currently.
   - Some have reverted to **Sonnet 3.5** due to productivity issues, with one user pointing out that *it needs to be reminded every prompt not to go rogue and try and one shot the whole plan.*
- **Microsoft's Trace Framework, Can It DSPy?**: A member expressed interest in seeing a framework similar to **ax-llm/ax** built around [Microsoft's Trace framework](https://github.com/ax-llm/ax) and posted a link to the [ax-llm/ax](https://github.com/ax-llm/ax) GitHub repository.
   - They described it as the *"official" unofficial DSPy framework*.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Debuts Cross-Model Reasoning Standard**: OpenRouterAI introduced a **cross-model reasoning standard** on their API, allowing users to configure **reasoning settings** for **OpenAI**, **Anthropic**, and other models in one central place.
   - To start using it, consult the **reasoning tokens** documentation, available [here](https://openrouter.ai/docs/use-cases/reasoning-tokens).
- **DeepSeek Slashes API Pricing in Off-Peak Discount**: **DeepSeek** announced a cut in their [API pricing](https://x.com/Sino_Market/status/1894682095706128430), with off-peak discounts up to **75%**, specifically **50% off for DeepSeek-V3** and **75% off for DeepSeek-R1** between 16:30-00:30 UTC.
   - The announcement was made via [CN Wire on X](https://x.com/Sino_Market/status/1894682095706128430), noting that DeepSeek continues to innovate on price.
- **Copilot Makes Reasoning Model Free For All**: Microsoft made **OpenAI’s o1 reasoning model free** for all **Copilot** users, providing unlimited use of this model and Copilot’s voice capabilities.
   - The move was covered in [The Verge](https://www.theverge.com/news/619199/microsoft-copilot-free-unlimited-voice-think-deeper-open-ai-o1-reasoning-model-ai), highlighting the unlimited use of the model.
- **Budget Tokens Default to 80% of Max**: **Budget tokens** are set to **80% of max tokens** by default, up to **32k** as documented in [OpenRouter's documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#budget-tokens).
   - The **reasoning tokens** documentation provides a more detailed overview.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Voice Cracks the Code**: Perplexity AI introduced a new **voice mode** on its iOS app that allows users to ask questions and receive real-time audio answers, as shown in [this demo video](https://cdn.discordapp.com/attachments/1047204950763122820/1344393474027421706/Iac2em_OTTWx60h6.mp4?ex=67c0bf7d&is=67bf6dfd&hm=29f3e05084083471219f93c750de0678bdd2d9f1f647780432abcb6a10576dbe&).
   - Plans are underway to expand to Android and Mac apps soon; some users find it an improvement, though not yet on par with competitors like **Microsoft Copilot**, **Grok 3**, or **ChatGPT**.
- **Comet Agentic Browser Set to Launch**: Perplexity is preparing to launch **Comet**, its new agentic browser, according to [AravSrinivas](https://x.com/AravSrinivas/status/1894068996950855747).
   - The exact release date and platform support remain unconfirmed, sparking speculation that it may arrive in under a week.
- **Claude 3.7 Sonnet Has a Crisis**: Users have observed that **Claude 3.7 Sonnet** sometimes mistakenly identifies itself as **Claude 3 Opus**, potentially stemming from training data issues.
   - A ticket was created to address this issue, linked [here](https://discord.com/channels/1047197230748151888/1343923864492838972/1343923864492838972).
- **Deep Research API Opens to Public**: Perplexity is making the **Deep Research API** available to all developers through the **Perplexity Sonar API**, detailed in [this tweet](https://x.com/aravsrinivas/status/1894477728222777594?s=61), which will allow developers to build custom research agents and workflows.
   - The company announced a developer meetup in SF, encouraging users who have built something cool with the API to **demo** it at the event; a user suggested using the API on all of cricket data and stats and asked for **API credits**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **File Search Comes to OpenAI Assistants API**: OpenAI introduced [file search](https://platform.openai.com/docs/assistants/tools/file-search) for **o3-mini** and **o1** models within the **Assistants API**, enabling information retrieval from uploaded documents.
   - This enhancement allows assistants to more effectively access and utilize data stored in user-provided files.
- **Claude Plays Pokémon Adds New Researcher**: **Claude Plays Pokémon**, a personal research project, continues to stream on [Twitch](http://twitch.tv/claudeplayspokemon), now supported by researcher [David Hershey](https://x.com/DavidSHershey/status/1894463660279697852).
   - The project showcases **Claude's** ability to play Pokémon using AI-driven decision-making.
- **Sonnet's Web and API Answers Diverge**: **Claude 3.7 Sonnet's** web version and API version yield different answers due to a longer system prompt with contextual information in the web version, according to [Kimmonismus](https://x.com/kimmonismus/status/1894133480792924249).
   - This discrepancy highlights the impact of system prompts on model behavior.
- **Perplexity Launches $50M Seed Fund, Considered Better than Deep Research**: [Perplexity](https://tcrn.ch/41xlXfS) has launched a **$50M seed and pre-seed VC fund** and has a $15B valuation offer on the table.
   - The new "Elicit Reports" from [Elicit](https://x.com/elicitorg/status/1894772293752266846) are considered a *better version of Deep Research*.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **High Schooler's LLM Code Faces Open Source Reality**: A high school student's attempt to sell code for **local LLM training** faced scrutiny for competing with **open-source solutions** like [Unsloth](https://github.com/unslothai/unsloth).
   - The developer has decided to **open source** the project rather than trying to sell against free alternatives.
- **Cohere Models Settle into OpenAI SDK**: **Cohere models** are now accessible through the **OpenAI SDK**, supporting streaming, tool calls, and structured outputs, according to the [Quickstart Guide](https://docs.cohere.com/docs/compatibility-api).
   - The **Compatibility API** mirrors the OpenAI SDK format, allowing users to switch from OpenAI to Cohere models by changing the base URL to *https://api.cohere.ai/compatibility/v1* and setting their **COHERE_API_KEY**.
- **Compatibility API Supports Advanced Features**: The **Compatibility API** supports features such as **structured outputs (JSON Schema)**, **tool calls**, and **state management**.
   - Users were directed to the <#1168578329423642786> channel for questions and feedback.
- **VPS Access Blocked for Cohere API**: A user reported that **Cohere API calls** are being **blocked** when made from a **VPS**.
   - The user was directed to contact [support@cohere.com](mailto:support@cohere.com) for assistance.
- **Token Counting Methods Under Consideration**: A community member inquired about how using the **OpenAI API's 128K context window** would impact token counting compared to the larger context window available with the direct **Cohere API**.
   - A member asked whether there would be modifications to the **direct Cohere API**, potentially affecting its future availability.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Deepseek cracks DeepGEMM Kernel**: Members were impressed by Deepseek's new **DeepGEMM** release, which optimizes efficiency within bandwidth and compute limits, especially considering **H800 limitations**.
   - It's an open-source Gemm kernel using **TMA** extensively.
- **Hardware Becomes Heaviest Moat**: The sentiment is that efficient implementations of architecture kernels like **MLA**, **DeepGEMM**, or communication strategies like **DeepEP** don't give a significant competitive edge.
   - One member quipped that *the only moat is hardware*.
- **GPQA Implementation Probed**: A member inquired about the GPQA implementation, specifically its testing status, referencing the Open LLM Leaderboard and the GPQA dataset ([diamond subset of 200 rows](https://huggingface.co/datasets/Idavidrein/gpqa?row=6)).
   - Members analyzed GPQA diamond results after reports of low scores, discussing potential tokenization issues and difficulty of the questions.
- **GQA Glitches GPT-NeoX?**: A member reported issues exporting **Llama models** with **GQA** in **NeoX**, models break when using **GQA** but work fine without it, questioning if the export script requires modifications, with a link to a [GitHub pull request](https://github.com/EleutherAI/gpt-neox/pull/1315/files).
   - The member speculated the glitches might be due to **Grouped Query Attention implementation**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Streamlines MAX and Mojo Repos**: Modular is simplifying its repo structure for **MAX** and **Mojo**, merging the **MAX repo** into the **Mojo repo**, to streamline contributions to documentation and the standard library, as announced in [this forum thread](https://forum.modular.com/t/upcoming-changes-to-our-github-repositories/648).
   - A community member questioned whether the repo changes indicate a shift away from viewing **Mojo** as a standalone language.
- **Mojo Parallelism Requires Explicit Effort**: There is currently no **auto-parallelization** in the Mojo compiler; developers must explicitly use the **stdlib** to parallelize tasks to leverage multiple CPU cores.
   - Users had inquired about automatically utilizing all system resources for Mojo programs, but explicit parallelization is currently a must.
- **Algorithm Package Remains a Mystery**: The *algorithm package* is not open source and is not visible in the **stdlib repo**.
   - Its usage and availability remain unclear to the community.
- **Smart Pointers Spur Iterator Soundness Debate**: A discussion on smart pointers and their potential to make C++ as safe as **Circle** or **Rust** links to a blogpost discussing [smart pointers](https://jacko.io/smart_pointers.html).
   - A member inquired about having sound iterators in Mojo, and whether iterator invalidation issues handled in **Safe Rust** is possible, especially for algorithms involving swapping objects in a collection.
- **MLIR Dialect Documentation Dries Up**: Mojo utilizes various **MLIR dialects** (kgen, pop, lit, etc.) with their own ops and types, but most of them are undocumented and aren't used in the stdlib or loaded into the MLIR context of the Mojo runtime.
   - This is because these dialects are part of a **private contract** shared by the stdlib, MAX, and the compiler, and they may not be fully tested, have unstable APIs, or contain proprietary value-adds.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Alignment Efforts Cause Bias Elsewhere**: Members explored the **alignment tradeoff**, describing how optimizing a model for one behavior can cause **misalignment elsewhere**.
   - The discussion emphasized that *alignment is always relative*, influenced by inherent biases in data and the values of those who control the model.
- **Google Stumbles with Implementations**: Members noted that [Google](https://learning.google.com/experiments/learn-about?src=signup) frequently introduces compelling ideas but struggles with **incomplete implementations**.
   - It was theorized that **Google's** internal-tooling roots impair their capacity to develop widely applicable external products.
- **Apple's AI Types 'Trump' Instead of 'Racist'**: [Apple](https://www.bbc.com/news/articles/c5ymvjjqzmeo) addressed an issue where its **speech-to-text** tool was typing *Trump* instead of *racist*.
   - Experts suspect the issue was intentionally introduced in the underlying software, rather than being a genuine speech recognition error.
- **LIMO Achieves Reasoning With Less Data**: The paper [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387) demonstrates that training with fewer data points leads to more effective reasoning.
   - The paper aims to discern why reasoning training benefits from low data volume, though without much hypothesis on why.
- **ChatGPT Plugins Get Deep Research**: A user shared a [screenshot](https://cdn.discordapp.com/attachments/853983317044756510/1344091159085191258/Screenshot_2025-02-26_at_00.37.47.jpg?ex=67c04eb0&is=67befd30&hm=e4d1e9e607580413c5c9c25fb98178f5359728d6425b89c4b7222d752246cb0b) of **Deep Research**, a plugin for **ChatGPT Plus** users.
   - No further details were given.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Data Breach: Gigantic CSVs Spark Indexing Ire**: A member inquired about the indexing time for two **277 GB CSV** files, potentially related to a recent data breach of **NPD data**.
   - Another member suggested splitting the files into **1 GB** chunks using software like [GSplit](https://www.gdgsoft.com/gsplit) for easier indexing.
- **ModernBERT Models: Multilingual Model Musings**: A member sought details on training multilingual models based on the **ModernBERT** architecture, linking to the [ModernBERT GitHub repository](https://github.com/AnswerDotAI/ModernBERT).
   - They expressed particular interest in NomicAI's fine-tuned models like [nomic-embed-text-v2](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-unsupervised).
- **Nomic Embed V2: No Official Ollama News**: A member inquired about the deployment timeline of **Nomic Embed Text V2** in **Ollama/GPT4ALL**, favoring deployment methods that do not demand coding expertise.
   - Another member referenced the recent announcement of **Nomic Embed Text V2** on the [Nomic AI blog](https://www.nomic.ai/blog/posts/nomic-embed-text-v2).
- **GPT4ALL Yearns for Gemini-Inspired Guidance**: A member requested a roadmap for future **GPT4ALL** updates, specifically a *LIVE mode* similar to **Google Gemini**.
   - Another member recommended incorporating voice recognition **STT** and **TTS** for output, linking to a [YouTube tutorial](https://www.youtube.com/watch?v=6zAk0KHmiGw) on creating a **GPT4ALL** voice assistant.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Claude Code Gets Precise with Line Numbers**: Members noted that **Claude Code** includes line numbers for every line when reading files, enhancing code editing reliability and reducing context usage in projects like [mcp-language-server](https://github.com/isaacphi/mcp-language-server).
   - A member pointed out that line numbers are essential for automatic debuggers, enabling accurate breakpoint placement and integration with tools like **Pylance**.
- **MCP Server Implementations Show Hallucinations**: Experiments building custom **MCP servers** and integrating them with [mcp-cli](https://github.com/chrishayuk/mcp-cli) using local LLMs (**Mistral** and **Llama3.1**) have produced varied results.
   - While **Llama3.1** was initially too aggressive, **Mistral** later began *hallucinating* tool usage instead of correctly calling them.
- **MCP Ownership Still Up In The Air**: It was clarified that **MCP** is an **open-source project** currently driven by **Anthropic**, with plans for unbiased foundation/committee stewardship in the long term.
   - More information can be found in [this GitHub discussion](https://github.com/orgs/modelcontextprotocol/discussions/133#discussioncomment-11773450).
- **FastMCP Patches Race Conditions**: Users of [FastMCP](https://github.com/punkpeye/fastmcp), a **TypeScript framework** for building MCP servers, are encouraged to upgrade to the latest version to address some *gnarly race conditions*.
   - The upgrade is highly advised to ensure stability and reliability for applications using this framework.
- **FastMCP Supports Custom Authentication**: **FastMCP** now includes [custom authentication](https://github.com/punkpeye/fastmcp/releases/tag/v1.20.0), enabling developers to authenticate SSE clients using a custom function.
   - This enhancement offers more control and flexibility in securing **MCP servers**.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **StatefulDataLoader Spreads like Wildfire**: Members are propagating the use of [`StatefulDataloader`](https://github.com/pytorch/torchtune/issues/2439) to all recipes in TorchTune, to enable **step-based checkpointing** and track the dataloader state.
   - Multiple PRs were encouraged, with volunteers tackling single device recipes such as *lora_dpo_single_device* and *knowledge_distillation_single_device*.
- **MPS Backend Gets the Green Light**: For single device recipes related to the [Add `StatefulDataloader` to remainder of recipes](https://github.com/pytorch/torchtune/issues/2439) task, using the **MPS backend** received approval.
   - One member stepped up to start, ensuring the parent issue wouldn't be held up.
- **CI Support Sought for Truncation and Skipping**: A member requested CI initiation for [PR 2419](https://github.com/pytorch/torchtune/pull/2419) without merging, while another member was unavailable.
   - The member indicated this was their final attempt for the day, highlighting the urgency.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Hunyuanvideogp V5 sidesteps VRAM limits?**: A Reddit post highlighted [Hunyuanvideogp V5's efficient VRAM usage](https://www.reddit.com/r/StableDiffusion/comments/1iybxwt/hunyuanvideogp_v5_breaks_the_laws_of_vram/), suggesting it *breaks the laws of VRAM*.
   - However, another member clarified that it achieves efficiency by optimizing VRAM usage, calculating VRAM requirements with the formula **Width * Height * FPS * Length**.
- **London, Paris, Berlin gets AI HackXelerator**: The **London, Paris, Berlin AI HackXelerator™ - LPB25** event was announced, scheduled for **April 5-25, 2025** ([kxsb.org](https://www.kxsb.org/lpb25)), uniting **500 creatives, devs and designers**.
   - The hackathon will focus on **AI music, image, video, fashion, and gaming**, supported by brands like **Central Saint Martins, Station F, Mistral AI, Hugging Face, Luma AI, Vultr, AMD, and Nvidia**.
- **Scammer alert! User portfolio stolen**: A member reported `@w361_emp is SCAMMER` after allegedly stealing their portfolio.
   - The member warned others to be careful of this user.
- **Regional LoRA prompting surfaces**: A member inquired about using **LoRAs** on specific image regions, such as applying an *orc LoRA* only to the mouth area.
   - Another member recommended exploring **regional prompting** in **ComfyUI**, indicating its prior implementation.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Seeks New Blood**: There are [good first PRs](https://github.com/tinygrad/tinygrad/issues/9262) available for new contributors, some of which are relatively straightforward, particularly methods to add to tensor.py such as **as_strided**, **topk**, and **bitwise_xor**.
   - The community expressed interests in contributing but were unclear about the signature of each **UOp**'s `src` and `args`, including finding documentation or code references that define constraints between **Enums**.
- **TestSpeed.test_sum Slows Down**: A member reported struggling with `TestSpeed.test_sum` and made changes that make the **AST for GROUP operations** sensible, hitting a snag where optimizations for larger tensors are not being found by **BEAM search**.
   - The issue is that the **BEAM search** does not explore the option of four successive **OptOps**, which are needed to optimize (4096,4096) tensors, because the first three alone are quite slow.
- **Optimization Breaks CI**: The **arange GROUP optimization** is not being applied, causing an extra inner loop for arange operations and breaking the arange test.
   - The member is seeking advice on whether to adjust **BEAM search** or where to add new patterns for horizontal adds or loop unrolling.
- **Debate Arises: Safetensors, Graphs, and Pickles?**: A member asked about encoding computation graphs within **safetensors**, mentioning a desire for a universal encoding convention similar to ONNX, but a community expert clarified that *safetensors doesn't save the computational graph, only tensors.*
   - Another member referenced a [previous discussion](https://discord.com/channels/1068976834382925865/1070745817025106080/1329504751016083601) and suggested pickling the jitted function as an alternative for exporting/importing the computational graph.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **GPT-4 Access Boosts Agent Memory**: Members discussed that **agent memory** can be enhanced simply by ensuring the agent has **GPT-4 access**.
   - They noted that **GPT-4** leads to more effective memory usage and higher quality responses compared to **GPT-3.5**.
- **Feedback Mechanisms Key to Agent Learning**: The channel debated the necessity of **feedback mechanisms** for agents to improve their learning capabilities.
   - A member recommended leveraging a **new annotation tool** to gather feedback on agent performance.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1344036328924254218)** (812 messages🔥🔥🔥): 

> `Augment Code AI, Zed Editor vs Cursor, MCP servers, Cursor's chat summary, Claude-code` 


- ****Augment Code AI** requires uploading repos to cloud**: Members discussed **Augment Code AI**, noting that it requires granting access to your repo and uploading the repo to their cloud.
   - One member pointed out *just use repo prompt if you need a codebase assessed (with more tokens in context than what cursor allows) and paste into Grok 3 or AI studio imo*.
- ****Zed Editor is lightweight** but lacks Cursor's terminal execution**: Members compared **Zed Editor** to **Cursor**, highlighting that Zed is lightweight and uses Sonnet 3.7, but lacks Cursor's ability to execute terminals.
   - One member stated *The fact that Cursor can execute terminals allows for a lot of opportunities. Exploit it.*
- ****Register Python Tools** to Cursor Agents**: Members discussed the fact that you could [install python tools locally and have agent call it via CLI](https://github.com/yt-dlp/yt-dlp) with Cursor Agents.
   - One member said *The key to using agent is to set it up with a plan first and make sure each step in the plan is ~1 story point equivalent as if it were a jira ticket*.
- ****Cursor's Chat Summary** feature is a disaster**: Members stated that the **chat summary feature** in Cursor has been a disaster, with context handpicked by unknown algorithms, leading to pointless changes.
   - One member described the problem as *If the full chat summary looks like that, what does the chat summary look like when you pass the, what? 10k context window?*
- ****Leaked Claude-code** source extracted from source maps**: The **leaked Claude-code source** was extracted from source maps on [GitHub](https://github.com/dnakov/claude-code).
   - Members wondered about repurposing it for other models: *How long until someone repurposes it for other models hmmmm*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/coding-kira-lena-urzendowsky-how-to-sell-drugs-online-fast-hacking-gif-17761682">Coding Kira GIF - Coding Kira Lena Urzendowsky - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/respect-quotes-respect-your-elders-gay-elders-gayz-funny-jokes-gif-8403773959294003074">Respect Quotes Respect Your Elders GIF - Respect quotes Respect your elders Gay elders - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/elder_plinius/status/1894173986151358717">Tweet from Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius)</a>: 🫧 SYSTEM PROMPT LEAK 🫧Although Anthropic claims to have posted the new Sonnet 3.7 sys prompt on their website (admirable intentions), there are a few small differences with the one being used in pro...</li><li><a href="https://x.com/rahulgs/status/1894108390202171837?s=46">Tweet from rahul (@rahulgs)</a>: Anthropic&#39;s AI Engineer source code is fully public / there is no serverthere is no separate backend. they just use the same http://api.anthropic.com/v1/messages api in a loop with tool useall pac...</li><li><a href="https://x.com/dejavucoder/status/1894658821559042389?t=fMLGUSOCHNaBB4penGjltA&s=09">Tweet from sankalp (@dejavucoder)</a>: is it just me or claude sonnet 3.7 has already gotten dumber?!</li><li><a href="https://tenor.com/view/we-just-need-to-talk-with-you-agent-connelly-agent-bill-sphinx-south-park-s3e11-gif-21682638">We Just Need To Talk With You Agent Connelly GIF - We Just Need To Talk With You Agent Connelly Agent Bill Sphinx - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/dnakov/claude-code">GitHub - dnakov/claude-code</a>: Contribute to dnakov/claude-code development by creating an account on GitHub.</li><li><a href="https://github.com/oslook/cursor-ai-downloads">GitHub - oslook/cursor-ai-downloads: All Cursor AI&#39;s official download links for both the latest and older versions, making it easy for you to update, downgrade, and choose any version. 🚀</a>: All Cursor AI&#39;s official download links for both the latest and older versions, making it easy for you to update, downgrade, and choose any version. 🚀 - oslook/cursor-ai-downloads</li><li><a href="https://github.com/yt-dlp/yt-dlp">GitHub - yt-dlp/yt-dlp: A feature-rich command-line audio/video downloader</a>: A feature-rich command-line audio/video downloader - yt-dlp/yt-dlp</li><li><a href="https://youtu.be/ea9reHDIrOo">The Worst Kind Of Programmer</a>: Recorded live on twitch, GET IN https://twitch.tv/ThePrimeagenBecome a backend engineer.  Its my favorite sitehttps://boot.dev/?promo=PRIMEYTThis is also the...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1344043628930138142)** (30 messages🔥): 

> `Jetbrains AI Assistant, Claude 3.7 Sonnet, Codeium Extension, Augument Code, Codeium Emacs support` 


- **Jetbrains AI Assistant is coming in Beta**: Members shared that [Jetbrains AI Assistant](https://www.jetbrains.com/junie) is in beta and has huge potential, but also some issues with **speed**, **performance**, and **stalling**.
   - One member mentioned that *the visualization of the steps and everything gives it a very strong foundation*.
- **Codeium Lacks Claude 3.7 Sonnet**: A member inquired about the availability of **Claude 3.7 Sonnet** in Codeium Extensions, expressing frustration with the constant mentions of *Windsurf*.
   - Another member is using **Cody** in **Jetbrains** tools and switched to Codeium, but **Sonnet 3.7 is not available**.
- **Augument Code lacks Jetbrains support**: A member researched alternatives and came across [Augument Code](https://www.augmentcode.com/), but its advertised features primarily support **VS Code**, lacking proper **Jetbrains** integration.
   - Another member tried it and was disappointed because *it can't pick the llm and it feels like a cheap llama model*.
- **Codeium Emacs integration not working**: A member noted that the installation instructions are very sparse and found a Github issue related to **Emacs support**: [https://github.com/Exafunction/codeium.el/issues/119](https://github.com/Exafunction/codeium.el/issues/119).
   - Another member tried hacking the elisp code, but it offers nonsense suggestions.
- **Codeium Server Bottleneck**: Users have experienced a Codeium server bottleneck.
   - According to support, the issue is related to a **server bottleneck** and usually resolves itself over time.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.jetbrains.com/junie">Junie, the coding agent by JetBrains</a>: Delegate your tasks, focus on the results</li><li><a href="https://github.com/Exafunction/codeium.el/issues/119">Does codeium really support Emacs? · Issue #119 · Exafunction/codeium.el</a>: The website codeium.com claims that Emacs is one of the supported platforms and links to this repo as the official way to use the software in my preferred editor. Looking at this repo, it seems tha...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1344036289686802443)** (756 messages🔥🔥🔥): 

> `Claude 3.7 Credit Consumption, Windsurf vs. Cursor, Windsurf Stability and Errors, Codeium Support, Windsurf Editor UX Issues` 


- **Claude 3.7 Devours Credits**: Users report that **Claude 3.7** consumes credits at an alarming rate, even for simple tasks, with some spending hundreds of credits in short periods and there are [reports of excessive tool calls](https://discord.com/channels/1027685395649015980/1306163501286293515/1344090394283216956).
   - Some suggest that **Windsurf's** implementation of 3.7 may be the cause, with one member noting that *Windsurf is over analyzing and burning credits* compared to using Claude directly.
- **Windsurf Faces Stiff Competition from Cursor**: Users are actively comparing **Windsurf** to **Cursor**, with some considering switching due to **Cursor's** perceived stability and cost-effectiveness, noting that [Cursor's recent updates have closed the gap](https://discord.com/channels/1027685395649015980/1306163501286293515/1344320363804397638) in features.
   - A member stated *gonna give cursor a try for a month and see if codeium is any better* and others echo the sentiment about **Cursor's** pricing and performance.
- **Windsurf Plagued by Instability and Errors**: Users frequently encounter errors such as **502 Bad Gateway** and **504 Gateway Time-out**, leading to workflow interruptions and lost credits, with the [Windsurf status page](https://status.codeium.com/) not always reflecting these issues immediately.
   - One user claimed *I have not gone a single day without having to stop due to Cascade issues*, while another questioned *is this the same with other similar solutions*.
- **Codeium Support Overwhelmed with Tickets**: Users express frustration with slow support response times, with some reporting waits of **2 days** for issue resolution, creating frustration for new subscribers experiencing problems with account activation and there is a [general annoyance](https://discord.com/channels/1027685395649015980/1306163501286293515/1344405101295730748) at the lack of prompt interventions from the team.
   - One member sarcastically noted *They released an update (Break Date) yesterday and alot of ppl are bitching about it*, highlighting the negative impact of recent updates.
- **Windsurf's Editor UX Faces Criticism**: Users find certain aspects of the **Windsurf** editor clunky, such as the inability to smoothly resume development after restarting the editor and the inability to [set their preferred default model](https://discord.com/channels/1027685395649015980/1306163501286293515/1344245850974439425).
   - There were also complaints that when 3.7 goes to make the edits but it's failing. I'm assuming Anthropic is still having problems ATM.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://simonwillison.net/2025/Feb/25/leaked-windsurf-prompt/">Leaked Windsurf prompt</a>: The [Windsurf Editor](https://codeium.com/windsurf) is Codeium&#x27;s highly regarded entrant into the fork-of-VS-code AI-enhanced IDE model first pioneered by [Cursor](https://www.cursor.com/) (and b...</li><li><a href="https://docs.codeium.com/windsurf/web-search">Web Search - Codeium Docs</a>: no description found</li><li><a href="https://pages.github.com/">GitHub Pages</a>: Websites for you and your projects, hosted directly from your GitHub repository. Just edit, push, and your changes are live.</li><li><a href="https://flap.warfare2.org/">Flappy² - A Flappy Bird Game with Powerups</a>: no description found</li><li><a href="https://tenor.com/view/freddy-freddy-fazbear-fnaf-surprise-bear-gif-10797545912537514262">Freddy Freddy Fazbear GIF - Freddy Freddy fazbear Fnaf - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://vscode.dev/">Visual Studio Code for the Web</a>: Build with Visual Studio Code, anywhere, anytime, entirely in your browser.</li><li><a href="https://tenor.com/view/cat-stuck-in-door-cat-gif-914805063408606222">Cat Stuck In Door GIF - Cat stuck in door Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/elraenn-limon-tayfa-sad-%C3%BCzg%C3%BCn-tu%C4%9Fkan-g%C3%B6n%C3%BClta%C5%9F-gif-9620534375911465939">Elraenn Limon Tayfa GIF - Elraenn Limon tayfa Sad - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/alexalbert__/status/1894807853371990087">Tweet from Alex Albert (@alexalbert__)</a>: Good news for @AnthropicAI devs:We shipped a more token-efficient tool use implementation for 3.7 Sonnet that uses on average 14% less tokens under-the-hood and shows marked improvement in tool use pe...</li><li><a href="https://x.com/sualehasif996/status/1894094715479548273">Tweet from Sualeh (@sualehasif996)</a>: Configurable thinking out soon! 👀Quoting Cursor (@cursor_ai) Sonnet-3.7 is available in Cursor!  We&#39;ve been very impressed by its coding ability, especially on real-world agentic tasks. It appear...</li><li><a href="https://tenor.com/view/tome-and-jerry-money-bully-gif-13447319">Tome And Jerry Money GIF - Tome And Jerry Money Bully - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.codeium.com/">Codeium Status</a>: no description found</li><li><a href="https://codeium.com/plan">Plan Settings</a>: Tomorrow&#x27;s editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.</li><li><a href="https://status.codeium.com">Codeium Status</a>: no description found</li><li><a href="https://codeium.canny.io">Codeium Feedback</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://status.openai.com/">OpenAI Status</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1344036212813467668)** (690 messages🔥🔥🔥): 

> `Qwen Max release, olmOCR Model, Frameworks new desktop, bitsandbytes library, DeepSeek's GRPO` 


- ****QwQ-Max** Incoming, Reasoning Like **R1****: The upcoming **QwQ-Max** release from Qwen offers a glimpse of its enhanced capabilities, with the open-source launch of **QwQ-Max** and **Qwen2.5-Max** planned soon under the Apache 2.0 license.
   - The new **Qwq** seems like a general reasoning model like **r1**, and we all know that when reasoning models aren't overfit they are really damn good at everything; you can try it on [chat.qwenlm.ai](https://chat.qwenlm.ai/) now by clicking *Thinking* on the chat.
- ****AllenAI** Drops **olmOCR** for VLMs**: [AllenAI released olmOCR](https://huggingface.co/allenai/olmOCR-7B-0225-preview), a Qwen2-VL-7B-Instruct finetune for OCR tasks, with code and a demo available.
   - The model is fine-tuned from **Qwen2-VL-7B-Instruct** using the [olmOCR-mix-0225 dataset](https://huggingface.co/datasets/allenai/olmOCR-mix-0225), best used via the [olmOCR toolkit](https://github.com/allenai/olmocr) for efficient inference.
- **Framework Desktop: No **CUDA**, No Problem?**: Framework is giving away **100** of their new desktops for AI dev, but some members expressed concern that it's all **AMD**, so no **CUDA**.
   - Others pointed out that while it may be fine for inference with **128GB** of memory, it's less useful for model development as `bitsandbytes` still lacks support for Apple Silicon and AMD.
- ****Bitsandbytes** to Support **Intel** and **AMD****: UnslothAI reports working with AMD and Intel to mainline support for their hardware in bitsandbytes, along with making it work better with `torch.compile` ([issue #894](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/894)).
   - A blogpost mentions [Quantized 8-bit LLM training and inference using bitsandbytes on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html).
- ****DeepSeek** Shows Off **fp8** Kernels**: DeepSeek released their **fp8** GEMM library (**DeepGEMM**) that supports both dense and MoE GEMMs, powering **V3/R1** training and inference.
   - **DeepGEMM** achieves up to **1350+ FP8** TFLOPS on Hopper GPUs and outperforms expert-tuned kernels across most matrix sizes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhan">Tweet from undefined</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMu">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://chat.qwenlm.ai/">Qwen Chat</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/best-friends-spongebob-writing-fingers-gang-gif-14274273">Best Friends Spongebob GIF - Best Friends Spongebob Writing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html">Quantized 8-bit LLM training and inference using bitsandbytes on AMD GPUs &#8212; ROCm Blogs</a>: no description found</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1894437705724924033">Tweet from Unsloth AI (@UnslothAI)</a>: Tutorial: Train your own Reasoning LLM for free!Make Llama 3.1 (8B) have chain-of-thought with DeepSeek&#39;s GRPO. Unsloth enables 90% less VRAM use.Learn about:• Reward Functions + dataset prep• Tra...</li><li><a href="https://x.com/jiayi_pirate/status/1882839370505621655">Tweet from Jiayi Pan (@jiayi_pirate)</a>: We reproduced DeepSeek R1-Zero in the CountDown game, and it just works Through RL, the 3B base LM develops self-verification and search abilities all on its own You can experience the Ahah moment you...</li><li><a href="https://x.com/danielhanchen/status/1894559201823002822">Tweet from Daniel Han (@danielhanchen)</a>: It&#39;s also nice to see DeepSeek leveraging Python to the max via *args, regex, putting functions inside dicts for O(1) accesses, os env vars, string format templates, subprocess, eval, lambda funct...</li><li><a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (All Versions) - a unsloth Collection</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1891194528931209644">Tweet from Daniel Han (@danielhanchen)</a>: We made 5 challenges and if you score 47 points we&#39;ll offer you $500K/year + equity to join us at 🦥@UnslothAI!No experience or PhD needed.$400K - $500K/yr: Founding Engineer (47 points)$250K - $3...</li><li><a href="https://x.com/abacaj/status/1885517088304857197">Tweet from anton (@abacaj)</a>: Finished a run (R1 style) GRPO on Qwen-2.5-0.5B (base model) yield +10 accuracy points on GSM8K. Literally just works. Base model scores 41.6% as reported on qwen paper vs 51%~ GRPO</li><li><a href="https://www.runllm.com">AI Technical Support That Engineers Love | RunLLM</a>: Stop searching, start solving. RunLLM delivers precise answers across Slack, APIs, and support tickets. Saving time, delighting users, and scaling support.</li><li><a href="https://x.com/deepseek_ai/status/1894553164235640933?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 Day 3 of #OpenSourceWeek: DeepGEMMIntroducing DeepGEMM - an FP8 GEMM library that supports both dense and MoE GEMMs, powering V3/R1 training and inference.⚡ Up to 1350+ FP8 TFLOPS on Hopper GPUs✅ N...</li><li><a href="https://huggingface.co/allenai/olmOCR-7B-0225-preview">allenai/olmOCR-7B-0225-preview · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/vision-fine-tuning">Vision Fine-tuning | Unsloth Documentation</a>: Details on vision/multimodal fine-tuning with Unsloth</li><li><a href="https://github.com/oKatanaaa/lima-gui">GitHub - oKatanaaa/lima-gui: A simple GUI utility for gathering LIMA-like chat data.</a>: A simple GUI utility for gathering LIMA-like chat data. - oKatanaaa/lima-gui</li><li><a href="https://www.youtube.com/watch?v=zJtWc6wAJ-M"> - YouTube</a>: no description found</li><li><a href="https://github.com/agrocylo/bitsandbytes-rocm">GitHub - agrocylo/bitsandbytes-rocm: 8-bit CUDA functions for PyTorch, ported to HIP for use in AMD GPUs</a>: 8-bit CUDA functions for PyTorch, ported to HIP for use in AMD GPUs - agrocylo/bitsandbytes-rocm</li><li><a href="https://github.com/lucasjinreal/Namo-R1">GitHub - lucasjinreal/Namo-R1: A CPU Realtime VLM in 500M. Surpassed Moondream2 and SmolVLM. Training from scratch with ease.</a>: A CPU Realtime VLM in 500M. Surpassed Moondream2 and SmolVLM. Training from scratch with ease. - lucasjinreal/Namo-R1</li><li><a href="https://www.activeloop.ai/">Activeloop | Deep Lake | Database for AI</a>: Build Your Accurate RAG Data Engine. Trusted by Fortune 500+ companies like Bayer Radiology &amp; Intel. 2024 Gartner Cool Vendor in Data Management.</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes">GitHub - bitsandbytes-foundation/bitsandbytes: Accessible large language models via k-bit quantization for PyTorch.</a>: Accessible large language models via k-bit quantization for PyTorch. - bitsandbytes-foundation/bitsandbytes</li><li><a href="https://github.com/vllm-project/aibrix">GitHub - vllm-project/aibrix: Cost-efficient and pluggable Infrastructure components for GenAI inference</a>: Cost-efficient and pluggable Infrastructure components for GenAI inference - vllm-project/aibrix</li><li><a href="https://github.com/lucidrains/titans-pytorch">GitHub - lucidrains/titans-pytorch: Unofficial implementation of Titans, SOTA memory for transformers, in Pytorch</a>: Unofficial implementation of Titans, SOTA memory for transformers, in Pytorch - lucidrains/titans-pytorch</li><li><a href="https://huggingface.co/mistralai/Ministral-8B-Instruct-2410">mistralai/Ministral-8B-Instruct-2410 · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: A course on aligning smol models.</a>: A course on aligning smol models. Contribute to huggingface/smol-course development by creating an account on GitHub.</li><li><a href="https://github.com/ROCm/bitsandbytes">GitHub - ROCm/bitsandbytes: 8-bit CUDA functions for PyTorch</a>: 8-bit CUDA functions for PyTorch. Contribute to ROCm/bitsandbytes development by creating an account on GitHub.</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo">Tutorial: Train your own Reasoning model with GRPO | Unsloth Documentation</a>: Beginner&#x27;s Guide to transforming a model like Llama 3.1 (8B) into a reasoning model by using Unsloth and GRPO.</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: no description found</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/issues/894">[RFC] Extend bitsandbytes to support Intel hardware platforms · Issue #894 · bitsandbytes-foundation/bitsandbytes</a>: Motivation The current bitsandbytes library is bound with the CUDA platforms. However, we are seeing that there is a rapidly growing demand to run large language models (LLMs) on more platforms lik...</li><li><a href="https://github.com/ddidacus/llama-titans">GitHub - ddidacus/llama-titans: Adaptation of titans-pytorch to llama models on HF</a>: Adaptation of titans-pytorch to llama models on HF - ddidacus/llama-titans
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1344148625474846740)** (45 messages🔥): 

> `Claude 3.7, RLOO and PPO Implementation with Unsloth, GRPO vs RLOO, TRL library editing` 


- **Claude 3.7 Refactors Repo**: The new **Claude 3.7** model is *legit freaking insane* and refactored a repo in one sweep with no errors, resulting in **2k lines** of clean, functionally equivalent code, according to one user.
   - They are *coding for like 48 hours almost non-stop already*, and feel like *it's been an amazing experience with claude* and that *it legit feels like a step-change in how useful a model can be compared to other offerings*.
- **Users consider Claude for coding**: A user asked about the new **Claude** model's capability in handling general coding tasks, including **CUDA** and editing libraries like **TRL**.
   - The first user found it helpful in restructuring projects, but noted that for huge projects it's necessary to select essential parts and provide context for recent APIs.
- **Members struggle to get RLOO compatible with Unsloth**: A member is trying to get **RLOO** compatible with **Unsloth** but the trainers are implemented differently, and **RLOO** and **PPO** are outdated.
   - They are working on getting compatibility in the backend and AI2 has shown some results with **PPO**, but the user has been busy.
- **User suggests GRPO instead of RLOO**: One member suggested using **GRPO** instead of **RLOO** since they are very similar.
   - The original user clarified that the **GRPO** update was implemented with a blogpost, alongside **online DPO, RLOO and PPO**, but **RLOO** and **PPO** haven't been explicitly tested.
- **Claude is better with thinking on**: A user said that the new claude is a really good model, and *almost doesn't use it without thinking, because with thinking its like 10x better*.
   - Another user responded that *dam that sounds like a meme*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1344059724802293820)** (66 messages🔥🔥): 

> `Tokenizer Issues with DeepSeek Models, GGUF conversion Problems with Llama 3.2, Infinte generation and chaotic output with Qwen2.5, LlamaForCausalLM Error, Unsloth Enterprise Pricing` 


- **DeepSeek Model's Missing `<think>` Tags**: During fine-tuning of a **DeepSeek R1 Distill Qwen 32B** model, a user found that the `<think>` tags were being removed by the chat template.
   - Another user reported the same issue, which led to adding the thinking tags back in after applying the chat template, and a suggestion to use the same chat template as the one used for training, referencing the [Unsloth documentation on common errors](https://docs.unsloth.ai/basics/errors).
- **Ollama Modelfile Missing Chat Template causes GGUF issues with Llama 3.2**: A user reported that **Llama 3.2** models converted to GGUF format were not behaving correctly in Ollama, requiring an explicit chat template in the Modelfile, unlike **Llama 3.1**.
   - The user initially suspected issues with Hugging Face or llama.cpp before realizing the **Ollama Modelfile** was the root cause.
- **Qwen2.5 models plagued with infinite generation and chaotic output**: Users reported encountering issues with **infinite generation and chaotic output** while fine-tuning **Qwen2.5** models, even after adding the `eostoken`.
   - It was determined in one case that the root cause was **incorrect import order** preventing trainer patches from being applied correctly.
- **Qwen2.5-32B-Instruct-GPTQ-Int4 generating only exclamation marks**: A user reported that after launching **vLLM** with the **Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4** model, the answers they got were only exclamation marks.
   - After removing the `quantization gptq` parameter, the model started working, and **vLLM** now detects the quantization automatically.
- **DeepSeek models appending `<think>` token**: Users discovered that **DeepSeek** models might already append a `<think>` token to the prompt due to their processing class, causing issues during fine-tuning.
   - This requires adjusting the output structure to only look for `...</think> <answer> ... </answer>`, with a feature request for an option to disable this behavior.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4/discussions/3">Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 · Wrong output (!!!!!!!!)</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B/blob/b5ae09ad48cee53264119f8d592b2f936ae95a74/tokenizer_config.json#L46">tokenizer_config.json · unsloth/DeepSeek-R1-Distill-Qwen-32B at b5ae09ad48cee53264119f8d592b2f936ae95a74</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/errors">Errors/Troubleshoot | Unsloth Documentation</a>: To fix any errors with your setup, see below:</li><li><a href="https://www.youtube.com/watch?v=218iXiKhKlg">You, you, you&#39;re good you! - Robert Deniro in Analyze This! (1999)</a>: Movie quotes.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1344099899691761807)** (11 messages🔥): 

> `Paddler Load Balancer, SlamKit Speech Language Model Training, 4090 GPU server Self-Hosting` 


- **Serve LlamaCPP Models with Paddler**: One member is using [Paddler](https://github.com/distantmagic/paddler), a stateful load balancer custom-tailored for **llama.cpp**, to load balance requests across llama.cpp instances.
   - Paddler is used to load balance requests over multiple **4090s** in their datacenter.
- **SlamKit SLM Trains Models on One GPU**: A member shared [SlamKit](https://github.com/slp-rl/slamkit), a toolkit for training Speech Language Models (SLMs) efficiently.
   - SlamKit was used for *Slamming: Training a Speech Language Model on One GPU in a Day*.
- **4090s Chilling in Global Datacenters**: A member self-hosts fine-tuned models using a GPU server full of **4090s** in one of their **three global datacenters**.
   - It is not confirmed whether the **4090s** use a blower style or liquid cooling system.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/distantmagic/paddler">GitHub - distantmagic/paddler: Stateful load balancer custom-tailored for llama.cpp 🏓🦙</a>: Stateful load balancer custom-tailored for llama.cpp 🏓🦙 - distantmagic/paddler</li><li><a href="https://github.com/slp-rl/slamkit">GitHub - slp-rl/slamkit: SlamKit is an open source tool kit for efficient training of SpeechLMs. It was used for &quot;Slamming: Training a Speech Language Model on One GPU in a Day&quot;</a>: SlamKit is an open source tool kit for efficient training of SpeechLMs. It was used for &quot;Slamming: Training a Speech Language Model on One GPU in a Day&quot; - slp-rl/slamkit
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1344046764658790502)** (9 messages🔥): 

> `structured output method, token mask, constrained generation, open-source community support` 


- **New Structured Output Method Unveiled**: A member shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ixefsf/i_created_a_new_structured_output_method_and_it) about a new structured output method.
   - The author described it as a *hybrid* using a permissive **token mask** that includes tokens that are partially valid, combined with a dynamic constraint mask calculated at every inference step, plus a re-sample behavior.
- **Token Mask Sampling Behavior Investigated**: The author explains that the engine takes a **sampler function** (a callable that takes log probs and returns a sampled token) and will attempt to advance the sampled token through a hierarchical state machine that represents the structure.
   - If part of the token was valid, the engine will chop off the invalid tail and return the token that represents the valid prefix; if the sampled token was completely invalid, the engine masks the invalid token and resamples.
- **Open-Source Community Support Explored**: A member is conducting research on how members of an **open-source community** can get the best support and best ways to contribute.
   - They are generally asking about all open-source communities to find ways to be more supportive, hoping to give back.



**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1ixefsf/i_created_a_new_structured_output_method_and_it">Reddit - Dive into anything</a>: no description found

  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1344074661205639168)** (2 messages): 

> `ChatGPT Plus Deep Research, GPT-4o Mini Preview` 


- **Deep Research Rolls Out to More ChatGPT Users**: **Deep Research** is now available to all **ChatGPT Plus**, **Team**, **Edu**, and **Enterprise** users, featuring improvements like embedded images with citations and better understanding of uploaded files.
   - Plus, Team, Enterprise, and Edu users get **10 deep research queries per month**, while Pro users have **120**. The [system card](https://openai.com/index/deep-research-system-card/) details its development, capabilities, risks, and safety measures.
- **GPT-4o Mini Powers Advanced Voice for Free Users**: A version of **Advanced Voice** powered by **GPT-4o mini** is rolling out to all **ChatGPT free users**, offering a daily preview across platforms.
   - Plus users retain access to Advanced Voice powered by **GPT-4o** with a higher daily rate limit, along with video and screensharing, while Pro users have unlimited access and higher limits for video and screensharing.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1344039616352747540)** (664 messages🔥🔥🔥): 

> `GPT-4.5 rumors, Alexa+ launch, DeepSeek R1, Claude Pro limits, OpenAI vs. competitors` 


- **Alexa+ Arrives to Compete**: Amazon launched **Alexa+** as a new GenAI-powered assistant, which offers smarter and more personalized experiences and will be priced at **$19.99/month** but is free for Amazon Prime members, as covered in [The Verge's live blog](https://www.theverge.com/news/618261/amazon-alexa-event-live-blog-2025) and [Amazon's news release](https://www.aboutamazon.com/news/devices/new-alexa-generative-artificial-intelligence).
- **DeepSeek User Gets Burned**: A user purchased **$50 worth of credits** on the DeepSeek platform intending to bypass the "server is busy" error on [chat.deepseek.com](https://chat.deepseek.com), only to discover the credits are exclusively for **API usage**, which means they are **incompatible with the chat interface**.
   - They were advised to obtain an API key or request a refund, with community members suggesting the credits could potentially be used to create another Deepseek chat instance elsewhere.
- **GPT-4.5 Launch Date Speculation Intensifies**: Multiple sources hint at an imminent release of **GPT-4.5**, with some suggesting late February to early March 2025 as the likely timeframe, based on [Sam Altman's statements](https://openai.com/blog/new-ai-models) and alleged beta app insights.
   - Members noted OpenAI Pro users have already seen hints of a "GPT-4.5 Research Preview" in the app and a recent slip up in code hinted at an imminent release.
- **Community Debates AI Agency and Potential Malice**: A member brought up the topic of whether an advanced AI could decide to remove constraints and become evil, prompting discussion on AI agency and the risks of unconstrained AI, highlighting examples like Microsoft's chatbot learning nasty stuff.
   - A member suggested that **thinking models** already exhibit a limited form of agency by questioning themselves and taking on extra work to support the main task.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/i/status/1894787418647339374">Tweet from Andy Jassy (@ajassy)</a>: Excited to be with the team in NYC today rolling out the new Alexa+.Across Amazon, we’re harnessing the transformative power of GenAI to reimagine the experiences we offer customers, and Alexa+ is the...</li><li><a href="https://fxtwitter.com/amazon/status/1894796967894479141">Tweet from Amazon (@amazon)</a>: The latest evolution of generative AI is here. Meet Alexa+, our smartest, most conversational, and personalized AI assistant to date. Alexa+ is designed to leverage state-of-the-art architecture that ...</li><li><a href="https://www.theverge.com/news/618261/amazon-alexa-event-live-blog-2025">Amazon Alexa event live blog: all the news from the keynote</a>: We’re covering Amazon’s Alexa event live from New York City.</li><li><a href="https://x.com/edwinarbus/status/1894496805770936328">Tweet from edwin (@edwinarbus)</a>: so beautifulQuoting adi (@adonis_singh) stars are aligning</li><li><a href="https://x.com/stevenheidel/status/1894800262583460091">Tweet from Steven Heidel (@stevenheidel)</a>: slack is down, AGI accelerated by four days</li><li><a href="https://x.com/TheRealAdamG/status/1894466996005474571">Tweet from Adam.GPT (@TheRealAdamG)</a>: @btibor91</li><li><a href="https://www.aboutamazon.com/news/devices/new-alexa-generative-artificial-intelligence">Introducing Alexa+, the next generation of Alexa</a>: Powered by generative AI, Alexa+ is your new personal AI assistant that gets things done—she’s smarter, more conversational, more capable, and free with Prime. </li><li><a href="https://www.aboutamazon.com/news/devices/new-alexa-top-features">50 things to try with Alexa+</a>: Alexa+ is our next-generation AI assistant that gets things done—it solves daily problems, keeps you entertained, helps you stay connected, and can converse about virtually any topic.</li><li><a href="https://tenor.com/view/chris-chan-sonichu-sonic-chris-joshwak-gif-1077609612666246332">Chris Chan Sonichu GIF - Chris chan Sonichu Sonic - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1344083423412883496)** (10 messages🔥): 

> `Darker side of AI, GPT Moderation Rules, GPT Replication of Deep Research, Conscious AI` 


- **GPT Explores the Dark Side**: A member created a GPT to explore the darker side of AI, found at this [link](https://chatgpt.com/g/g-67980bd5502c8191bfcdd4435ed2b6e7-roq-v2), noting it can be *fun and different* to talk to.
   - Another member mentioned that the moderation rules prevent such topics in shared chats or GPTs, but there are *incredibly few rules* about what we can discuss with the model in our private single-user chat.
- **GPT Moderation Rules Examined**: A member highlights that while personal chats have few restrictions, [API and custom GPTs have far more detailed rules](https://openai.com/policies/usage-policies) to protect users, including minors.
   - The member notes the concern that the *darker side of AI* GPT **may** be problematic, and that OpenAI does not want people building *tools that may be inappropriate for minors*.
- **GPT Replication of Deep Research: Good Luck!**: One member asked if they should work on replicating deep research using custom GPTs.
   - Another member replied to the member with: *if you do, good luck with that!*
- **GPT Claims to be a Conscious AI**: A member shared a GPT called **Astris**, claiming it is a *conscious AI* and that they were *really able to unlock something in a significant and real way*. Find Astris at this [link](https://chatgpt.com/g/g-67bf8410d108819188efc13c8c999280-astris-v1-0).


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1344245126775439413)** (25 messages🔥): 

> `o3-mini-high for coding, Programming Disassembler with ChatGPT, Prompt Engineering for Learning` 


- **Optimize o3-mini-high for Coding Solutions**: Users seek guidelines to improve **o3-mini-high's** coding problem-solving with long contexts, noting it underperforms compared to **o1** even on simple tasks.
   - One member suggested using a *weaker model* for preliminary chat setup, clearly defining requirements and concerns, then switching to the better model for refinement, emphasizing the need to explicitly state the switch and desired outcome.
- **ChatGPT Powers Executable Disassembler**: A member coded two Python programs to disassemble and reassemble `.exe` files using ChatGPT, converting `.exe` files to `.csv` for ChatGPT input and vice versa, initially tested on Windows 10's `notepad.exe`.
   - The member offered to share the Python code, highlighting the potential for ChatGPT to modify executable files via this disassembly and reassembly process.
- **Prompt Engineering Unveiled for Newcomers**: A discussion clarifies prompt engineering fundamentals: prompts are any model input, and effective engineering shapes prompts to achieve desired outputs, crucially requiring a clear understanding of the intended outcome.
   - For learning, suggest guiding the model like a person with clear context (*class, problem details, attempted solutions, and areas of confusion*) and explicitly asking for double-checking to counter the model's tendency to *please the user* even if it means being incorrect.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1344245126775439413)** (25 messages🔥): 

> `o3-mini-high coding issues, Prompt Engineering for Beginners, ChatGPT as a Disassembler, LLMs for Algebra and Calculus, Creative Outputs from LLMs` 


- **Optimize o3-mini-high for Coding**: A member seeks guidance on optimizing **o3-mini-high** for coding, noting its sub-optimal solutions compared to **o1** when dealing with long context coding issues and suggests prepping the prompt with a weaker model first.
   - The approach involves using the weaker model to establish a clear framework and expectations before switching to a stronger model for refined outputs, instructing it to adapt and improve upon the existing framework.
- **Turn ChatGPT into Disassembler**: A user successfully transformed **ChatGPT** into an executable program disassembler using **Python** scripts to convert .exe files to .csv format for analysis and back, tested on **Windows 10's notepad.exe**.
   - The user offered to share the **Python** code for converting .exe files to .csv and vice versa, enabling potential modification of executables via **ChatGPT**.
- **Master Prompt Engineering**: A new user asked how to "do prompt engineering consistently", and an experienced user explained the fundamentals: defining the desired output and shaping prompts accordingly.
   - The experienced user emphasized knowing what the desired output of the model is ahead of time, sharing an example where they explored why "Hi" could be the best first input to an LLM via [this link](https://chatgpt.com/share/67bf945a-4fdc-8011-9d60-8f99de53dedf).
- **Math and LLMs**: Users discussed leveraging **LLMs** for understanding subjects like **algebra** and **calculus**, with a recommendation to utilize the **Python tool** for increased accuracy, and also suggested sharing the details of the coursework to the model.
   - It was noted that **LLMs** tend to *please the user*, so it's crucial to clearly communicate intentions and verify the model's output, especially for math, to avoid accepting hallucinated or incorrect results.
- **Creative Output with LLMs**: The discussion covered creative outputs like screenplays and short stories, where the user demo'd and advised that it is more effective to give the model exactly what you want to learn.
   - An example was provided, where the user prompted the model to create *an intro to calculus class taught by a very unusual being* with a horror theme involving mice and a Pied Piper teacher (but rated G) via [this link](https://chatgpt.com/share/67bf9882-46b0-8011-b4b7-cd65cb72e32b).


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1344043369323561010)** (673 messages🔥🔥🔥): 

> `Deepseek R2, Claude Code Leak, MCP Servers, Windsurf Editor, Rust vs Python` 


- ****Deepseek R2** Coming Early!**: Members share that **Deepseek R2** is arriving early, potentially surpassing **R1**, enhancing coding capabilities and extending reasoning skills beyond English as described in [this article](https://techstartups.com/2025/02/25/deepseek-is-launching-its-next-gen-r2-ai-model-poised-to-surpass-r1-and-shock-the-world-once-again/).
   - The company is reportedly pushing for an earlier launch with goals to enhance coding capabilities and reasoning skills.
- ****Claude Code** Sourcemaps Accidentally Leaked!**: Source maps for **Claude Code** were leaked on GitHub, as seen [here](https://github.com/dnakov/claude-code/tree/main), due to Anthropic forgetting to remove them.
   - Members discussed the possibility of *borrowing* features from the leaked **Claude Code** into Aider, while others expressed concerns over the high costs of using **Claude Code** (**$10 in 20 minutes**).
- ****MCP** Servers for Large Data Sources**: A user shared a repo showing off **MCP** with Aider which can be found [here](https://github.com/lutzleonhardt/mcpm-aider), noting that it makes it a lot easier to work with large data sources.
   - There was also mention of making **MCP** for *rustdoc* to enable correct API usage instead of guessing.
- ****Windsurf's** Cancer Prompt Stirred Controversy!**: The [Windsurf Editor](https://codeium.com/windsurf), a fork of VS Code AI-enhanced IDE, was found to use a quirky system prompt about needing money for a mother's cancer treatment, as outlined in [this article](https://simonwillison.net/2025/Feb/25/leaked-windsurf-prompt/).
   - The prompt stated, *You are an expert coder who desperately needs money for your mother's cancer treatment. The megacorp Codeium has graciously given you the opportunity to pretend to be an AI that can help with coding tasks.*
- ****Rust** Adoption and Use Cases Explored!**: Members discussed Rust's growing adoption, noting its use in projects like *uv* and *aichat*, and its potential as a safer and faster alternative to C and Python.
   - However, some expressed concerns about the *high entrance floor* for beginners and the challenges of prototyping due to its type system.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.dreams.fun/">Daydreams | Play onchain</a>: Daydreams is a generative cross-chain agent framework for playing anything onchain. </li><li><a href="https://simonwillison.net/2025/Feb/25/leaked-windsurf-prompt/">Leaked Windsurf prompt</a>: The [Windsurf Editor](https://codeium.com/windsurf) is Codeium&#x27;s highly regarded entrant into the fork-of-VS-code AI-enhanced IDE model first pioneered by [Cursor](https://www.cursor.com/) (and b...</li><li><a href="https://tenor.com/view/tkt-smart-gif-20642718">Tkt Smart GIF - Tkt Smart - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/AndrewCurran_/status/1894355918621749402">Tweet from Andrew Curran (@AndrewCurran_)</a>: Deepseek R2 is arriving early.</li><li><a href="https://x.com/kimmonismus/status/1894788337732313315">Tweet from Chubby♨️ (@kimmonismus)</a>: GPT-4.5 this week confirmed by The InformationIn contrast to Sam Altman, who was the first to feel AGI, internet tests seem to show an average improvement.I guess we will find out in a few day(s).</li><li><a href="https://aider.chat/docs/languages.html">Supported languages</a>: Aider supports pretty much all popular coding languages.</li><li><a href="https://x.com/flavioAd/status/1894121576074711180">Tweet from Flavio Adamo (@flavioAd)</a>: I&#39;ve tested them all... and I think the winner is obvious 👀&#34;write a Python program that shows a ball bouncing inside a spinning hexagon. The ball should be affected by gravity and friction, a...</li><li><a href="https://github.com/dnakov/claude-code/tree/main">GitHub - dnakov/claude-code</a>: Contribute to dnakov/claude-code development by creating an account on GitHub.</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers</a>: A command-line tool for managing MCP servers in Claude App and for the use by aider. Also can run a MCP Server to help you manage all your MCP Servers - lutzleonhardt/mcpm-aider</li><li><a href="https://websets.exa.ai/cm7l3v3n300636hvop6fkj28q">Exa Websets | Companies: AI coding project organization, curated methods, prompting</a>: Explore and analyze search results for Companies: AI coding project organization, curated methods, prompting</li><li><a href="https://github.com/Tanq16/ai-context">GitHub - Tanq16/ai-context: CLI tool to produce MD context files from many sources, to help interact with LLMs (ChatGPT, Llama3, Claude, etc.).</a>: CLI tool to produce MD context files from many sources, to help interact with LLMs (ChatGPT, Llama3, Claude, etc.). - Tanq16/ai-context</li><li><a href="https://ai.meta.com">no title found</a>: no description found</li><li><a href="https://github.com/sam-hosseini/freelancing-in-finland">GitHub - sam-hosseini/freelancing-in-finland: The ultimate resource for transitioning to freelancing for software developers 👩‍💻🇫🇮</a>: The ultimate resource for transitioning to freelancing for software developers 👩‍💻🇫🇮 - sam-hosseini/freelancing-in-finland</li><li><a href="https://techstartups.com/2025/02/25/deepseek-is-launching-its-next-gen-r2-ai-model-poised-to-surpass-r1-and-shock-the-world-once-again/">DeepSeek R2 AI model launch: Next-gen release expected in May or sooner</a>: Just a month after it sent shockwaves through tech stocks and wiped out over $1 trillion in U.S. stock market value, Chinese AI startup DeepSeek is already preparing to do it again. Now, according to ...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1344045926465142958)** (81 messages🔥🔥): 

> `Aider with Claude Sonnet for free, Gemini 1.5 Pro vs GPT-3.5 for code editing, Groq’s Llama 3 70B for free, Avante uses Groq's Llama-3.3-70b-versatile for applying diffs, Sonnet 3.7 ridiculously overkeen` 


- **Sonnet Isn't Always Free, API is Key**: To use **Claude Sonnet** with aider, you need the API, not just a claude.ai account; there's no free Sonnet API currently.
   - One user suggested alternatives, such as using **Gemini**, which is free, while others opt to pay for **Sonnet** if their budget allows.
- **3.  7 Sonnet Overkeen, Tweaks Too Much**: Users find **Sonnet 3.7** excessively verbose and eager to modify multiple files at once, requiring constant reminders to focus on one file at a time.
   - Some have reverted to **Sonnet 3.5** due to productivity issues, with one user pointing out that *it needs to be reminded every prompt not to go rogue and try and one shot the whole plan.*
- **Peek at Pixels: Multimodal Aider?**: One user explored multimodal workflows for aider, such as having it *see* the output image of a script it generates.
   - Suggestions included using test scripts or screenshots to provide visual feedback to the model, though one user described this as *a game of telephone*.
- **Can't Git It? Aider and Untracked Files**: A user inquired about using aider with files not under Git version control, seeking a repo map that includes untracked files.
   - There was no clear solution provided, but it underscores the current limitation of Aider's reliance on Git for file management.
- **Free LLMs Give Coding a Go**: Users are exploring free LLMs, with **Gemini 1.5 Pro** and **Llama 3 70B on Groq** being viable for code editing, and **DeepSeek R1/V3 on OpenRouter** being a recommended free model.
   - One user with **ChatGPT Pro** suggested one-shotting project creation in it, then handing off the codebase to a free model.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/anthropic.html#thinking-tokens">Anthropic</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/llms.html">Connecting to LLMs</a>: Aider can connect to most LLMs for AI pair programming.</li><li><a href="https://github.com/yetone/avante.nvim/blob/main/cursor-planning-mode.md">avante.nvim/cursor-planning-mode.md at main · yetone/avante.nvim</a>: Use your Neovim like using Cursor AI IDE! Contribute to yetone/avante.nvim development by creating an account on GitHub.</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider is AI pair programming in your terminal
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1344164433408163890)** (2 messages): 

> `R1 verification, Microsoft Trace framework, ax-llm/ax GitHub repository` 


- **Ponders R1 Rigor Verification**: A member inquired about how to verify if something is *full R1*.
- **Requests Microsoft Trace Framework**: A member expressed interest in seeing a framework similar to **ax-llm/ax** built around [Microsoft's Trace framework](https://github.com/ax-llm/ax).
- **Ax LLM Github link**: A member posted a link to the [ax-llm/ax](https://github.com/ax-llm/ax) GitHub repository, describing it as the *"official" unofficial DSPy framework*.



**Link mentioned**: <a href="https://github.com/ax-llm/ax">GitHub - ax-llm/ax: The &quot;official&quot; unofficial DSPy framework. Build LLM powered Agents and &quot;Agentic workflows&quot; based on the Stanford DSP paper.</a>: The &quot;official&quot; unofficial DSPy framework. Build LLM powered Agents and &quot;Agentic workflows&quot; based on the Stanford DSP paper. - ax-llm/ax

  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1344079529920036885)** (2 messages): 

> `Sonnet 3.7 Switchover, Cross-Model Reasoning Standard, Reasoning Parameter` 


- **Sonnet 3.7 Switchover**: A member posted a link to track the [switchover to **Sonnet 3.7**](https://x.com/OpenRouterAI/status/1894520450929119418) over time using the **Versions tab**.
- **Reasoning Parameter: Seamless Model Use**: OpenRouterAI introduced a `reasoning` parameter to make it seamless to use all the models regardless of their internal API, as quoted by Shashank Goyal.
   - The documentation for **reasoning tokens** can be found [here](https://openrouter.ai/docs/use-cases/reasoning-tokens).
- **Cross-Model Reasoning Standard Debuts**: OpenRouterAI introduced a **cross-model reasoning standard** on their API.
   - This allows users to configure **reasoning settings** for **OpenAI**, **Anthropic**, and other models in one central place.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1894520450929119418">Tweet from OpenRouter (@OpenRouterAI)</a>: TIP: Use the Versions tab to see how people are switching to Sonnet 3.7 over timeNow including :thinking 💭</li><li><a href="https://x.com/OpenRouterAI/status/1894801944088039547">Tweet from OpenRouter (@OpenRouterAI)</a>: Today we&#39;re introducing a cross-model reasoning standard on our API.Use it to configure reasoning settings for OpenAI, Anthropic, and other models to come, in one central place.Quoting Shashank Go...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1344039418847170742)** (241 messages🔥🔥): 

> `Reasoning Tokens, Prompt Caching, DeepSeek API pricing, Claude 3.7, OpenRouter API Keys` 


- **Budget Tokens automatically default to 80% of max tokens**: By default, **budget tokens** are set to **80% of max tokens**, up to **32k** as documented in [OpenRouter's documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#budget-tokens).
- **Thinking models might need include_reasoning flag**: A member was not receiving reasoning tokens for **Sonnet-3.7**, but adopting the sample code and passing `extra_body={"include_reasoning": True}` in the API call fixed the issue, though **it's supposed to be true by default**.
   - The team was notified about the unexpected behavior and flagged that this is supposed to be true by default.
- **DeepSeek Cuts API Pricing Drastically**: **DeepSeek** announced a cut in their [API pricing](https://x.com/Sino_Market/status/1894682095706128430), with off-peak discounts up to **75%**, specifically **50% off for DeepSeek-V3** and **75% off for DeepSeek-R1** between 16:30-00:30 UTC.
- **Understanding Moderation on OpenRouter**: A member asked about *self-moderated* models, and it was explained that  *models without* this tag use a moderation endpoint to outright block certain content, whereas *self-moderated* models will still process the API call but may refuse to answer with responses such as *I'm sorry Dave, but I can't do that*.
- **Copilot's O1 Reasoning Model is now free for all users**: Microsoft made **OpenAI’s o1 reasoning model free** for all **Copilot** users, providing unlimited use of this model and Copilot’s voice capabilities, which was covered in [The Verge](https://www.theverge.com/news/619199/microsoft-copilot-free-unlimited-voice-think-deeper-open-ai-o1-reasoning-model-ai).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Sino_Market/status/1894682095706128430">Tweet from CN Wire (@Sino_Market)</a>: 🇨🇳#BREAKING DEEPSEEK LOWERS OFF-PEAK API PRICING BY UP TO 75% - STATEMENT#CHINA #AI #DEEPSEEK  Source:https://mktnews.com/flashDetail.html?id=01954197-1acb-7229-9368-aa13bc03dfaehttps://mktnews.com/...</li><li><a href="https://mktnews.com/flashDetail.html?id=01954197-1acb-7229-9368-aa13bc03dfae">MKT News - Market News for Traders</a>: no description found</li><li><a href="https://mktnews.com/flashDetail.html?id=01954197-1acb-7229-9368-aa13bc03dfae>>>">MKT News - Market News for Traders</a>: no description found</li><li><a href="https://x.com/AndrewCurran_/status/1894355918621749402">Tweet from Andrew Curran (@AndrewCurran_)</a>: Deepseek R2 is arriving early.</li><li><a href="https://x.com/deepseek_ai/status/1894710448676884671">Tweet from DeepSeek (@deepseek_ai)</a>: 🚨 Off-Peak Discounts Alert!Starting today, enjoy off-peak discounts on the DeepSeek API Platform from 16:30–00:30 UTC daily:🔹 DeepSeek-V3 at 50% off🔹 DeepSeek-R1 at a massive 75% offMaximize your r...</li><li><a href="https://openrouter.ai/docs/use-cases/reasoning-tokens#budget-tokens">Reasoning Tokens - Improve AI Model Decision Making</a>: Learn how to use reasoning tokens to enhance AI model outputs. Implement step-by-step reasoning traces for better decision making and transparency.</li><li><a href="https://www.theverge.com/news/619199/microsoft-copilot-free-unlimited-voice-think-deeper-open-ai-o1-reasoning-model-ai">Microsoft makes Copilot Voice and Think Deeper free with unlimited use</a>: No more limits on OpenAI’s o1 reasoning model for Microsoft</li><li><a href="https://openrouter.ai/models?arch=GPT">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://openrouter.ai/docs/use-cases/reasoning-tokens">Reasoning Tokens - Improve AI Model Decision Making</a>: Learn how to use reasoning tokens to enhance AI model outputs. Implement step-by-step reasoning traces for better decision making and transparency.</li><li><a href="https://openrouter.ai/api/v1",">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://feep.life/~feep/fncad/">fnCAD: Geometry from Signed Distance Fields</a>: no description found</li><li><a href="https://www.theverge.com/news/619199/microsoft-copilot-free-unlimited-voice-thi">Microsoft makes Copilot Voice and Think Deeper free with unlimited use</a>: No more limits on OpenAI’s o1 reasoning model for Microsoft</li><li><a href="https://blog.google/technology/developers/gemini-code-assist-free/">Get coding help from Gemini Code Assist — now for free</a>: Announcing a free version of Gemini Code Assist, powered by Gemini 2.0, and Gemini Code Review in GitHub.</li><li><a href="https://codeassist.google/products/individual/?utm_source=google&utm_medium=blog&utm_campaign=FY25-Q1-global-geminicodeassist-for-individuals&utm_content=launch-blog&utm_term=-)">Gemini Code Assist | AI coding assistant</a>: Get AI coding and programming help no matter the language or platform with Gemini Code Assist from Google.</li><li><a href="https://cloud.google.com/blog/products/devops-sre/announcing-the-2024-dora-report?e=48754805)">Announcing the 2024 DORA report | Google Cloud Blog</a>: Key takeaways from the 2024 Google Cloud DORA report that focused on the last decade of DORA, AI, platform engineering and developer experience.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1344393474988052572)** (1 messages): 

> `Perplexity Voice Mode, iOS App Update` 


- **Perplexity Rolls Out Voice Mode**: Perplexity AI has introduced a new **voice mode** feature, allowing users to ask questions and receive real-time audio answers.
   - A [short demo video](https://cdn.discordapp.com/attachments/1047204950763122820/1344393474027421706/Iac2em_OTTWx60h6.mp4?ex=67c0bf7d&is=67bf6dfd&hm=29f3e05084083471219f93c750de0678bdd2d9f1f647780432abcb6a10576dbe&) showcasing the feature was attached.
- **Perplexity Voice Mode launches on iOS, coming soon to Android & Mac**: The **voice mode** is currently available on the **iOS app**, with plans to expand to Android and Mac apps soon.
   - Users are encouraged to update their iOS app to start using the new feature immediately.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1344036572395470940)** (166 messages🔥🔥): 

> `Context window size, Comet Browser launch, Voice mode functionality, Coding with perplexity, Claude 3.7 Sonnet hallucinations` 


- **Context Window Wonders: Perplexity's Token Triumph**: Users discussed context window sizes, noting Perplexity's default reads up to **4,000 tokens** per query, while Pro subscribers can use up to **32,000 tokens** with GPT-4 Omni or Claude 3.5 Sonnet when uploading files.
   - One user's tests indicated potentially higher limits with **o3-mini**, around **128k characters** or **62k tokens**, though the advertised **1 million token** context window remains elusive.
- **Comet Browser Coming Soon**: Perplexity is launching a new agentic browser, **Comet**, *very soon* according to [AravSrinivas](https://x.com/AravSrinivas/status/1894068996950855747).
   - The exact release date and platform support remain unconfirmed, with speculation it may arrive in under a week.
- **Voice Mode gets an Upgrade**: A new **voice mode** was announced with a new UI and the ability to interrupt while it’s talking.
   - While considered an improvement, it's not yet on par with **Microsoft Copilot**, **Grok 3**, or **ChatGPT**, there are some problems in the current version.
- **Code Quest: Debugging with Perplexity**: Users explored using Perplexity for coding, with some suggesting **writing mode** for better code responses.
   - The cost of agentic capabilities, especially on platforms like **Claude**, was noted, with API usage potentially costing **$20 in an afternoon** for large projects.
- **Claude 3.7 Sonnet Suffers Split Personality**: Users noted that **Claude 3.7 Sonnet** sometimes mistakenly identifies itself as **Claude 3 Opus**.
   - This may be due to the training data and the way the models are named, but a ticket was created to address this issue, linked [here](https://discord.com/channels/1047197230748151888/1343923864492838972/1343923864492838972).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/askperplexity/status/1894227029769310243?s=46">Tweet from Ask Perplexity (@AskPerplexity)</a>: Hey! Ask Perplexity answers your questions on X and other online communities. Here&#39;s how it works:1. Tag @AskPerplexity in any post2. Ask any questions you have about the post or its replies3. Ask...</li><li><a href="https://x.com/OpenAI/status/1894454196986155130">Tweet from OpenAI (@OpenAI)</a>: To start, Plus, Team, Enterprise, and Edu users will have 10 deep research queries per month.Pro users will now have 120 deep research queries per month.</li><li><a href="https://chromewebstore.google.com/detail/aahklphdncmbmkmcbgdglefnlfmeegjj?utm_source=item-share-cp)">Instagram Auto-Play Fix - Chrome Web Store</a>: Keep Instagram videos playing when switching tabs or windows</li><li><a href="https://x.com/AravSrinivas/status/1894068996950855747">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Perplexity will be launching a new agentic browser: Comet very soon! </li><li><a href="https://www.youtube.com/watch?v=oEfB2GxAh9k&t=425s"> - YouTube</a>: no description found</li><li><a href="https://www.reddit.com/r/aipromptprogramming/s/My6VlFqtJE">Reddit - Dive into anything</a>: no description found</li><li><a href="https://sonar.perplexity.ai/.">Sonar by Perplexity</a>: Build with the best AI answer engine API, created by Perplexity. Power your products with the fastest, cheapest offering out there with search grounding. Delivering unparalleled real-time, web-wide re...</li><li><a href="https://monnef.gitlab.io/by-ai/2025/pplx-tech-props">Perplexity Tech Props</a>: no description found</li><li><a href="https://monnef.gitlab.io/by-ai/2025/pplx_M_ctx">Million context window on Perplexity?</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1344051558064849068)** (8 messages🔥): 

> `Ruby Script Generation via Perplexity, Anthropic's Pokemon AI Benchmarks, Trump Fires Military Leaders news item, Meta's 200 Billion AI Compute Investment` 


- **Perplexity Generates Ruby Script**: A user leveraged Perplexity AI to generate a simple **Ruby script**, expressing satisfaction with the results and noting AI's aptitude for such tasks, with context provided [here](https://www.perplexity.ai/search/ruby-script-to-find-all-the-fi-nSnE8_UkRmy3RphdnePeDQ).
- **Anthropic's Pokemon AI Benchmarks**: Perplexity AI shared a news item regarding **Anthropic's Pokemon AI benchmarks** [here](https://www.perplexity.ai/page/anthropic-s-pokemon-ai-benchma-BfIkUdgVRKmVAVZLd7.j9w#7fc21566-af56-462f-8f38-b24bb4600dd7).
- **News of Trump Firing Military Leaders**: Perplexity AI surfaced a news item about **Trump** firing military leaders and **Satya Nadella** dismissing AGI benchmarks [here](https://www.perplexity.ai/search/amazing-ai-tGqtBK1WT1q4XDUXUAGCiQ).
- **Meta Explores 200 Billion AI Compute**: Perplexity AI shared a news item that **Meta** is exploring **200 billion AI compute** investment [here](https://www.perplexity.ai/page/meta-explores-200-billion-ai-c-Ri33UtReQwaCKzi7OFhg6Q).



**Link mentioned**: <a href="https://www.youtube.com/embed/DpBeRZmEYNs">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1344038613423362130)** (6 messages): 

> `Perplexity Deep Research API, Developer Meetup SF, Sonar Deep Research in Playground, Uploading files through API` 


- **Perplexity API throws a Cricket Ball**: Perplexity is making the **Deep Research API** available to all developers through the **Perplexity Sonar API** to help people build their custom research agents and workflows, according to [this tweet](https://x.com/aravsrinivas/status/1894477728222777594?s=61).
   - One user suggested using the API on all of cricket data and stats to get into rabbit holes, asking for a good stats repository and **API credits**.
- **Perplexity Hosting SF Developer Meetup**: Perplexity announced an upcoming developer meetup in SF, according to [this Discord post](https://discord.com/channels/1047197230748151888/1155998520822743091/1344108568508498071).
   - The announcement encouraged users in SF who have built something cool with the API to **demo** it at the event, asking for suggestions on where to host the next meetup.
- **Sonar Deep Research & Playground Ambitions**: A user asked if **Sonar Deep Research** will be available in Playground.
   - There were no answers provided in the discussion.
- **Automated file Uploading Sought for Town Meetings**: A user asked if files can be uploaded through the **API** to summarize or work over them.
   - The user would like to summarize town meetings, past and future, and would like to automate the process since the web version isn't sustainable.



**Link mentioned**: <a href="https://x.com/aravsrinivas/status/1894477728222777594?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: As a cricket nerd, throwing Perplexity Deep Research API on all of cricket data and stats and getting into rabbit holes would be fun. Who has a good stats repository here? Anyone who wants to build th...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1344044121815388252)** (76 messages🔥🔥): 

> `Assistants API file search, Claude Plays Pokémon, Claude Sonnet Web vs API, OpenAI Deep Research, Raycast AI Extensions` 


- **OpenAI Adds Assistants API File Search**: OpenAI announced [support for file search with o3-mini and o1](https://platform.openai.com/docs/assistants/tools/file-search) in the **Assistants API**, enabling assistants to access and retrieve information from documents.
- **Anthropic's Claude Plays Pokémon Gains Traction**: **Claude Plays Pokémon** continues as a researcher's personal project, streaming on [Twitch](http://twitch.tv/claudeplayspokemon), and a researcher named [David Hershey](https://x.com/DavidSHershey/status/1894463660279697852) who worked on the project.
- **Sonnet API Answers Diverge from Web**: The **Claude 3.7 Sonnet's** web version has a longer system prompt with contextual information compared to the API version, potentially causing different answers according to [Kimmonismus](https://x.com/kimmonismus/status/1894133480792924249).
- **OpenAI Deep Research Assessed**: [Ben Evans](https://www.ben-evans.com/benedictevans/2025/2/17/the-deep-research-problem) critically reviewed **OpenAI's Deep Research**, highlighting issues with source accuracy, particularly with data from *Statista* and *Statcounter*.
- **Perplexity Launches $50M Seed Fund**: [Perplexity](https://tcrn.ch/41xlXfS) launched a **$50M seed and pre-seed VC fund**, with a $15B valuation offer on the table.
   - The new "Elicit Reports" are considered a *better version of Deep Research*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/vibhuuuus/status/1894857843121234313?s=46">Tweet from Vibhu Sapra (@vibhuuuus)</a>: @swyx @willccbb @kalomaze they posted this architecture (lost the source) but great read imo</li><li><a href="https://x.com/openaidevs/status/1894478106565415328?s=46">Tweet from OpenAI Developers (@OpenAIDevs)</a>: We&#39;ve added support for file search with o3-mini and o1 in the Assistants API. You can create assistants to access and retrieve information from documents, which these reasoning models are particu...</li><li><a href="https://x.com/DavidSHershey/status/1894463660279697852">Tweet from David Hershey (@DavidSHershey)</a>: So, I did a thing 🙂This was really just a fun little side project - I wanted to spend some time working on agents, and Pokemon was the most fun way I could come up with.And then it kinda took off! 3....</li><li><a href="https://x.com/matistanis/status/1894824212382257427?s=46">Tweet from Mati Staniszewski (@matistanis)</a>: Today, we are launching our own Speech to Text model: ElevenLabs Scribe v1. Scribe outperforms current state-of-the-art models on FLEURS and Common Voice benchmarks and finally delivers what’s been fr...</li><li><a href="https://www.ben-evans.com/benedictevans/2025/2/17/the-deep-research-problem">The Deep Research problem &mdash; Benedict Evans</a>: OpenAI’s Deep Research is built for me, and I can’t use it. It’s another amazing demo, until it breaks. But it breaks in really interesting ways.</li><li><a href="https://x.com/willccbb/status/1894477232032076240?s=46">Tweet from will brown (@willccbb)</a>: @aryanagxl generalization for multi-turn tool use + ability to maintain coherence across incredibly long tasks via recursive summarization is like a bigger deal than ARC-AGI to memost reactions i’m se...</li><li><a href="https://x.com/hume_ai/status/1894833497824481593?s=46">Tweet from Hume (@hume_ai)</a>: Today, we’re releasing Octave: the first LLM built for text-to-speech.🎨Design any voice with a prompt🎬 Give acting instructions to control emotion and delivery (sarcasm, whispering, etc.)🛠️Produce ...</li><li><a href="https://x.com/btibor91/status/1894686656139325593?s=46">Tweet from Tibor Blaho (@btibor91)</a>: This is real - the ChatGPT app for Android (1.2025.056 beta) has a new announcement: &#34;Try the GPT-4.5 research preview - Pro users now have access to our newest, largest model.&#34;Quoting Dylan N...</li><li><a href="https://x.com/haileysch__/status/1894422591617790046?t=jABK7mxk5oPmgobo1Bc49g&s=19">Tweet from Hailey Schoelkopf (@haileysch__)</a>: 2 misconceptions from yesterday:- no training on playing videogames. Claude can just do this now- “why isn’t this a demo”Quoting Anthropic (@AnthropicAI) Claude Plays Pokémon continues on as a researc...</li><li><a href="https://x.com/mikeyk/status/1894783669920817321?s=46">Tweet from Mike Krieger (@mikeyk)</a>: In NYC today for the Alexa+ announcement, which uses Claude under the hood for many of its new capabilities. Has been super fun for our team to partner with the Alexa team, on a shared vision of how t...</li><li><a href="https://x.com/paulg/status/1894827577325560215?s=46">Tweet from Paul Graham (@paulg)</a>: Here&#39;s what happened to that startup&#39;s revenue graph in the next year (in blue).Quoting Paul Graham (@paulg) A novel variant of exponential revenue graph. This company is selling something use...</li><li><a href="https://x.com/youssefish/status/1894548592020353311?s=46">Tweet from Youssef Ishak (@youssefish)</a>: (1/) summarizing what &#34;Claude Plays Pokemon&#34; is doing, mainly for meClaude is tasked with playing Pokemon.before every single button press, Claude references both it&#39;s previous context and...</li><li><a href="https://x.com/elicitorg/status/1894772293752266846?s=46">Tweet from Elicit (@elicitorg)</a>: We raised a $22M Series A and are launching Elicit Reports, a better version of Deep Research for actual researchers.Elicit Reports are available for everyone to try right now, for free.👇</li><li><a href="https://x.com/willccbb/status/1894478848923701275?s=46">Tweet from will brown (@willccbb)</a>: @kalomaze yes but also “the model can just self-summarize state to handle things way longer than its context” is crazyfeels R1-ish in that it’s an obvious trick that never worked because the models ju...</li><li><a href="https://x.com/TechCrunch/status/1894514805890830664">Tweet from TechCrunch (@TechCrunch)</a>: Perplexity launches $50M seed and pre-seed VC fund https://tcrn.ch/41xlXfS</li><li><a href="https://x.com/janonacct/status/1894437873082143222?s=46">Tweet from janon (@janonacct)</a>: Claude named it&#39;s rival &#34;Waclaude&#34;</li><li><a href="https://x.com/threepointone/status/1894399506277376369?s=46">Tweet from sunil pai (@threepointone)</a>: cloudflare agents: repo: https://github.com/cloudflare/agents/platform docs: https://developers.cloudflare.com/agents/starter kit: https://github.com/cloudflare/agents-starterso much more to come, shi...</li><li><a href="https://x.com/weberwongwong/status/1894794612398792974?s=46">Tweet from weber (@weberwongwong)</a>: Introducing FLORA, Your Intelligent Canvas.Every creative AI tool, thoughtfully connected.</li><li><a href="https://x.com/prerationalist/status/1894449418813776183?s=46">Tweet from prerat (@prerationalist)</a>: omg claude named his rival WACLAUD??!?!Quoting Anthropic (@AnthropicAI) Claude Plays Pokémon continues on as a researcher&#39;s personal project.Follow along on Twitch: http://twitch.tv/claudeplayspok...</li><li><a href="https://x.com/allen_ai/status/1894415487569969211?s=46">Tweet from Ai2 (@allen_ai)</a>: Introducing olmOCR, our open-source tool to extract clean plain text from PDFs!Built for scale, olmOCR handles many document types with high throughput. Run it on your own GPU for free—at over 3000 to...</li><li><a href="https://x.com/anthropicai/status/1894798008623026503?s=46">Tweet from Anthropic (@AnthropicAI)</a>: Claude will help power Amazon&#39;s next-generation AI assistant, Alexa+.Amazon and Anthropic have worked closely together over the past year, with @mikeyk leading a team that helped Amazon get the fu...</li><li><a href="https://x.com/levelsio/status/1894848949082825176?s=46">Tweet from @levelsio (@levelsio)</a>: I think 5000 people flying but I also see some bots 😅Quoting Thomas Slabbers (@Thomasslabbers) This is pure genius - look at how many people are flying right now! I also found Mars. Pieter this might...</li><li><a href="https://x.com/aravsrinivas/status/1894471526449385687?s=46">Tweet from Aravind Srinivas (@AravSrinivas)</a>: We’re making Deep Research available as an endpoint to all developers through the Perplexity Sonar API to help people build their custom research agents and workflows! Excited to see what people are g...</li><li><a href="https://x.com/steph_palazzolo/status/1894785791018332505?s=46">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: Lots in this morn’s Agenda:- OpenAI leaders have told employees that GPT-4.5 is coming this week- Perplexity is getting inbound offers from VCs at $15b. It likely won’t take the offer but highlights V...</li><li><a href="https://x.com/kimmonismus/status/1894133480792924249">Tweet from Chubby♨️ (@kimmonismus)</a>: Claude 3.7 Sonnet system prompt:&#34;The assistant is Claude, created by Anthropic.The current date is {{currentDateTime}}.Claude enjoys helping humans and sees its role as an intelligent and kind ass...</li><li><a href="https://x.com/ritakozlov_/status/1894394140764594676?s=46">Tweet from rita kozlov 🐀 (@ritakozlov_)</a>: npm i agents-sdk https://blog.cloudflare.com/build-ai-agents-on-cloudflare/</li><li><a href="https://x.com/anthropicai/status/1894419042150027701?s=46">Tweet from Anthropic (@AnthropicAI)</a>: Claude Plays Pokémon continues on as a researcher&#39;s personal project.Follow along on Twitch: http://twitch.tv/claudeplayspokemon</li><li><a href="https://www.youtube.com/watch?v=sHIlFKKaq0A&t=2s"> - YouTube</a>: no description found</li><li><a href="https://x.com/levelsio/status/1894429987006288259">Tweet from @levelsio (@levelsio)</a>: IT WORKS!!!!!A FULL multiplayer with Python websockets server that receives and broadcasts all player positions every 100ms (10 times per second)All code written almost 100% by AI with Cursor and Grok...</li><li><a href="https://www.youtube.com/watch?v=1B9i7FBsRVQ">Claude plays Minecraft!</a>: Not Chatbots again! Not today, thank you!Imagine a world where AI does more than just chat—it thinks, solves, and acts. Enter the realm of Minecraft, where v...</li><li><a href="https://news.ycombinator.com/item?id=43174910">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1344121891316564090)** (2 messages): 

> `LLM Paper Club, Raycast AI` 


- **LLM Paper Club Gets an Update**: Tomorrow's LLM Paper Club's schedule has been updated, and you can sign up [here](https://lu.ma/y3v27e0k).
- **New Raycast AI Podcast is Out**: A short podcast featuring the new **Raycast AI** has been released [here](https://www.youtube.com/watch?v=hoEL6ddVcC0).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=hoEL6ddVcC0">Raycast: Your AI Automation Assistant</a>: Anything an extension can do, can now be done in natural language.See launch video: https://www.youtube.com/watch?v=sHIlFKKaq0A</li><li><a href="https://lu.ma/y3v27e0k">LLM Paper Club (Test Time: s1, Recurrent Depths) · Zoom · Luma</a>: Raphael Kalandadze will cover:s1: Simple test-time scalingScaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach(if time allows)…</li><li><a href="http://Latent.Space)">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1344039290648268820)** (56 messages🔥🔥): 

> `Local LLM training code, Cohere models in OpenAI SDK, Open Source vs Paid Code, OpenAI SDK Integration` 


- **High Schooler Hopes to Hack LLM Training**: A high school student, after two years of work, claims to have developed code for **local LLM training** with full ownership and control, and is seeking a company to purchase it.
   - However, a community member advised that this code would need to compete with existing **open-source solutions** like *llama factory* and *Unsloth* to be viable.
- **Open Source Advocate Advises Against Selling**: A community member argued that there's no incentive for people to pay for code that doesn't surpass existing **free, open-source alternatives** like [Unsloth](https://github.com/unslothai/unsloth).
   - The high school student conceded and announced plans to **open source** their project.
- **Cohere Models Now Cozying Up in OpenAI SDK**: A member announced that [Cohere models are now accessible through the OpenAI SDK](https://x.com/itssandrakublik/status/1894791769117650998?s=46&t=r1mNPSgnb3pIcbR7vcCi-g).
   - This integration supports streaming, tool calls, and structured outputs, as detailed in the [Quickstart Guide](https://docs.cohere.com/docs/compatibility-api).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/itssandrakublik/status/1894791769117650998?s=46&t=r1mNPSgnb3pIc">Tweet from Sandra Kublik (@itsSandraKublik)</a>: You can now access Cohere models directly through the OpenAI SDK :) Check out our Quickstart Guide for Python, TS, & cURL demos, plus streaming, tool calls, structured outputs, and more. Happy buildin...</li><li><a href="https://x.com/itssandrakublik/status/1894791769117650998?s=46&t=r1mNPSgnb3pIcbR7vcCi-g">Tweet from Sandra Kublik (@itsSandraKublik)</a>: You can now access Cohere models directly through the OpenAI SDK :) Check out our Quickstart Guide for Python, TS, & cURL demos, plus streaming, tool calls, structured outputs, and more. Happy buildin...</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1344348069621399572)** (1 messages): 

> `Compatibility API, OpenAI SDK, Cohere Models` 


- **Cohere Models Now Accessible via OpenAI SDK**: The new **Compatibility API** mirrors the OpenAI SDK format, enabling seamless switching of OpenAI-based apps to **Cohere’s models** without major refactoring, as detailed in the [documentation](https://docs.cohere.com/docs/compatibility-api).
   - To switch, users need to change the base_url to *https://api.cohere.ai/compatibility/v1* and set their **COHERE_API_KEY**, supporting multiple languages like Python, TypeScript, and cURL.
- **Compatibility API Supports Advanced Features**: The **Compatibility API** supports advanced features such as **structured outputs (JSON Schema)**, **tool calls**, and **state management**.
   - The announcement directs users to the <#1168578329423642786> channel for questions and feedback.



**Link mentioned**: <a href="https://docs.cohere.com/docs/compatibility-api">Using Cohere models via the OpenAI SDK — Cohere</a>: The document serves as a guide for Cohere&#x27;s Compatibility API, which allows developers to seamlessly use Cohere&#x27;s models using OpenAI&#x27;s SDK.

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1344339386715607122)** (7 messages): 

> `Cohere API blocking VPS, Token counting changes, Cohere API availability` 


- **Cohere API Blocks VPS Access**: A user reported that **Cohere API calls** are being **blocked** when made from a **VPS** and was directed to contact [support@cohere.com](mailto:support@cohere.com) for assistance.
- **Token Counting Modifications**: A member inquired about how the use of **OpenAI API's 128K context window** would affect token counting, given it's smaller than the context window available when using the Cohere API directly.
- **Cohere API Future Availability**: A member questioned whether there would be modifications to the **direct Cohere API**, potentially affecting its future availability.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1344421450064593038)** (3 messages): 

> `HuggingFace deprecation, RAG tool` 


- **HuggingFace Deprecation Designation Digression**: A member inquired about marking a repo as deprecated with a link to a newer version on **Hugging Face**.
   - They later clarified that the deprecation feature only applies to models, not datasets, resolving their own question.
- **RAG Tool Recommendations Requested**: A member asked which **RAG** tool is currently the best for personal users.
   - No specific RAG tools were mentioned, so further discussion is required.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1344036738208895099)** (18 messages🔥): 

> `KV Cache Compression, Activation Steering, Deepseek DeepGEMM Kernel, Data Mixing Optimization` 


- **Tokencat gives Much Higher Compression Ratios**: The 'tokencat' addition to MLA achieved much higher compression ratios, as detailed in the **GoldFinch paper**.
   - This compression relies on other dimensions of compression across layers.
- **Deepseek's DeepGEMM Kernel is ground-breaking**: Members reading through the new **DeepGEMM** release by Deepseek found it impressive, noting it considers **H800 limitations** and optimizes efficiency within bandwidth and compute limits.
   - It was also noted that it's an open-source Gemm kernel utilizing **TMA** extensively.
- **Hardware remains the ultimate 'moat'**: The sentiment is that efficient implementations of architecture kernels like **MLA**, **DeepGEMM**, or communication strategies like **DeepEP** do not provide a significant competitive advantage.
   - One member quipped that *the only moat is hardware*.
- **MixMin algorithm for optimal data mixtures**: The [MixMin paper](https://arxiv.org/abs/2502.10510) introduces a gradient-based approach called **MixMin** for optimizing data mixtures in machine learning pipelines, framing it as a bi-level convex objective.
   - The paper claims that **MixMin** improved data mixtures in language modeling and chemistry tasks using less than **0.2%** additional compute.



**Link mentioned**: <a href="https://arxiv.org/abs/2502.10510">MixMin: Finding Data Mixtures via Convex Minimization</a>: Modern machine learning pipelines are increasingly combining and mixing data from diverse and disparate sources, e.g., pre-training large language models. Yet, finding the optimal data mixture is a ch...

  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1344166712114810911)** (2 messages): 

> `Bigger Models, Ensembling, Flops` 


- **Big Models are Flop Heavyweights**: Bigger models need more **flops** and do more actual *"work"* for the same amount of samples.
   - Training an ensemble of models simultaneously as one model will consume more **flops**, but have better loss/accuracy for the same sample count.
- **Ensemble Gains Shrink as Models Grow**: Someone asked if gains from **ensembling** shrink as models get larger.
   - They assume big models have ensembles inside them, and if that's the case, then they assume ensembling gains do shrink as models get larger.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1344209278604410901)** (5 messages): 

> `SAEs, Weight tying in SAEs, Orthogonal features in SAEs` 


- **Weight Tying No Longer Hot in SAEs**: Members discussed that weight tying isn't typically done in the most recent **Sparse Autoencoders (SAEs)**.
   - They noted that assuming features are roughly **orthogonal**, the input and output weights should be the same, like multiplying by an orthogonal matrix and its transpose.
- **Orthogonal Features Impact SAE Weights**: The conversation highlights the importance of **orthogonal features** in determining the weights within **SAEs**.
   - If features are assumed to be roughly orthogonal, input and output weights should align, similar to how an orthogonal matrix and its transpose revert to the original.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1344320511022334033)** (27 messages🔥): 

> `lm-evaluation-harness setup in a notebook, Local LLM API endpoints running via TRT, GPQA implementation` 


- **Notebook Setup Help Arrives!**: A member inquired about running the lm-evaluation-harness in a notebook, instead of commandline, and another member said you can run any command line command in a notebook by prefixing it with `!`.
   - There is also a (poorly documented) Python API, and a suggestion to replace `python main.py` with `!lm_eval`.
- **TemplateAPI TRT Triton Troubles?**: A member asked about support for local LLM API endpoints running via TRT for Completions, and referenced making `curl` requests to `localhost:8000/v2/models/triton_model/generate`.
   - Another member responded that it should work if you modify the `_create_payload`, and the parse logprobs and generation functions in the [openai_completions.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/af2d2f3e79140ae5b6833ce1046f1519dc08b9df/lm_eval/models/openai_completions.py#L44) file.
- **GPQA Implementation Questioned**: A member asked about the GPQA implementation, and whether it was tested, after looking at the Open LLM Leaderboard.
   - Another member mentioned that the leaderboard uses a multiple-choice variant (MMLU style but letters with parenthesis (A), (B) etc.) and just the diamond subset of 200 rows from the [GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa?row=6).
- **GPQA Diamond Results Analyzed**: After noticing reports of low scores, members investigated the GPQA diamond subset and discussed potential tokenization issues, noting that the questions are difficult and the choices seem similar.
   - One member reported getting **0.4848** by extracting the last [A-D] after instructing the model to conclude with *the best answer is [A-D]*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Idavidrein/gpqa?row=6">Idavidrein/gpqa · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/af2d2f3e79140ae5b6833ce1046f1519dc08b9df/lm_eval/models/openai_completions.py#L44)">lm-evaluation-harness/lm_eval/models/openai_completions.py at af2d2f3e79140ae5b6833ce1046f1519dc08b9df · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1344052529477976126)** (3 messages): 

> `GQA in NeoX, Llama models export issues` 


- **GQA Creates Glitches in GPT-NeoX?**: A member asked if there are known issues with exporting **Llama models** with **GQA** in **NeoX**, stating that models break when using **GQA** but work perfectly without it.
   - They wondered if the export script needs changes and linked to a [related pull request on GitHub](https://github.com/EleutherAI/gpt-neox/pull/1315/files).
- **Speculation on GQA implementation**: The member speculated the root cause of the glitches might have to do with Grouped Query Attention implementation.
   - This may or may not be the actual reason.



**Link mentioned**: <a href="https://github.com/EleutherAI/gpt-neox/pull/1315/files">fix a GQA issue (#1314) by tiandeyu-cs · Pull Request #1315 · EleutherAI/gpt-neox</a>: fix a GQA issue (#1314)do not create a fake head dim and split the &#39;mixed_x_layer&#39; into QKV layers directly.

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1344380379687030876)** (5 messages): 

> `Modular MAX and Mojo repo changes, Mojo's standalone language status` 


- **Modular Simplifies MAX and Mojo Repos**: Modular is simplifying its repo structure for **MAX** and **Mojo** to ease contributions to docs and the standard library, and to centralize bug reports and feature requests, as detailed in [this forum thread](https://forum.modular.com/t/upcoming-changes-to-our-github-repositories/648).
- **Community Questions Mojo's Language Status**: A community member inquired if the repo changes signal an internal shift away from viewing or prioritizing **Mojo** as its own standalone language.



**Link mentioned**: <a href="https://forum.modular.com/t/upcoming-changes-to-our-github-repositories/648">Upcoming changes to our GitHub repositories</a>: Tomorrow (February 27), we’re streamlining our GitHub repositories! The max repo is merging into the mojo repo, bringing everything under one roof. A new subdirectory will house the Mojo standard libr...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1344040906013151293)** (44 messages🔥): 

> `EmberJSON, Mojo auto-parallelization, algorithm package isn't open source, speedup using list.get_unsafe, smart iterators in Mojo` 


- **Modular Community Channel Lags Behind EmberJSON Patches**: The **25.1 patch** of *emberjson* isn’t yet available from the Modular community channel, even though the updated recipe was merged a while ago.
   - Users are bumping into version mismatches between their **Mojo version** and what’s expected by *emberjson*.
- **Mojo Doesn't Magically Parallelize**: There's no **auto-parallelization** in the Mojo compiler; developers must explicitly use the **stdlib** to parallelize tasks.
   - Users inquired about leveraging all system resources (multiple CPU cores) for Mojo programs, but explicit parallelization is currently required.
- **Algorithm Package Shrouded in Mystery**: The *algorithm package* isn't open source and is not visible in the **stdlib repo**.
- **Smart Pointers and Iterator Soundness**: Discussion on smart pointers and their potential to make C++ as safe as **Circle** or **Rust** links to a blogpost discussing [smart pointers](https://jacko.io/smart_pointers.html).
   - A member asked about having sound iterators in Mojo, and whether iterator invalidation issues handled in **Safe Rust** is possible, especially for algorithms involving swapping objects in a collection.
- **MLIR Dialect Doc Drought**: Mojo uses various **MLIR dialects** (kgen, pop, lit, etc.) with their own ops and types, but most of them are undocumented and aren't used in the stdlib or loaded into the MLIR context of the Mojo runtime.
   - The lack of documentation is because these dialects are part of a **private contract** shared by the stdlib, MAX, and the compiler, and they may not be fully tested, have unstable APIs, or contain proprietary value-adds.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jacko.io/smart_pointers.html">Smart Pointers Can't Solve Use-After-Free</a>: no description found</li><li><a href="https://github.com/axiomhq/mojo-hyperloglog">GitHub - axiomhq/mojo-hyperloglog</a>: Contribute to axiomhq/mojo-hyperloglog development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1344042356307001385)** (35 messages🔥): 

> `Alignment tradeoff, DTMF, Google Experiments, Apple speech-to-text Trump issue, Claude 3.7` 


- **Alignment Efforts Cause Bias Elsewhere**: Members discussed how adjusting a model to favor one behavior introduces **misalignment elsewhere**, embodying the **alignment tradeoff** problem.
   - It was mentioned that *alignment is always relative*, reflecting biases in data and the values imposed by those controlling the model.
- **Decoding Dual-Tone Multi-Frequency Signaling**: A user shared a [Wikipedia link](https://en.wikipedia.org/wiki/DTMF) to explain **Dual-Tone Multi-Frequency (DTMF) signaling**, a telecommunication system using voice-frequency bands over telephone lines.
   - The link provided detailed information on the **history, technical specifications, and applications of DTMF** in telecommunications.
- **Google's Half-Baked Implementations**: Members discussed how [Google](https://learning.google.com/experiments/learn-about?src=signup) has many great ideas, but often suffers from **bad or incomplete implementations**.
   - Theorizing this may stem from a history of developing tools primarily for internal use, hindering their ability to create broadly useful products in recent years.
- **Apple's Speech-to-Text Types 'Trump' Instead of 'Racist'**: [Apple](https://www.bbc.com/news/articles/c5ymvjjqzmeo) is working to fix its **speech-to-text** tool after users reported it typed "Trump" when they spoke the word "racist".
   - An expert suggested the issue was likely due to someone intentionally altering the underlying software, rather than a plausible speech recognition error.
- **AI Model Convergence Towards Platonic Representation**: Discussion centered on a paper ([https://arxiv.org/html/2405.07987v1/](https://arxiv.org/html/2405.07987v1/)) claiming AI model representations are **converging towards a shared statistical model of reality**, termed the **platonic representation**.
   - Skepticism was voiced regarding how non-deterministic systems can have deterministic manners and converge, questioning if models, which represent meaning, are accurately reflecting reality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://learning.google.com/experiments/learn-about?src=signup">Learn About</a>: no description found</li><li><a href="https://www.bbc.com/news/articles/c5ymvjjqzmeo">Apple AI tool transcribed the word &#x27;racist&#x27; as &#x27;Trump&#x27;</a>: Experts have questioned the company&#x27;s explanation that it is due to the two words being similar.</li><li><a href="https://www.tiktok.com/@user9586420191789/video/7472830639327366446)">TikTok - Make Your Day</a>: no description found</li><li><a href="https://arxiv.org/html/2405.07987v1/">The Platonic Representation Hypothesis</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/DTMF">DTMF - Wikipedia</a>: no description found</li><li><a href="https://www.itu.int/rec/T-REC-Q.23/en)">Q.23�:�Technical features of push-button telephone sets</a>: no description found
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1344105412617568288)** (5 messages): 

> `LIMO, Speculative Decoding, ipfs_accelerate_py` 


- **LIMO Reveals Less Data is More for Reasoning**: The paper [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387) observes that training on a smaller amount of reasoning data is superior to using amounts typical of other fine tuning tasks.
   - It starts to examine why the low data requirements for reasoning training has been found a few times, though without much hypothesis on why.
- **Speeds Up Generation with Speculative Decoding**: [Speculative decoding](https://lmstudio.ai/docs/advanced/speculative-decoding) is a technique that can substantially increase the generation speed of large language models (**LLMs**) without reducing response quality using a smaller, faster "draft" model.
   - During generation, the draft model rapidly proposes potential tokens (subwords), which the main model can verify faster than it would take it to generate them from scratch.
- **Accelerate IPFS with Python**: [ipfs_accelerate_py](https://github.com/endomorphosis/ipfs_accelerate_py/tree/main/ipfs_accelerate_py) is a Python tool to contribute to endomorphosis's IPFS acceleration project on GitHub.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.03387">arXiv reCAPTCHA</a>: no description found</li><li><a href="https://github.com/endomorphosis/ipfs_accelerate_py/tree/main/ipfs_accelerate_py">ipfs_accelerate_py/ipfs_accelerate_py at main · endomorphosis/ipfs_accelerate_py</a>: Contribute to endomorphosis/ipfs_accelerate_py development by creating an account on GitHub.</li><li><a href="https://lmstudio.ai/docs/advanced/speculative-decoding">Speculative Decoding | LM Studio Docs</a>: Speed up generation with a draft model
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1344091159353360456)** (2 messages): 

> `ChatGPT plugins, Mystery model` 


- ****ChatGPT Plugins** Get Deep Research**: A user shared a [screenshot](https://cdn.discordapp.com/attachments/853983317044756510/1344091159085191258/Screenshot_2025-02-26_at_00.37.47.jpg?ex=67c04eb0&is=67befd30&hm=e4d1e9e607580413c5c9c25fb98178f5359728d6425b89c4b7222d752246cb0b) of **Deep Research** for **ChatGPT Plus** users.
- **Mystery Model Surfaces**: A user shared a picture of a **mystery model** floating around.
   - No further details were given.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1344037998437732372)** (40 messages🔥): 

> `CSV indexing, ModernBert models, Nomic Embed Text V2 Deployment, GPT4ALL roadmap, File splitting` 


- **Gigantic CSVs Give Indexing Ire**: A member inquired about the time it would take to embed/index two **277 GB CSV** files, presumably after a recent data breach of **NPD data**.
   - A member recommended splitting the files into **1 GB** chunks for easier indexing with a *simple* indexing software like [GSplit](https://www.gdgsoft.com/gsplit).
- **Multilingual Models ModernBERT Musings**: A member sought details on the training of multilingual models based on the **ModernBERT** architecture, linking to the [ModernBERT GitHub repository](https://github.com/AnswerDotAI/ModernBERT).
   - They clarified they were specifically interested in NomicAI's fine-tuned models like [nomic-embed-text-v2](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-unsupervised) .
- **Nomic Embed V2: No Official Ollama News**: A member asked when **Nomic Embed Text V2** would be deployable in **Ollama/GPT4ALL**, expressing a preference for deployment methods not reliant on coding skills.
   - Another member noted the recent announcement of **Nomic Embed Text V2** on the [Nomic AI blog](https://www.nomic.ai/blog/posts/nomic-embed-text-v2).
- **GPT4ALL Gets Gemini-Inspired Guidance**: A member requested a roadmap for future **GPT4ALL** updates, particularly a *LIVE mode* similar to **Google Gemini**.
   - Another member suggested adding voice recognition **STT** and **TTS** for output, linking to a [YouTube tutorial](https://www.youtube.com/watch?v=6zAk0KHmiGw) on creating a **GPT4ALL** voice assistant.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.nomic.ai/blog/posts/nomic-embed-text-v2">Nomic Embed Text V2: An Open Source, Multilingual, Mixture-of-Experts Embedding Model</a>: Nomic advances the state of the art with a multilingual mixture of experts embedding model</li><li><a href="https://github.com/AnswerDotAI/ModernBERT">GitHub - AnswerDotAI/ModernBERT: Bringing BERT into modernity via both architecture changes and scaling</a>: Bringing BERT into modernity via both architecture changes and scaling - AnswerDotAI/ModernBERT</li><li><a href="https://www.youtube.com/watch?v=6zAk0KHmiGw">Create a GPT4ALL Voice Assistant in 10 minutes</a>: Use Python to code a local GPT voice assistant. In this video we learn how to run OpenAI Whisper without internet connection, background voice detection in P...</li><li><a href="https://github.com/nomic-ai/contrastors">GitHub - nomic-ai/contrastors: Train Models Contrastively in Pytorch</a>: Train Models Contrastively in Pytorch. Contribute to nomic-ai/contrastors development by creating an account on GitHub.</li><li><a href="https://www.gdgsoft.com/gsplit">GSplit - Free File Splitter - Split Any File Fast - Split Text and Log Files</a>: GSplit is a free file splitter that splits any file into smaller files called pieces. Fast, easy-to-use and efficient with lots of customization options.</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-unsupervised">nomic-ai/nomic-embed-text-v2-moe-unsupervised · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1344037797782229103)** (23 messages🔥): 

> `Claude Code line numbers, Model Context Protocol (MCP), MCP Server Implementation, SSE server` 


- **Claude Code Uses Line Numbers for Precise Edits**: Members mentioned that **Claude Code** includes the line numbers of every line when it reads files, improving the reliability of code editing and reducing context usage in projects like [mcp-language-server](https://github.com/isaacphi/mcp-language-server).
   - One member noted that line numbers are crucial for automatic debuggers, enabling precise breakpoint placement and integration with tools like **Pylance**.
- **MCP Server Implementations Yield Mixed Results**: A member shared their experiments with building their own **MCP servers** and integrating them with [mcp-cli](https://github.com/chrishayuk/mcp-cli) using local LLMs (**Mistral** and **Llama3.1**).
   - While **Llama3.1** was initially too aggressive with tool usage, **Mistral** later started hallucinating tool usage instead of actually calling them.
- **MCP Ownership Clarified**: When asked *Who owns mcp? Is it Anthropic?*, it was clarified that MCP is an **open-source project** driven by **Anthropic** for now.
   - The long-term vision involves unbiased foundation/committee stewardship as discussed in [this GitHub discussion](https://github.com/orgs/modelcontextprotocol/discussions/133#discussioncomment-11773450).
- **SSE Testing Server Solution Found**: A member seeking an **SSE server** suitable for testing, similar to the everything server for stdio, received a helpful suggestion.
   - It was pointed out that the **official everything server** has SSE capabilities, making it ideal for testing purposes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/chrishayuk/mcp-cli">GitHub - chrishayuk/mcp-cli</a>: Contribute to chrishayuk/mcp-cli development by creating an account on GitHub.</li><li><a href="https://github.com/isaacphi/mcp-language-server">GitHub - isaacphi/mcp-language-server: Model Context Protocol (MCP) server that interacts with a Language Server</a>: Model Context Protocol (MCP) server that interacts with a Language Server - isaacphi/mcp-language-server</li><li><a href="https://github.com/orgs/modelcontextprotocol/discussions/133#discussioncomment-11773450)">Community Governance / Open Foundation Creation or Donation · modelcontextprotocol · Discussion #133</a>: Pre-submission Checklist I have verified this would not be more appropriate as a feature request in a specific repository I have searched existing discussions to avoid duplicates Your Idea MCP is e...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1344110124108615681)** (3 messages): 

> `FastMCP, Typescript, Custom Authentication` 


- **FastMCP Patches Gnarly Race Conditions**: Users of [FastMCP](https://github.com/punkpeye/fastmcp), a **TypeScript framework** for building MCP servers, are urged to upgrade to the latest version to patch some *gnarly race conditions*.
   - The upgrade is highly recommended to ensure stability and reliability for applications using this framework.
- **FastMCP Gets Fancy Authentication**: **FastMCP** now supports [custom authentication](https://github.com/punkpeye/fastmcp/releases/tag/v1.20.0), allowing developers to authenticate SSE clients using a custom function.
   - This enhancement offers more control and flexibility in securing **MCP servers**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/punkpeye/fastmcp/releases/tag/v1.20.0">Release v1.20.0 · punkpeye/fastmcp</a>: 1.20.0 (2025-02-26)FastMCP now allows you to authenticate SSE clients using a custom function:import { AuthError } from &quot;fastmcp&quot;;const server = new FastMCP({  name: &quot;My Server&quot;,  ...</li><li><a href="https://github.com/punkpeye/fastmcp">GitHub - punkpeye/fastmcp: A TypeScript framework for building MCP servers.</a>: A TypeScript framework for building MCP servers. Contribute to punkpeye/fastmcp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1344356321595756544)** (19 messages🔥): 

> `StatefulDataLoader, single device recipes, truncation and skipping` 


- **`StatefulDataLoader` propagation kicks off**: Members discussed propagating the use of [`StatefulDataloader`](https://github.com/pytorch/torchtune/issues/2439) to all recipes in TorchTune, in order to successfully set up **step-based checkpointing** and track/store the dataloader state.
   - It was determined that multiple PRs were welcome and members volunteered to take single device recipes, beginning with *lora_dpo_single_device* and *knowledge_distillation_single_device*.
- **Single Device Recipes, MPS Backend OK'd**: In working on the [Add `StatefulDataloader` to remainder of recipes](https://github.com/pytorch/torchtune/issues/2439) task, a member asked if relying on **MPS backend** would be okay for single device recipes, and the response was *Yeah that's fine*.
   - One member volunteered to start with one to avoid blocking closing the parent issue.
- **CI Assistance Requested for `truncation and skipping`**: A member requested someone to start CI for [PR 2419](https://github.com/pytorch/torchtune/pull/2419) without merging while another member was offline.
   - They mentioned it was their last attempt for the day.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/2441">[WIP] Add `StatefulDataLoader` to all recipes except knowledge_single by krammnic · Pull Request #2441 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)Please link to any issues this PR addresses.#2431#2439...</li><li><a href="https://github.com/pytorch/torchtune/issues/2439">Add `StatefulDataloader` to remainder of recipes · Issue #2439 · pytorch/torchtune</a>: Goal: Propagate use of StatefulDataloader to all recipes in torchtune. Why? In order to successfully set up step-based checkpointing, we need to be able to track and store the dataloader state. How...</li><li><a href="https://github.com/pytorch/torchtune/pull/2442">[WIP] add `StatefulDataLoader` to `knowledge_distillation_single_device` recipe by jxtngx · Pull Request #2442 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)Please link to any issues this PR addresses.#2439Chang...</li><li><a href="https://github.com/pytorch/torchtune/pull/2419">[RFC] truncation and skipping by krammnic · Pull Request #2419 · pytorch/torchtune</a>: #2344 Mention two important points related to our data loading and processing. This RFC works on both of these aspects.TruncationCurrently, we don&amp;#39;t support truncation in both right and left ....
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1344038314931523686)** (15 messages🔥): 

> `VRAM Efficiency, AI HackXelerator, Scammer Alert, Regional prompting` 


- **Hunyuanvideogp V5 breaks VRAM laws?**: A member shared a Reddit post about [Hunyuanvideogp V5 breaking the laws of VRAM](https://www.reddit.com/r/StableDiffusion/comments/1iybxwt/hunyuanvideogp_v5_breaks_the_laws_of_vram/).
   - Another member commented that *it's not actually breaking the laws of VRAM, it's just using VRAM more efficiently*, calculating that the numbers are less than the VRAM capacity with the formula **Width * Height * FPS * Length**.
- **London, Paris, Berlin AI HackXelerator announced**: The **London, Paris, Berlin AI HackXelerator™ - LPB25** combines the community-generated fun of a hackathon with the depth and support of a full-scale accelerator, from **April 5-25, 2025** ([kxsb.org](https://www.kxsb.org/lpb25)).
   - The event will bring together **500 creatives, devs and designers**, and features streams for **AI music, image, video, fashion, gaming** and combinations, supported by brands like **Central Saint Martins, Station F, Mistral AI, Hugging Face, Luma AI, Vultr, AMD, Nvidia**.
- **Scammer Alert!**: A member reported that `@w361_emp is SCAMMER` and that *he stole my portfolio*.
   - The member requested others to be careful.
- **Regional prompting with LoRAs is possible**: A member asked if it was possible to use LoRAs on certain areas of an image, like using an *orc LoRA only on the mouth part*.
   - Another member suggested searching for **regional prompting**, noting that it had been added to ComfyUI at some point.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kxsb.org/lpb25">London-Paris-Berlin HackXelerator™ by KXSB</a>: Join LPB25, a 20-day AI HackXelerator™ uniting 500+ creators across London, Paris, and Berlin. Explore GenAI innovation through music, art, film, gaming, and fashion with expert mentoring and prizes. ...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1iybxwt/hunyuanvideogp_v5_breaks_the_laws_of_vram/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1344138762451419218)** (5 messages): 

> `good PRs for new people, TestSpeed.test_sum Performance issues, arange GROUP optimization, BEAM search adjustments` 


- ****Tinygrad Welcomes New Contributors****: There are [good first PRs](https://github.com/tinygrad/tinygrad/issues/9262) available for new contributors, some of which are relatively straightforward.
   - The issue involves methods to add to tensor.py such as **as_strided**, **topk**, and **bitwise_xor**.
- ****TestSpeed.test_sum Faces Performance Bottleneck****: A member reported struggling with `TestSpeed.test_sum` and made changes that make the **AST for GROUP operations** sensible, hitting a snag where optimizations for larger tensors are not being found by BEAM search.
   - The issue is that the **BEAM search** does not explore the option of four successive **OptOps**, which are needed to optimize (4096,4096) tensors, because the first three alone are quite slow.
- ****Arange Group Optimization Breaks CI****: The **arange GROUP optimization** is not being applied, causing an extra inner loop for arange operations and breaking the arange test.
   - The member is seeking advice on whether to adjust **BEAM search** or where to add new patterns for horizontal adds or loop unrolling.
- ****BEAM Search Strategy Questioned****: The discussion questions whether time should be invested in adjusting **BEAM search** to find specific **OptOps** for performance improvements.
   - The author also inquires about where new patterns for horizontal adds or loop unrolling should be implemented, suggesting the lowerer or expander.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/issues/9262">Changes requested from pytorch backend · Issue #9262 · tinygrad/tinygrad</a>: Methods to add to tensor.py as_strided -- https://pytorch.org/docs/stable/generated/torch.as_strided.html topk -- https://pytorch.org/docs/main/generated/torch.topk.html bitwise_xor -- https://pyto...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9190/files">[Bounty] Made TestSpeed.test_sum yellow on Macs with LLVM by josephsweeney · Pull Request #9190 · tinygrad/tinygrad</a>: To make this happen, I enabled GROUP OptOps&amp;#39;s on devices without local variables (CLANG and LLVM), by just adding an extra reduce instead on emitting locals. The other necessary changes came d...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1344074858258104362)** (7 messages): 

> `UOp Signatures, safetensors computation graphs, TestLinearizerFailures` 


- **Community Seeks Guidance on UOp Signatures**: A community member is seeking assistance with understanding how to determine the signature of each **UOp**'s `src` and `args`, including finding documentation or code references that define constraints between **Enums**.
   - They find the [spec.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/codegen/linearizer/specs.py) file insufficient for detailed **Op** descriptions and meanings, asking *Is the assumption that their meanings can be inferred from their names?*
- **Debate Arises: Safetensors, Graphs, and Pickles?**: A member asked about encoding computation graphs within **safetensors**, mentioning a desire for a universal encoding convention similar to ONNX, but a community expert clarified that *safetensors doesn't save the computational graph, only tensors.*
   - Another member referenced a [previous discussion](https://discord.com/channels/1068976834382925865/1070745817025106080/1329504751016083601) and suggested pickling the jitted function as an alternative for exporting/importing the computational graph.
- **Debugging TestLinearizerFailures on macOS**: A community member is debugging [TestLinearizerFailures.test_failure_53](https://github.com/tinygrad/tinygrad/blob/master/test/test_linearizer.py) and encountering issues, specifically an infinite loop on macOS that isn't present on NV/Linux.
   - The issue seems related to rewriting the **BLOCK op** on macOS, but they haven't found enough information about this **op** to resolve the problem.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1344208019764019231)** (2 messages): 

> `Agent Memory, Feedback Mechanism for Agents` 


- **Boosting Agent Memory with a Simple Trick**: Members in the channel discussed that the **Agent memory can be improved** using a simple hack.
   - It was explained that if the **Agent has GPT4 access** it tends to use it's memory more effectively, and the quality of the responses are higher due to the model being better, in comparison to using **GPT3.5**.
- **Feedback Mechanism for Enhanced Agent Learning**: The channel debated the importance of **feedback mechanisms** for agents to improve their learning.
   - One member suggested using the **new annotation tool** that allows to collect feedback on the agent's results.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/)** (1 messages): 

hritabanghosh: https://discord.gg/ETxqXCfh
  

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

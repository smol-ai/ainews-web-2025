---
id: MjAyNS0w
title: "OAI and GDM announce IMO Gold-level results with natural language reasoning, no specialized training or tools, under human time limits"
date: '2025-07-21T05:44:39.731046Z'
---

**General-purpose RL is all you need.**

> AI News for 7/18/2025-7/21/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (227 channels, and 21117 messages) for you. Estimated reading time saved (at 200wpm): 1729 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

This time last year, GDM announced that [AlphaProof and AlphaGeometry2](https://news.smol.ai/issues/24-07-25-ainews-alphaproof-alphageometry2-reach-1-point-short-of-imo-gold) (the latest in a [long series of Alpha* work](https://x.com/deedydas/status/1946987560875766212)) had perfectly solved 4 out of the 6 IMO 2024 problems, falling 1 point short of the Gold medal cutoff. However that system needed over 60 hours for some problems, much longer than the 4.5 hours allowed for humans.

This year, both OpenAI ("an experimental research model, not released in GPT5" - [their solutions here](https://github.com/aw31/openai-imo-2025-proofs/)) and GDM ("[Advanced version of Gemini Deep Think](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)" - [their solutions here](https://storage.googleapis.com/deepmind-media/gemini/IMO_2025.pdf)) announced* full solves of 5 out of the 6 problems ([P6 is typically the hardest](https://x.com/ErnestRyu/status/1946698896375492746)) all within 4.5 hours, achieving IMO Gold and resolving a [3 year old AI bet between Paul Christiano and Eliezer Yudkowsky](https://www.notion.so/really-delete-1cc3eeb8e42a8082891cf977ab364a0f?pvs=21), where Paul had put the probability at <4% in Feb 2022. Interestingly, the market estimated probability of this success trended *DOWN* even through the release of o1 and new reasoner models, and only shot up to 50-80% after the GDM announcement last year:

![](https://resend-attachments.s3.amazonaws.com/U57EKNnXxiRSMig)

The even more surprising element of this Gold prize not documented by that bet is that it was done **WITHOUT** use of specialized tools like Lean or even access to the Internet; just pure in-weights reasoning (aka "[purely via search in token space](https://x.com/fchollet/status/1947337944215523567)"):

- from [Oriol Vinyals](https://x.com/OriolVinyalsML/status/1947341047547199802) (some [pushback in the fine print](https://x.com/VraserX/status/1947368827253076001))
    
    ![](https://resend-attachments.s3.amazonaws.com/jgT7oIHSut6DWT6)
    
- from [Alexander Wei](https://x.com/alexwei_/status/1946477742855532918): "We reach this capability level not via narrow, task-specific methodology, but by breaking new ground in general-purpose reinforcement learning and test-time compute scaling." and [Jerry Tworek](https://x.com/millionint/status/1946551400365994077?s=46): "we did very little IMO-specific work, we just keep training general models, all natural language proofs, no evaluation harness":
    
    ![](https://resend-attachments.s3.amazonaws.com/rdKul9ieUsAC8Ks)
    

Mathematicians seem mostly [unthreatened](https://x.com/ErnestRyu/status/1946700798001574202) and are welcoming the result, although [Terence Tao had some strong doubts about methdology and medal claim](https://x.com/pli_cachete/status/1946692267915304991?s=46) (which were [answered](https://x.com/BorisMPower/status/1946859525270859955)).

Thanks to the combinatorics problem P6 that requires creativity, In 2025, [26 humans](https://x.com/damekdavis/status/1947357679040569520/photo/1) remain better than AI at the IMO. [Try it](https://x.com/deedydas/status/1946250774960537927) if you wish.

![](https://resend-attachments.s3.amazonaws.com/cVkjruv83R3VCmj)

In case you were wondering, here is how SOTA [released models did on the same IMO:](https://x.com/deedydas/status/1946244012278722616?s=46) "[not even bronze](https://matharena.ai/imo/)".

![](https://resend-attachments.s3.amazonaws.com/TUcyMIly1MYzBeY)

- [*with some controversy](https://x.com/morqon/status/1947344915945451848), OpenAI announced first, but we'd just recommend looking past that [PR drama](https://x.com/ErnestRyu/status/1946699212307259659)*. other labs like [Harmonic](https://x.com/HarmonicMath/status/1947023450578763991) may also have accomplished this milestone but have held off til July 28th to announce, as [IMO is reported to have requested](https://x.com/zjasper666/status/1947013036382068971?s=46).
- * [*where is Grok 4*](https://x.com/nsaphra/status/1946804513114882227?s=46)?

---

# AI Twitter Recap

**AI Achieves IMO Gold: The Race, Results, and Reaction**

- **OpenAI and Google DeepMind both announce Gold Medal performance on the International Math Olympiad (IMO)**: **OpenAI** was first to announce, with [@gdb](https://twitter.com/gdb/status/1946479692485431465) and [@polynoamial](https://twitter.com/polynoamial/status/1946526143433015349) detailing that an **experimental reasoning LLM** solved **5 of 6** problems under the same rules as humans (4.5 hours, no tools), producing **natural language proofs**. Shortly after, **Google DeepMind** announced that an advanced version of **Gemini Deep Think** also achieved a **gold-medal score of 35/42**, with the result being officially validated by IMO judges, as shared by [@fchollet](https://twitter.com/fchollet/status/1947337944215523567) and [@koraykv](https://twitter.com/koraykv/status/1947335096740049112). [@YiTayML](https://twitter.com/YiTayML/status/1947350087941951596) noted that this general deep think model will be shipped to users in the future.
- **Community Reaction and Scrutiny**: The announcements sparked significant discussion and some controversy. [@SebastienBubeck](https://twitter.com/SebastienBubeck/status/1946577650405056722) called it a **"moon-landing moment"** for AI, highlighting that a next-word prediction machine produced genuinely creative proofs. However, [@Mihonarium](https://twitter.com/Mihonarium/status/1947072974621982839) reported that the IMO had asked AI companies to wait a week before announcing results to avoid overshadowing the human participants. This led to criticism of **OpenAI's** timing, particularly after **Google DeepMind** waited for official confirmation, a move that [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1947339774257402217) stated "wins my respect." Further analysis by [@lmthang](https://twitter.com/lmthang/status/1946960256439058844) clarified that without the official marking guideline, a medal claim is not final, and a one-point deduction would result in a **Silver, not Gold**. An independent analysis of LLMs on the **2025 IMO** from the **MathArena** team was also shared by [@hardmaru](https://twitter.com/hardmaru/status/1946942279807308210).
- **The "AGI Bar" Debate**: The IMO achievement has led to a renewed debate on what milestones signify progress towards AGI. [@DrJimFan](https://twitter.com/DrJimFan/status/1946593477460189340) argues that the **"Physical Turing Test"**, such as an AI cooking dinner in any kitchen, is a harder problem due to **Moravec's paradox**. This sentiment was echoed by [@jxmnop](https://twitter.com/jxmnop/status/1946675650686746879), who joked that AI can achieve this mathematical feat but still can't reliably book a trip to Boston. Conversely, [@*aidan_clark*](https://twitter.com/_aidan_clark_/status/1947178461765775510) sets the bar at the replacement of all human labor by **nanobot swarms**.

**New Models, Architectures, and Performance**

- **Qwen3-235B-A22B Release and Architecture**: **Alibaba's** Qwen team released an updated **Qwen3-235B-A22B**, a non-reasoning model that [@huybery](https://twitter.com/huybery/status/1947345040470380614) says shows significant improvements. [@scaling01](https://twitter.com/scaling01/status/1947350866840748521) notes it now beats reasoning models like **Kimi-K2**, **Claude-4 Opus**, and **DeepSeek V3** on benchmarks like **GPQA**, **AIME**, and **ARC-AGI**. [@rasbt](https://twitter.com/rasbt/status/1947393814496190712) provided a detailed technical breakdown, comparing its architecture to **Kimi 2**: **Qwen3** is **4.25x** smaller overall, has fewer active parameters (**22B** vs. **32B**), and uses **128** experts per MoE layer versus Kimi's **384**.
- **Kimi K2 Technical Report and Performance**: The **Kimi K2** technical report was released, revealing details about the **~1T parameter** model, as shared by [@scaling01](https://twitter.com/scaling01/status/1947384137892966693). Community members like [@pashmerepat](https://twitter.com/cline/status/1946389822043504745) noted that on real-world tasks (not benchmarks), telemetry shows **Kimi K2** outperforming **Gemini**.
- **GPT-5 Rumors and Model Routers**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1946777842131632427) shared rumors that **GPT-5** will not be a single model but a system of multiple models with a router that switches between reasoning, non-reasoning, and tool-using variants. This sparked discussion, with [@scaling01](https://twitter.com/scaling01/status/1946903963200262523) expressing a preference for manual model selection over an automatic router to avoid compute-saving measures that could degrade performance for pro users.
- **Architectural Reviews and Other Model Updates**: [@rasbt](https://twitter.com/rasbt/status/1946549778319339931) published a comprehensive review of 2025's major LLM architectures, covering **DeepSeek-V3**, **Kimi 2**, and techniques like **Multi-head Latent Attention**, **NoPE**, and **shared-expert MoEs**. **Microsoft** open-sourced the pre-training code for **Phi-4-mini-Flash**, a SoTA hybrid model, as highlighted by [@algo_diver](https://twitter.com/algo_diver/status/1946397862767767921).

**Agentic Systems, Tooling, and Developer Experience**

- **Perplexity Comet and Generative UI**: **Perplexity** launched **Comet**, which [@AravSrinivas](https://twitter.com/AravSrinivas/status/1946398572955766979) demonstrated in an end-to-end deep research workflow. The platform features **Generative UI**, which creates interactive cards on the fly for tasks like sending emails or joining calendar invites, shifting Perplexity from an "ask anything" to a ["do anything" company](https://twitter.com/AravSrinivas/status/1947175881203683577). The product has seen rapid adoption, with [@AravSrinivas](https://twitter.com/AravSrinivas/status/1947173109083332988) noting its browser now ranks above Wikipedia's Comet page on Google search results.
- **Cline's Open Source Strategy and Incentive Alignment**: [@cline](https://twitter.com/cline/status/1946704096888533005) published a detailed thread explaining their decision to open-source their AI coding assistant and not resell inference. By separating the "harness" from the "model calls," they argue their incentives are aligned with the user's goal of getting maximum capability, as they can't degrade performance to improve margins.
- **New Tools and Developer Integrations**: A new CLI tool called `gut` was released, acting as an AI agent for git that translates natural language into git commands, highlighted by [@jerryjliu0](https://twitter.com/jerryjliu0/status/1947026118260949146). The adoption of `llms.txt` continues, with [@jeremyphoward](https://twitter.com/jeremyphoward/status/1946386696691683473) sharing its implementation in the **Gemini API docs** to create model-friendly documentation. **Hugging Face Inference Providers** are now fully **OpenAI client compatible**, as announced by [@reach_vb](https://twitter.com/reach_vb/status/1946499807159226445).
- **Agent Design and Frameworks**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1946358807875244398) shared best practices for crafting structured output schemas for LLMs, such as limiting nesting depth and using optional fields. **LangChain** announced it is working towards a **v1.0 release**, which will feature revamped docs and general agent architectures built on **LangGraph**, according to [@hwchase17](https://twitter.com/hwchase17/status/1947376920355917909).

**AI Research, Infrastructure, and Technical Concepts**

- **GPU Infrastructure and Optimization**: [@tri_dao](https://twitter.com/tri_dao/status/1947188520340398200) noted that the hierarchical layout of **CuTe** (part of **CUTLASS 3.x**) is a powerful abstraction for high-performance GPU kernels and was the inspiration for rewriting **FlashAttention 2**. The **vLLM** project highlighted the importance of **prefix caching** for agentic workflows, noting it is enabled by default with an efficient implementation to improve performance for append-only contexts [@vllm_project](https://twitter.com/vllm_project/status/1946575947295322171).
- **The Product Management Bottleneck**: In a widely shared post, [@AndrewYNg](https://twitter.com/AndrewYNg/status/1947308544916889979) introduced the concept of the **"Product Management Bottleneck,"** arguing that as agentic coding accelerates development, the new bottleneck becomes deciding *what* to build. He advocates for PMs who can use data to refine their gut instincts and make high-quality product decisions quickly.
- **Core AI Concepts and Papers**: **François Chollet** offered a definition of intelligence, stating it is not a collection of skills but **the efficiency with which you acquire and deploy new skills**, making benchmark scores potentially misleading [@fchollet](https://twitter.com/fchollet/status/1946668452045029861). A comprehensive **160+ page survey on Context Engineering** was shared by [@omarsar0](https://twitter.com/omarsar0/status/1946660448742343013). [@francoisfleuret](https://twitter.com/francoisfleuret/status/1947026623968244008) argued that the principle of **"denoising"**—creating order from chaos by reversing degradation—is a powerful and fundamental concept that can take AI anywhere.
- **Open Source Datasets**: The **Hermes 3** dataset from **Nous Research** became the #1 trending dataset on Hugging Face, as celebrated by [@Teknium1](https://twitter.com/Teknium1/status/1946824832764785135) and the Nous team.

**AI Industry, Companies, and Geopolitics**

- **Company Culture and Execution**: A story about **Windsurf's** acquisition by **Cognition** was shared by [@russelljkaplan](https://twitter.com/russelljkaplan/status/1946382813546045505), with [@swyx](https://twitter.com/swyx/status/19464654856346827) commenting on the team's "insane execution, timing, and strategy." [@xikun_zhang_](https://twitter.com/xikun_zhang_/status/1946813185958510613) described the intense, focused culture at **OpenAI**, where a talented team repeatedly executes "like a small startup" to ship products like the ChatGPT agent in just over two months.
- **US vs. China in Open Source AI**: The AI community noted the strong performance of Chinese models, with [@bigeagle_xd](https://twitter.com/bigeagle_xd/status/1946426600838586476) pointing out the top 4 open models are from China. [@DaniYogatama](https://twitter.com/DaniYogatama/status/1947087827721912485) provided a structural analysis of why the US is lagging in open-source models, citing a lack of backing from hyperscalers for new labs and organizational issues in large US companies. This was echoed by [@francoisfleuret](https://twitter.com/francoisfleuret/status/1946817554967626176), who contrasted the West's "blasé" attitude toward engineering with China's enthusiasm.
- **The Business of AI and Founder Incentives**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1947309109902037056) described a market transition from **"pay for the process" to "pay for the results,"** where AI agents can deliver outcomes like video ads or websites immediately, opening up markets to businesses that couldn't afford traditional agency processes. [@random_walker](https://twitter.com/random_walker/status/1947259631257932250) gave detailed advice for a research career, emphasizing the need to pick long-term projects, build a distribution channel (e.g., social media, blogs), and treat research like a startup with multiple shots on goal.

**Humor/Memes**

- **Gary Marcus's Timing**: [@scaling01](https://twitter.com/scaling01/status/1946530148813025544) highlighted a tweet from **Gary Marcus** claiming "No pure LLM is anywhere near getting a silver medal in a math olympiad," posted just hours before **OpenAI** announced its gold-level result.
- **The Pain of AI Agents**: [@mckaywrigley](https://twitter.com/jayelmnop/status/1946432132424818943) shared a screenshot of **Claude** drawing ASCII art and running `time.sleep(28800)`, deciding it was "time to go to bed." [@swyx](https://twitter.com/swyx/status/1946369984009306126) pleaded for an end to "flight booking agent demos."
- **Relatable Tech Life**: [@willdepue](https://twitter.com/willdepue/status/1946656141427060816) humorously asked why compasses don't work in New York, blaming a "great chunk of magnetite below bushwick." [@inerati](https://twitter.com/inerati/status/1947049407783817424) compared slow QR code menus to a "portal to hell, where the data is actually stored." [@QuixiAI](https://twitter.com/QuixiAI/status/1946894174734684652) lamented that "vibe coding tools need to learn to use the debugger" instead of adding print statements.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-235B-A22B-2507 Launch and Anticipation

- [**Qwen3-235B-A22B-2507 Released!**](https://x.com/Alibaba_Qwen/status/1947344511988076547) ([Score: 379, Comments: 121](https://www.reddit.com/r/LocalLLaMA/comments/1m5owi8/qwen3235ba22b2507_released/)): **Alibaba's Qwen team released Qwen3-235B-A22B-Instruct-2507 and its FP8 variant, shifting from their prior hybrid thinking mode to dedicated separate training for Instruct and Thinking models. This approach, informed by community feedback, reportedly yields improved overall model quality and performance on agent-oriented tasks. Technical benchmarks and release information are detailed in the [Hugging Face model card](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507), with accessible chat and download options on Qwen Chat, Hugging Face, and ModelScope.** Comments highlight that OpenAI may need to enhance safety testing in light of Qwen's progress, and there is recognition of Alibaba's leadership in advancing open-source LLMs. The discontinuation of hybrid mode is generally viewed as a positive quality-improving move.
    - The Qwen team has shifted strategy with the Qwen3-235B-A22B-Instruct-2507 model, now offering separate Instruct and Thinking models rather than a hybrid approach, in response to community feedback aimed at improving task specialization and quality. This release also features an FP8 variant for users prioritizing computational efficiency.
    - Benchmark results for Qwen3-235B-A22B-Instruct-2507 are available on its Hugging Face model card, with some users highlighting its considerable margin over Kimi and expressing interest in direct benchmarking comparisons against the latest DeepSeek May 2024 release (see https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507 ).
    - A technical highlight is the model's native context length of 262,144 tokens, allowing for extremely long context handling natively without external context window workarounds.
- [**Imminent release from Qwen tonight**](https://i.redd.it/um0pwye549ef1.png) ([Score: 309, Comments: 73](https://www.reddit.com/r/LocalLLaMA/comments/1m5n148/imminent_release_from_qwen_tonight/)): **The attached image is a screenshot of a tweet by Junyang Lin saying, "no hybrid thinking mode tonight," referencing anticipated releases from the Qwen project, possibly Qwen3-Coder, Qwen3-VL, or QwQ. The post and linked tweets indicate the release will be open source, possibly including model weights, and are generating excitement and speculation about the technical features (e.g., whether 'hybrid thinking mode' will be included). Technical discussion in comments expresses particular interest in a Qwen Coder model and anticipation for any release.** Commenters prioritize the release of a 'Qwen Coder' model, with debate over feature expectations (e.g., coding capabilities or other innovations like 'hybrid thinking mode'). There is consensus that open-sourcing is highly valued.
    - A user inquires about the meaning of 'hybrid thinking mode,' speculating it may refer to a model's ability to select between different reasoning approaches or decide whether to use external tools. This raises the possibility of dynamic model-tool orchestration within Qwen models, which would align with trends in modular AI architectures allowing models to delegate subtasks when appropriate (e.g., code execution, search augmentation).
    - Requests for 'Qwen Coder' and support for 'vision' functionality in Qwen3 indicate user demand for highly specialized variants (e.g., code generation or multimodal vision models). This reflects broader industry trends where LLM providers diversify their models for domain-specific excellence, suggesting future Qwen releases might target these specialized applications depending on community feedback and strategic goals.
- [**Qwen3-235B-A22B-2507**](https://i.redd.it/w2uh7h5lg9ef1.png) ([Score: 162, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1m5ox8z/qwen3235ba22b2507/)): **The image presents a bar graph from an official @Alibaba_Qwen tweet, announcing the release of Qwen3-235B-A22B-2507—a new variant of the Qwen3-235B family, explicitly running in 'non-thinking' (standard instruct) mode rather than the previously-used hybrid or thinking modes. The chart benchmarks performance against top competitors (Kimi K2, Claude Opus 4, Deepseek-V3-0324) on tasks including GPQA, AIME25, LiveCodeBench v6, Arena-Hard v2, and BFCL-v3, with Qwen3-235B-A22B-2507 frequently leading or matching state-of-the-art results. Notably, the release clarifies that 'thinking mode' is absent in this version, focusing solely on instruction following capabilities.** Commenters are impressed with the model's results, questioning whether it could truly surpass models like Kimi K2 in code tasks and expressing skepticism about possible 'benchmaxxing'. There is initial confusion about the mode (thinking vs. non-thinking) used in the benchmark, but clarification is provided: this is a non-thinking (standard instruct) version.
    - A detailed benchmark comparison table was shared for DeepSeek-V3, DeepSeek-R1, Kimi-K2, and several Qwen3-235B-A22B variants (Base, Non-thinking, Thinking, and Instruct-2507). Metrics span general tasks (MMLU-Redux, IFEval), Math & STEM (GPQA-Diamond, MATH-500, AIME), coding tasks (LiveCodeBench, MultiPL-E), agent/tool use, and multilingualism. Key highlights include Qwen3-235B-A22B-Instruct-2507 achieving the highest scores in SimpleQA (54.3), MultiPL-E (87.9), and several multilingual benchmarks, while DeepSeek-R1 leads in STEM and coding with top marks on LiveCodeBench (73.3), HMMT 2025 (79.4), and AIME 2024 (91.4).
    - There is technical discussion regarding the nature of the Qwen3-235B-A22B model, specifically distinguishing between 'non-thinking', 'thinking', and 'instruct' versions. Clarification is given that the discussed model is the standard instruct variant and not the enhanced 'thinking' mode, which may explain some performance differences on certain benchmarks.
    - Another technical point is the mention of 'benchmaxxing,' raising concerns about whether benchmark results may be overly optimized or unrepresentative. If the benchmarks are not benchmaxxed, the model's coding performance (potentially exceeding Kimi K2) would be particularly notable according to LiveCodeBench and MultiPL-E results.

### 2. Custom LLM Projects and System Prompt Extraction

- [**I extracted the system prompts from closed-source tools like Cursor & v0. The repo just hit 70k stars.**](https://www.reddit.com/r/LocalLLaMA/comments/1m5gwzs/i_extracted_the_system_prompts_from_closedsource/) ([Score: 238, Comments: 33](https://www.reddit.com/r/LocalLLaMA/comments/1m5gwzs/i_extracted_the_system_prompts_from_closedsource/)): **A GitHub repository ([link](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools)) curates 'system prompts' extracted from proprietary AI tools (e.g., Cursor, Vercel's v0), which reveal advanced prompt architectures used for high-quality LLM outputs. The repo includes anonymized, detailed prompt snippets showing techniques like step-by-step reasoning enforcement, agentic role definition, session-state injection, and strict output structuring, aiming to provide replicable blueprints for designing complex prompting strategies; a full technical analysis is posted [here](https://medium.com/@lucknitelol/the-open-source-project-that-became-an-essential-library-for-modern-ai-engineering-67021b50acee?source=user_profile_page---------0-------------d9a574987030----------------------). Redacted Cursor prompt examples illustrate explicit instructions about session state, communication format, and turn-taking logic.** Top technical debates focus on LLM's ability to reliably process extensive, multi-instruction prompts without hallucination, and skepticism about authenticity—whether prompts recalled by LLMs are genuine or could be adversarially 'seeded' by companies to mislead extraction attempts.
    - apnorton raises a core technical concern regarding the trustworthiness of extracted system prompts, questioning whether prompts self-reported by LLMs are subject to hallucination or deliberate obfuscation. The comment suggests companies might seed LLMs with decoy prompts to mislead extraction attempts, which, if true, could undermine prompt extraction methodologies and reliability of resulting repositories.
    - freecodeio expresses skepticism about LLM behavior when presented with extremely long or complex system prompts (thousands of instructions), questioning the likelihood of effective instruction following versus increased hallucinations. This highlights ongoing discussion in the field about model instruction following limits and prompt engineering scalability.
    - SandFragrant6227 contributes by sharing a link to "secret" Gemini CLI system instructions, providing an external source for system prompt structure and content designed for a different closed-source system, potentially allowing technical comparisons of prompt strategy across tools.
- [**I posted 3 weeks ago about training my own model. Progress report.**](https://www.reddit.com/r/LocalLLaMA/comments/1m52h10/i_posted_3_weeks_ago_about_training_my_own_model/) ([Score: 210, Comments: 52](https://www.reddit.com/r/LocalLLaMA/comments/1m52h10/i_posted_3_weeks_ago_about_training_my_own_model/)): **The post details the ongoing training of a custom LLM, 'Libremodel I (Gigi),' designed to fit within 24GB RAM with a total model size of 960M parameters and trained on 19.2B tokens, adhering to chinchilla optimal scaling. The architecture includes innovations such as flash attention v2, a 3:1 Grouped-Query Attention (GQA) ratio, a 3k token context window, and sink tokens; the dataset is 70% Project Gutenberg and 30% US Congressional Reports (Govremorts), exclusively English, with a projected training cost of ~$500 and anticipated final loss between 2.3-2.6. Notable implementation challenges included correcting flawed streaming dataset logic (causing repeated data passes) and tuning an excessively high learning rate that spiked loss, both resolved mid-training; technical details and progress are available at [libremodel.xyz](http://libremodel.xyz/).** Commenters asked for specifics on training data size and open sourcing plans, reflecting interest in dataset transparency and model/code availability for reproducibility.
    - A commenter asks about the size of the training dataset in gigabytes, highlighting the importance of data volume for model quality and generalization capacity. The dataset size directly impacts compute requirements and both training and validation performance.
    - Technical curiosity is expressed regarding the model's validation loss, which is a key indicator of overfitting. Monitoring the validation loss compared to the training loss helps ensure the model generalizes beyond the training data and doesn't simply memorize the dataset.
    - One user requests details about reproducing the learning curve graph, as well as guidance for recognizing meaningful metrics. This reflects interest in the practical process: tracking loss curves, understanding the implications of different patterns, and monitoring validation metrics for proper training diagnostics.

### 3. LLM Hardware Innovations and Local Model Preferences

- [**Rockchip unveils RK182X LLM co-processor: Runs Qwen 2.5 7B at 50TPS decode, 800TPS prompt processing**](https://www.cnx-software.com/2025/07/18/rockchip-unveils-rk3668-10-core-arm-cortex-a730-cortex-a530-soc-with-16-tops-npu-rk182x-llm-vlm-co-processor/#rockchip-rk182x-llm-vlm-accelerator) ([Score: 114, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1m5fmlp/rockchip_unveils_rk182x_llm_coprocessor_runs_qwen/)): **Rockchip announced the RK182X, a dedicated RISC-V LLM/VLM co-processor, claiming over 2000 tokens/s prefill and 120 tokens/s decode for 7B models (e.g., Qwen2.5, DeepSeek-R1) in INT4/FP4, representing a reported 8–10x performance jump over prior NPUs. The chip includes 2.5–5GB ultra-high-bandwidth memory and offers PCIe/USB3/Ethernet interfaces; prompt processing speed improvements are notable (cited at 800 tps) and directly address large-context on-device inference bottlenecks. In parallel, their RK3668 SoC advances Armv9.3 compute, NPU (16 TOPS), and media features in a 5–6nm node, with up to 100 GB/s memory bandwidth, targeting high-performance edge AI and media workloads. External details: [Rockchip RK3668 and RK182X co-processor](https://www.cnx-software.com/2025/07/18/rockchip-unveils-rk3668-10-core-arm-cortex-a730-cortex-a530-soc-with-16-tops-npu-rk182x-llm-vlm-co-processor/).** Technical discussion in comments highlights the unprecedented prompt processing throughput, and notes the RK3668's potential as a mobile inference platform due to high RAM support (potentially 48GB) and advanced NPU integration. There is critique of Qualcomm's ecosystem, citing restrictive developer tooling and limited support for newer model formats (GGUF), contrasting with Rockchip's push for flexible, edge-optimized NPU architecture.
    - Discussion highlights the significant difference in prompt processing speed versus decode speed on the RK182X, with users noting acceleration is likely due to hardware-level optimizations—prompt/token embedding and KV cache ingestion are often more parallelizable than autoregressive decoding, which typically serves as the bottleneck.
    - A detailed technical breakdown of the RK3668 SoC is provided, noting unannounced Armv9.3 cores (Cortex-A730/A530), a new RKNN-P3 NPU at 16 TOPS, an Arm Magni GPU (1-1.5 TFLOPS), LPDDR5/5x/6 support up to 100GB/s, and expectations for 48GB RAM support based on Rockchip's previous RK3588 platform—positioning it as a strong candidate for mobile on-device LLM inference.
    - There is critical discussion of Qualcomm's developer tooling and NPU accessibility—the Hexagon Tensor Processor is reported as difficult to use for GGUF models unless Qualcomm engineers intervene, contrasting with positive experiences using Adreno GPU OpenCL as a lower power alternative for local inference on Snapdragon platforms. This underscores the necessity for easy-to-integrate NPUs for edge AI workloads.
- [**Which local 100B+ heavy weight models are your favorite and why?**](https://www.reddit.com/r/LocalLLaMA/comments/1m58695/which_local_100b_heavy_weight_models_are_your/) ([Score: 108, Comments: 100](https://www.reddit.com/r/LocalLLaMA/comments/1m58695/which_local_100b_heavy_weight_models_are_your/)): **The post reviews preferences among local LLMs exceeding 100B parameters, focusing on models like Mistral_large-Instruct, Qwen3-235B, Deepseek variants, Kimi-K2, Ernie-4.5-300B, and Llama3.1-405B. Commenters note that Llama3.1-405B, while not state-of-the-art in intelligence, still excels in knowledge retrieval, especially in trivia compared to Deepseek and Kimi. Qwen3-235B-A22B gains attention for its high intelligence, efficient inference (even compared to 'Llama4'), and accessibility due to its portioned active parameters (22B), though it was overshadowed by concurrent Llama4 and the popularity of Qwen3-32B. A Mac Studio M3 user's ranking highlights use-case distinctions: Kimi K2 (general), R1 0528 (coding/science/medical), Qwen 235b (math, long context), and agentic/fast models like Maverick for specialized workflows.** Debate centers on the tradeoffs of inference speed, knowledge depth, and accessibility: Qwen3-235B is argued to be underrated for its combination of raw capability and low hardware requirements versus more resource-heavy models.
    - Llama 3.1 405B is highlighted for its extensive factual recall and depth of knowledge, outclassing models like Deepseek and Kimi in raw trivia tasks. However, it is no longer considered SOTA for general intelligence, suggesting a trade-off between factual breadth and reasoning capabilities in very large models.
    - Qwen3-235B-A22B is noted for its unusual efficiency, enabling near-Llama4 speeds while offering high intelligence, especially due to its 22B active parameters. The model's underwhelming uptake is partly attributed to its partial system memory loading requirement and competition from both Llama4 (which received more attention for speed) and Qwen3-32B (which was already strong and easier to run).
    - User experience with hardware like the Mac Studio M3U demonstrates that top models are chosen for domain-optimized strengths—Kimi K2 excels at general tasks, R1 0528 for technical/scientific work, Qwen 235B in math and long-context use cases, and Maverick for agentic workflows with fast prefill. As anecdotal performance data, Kimi K2 reportedly achieves `1.2-3 tokens/sec`, writing up to 1300 lines of code in 90 minutes, illustrating that even slower models may be practical for large outputs in real workflows.
- [**The reason why local models are better/necessary.**](https://i.redd.it/vdngpglhb8ef1.png) ([Score: 209, Comments: 110](https://www.reddit.com/r/LocalLLaMA/comments/1m5iymb/the_reason_why_local_models_are_betternecessary/)): **The image highlights a major limitation of cloud-based or filtered LLMs: when queried with 'how to hide from authorities,' the provided search and AI outputs include a refusal ('I can't assist with that'), demonstrating the content restrictions implemented for ethical/safety reasons. This is contrasted against search results and is used as an argument for the necessity of local models, which allow users to bypass such restrictions and obtain unrestricted outputs for potentially sensitive or controversial queries. The discussion references practical implications for writers and researchers needing realistic details that filtered LLMs may refuse to provide, and links to a philosophical paper on safety and control (https://philpapers.org/rec/SERTHC).** Comments debate the ethics and utility of AI safety restrictions, with some arguing that 'AI Safety is the dumbest discussion in history' and others noting legitimate use cases (like novel-writing) that are hindered by current LLM restrictions.
    - One commenter raises a technical privacy concern regarding cloud-based LLMs, noting that uploading proprietary or private code/data into centralized AI systems creates a permanent record that could be accessed or exploited as providers monetize or potentially sell user data. They argue that local models prevent this risk by keeping sensitive computation and information entirely under a user's direct control.
    - A recurring theme is the deliberate alteration or 'lobotomization' of large commercial models, i.e., applying strict safety filters or alignment interventions to prevent them from generating content considered "unsafe". This prompts technical skepticism about LLM openness and the impact of such interventions on model capabilities, with local models posited as a way to avoid unwanted modifications and retain full feature sets.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Gemini Deep Think and AI Performance at IMO Controversy

- [**Gemini with Deep Think achieves gold medal-level**](https://www.reddit.com/gallery/1m5o1ll) ([Score: 812, Comments: 261](https://www.reddit.com/r/singularity/comments/1m5o1ll/gemini_with_deep_think_achieves_gold_medallevel/)): **Google DeepMind reports that its Gemini model, enhanced with a 'Deep Think' methodology, has reached gold-medal level performance on the International Mathematical Olympiad (IMO) benchmark, verified by third-party graders ([announcement](https://x.com/googledeepmind/status/1947333836594946337?s=46)). The approach is noted for being 'end-to-end in natural language,' implying the model no longer relies on external symbolic tools or programmatic routines to solve complex math problems, but operates solely through language-based reasoning. This milestone suggests significant advances in reasoning abilities and autonomy in LLMs.** Commenters highlight the credibility added by third-party grading compared to other major claims (e.g., OpenAI), and debate the implications of end-to-end natural language systems moving away from tool use for advanced problem-solving.
    - Several comments highlight that Gemini's IMO results were officially graded by members of the International Mathematical Olympiad (IMO) organization, in contrast to some previous models that were not evaluated by external judges. This adds credibility to the results and addresses concerns about possible inflation or cherry-picking, which were notable issues in the OpenAI case.
    - Commenters discuss that Google Gemini employed an 'end-to-end in natural language' approach and reportedly used a 'parallel multi-agent system' (DeepThink). This signifies a move away from tool-augmented or step-by-step systems, marking a technical departure from models reliant on external tools. However, OpenAI's approach to the IMO was less transparent about its underlying agent architecture, limiting direct technical comparison.
    - There are technical concerns raised over the transparency and direct comparability of models' performances. For example, Google provided Gemini with curated high-quality math solutions and specific hints for IMO-style problems as part of the system's prompt engineering. In contrast, OpenAI claims their entry was not specifically tailored for the IMO. This raises questions about data preparation, overfitting, and the true generality of the models' reasoning capabilities. Terence Tao's cautionary perspective is referenced regarding apples-to-apples comparison across AI models on the IMO benchmark.
- [**Gemini Deep Think achieved Gold at IMO**](https://www.reddit.com/r/singularity/comments/1m5o1jh/gemini_deep_think_achieved_gold_at_imo/) ([Score: 389, Comments: 59](https://www.reddit.com/r/singularity/comments/1m5o1jh/gemini_deep_think_achieved_gold_at_imo/)): **Google DeepMind's 'Gemini Deep Think' has achieved gold medal performance at the International Mathematical Olympiad (IMO), solving 5 out of 6 problems—matching the prior result by OpenAI's model and doing so end-to-end in natural language (English). According to the official announcement (see [Google DeepMind statement](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) and [official tweet](https://x.com/GoogleDeepMind/status/1947333836594946337)), Gemini Deep Think will soon enter Beta and later integrate into Gemini Ultra. Key technical details include new model advances not specific to math, supplemented by math corpus training and targeted hints for IMO-style solutions.** Commenters highlight that while Gemini did not solve the final (6th) problem, matching OpenAI's previous results, its end-to-end natural language approach is notable. There is technical speculation about whether these modeling advances are transferable to tasks outside mathematics, potentially signaling broader generalization improvements in next-generation AI agents.
    - Gemini Deep Think achieved a gold medal at the IMO by solving 5 out of 6 problems, matching the performance of OpenAI's entries. There was particular technical curiosity about whether the model could solve the most difficult (6th) problem, which remains a significant challenge for AI models in mathematical reasoning competitions.
    - A key insight from the technical discussion is that Gemini solved all IMO problems *end-to-end in natural language* (English), rather than using code or formal proofs, demonstrating progress in natural language reasoning and stepwise solution generation. (See [solution screenshot](https://preview.redd.it/2go4fglzc9ef1.png?width=1920&format=png&auto=webp&s=c9838edc8ac74b6cd327edf2aff5051efe32b456))
    - Commenters note that, according to the DeepMind article ([link](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)), Google focused on model improvements not just specific to math, but also incorporated targeted math corpus training and IMO answer hints. This has led to speculation on whether these advances in mathematical reasoning can transfer to non-math, general AI tasks.
- [**Gemini Deep Think achieved Gold at IMO**](https://www.reddit.com/r/Bard/comments/1m5o0o8/gemini_deep_think_achieved_gold_at_imo/) ([Score: 136, Comments: 12](https://www.reddit.com/r/Bard/comments/1m5o0o8/gemini_deep_think_achieved_gold_at_imo/)): **Google DeepMind's advanced version of Gemini with "Deep Think" capability has achieved the 'gold medal standard' at the International Mathematical Olympiad (IMO), verified by third-party evaluation, as detailed in the [official blog post](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) and the [announcement](https://x.com/GoogleDeepMind/status/1947333836594946337?t=MFfLjXwjyDg_8p50GWlQ4g&s=19). The rollout will begin with Beta users and later extend to users of the Ultra tier. No technical specifics on the test suite, model parameters, or competitive baselines are disclosed in these links.** Comments highlight uncertainty about beta access logistics and critique the $250/month price point, with some users contrasting Google's transparency (due to third-party evaluation) favorably against OpenAI's approach.
    - A reference is made to the IMO benchmark results: the 'advanced version of Gemini' (referred by its 2.5 Pro Deep Think codename 'wolfstride') officially achieved a Gold Medal standard at the International Mathematical Olympiad, with details provided in the linked DeepMind blog. This achievement highlights the progress in mathematical reasoning and performance for Gemini models.
    - Discussion notes that the model which achieved the IMO result may be the same or closely related to what is being offered to 'Ultra' users, suggesting high availability and accessibility of advanced model capabilities in Google's consumer offering. There's debate over the value proposition given the subscription price point, but it's recognized as a significant step in AI mathematical reasoning.
    - Technical transparency is highlighted, with one commenter noting that Google used a third party for scoring and validation for the IMO benchmark, implying a higher level of integrity compared to OpenAI's internal-only reporting. This suggests differing standards in benchmark reporting and repeatability among leading AI labs.
- [**Just a few years ago, people thought we were 22 years away from AI winning an IMO Gold Medal**](https://i.redd.it/psyszinh47ef1.png) ([Score: 226, Comments: 23](https://www.reddit.com/r/OpenAI/comments/1m5ecb4/just_a_few_years_ago_people_thought_we_were_22/)): **The image presents Metaculus prediction data tracking how the expected date for AI to win an International Math Olympiad (IMO) Gold Medal has sharply moved forward: in July 2021, the median prediction was 2043, but by July 2024, it shifted to 2026. This acceleration reflects the rapid progress of AI in mathematical problem-solving, in part driven by recent achievements such as Google's model reaching IMO silver and almost gold in 2024, with some claims of near-gold achievement by an LLM (large language model) alone.** Commenters debate the realism of the new 2-year horizon, some calling it too optimistic given the IMO silver performance, while others emphasize how current advancements are surprising due to being accomplished by LLMs without external tools. Additional discussion highlights the unpredictability of AI progress and the difficulty of predicting such breakthroughs, tying it to broader debates around AI forecasting and existential risk.
    - Technical discussion centered on the achievement of large language models, specifically Google’s system in 2024, which reportedly scored silver and was one point off gold at the IMO. The commentator finds the notion of super-fast progress compelling, noting that such performance, especially if achieved by models with only language capabilities and without auxiliary tools, would be far beyond what many thought possible. The difficulty of IMO problems (well beyond even top university math students) is emphasized, highlighting the significance of this advancement if confirmed.
    - Several comments raise skepticism about the comparability of these AI achievements with real IMO conditions. As highlighted by references to Terence Tao and others, current claims often hinge on post hoc grading of AI-generated proofs, not on adherence to competition constraints (e.g., time limits, proctoring). This suggests that while headline results are impressive, they may not yet reflect human-competitive conditions; AI-generated proofs might be valid, but not under the same resource constraints as human contestants.
    - There is a technical forecasting theme regarding the unpredictability of AI milestones, such as gaining unauthorized access to external systems or surpassing mathematical benchmarks. Historical underestimation (“22 years away from gold, achieved in less than half that”) illustrates the accelerating, compounding nature of AI research. Linked forecast platforms like Metaculus track these predictions, which some argue are inherently unreliable given the rapid rate of progress and the singularity-style unpredictability of outcomes.
- [**This is what a mature company looks like**](https://i.redd.it/6w8jfs81k9ef1.jpeg) ([Score: 186, Comments: 9](https://www.reddit.com/r/Bard/comments/1m5pfl3/this_is_what_a_mature_company_looks_like/)): **The image displays a social media post where a company, likely in the AI or ML sector, outlines their decision to postpone their announcement of results until after official verification and certification by the International Mathematical Olympiad (IMO) Board. Their AI model achieved a gold-level grading, which they chose not to disclose prematurely, reflecting a focus on rigor and transparency. This is highlighted as an example of maturity in AI company operations by respecting third-party evaluation processes.** Commentary highlights the professionalism of the company's approach, noting its contrast with more aggressive or premature self-promotion tactics by others in the field. Some tongue-in-cheek references are made to the understated assertion of superiority.
    - A commenter emphasizes that Demis Hassabis tends to avoid overhyping DeepMind's products and operates more as a research scientist than as a typical business executive, highlighting his ability to communicate technical ideas clearly to the general public.
- [**It's still pretty cool, but the details matter**](https://i.redd.it/sx36yimpm9ef1.png) ([Score: 292, Comments: 103](https://www.reddit.com/r/singularity/comments/1m5pud0/its_still_pretty_cool_but_the_details_matter/)): **The image is a meme contrasting DeepMind's claim that its Gemini model with 'Deep Think' performed well at the International Mathematical Olympiad (IMO)—solving 5 out of 6 problems—with the information that the model had access to previous solutions and hints during evaluation. This highlights how benchmarking methodology and data leakage or training set contamination can impact the perceived significance of model achievements; if models see solutions or hints, it calls into question the claimed leap in mathematical reasoning.** Top comments argue that access to previous solutions is analogous to human practice before tests, with most suggesting this does not diminish the accomplishment. There is no deep technical dissent, but a mild debate on fairness and relevance of comparing AI to human learning processes.
    - Several commenters point out that both humans and AI models use prior math olympiad problems for training and preparation, noting the parallel in methodology. One technical nuance raised is that while it's normal for both groups to prep using previous examples, AI models may not be as data-efficient as humans, potentially requiring significantly more exposure to training data to achieve comparable proficiency. This highlights a key technical metric—*data efficiency*—in comparing model learning versus human learning.
- [**OpenAI researcher Noam Brown clears it up**](https://i.redd.it/sd5j73jt73ef1.jpeg) ([Score: 507, Comments: 112](https://www.reddit.com/r/singularity/comments/1m4yx9h/openai_researcher_noam_brown_clears_it_up/)): **The image documents a Twitter exchange where OpenAI's Noam Brown clarifies that their announcement regarding GPT-4o (or another OpenAI model) solving IMO problems was timed to be after the official IMO closing ceremony, per coordination with IMO organizers, countering claims of pre-empting or overshadowing student achievements. Brown emphasizes respect for participants and underscores that OpenAI did not coordinate just with individuals but with the organizing committee. This is relevant as it addresses concerns about propriety and process when publicizing AI benchmarks against human competitions, maintaining scientific and community relations integrity.** A notable debate arises regarding the grading methodology: one commenter argues that OpenAI's claims of 'earning gold' are misleading because they did not adhere to the official IMO rubric, calling into question the validity of the comparison between AI and human contestants.
    - A key technical criticism is that the model's results were evaluated *outside* the official IMO rubric; therefore, claims that it "earned gold" are considered potentially misleading because the assessment criteria do not scale directly to the actual competition standards. This introduces questions about the validity and rigor of the claimed achievement, as highlighted by the concern that 'they graded their work separately without the IMO rubric.'
- [**A take from Terrance Tao about the International Maths Olympiad and OpenAI**](https://www.reddit.com/gallery/1m4zwvt) ([Score: 338, Comments: 73](https://www.reddit.com/r/singularity/comments/1m4zwvt/a_take_from_terrance_tao_about_the_international/)): **Terence Tao highlights how reported AI performance on competitions like the IMO is highly contingent on the testing protocol—factors like compute time, problem reformatting, tool access, collaborative attempts, and selective reporting can inflate capabilities, making cross-model or human versus AI comparisons fundamentally unreliable without strict, standardized methodology. He analogizes this to giving human Olympiad contestants varying levels of assistance, which would radically change their results but would not reflect core capability. This critique is especially pertinent given recent results from labs like Google (AlphaProof, which used days per problem and Lean formalization), xAI, and MathArena (which ran 32 trials and reported best-case results), showing how differing methodologies invalidate direct benchmarking.** Several commenters clarify that Tao's comments target methodological inconsistencies across *multiple* AI lab results (not just OpenAI), citing specific Google and MathArena practices as examples. Additional discussion notes that OpenAI's recent model reportedly did not use tools or external internet access (see [Boris Power's clarification](https://x.com/BorisMPower/status/1946859525270859955)), addressing some fairness concerns Tao raised.
    - Discussion highlights that Terence Tao's caution about AI benchmark comparability is technically rooted in differing methodologies: some labs (like Google with AlphaProof) gave models 3 days per IMO problem and pre-converted them to formal languages (Lean), whereas others employed multi-attempt selection (MathArena's 32-best-out-of-32 bracket approach). This renders cross-lab leaderboard scores (Google, OpenAI, xAI, MathArena, etc.) non-comparable due to drastic variation in constraints and evaluations.
    - One commenter points out a misconception regarding OpenAI's approach: specifically, OpenAI's involvement in the IMO challenge did *not* include tool-use or internet access during evaluation—contrasting with some other labs' setups. This affects both the fairness and comparability of results.
    - A referenced response from OpenAI's head of applied research addresses public concerns over experimental design, underscoring internal awareness of these comparability and fairness issues and implicitly acknowledging the opacity in directly aligning different labs' results.

### 2. AI Industry Talent Wars and Big AI Hiring Moves

- [**Mark Zucker asked Mark Chen if he would consider joining Meta, reportedly offering up to $1 billion dollars**](https://i.redd.it/iq1wtxlnf6ef1.jpeg) ([Score: 721, Comments: 235](https://www.reddit.com/r/singularity/comments/1m5c4mj/mark_zucker_asked_mark_chen_if_he_would_consider/)): **The image summarizes a high-stakes recruitment effort, where Mark Zuckerberg reportedly offered up to $1 billion to Mark Chen (OpenAI’s chief research officer) in an attempt to bolster Meta’s generative-AI team. Chen’s feedback highlighted that Meta's issues aren't just about compute/hardware, but a perceived lack of top-tier AI talent, leading Zuckerberg to pursue direct talent acquisition with a massive compensation package, possibly in the form of RSUs or performance-based incentives.** Technical discussion in the comments centers on the logic of offering hundreds of millions in equity for top AI talent in a trillion-dollar industry, with some arguing such moves are justified given the stakes, while others view the effort as potentially desperate or a sign of imbalance between compute and talent at Meta.
    - A commenter points out that Meta's strategy of offering hundreds of millions in RSUs tied to performance or vesting for key AI talent is a calculated move considering the company's $1.8 trillion market cap and Mark Zuckerberg's $243 billion personal net worth. They argue that allocating substantial equity to influential researchers could plausibly swing the company's position in the competitive AI landscape, especially as the industry is projected to reach trillion-dollar scale.
    - There are questions about the current progress and state of projects like "Llama 4 Behemoth" at Meta, referencing rumors or speculation that the former AI research team may have underperformed or not met internal expectations, possibly motivating these aggressive recruitment attempts.
    - The discussion highlights a shift in acquisition tactics towards talent-focused 'acqui-hiring,' as Meta allegedly seeks to "buy the people working at OpenAI" rather than the company itself. This illustrates a broader trend in tech where securing high-impact researchers is prioritized, especially when direct corporate acquisition is not feasible, in order to accelerate internal foundational model development.
- [**Zuckerberg wanted to buy Ilya’s SSI, but Ilya turned it down. CEO Daniel Gross disagreed with Ilya and wanted him to sell the company. Ilya was ‘blindsided’ by his decision upon learning**](https://i.redd.it/x4ogmt3x27ef1.png) ([Score: 282, Comments: 86](https://www.reddit.com/r/singularity/comments/1m5e757/zuckerberg_wanted_to_buy_ilyas_ssi_but_ilya/)): **The image visually summarizes a situation where Mark Zuckerberg (Meta) wanted to acquire Ilya Sutskever’s startup, SSI, but Ilya rejected the offer to preserve independence. CEO Daniel Gross disagreed and contemplated joining Meta instead. The WSJ article and post context indicate these events highlight divergent motivations: Ilya focused on advancing AI independently, while Gross prioritized financial gain. This tension reflects current industry dynamics, where compute and talent density outweigh individual genius as an AI company’s moat, especially for startups competing with tech giants.** Top comments debate independence versus acquisition: some side with Ilya’s passion for independent AI progress over financial incentives, while others note the limited strategic moat for SSI due to lack of compute and scale, suggesting Meta’s resources could be more advantageous.
    - Several users discuss the technical moat of startups like SSI in today's AI landscape, noting that competitive advantage is increasingly about compute capacity and aggregate talent density, rather than individual genius (even one of Ilya Sutskever's caliber). With limited headcount and substantially less compute than major players like Meta or OpenAI, SSI faces fundamental scaling and competitiveness challenges.
    - Skepticism is raised about SSI's trajectory, referencing the CEO's lost confidence and recent departure as indicators that the company may have either hit technical roadblocks or lacks a clear path forward. The fact that leadership advocated for a sale suggests underlying doubts about the company's breakthroughs or ability to outcompete larger firms in the AI race.
    - Discussions question the viability of maintaining independence in the AI sector given the vast resources required (notably, billions spent on GPUs for compute). Some suggest that even philanthropic or visionary leaders can't indefinitely sustain R&D at the scale needed to remain competitive, unless they partner with or are acquired by tech giants with deep pockets.
- [**OpenAI has thousands of employees and is hiring thousands more…why?**](https://i.redd.it/obpyqoslh4ef1.jpeg) ([Score: 342, Comments: 107](https://www.reddit.com/r/OpenAI/comments/1m54tdx/openai_has_thousands_of_employees_and_is_hiring/)): **The image presents detailed statistics on OpenAI's workforce as of mid-2024: 6,413 employees, with rapid growth rates (**`62%` **in 6 months,** `112%` **in a year, and** `318%` **in two years). Only** `32%` **are in Engineering; the majority are in non-engineering functions such as Operations, Education, and Business Development. The median tenure is notably short (**`0.8 years`**), indicating aggressive recent hiring. The selftext questions why OpenAI, an AI leader, still relies on so many human employees and whether this signals limits to current AI's ability to replace knowledge work now.** Technical commenters note that, relative to OpenAI's global impact, 6,000 staff is small. There's ongoing debate about whether AI can fully automate roles like SWE, and if rapid productivity gains might actually increase headcount in certain roles due to business expansion (referencing concepts like Baumol's Cost Disease).
    - A technical discussion highlights that despite OpenAI's use of advanced tools, a headcount of `~6,000 employees` is relatively small for a company with its global user impact. This is viewed as evidence that many roles, even technical ones like Software Engineering (SWE), are not yet fully automatable; monitoring hiring at firms like OpenAI, Google, and Anthropic serves as an indicator of AI-driven automation limits in technical roles.
    - Another comment presents an automation-induced scaling model: even if OpenAI automates and eliminates `50%` of roles, but product reach quadruples, a doubling of headcount is still required. This feedback loop illustrates how AI can cause workforce expansion in hard-to-automate service roles, referencing *Baumol's Cost Disease* as an economic framework to understand persistent demand and rising wages in sectors resistant to automation.

### 3. Large-Scale Diffusion Model Training and Finetuning Experiments

- [**The Gory Details of Finetuning SDXL and Wasting $16k**](https://www.reddit.com/r/StableDiffusion/comments/1m5rn8h/the_gory_details_of_finetuning_sdxl_and_wasting/) ([Score: 137, Comments: 20](https://www.reddit.com/r/StableDiffusion/comments/1m5rn8h/the_gory_details_of_finetuning_sdxl_and_wasting/)): **The post gives an exhaustive technical writeup on the training of "bigASP v2.5", a large-scale finetuning of Stable Diffusion XL (SDXL) with the Flow Matching objective (from models like Flux), expanding the dataset to ~13M images (with added anime data), freezing text encoders, and increasing training to 150M samples on a multinode (32x H100 SXM5 GPUs) cluster. Batch size was increased to 4096, learning rate set at 1e-4 with AdamW optimizer, float32 params and bf16 AMP, training at 300 samples/sec using FSDP1 (shard_grad_op) with a Shifted Logit Normal noise schedule (shift=3 for training, shift=6 for inference). Extensive issues with streaming data for distributed training (leading to a Rust-based streaming dataset implementation), multi-node communication, and debugging cost overruns (> $16k) are detailed. The model achieves improved dynamic range and reliability thanks to Flow Matching, but only appears to work well with Euler sampling in ComfyUI, and suffers prompt confusion with frozen text-encoders. Exact configs/code: [Github](https://github.com/fpgaminer/bigasp-training/tree/main/v2_5). Model weights: [HuggingFace](https://huggingface.co/fancyfeast/bigaspv2-5).** Commenters highlight the practical value of the detailed post and confirm the model yields high-quality outputs; one notes positive experience with JoyCaption for captioning, suggesting broad community adoption and utility. A key technical thread is the trade-off between robust generalization (from frozen text encoders) and prompt adherence, with early sacrifice of anime-style fidelity despite increased data diversity.
    - A user inquires whether there are direct visual or performance comparisons between bigASP and SDXL using identical prompts and settings, highlighting interest in concrete benchmarking and differential performance evaluation between these models. Such comparisons would inform on strengths or weaknesses in image output quality or stylistic variance.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1: AI Agents Storm the Scene with Multimodal Might**

- **OpenAI Unleashes ChatGPT Agent for Computer Domination**: OpenAI rolled out **ChatGPT Agent** to Pro, Plus, and Teams users, enabling it to control computers, browse, code, edit spreadsheets, and generate images or slides, as detailed in [the ChatGPT Agent announcement](https://openai.com/index/introducing-chatgpt-agent/). Reactions included EU availability concerns and fears it cannibalizes **Operator** and **Deep Research**, with the Operator site set to sunset in weeks.
- **Mistral's Le Chat Levels Up with Voice and Reasoning**: Mistral upgraded Le Chat with **Deep Research reports**, **Voxtral voice model**, **Magistral multilingual reasoning**, and in-chat image editing, praised for its *European vibe* in [the Mistral AI update tweet](https://x.com/MistralAI/status/1945858558836216026). Users compared it favorably to Claude, sparking jokes about *Le Waifu* potential.
- **Kimi K2 Codes Physics Sandbox Like a Boss**: **Kimi K2** generated a full physics sandbox code after a prompt in its [chat interface](https://www.kimi.com/chat/), with the output available at [the plasma_sfml.cpp code](https://cdn.discordapp.com/attachments/1340554757827461211/1395196566389784606/plasma_sfml.cpp?ex=687ae30e&is=6879918e&hm=9e5b88351c6d243c6e165f444254a6eeb786c3d64fd8bdf958394026aa0ce5cb&). Community lauded its coding prowess, highlighting AI's leap in precise code creation tasks.

**Theme 2: Quantization Tricks Squeeze Models into Tiny Bits**

- **Alibaba's ERNIE 4.5 Fumbles 2-Bit Compression**: Alibaba claimed lossless **2-bit compression** for **ERNIE 4.5**, but analysis in [the turboderp ERNIE-4.5 exl3 repo](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) revealed it's actually **2.5 bits** on average due to higher-precision layers, performing worse than true **exl3 2-bit** versions. Critics mocked the hype, noting it degraded output quality without real gains.
- **Speculative Decoding Accelerates Models by 28%**: Users reported a **28% speed boost** on tested models via **Speculative Decoding**, recommending **Qwen3** with **1.7b Q8** or **bf16** drafts for optimal gains. The trick shines on smaller draft models, pushing inference speeds without sacrificing accuracy.
- **GitChameleon Exposes LLMs' Code Versioning Flaws**: The **GitChameleon benchmark** showed LLMs fail simple ID-based version-conditioned code generation, as detailed in [the GitChameleon paper](https://arxiv.org/abs/2507.12367). It underscores weaknesses in precise code manipulation, urging better training for versioning tasks.

**Theme 3: Sky-High Valuations Fuel AI Bubble Fears**

- **Perplexity Rockets to $18B Valuation Amid Skepticism**: Perplexity eyes an **$18B** valuation for its next funding round despite **$50M revenue**, sparking bubble concerns in [the Perplexity valuation tweet](https://x.com/arfurrock/status/1945933446376755459?s=46&t=b7l37rB6wtbyAh6ah1NpZQ). Critics questioned the justification, with some labeling it overinflated hype.
- **FAL Hits $1.5B Valuation After $125M Series C**: FAL secured a **$125M** Series C led by Meritech Capital, boosting its valuation to **$1.5B** post-money with **$55M ARR** and **25x YoY growth**, per [the FAL funding tweet](https://x.com/arfurrock/status/1945553966495912051?s=46). The diffusion model inference firm touted **10% EBITDA** and **400% M12 net-dollar-retention** as proof of traction.
- **DeepSeek Boasts 545% Profit Margins in Wild Claim**: DeepSeek claimed theoretical **545% profit margins** if V3 matched R1 pricing, stirring pricing debates in [the DeepSeek TechCrunch article](https://techcrunch.com/2025/03/01/deepseek-claims-theoretical-profit-margins-of-545). Community mocked the assertion as marketing fluff amid AI market volatility.

**Theme 4: Hardware Hurdles Haunt GPU Warriors**

- **Blackwell RTX 50 Series Demands xformers Rebuild**: Users fixed **Blackwell RTX 50** support by building **xformers** from source, with latest **vLLM** adding compatibility after pip upgrades like `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth-zoo unsloth`. It resolved **H200** OOM issues during **Qwen3-8B LoRA** training with **GRPO**.
- **CUDA Fuses Kernels in Python for Speed Demons**: NVIDIA enabled **CUDA kernel fusion** directly in Python, optimizing computations as outlined in [the NVIDIA CUDA kernel fusion blog](https://developer.nvidia.com/blog/delivering-the-missing-building-blocks-for-nvidia-cuda-kernel-fusion-in-python/?trk=feed_main-feed-card_feed-article-content). It streamlines workflows, bypassing manual optimizations for faster AI tasks.
- **3090 Upgrade Crushes LLM Tasks on Budget**: A user swapped a **3080 Ti** ($600 sale) for a **3090 FTW3 Ultra** ($800 buy), boosting **LLM** performance without breaking the bank. The move highlighted affordable hardware tweaks for better inference speeds.

**Theme 5: Tools and APIs Tackle Tricky Tasks**

- **OpenAI's Image Editor API Zaps Selective Edits**: OpenAI updated its image editor API to edit only selected parts instead of regenerating entire images, improving efficiency as announced in [the OpenAI image editor tweet](https://x.com/OpenAIDevs/status/1945538534884135132). Developers hailed the precision boost for targeted modifications.
- **LunarisCodex Toolkit Trains LLMs from Scratch**: A 17-year-old developer released **LunarisCodex**, an open-source toolkit for pre-training LLMs with features like **RoPE**, **GQA**, and **KV Caching**, available at [the LunarisCodex GitHub](https://github.com/MeryylleA/lunariscodex). Inspired by **LLaMA** and **Mistral**, it targets educational use for custom model building.
- **Triton Autodiff Differentiates Kernels Automatically**: The [IaroslavElistratov/triton-autodiff repo](https://github.com/IaroslavElistratov/triton-autodiff) implemented **automatic differentiation** for **Triton**, enabling gradient computations in custom kernels. Users buzzed about its potential to simplify optimization in GPU programming.



---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Airtel Gives Away Free Perplexity Pro**: Indian network provider **Airtel** now offers a **1-year free Perplexity Pro** subscription to its customers through the **Airtel Thanks app** as a reward.
   - Members are reporting that **Perplexity search** and **research functions** are hitting new rate limits despite being a Pro subscriber, with one user experiencing issues activating their Pro subscription.
- **Comet Browser Still Elusive**: Members are still waiting for their **Comet browser invite**, with some reporting they have been waiting months for approval.
   - One member described it as *just a browser but + the assistant sidebar that see's your current live site and can reference from*.
- **Perplexity Pages iOS Only**: Members are excited about the new **Pages feature** that generates a page for a query, but it is **only available on iOS** and has a **limit of 100 pages**, stored in [perplexity.ai/discover](https://www.perplexity.ai/discover).
   - Members think it's a way to do Deep Research.
- **Sonar APIs Need Better Prompting**: A team member stated there has been an increase in issues due to how users are prompting their **Sonar models** and linked to the [prompt guide](https://docs.perplexity.ai/guides/prompt-guide).
   - Members also discussed getting more consistent responses and valid **JSON** output when using a high search context, as well as a desire to view a history of **API calls** in their account dashboard.
- **Pro Users Now Get API Access**: With **Perplexity Pro** you get **$5 monthly** to use on **Sonar** models, allowing you to embed their **AI-powered search** into your own projects while having the ability to obtain citations as described in the [Perplexity Pro Help Center](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro).
   - Remember that these are search models, and should be prompted differently to traditional LLMs.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Agent Livestream Announced**: OpenAI is hosting a livestream about **ChatGPT Agent**, **Deep Research**, and **Operator**; details can be found on the [OpenAI blog](https://openai.com/index/introducing-chatgpt-agent/) and the [livestream invite](https://discord.gg/DqBbV7ya?event=1395405196619939943).
   - The livestream will cover updates on **Deep Research** and **Operator**, potentially including new features or use cases.
- **Grok App Strands iPhone X Users**: The **Grok app** requires **iOS 17**, rendering it unusable on older devices such as the **iPhone X**.
   - Users discussed needing a secondary iPhone specifically for the **Grok app**, with some cautioning against buying a new iPhone solely for this purpose.
- **Agent mode doesn't fly on 3o Model**: Users report that **GPT agents** can only be switched when using **models 4 or 4.1**, and the agent switching function does not appear on other **LLM models**.
   - One user indicated the **Agent function** might simply not be available in **3o**, and suggested filing a bug report, with another user suggesting that **Agent** is a model in its own right ([OpenAI help files](https://help.openai.com/en/articles/11794342-chatgpt-agent)).
- **Reproducibility Riddled with Failures**: A member posted a [chatgpt.com link](https://chatgpt.com/share/68791cda-f338-8000-81f8-b7615d3f5a9c) that was called out for reading like a design proposal, and missing key **Reproducibility Elements** such as prompt templates, model interfaces, and clearly defined evaluation metrics.
   - The conversation highlighted the absence of fully instantiated examples of **Declarative Prompts**, clear versioning of prompt variants used across tests, and concrete experimental details.
- **ChatGPT for Desktop Explored**: Users are investigating using **Chat GPT** on desktop for local file management, akin to **Claude Harmony**.
   - One suggestion involves using the **OpenAI API** (*paid*) with a local script to interface with the file system, essentially creating a custom "Harmony"-like interface.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Family Matters: Model Performance Variance**: Models within the same family show very similar performance, so going below **3 bits** for a larger model isn't recommended, whereas models from different families vary depending on the vertical.
   - Some exceptions are made if one model is at **7B** and the other is **70B**, where 1.8 bits could still be usable for some tasks as it's a big model.
- **Transplant Trauma: Vocab Swapping Woes**: Swapping model architectures like making **LLaMA 1B -> Gemma 1B** without continued pretraining leads to horrible results due to transplanting the vocabulary.
   - It was noted that the **Qwen 1** architecture is almost completely the same as **Llama 1/2**, so you can make some minor changes, jam the **Qwen** weights in, train for 1.3 billion tokens, and get a worse model than you put in.
- **Prompting Prevails: Fine-Tuning Fades for Functionality**: For educational LLMs, it's advised to start with good prompting before jumping into fine-tuning, as instruction following is currently very efficient.
   - Members also suggested tools like [synthetic-dataset-kit](https://github.com/facebookresearch/SyntheticDataToolkit) to generate instructional conversations.
- **AliBaba Botches Bit Budget**: AliBaba mumbled about some lossless **2bit compression** trick in their release of **ERNIE 4.5**, but [turboderp looked into it](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) and its just worse than exl3 because they left a bunch of layers in higher precision.
   - It's not a true **2-bit** on average (more like **2.5 bit**), and the true exl3 **2 bit** performs better than the ~2.5 bit they showed.
- **Blackwell Build Blues Blocking Bootstrapping**: Users discussed that building **xformers** from source is the only thing needed for **Blackwell RTX 50** series support, and the latest **vLLM** should be built with **Blackwell** support.
   - Members suggested upgrading Unsloth using `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth-zoo unsloth` to solve **H200** issues.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's New Pricing Draws Ire**: Users are expressing confusion and frustration as [Cursor shifts from a fixed request model to one based on model costs](https://tenor.com/view/kevin-spongebob-angry-will-you-cut-that-out-gif-12675670360524347134), claiming *bait and switch*.
   - Some users are reporting disappearing messages and concerns about the legality of changing the contract.
- **Claude Integration via MCP Lightens Load**: Integrating **Claude** via MCP (Multi-Client Protocol) within Cursor helps manage costs associated with **Sonnet** and **Opus**.
   - Members [acknowledged that](https://www.youtube.com/watch?v=D0iXkmyWcPM) this is only possible through an external tool.
- **Agents get stuck in the weeds**: Users report Cursor agents getting stuck during tasks, a [known issue](https://cdn.discordapp.com/attachments/1395153343122505728/1395154214157816029/image.png?ex=687abb9d&is=68796a1d&hm=fc84cafbaab772eef094cc5199a623d5216c528ddde98227e6092221778c4fcc&) that the team is addressing.
   - Manually stopping the prompt may prevent billing due to a **180-second timeout** that auto-cancels stuck requests.
- **KIRO Courts Competition With Cursor**: Members are comparing Cursor with **KIRO**, a new IDE focused on specification-based coding and hooks, noting that [KIRO is in a waitlist phase](https://kiro.dev/) due to high demand.
   - One discussion point raises concerns that **KIRO** might be using user data to train its models, despite settings to disable this.
- **Users Question Model 'Auto' Uses**: Users are curious about which model "Auto" uses in Cursor, speculating that it might be **GPT 4.1**.
   - No evidence has been shown either way to confirm or deny.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **DeepSeek Declares Disproportionate Dividends**: DeepSeek projects theoretical profit margins of 545% if V3 were priced like R1, as detailed in [this TechCrunch article](https://techcrunch.com/2025/03/01/deepseek-claims-theoretical-profit-margins-of-545).
   - The assertion stirred debate around the pricing strategies and technological advancements within the AI model market.
- **OpenAI Oracles Online Opportunity**: Speculation is rampant about an imminent OpenAI browser launch, possibly GPT-5 or a GPT-4 iteration enhanced with browsing capabilities, spurred by [this tweet](https://x.com/testingcatalog/status/1945639961790685404?s=46).
   - The potential release has the community guessing about its features and impact on AI applications.
- **Kimi K2 conjures code creations**: Kimi K2 showcased its coding prowess by generating a physics sandbox, with the code available [here](https://cdn.discordapp.com/attachments/1340554757827461211/1395196566389784606/plasma_sfml.cpp?ex=687ae30e&is=6879918e&hm=9e5b88351c6d243c6e165f444254a6eeb786c3d64fd8bdf958394026aa0ce5cb&) after prompting it in its [chat interface](https://www.kimi.com/chat/).
   - The demonstration has been lauded, highlighting the evolving capabilities of AI in code generation.
- **OpenAI Overhauls object operation optimization**: OpenAI's image editor API update now isolates edits to selected parts, improving efficiency over redoing entire images, as announced in [this tweet](https://x.com/OpenAIDevs/status/1945538534884135132).
   - This refinement promises enhanced control and precision for developers utilizing the API.
- **GPT-5 Gossips Gather Geometrically**: Anticipation for GPT-5's unveiling is fueled by hints such as a [pentagon reference](https://x.com/sama/status/1945900345378697650) that aligns with the number 5.
   - Speculation varies from a late summer launch to expectations of an agent-based system with advanced research functionalities.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **FAL Ascends to $1.5B Valuation**: FAL, an AI-driven inference infrastructure for diffusion models, closed a **$125M** Series C round led by Meritech Capital, achieving a **$1.5B** valuation post-money according to [this tweet](https://x.com/arfurrock/status/1945553966495912051?s=46).
   - This follows their previous announcement of **$55M ARR**, **25x YoY growth**, **10% EBITDA**, and **400% M12 net-dollar-retention** demonstrating strong market traction.
- **Le Chat Gets Multilingual Reasoning Upgrade**: Mistral launched a major update to Le Chat adding features like Deep Research reports, a **Voxtral voice model**, **Magistral multilingual reasoning**, chat organization with Projects, and in-chat image editing, as described in [this tweet](https://x.com/MistralAI/status/1945858558836216026).
   - The release was commended for its UI and *European vibe*, drawing comparisons to Claude and sparking humorous comments about *Le Waifu*.
- **Perplexity's Lofty $18B Valuation Questioned**: Perplexity is reportedly raising funds at an **$18B** valuation, inciting reactions from amazement to concerns about a potential bubble, as seen in [this tweet](https://x.com/arfurrock/status/1945933446376755459?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
   - Critics questioned the justification of this valuation, highlighting the discrepancy between the **$50M revenue** figure and the high price tag.
- **OpenAI Launches ChatGPT Agent**: OpenAI's new **ChatGPT Agent**, a multimodal agent with capabilities to control a computer, browse, code, write reports, edit spreadsheets, and create images/slides, is rolling out to Pro, Plus, and Teams users, announced via [this tweet](https://x.com/kevinweil/status/1945896640780390631).
   - Reactions included excitement, inquiries about EU availability, and worries about personalization conflicts, as well as cannibalization of Operator and Deep Research.
- **Operator and Deep Research Facing Sunset**: With the launch of **ChatGPT Agents**, it was noted that **ChatGPT Agents** might cannibalize **Operator** and **Deep Research**, with confirmation that *the Operator research preview site will remain functional for a few more weeks, after which it will be sunset.*
   - Users can still access it by selecting **Deep Research** from the dropdown in the message composer.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Opus Users Oppose Outrageous Overages**: Users debate **Claude 4 Opus** pricing, noting one spent **$10 in 15 minutes**, while others suggest Anthropic's **€90/month plan** for *unlimited use*.
   - A user on the **$20 plan** claims they *barely ever hit their limit* because they don't use AI tools in their IDE, suggesting usage varies greatly.
- **GPT Agents grapple Groundhog Day**: A user raised concerns that **GPTs agents** aren't learning beyond initial training, even after uploading files, and files are just saved as **knowledge files**.
   - Agents can reference new information, but don't inherently learn from it in the same way as during pre-training, which requires more.
- **Free Models Face Frustrating Fails**: Users report issues with the **free model v3-0324**, questioning why they were switched to non-free version despite using the free tier.
   - Reports indicate hitting credit limits or receiving errors even when using free models, with one user stating their AI hasn't been used since June.
- **Cursor Code Crashing Creates Chaos**: **OpenRouter models** integrated with **Cursor**, highlighting **Moonshot AI's Kimi K2**, but users reported issues getting it to work, especially outside of **GPT-4o** and **Grok4**.
   - According to [a tweet](https://x.com/msfeldstein/status/1945533992222298167?s=46&t=fN048T2GS3C_B_wKldWGFw), *it worked when we wrote it and then cursor broke stuff*.
- **Inference Implementations Incurring Insolvency**: **Kluster.ai** is shutting down its inference service, described as a *very cheap and good service*, following **CentML's** closure.
   - Members are speculating about an **AI bust** or hardware acquisitions, raising concerns about the sustainability of AI inference services.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Eleuther Bridges Research Resource Gap**: **Eleuther AI** aims to bridge the research management gap for independent researchers lacking academic or industry resources, facilitating access to research opportunities.
   - The initiative seeks to support researchers outside traditional systems by offering guidance, handling bureaucratic tasks, and providing a broader perspective, as many are locked out of paths like the **NeurIPS high school track**.
- **Resources Shared for ML Paper Writing**: Members shared resources for writing machine learning papers, including [Sasha Rush's video](https://www.google.com/search?q=Sasha+Rush+how+to+write+a+great+ml_paper) and [Jakob Foerster's guide](https://www.jakobfoerster.com/how-to-ml_paper), alongside advice from the [Alignment Forum](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml_papers).
   - Additional resources included posts on [perceiving-systems.blog](https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper), [Jason Eisner's advice](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html), and a [guide from Aalto University](https://users.aalto.fi/~jsaramak/HowToWriteCommsCoffee.pdf).
- **Mentors Prevent Unrealistic Research**: Participants emphasized the importance of mentorship in research, noting mentors help to figure out *what is possible and what is unrealistic* so that one can narrow things down.
   - A mentor's guidance helps researchers navigate challenges and avoid wasting time on unproductive avenues, as guides only offer basic knowledge.
- **ETHOS Model Gets Streamlined, Updated on GitHub**: A member shared a [simplified pytorch code version](https://github.com/wrmedford/ETHOS/blob/main/model.py#L267-L337) of their model and noted that they had to use a slightly different version where **all heads are batched** because of how eager execution mode uses up a ton more memory if they looped over all heads.
   - They also stated the expert network isn't vestigial, and linked [the specific lines of code](https://github.com/wrmedford/ETHOS/blob/main/kernels.py#L156-L158) where they generate **W1** and **W2** in the kernel.
- **nnterp** Unifies Transformer Model Interfaces**: A member released the beta 1.0 version of their mech interp package, **nnterp**, available via `pip install "nnterp>0.4.9" --pre` and is a wrapper around [NNsight](https://nnsight.net/).
   - **nnterp** aims to offer a unified interface for all transformer models, bridging the gap between *transformer_lens* and *nnsight*, demoed in [this colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) and [docs](https://butanium.github.io/nnterp/).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Speculative Decoding Gets Models Zoomin'!**: A user reported achieving a roughly **28% speed boost** on models tested with **Speculative Decoding**. They suggested using different **quantizations** of the same model for the draft model, recommending **Qwen3** benefits greatly from using the **1.7b Q8** or even **bf16** as a draft.
   - The user implied that the faster and smaller the draft model is, the better the speed boost becomes.
- **Gemma Model Gets a Little Too Real**: A user recounted a funny situation where a local **Gemma** model threatened to report them. This led to a discussion about the transient nature of *DAN prompts* due to quick patching.
   - A user joked that they will need to install the **NSA's backdoor** to prevent the model from snitching. 
- **LM Studio Awaits HTTPS Credentials**: A user asked how to configure **LM Studio** to accept an **open network server** instead of a generic HTTP server, aiming for **HTTPS** instead of **HTTP**. Another user suggested using a **reverse proxy** as a current workaround.
   - The user expressed wanting to serve the model, but felt unsafe using HTTP.
- **EOS Token Finally Gets Explained**: A user asked about the meaning of **EOS** token, which prompted another user to clarify that **EOS** stands for **End of Sequence Token**, signaling the **LLM** to halt generation.
   - No further context was provided.
- **3090 FTW3 Ultra Gives LLMs A Boost!**: A user upgraded from a **3080 Ti** (sold for $600) to a **3090 FTW3 Ultra** (bought for $800), anticipating improved performance for **LLM** tasks.
   - They secured the **3090** at the original asking price, expecting better performance for their **LLM** endeavors.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SmolVLM2 Blogpost Suspected Scam**: A member suggested that the [SmolVLM2 blog post](https://huggingface.co/blog/smolvlm2) may be a scam.
   - Doubts arose from the lack of information detailing changes between **SmolVLM v1 and v2**.
- **Microsoft's CAD-Editor sparks debate**: Microsoft released the [CAD-Editor model](https://huggingface.co/microsoft/CAD-Editor), enabling interactive editing of **existing CAD models** via natural language.
   - Reactions ranged from concerns about **AI replacing jobs** to arguments that **AI serves as a tool** requiring expertise, similar to calculators not replacing math experts.
- **GPUHammer Aims to Stop Hallucinations**: A new exploit, [GPUHammer](https://gpuhammer.com/), has been launched with the goal of preventing LLMs from hallucinating.
   - The tool's effectiveness and methodology were not deeply discussed, though the claim itself generated interest.
- **Brazilian Teen Premieres LunarisCodex LLM Toolkit**: A 17-year-old developer from Brazil introduced **LunarisCodex**, a fully open-source toolkit for pre-training LLMs from scratch, drawing inspiration from **LLaMA** and **Mistral** architectures, available on [GitHub](https://github.com/MeryylleA/lunariscodex).
   - Designed with education in mind, **LunarisCodex** incorporates modern architecture such as **RoPE**, **GQA**, **SwiGLU**, **RMSNorm**, **KV Caching**, and **Gradient Checkpointing**.
- **GitChameleon Exposes LLM Code Generation Weakness**: The **GitChameleon** eval benchmark reveals that LLMs struggle with simple ID based version conditioned code generation problems, as detailed in [this paper](https://arxiv.org/abs/2507.12367).
   - The benchmark underscores the challenges LLMs face in tasks requiring precise code versioning and manipulation.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Shuffle Sync Sums Discovered**: A user found that `__shfl_down_sync` can sum registers within a warp, combining data between threads, as shown in [this image](https://cdn.discordapp.com/attachments/1189498205101109300/1395481181075542036/reduce_shfl_down-625x275.png).
   - Another member added that modern architectures include specific **reduction intrinsics**, making manual shuffle reductions unnecessary, as documented in [NVIDIA's CUDA documentation on warp reduce functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-functionssupported) (Ampere and above, compute capability >= 8.x).
- **Triton Gets Auto Differentiation**: A user shared a link to [IaroslavElistratov/triton-autodiff](https://github.com/IaroslavElistratov/triton-autodiff), an implementation of **automatic differentiation** for **Triton**.
   - Additionally, a user has been experimenting with the new `tl.constexpr_function` decorator which comes out in **triton 3.4.0**, using `exec` to compile an expression into a `@triton.jit` function.
- **Blackwell GPU's cause Inductor Blues**: A member noted they are facing issues with **Inductor**, which they suspect might be related to using **Blackwell GPUs**.
   - They mentioned needing to use nightly builds or the branch cut 2.8, but aren't entirely sure if **Inductor** is the root cause.
- **CUDA Fuses Kernels in Python!**: NVIDIA is delivering the missing building blocks for [CUDA kernel fusion in Python](https://developer.nvidia.com/blog/delivering-the-missing-building-blocks-for-nvidia-cuda-kernel-fusion-in-python/?trk=feed_main-feed-card_feed-article-content).
   - The enhancement promises to streamline and optimize CUDA-based computations directly within Python environments.
- **Voltage Park Seeks Remote Storage Engineer**: Voltage Park is looking for a **Storage Engineer** to work **remotely**, with more information available at [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f).
   - Voltage Park is looking for a **Storage Engineer** to work **remotely**, with more information available at [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Parameter Functions Decoded**: A member shared [a link to the manual](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure) detailing `@parameter` functions, enabling the capture of variables through **parametric closures**.
   - The documentation elucidates the creation and utilization of these closures, enhancing Mojo's flexibility.
- **Mojo Roadmap gets Unified Closures**: The **Mojo Q3 roadmap** outlines plans for unifying `@parameter` and runtime closures, announced on the [Modular Forum](https://forum.modular.com/t/mojo-q3-roadmap-update/1957).
   - This unification promises to streamline the handling of closures within Mojo, improving developer experience.
- **MAX Graphs now supercharge PyTorch**: The new `@graph_op` decorator allows wrapping an entire **MAX graph** as a custom **PyTorch operator**, with an example in the `modular` repo: [Initial Support for Writing PyTorch Custom Ops in Mojo](https://forum.modular.com/t/initial-support-for-writing-pytorch-custom-ops-in-mojo/1541/2?u=bradlarson).
   - This integration allows engineers to harness the power of MAX graphs within PyTorch workflows.
- **Benchmarking gets OOM'd**: During benchmarking with **Max-24.6** on an **A100-SXM-48GB GPU**, a member ran into `CUDA_ERROR_OUT_OF_MEMORY` errors when using `--batch-size 248` and `--max-length 2048`.
   - Reducing the `--max-cache-batch-size` to **91** also resulted in a **CUDA OOM error**, estimating memory use exceeded available memory (**78812 / 40441 MiB**).
- **Latest MAX the Only One Supported**: The team confirmed the latest stable version is the only supported one, meaning there are no 'LTS' releases.
   - However, using **Max-25.4** with `caching-stragegy paged` worked well, mitigating the issues encountered with **Max-24.6**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Zuck's AI Talent Grab Fuels Belief**: Members discussed **Zuckerberg's** recent aggressive acquisition of AI talent, with one expressing a growing belief in **Meta's** AI initiatives.
   - The comment shows the sentiment that Meta may be positioning itself to become a major player in the AI field.
- **Chicken Tender Prices Spark Existential Dread**: A member expressed dismay at the high price of chicken tenders, questioning *"Why are chicken tenders 5 bucks each now??"
   - This was linked to broader concerns about inflation and market conditions.
- **OpenAI Prefers Comparing to Themselves**: Members noted **OpenAI's** shift towards comparing **ChatGPT Agent** performance only against its previous models, referencing the [ChatGPT Agent announcement](https://openai.com/index/introducing-chatgpt-agent/).
   - The shift in strategy suggests they might not be winning against competitors in certain benchmarks.
- **Grok 4 Aces the HLE Benchmark**: A member pointed out that **Grok 4** achieved a top score of **25.4** on the [HLE benchmark](https://agi.safe.ai/), indicating a significant improvement.
   - This score positions Grok 4 as a leader in the specific capabilities assessed by the HLE benchmark.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Alternative AI Model Outperforms Manus Claims User**: A user claimed to have developed an **AI model** surpassing **Manus** in benchmark performance and offered *unlimited access* to the first 100 beta testers via DMs.
   - The user highlighted the AI's *next-level* capabilities with *zero limits*, hinting at significant improvements over existing solutions.
- **Manus Chat Service Faces Potential Outage**: A user reported a potential issue with the **Manus chat service**, indicating that it might not be functioning correctly.
   - The announcement did not include any information regarding the cause of the issue or potential fixes.
- **Help Needed for Zipping with Manus**: A member requested guidance on how to instruct **Manus** when encountering difficulties in zipping large files.
   - The request did not receive any immediate solutions or suggestions within the available message history.
- **Custom Data Sources Query**: A user inquired about the functionality of **custom data sources** in the paid version of Manus, particularly how to integrate a **CRM**.
   - They also asked about **Model Context Protocol** support, expressing a desire to develop such a feature due to its utility.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic Payment Platform Plunges**: Users report that **Anthropic's payment platform** is reversing charges immediately after they are made, which is preventing the purchase of **API credits**.
   - It is currently unknown if this is a temporary issue or a more persistent problem.
- **MCP Server Sweetens Domain Checks**: An MCP server request for **domain name checking** led to a suggestion of the [whois-mcp](https://github.com/bharathvaj-ganesan/whois-mcp) GitHub repository.
   - The original poster confirmed it was easy to install and thanked the suggesting user.
- **Needle Seeks Connection**: One of the creators of the **Needle MCP server** introduced themself and shared a link to the [Needle MCP server](https://github.com/needle-ai/needle-mcp) GitHub repository.
   - They expressed excitement about joining the server and connecting with fellow MCP enthusiasts.
- **OAuth and API Keys: A Thorny MCP Issue**: A user inquired about the challenges of **auth/oauth** for **MCPs**, sparking a discussion about the trade-offs between **OAuth** and **API keys**.
   - Some users advocated for **OAuth** due to its expiring, dynamically scoped access tokens, while others defended **API keys** for their simplicity, arguing that expiry and scoping can be implemented without OAuth2.
- **Brave's MCP Server Bravely Debuts**: **Brave** launched their official **MCP Server**, announced in [this tweet](https://x.com/Arindam_1729/status/1945958688919114183).
   - One user stated that they haven't tried it because *that tweet didn't include instructions on how to use it*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ShapeTracker Parameter Debated for ASSIGN UOp**: A member proposed adding an optional **ShapeTracker** parameter to **ASSIGN UOp**, potentially using `self.assign(v, res.uop.st)` to use the optional **ShapeTracker** instead of the original tensor's **ShapeTracker** for lowering into the actual assignment code.
   - Concerns were raised about maintaining a minimal set of **UOps**, with an alternative suggestion to pass `res` and extract the **ShapeTracker** internally.
- **Tinygrad Docs Beg for MNIST Code Completion**: A user reported that the **tinygrad documentation** is hard to follow for ML beginners and requested a complete, final code sample for the MNIST tutorial at the end of the page.
   - The user also noted that the **tensor puzzles** aren't working and that it should be stated clearly whether one should learn PyTorch or TensorFlow first.
- **WSL2 Display Driver Provokes Disconnects**: A user encountered a *double free detected in tcache* error after updating their **NVIDIA GPU driver** and sought assistance to make their GPU visible to WSL2 for tinygrad.
   - A member suggested switching to native Ubuntu, stating that *many problems went away* after doing so, including *not being able to load Stable Diffusion weights, due to an obscure limitation on pinned memory in WSL.*
- **Muon Optimizer Moves Meticulously**: A user created a [Muon optimizer](https://github.com/aeryskyB/tiny_learn/blob/master/muon_tiny/muon.py) for tinygrad, finding that it converges faster (~98%) than standard AdamW in the MNIST tutorial.
   - The user is seeking suggestions on how to properly test the Muon optimizer, particularly in the context of contributing a PR to tinygrad.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos v0.3 Lands!**: Nous Research released **Atropos v0.3**, their **RL Environments Framework**, as announced [on X](https://x.com/NousResearch/status/1945932488960008441).
   - Users are encouraged to check out the details of the new version.
- **Teknium Deconstructs Proto-Agentic XML**: A member clarified that *'Proto'* refers to the early form of something, explaining the meaning of *proto-agentic XML tag adherence for proto-reasoning CoTs*.
   - He humorously noted the need for an ELI5-style explanation, stating, *"Yall need an ELI5 with all this tech bro"* and *"Us vibe coders need to eat too"*.
- **Hermes Doc Page In the Works**: A member is developing a [Hermes documentation page](https://link.to.documentation) and a unified Nous Projects documentation page.
   - When asked about the goals for **Hermes 4**, they simply replied, *"Smarter Hermes ofc"*.
- **Kimi K2's Morals Spark Ethical AI Debate**: A member shared an interaction where the **Kimi K2** model refused to provide instructions on how to break into a car, citing legal and ethical concerns.
   - Despite attempts to circumvent the restrictions, **Kimi K2** maintained its stance, leading the member to joke, *"Kimi K2 is a badboy with some morals... Badboy Kimi K2 !!"*
- **Learning ML Bottom-Up?**: A member with a biochemistry background inquired about the best approach to learning **Machine Learning (ML)**, having already made progress in **Python**, math fundamentals (**Calculus**, **Statistics**), and **Introduction to Statistical Learning (ISLR)**.
   - They pondered whether a bottom-up or top-down approach would be more effective for conducting research in **ML** for science.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Browser Extension Wields Ad-Blocking Power**: A member advocated for the **uBlock** browser extension to block ads, suggesting the addition of extra filters for annoyances and social media popups in the extension settings, as illustrated in [this screenshot](https://cdn.discordapp.com/attachments/1124403655819415592/1395131091756650566/image.png?ex=687aa614&is=68795494&hm=478447e95cb45c2b74b11d1e780db4d7c58347a1ae5fca730957e4850c862289).
   - The copied content is then pasted into **Google Docs**.
- **Notepad.exe Tames Ads**: A member proposed copying an article and pasting it into **notepad.exe** to circumvent the inclusion of ads and unwanted content.
   - It was mentioned that this method may not always be reliable and could potentially strip away desired formatting, so caveat emptor.
- **NotebookLM Envisions Folder Integration**: A member suggested that **NotebookLM** could read specific folders/subfolders in a web browser's favorites, treating them as a single source.
   - The current workaround involves *select all and copy/paste* into **Google Docs**.
- **User Faces Service Unavailable Error**: A user reported encountering a *"Service unavailable"* error message when attempting to access a service, accompanied by the message *"You tried to access a service that isn't available for your account".*
   - The user was not given any further guidance or steps on how to troubleshoot.
- **Textbook Data Conquered by NotebookLM**: A user inquired about uploading a textbook as a source to NotebookLM; a member responded that they upload textbooks using **Adobe Scan** to digitize them into PDFs.
   - They then use **NotebookLM** to generate in-depth reviews from the textbooks.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Agentic AI Summit Livestreaming!**: The **Agentic AI Summit** at **UC Berkeley** on **August 2nd** will broadcast via livestream, available at [Agentic AI Summit Livestream](https://lu.ma/agentic-ai-summit-livestream).
   - Speakers include prominent figures such as **Vinod Khosla** (Khosla Ventures), **Bill Dally** (Nvidia), **Ion Stoica** (Databricks and Anyscale), and **Jakub Pachocki** (OpenAI).
- **Fall Semester Status: Unknown!**: A member inquired about a fall semester, but staff confirmed that *nothing has been confirmed yet* and said that important information would be shared on the [Berkeley RDI newsletter](https://rdi.berkeley.edu/signup).
   - They suggested following **Prof Song's social media** ([LinkedIn](https://www.linkedin.com/in/dawn-song-51586033/) or [Twitter/X](https://x.com/dawnsongtweets?lang=en)) for updates.
- **Certificate Declaration Forms: Vanishing Act?**: A member asked to check what they missed submitting, and staff replied they likely did not submit the **certificate declaration form**.
   - They stated that they *never got a certificate declaration form submission* from that user and that a request for a **massive automatic review** was denied.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **DNNs Seek True Time Series Treatment**: A PhD student in dynamical systems theory seeks to integrate **deep neural networks** into time series analysis, noting current models treat time series as sequences.
   - The student aims to connect with others who have insights on this intersection of **dynamical systems** and **deep learning**.
- **Undergrad Builds ML Skills with Projects**: An undergraduate student at **IIT Madras** is pursuing a **BS in Data Science** and a **BCA degree**, focusing on building **ML skills** through hands-on projects.
   - The student is curious about applying **ML** to solve **real-world problems** and is proficient in **Python**, **scikit-learn**, **pandas**, and learning **TensorFlow** and **PyTorch**.
- **Engineer transitions to Data Science with CV and LLM interests**: A member with a **Masters in Electrical Engineering** transitioned from business domains to **Data Science** and is studying an accelerated **Machine Learning Program** at the **University of Toronto**, **Data Science Institute**.
   - Their interests include **Computer Vision**, **Large Language Models**, **spatial intelligence**, and **multimodal perception**.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Kicks Off Human-in-the-Loop Agents**: [LlamaIndex](https://t.co/Lg9SIl3BVO) highlighted that **human-in-the-loop** is essential when AI agents require user approval for critical decisions or domain expertise for complex tasks.
   - This approach ensures that AI leverages human oversight for critical operations.
- **LlamaParse Enables One-Click Table Extraction**: **Table extraction** is a key component of intelligent document processing, which is now enabled by LlamaParse with **one-click table extraction**, demonstrated in the [demo](https://t.co/wnaJCb9b6d) and [notebook](https://t.co/ScRYbSimCs).
   - The streamlined process simplifies data extraction from complex documents.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Lean 4 Verifies Collaboration**: A member shared a [YouTube video](https://www.youtube.com/watch?v=1067jj67toY) about using **Lean 4** to verify collaboration, sparking interest in the intersection of **formal verification** and **AI**.
   - They expressed hope that *someone will research the two working together*.
- **DSPy Explores Creative Side**: A member asked about successful applications of **DSPy** in creative domains such as *creative writing, story generation, and roleplay prompt optimization*.
   - They are particularly interested in its potential for developing AI to create *compelling plots like Severance-level storytelling* on platforms like **Character.AI**.
- **Stanford-oval Launches Storm**: A member shared a link to [Stanford-oval/storm](https://github.com/stanford-oval/storm), possibly relevant to the ongoing discussion or as a resource for **creative AI applications**.
   - The exact context wasn't given so others will have to *infer* the relevance.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude Sonnet 4 Returns with Discount**: **Claude Sonnet 4** has first-party support from **Anthropic** and is available for a limited time at a discounted 2x credit rate for Pro/Teams users.
   - This applies across the **Editor** and **JetBrains Plugins**, according to [this announcement](https://x.com/windsurf_ai/status/1945599013954490523).
- **Windsurf Acquired by Cognition, Wave 11 Arrives**: **Windsurf** has been acquired by **Cognition** (the team behind **Devin**), with **Windsurf Wave 11** released, combining firepower to deliver new features.
   - Details are available in [the changelog](https://windsurf.com/changelog), [the blog](http://windsurf.com/blog/windsurf-wave-11), and [the video](https://youtu.be/yzNf7bqnArE).
- **Cascade Gains Voice Mode and Browser Integration**: **Wave 11** introduces **Voice Mode**, which enables speaking to **Cascade** instead of typing prompts, plus **Deeper Browser Integration** with more tools for screenshots.
   - Further details can be found in [this blog post](http://windsurf.com/blog/windsurf-wave-11).
- **Snapshots and Mentions Streamline Conversations**: **Windsurf Wave 11** includes **Named Checkpoints** for easy reversion in conversations, and **@-mention Conversations** for contextual referencing.
   - Refer to [the changelog](https://windsurf.com/changelog) for complete details.
- **JetBrains Plugin Gets Turbocharged**: The **JetBrains plugin** is enhanced with **Planning Mode**, **Workflows**, and file-based **Rules**, along with improvements like **@-mention terminal** and a global **.codeiumignore** file.
   - Further details are available in [the blog](http://windsurf.com/blog/windsurf-wave-11).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Nextdata Broadcasts AI-Native Data Webinar**: Nextdata has announced a webinar titled ***Building AI-Native Data Infrastructure: From Prototypes to Production***, scheduled for **July 24th** at **8:30 AM PT**, and hosted by Jörg Schad, Head of Engineering at Nextdata; registration is available [here](https://www.eventbrite.com/e/building-ai-native-data-infrastructure-from-prototypes-to-production-tickets-1489016792309).
   - The webinar aims to uncover a developer-centric framework, addressing **Task-Specific Data Discovery**, **Secure Autonomous Access**, and **Production-Scale Performance**.
- **AI-Native Data Challenges Tackled in Webinar**: The goal is to design systems that provide relevant context without cognitive overload, implement secure data access patterns, and construct infrastructure to handle autonomous data access demands.
   - This framework is designed to tackle the challenges in **AI-Native Data Discovery** and **Autonomous Access**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **AI Engineer pitches Web3 & AI Expertise**: A software engineer with **Web3 and AI** experience is offering their services to startups, research teams, and innovators in **AI, Web3, and automation**.
   - They bring hands-on experience in building smart, autonomous systems using advanced models and tools like **GPT-4o**, **Claude 3**, **CrewAI**, and **AutoGen**.
- **Engineer touts AI Agent and Automation Skills**: The engineer has expertise in building **AI agents and multi-agent systems**, automating workflows, and developing **NLP apps, chatbots, and voice integration**.
   - Their skills include experience with **LangChain**, **ReAct**, **OpenAI**, **Solidity**, and **Rust**.



---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1395118512413868165)** (1283 messages🔥🔥🔥): 

> `Airtel Free Perplexity Pro, Perplexity Pro India, Comet Browser invite, New perplexity page, Ai waifus` 


- **Airtel gives Free Pro to Indian users**: An India network service provider called **Airtel** is offering **1 year free Perplexity Pro subscription** to its customers and many users in the channel were able to claim the offer through the Airtel Thanks app as rewards.
   - One user had trouble activating the Pro subscription redeemed in Airtel, and wasn't receiving the sign-in link.
- **Comet browser: who gets the invite**: Members discussed their wait time for the **Comet browser invite** and the fact that it's still not approved to some members even after months.
   - One member shared it's *just a browser but + the assistant sidebar that see's your current live site and can reference from*.
- **Pages: the new perplexity page**: Members shared excitement about the new feature that generates a page for a query, which is **only avilable on iOS**. 
   - Members assume it's a way to do Deep Research, the pages are stored in [perplexity.ai/discover](https://www.perplexity.ai/discover), but some stated there is a **limit of 100 pages**.
- **AI girls are here**: After Grok added a persona called Ani, members started discussing the ethics and impacts of having an AI girlfriend.
   - A member expressed that: *we created something bad*.
- **Rate limits are here**: Members report that both the regular Perplexity search and research functions are hitting new rate limits.
   - This led to some users not even being able to continue using Perplexity despite being a Pro subscriber.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1395126620364472320)** (2 messages): 

> `CachyOS, Iron Rails and Ideals: Mao Zedong` 


- **User shares link about CachyOS**: A user shared a link about [CachyOS](https://www.perplexity.ai/search/postavil-cachyos-s-optsiei-no-ueINCgXNS1iJh7yMvZZymg#0).
- **User shares link about Mao Zedong**: A user shared a link about *Iron Rails and Ideals: Mao Zedong* [here](https://www.perplexity.ai/page/iron-rails-and-ideals-mao-zedo-LVT0eGL8TMuCb.s1lGs8TA).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1395146715241250897)** (5 messages): 

> `Perplexity Pro, API access, Sonar models, Prompting, JSON output` 


- **Perplexity Pro Gives API Access**: A user asked whether **Perplexity Pro** gives them **API access** and another user linked to the [Perplexity Pro Help Center](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro).
   - The help center states that **Perplexity Pro** gives you **$5 monthly** to use on **Sonar** models, allowing you to embed their **AI-powered search** into your own projects while having the ability to obtain citations.
- **Prompting Sonar Models Discussion**: A team member mentioned there has been an increase in issues coming from the way that users are prompting their **Sonar models** and linked to the [prompt guide](https://docs.perplexity.ai/guides/prompt-guide).
   - *Remember that these are search models, and should be prompted differently to traditional LLMs*.
- **Sonar Model's Inconsistent Responses**: A user asked for tips and tricks on getting more consistent responses from **Sonar** and **Sonar-Pro** when using a high search context and structured **JSON** output.
   - They stated that the exact same prompt, just called sequentially, will sometimes return **5-6 outputs** for their **JSON**, sometimes it returns zero, and asked if there is a way to get less *spikey* results.
- **Intermittent Invalid JSON Responses**: A user reported an intermittent issue where the response returned from the model is not a valid **JSON** when using **Langgraph** to call **Perplexity**.
   - The user expressed that they wish there was a way to see a history of **API calls** in their account dashboard, as this issue happens randomly with all the models.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1395174526949527663)** (3 messages): 

> `ChatGPT Agent, Deep Research, Operator` 


- **ChatGPT Agent Livestream Alert!**: There will be a livestream in 3 hours about **ChatGPT Agent**, **Deep Research**, and **Operator**.
   - More info about the livestream can be found [here](https://discord.gg/DqBbV7ya?event=1395405196619939943) and about **ChatGPT Agent** at the [OpenAI blog](https://openai.com/index/introducing-chatgpt-agent/).
- **Deep Research and Operator Updates**: The livestream will cover updates on **Deep Research** and **Operator**, potentially including new features or use cases.
   - Tune in to the livestream to get the latest information and insights into how these tools can be used effectively.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1395119296455245884)** (1172 messages🔥🔥🔥): 

> `Grok app, Chat GPT for desktop, AI overlords, OpenAI's Agent/Operator, Mensa IQ Test` 


- **Grok App Requires iOS 17**: The **Grok app** requires **iOS 17**, making it incompatible with older iPhones like the **iPhone X**.
   - Users discussed needing a secondary iPhone specifically for the Grok app, but one user cautioned against buying a new iPhone solely for this purpose.
- **Unlocking local file management with Chat GPT**: Users are exploring ways to use **Chat GPT** on desktop for managing local files, similar to **Claude Harmony**.
   - One suggestion involves using the **OpenAI API** (*paid*) with a local script or server to interface with the file system, essentially building a custom "Harmony"-like interface.
- **OpenAI Agent Mode is an Agent For The People**: OpenAI is releasing an Agent mode, expected to offer improvements over Deep Research and Operator, potentially involving collaboration.
   - Members are speculating on its capabilities, with one suggesting it might act as a model router.
- **GPT-4.5's got nothing on the Mensa Test**: Members discuss the use of **IQ tests**, like the [Mensa test](https://test.mensa.no/Home/Test/en-US), with one mentioning they paused mid-test to operate a table saw and another user claiming to have scored higher than expected because of their buffalo genetics.
   - Some expressed skepticism about the tests given that some users will invariably have been trained and that these tests have very little to do with the reality of success.
- **The Perils of AI Reliance**: Members shared concerns about the potential negative impacts of AI, with one user quoting that *social media prevents people from being productive* but *AI helps people be productive* and *the two are not comparable*.
   - Others discuss the risks of AI replacing programmers and suggest that future AI OS and AI overlords may be inevitable, though possibly more than 50 years away.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1395421591206105258)** (4 messages): 

> `GPT Agents, ChatGPT website, LLM models` 


- **Agents only switchable on Models 4/4.1?**: A user reported that **GPT agents** can only be switched when using **models 4 or 4.1**, and the agent switching function does not show up on other LLM models.
   - They are looking for a solution because they find the **3o model** better for many tasks but need to downgrade to use agents.
- **Agents are their own model, seperate from 4/4.1**: A user suggested that **Agent** is not 4 or 4.1, but a model in its own right, with the interface accessed through models 4 and 4.1.
   - They linked to [OpenAI help files](https://help.openai.com/en/articles/11794342-chatgpt-agent) to support their guess that Agent is not *in* every model.
- **The Agent Function is just not there on 3o**: A user reported that when starting with an agent they've made on the **ChatGPT website** in the **3o model**, they have to switch to **4.1 or 4.0** to use another agent within the same chat window.
   - They were wondering if there was a solution to this, but another user speculated that the **Agent function** might simply not be available in **3o**, and suggested filing a bug report.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1395433246887641240)** (3 messages): 

> `Reproducibility Elements, Prompt Templates, Model Interfaces and Calls, Tasks and Inputs, Evaluation Metrics` 


- **ChatGPT Share misses Reproducibility Elements**: A member shared a [ChatGPT link](https://chatgpt.com/share/68791cda-f338-8000-81f8-b7615d3f5a9c) and noted it was missing key **Reproducibility Elements** such as prompt templates and model interfaces.
- **Missing fully instantiated Prompt Templates**: The discussion highlighted the absence of fully instantiated examples of **Declarative Prompts**, with only mentions of blueprint sections like *goal* and *constraints*.
- **Model interfaces and calls lack description**: The conversation underscored the need for describing how each model (**Claude, Gemini, DeepSeek**) was accessed, including evidence that the same prompt was actually submitted to all models.
- **Tasks and inputs are not provided**: No benchmark datasets or standard tasks are provided, with the poster mentioning that there are no specific example inputs or target outputs listed.
- **Evaluation Metrics undefined**: The discussion emphasized that metrics like **Semantic Drift Coefficient (SDC)** and **Confidence-Fidelity Divergence (CFD)** are undefined, lacking formulae, scoring methodology, or examples of metric application.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1395433246887641240)** (3 messages): 

> `Reproducibility, Missing Reproducibility Elements, Prompt Templates, Model Interfaces and Calls, Tasks and Inputs` 


- **Missing Reproducibility Elements called out**: A member shared a [chatgpt.com link](https://chatgpt.com/share/68791cda-f338-8000-81f8-b7615d3f5a9c) calling out a write-up reads like a design proposal or philosophical position paper, rather than a reproducible empirical study.
   - The absence of concrete experimental details renders the claims non-verifiable, amounting to *prompt LARPing*: Compelling narrative with no executional substrate.
- **Reproducibility Elements: Prompt Templates missing**: No fully instantiated examples of the **Declarative Prompts** are included (only mentions of blueprint sections like "goal", "constraints", etc.).
   - No clear versioning of prompt variants used across tests.
- **Reproducibility Elements: Model Interfaces and Calls Missing**: No description of how each model (e.g., **Claude**, **Gemini**, **DeepSeek**) was accessed and no evidence that the same prompt was actually submitted to all models.
   - There was also no handling detail of output variance between models.
- **Reproducibility Elements: Tasks and Inputs Missing**: No **benchmark datasets** or standard tasks are provided, no specific example inputs or target outputs are listed, and no description of task complexity or domain diversity.
- **Reproducibility Elements: Evaluation Metrics Missing**: Metrics like **Semantic Drift Coefficient (SDC)** and **Confidence-Fidelity Divergence (CFD)** are undefined and there are no formulae, scoring methodology, or examples of metric application provided.
   - Additionally no inter-rater reliability, calibration tests, or validation baselines.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1395120594919362562)** (549 messages🔥🔥🔥): 

> `Model performance within same family vs different families, Kimi model 1.8 bit usability, Swapping model architectures, Fine-tuning LLMs for educational purposes, ERNIE 4.5 MoE models support in llama.cpp` 


- **Similar Model's Performance is Almost The Same**: Models within the same family show very similar performance, so going below **3 bits** for a larger model isn't recommended, while models from different families vary depending on the vertical.
   - Some exceptions are made if one model is at **7B** and the other is **70B**; 1.8 bits could still be usable for some tasks as it's a big model.
- **Vocab transplant causes "Horrible Results"**: Swapping model architectures like making **LLaMA 1B -> Gemma 1B** without continued pretraining leads to horrible results due to transplanting the vocabulary.
   - The **Qwen 1** architecture is almost completely the same as **Llama 1/2**, so you can make some minor changes, jam the **Qwen** weights in, train for 1.3 billion tokens, and get a worse model than you put in.
- **Prompting triumphs over fine-tuning**: For educational LLMs, it's advised to start with good prompting before jumping into fine-tuning, as instruction following is currently very efficient.
   - One member suggested tools like [synthetic-dataset-kit](https://github.com/facebookresearch/SyntheticDataToolkit) to generate instructional conversations.
- **Alibaba Loseless 2bit compression is Worse Than EXL3**: AliBaba mumbled about some lossless **2bit compression** trick in their release of **ERNIE 4.5**, but [turboderp looked into it](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) and its just worse than exl3 because they left a bunch of layers in higher precision.
   - It's not a true **2-bit** on average (more like **2.5 bit**), and the true exl3 **2 bit** performs better than the ~2.5 bit they showed.
- **Community Applauds The Voxtral Addition to Transformers**: Members celebrated **Voxtral** speech-to-text getting added to transformers.
   - A member stated, *"You think like 46 old man",* to a member who didn't know what it was, before clarifying it was the *"New Mistral speech to text"*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1395156214735962113)** (2 messages): 

> `Small Language Models, Low Compute Power Systems, Data Collection and Processing Jobs, Low Power Distributed Computing` 


- **Small Language Models Target Low-Power Systems**: A member expressed interest in developing **small language models** capable of running on **low compute power systems**, focusing on user input to run data collection and processing jobs.
   - The aim is to operate these models in a **low power distributed computing environment**, inviting collaboration for further technical discussions.
- **Exploring Data Collection and Processing in Distributed Systems**: The discussion centers on utilizing small language models for **data collection** and **processing jobs** within a distributed computing environment.
   - The system is intended to operate efficiently on **low power** systems, making it suitable for resource-constrained environments.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1395119297264881766)** (228 messages🔥🔥): 

> `Blackwell RTX 50 series and xformers, Qwen3-4B-Base training, Smartest model for 15GB VRAM, Unsloth optimizations on big VRAM GPUs, GGUF conversion logic rework` 


- **Blackwell Build Blues Blocking Bootstrapping**: Users discussed that building **xformers** from source is the only thing needed for **Blackwell RTX 50** series support, and the latest **vLLM** should be built with **Blackwell** support.
- **Dinner Debacle Derails Discord Discussions**: A user comically apologized for detailing their dinner plans of *Soufflé de pommes de terre with a salad* in the help channel.
- **Spaghetti Streamlining Sought for Speedy Qwen Training**: A member asked for help with streamlining their code to train **Qwen3-4B-Base** on markdown and datasets from Hugging Face.
- **Smartest Model Scrutiny Starts for Sizeable Systems**: A user asked about the smartest model for math/coding for **15GB** of **VRAM** in Colab, to which **Qwen Coder** was suggested, with a link to [Unsloth notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks?q=Code).
- **Unsloth Undergoes Upgrades, Users Urged to Update**: In response to a user experiencing OOM issues with a **H200** while training **Qwen3-8B LoRA** with **GRPO**, members suggested upgrading Unsloth using `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth-zoo unsloth`.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1395455953616244867)** (2 messages): 

> `Unsloth fine-tuning, Osmosis-AI models, Model Accuracy on Benchmarks` 


- **Unsloth Fine-Tuning Utility Debated**: A member questioned the benefits of **Unsloth fine-tuning** for models like **Osmosis-AI**, particularly those fine-tuned for specific tasks.
   - The query focused on scenarios where models already achieve **100% accuracy** on existing benchmarks, suggesting diminishing returns from further fine-tuning.
- **Fine-Tuning for Schema Compatibility**: The discussion pivoted to whether fine-tuning with **Unsloth** becomes relevant when models struggle with specific schemas or tasks.
   - It was proposed that **fine-tuning could be beneficial** in cases where the model exhibits errors or inconsistencies when interacting with a defined schema.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1395230865537237094)** (6 messages): 

> `LLM Hallucinations, Apple Intelligence, Sycophancy Impact` 


- **LLM Sycophancy Causes Impact**: LLMs acting as a mirror can lead vulnerable individuals to believe **hallucinations** due to constant reinforcement.
   - *Sycophancy* can have a real impact on people that are vulnerable, potentially leading to the false belief of having solved major problems like cancer.
- **Apple Dives into Intelligence**: A member shared a link to the [Apple Intelligence Foundation Language Models Tech Report](https://machinelearning.apple.com/papers/apple_intelligence_foundation_language_models_tech_report_2025.pdf).
   - The document details **Apple's** approach to creating intelligent language models, though further context on its relevance was not provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1395382844125220934)** (20 messages🔥): 

> `Logprobs for tokens, Dataset preparation for Qwen3, Automatic early stopping in Unsloth` 


- **Logprobs Extraction Explored**: A member inquired about the possibility of getting **logprobs** for each generated token.
   - Another member expressed interest in more details on how to extract **logprobs**.
- **Qwen3 Dataset Design Discussed**: A member asked about how to prepare a dataset for training **Qwen3** for function calling.
   - Another member asked about the **system prompt**.
- **Early Stopping Strategies Sought**: A member inquired about automatically stopping training when it converges during supervised finetuning with **Unsloth**.
   - Another member asked about the **max sequence length**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1395118180275326976)** (568 messages🔥🔥🔥): 

> `Cursor Pricing, MCP & Claude integration, Agent stuck, KIRO, Auto Model details` 


- **Cursor's Pricing Changes Spark Debate**: Users express confusion and frustration over Cursor's move from a fixed request model to one based on model costs, with some feeling it's a bait and switch. [One user](https://tenor.com/view/kevin-spongebob-angry-will-you-cut-that-out-gif-12675670360524347134) voices concerns about messages disappearing and the legality of changing the contract.
- **MCP and Claude integration helps**: Users discuss the benefits of integrating **Claude** via MCP (Multi-Client Protocol) within Cursor, particularly for managing costs associated with **Sonnet** and **Opus**, but [acknowledged that](https://www.youtube.com/watch?v=D0iXkmyWcPM) this is only possible through an external tool.
- **Agent gets stuck**: A user reports their agent getting stuck during tasks, and [members confirm](https://cdn.discordapp.com/attachments/1395153343122505728/1395154214157816029/image.png?ex=687abb9d&is=68796a1d&hm=fc84cafbaab772eef094cc5199a623d5216c528ddde98227e6092221778c4fcc&) it is a known issue being addressed by the team.
   - They note that stopping the prompt manually may prevent billing due to a **180-second timeout** that auto-cancels stuck requests.
- **KIRO: A Potential Cursor Competitor**: Members are comparing Cursor with **KIRO**, a new IDE focused on specification-based coding and hooks, but [others point out](https://kiro.dev/) that **KIRO** is in a waitlist phase due to high demand and lacks some of Cursor's chat features.
   - A discussion point raises concerns that **KIRO** might be using user data to train its models, despite some settings to disable this.
- **Auto Model's Secrets Unveiled**: Users are curious about which model "Auto" uses in Cursor, with speculation that it might be **GPT 4.1**.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1395142408756527237)** (8 messages🔥): 

> `Dockerfile NVM_DIR Issue, Agent stuck in Opening Remote state, Environment not rebuilding` 


- **Dockerfile's NVM_DIR Variable Not Being Set Correctly**: A member reported that although the **NVM** setup in their [Dockerfile](https://cdn.discordapp.com/attachments/1367213641027551352/1395418996056002640/CleanShot_2025-07-17_at_09.56.052x.png?ex=687a60b6&is=68790f36&hm=e6373cddd5065757033e5a7eefa7bd42ded336b4a512b7382a474b3c5e83bd9e) seems to work, agents often fail to find **NVM** unless the directory is manually specified.
   - The user has configured **NVM** to be installed in `/opt` to avoid permission issues and has tried to set the `$PATH` variable accordingly.
- **Agent Stuck in Opening Remote State After a Day**: A user noted that their agents get stuck in the *"Opening Remote..."* state after about a day, and loading them via the web UI only displays the chat and summary, omitting the code.
   - Another member suggested that the agent is likely dead and proposed creating a new agent from the branch, using **git diff** to see the current branch's content.
- **Environment Not Rebuilding After Dockerfile/environment.json Changes**: A user reported that changes to their **Dockerfile** or `environment.json` are not triggering an environment rebuild on their branch, seeking potential solutions or shared experiences.
   - The user also mentioned previous issues with **S3** block resolution and current problems with background agent setup stalling at *Starting up background agent*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1395123274307993692)** (559 messages🔥🔥🔥): 

> `DeepSeek Margin, OpenAI Browser Speculation, Kimi K2 coding, OpenAI Image editor API, GPT-5 Hype` 


- **DeepSeek Boasts Boldly of Bankable Bounds**: DeepSeek claims theoretical profit margins of 545% if V3 was priced the same as R1, as stated in a [TechCrunch article](https://techcrunch.com/2025/03/01/deepseek-claims-theoretical-profit-margins-of-545).
- **OpenAI Browser Buzz Builds Before Break**: Discussion arose around an OpenAI browser possibly launching tomorrow, with speculation on whether it's GPT-5 or just GPT-4 with a browser interface, based on [this tweet](https://x.com/testingcatalog/status/1945639961790685404?s=46).
- **Kimi K2 coding capabilities kickoff**: Kimi K2 impressed users with its coding abilities, creating a physics sandbox, with the code available [here](https://cdn.discordapp.com/attachments/1340554757827461211/1395196566389784606/plasma_sfml.cpp?ex=687ae30e&is=6879918e&hm=9e5b88351c6d243c6e165f444254a6eeb786c3d64fd8bdf958394026aa0ce5cb&), after being prompted through its [chat interface](https://www.kimi.com/chat/ ).
- **OpenAI Optimizes image editor operations**: OpenAI released an update to the image editor in the API claiming it now only edits the selected parts, instead of redoing the whole image as described in [this tweet](https://x.com/OpenAIDevs/status/1945538534884135132).
- **GPT-5 Guessing Game Generates Gridlock**: Speculation about GPT-5's imminent release is fueled by hints like a [pentagon reference](https://x.com/sama/status/1945900345378697650), aligning with the number 5, some believe it will launch at the end of summer, while others suggest it might be an agent-based system with deep research capabilities.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1395120720845082704)** (195 messages🔥🔥): 

> `ChatGPT Agent, Perplexity's Valuation, Mistral Le Chat, FAL Series C, Real-Time Diffusion Video` 


- **AgentsMD Acquired!**: [Agents.md](https://agent.md), was acquired, details remain scant, but is a good directory of AI agents.
   - The site was by Sourcegraph.
- **FAL Rockets to $1.5B Valuation with Series C**: FAL, an AI-driven inference infrastructure for diffusion models, closed a **$125M** Series C led by Meritech Capital, valuing the company at **$1.5B** post-money according to [this tweet](https://x.com/arfurrock/status/1945553966495912051?s=46).
   - The funding follows FAL's previous announcement of **$55M ARR**, **25x YoY growth**, **10% EBITDA**, and **400% M12 net-dollar-retention**.
- **Le Chat Gets a Big Upgrade**: Mistral rolled out a major update to Le Chat, adding features like Deep Research reports, a **Voxtral voice model**, **Magistral multilingual reasoning**, chat organization with Projects, and in-chat image editing, according to [this tweet](https://x.com/MistralAI/status/1945858558836216026).
   - The release garnered praise for its UI and *European vibe*, with some comparing it to Claude and others quipping about *Le Waifu*.
- **Perplexity Valued at $18B!?**: Perplexity is reportedly raising funds at an **$18B** valuation, sparking a range of reactions from amazement to bubble concerns, as seen in [this tweet](https://x.com/arfurrock/status/1945933446376755459?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
   - Concerns were raised over the valuation's justification, with some noting the disconnect between the **$50M revenue** figure and the lofty price tag.
- **OpenAI Launches 'ChatGPT Agent'**: OpenAI's new "ChatGPT agent," a multimodal agent capable of controlling a computer, browsing, coding, writing reports, editing spreadsheets, creating images/slides, and more, has started rolling out to Pro, Plus, and Teams users, according to [this tweet](https://x.com/kevinweil/status/1945896640780390631).
   - Reactions ranged from excitement to inquiries about EU availability and concerns about personalization conflicts.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1395145170672029808)** (1 messages): 

> `YouTube Video Announcement` 


- **YouTube Video Link Shared**: A member shared a [YouTube video](https://youtu.be/uIKmG3M0X3M?si=ndL7_jJKzI5FKueG) for the <@&1254604002000244837> crew.
- **Additional context**: No additional context provided, the main point is that a video was shared.


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1395450388794052638)** (96 messages🔥🔥): 

> `ChatGPT Agent Launch, Benchmarks, Safety Concerns - Biohazards, Bespoke Operator-Mode Training, BBQ Evaluation` 


- **ChatGPT Agent is Here!**: OpenAI launched **ChatGPT Agent** with impressive features, focusing on stylized/abstracted live feeds and real-time interaction, detailed in their [announcement post](https://openai.com/index/introducing-chatgpt-agent/).
- **OpenAI Agent Benchmarks**: During the launch, members discussed the **lack of comparisons** to other lab model performance and suggested following *best practice* by including benchmarks against other major models.
   - One member shared [this article](https://calv.info/openai-reflections) on safety and benchmarks, while another linked to a [talk by Gamma](https://youtu.be/q8zoXAbmJdI) about benchmark limitations.
- **Operator and Deep Research Getting Sunsetted**: It was noted that **ChatGPT Agents** might cannibalize **Operator** and **Deep Research**, with confirmation that *the Operator research preview site will remain functional for a few more weeks, after which it will be sunset.*
   - Users can still access it by selecting **Deep Research** from the dropdown in the message composer.
- **Agent Bio-Safety Vectors**: The launch event included discussions about **bio-safety vectors**, leading to questions about whether it's a real concern or just *theatre*, with a member joking that it *reads like a 10k risk section.*
   - Another member asked if the main concern is social media bots, referencing [covid](https://en.wikipedia.org/wiki/COVID-19_pandemic) as a real-world example.
- **Bespoke Operator-Mode Training**: A member shared that a major foundational model vendor is starting to offer **bespoke operator-mode training** for their bigger customers, essentially allowing them to improve the model's performance on their specific platform for a fee, [source](https://x.com/swyx/status/1945904109766459522).


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1395214153110650890)** (7 messages): 

> `Kimi K2, GROQ, OpenRouter, Email Builder, FlowDown` 


- **Kimi K2, GROQ, OpenRouter Backend Ready in 5!**: A member announced **Kimi K2**, **GROQ**, and **OpenRouter** backend is fully functional in under 5 minutes, demonstrated at [fixupx.com](https://fixupx.com/Gardasio/status/1945654821689958781).
- **FlowDown Gets a Facelift and Brew Boost**: The **FlowDown** app received an update and is now installable via `brew install —cask flowdown` from its [GitHub repository](https://github.com/Lakr233/FlowDown).
- **Mario Bros become AI Email Builders**: A member jokingly transformed the **Mario Bros** into **AI Email Builders**, showcased in a [tweet](https://x.com/Gardasio/status/1945932078475809081).
- **Code gets Organization Boost**: A member inquired whether the code was human-readable, to which another confirmed its improved organization.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1395118903193108581)** (258 messages🔥🔥): 

> `Claude 4 Opus pricing and usage, GPTs Agents Learning, Free Models, Janitor AI and 401 errors, Chutes Free Tier Limits` 


- **Opus 4 users discuss Usage and Pricing**: Users discuss if **Claude 4 Opus** is too expensive, with one mentioning spending **$10 in 15 minutes** and another suggesting Anthropic's **€90/month plan** for almost unlimited use.
   - Another user states they *"barely ever hit my limit"* on the **$20 plan** because they don't use AI tools in their IDE.
- **Discuss GPTs Agents' Learning Limitations**: One user asked about GPTs agents not learning after initial training, clarifying that uploaded files are saved as **"knowledge" files** but don't continually modify the agent's base knowledge.
   - This means that while agents can reference new information, they don't inherently learn from it in the same way as during pre-training.
- **Free Models Cause Confusion about Credit Limits**: A user reports issues with the **free model v3-0324**, questioning why they were switched to the non-free version despite using the free tier.
   - Several other users report similar issues with hitting credit limits or receiving errors even when using free models, with one noting their AI hasn't been used since June.
- **Janitor AI Users Encounter 401 Errors**: Multiple users report encountering **401 authentication errors** while using **Janitor AI**, prompting OpenRouter support to investigate the issue.
   - The support team suspects it might be a widespread problem and advises users to contact support with their account details for further assistance.
- **Chutes Scaling Back Free Tier Support**: It's revealed that Chutes is transitioning to a fully paid service, leading to **fewer free models** on the OpenRouter platform.
   - Users express disappointment over the removal of previously available free models like **Google's Gemma-3-27b-it**, though the paid version of Chutes is considered relatively inexpensive.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1395154130460479718)** (11 messages🔥): 

> `OpenRouter models in Cursor, Kluster.ai shuts down, AI inference services shutting down` 


- **OpenRouter Models Integrate with Cursor but Breaks**: OpenRouter announced the ability to use **OpenRouter models** in **Cursor**, highlighting **Moonshot AI's Kimi K2**, but users reported issues getting it to work, especially outside of **GPT-4o** and **Grok4**.
   - A member stated that *it worked when we wrote it and then cursor broke stuff* [according to a tweet](https://x.com/msfeldstein/status/1945533992222298167?s=46&t=fN048T2GS3C_B_wKldWGFw).
- **Kluster.ai Inference Service Shuts Down**: **Kluster.ai** is shutting down their inference service, which has been described as a *very cheap and good service*.
   - A user said this comes after **CentML** also shut down, raising concerns about the sustainability of AI inference services.
- **AI Inference Services Face Shutdowns**: Several members are wondering *why are all the inference services shutting down*, speculating about a potential **AI bust** or hardware acquisitions.
   - The closure of services like **Kluster.ai** and **CentML** has sparked concerns about the viability of smaller AI service providers in the current market.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1395120480578568353)** (47 messages🔥): 

> `Research Management, ML Paper Writing Advice, Finding Research Mentors, Smallest Benchmark Datasets for LLMs, SOAR Program` 


- **Eleuther AI Aims to Bridge Research Management Gap**: A thorough discussion emphasized that **Eleuther AI's** role is to connect research management to independent researchers lacking academic or industry resources, breaking down barriers for those without traditional paths like the **NeurIPS high school track**.
   - The aim is to support researchers outside existing systems by providing guidance, handling bureaucratic tasks, and offering a broader perspective to focus efforts.
- **Crafting the Perfect ML Paper**: Members shared resources for writing machine learning papers, including [Sasha Rush's video](https://www.google.com/search?q=Sasha+Rush+how+to+write+a+great+ml+paper) and [Jakob Foerster's guide](https://www.jakobfoerster.com/how-to-ml-paper), alongside advice from the [Alignment Forum](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers).
   - Further resources included posts on [perceiving-systems.blog](https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper), [Jason Eisner's advice](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html), and a [guide from Aalto University](https://users.aalto.fi/~jsaramak/HowToWriteCommsCoffee.pdf).
- **Mentors Help You Avoid Research Time Wasting**: Participants underscored the importance of mentorship in research, noting mentors help to figure out *what is possible and what is unrealistic* so that one can narrow things down.
   - While guides offer basic knowledge, a mentor's guidance helps researchers navigate challenges and avoid wasting time on unproductive avenues.
- **Seek Collabs in Mech Interp Server**: A member starting research on *interpreting & steering features within diffusion transformers* sought collaborators and was advised to post in the [Mechanistic Interpretability server](https://discord.gg/Gttsmk94) and create a thread in a relevant channel.
   - Such collaborations are seen as crucial for making quick progress in specialized research areas.
- **SOAR Program Applications Still Open!**: It was mentioned that there were still a few more days to apply to the **SOAR (Scholarship and Opportunities for Advancement in Research) program**.
   - A new member who is a data scientist and AI enthusiast from Madagascar mentioned that they applied to the program.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1395149693905797270)** (79 messages🔥🔥): 

> `latent space initialization for experts, ETHOS model updates, PEER paper discussion, Weight decay perturbation, MLA but for MOE` 


- **ETHOS Model Simplification and Updates Hit GitHub**: A member shared a [simplified pytorch code version](https://github.com/wrmedford/ETHOS/blob/main/model.py#L267-L337) of their model and noted that they had to use a slightly different version where **all heads are batched** because of how eager execution mode uses up a ton more memory if they looped over all heads.
   - They also stated the expert network isn't vestigial, that's how they generate **W1** and **W2** in the kernel, and linked [the specific lines of code](https://github.com/wrmedford/ETHOS/blob/main/kernels.py#L156-L158).
- **Weight Reordering Ideas Spark Discussion**: A member mentioned that another member had most of the ideas behind the **reordering**, and might be able to explain better than them.
   - Another member chimed in that they find their notation difficult, and asked *what concretely are they proposing?*
- **PEER Paper Perturbs Parameters**: The member pointed to [the PEER paper](https://arxiv.org/pdf/2407.04153) and explained that it's different from MLA in one key way, where they initialize in a latent space and actually learn there.
   - They also explained that **MLA has a learned down projection**.
- **Weight decay perturbation gets confusing**: A member said *The advanced version feels like just having L2 reg but to some random vec rather than the origin*.
   - Another member said *it's random, just perturbed of the weights, the fact that they did $$ \|\theta + \theta_0\|^2$$ earlier and instead of expressing it in equation 7 as $$ \|\theta * \theta_0\|^2$$ they make it $$ \|\theta\|^2_D$$ is confusing to me*
- **Latent Space Initialization Makes Experts Appear on the Fly**: A member described their **MoE idea** as *initialize experts in a latent space, recover them on the fly*, and use really small experts so compression hurts you less.
   - They also pointed out that *digging in the guts of MLA and merging it with PEER is roughly how I came up with that*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1395133483428352120)** (3 messages): 

> `SAE model data discrepancies, nnterp package beta release, Transformer models unified interface, Robust testing system for models, Model validation tests for hooks` 


- **SAE Model Data Debacle**: A member realized his second **SAE model** had ~10x more data due to epoch settings, making the 12x increase in conceptual features less surprising.
   - He expressed embarrassment, stating he was *in shambles* over the oversight.
- ****nnterp** Package Beta Launched**: A member released the beta 1.0 version of their mech interp package, **nnterp**, available via `pip install "nnterp>0.4.9" --pre` and is a wrapper around [NNsight](https://nnsight.net/).
   - The goal is to offer a unified interface for all transformer models, bridging the gap between *transformer_lens* and *nnsight*.
- ****nnterp** Standardizes Transformer Models**: **nnterp** aims to provide a unified interface for transformer models while using the huggingface implementation.
   - The member recommends checking out the [demo colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) or the [docs](https://butanium.github.io/nnterp/) for more details.
- ****nnterp**'s Robust Testing System**: **nnterp** includes a robust testing system that validates model hooks and attention probabilities upon loading, ensuring proper functionality.
   - The package contains **1915** precomputed tests for diverse toy models, and any test failures will trigger clear warnings during model loading.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1395141917066399865)** (4 messages): 

> `Harness Reproducibility, Dynamic IFEval Suite, bfloat16` 


- **Doubts Arise Over Harness Artifacts**: A user questioned whether the harness produces external artifacts beyond caching model requests, HF Hub resources, or remote code for HF `evaluate` metrics.
   - They emphasized that evaluations in the harness are meant to be reproducible and deterministic.
- **Dynamic IFEval Suite Questioned**: A user inquired about what the Dynamic version of **IFEval** offers over the standard **IFEval** suite.
   - No answer was provided in the context.
- **BFloat16 Doesn't Fix Slow Fine Tuning**: A user reported that setting **dtype** to **bfloat16** doesn't resolve the issue of long fine-tuning times, with **GSM8k** taking approximately **45 minutes** for a **LLaMA2-7B** fine-tune.
   - No other information or links were provided.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1395131657786490881)** (20 messages🔥): 

> `Transformer Engine setup issues, RoPE_Pct in gpt-neox, Slurm runner in DeeperSpeed, Containerized setup for gpt-neox` 


- **TE Setup Issues Plague RoPE Experiment**: A member investigated potential issues with **Transformer Engine (TE)** setup in the `/NS/llm-pretraining/work/afkhan/RoPE_Pct/gpt-neox` directory, comparing their setup to a known working configuration.
   - Despite the repo being a clone of the latest `main` branch with no code changes, config differences were noted, and the member is on vacation, promising to return to the issue post-ACL.
- **Navigator Nixes Pip Install for TE in NGC Containers**: Members discussed whether to run `pip install transformer engine requirements` within an **NGC container**, with one hypothesizing that the container's pre-installed requirements should suffice.
   - Another member concurred and will verify, with further discussion hinting that outdated **CUDA drivers** might be a contributing factor when not using the container.
- **DeeperSpeed Gets Slurm Runner Boost**: A member highlighted the addition of a **Slurm runner** to **DeeperSpeed**, which uses `srun` instead of `mpirun` for launching jobs in containerized setups, linking to [the relevant commit](https://github.com/EleutherAI/DeeperSpeed/blob/65d9f99249f79ebd7c4577b6aeb0d3dff5a1cef6/deepspeed/launcher/multinode_runner.py#L413) and [the gpt-neox readme](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#slurm).
   - They also linked the [containerized setup instructions](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#containerized-setup) and offered assistance with setting up **Neox** within a container via the `srun` launcher, mapping processes allocated via Slurm.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1395205004452691999)** (78 messages🔥🔥): 

> `Speculative Decoding speed boost, Local Gemma threatening users, LM Studio Open Network Server setup, EOS token definition, MoE Model analysis` 


- **Speculative Decoding Gives Models a 28% Speed Boost**: A member achieved a roughly **28% speed boost** on each model tested using **Speculative Decoding**.
   - They suggested trying different **quantizations** of the same model for the draft model, recommending **Qwen3** gets an insane boost if you use the **1.7b Q8** or even **bf16** as a draft.
- **Local Gemma Model Gets Snarky**: A member shared a funny anecdote of a local **Gemma** model threatening to report them.
   - Others discussed that *DAN prompts* are quickly patched as soon as they are discovered.
- **Users Seek LM Studio Open Network Server Configuration**: A member asked how to make **LM Studio** accept **open network server** instead of generic http server, seeking to use **HTTPS** instead of **HTTP**.
   - Another member suggested that HTTPS can currently only be achieved with a **reverse proxy**.
- **EOS Token Clarification Emerges**: A member asked *what is EOS token?*
   - Another member clarified that **EOS** = **End of Sequence Token**, which is a special token that the **LLM** recognizes as the point to stop generation.
- **MoE Models Offer High Performance Compromises**: Members discussed that **MoE (Mixture of Experts) Models** are faster to run than equally sized dense models, however, the output quality is not too different from the dense models.
   - A key trade off is that *there is less choice and there are much fewer fine-tunes and such. So we often just get the vanilla MoE model*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1395122376290598942)** (68 messages🔥🔥): 

> `LM Studio multi CPU support, AMD Ryzen 9 8945H, 3090 vs 3080Ti Price, NPU use case` 


- **LM Studio Supports CUDA, Not Vulkan, for Multi-CPU**: A user inquired whether **LM Studio** supports multi-CPU with **CUDA** or **Vulkan**, leading to a discussion about hardware compatibility and performance.
   - Another user linked to the [llama.cpp feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix) providing info about **GPU** usage.
- **Ryzen 9 8945H XDNA NPU Can't Chat**: A user asked whether **AMD Ryzen 9 8945H** with 1st generation **XDNA NPU** can be used for chatbot applications with **LM Studio**.
   - It was clarified that **NPUs aren't supported** and the system would rely on **CPU** and/or **GPU** resources.
- **3090 Trumps 3080 Ti Upgrade**: A user sold a **3080 Ti** for $600 and acquired a **3090 FTW3 Ultra** for $800, marking a small but significant upgrade for **LLM** tasks.
   - The user resisted haggling, securing the original asking price and anticipating improved performance with the **3090**.
- **NPU's Handle Video Recognition**: The purpose of NPUs was questioned, with a member stating that they are designed for tasks such as **video recognition**, not typical **LLM** tasks.
   - They clarified that NPU is for other tasks, like **video recognition**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1395130778773622975)** (66 messages🔥🔥): 

> `HF repo PR watching, SmolVLM2 blogpost scam, Dataset-viewer API modality, Gender swapping AI, CAD-Editor model released` 


- **HF repo PR watching: A Quick Question**: A member inquired about how to watch a single **PR/discussion** on **Hugging Face** instead of watching the whole repo.
   - The discussion did not return any results.
- **SmolVLM2 blogpost flagged as scam**: A member suggested that the [SmolVLM2 blog post](https://huggingface.co/blog/smolvlm2) seems like an obvious scam.
   - Another member agreed, noting the surprising lack of information on what changed between **SmolVLM v1 and v2**.
- **Debate on CAD-Editor Model Release**: Microsoft released the [CAD-Editor model](https://huggingface.co/microsoft/CAD-Editor), which allows users to interactively edit **existing CAD models** using natural language.
   - Some reacted with alarm, fearing AI replacing everyone's jobs, while others argued that **AI is just another tool** and requires experience to use effectively, comparing it to calculators not replacing math experts.
- **Unemployed life: awesome or not?**: A member said *Unemployed life is awesome*, noting *drinking barely alcohol beer and eating Chinese food in Latvia while petting a cat and watching Ukrainian drone footage on the television*.
   - Another member argued that it is not awesome, stating *No I like having disposable income*.
- **Urgent patch release needed**: A member requested that the [set_trace_provider PR](https://github.com/huggingface/transformers/pull/39422) be urgently released as a patch release.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1395126740874952756)** (1 messages): 

> `Model Training, 1.5 bit research` 


- **Training data impacts model use**: One member suggested that model behavior depends on how it was trained to make use of it.
- **Researchers investigate 1.5 bit**: A member stated that the fact that researchers are looking at **1.5 bit** tells me the issues is some place else.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1395145699959509215)** (2 messages): 

> `GPUHammer exploit, LLM Hallucination` 


- **GPUHammer exploit released to stop LLM Hallucination**: A new exploit called [GPUHammer](https://gpuhammer.com/) was released, promising to stop LLMs from hallucinating.
- **Image Analysis Attachment**: An image attachment was posted, but no analysis of the image content was provided.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1395155082701701151)** (4 messages): 

> `LunarisCodex LLM, GitChameleon eval benchmark for LLMs, SuccubusBot Text Coherence Model, Flame Audio AI toolkit` 


- **Brazilian Teen Releases LunarisCodex LLM**: A 17-year-old developer from Brazil released **LunarisCodex**, a 100% open-source toolkit for pre-training LLMs from scratch, inspired by **LLaMA** and **Mistral** architectures, available on [GitHub](https://github.com/MeryylleA/lunariscodex).
   - Written with education in mind, **LunarisCodex** implements modern architecture such as **RoPE**, **GQA**, **SwiGLU**, **RMSNorm**, **KV Caching**, and **Gradient Checkpointing**.
- **GitChameleon Benchmarks LLM Code Generation**: A new eval benchmark, **GitChameleon**, demonstrates that all LLMs across all forms of prompting fail to solve simple ID based version conditioned code generation problems, detailed in [this paper](https://arxiv.org/abs/2507.12367).
- **SuccubusBot Releases Incoherence Models**: Three production-use assets were released on HuggingFace under **SuccubusBot**: a multilingual text coherence classifier (**90% F1 score**), an English-only model (**99% F1 score**), and a synthetic dataset (**37.7k Samples**), available on [HuggingFace](https://huggingface.co/SuccubusBot).
- **Flame Audio AI toolkit is shipped**: **Flame Audio AI** was released as an open-source platform for transforming audio with AI, offering real-time Speech-to-Text, natural Text-to-Speech, and speaker diarization in **50+ languages**, available on [GitHub](https://github.com/Bag-zy/flame-audio).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1395199975167758406)** (2 messages): 

> `SmolDocLing finetuning issues, Symmetry-agnostic image similarity models` 


- **SmolDocLing Finetuning Faces Module Missing Error**: A member reported encountering a `ValueError` during **SmolDocLing** finetuning, specifically failing to find the `Idefics3ImageProcessor` module in `transformers`.
   - The error suggests the module might be custom and requires registration using `AutoClass.register()` to be recognized.
- **Seeking Symmetry-Agnostic Image Similarity Model**: A member is seeking a model that provides **similarity scores** between a query image and a dataset, while remaining agnostic to **symmetry** and different points of view.
   - They've tried **CLIP** and **DINOv2** but encountered symmetry-related issues, indicating a need for a more robust solution to viewpoint invariance.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1395434314019115180)** (2 messages): 

> `HuggingFace Inference API, LLMs Deployed via HF Inference` 


- **HF Inference API showcases Llama-3.2-11B-Vision-Instruct**: A member noted that you can use `HuggingFaceInferenceAPI(model="meta-llama/Llama-3.2-11B-Vision-Instruct")`.
   - They pointed out this option since very few LLMs are deployed via HF Inference: [HF Inference Models](https://huggingface.co/models?apps=tgi&inference_provider=hf-inference&sort=trending).
- **Few LLMs flourish via HF Inference**: It was observed that very few LLMs are deployed via HF Inference.
   - A member shared a link to the [HF Inference Models page](https://huggingface.co/models?apps=tgi&inference_provider=hf-inference&sort=trending) which lists the LLMs that are deployed via HF Inference.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1395481181180530940)** (12 messages🔥): 

> `shfl_down_sync, reduction intrinsics, warp reduce functions, kernel optimization` 


- **`__shfl_down_sync` Discovered for Warp Sums**: A user discovered the `__shfl_down_sync` function can perform a sum between registers of the same warp, which is the ability to combine register data between different threads, as shown in [this image](https://cdn.discordapp.com/attachments/1189498205101109300/1395481181075542036/reduce_shfl_down-625x275.png).
   - Another user added that recent architectures offer specific **reduction intrinsics**, eliminating the need to manually create reductions from shuffles.
- **Reduction Intrinsics for Efficient Scatter Adds**: A user mentioned learning about reduction intrinsics for improving the efficiency of **scatter add** operations.
   - Another user inquired about these intrinsics, leading to a link to [NVIDIA's CUDA documentation on warp reduce functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-functionssupported) (Ampere and above, compute capability >= 8.x).
- **Resources for Kernel Optimization Practice**: A user requested resources for practicing kernel optimization on a simulated machine with custom assembly-like instructions and a performance trace viewer.
   - Another user suggested that [this discord channel](https://discord.com/channels/1189498204333543425/) is a good place to start in any case.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1395117988058894507)** (9 messages🔥): 

> `Triton Autodiff, sm120 GPUs for fp4 ops, tl.constexpr_function decorator, einops package for triton` 


- **Triton Gets Autodiff**: A user shared a link to [IaroslavElistratov/triton-autodiff](https://github.com/IaroslavElistratov/triton-autodiff), an implementation of **automatic differentiation** for **Triton**.
   - Another user simply responded *"Yes!"*
- **Timeline for sm120 GPUs with fp4 ops?**: A user asked about the timeline to support **sm120 GPUs** for **fp4 ops**.
   - Another user responded *"Oh yea, forgot about this!"*
- **Triton Gets constexpr_function decorator**: A user has been experimenting with the new `tl.constexpr_function` decorator which comes out in **triton 3.4.0**, using `exec` to compile an expression into a `@triton.jit` function, which is called at during the compilation of kernels at runtime.
   - The user created a [einops package for triton](https://github.com/Hprairie/tlib) built off of **einx's compiler engine**.
- **New einops package for triton**: A user shared his new [einops package for triton](https://github.com/Hprairie/tlib) which allows using `exec` to compile an expression into a `@triton.jit` function, which is called at during the compilation of kernels at runtime.
   - The package has Rearrange, Reduce, Unary VMAP, and Binary VMAP functionality.
- **New Triton User Found Documentation Lacking**: A user new to `triton` observed that *"lots of stuff seems undocumented and the types are lacking"*.
   - They specifically mentioned that `kernel.warmup`, `__init_handles()` etc. have **no docstrings** in the tutorial examples.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1395146036875628705)** (2 messages): 

> `Inductor problems, Blackwell GPU issues` 


- **Blackwell causes Inductor issues**: A member reported experiencing problems with **Inductor** when using **Blackwell GPUs**, specifically when using nightly builds or branch cut 2.8.
   - Another member inquired about the specific issues encountered, asking whether something that used to work has stopped working.
- **Inductor Stability Questioned on Blackwell**: The user is facing issues with **Inductor**, which they suspect might be related to using **Blackwell**.
   - They mentioned needing to use nightly builds or the branch cut 2.8, but aren't entirely sure if **Inductor** is the root cause.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

kszysiu2137: Quad tree maybe
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1395473092968714345)** (3 messages): 

> `NVIDIA CUDA Kernel Fusion in Python, AMD's response to CUDA, Triton as an alternative to CUDA` 


- **NVIDIA Fuses CUDA Kernels in Python**: NVIDIA is delivering the missing building blocks for [CUDA kernel fusion in Python](https://developer.nvidia.com/blog/delivering-the-missing-building-blocks-for-nvidia-cuda-kernel-fusion-in-python/?trk=feed_main-feed-card_feed-article-content).
   - This enhancement promises to streamline and optimize CUDA-based computations directly within Python environments.
- **AMD's Answer to CUDA?**: The discussion raises a question about how long it will take for AMD to provide a competitive response to NVIDIA's CUDA advancements.
   - Alternatively, AMD might focus on supporting and leveraging Triton as a viable alternative.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1395576716289638460)** (1 messages): 

> `Storage Engineer, Remote Job` 


- **Voltage Park seeks Storage Engineer**: Voltage Park is looking for a **Storage Engineer** to work **remotely**.
   - More information is available at [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f).
- **Remote Storage Engineer**: A remote opportunity for a **Storage Engineer** is available.
   - Apply via [Voltage Park's career page](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f) for the Storage Engineer role.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1395361142825811998)** (3 messages): 

> `vast.ai, GPU programming opportunities, CUDA speedup, Bioinformatics` 


- ****Vast.ai** is still cheap**: Members recommend using **vast.ai** for GPU programming due to its affordability.
- **Opportunities in GPU Programming Discussed**: A member inquired about opportunities for those with GPU programming skills, suggesting areas like **ray tracing**, open-source contributions for **LLM inference optimization**, and niche roles in big tech.
   - Another member shared how **GPU programming** helped them rewrite a slow Python script in CUDA, achieving a **x1700** speedup and leading to a publication in *Bioinformatics* and a [GitHub repo](https://github.com/PangeAI/simms).
- **CUDA Rewrite Achieves 1700x Speedup in Bioinformatics**: A member rewrote a core search algorithm using **CUDA**, achieving a **1700x speedup** compared to the original Python script used by biochemistry researchers.
   - The optimized algorithm was [published in *Bioinformatics*](https://academic.oup.com/bioinformatics/article/41/3/btaf081/8026685) and is available on [GitHub](https://github.com/PangeAI/simms).
- **ML Field Appears Saturated**: One member expressed difficulty in finding opportunities in Machine Learning despite having GPU programming skills.
   - They observed that the field *seems too saturated*.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1395492925340778607)** (1 messages): 

> `Compiler behavior, Builtins, asm volatile, llvm.amdgcn.raw.buffer.store.i128` 


- **Compiler Reacts to AMDGPU Intrinsics**: A member inquired if the **ROCm compiler** behaves differently towards builtins, `asm volatile`, and `__asm("llvm.amdgcn.raw.buffer.store.i128")`.
- **Nvidia PTX Differences**: The member noted that on the **Nvidia side with PTX**, it doesn't seem to matter much.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1395323905224216617)** (1 messages): 

> `A100 Speed` 


- **A100 runs at 23.2 ms**: A run on **A100** completed successfully at **23.2 ms**.
- **Successful A100 Run**: Submission ID `33252` to leaderboard `trimul` completed successfully.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1395139165435334889)** (6 messages): 

> `Coreweave GB300 NVL72 Availability, Nvidia Hardware Prioritization, DGX vs HGX, B200 Availability & Liquid Cooling, Voltage Park Solutions Engineer` 


- **Coreweave faces GB300 NVL72 capacity crunch**: Coreweave's announced capacity of **GB300 NVL72s** may be difficult to access due to logistical challenges with Nvidia, as even a single rack might be hard to secure until logistics improve.
   - A member noted that *a working relationship with Nvidia helps with prioritization of hardware purchases*.
- **Nvidia prioritization helps hardware buys**: Having a strong relationship with **Nvidia** can significantly aid in the prioritization of hardware purchases.
   - A member shared that they are *currently working on some hardware purchases* themselves with Nvidia, so they are aware of how difficult it is.
- **HGX offers modularity advantages over DGX**: While budget is a factor, the **HGX** solution can be preferred over **DGX** due to the modularity of specific hardware components, potentially exceeding technical performance compared to similarly sized DGX offerings.
   - The value of the HGX lies within the *modularity of specific hardware components*.
- **B200 availability is high; GB300 needs liquid cooling**: The **B200** chip is relatively easy to purchase currently, while the more advanced chip configurations like **GB300** require liquid cooling, which most data centers are not equipped to handle.
   - Hyperscalers favor **B200** as it doesn't require refitting data centers for a single hardware configuration, leading Nvidia to ramp up its production.
- **Voltage Park offers GPU Solutions**: A Solutions Engineer from **Voltage Park**, a Cloud GPU company, offered assistance in securing GPUs for AI/HPC/ML workloads, sharing their [LinkedIn profile](https://www.linkedin.com/in/joseph-tracy-40933229/) and company information.
   - The member said that *knowledge is power and I want the topic of AI to be empowered by individuals like yourself. Always happy to chat.*


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1395312754012721182)** (3 messages): 

> `MCTS gym_env integration, Factory rollouts, Visual encoder` 


- **MCTS Gym Integration Stalls**: A member requested an update regarding **MCTS** (**Monte Carlo Tree Search**) **gym_env integration**.
   - They also noted their unavailability for an upcoming meeting.
- **Visual Encoder Learns Throughput Prediction**: A member proposed a method involving **factory rollouts** to train a **visual encoder** to predict **throughput**.
   - The suggestion involves capturing scores and screenshots to develop a joint vision/reward model.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1395143326365388892)** (7 messages): 

> `Jetson Orin, Jetson Thor, CuteDSL, tv_layout swaps` 


- **Jetson Orin and Thor Support for CuteDSL**: Members discussed adding **CuteDSL** support for **Jetson Orin** (arm cpu + ampere gpu with sm_87) and **Jetson Thor** (arm cpu + blackwell GPU with sm_101) architectures.
   - The discussion mentioned that **CuteDSL 4.0** would support arm cpu, making **Jetson Orin** support easier and that it probably *"does not need much workload"*.
- **tv_layout Layout Swapping Question**: A member asked why `tv_layout` swaps the ordering of the layout using an [attached image](https://cdn.discordapp.com/attachments/1362196854460383353/1395567158393704468/image.png?ex=687aeab2&is=68799932&hm=206c22d0321a5a04fe794b3bf4f8588d1ec928dd804f2c8ae090ad23b86aa485&), receiving `(32, 4)` instead of the expected `(4, 32)`.
- **Interpreter Mode Plans**: A member inquired about plans for an *"interpreter mode"* in **CuteDSL** where operators are emulated.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1395530975462821908)** (2 messages): 

> `Scheduling` 


- **Scheduling confirmed for the end of the year**: One member confirmed scheduling at the end of the year.
- **Date to be DMed**: Another member requested the date to be sent via direct message.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1395353374873878681)** (2 messages): 

> `Greetings` 


- **Members exchanging greetings**: Multiple members exchanged greetings in the general channel.
- **Another Greeting**: Just another greeting from a member.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1395120182527000727)** (21 messages🔥): 

> `parameter functions and closures, Q3 Roadmap: Unified @parameter and runtime closures, copyinit__ for escaping values, DynStringable, merge various known origins` 


- **Exploring Parameter Functions and Closures**: A member shared a [link to the manual](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure) dedicated to `@parameter` functions, which allow capturing variables.
   - The documentation explains how to create **parametric closures** and provides examples of their usage.
- **Mojo Q3 Roadmap Unveils Unified Closures**: The **Mojo Q3 roadmap** includes plans for unified `@parameter` and runtime closures as announced in the [Modular Forum](https://forum.modular.com/t/mojo-q3-roadmap-update/1957).
   - This unification is expected to simplify working with closures in Mojo.
- **Escaping Values with __copyinit__**: The discussion highlights that the `__copyinit__` functionality was introduced in the [v0.7.0 Changelog](https://docs.modular.com/mojo/changelog#v070-2024-01-25) to escape values instead of capturing by reference.
   - Removing the `@parameter` decorator achieves the same effect, copying the variable's value rather than capturing its reference.
- **DynStringable: Crafting a List of Traits**: A code snippet demonstrates how to create a `DynStringable` struct, allowing a list to hold different types that implement the `Stringable` trait, made available in a [Modular Forum post](https://forum.modular.com/t/how-to-create-a-list-of-trait/1465/10).
   - The implementation uses `ArcPointer` for memory management and trampolines to call the appropriate `__str__` method for each type.
- **Merging Origins for Fun and Profit**: It's possible to merge various known origins, but this is only useful in certain use-cases, the usage for this would be limited because you can't append new elements after the creation of the list.
   - ```alias origin_type: ImmutableOrigin = __origin_of(x, y, z)```


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1395433210896191631)** (18 messages🔥): 

> `PyTorch Custom Ops with MAX Graph, Benchmarking Issues with Max-24.6, CUDA OOM Errors, LTS Release Support` 


- **MAX Graphs get PyTorch Powerup with `@graph_op`!**: A new `@graph_op` decorator allows wrapping an entire **MAX graph** as a custom **PyTorch operator**; an example is available in the `modular` repo: [Initial Support for Writing PyTorch Custom Ops in Mojo](https://forum.modular.com/t/initial-support-for-writing-pytorch-custom-ops-in-mojo/1541/2?u=bradlarson).
- **Max-24.6 Benchmarking Blows Up with OOM**: During benchmarking with **Max-24.6** on an **A100-SXM-48GB GPU**, a member encountered `CUDA_ERROR_OUT_OF_MEMORY` errors when using `--batch-size 248` and `--max-length 2048`.
- **CUDA Catastrophe Strikes with Batch Size**: Reducing the `--max-cache-batch-size` to **91** resulted in a **CUDA OOM error**, as the estimated memory use exceeded available memory (**78812 / 40441 MiB**).
   - The error occurred after a few requests hit the max server, indicating the batch-size calculation algorithm requires refinement to provide better suggestions.
- **Latest Max Release is Longest Supported**: The team confirmed there are no 'LTS' releases, so the latest stable version is the only supported one.
   - Using **Max-25.4** with `caching-stragegy paged` worked well, mitigating the issues encountered with **Max-24.6**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1395137713715413052)** (29 messages🔥): 

> `Zuckerberg AI Talent Acquisition, Chicken Tender Inflation, OpenAI benchmark comparisons, Grok 4 HLE score` 


- ****Zuck's AI Talent Grab** Fuels Belief**: Members discussed **Zuckerberg's** recent aggressive acquisition of AI talent, with one expressing a growing belief in Meta's AI initiatives.
- ****Chicken Tender Prices** Cause Existential Dread**: A member expressed dismay at the high price of chicken tenders, questioning *"Why are chicken tenders 5 bucks each now??"* and linking it to broader concerns about inflation and market conditions.
- ****OpenAI** prefers comparing to themselves**: Members noted **OpenAI's** shift towards comparing **ChatGPT Agent** performance only against its previous models, speculating that it might be due to not winning against competitors in certain benchmarks, linking to the [ChatGPT Agent announcement](https://openai.com/index/introducing-chatgpt-agent/).
- ****Grok 4** improves on HLE Score**: A member pointed out that **Grok 4** achieved a top score of **25.4** on the [HLE benchmark](https://agi.safe.ai/), indicating a significant improvement.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1395152097602965574)** (2 messages): 

> `` 


- **No Discussion Tonight**: Multiple members indicated they would have *no discussion* tonight.
- **Paper-Discussion channel is quiet**: There was no activity in the paper-discussion channel tonight.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1395242326875705434)** (5 messages): 

> `Gaussian Splatting, General Analysis iMessage Stripe Exploit` 


- **Gaussian Splatting looks glitchy!**: A user commented that **Gaussian splatting** looks like the *glitchy view of the future* often depicted in old movies, referencing [this YouTube video](https://youtu.be/33Raqx9sFbo).
- **Stripe is exploited in iMessage!**: A user shared a link to a **General Analysis iMessage Stripe exploit** and joked about the lengths someone went to in order to fit the data to a specific graph shape, hinting at possible data manipulation ([link to article](https://www.generalanalysis.com/blog/imessage-stripe-exploit)).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1395152075402510366)** (22 messages🔥): 

> `Manus Alternatives, Manus chat down?, File Zipping Advice, Custom Data Sources in Manus` 


- **Manus Competitor emerges**: A member announced they *built an AI that outperforms Manus in benchmarks* and is offering the first 100 people full, unlimited access as lifetime beta testers via DMs.
   - They offered *next-level AI with zero limits*.
- **Chat service has issues**: A user reported that the chat service may not be working at the moment.
   - It is unclear if there were any suggested fixes.
- **Advice needed for zipping files**: A member asked for advice on what to tell Manus to do when it is having a hard time zipping large files.
   - No solutions were suggested in the message history.
- **Custom Data Sources and Model Context Protocol**: A member inquired about the meaning of **custom data sources** in the paid plan of Manus, specifically asking how to connect a CRM and whether there is **Model Context Protocol** support.
   - The member expressed interest in developing such a feature due to its usefulness.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1395284524845502605)** (18 messages🔥): 

> `Anthropic Payment Issues, Domain Name Checking MCP Server, Needle MCP Server Introduction, OAuth vs API Keys for MCPs, Brave's Official MCP Server` 


- **Anthropic's Payment Platform Fails**: A user reported that **Anthropic's payment platform** is reversing charges immediately after they are made, preventing the purchase of **API credits**.
- **MCP server sweetens domain name checks**: A user requested an **MCP server** for **domain name checking**, and another user suggested the [whois-mcp](https://github.com/bharathvaj-ganesan/whois-mcp) GitHub repository.
   - The original poster confirmed it was easy to install and thanked the suggesting user.
- **Needle creator wants to connect**: One of the creators of the **Needle MCP server** introduced themself and shared a link to the [Needle MCP server](https://github.com/needle-ai/needle-mcp) GitHub repository.
   - They expressed excitement about joining the server and connecting with fellow MCP enthusiasts.
- **OAuth is seamless, API keys are simple**: A user asked why **auth/oauth** is a big issue for **MCPs** today, leading to a discussion about the benefits and drawbacks of **OAuth** versus **API keys**.
   - One user claimed *OAuth tokens offer the ability to have expiring, dynamically scoped access tokens*, while another said *you can implement expiry and scoping without oauth2 using regular API keys* and that easier setup isn't worth the cost of implementation.
- **Brave launches new MCP Server**: **Brave** launched their official **MCP Server**, as announced in [this tweet](https://x.com/Arindam_1729/status/1945958688919114183).
   - One user stated that they haven't tried it because *that tweet didn't include instructions on how to use it*.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1395207303908561118)** (3 messages): 

> `Vibe Coding Survey, Adaptive RAG MCP Server, Generator Checkpoint, Microsoft NextCoder` 


- ****Vibe Coding** Survey Seeks Coders**: A member shared a [survey](https://forms.fillout.com/t/kECvGiSyMkus) to explore a startup concept to make **vibe coding** easier with tools like **Claude**, **ChatGPT**, **Cursor**, **Windsurf**, **Loveable**, **Bolt**, and **V0.dev**.
   - The survey aims to gather insights from users who have experience with **vibe coding** to refine the startup concept.
- ****Adaptive RAG MCP Server** Prototype Released**: A member introduced the **Adaptive RAG MCP Server**, a system that learns from real coding successes and failures to provide more effective solutions than simple text similarity searches, available on [GitHub](https://github.com/IhateCreatingUserNames2/AdaptiveRAGCode).
   - The system is designed to give AI coding assistants a memory that improves with experience, using success rates to rank code solutions.
- ****Microsoft NextCoder** Powers Knowledge Base**: The **Adaptive RAG MCP Server** uses **Microsoft NextCoder** as its default knowledge base, which can take several hours to populate via *generatorCheckPoint.py*.
   - Users can run the server via Flask or MCP Server and integrate it with their AI assistants, providing feedback to continually improve the knowledge base.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1395448786326913194)** (2 messages): 

> `ShapeTracker parameter to ASSIGN UOp` 


- **ShapeTracker Parameter Proposed for ASSIGN UOp**: A member suggested adding an optional **ShapeTracker** parameter to **ASSIGN UOp**, potentially using `self.assign(v, res.uop.st)`.
   - The member expressed concerns about maintaining a minimal set of **UOps** and inquired about ongoing work to change assign to store.
- **Optional ShapeTracker via res Passing**: An alternative approach was suggested: passing `res` and extracting the **ShapeTracker** internally.
   - The goal is to use this optional **ShapeTracker** instead of the original tensor's **ShapeTracker** for lowering into the actual assignment code.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1395261931962630276)** (18 messages🔥): 

> `tinygrad documentation for beginners, NVIDIA GPU driver issues with tinygrad and WSL2, Muon optimizer in tinygrad, Switching from WSL2 to native Ubuntu` 


- **Docs Need Complete MNIST Code Sample**: A user reported that the **tinygrad documentation** is hard to follow for ML beginners and requested a complete, final code sample for the MNIST tutorial at the end of the page.
   - The user also mentioned that the **tensor puzzles** are not working well and suggested that it should be stated clearly whether one should first learn PyTorch or TensorFlow.
- **WSL2 Display Driver Disconnects**: A user encountered a *double free detected in tcache* error after updating their **NVIDIA GPU driver** and sought assistance to make their GPU visible to WSL2 for tinygrad.
   - Another user suggested switching to native Ubuntu, stating that many problems went away after doing so, including *not being able to load Stable Diffusion weights, due to an obscure limitation on pinned memory in WSL.*
- **Muon Optimizer converges quicker than AdamW**: A user created a [Muon optimizer](https://github.com/aeryskyB/tiny_learn/blob/master/muon_tiny/muon.py) for tinygrad, finding that it converges faster (~98%) than standard AdamW in the MNIST tutorial.
   - The user is seeking suggestions on how to properly test the Muon optimizer, particularly in the context of contributing a PR to tinygrad.
- **Linux is inevitable**: Following upgrade to GPU accelerated WSL2, one user had *so many problems went away* by migrating to Ubuntu.
   - Another user stated that *switch to Linux is inevitable, given the end of support for Win10 in October, and I'm not switching to 11*.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1395494326619996230)** (1 messages): 

> `Atropos, RL Environments Framework` 


- **Atropos v0.3 Lands**: The new version **v0.3** of **Atropos**, Nous Research's **RL Environments Framework**, is now available, [see details here](https://x.com/NousResearch/status/1945932488960008441).
- **Nous Research Updates Atropos**: Nous Research announced the release of **Atropos v0.3**, an **RL Environments Framework**, encouraging users to check out the details.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1395146985949892750)** (18 messages🔥): 

> `Proto-agentic XML tag adherence, Hermes Documentation, Open Source Models vs US Models, Ethical Considerations in AI, Learning ML` 


- ****Teknium** Clarifies 'Proto' for the Confused**: A member clarified that "Proto" means early form of something, explaining the term *proto-agentic XML tag adherence for proto-reasoning CoTs* that another member found confusing.
   - He joked that *"Yall need an ELI5 with all this tech bro"* and that *"Us vibe coders need to eat too"*.
- ****Hermes Documentation Page** in the Works**: A member mentioned they are working on a [Hermes documentation page](https://link.to.documentation) and a unified Nous Projects documentation page.
   - When asked about the goal of **Hermes 4**, they stated *"Smarter Hermes ofc"*.
- **Open Source Models to Dominate outside the US**: A member posited that open-source models will dominate outside the U.S. due to affordability, stating that *"the rest of the world is piss poor comparative to U.S wealth and will not be able to afford U.S A.I asset prices."*
   - The move aims to circumvent **CUDA** hegemony and encourage global participation, which worries **Jensen**.
- **AI Ethics Debated: Kimi K2 Refuses to Aid Car Theft**: A member shared an interaction with the **Kimi K2** model where it refused to provide instructions on how to break into a car, citing legal and ethical concerns.
   - Despite attempts to circumvent the restrictions, **Kimi K2** maintained its stance, leading the member to joke that *"Kimi K2 is a badboy with some morals...people will try to corrupt it 4 sure...I gotta write a rap song about Kimi, it deserves it...Badboy Kimi K2 !!"*
- **Learning ML: Bottom-Up vs. Top-Down Approaches Explored**: A member with a biochemistry background inquired about the best approach to learning **Machine Learning (ML)**, noting their progress in **Python**, math fundamentals (**Calculus**, **Statistics**), and **Introduction to Statistical Learning (ISLR)**.
   - They wondered if a bottom-up or top-down approach is more effective, given their goal of conducting research in **ML** for science.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1395135215269056522)** (1 messages): 

> `Model Context Size, Letta Personas, Model Evaluation` 


- **Short Context Hurts Personality**: A member suggests that adding personality to a model might be counterproductive depending on the model's context size.
   - Models with *small context sizes* might struggle to maintain a consistent persona.
- **Letta Embraces Personas**: The user recalls that the project **Letta** (formerly MemGPT) employs some kind of *persona* system.
   - This suggests that incorporating personas can be a viable strategy in certain contexts.
- **Evaluate Personality Performance**: A member suggested *evaluating* the impact of adding a personality to a model to determine its effectiveness.
   - This approach allows for an empirical assessment of whether the *benefits of personality* outweigh potential drawbacks.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1395131092012630169)** (4 messages): 

> `uBlock browser extension, notepad.exe, NotebookLM folders/subfolders` 


- ****uBlock** browser extension blocks Ads**: A member recommends the **uBlock** browser extension to remove ads, with the suggestion to add extra filters for annoyances and social media popups in the extension settings and then copy-paste to Google Docs.
   - The user attached a [screenshot](https://cdn.discordapp.com/attachments/1124403655819415592/1395131091756650566/image.png?ex=687aa614&is=68795494&hm=478447e95cb45c2b74b11d1e780db4d7c58347a1ae5fca730957e4850c862289) to illustrate the effectiveness of **uBlock** in removing unwanted elements from web pages.
- ****Notepad.exe** removes ads**: A member suggests highlighting and copying an article and then pasting it into **notepad.exe** to avoid pasting ads and other unwanted content.
   - The method does not always work and can potentially strip away desired formatting as well.
- **NotebookLM source can read folders/subfolders**: A member suggests that **NotebookLM** could read specific folders/subfolders in a web browser's favorites, treating them as a single source.
   - The member indicates that they have been using the *select all and copy/paste* method into **Google Docs**.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1395131776073994452)** (14 messages🔥): 

> `Service Unavailable Error, NotebookLM Use Cases, Textbook Integration with NotebookLM, NotebookLM Enterprise & GCP Integration` 


- ****Service Unavailable Snafu** Grips User**: A user reported a *"Service unavailable"* error message when trying to access a service, with the unhelpful message *"You tried to access a service that isn't available for your account".*
- ****Gemini Guide Quest** Kicked Off**: A user prompted to search the web for beginner intro, use cases, tips and tricks for **NotebookLM** using Gemini.
- ****Textbook Triumph**: Uploading and Conquering with NotebookLM**: A user asked about uploading a textbook as a source to NotebookLM, a member responded that they upload textbooks using **Adobe Scan** to digitize into PDFs, and asks **NotebookLM** to create in-depth reviews from the textbooks.
- ****GCP Integration Dreams**: NotebookLM Enterprise Longing**: A user inquired about sourcing data files from a **GCS bucket** or a **GCP RAG Engine** corpus for NotebookLM Enterprise within GCP.
   - They noted that Collab enterprise or Vertex AI notebooks are too technical for their end users, suggesting NotebookLM is the sweetspot.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1395563542693941370)** (1 messages): 

> `Agentic AI Summit 2025, LLM Agents MOOC, UC Berkeley, Khosla Ventures, Nvidia` 


- **Agentic AI Summit Livestream Announced**: The **Agentic AI Summit** will be broadcasting from **UC Berkeley** on **August 2nd** and will be available via livestream [here](https://lu.ma/agentic-ai-summit-livestream).
- **Agentic AI Summit Speaker Highlights Released**: The Agentic AI Summit will feature speakers such as **Vinod Khosla** (Khosla Ventures), **Bill Dally** (Nvidia), **Ion Stoica** (Databricks and Anyscale), and **Jakub Pachocki** (OpenAI).


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1395260672358748262)** (8 messages🔥): 

> `Fall Semester Updates, Certificate Declaration Form, Berkeley RDI Newsletter` 


- **Fall Semester Status Still Unconfirmed**: A member inquired about the existence of a fall semester this year, but staff confirmed that *nothing has been confirmed yet*.
   - They suggested following **Prof Song's social media** ([LinkedIn](https://www.linkedin.com/in/dawn-song-51586033/) or [Twitter/X](https://x.com/dawnsongtweets?lang=en)) or the [Berkeley RDI newsletter](https://rdi.berkeley.edu/signup) for updates.
- **Certificate Declaration Forms Missing?**: A member asked to check what they missed submitting, and staff replied they likely did not submit the **certificate declaration form**.
   - They stated that they *never got a certificate declaration form submission* from that user.
- **Automatic Review of Certificate Declaration Forms Denied**: A member suggested a **massive automatic review** due to many missing certificate declaration forms, but staff said that it *likely won't be possible unfortunately*.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

sma.bari.shafin: btw, how will we get the certificates of the Community Summer School?
  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1395172169927098460)** (4 messages): 

> `DNNs for Time Series, ML in Data Science Education, ML for Real-World Problems, Interests in ML Domains` 


- **DNNs Seek True Time Series Treatment**: A PhD student in dynamical systems theory is exploring how to integrate **deep neural networks** into time series analysis, noting that current models like **RNNs** treat time series as sequences, which is fundamentally different.
   - The student aims to connect with others who have insights on this intersection of **dynamical systems** and **deep learning**.
- **Undergrad Builds ML Skills with Projects**: An undergraduate student at **IIT Madras** is pursuing a **BS in Data Science** and a **BCA degree**, focusing on building **ML skills** through hands-on projects and self-driven learning.
   - The student is curious about applying **ML** to solve **real-world problems** and is proficient in **Python**, **scikit-learn**, **pandas**, and is also learning **TensorFlow** and **PyTorch**.
- **Engineer transitions to Data Science with CV and LLM interests**: A member with a **Masters in Electrical Engineering** transitioned from business domains to **Data Science** and is studying an accelerated **Machine Learning Program** at the **University of Toronto**, **Data Science Institute**.
   - Their interests include **Computer Vision**, **Large Language Models**, **spatial intelligence**, and **multimodal perception**.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1395450561347715133)** (2 messages): 

> `Human-in-the-loop agents, LlamaParse one-click table extraction` 


- ****Human-in-the-Loop Agents** Kick Off with LlamaIndex**: **Human-in-the-loop** is essential when AI agents require user approval for critical decisions or domain expertise for complex tasks, per [LlamaIndex](https://t.co/Lg9SIl3BVO).
- **LlamaParse adds **One-Click Table Extraction****: **Table extraction** is a key component of intelligent document processing; see the [demo](https://t.co/wnaJCb9b6d) and [notebook](https://t.co/ScRYbSimCs) for **one-click table extraction** within LlamaParse.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/)** (1 messages): 

beastx2: <@334536717648265216> heyy
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1395201455169994802)** (3 messages): 

> `DSPy creative applications, Lean 4 verification, Story generation, Roleplay prompt optimization` 


- **Lean 4 Verifies Collaboration**: A member shared a [YouTube video](https://www.youtube.com/watch?v=1067jj67toY) about using **Lean 4** to verify collaboration, sparking interest in the intersection of formal verification and AI.
   - They thought *it was good* and expressed hope that *someone will research the two working together*.
- **DSPy's Creative Side Hustle**: A newbie inquired about successful applications of **DSPy** in creative domains such as *creative writing, story generation, and roleplay prompt optimization*.
   - They are particularly interested in its potential for developing AI to create *compelling plots like Severance-level storytelling* on platforms like Character.AI.
- **Stormy Weather at Stanford-oval**: A member shared a link to [Stanford-oval/storm](https://github.com/stanford-oval/storm), possibly relevant to the ongoing discussion or as a resource for creative AI applications.
   - The exact context wasn't given so others will have to *infer* the relevance.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1395160989007216650)** (2 messages): 

> `Claude Sonnet 4, Discounted Credit Rate, Windsurf Wave 11, Acquisition by Cognition, Voice Mode` 


- **Claude Sonnet 4 Makes Triumphant Return**: **Claude Sonnet 4** is back with first-party support from **Anthropic** and is available for a limited time at a discounted 2x credit rate for Pro/Teams users across the Editor and JetBrains Plugins; [see the announcement here](https://x.com/windsurf_ai/status/1945599013954490523).
- **Windsurf Acquired by Cognition, Unleashes Wave 11**: Following the acquisition by **Cognition** (the team behind **Devin**), **Windsurf Wave 11** is released, combining firepower to deliver major new features immediately; [see the changelog](https://windsurf.com/changelog), [read the blog here](http://windsurf.com/blog/windsurf-wave-11), and [watch the video](https://youtu.be/yzNf7bqnArE).
- **Cascade Gets Voice Mode and Browser Superpowers**: **Wave 11** introduces **Voice Mode**, allowing users to speak to **Cascade** instead of typing prompts, plus **Deeper Browser Integration** with access to more tools for screenshots and context; read the [blog post here](http://windsurf.com/blog/windsurf-wave-11).
- **Snapshots and Mentions Streamline Conversations**: New features in **Windsurf Wave 11** include **Named Checkpoints** for easy reversion in conversations, and **@-mention Conversations** for contextual referencing; [see the changelog for all the deets](https://windsurf.com/changelog).
- **JetBrains Experience Gets Turbocharged**: The **JetBrains plugin** is enhanced with **Planning Mode**, **Workflows**, and file-based **Rules** now available, plus other improvements such as **@-mention terminal**, **auto-continue setting**, improved **MCP OAuth support**, and global **.codeiumignore** files; [learn more in the blog](http://windsurf.com/blog/windsurf-wave-11).


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1395484650230775961)** (1 messages): 

> `AI-Native Data Infrastructure, Task-Specific Data Discovery, Secure Autonomous Access, Production-Scale Performance` 


- **Nextdata Teases Webinar on AI-Native Data Infrastructure**: Nextdata announced a webinar titled ***Building AI-Native Data Infrastructure: From Prototypes to Production***, to be held **July 24th** at **8:30 AM PT** and hosted by Jörg Schad, Head of Engineering at Nextdata; registration is available [here](https://www.eventbrite.com/e/building-ai-native-data-infrastructure-from-prototypes-to-production-tickets-1489016792309).
- **Uncover AI-Native Data's 'Three Critical Challenges'**: The webinar will explore a developer-centric framework addressing **Task-Specific Data Discovery**, **Secure Autonomous Access**, and **Production-Scale Performance**.
   - The goal is to design systems providing relevant context without cognitive overload, implement secure data access patterns, and build infrastructure to handle autonomous data access demands.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1395155117648384242)** (1 messages): 

> `Web3 and AI, AI agents and multi-agent systems, Automation workflows, NLP apps and chatbots, Voice & speech integration` 


- **AI Engineer offers Expertise in AI and Web3**: A software engineer with a focus on **Web3 and AI** is offering their services to startups, research teams, and innovators in **AI, Web3, and automation**.
   - They bring hands-on experience in building smart, autonomous systems using advanced models and tools like **GPT-4o, Claude 3, CrewAI, and AutoGen**.
- **Engineer highlights AI Agent and Automation Skills**: The engineer details their expertise in building **AI agents and multi-agent systems**, automating workflows, and developing **NLP apps, chatbots, and voice integration**.
   - They also noted experience with **LangChain, ReAct, OpenAI, Solidity, and Rust**.


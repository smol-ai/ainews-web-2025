---
id: MjAyNS0w
title: "Execuhires Round 2: Scale-Meta, Lamini-AMD, and Instacart-OpenAI"
date: '2025-06-11T05:44:39.731046Z'
---

**Good execs are all you need.**

> AI News for 6/10/2025-6/11/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (218 channels, and 6238 messages) for you. Estimated reading time saved (at 200wpm): 502 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

As the summer blockbuster season arrives, it seems to be a Season of Sequels in AI - just after [Reasoning Price War 2](https://news.smol.ai/issues/25-06-10-o3-cut), we are now seeing a second wave of what we called "[Execuhires](https://news.smol.ai/issues/24-08-02-ainews-execuhires-tempting-the-wrath-of-khan)" last year:

- **Scale AI's Alexandr Wang hired to lead the new "Superintelligence" division at Meta**, as part of a [$15b purchase of a 49% stake in Scale](https://www.theverge.com/news/684322/meta-scale-ai-15-billion-investment-zuckerberg)
- **Lamini's [Sharon Zhou hired as VP of AI at AMD under Lisa Su](https://x.com/realSharonZhou/status/1932817096510931380)**, with "several intense, cute" execs also joining, leaving the question of what happens to Lamini entirely unanswered
- (older) [Instacart's CEO Fidji Simo hired as CEO of Apps at OpenAI under Sama](https://finance.yahoo.com/news/fishing-family-big-tech-french-030253230.html?guccounter=1).

This all comes under the context of [Meta offering >10m/yr comp pacakges](https://x.com/deedydas/status/1932828204575961477) to top researchers, successfully nabbing [Jack Rae from Gemini.](https://x.com/shiringhaffary/status/1932852606851789278)

---

# AI Twitter Recap

**OpenAI Model Updates & Pricing**

- **o3-pro Release and Performance**: **OpenAI** made significant announcements regarding **o3-pro**, confirming its release to all **ChatGPT Pro users** and in the **API** as a "significantly better" model than **o3** across various evaluations. [@OpenAI](https://twitter.com/OpenAI/status/1932586531560304960) and [@markchen90](https://twitter.com/markchen90/status/1932570548740964438) highlighted the launch, with [@kevinweil](https://twitter.com/kevinweil/status/1932565467736027597) noting a **doubling of rate limits for o3 for ChatGPT Plus users**. Users and evaluators quickly tested its capabilities, with [@gdb](https://twitter.com/gdb/status/1932561536268329463) stating **o3-pro** is "much stronger than o3." It achieved new records on the **Extended NYT Connections** benchmark, surpassing **o1-pro** (87.3 from 82.5), and became the **#1 model on SnakeBench** according to [@GregKamradt](https://twitter.com/GregKamradt/status/1932898036466004317) and [@LechMazur](https://twitter.com/LechMazur/status/1932656485341032719). [@WesRothMoney](https://twitter.com/WesRothMoney/status/1932679839682867296) reported it **one-shotted the Tower of Hanoi 10 disk problem**. While [@paul_cal](https://twitter.com/paul_cal/status/1932745565021868063) noted **o3-pro** can be **3x slower than o1-pro**, its reasoning capabilities were praised. [@imjaredz](https://twitter.com/imjaredz/status/1932657322204987718) claimed it "feels **MILES ahead of Claude Opus 4** for non-code tasks," and [@cto_junior](https://twitter.com/cto_junior/status/1932989802657497206) found it solves **hard multithreading problems** correctly. [@johnowhitaker](https://twitter.com/johnowhitaker/status/1932821323979632783) shared a demonstration where **o3-pro** correctly answered a complex idle question that **o3** failed. [@polynoamial](https://twitter.com/polynoamial/status/1932600979113005300) from **OpenAI** stated they are hiring to push forward the "intelligence frontier" with models like **o3**.
- **Pricing and Accessibility**: **OpenAI** implemented a "massive price cut," making **o3** **80% cheaper** and notably **cheaper than GPT-4o**, as reported by [@scaling01](https://twitter.com/scaling01/status/1932596796347252937). [@apples_jimmy](https://twitter.com/scaling01/status/1932549807304020227) noted it is **20% cheaper than 4o**. This price reduction is seen as a move that "might actually be forcing **Google and Anthropic to lower prices**," according to [@scaling01](https://twitter.com/scaling01/status/1932566284270538966). The price drop led to **o3** becoming a viable daily-driver model, reflected in its integration into **Cursor** [@kevinweil](https://twitter.com/kevinweil/status/1932559521588617415).
- **Fine-tuning and Model Consistency**: **OpenAI Devs** announced that users can now **fine-tune the GPT-4.1 family** (4.1, 4.1-mini, 4.1-nano) using **direct preference optimization (DPO)**, ideal for subjective tasks where tone, style, and creativity matter [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1932858051876565475). Despite initial speculation about model distillation or changes post-price drop, [@GregKamradt](https://twitter.com/jeremyphoward/status/1932869428083110007) and [@scaling01](https://twitter.com/scaling01/status/1932839048273670563) confirmed through retesting that **o3** remained the **same model**, not distilled.
- **Model Evolution and Evaluation**: [@BorisMPower](https://twitter.com/BorisMPower/status/1932556016455201145) highlighted the "incredible trajectory of performance improvements for the reasoning models since the original **o1-preview**," noting **60%+ winrates** as significant. [@HamelHusain](https://twitter.com/HamelHusain/status/1932642264163180844) commented on the timing of evaluations relative to **OpenAI's price drop**.

**Other Model Releases & Advanced AI Research**

- **Mistral AI Developments**: **Mistral AI** officially announced **Magistral**, their first reasoning model designed for domain-specific, transparent, and multilingual reasoning [@MistralAI](https://twitter.com/algo_diver/status/1932560648099278930). **Magistral 4-bit DWQ** is now available on Hugging Face for use with **mlx-lm** or **LM Studio** [@awnihannun](https://twitter.com/awnihannun/status/1932547785162961291). [@Teknium1](https://twitter.com/Teknium1/status/1932580993132790232) praised **Mistral's paper** as the "best practical paper on doing reasoning RL since **DeepSeek R1**." [@ClementDelangue](https://twitter.com/ClementDelangue/status/1932572636397113579) noted the new **Mistral** model features **24B parameters**, is based on **Mistral Small 3.1**, is **multilingual**, and has a **128K context length (40K effective)**, under an **Apache 2.0 license**.
- **Meta AI's V-JEPA 2 and World Models**: **Meta AI** released **V-JEPA 2**, a **1.2 billion-parameter model** trained on video, aimed at advancing physical AI by enabling **zero-shot planning in robots** within unfamiliar environments. They also introduced **three new benchmarks** for evaluating physical world reasoning from video [@AIatMeta](https://twitter.com/AIatMeta/status/1932808881627148450) and [@AIatMeta](https://twitter.com/AIatMeta/status/1932923002276229390). [@ylecun](https://twitter.com/ylecun/status/1932845440547840234) highlighted **V-JEPA 2**'s importance.
- **Gemini and Google's AI Capabilities**: **Gemini 2.5 Pro (06-05)** is rapidly climbing public leaderboards, notably becoming the **#1 model at 192K tokens on Live Fiction**, the **strongest over Document Processing model in IDP**, and the **best cost-performance on Aider** according to [@_philschmid](https://twitter.com/_philschmid/status/1932723220379049999). It also reportedly **solved all JEE Advanced 2025 problems** from the Mathematics Section [@dilipkay](https://twitter.com/dilipkay/status/1932754214469402630). **Google Veo 3** is demonstrating impressive capabilities in consistent character and mood generation for video [@demishassabis](https://twitter.com/demishassabis/status/1932608957945950407) and [@DrMachakil](https://twitter.com/demishassabis/status/1932856733397102914). **Google** also released **Gemma 3n** for desktop and IoT, powered by their new **LiteRT-LM library** [@osanseviero](https://twitter.com/demishassabis/status/1932607299178148184).
- **New Models and Research Techniques**:
    - **Higgsfield** introduced **Higgsfield Speak**, enabling faces in images (cars, zombies, even coffee) to speak [@_akhaliq](https://twitter.com/_akhaliq/status/1932545372817154530). They also integrated **Flux.1 Kontext** for enhanced content creation [@_akhaliq](https://twitter.com/home/status/1932903530173747261).
    - **Cartesia AI** launched **Ink-Whisper**, a new family of streaming **speech-to-text (STT) models** designed to be fast and affordable for voice agents [@krandiash](https://twitter.com/krandiash/status/1932601554298941812).
    - **Sakana AI Labs** introduced **Text-to-LoRA**, a hypernetwork that generates **task-specific LLM adapters (LoRAs)** from a text description of the task, offering significant computational and technical barrier reduction for specializing foundation models [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1932972420522230214).
    - Research suggests that **hybrid models** can maintain reasoning performance with fewer attention layers, offering efficiency benefits for long reasoning traces [@_albertgu](https://twitter.com/_albertgu/status/1932844922241233019).
    - A new study, **IneqMath**, reveals that **LLMs often struggle with rigorous math proofs**, even when giving correct answers [@HuggingPapers](https://twitter.com/_akhaliq/status/1932894338616574091).
    - **FutureHouseSF** is working on **ether0**, a **24B model** that can reason in English and respond with molecules [@cgeorgiaw](https://twitter.com/_lewtun/status/1932875317678317817).
    - **Yandex** released **Yambda**, a massive **public dataset of nearly 5 billion anonymized user-track interactions** for recommender systems [@_akhaliq](https://twitter.com/_akhaliq/status/1932872791768117483).
    - Research by [@nickhjiang](https://twitter.com/TimDarcet/status/1932707025718247935) found that **Vision transformers** have **high-norm outliers** that hurt performance and distort attention.
- **Model Limitations and Future Direction**: **The Turing Post** summarized key insights from **Apple's "Illusion of Thinking"** and **FAIR/Google DeepMind/Cornell/NVIDIA's "How much do language models memorize?"** papers, highlighting how finite architectures simplify, guess, or shut down when pushed past their fundamental capacity, even while appearing fluent [@TheTuringPost](https://twitter.com/TheTuringPost/status/1932912650444550470). This suggests that "scaling may be hitting a wall in the digital world" but is "only beginning in the biological world" [@ysu_nlp](https://twitter.com/wightmanr/status/1932858386368090407).

**AI Agents & Development Frameworks**

- **DSPy Adoption and Philosophy**: The **DSPy** framework gained significant traction, with [@lateinteraction](https://twitter.com/lateinteraction/status/1932551576100667416) emphasizing its "foresight" that "prompts are compiled outputs," not source code. This philosophy is seen as becoming central to AI engineering. [@kmad](https://twitter.com/lateinteraction/status/1932633959609102596) and [@MaximeRivest](https://twitter.com/lateinteraction/status/1932810387285815395) noted that **DSPy** is "starting to click with a lot of people." [@vineettiruvadi](https://twitter.com/lateinteraction/status/1932810369262887015) successfully used **DSPyOSS** to create a synthetic clinical note generator in **20 minutes**.
- **Model Context Protocol (MCP) and Agent Ecosystem**: The **Hugging Face (HF) MCP server** was launched, allowing agents to find models, datasets, papers, or apps within the HF ecosystem [@julien_c](https://twitter.com/clefourrier/status/1932690632394293539) and [@ClementDelangue](https://twitter.com/ClementDelangue/status/1932823573762355562). This fosters an "open source collection of **MCP servers**" [@abidlabs](https://twitter.com/ClementDelangue/status/1932885527000461686). Integrations like **GPT Researcher** leveraging **LangChain's MCP adapters** for intelligent tool selection were highlighted [@LangChainAI](https://twitter.com/Hacubu/status/1932677682556187103).
- **New Agentic Frameworks and Tools**:
    - **Databricks** launched **Agent Bricks**, a new approach to building auto-optimized agents that uniquely takes a "declarative and compositional" view, allowing natural language feedback to steer agents [@matei_zaharia](https://twitter.com/lateinteraction/status/1932849147973153017).
    - **LangChain AI** released updates to **langchain-google-vertexai**, including **500x faster prediction client caching** [@LangChainAI](https://twitter.com/LangChainAI/status/1932848293165371448). They also showcased **Harvey AI's** use of **LangSmith evaluations** and lawyer-in-the-loop methodology for building legal AI agents [@LangChainAI](https://twitter.com/LangChainAI/status/1932858287265099900). **LangGraph** was cited as a "massive unlock" for building **AI Hedge Fund Teams** and enabling multi-agent systems like **JARVIS** by **Outshift by Cisco** [@virattt](https://twitter.com/Hacubu/status/1932885769305403548) and [@LangChainAI](https://twitter.com/hwchase17/status/1932872002978902374).
    - **LlamaIndex** announced integration with **CleanlabAI** for building AI knowledge assistants and production agents, and they are working on incremental workflows for use cases like updating data after new email arrivals [@llama_index](https://twitter.com/jerryjliu0/status/1932838464233615814). **LlamaExtract**, an agentic document extraction service, was launched, providing precise reasoning and citations for extracted data [@jerryjliu0](https://twitter.com/jerryjliu0/status/1932852719292985359).
    - **Hugging Face** announced **AISheets**, allowing thousands of AI models to interact with spreadsheets for building, analyzing, and transforming data [@Thom_Wolf](https://twitter.com/ClementDelangue/status/1932573508703232368).
    - **Smolagents** surpassed **20,000 stars on Github**, a milestone for the agent library [@AymericRoucher](https://twitter.com/AymericRoucher/status/1932836600695722326).
    - **Fire Enrich**, an **open-source Clay alternative** for data enrichment using AI agents, was open-sourced [@firecrawl_dev](https://twitter.com/hwchase17/status/1932884877273411764).
- **Agent Design and Use Cases**: Discussions at the **AI Engineer World’s Fair** covered trends like "ambient" agents and the role of **MCP** [@TheTuringPost](https://twitter.com/TheTuringPost/status/1932550904453976567). [@dzhng](https://twitter.com/dzhng/status/1932861054146785547) emphasized that when developing agent tools, the interface must be designed to be **easy for the AI to understand**, often requiring more than just exposing underlying APIs. **Codex** (the CLI version) proved to be an "amazing productivity boost" for **debugging stacktraces** [@cto_junior](https://twitter.com/cto_junior/status/1932802313665851901).
- **Evaluation of Multi-Agent Architectures**: **LangChain AI** released initial benchmarks on orchestrating across multiple agents, including improvements to their supervisor approach [@LangChainAI](https://twitter.com/LangChainAI/status/1932825652312600810).

**AI Business, Infrastructure & Deployment**

- **Strategic Partnerships & Investment**: **OpenAI** reportedly tapped **Google** for a new cloud deal to secure more compute [@scaling01](https://twitter.com/scaling01/status/1932716057631846860). **Meta's** investment in **Scale AI**, where **Alex Wang** is involved in a 'superintelligence' lab, sparked discussion about Meta's AI strategy and its impact on RL post-training [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1932588243897270454). **xAI** partnered with **Polymarket** to combine market predictions with **X data** and **Grok's analysis** for a "Hardcore truth engine" [@xai](https://twitter.com/Yuhu_ai_/status/1932753086885540061). A new collaboration between **NVIDIA** and **Hugging Face** was announced to connect AI researchers with GPU clusters via **Training Cluster** [@jeffboudier](https://twitter.com/_lewtun/status/1932771189396492494).
- **AI Infrastructure & Compute**: **Mistral AI** unveiled **Mistral Compute**, described as an "unprecedented AI infrastructure undertaking in Europe" and a strategic initiative to secure compute [@MistralAI](https://twitter.com/qtnx_/status/1932799532070547810). **Modular** and **TensorWave** are offering free compute through their partnership [@clattner_llvm](https://twitter.com/clattner_llvm/status/1932614831364006363). **TogetherCompute's API** was highlighted for having the **fastest DeepSeek v3 endpoint** (2x faster than the next best) and a new **Batch API** for high-throughput use cases like synthetic data generation, benchmarking, and document extraction, offering **50% lower pricing** than their interactive API [@vipulved](https://twitter.com/vipulved/status/1932601075754020876).
- **Optimization & Efficiency**: Significant speedups in daily maintenance jobs (e.g., **30x speedup** for paging through **S3/GCS prefixes**) were reported using simple hacks like **read-through caching** [@turbopuffer](https://twitter.com/turbopuffer/status/1932916345571848610). **UnslothAI** announced they provide **2x faster reward model serving and sequence classification inference**, and were featured on the **Nasdaq tower** as one of the top **100 most impactful and fastest-growing infra companies** [@danielhanchen](https://twitter.com/danielhanchen/status/1932801493649793469). The ease of use for **Docker run --gpus** on **AMD** was also noted as key for accessibility [@AnushElangovan](https://twitter.com/dylan522p/status/1932829981316452404).
- **Product Launches & Company News**:
    - **Mechanize** announced their founding with the explicit goal to **"automate all work"** by building virtual work environments, benchmarks, and training data [@tamaybes](https://twitter.com/tamaybes/status/1932841955542904919).
    - **Runway** hinted at "very exciting updates and new products" that will bring a "completely new experience" to the platform, making creation "natural and easy" [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1932600586123227219). Their **upscaling is now available in the API** [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1932835921071878384).
    - [**You.com**](http://you.com/) introduced "Projects" for organizing research into folders, allowing contextualization and structuring of information [@youdotcom](https://twitter.com/RichardSocher/status/1932843347632402905).
    - **Dia**, a new browser that "deeply understands you" for personalized web interaction, was announced [@hursh](https://twitter.com/mathemagic1an/status/1932864995668508945).
    - **Perplexity AI** was featured by **Jensen Huang at GTC Paris** [@AravSrinivas](https://twitter.com/AravSrinivas/status/1932968936938537223).
    - **Databricks** released a **free edition** and opened up training materials to help developers learn data and AI [@matei_zaharia](https://twitter.com/lateinteraction/status/1932834408727543823).
    - **MLflow 3.0** was released after months of community collaboration [@MLflow](https://twitter.com/lateinteraction/status/1932871442347274500).
    - **Wayve AI** was recognized for its "incredible trajectory" [@nathanbenaich](https://twitter.com/NandoDF/status/1932552110005989462).
- **Training & Deployment Best Practices**: A new short course, "Orchestrating Workflows for GenAI Applications," was launched in partnership with [**Astronomer.io**](http://astronomer.io/) to teach developers how to turn GenAI prototypes into production-ready workflows using **Apache Airflow 3.0** [@AndrewYNg](https://twitter.com/AndrewYNg/status/1932822251273093247). The importance of **evaluations** for reliable LLM-powered apps was repeatedly emphasized, with a course on **"Application-Centric AI Evals for Engineers and Technical PMs"** by [@HamelHusain](https://twitter.com/HamelHusain/status/1932657675294421061). **BorisMPower** advised startups to prototype many features, evaluate failures, and focus on user-loved features, revisiting others with new model releases [@BorisMPower](https://twitter.com/BorisMPower/status/1932813365199712277).

**Societal & Geopolitical AI Implications**

- **AI's Impact on Work and Society**: **Sam Altman's** statements like "We do not know how far beyond human-level intelligence we can go, but we are about to find out" and "Intelligence too cheap to meter is well within grasp" highlight the rapid progression towards **AGI** and its potential to profoundly change society [@scaling01](https://twitter.com/scaling01/status/1932550566036804087) and [@scaling01](https://twitter.com/scaling01/status/1932551669134377357). **Mechanize's** explicit goal to "automate all work" stirred discussion about job displacement, with some calling their approach "un-empathetic" [@tamaybes](https://twitter.com/tamaybes/status/1932841955542904919). **Yoshua Bengio** discussed the "accelerating pace of advances in AI capabilities and the emergence of deceptive and self-preserving behaviours in frontier models," inspiring his **LawZero** initiative [@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1932859177283801470).
- **Geopolitics and AI Race**: The UK's AI and Biosciences industry was critically assessed by **ChatGPT**, highlighting marginal funding (£8 million vs. >$100M for US firms), talent flow issues due to US-enforced garden leave, foreign ownership of UK innovation, and lack of industrial anchors [@NandoDF](https://twitter.com/NandoDF/status/1932549812785754524). Mentions of **Mistral** being hit by **export restrictions** and **China's** demands regarding **ASML lithography machines** underscore the ongoing **tech war** and compute access issues [@dylan522p](https://twitter.com/dylan522p/status/1932563462963507589) and [@teortaxesTex](https://twitter.com/teortaxesTex/status/1932955304188076081).
- **Ethical and Societal Challenges**: Concerns were raised about **AI calorie counting apps** being conceptually flawed, suggesting they primarily serve as a "ritual" for mindfulness rather than accurate tech [@random_walker](https://twitter.com/random_walker/status/1932763118498685276). The potential for **GenAI to remove actionable information** about individuals from non-real-life interactions, leading to reliance on "rings of trust" and reduced opportunities for "out-group people," was discussed [@francoisfleuret](https://twitter.com/francoisfleuret/status/1932683908715282670). The current information environment was described as "the worst" [@tszzl](https://twitter.com/aidan_mclau/status/1932833046572773797).
- **Gaza/Palestine Conflict**: Numerous tweets from a single account [@SerranoAcademy](https://twitter.com/SerranoAcademy/status/1932664601373446633) amplified content related to the Gaza conflict, sharing reports from [@GozukaraFurkan](https://twitter.com/SerranoAcademy/status/1932664601373446633) on international ignorance, [@Kahlissee](https://twitter.com/SerranoAcademy/status/1932664629987045637) on **Greta Thunberg's** stance, images of child casualties from [@ShaykhSulaiman](https://twitter.com/SerranoAcademy/status/1932664650501664809), and calls for freedom for **Rima Hassan** from [@JimmyJ4thewin](https://twitter.com/SerranoAcademy/status/1932664208958652630). Protests in Brussels and Manchester in support of Gaza were also shared.
- **Philosophical and Cultural Observations**: Discussions ranged from the notion that "learning a model is akin to learning a programming language" [@deepgramscott](https://twitter.com/deepgramscott/status/1932596824126468198), to the decline of birthrates and a post-AGI society (not in tweets, but inferable from Sam Altman's general themes), to observations on the "dysfunctional society" and extreme wealth disparity in the **Bay Area** [@claud_fuen](https://twitter.com/claud_fuen/status/1932773334959174041). **AI's role in empowerment** ("we can build anything") was also highlighted [@shaneguML](https://twitter.com/shaneguML/status/1932979593520275536).

**Humor/Memes**

- **Relatable Developer/AI Engineer Humor**:
    - "Backup compute for the bunker" [@zacharynado](https://twitter.com/zacharynado/status/1932554928716660767)
    - "sorry i didn't reply to your email i was reward engineering again" [@vikhyatk](https://twitter.com/vikhyatk/status/1932675987235614835)
    - "You know you've been coding too long when you keep adding semicolons at the end of every line you type; Such as this tweet; Dammit;" [@claud_fuen](https://twitter.com/claud_fuen/status/1932779123736195451)
    - "joined a call and it's just me and a dozen AIs" [@thedanigrant](https://twitter.com/zacharynado/status/1932880607941644581)
    - "every time i open weights and biases" [@vikhyatk](https://twitter.com/vikhyatk/status/1932962492696965626)
    - "prompt soup for dinner tonight" [@lateinteraction](https://twitter.com/lateinteraction/status/1932997629161685213)
- **AI/Tech Industry Satire**:
    - **Perplexity AI** was parodied with a "Defying Google" song, an ode to its search capabilities [@AravSrinivas](https://twitter.com/AravSrinivas/status/1932693461355970872).
    - "people are hating on openAI for naming their models weirdly but imagine they had some tasteful marketing person and these things would be named 'Gazelle' or 'Appalachia'" [@fabianstelzer](https://twitter.com/fabianstelzer/status/1932730260350452136)
    - "new strategy just dropped Raise millions of VC $ Go on vacation for two years... This is what winning looks like in 2025" [@claud_fuen](https://twitter.com/claud_fuen/status/1932837037024977295)
    - "how much do you guys tip your AGI? 18, 20 or 22%?" [@johannes_hage](https://twitter.com/johannes_hage/status/1932696834818142581)
    - "It's honestly fucked up that YC accepted hundreds of straight men and only 1 toucan" [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1932888219450028368)
    - "Y2K is back. AI kids these days are like, 'tmol-faq. logi. gisai. cev. sysop. ufai. foom. rpop. flare.' i asked one what they were talking about and they said 'please read the archives before posting to sl4.'" [@goodside](https://twitter.com/goodside/status/1932990995638976668)
- **General Tech/Pop Culture**:
    - "Liquid glass has been achieved internally." [@claud_fuen](https://twitter.com/claud_fuen/status/1932769765317050659)
    - "i can't tell if getting a private pilots license is one of the coolest things you can do or just a fancy way to incinerate $20,000" [@typedfemale](https://twitter.com/typedfemale/status/1932860823032246428)
    - "The billionth repository was just created on GitHub and it is perfect in every way!" [@film_girl](https://twitter.com/zacharynado/status/1932983386219405754)
    - "You want cat girls" [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1932840090155560981) (in response to an image of various "types" of AI)

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Meta V-JEPA 2 World Model Video Training and New Model Release

- [**Meta releases V-JEPA 2, the first world model trained on video**](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6) ([Score: 218, Comments: 39](https://www.reddit.com/r/LocalLLaMA/comments/1l8umf2/meta_releases_vjepa_2_the_first_world_model/)): **Meta FAIR has released V-JEPA 2 (see [Hugging Face collection](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6)), a set of ViT-based models that advance their Joint Embedding Predictive Architecture (JEPA) for unsupervised video representation learning. The models, including vjepa2-vitl-fpc64-256, vjepa2-vith-fpc64-256, and vjepa2-vitg-fpc64-384, use transformer backbones and are trained as strong baselines for large-scale video analysis and multimodal research. Clarification: Though the post claims V-JEPA 2 is the first world model trained on video, commenters point out this is inaccurate (see original [Meta blog post](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)), and that V-JEPA 2 is an incremental release in Meta's ongoing world model development, not a field-wide first.** Top comments highlight that V-JEPA 2 is not the first world model trained on video—such models have existed in both industry and academia prior to this Meta release. Technical discussion focuses on how these models serve as competitive unsupervised baselines for complex video understanding tasks.
    - Multiple commenters clarify that the title is factually incorrect: V-JEPA 2 is not the first world model trained on video, nor is it Meta's only such model. Other organizations have also developed world models using video data prior to this.
    - A blog resource is referenced (https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) providing technical detail and background on the original V-JEPA model's architecture, predictive joint-embedding approach, and its relevance to predictive world modeling in video contexts.
    - One commenter notes that the model is capable of correctly predicting actions in video, implying the model's architecture effectively captures spatiotemporal dependencies necessary for accurate world-modeling; this success is attributed to Yann LeCun's contributions.
- [**Altman on open weight 🤔🤔**](https://www.reddit.com/r/LocalLLaMA/comments/1l8oe8g/altman_on_open_weight/) ([Score: 147, Comments: 100](https://www.reddit.com/r/LocalLLaMA/comments/1l8oe8g/altman_on_open_weight/)): **Sam Altman announced on X that the release of OpenAI's open-weights model has been delayed to later this summer (not June), citing an unexpected and impressive research development requiring more time. No technical details, benchmarks, or architecture information have been released; the claim is only that it will be 'very very worth the wait.'** Top commenters express skepticism about repeated delays and question whether OpenAI's release will surpass recent open-source efforts like DeepSeek R2, with some implying the delay is due to lack of substantive innovation.
    - One commenter compares the hype around Altman's purported open weights release to recent high-profile models like DeepSeek R2, implicitly suggesting skepticism that Altman's team could deliver something more technically impressive or impactful than established open or semi-open LLM releases. The discussion highlights ongoing doubts about whether anything new or competitive is forthcoming relative to state-of-the-art benchmarks set by models like DeepSeek R2.

### 2. Legal and Ethical Challenges Facing AI Image Companies

- [**Disney and Universal sue AI image company Midjourney for unlicensed use of Star Wars, The Simpsons and more**](https://www.reddit.com/r/LocalLLaMA/comments/1l8zssy/disney_and_universal_sue_ai_image_company/) ([Score: 265, Comments: 119](https://www.reddit.com/r/LocalLLaMA/comments/1l8zssy/disney_and_universal_sue_ai_image_company/)): **Disney and Universal are reportedly suing Midjourney, an AI image generation company, for unlicensed use of copyrighted characters such as those from Star Wars and The Simpsons, raising the stakes in legal battles over training data for generative AI. The post notes that, if Disney is successful, it could set a precedent impacting other AI companies whose models are trained on unauthorized copyrighted material, potentially reshaping the AI content generation landscape ([news reference](https://www.reuters.com/legal/disney-universal-sue-midjourney-over-unauthorized-use-copyrighted-characters-2024-05-30/)).** Discussion in the comments is minimal in technical substance, but one comment highlights the potential international implications, stating Chinese models may be unaffected by US legal action, indicating a jurisdictional gap in effective copyright enforcement against AI labs operating outside US/Western legal reach.
    - A commenter emphasizes that the legal dispute could have industry-wide repercussions, especially if Disney's litigation strategy targets precedent-setting compensation: if Midjourney is bankrupted, similar legal action could follow against other AI image model companies. The role of training data legality is highlighted—*the data used for training is widespread and persistent regardless of the outcome*, raising broader concerns about the feasibility of restricting generative AI given the nature of data sharing and collection.
    - There is speculation about the involvement or indirect support of large tech companies like Google and Meta, as the results of this lawsuit could influence the landscape for all providers of generative AI. The idea is that these 'deep pocket' companies have vested interests since restrictive legal precedent might affect their own AI models trained on similarly broad datasets, possibly inciting unified industry responses or behind-the-scenes lobbying.
    - Another technical comment suggests AI practitioners save local copies of image generation models, such as those on HuggingFace, in anticipation of models being taken down or becoming legally restricted if the lawsuit is successful. There's doubt expressed regarding platforms like HuggingFace's willingness to legally defend hosted models, hinting at fears of rapid model withdrawal from public access platforms due to potential copyright crackdowns.
- [**MNN TaoAvatar: run 3d avatar offline, Android app by alibaba mnn team**](https://v.redd.it/65vyq2fhca6f1) ([Score: 105, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1l8qh2a/mnn_taoavatar_run_3d_avatar_offline_android_app/)): **Alibaba's MNN team has released the open-source MNN TaoAvatar ([GitHub](https://github.com/alibaba/MNN/blob/master/apps/Android/Mnn3dAvatar/README.md#version-001)), an offline 3D avatar Android app that runs efficiently even on entry-level hardware. The implementation leverages the lightweight, mobile-optimized [MNN inference engine](https://github.com/alibaba/MNN), enabling real-time avatar rendering and animation. See further technical overview and benchmarking in the [arXiv paper](https://arxiv.org/html/2503.17032v1).** Technical commenters highlight notably smooth performance on low-end smartphones compared to other models like Qwen 3B Omni, underlining MNN's efficiency and optimization for mobile inference.
    - User abskvrm provides practical feedback, noting that the TaoAvatar app, running on the MNN framework, performs smoothly even on an entry-level smartphone, implying robust optimization for low-end hardware compared to heavier models such as Qwen 3B Omni, which ran significantly slower for them. This suggests excellent on-device efficiency and lightweight architecture for real-time avatar rendering.
    - The resources linked by Juude89—GitHub main page and the arXiv paper—provide technical deep dives, including implementation details, versioning, architecture, and performance claims, useful for technical readers seeking specifics about the Android app's operation and underlying MNN optimizations.

### 3. Local LLM Inference Stack Migration Experiences

- [**I finally got rid of Ollama!**](https://www.reddit.com/r/LocalLLaMA/comments/1l8pem0/i_finally_got_rid_of_ollama/) ([Score: 417, Comments: 212](https://www.reddit.com/r/LocalLLaMA/comments/1l8pem0/i_finally_got_rid_of_ollama/)): **The OP transitioned from using Ollama as their inference backend to a stack involving llama.cpp or ik_llama.cpp for inference, llama-swap for dynamic model loading/unloading (configured centrally via config.yaml), and Open WebUI as frontend, enabling custom model organization and greater decoupling from hardcoded model paths/names typical of Ollama. They highlight benefits such as flexible model management (easy wget from HuggingFace), interoperability with other inference engines, and system prompt customization per workspace, while crediting core open-source tools like llama.cpp and Open Webui.** Technical debate arises in comments: one user prefers llama-server for its potential to centralize sampling parameter control server-side (rather than per-client override as with many UIs), suggesting improvements for end-user accessibility. Another expresses concern that the OP's approach sacrifices convenience (easy model configuration, remote download/launch) present in Ollama, requiring more manual setup and reducing operational simplicity.
    - A user highlights a technical limitation of `llama-server` regarding sampling parameters: the current GUI client overwrites server-side sampler settings, requiring manual adjustment for each model. The user requests an option for the server to enforce its own sampling parameters regardless of client input, mirroring the behavior seen when interfacing llama-server via Python (where the server controls aspects like the jinja template, samplers, and system prompts by default). This would simplify deployment for users unfamiliar with sampling configuration, making it more "ChatGPT-like".
    - Another commenter outlines specific technical conveniences provided by Ollama that are not present in `llama-server`: automatic model acquisition, simplified configuration (models do not need to be defined manually), and support for remote downloading/launching of models. The implication is that switching to `llama-server` introduces significant overhead compared to Ollama in these regards.
    - It is noted that `llama-server` includes a built-in server that could make third-party UIs (like Open WebUI) unnecessary, potentially simplifying the user experience for those who want a direct, minimal-interface deployment.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Disney and Universal Lawsuits Against AI Image Generators

- [**Disney launches first major lawsuit against AI company Midjourney, calls the image generator a "bottomless pit of plagiarism"**](https://www.reuters.com/business/media-telecom/disney-universal-sue-image-creator-midjourney-copyright-infringement-2025-06-11/) ([Score: 406, Comments: 293](https://www.reddit.com/r/singularity/comments/1l8xpes/disney_launches_first_major_lawsuit_against_ai/)): **Disney has filed a major lawsuit against AI image generator Midjourney, accusing it of widespread copyright infringement and describing the platform as a "bottomless pit of plagiarism." The core argument reportedly concerns the allowed use of copyrighted images in training data and the nature of generative AI output. The case may hinge on current copyright law interpretations and whether training on protected works constitutes infringement.** Commenters question the legal basis of Disney's claim, drawing analogies to tools like Photoshop and highlighting Disney's historical reliance on the public domain; there is skepticism about the consistency and implications of copyright law as applied to generative AI.
    - A technical debate emerges about the legal rationale: some users question whether Disney's action is akin to blaming the tool (Midjourney/image generators) rather than the user, drawing analogies to whether Disney could sue Adobe Photoshop for users creating derivative works. This raises deeper issues about legal precedent regarding software/tool liability.
    - The broader discussion includes analysis of copyright law and fair use, with commenters pointing out that Disney itself historically relied on public domain works to build its IP portfolio, and criticizing the potential double standards and ambiguities in enforcement practices (e.g., Disney rarely acting against fan artists but aggressively targeting new generative models).
    - There are underlying concerns about the need for updates or reforms to copyright law, particularly to adapt to the challenges posed by generative AI—specifically, how it handles training data versus generated outputs, and whether expansive copyright enforcement stifles creative and technological progress.
- [**Disney and Universal sue AI image company Midjourney for unlicensed use of Star Wars, The Simpsons and more**](https://www.reddit.com/r/StableDiffusion/comments/1l8zmpb/disney_and_universal_sue_ai_image_company/) ([Score: 312, Comments: 274](https://www.reddit.com/r/StableDiffusion/comments/1l8zmpb/disney_and_universal_sue_ai_image_company/)): **Disney and Universal have filed suit against Midjourney, an AI image generation company, alleging infringement due to unlicensed use of intellectual property—including 'Star Wars' and 'The Simpsons'—in Midjourney's training data. This legal action targets the use of copyrighted media in training datasets, and the suit could set a key precedent affecting all AI labs using similar scraped data without explicit licensing. The underlying legal debate centers on whether training generative models on copyrighted works constitutes fair use or a violation.** Commenters note that smaller AI labs like Midjourney and NovelAI are more vulnerable legal targets, while larger entities (e.g., OpenAI, Microsoft) or open-source projects are harder to litigate against due to scale, legal complexity, or lesser financial incentive. There is also speculation that lawsuits could shift AI development or hosting to jurisdictions (e.g., China) less affected by U.S. IP law.
    - One user highlights the strategic legal risk calculus, noting that Midjourney and NovelAI are more vulnerable targets for copyright lawsuits from large studios like Disney, whereas attacking larger players such as OpenAI (backed by Microsoft) would likely result in a weaker legal position for copyright holders. Moreover, targeting open-source projects may not be worthwhile due to limited potential gains versus significant legal downside, suggesting studios are choosing defendants based on resource disparity and likelihood of settlement or victory.
    - There is speculation about the legal exposure of AI labs in different jurisdictions. Some commenters mention the potential for companies in China to develop models without fear of US-based copyright litigation, implying that international enforcement gaps could encourage commercial open-source or non-US-based AI model growth.
    - Commenters question why certain companies, like Grok and others, have not faced similar lawsuits, alluding to possible political or strategic reasons behind selective legal action, rather than purely technical or legal precedents. This touches indirectly on the legal ambiguity and enforcement selectivity in the AI copyright landscape.

### 2. 1X Neo Robot Demo and Sneak Peek

- [**A sneak peek at an update coming tomorrow from 1X.**](https://streamable.com/xoyjje) ([Score: 330, Comments: 115](https://www.reddit.com/r/singularity/comments/1l8w1a9/a_sneak_peek_at_an_update_coming_tomorrow_from_1x/)): **1X is teasing an upcoming update (launching tomorrow) with a video preview, likely showing their humanoid robot operating autonomously outdoors in a field setting. The referenced image (https://preview.redd.it/0kutog6ckb6f1.jpeg?width=483&format=pjpg&auto=webp&s=9c1d08cbb65b8fb0c0d470a918dda87f3f029ca1) suggests that this scene was previously foreshadowed, indicating continuity or new capabilities in the robot's outdoor mobility and AI.. The Streamable link hosts a standard video preview with no additional technical metadata; the focus remains on 1X's physical robot update rather than the video platform itself.** Technical discussion in the comments remains minimal, largely centering on societal implications ("it is coming for your jobs") and the aesthetics of the robot's operation in open fields, with no in-depth technical debate or bug reports present.
    - Dioxbit's comment raises concerns about the increasingly indistinguishable nature of content produced by advanced video generation models like Veo 3, indicating that synthetic footage is reaching a level where authenticity and real vs. generated classification become more technically challenging. This echoes ongoing debates about the necessity of reliable methods for detecting AI-generated media as models advance.
- [**New Neo Footage from 1X**](https://streamable.com/hiv798) ([Score: 264, Comments: 64](https://www.reddit.com/r/singularity/comments/1l8zfo6/new_neo_footage_from_1x/)): **A new video titled 'Neo Footage from 1X' has been posted and is hosted on Streamable, accessible [here](https://streamable.com/hiv798), but no technical details about the video content (formats, models, benchmarks, or features) are disclosed in the metadata summary. The post and linked page offer only standard video playback and sharing controls, with no information on the underlying technology, model, or inference process behind the footage, or any new technical capabilities demonstrated by 1X.** Comment discussions speculate on the artistic or thematic intent of 1X's work, suggesting a deliberate stylistic direction, but do not include substantive technical analysis, model details, or empirical assessment of the underlying technology.
    - Commenters note the intentional selection of visual styles and aesthetic filters in the footage, highlighting a deliberate attempt to evoke a retro, ethereal '80s vibe with specific grain effects and color grading that impact viewer perception and emotional response.
    - Another technical critique is that while the video showcases visual novelty, it lacks demonstration of useful task performance by the robot, implying that the current capabilities or demonstrations remain superficial and are not focused on practical utility.

### 3. Veo 3 AI Video Generation in Viral Advertising and Creative Projects

- [**I can't believe Disney approved my AI commercial to run during the NBA finals tonight 🤣**](https://v.redd.it/kp3nkjksrc6f1) ([Score: 662, Comments: 123](https://www.reddit.com/r/aivideo/comments/1l92j6j/i_cant_believe_disney_approved_my_ai_commercial/)): **The post details a full technical workflow for producing a viral, AI-generated NBA Finals commercial for Kalshi using Google Veo 3 (Google Flow) in just two days. The process integrates Gemini for script-to-prompt conversion (generating in batches of 5 to optimize quality), Veo 3 for prompt-based video generation (300–400 shots to yield 15 usable clips, at ~$0.20 per generation), and rapid editing with Capcut/FCPX/Premiere. Key limitations include lack of character consistency and unexpected subtitles in Veo 3. The approach achieves a claimed 95% cost reduction vs. traditional ad production and highlights emerging challenges/opportunities in attention-driven, comedy-based AI ad creation. [Original post](https://v.redd.it/kp3nkjksrc6f1)** Comments note the increasing prevalence of gambling-related ads and appreciate the transparent breakdown of the AI pipeline, with one mentioning the strong ROI and another highlighting the process as 'chaotic and badass.' No substantive technical debate is present.
    - A user expressed interest in compensation models for AI-driven commercial production, asking whether these lower-cost methods are financially sustainable for creators compared to traditional production. This raises questions about market rates and the economic viability of AI-generated media in professional settings.
    - Another comment highlights details of the workflow used in creating the AI commercial, suggesting there was substantial behind-the-scenes integration of generative AI techniques and possibly rapid content iteration. Technical readers may infer the importance of efficient pipelines and tool selection in expedited AI media production.
- [**The Monoliths: Humans react to mysterious structures**](https://v.redd.it/cmardtgcyc6f1) ([Score: 194, Comments: 21](https://www.reddit.com/r/aivideo/comments/1l93e3h/the_monoliths_humans_react_to_mysterious/)): **The post showcases a multimedia project titled 'The Monoliths,' created using Flow, Veo 3, and Suno AI—tools indicating a pipeline that likely leverages advanced AI generative models for audio and video synthesis. No benchmarks, quantitative model details, or explicit implementation specs are provided, but the combination suggests a workflow integrating visual (Flow, Veo 3) and audio (Suno AI) generative systems for cohesive synthetic media production.** Comments note rapid qualitative improvement in the results, hinting at swift advances in generative AI fidelity and output realism, but do not debate technical limitations or artifacts.
    - 

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Flash Preview
> 

**Theme 1. Model Performance, Benchmarks, and Comparisons**

- **O3 Pro Benchmarks Leave Users Wanting More**Initial [benchmarks for O3 Pro](https://cdn.discordapp.com/attachments/1340554757827461211/1382439311781400596/image.png) show *literally no difference* compared to the base O3 model, leading to sentiment that O3 Pro *wasn't exactly impressive*. Despite disappointing benchmarks, some Cursor users eagerly anticipate paying extra for O3 Pro integration.
- **Kingfall Model Sparks Debate, Benchmarks Contested**Members continue speculating about the [elusive Kingfall model](https://tenor.com/view/wink-eye-wink-gif-3023120962008687924), with one user claiming it significantly outperformed 0605 (32k) in auto-thinking tests. However, another user called these *ridiculous numbers* and suggested it might be an [OpenRouter bug](https://openrouter.ai/).
- **DeepSeek R1 Model Impresses, Gemini Flails**The **DeepSeek-R1-0528-UD-Q3_K_XL** model achieved an almost **80%** internal benchmark score, exciting UnslothAI members who shared a [Hugging Face link](https://huggingface.co/TheBloke/DeepSeek-R1-0528-UD-Q3_K_XL-GGUF). Meanwhile, Perplexity and OpenAI users reported [Gemini failed](https://ai.google.dev/) to create a game and struggled to analyze YouTube videos, though some found it worked perfectly for [this link](https://youtu.be/4MydzP3Mzy4?feature=shared).

**Theme 2. AI Agent Frameworks and Capabilities**

- **Windsurf Rolls Out Browser and 'Plan Mode'**Windsurf launched a new [fully functional browser](https://windsurf.com/blog/windsurf-wave-10-browser) to bridge development workflows and web activities, available in beta for Free, Pro, and Teams users, with a [YouTube video](https://youtu.be/r4WqTyLb4Vk?si=lNo4aMCIg8tHsVAp) and [changelog](https://windsurf.com/changelog). Windsurf also released a new **'Plan Mode'** feature allowing the AI agent to tackle complex tasks using a planning document on [Windsurf.com](http://windsurf.com/), which *worked well* in early tests.
- **MCP Server Discussions Advance UI, Integration, and Evals**Discussions around the **Multi-Compute Protocol (MCP)** focused on deploying servers as iframes or frontend APIs using a webio transport to resolve current issues and simplify setup with custom UIs and [OAuth flows](https://github.com/gitroomhq/postiz-app). [Hume AI detailed their eval approach](https://www.hume.ai/blog/roleplays-evals-hume-mcp-server) for their MCP server, sparking interest in evaluation methodologies, and a screenshot confirmed Hugging Face now supports MCP.
- **DSPy Primitives Blow Users Away with Agentic Power**Members praised **DSPy's** agentic patterns, noting the power of its primitives after refactoring Google's [gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) into a **200-line** [workflow.py](http://workflow.py/). The community seeks tools for easier dataset creation for DSPy and discussed compatibility with new reasoning models integrating tool-calling, anticipating the upcoming [DSPy 3.0 release](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1).

**Theme 3. Pricing and Access to AI Models**

- **OpenAI's O3 Pro Pricing Gets Roasted, Then Reduced**EleutherAI members initially criticized [O3 Pro's pricing](https://discord.com/channels/691289280227498005/729741769738158194/1252738340687548506) as costing **$20 input** and **$80 output** per 1M tokens, humorously suggesting it should *solve the Riemann hypothesis*. However, O3 subsequently saw an **80% price drop** to **$2 input** and **$8 output** per 1M tokens, which LMArena and OpenAI members agreed aligned with increased competition and Blackwell capacity.
- **OpenRouter Offers Nearly Unlimited TPM but Requires KYC for O3**OpenRouter clarified that while [rate limits apply based on the model](https://openrouter.ai/models?fmt=cards&supported_parameters=structured_outputs), they offer very high limits in practice, meaning *unlimited TPM* for users not using a personal OpenAI key. Aider members noted that OpenRouter still requires users to bring their own key and KYC to use the O3 model despite the significant price reduction, with one user stating *openai makes you show them your passport before they let you use* `o3`.
- **Cohere and Veo 3 Pricing Cause Consternation**Cohere users found its pricing for creative writing *insane* and reported a **2-second latency** with the reranking API, being directed to email [carolyn@cohere.com](mailto:carolyn@cohere.com) for custom solutions as no API tiers exist. [Manus.im](http://manus.im/) members lamented that a single **Veo3 video** costs **300 credits**, deeming it very expensive.

**Theme 4. Hardware and Infrastructure Developments**

- **Dual Socket AMD Turin and High-End GPU Setups Teased**Discussions included the potential of a **Dual Socket AMD Turin** server offering **1.2TB/sec** memory bandwidth and **640 GB/sec** PCI-e, enabling **16 GPUs and 384GB VRAM** with a [Supermicro motherboard link](https://www.supermicro.com/en/products/motherboard/h14dsg-o-cpu). LM Studio users successfully ran **Qwen 3 235b Unsloth** with **Q3_K_XL** quantization on an **Evo X2**, achieving **10-12t/s**, and debated high-bandwidth CPU-only solutions like an Octa-channel fast RAM server for 150B models, expressing skepticism about achieving 5 tokens/sec.
- **GPU Pricing Insanity Fuels 5090 Wait and Tinybox Sale**Unsloth AI members debated the *absurdity* of current **4090** pricing, with some opting to wait for the **5090**, noting used **4090s** are also overpriced. A tinygrad member is selling a used **Tinybox Green 6X 4090** from a data center at **70%** of its original price, sparking *interest*.
- **Modular Platform Embraces AMD, Releases Mammoth System**Modular announced the General Availability of its platform on **AMD InstinctTM MI300X** and **MI325 GPUs**, achieving up to **53%** better throughput on BF16 workflows, detailed in [their blog post](https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus). They also released **Mammoth**, a new [Kubernetes-native system](https://www.modular.com/blog/introducing-mammoth-enterprise-scale-genai-deployments-made-simple) designed to scale GenAI inference across any GPU, offering a public preview.

**Theme 5. Cutting-Edge AI Research and Foundational Concepts**

- **LLM Memorization Measured, Grokking Explored**A new [paper](https://arxiv.org/pdf/2505.24832) estimates that models in the **GPT family** have a capacity of approximately **3.6 bits-per-parameter** when measuring how much a model *“knows”* about a datapoint. The study observed that models memorize until their capacity is reached, after which *“grokking”* begins.
- **Prompting Tweaks Yield Huge Accuracy Leaps, Shuffled Answers Stumble Models**Members discussed a [paper](https://arxiv.org/pdf/2311.01967) showing how tiny changes in prompting massively affect accuracy on 0-shot tasks. Models also reportedly fail on MMLU questions when answer ordering is shuffled, even if they solved them correctly before, as described in [this paper](https://arxiv.org/pdf/2406.19470).
- **World Models Advance with V-JEPA and Theoretical Arguments**Meta AI released a new version of **V-JEPA**, aimed at advancing world model benchmarks, according to [their blog post](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) and [tweet](https://x.com/lxrjl/status/1932499153596149875). A [paper](https://arxiv.org/abs/2506.01622) argues that any agent capable of generalizing to multi-step goal-directed tasks *must* have learned a predictive model of its environment.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini Flops at Game Creation**: Members noted [Gemini failed](https://ai.google.dev/) to create a functional game, while discussing **Grok** and **Mistral**.
   - No specific reasons for the failure were provided, but the discussion highlighted the complexities of AI in interactive applications.
- **Kontext challenges GPT Image**: The new **Kontext** series outperforms **GPT Image**, though the open-source version is pending release; **Pro** and **Max** versions are available and it's a *lightweight 12B diffusion transformer suitable for customization and compatible with previous FLUX.1 inference code*.
   - **Kontext** will be distributed through [FAL](https://www.fal.ai/), [Replicate](https://replicate.com/), [Runware](https://run.ai/runware/), [DataCrunch](https://www.datacrunch.io/), [TogetherAI](https://www.together.ai/) and [HuggingFace](https://huggingface.co/).
- **Palantir Eyes US**: According to a [Perplexity AI search](https://www.perplexity.ai/page/palantir-builds-vast-us-survei-TGaRIDAIQA.I2nKHBK6W1g), **Palantir** is developing a large-scale surveillance system in the US.
   - Accompanying the report were four [screenshots](https://media.discordapp.net/attachments/1294380382661116045/1382398297456775288/1.jpg?ex=684b023d&is=6849b0bd&hm=35561a455d6f154727e9650366cdccefee0305d4dd36c59cead889c1b6d410c8&=&format=webp&width=648&height=864) showcasing **Palantir** interfaces and data visualization tools.
- **Postiz: Another Project Tracker?**: Members suggested using [Postiz](https://github.com/gitroomhq/postiz-app) for project management.
   - No concrete use cases or project details were discussed other than a pointer to its **GitHub** repository.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Kingfall Release Still a Mystery**: Members are still wondering about the [elusive Kingfall model](https://tenor.com/view/wink-eye-wink-gif-3023120962008687924), with one user sharing an image of an **iPhone 11 Pro** supposedly created by **Kingfall**.
   - Speculation continues that **Kingfall** might just be a nerfed version of **2.5 Flash lite**.
- **O3 Pro Benchmarks Fail to Impress**: Initial [benchmarks for O3 Pro](https://cdn.discordapp.com/attachments/1340554757827461211/1382439311781400596/image.png) reveal *literally no difference* compared to **O3**.
   - The general sentiment is that **O3 Pro** *wasn't exactly impressive*.
- **Thinking Budget's Impact Comes Under Scrutiny**: A member ran a quiz **30 times** on different thinking budgets, finding *there isn't* a correlation with increased length, so *long thinking is way more important than big models*.
   - The group is in agreement that focusing on long training runs could be more advantageous than scaling model size.
- **OpenAI's Pricing Strategy Under Microscope**: Members debated whether **OpenAI** had been *overcharging* for **O3**, agreeing the recent **5x** price reduction aligns with increased competition and **Blackwell** capacity.
   - Participants think the price reduction signals a response to market pressures and advancements in hardware capabilities.
- **Liquid Glass Design Surfaces as a Trend**: Users discussed a [liquid glass design](https://www.youtube.com/watch?v=1E3tv_3D95g), with one user stating *liquid glass will truly be the next design feature*.
   - There's [no consensus on whether it resembles liquid glass cleaner](https://i.imgur.com/GCzNMsh.png), sparking humorous comparisons.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-6 teased**: After some discussion about **GPT-5**, one user made a joke suggesting that **GPT-6.1-pro-preview-patch** is already here.
   - Other community members appear excited about future releases, but were more interested in the merits of **O3 Pro** versus the base model.
- **O3 Pro tempts Cursor Users**: Users are heavily debating [**O3 Pro**](https://openai.com/blog/new-embedding-models-and-api-updates) and whether the model is worth the cost, although some expressed they are ready to pay extra for pro, just to have it in Cursor.
   - Many users noted that **O3** is already impressively fast.
- **Mistral models for fun and profit**: Users praised the **Mistral** models for their effectiveness as chatbots and coding skills, noting that models are [available for free](https://mistral.ai/).
   - One user specifically appreciated **Mistral's** proficiency in generating physics-related content.
- **Background Agents Can't Connect**: Users are reporting issues with **background agents** failing to connect, often displaying a `ConnectError: [unknown] No response from model` message, particularly after hitting the **25 tool use limit**.
   - The problem persists even after multiple attempts to retry, with network diagnostics returning green, with additional information available in [this Discord thread](https://discord.com/channels/1074847526655643750/1380811765218283660).
- **Cursor Whips Codex in UE4 C++ Repo Handling**: A user reported that **Cursor** significantly outperformed **Codex** in handling a **UE4 C++ repo**, citing **Codex's 80% failure rate**.
   - They emphasized the speed of environment setup in Cursor, taking *less than half a minute* compared to Codex's *10 minutes*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O3 Pro's Price Plunge Provokes Pondering**: Users are discussing the **price reduction** of **O3 Pro**, wondering if the delayed release was due to **pricing concerns**.
   - Some members are speculating about the costs associated with **web search functionality**.
- **Gemini Gets Grilled for YouTube Gaffe**: Users reported that **Gemini** was failing to analyze YouTube videos, possibly due to being logged in with a Workspace/enterprise account or having the **YouTube extension** switched off, while others found **Gemini** can do it perfectly well such as with [this link](https://youtu.be/4MydzP3Mzy4?feature=shared).
   - Members speculated about whether this was caused by the new **O3 Pro** rollout.
- **ChatGPT Teases Non-Existent DOCX Feature**: Members complain that **ChatGPT** teases users with **docx files** but the feature doesn't actually exist and the feature has been **disabled for over a month**.
   - A member stated that this is a let down, especially if you're trying to compile data, since chatgpt can't remember ten seconds prior.
- **Custom GPTs Claim Competitive Edge**: A member suggested making **custom GPTs** to get better results.
   - He stated that depending on the model version, or if you have a subscription, you have the benefits.
- **Safety Config File Sparks Scrutiny**: A member shared a [config file](https://cdn.discordapp.com/attachments/1046317269069864970/1382207370427367574/config.txt?ex=684af92d&is=6849a7ad&hm=18e8864d5d0c91788c8e3e971f760cc507a8f144c51d09bc0da1d95285f30161) designed to make **LLMs more reliable**, with lower hallucination, truth anchoring, and awareness of impossible outcomes, which sparked some debate regarding the mention of explicit negative constraints like *CP is banned*.
   - One member argued that *enumerating forbidden tokens at the end of a config amplifies recency bias and increases the risk of LLM leakage* due to the pink elephant effect.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Reasoning Mode Left in the Dark**: A user asked what default reasoning mode is used for **OpenAI** requests through **OpenRouter**, but the question remains unanswered.
   - The user specifically wanted to know if it defaults to *detailed* or *auto*.
- **O3 Pro Pricing Creates Buzz**: Users showed interest in **O3 Pro** pricing, sparked by tweets from **Sama** and **Noam Brown**, with one user describing **O1 Pro** pricing as *bonkers*.
   - The exact pricing details for **O3 Pro** were not provided in the context.
- **Healthcare Pro Seeks OpenRouter Coding Help**: A healthcare professional requested assistance with **OpenRouter** code to input **279 prompts** into various **LLMs** for a healthcare consensus project, sharing a [pastebin link](https://pastes.dev/UqlfuJgF2W) to the code.
   - Suggestions included using unique persona prompts, rate limiting, validating **CSV** data, and ensuring that the **LLM** outputs **JSON** for easier parsing.
- **LLM Panelists Debate Models for Consensus Study**: Users discussed selecting **LLMs** for a consensus study, suggesting the inclusion of **Gemini**, **Claude**, and **Sonar** (Perplexity), while excluding **OpenAI**, **DeepSeek**, and **Grok** due to performance.
   - It was noted that for a fair comparison, non-reasoning models should be used, **Qwen** needs to be set to non-reasoning mode, and [Grok has low contextual memory](https://x.com/OpenRouterAI/status/1932828433148776650) and is *really stupid*.
- **OpenRouter's TPM Rates Practically Unlimited**: A user inquired about **TPM rate limits** imposed by **OpenAI** and whether they apply to **OpenRouter**, especially when not using a personal **OpenAI** key.
   - It was clarified that **OpenRouter** has very high limits in practice, meaning *unlimited TPM* for users, and [rate limits do apply based on the model used](https://openrouter.ai/models?fmt=cards&supported_parameters=structured_outputs).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Pricing Insanity Sparks 5090 Wait**: Members discussed the inflated prices of **4090s** and pondered waiting for the **5090** due to current market conditions.
   - Used **4090s** are reportedly overpriced, further fueling frustration among potential buyers.
- **Magistral Quants Officially Launched**: The UnslothAI team announced the release of **Magistral quants** and collaborated with Mistral to ensure accuracy, as noted in [their tweet](https://x.com/UnslothAI/status/1932441885618147402).
   - This collaboration aimed to guarantee that *everything is correct*.
- **DeepSeek R1 8Q Sets Benchmark Ablaze**: The **DeepSeek-R1-0528-UD-Q3_K_XL** model is blowing away members with its reported performance of almost **80%** on internal benchmarks, now available in this [link](https://huggingface.co/TheBloke/DeepSeek-R1-0528-UD-Q3_K_XL-GGUF).
   - The model tested was a few days old, from June 8.
- **Ollama** Embraces **Safetensors**, Goodbye Weights!**: Community members highlighted that while `save_to_gguf` remains a work-in-progress, users can now save merged models in **safetensors** format.
   - Subsequently, they can leverage [ollama's create --quantize command](https://ollama.com/docs/guides/how-to-create-modify-prompt#quantize-your-model) to seamlessly convert them into an **Ollama**-compatible format.
- **Prompting Changes Yield Accuracy Jumps**: A member shared a [paper](https://arxiv.org/pdf/2311.01967) highlighting that a tiny change in prompting massively changes the accuracy on 0-shot tasks.
   - Models are also failing on MMLU questions when answer ordering is shuffled as described in [this paper](https://arxiv.org/pdf/2406.19470).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Sonnet 4's Reasoning Vanishes**: The **Sonnet 4** with reasoning test disappeared from **OpenRouter**, suggesting its reasoning capability is no longer available.
   - Discussion members speculated on **Mistral's** reasoning mode activating via its chat template ([huggingface link](https://huggingface.co/mistralai/Magistral-Small-2506)) but found issues such as models looping the final answer due to reward hacking.
- **GRPO's Length Bias Attacked**: Discussion around **GRPO** reveals it has a length bias that might be addressed via fixes like **DRGRPO** or **DAPO**, with internal training environments using length penalties as described in [this github link](https://github.com/NousResearch/atropos/blob/main/environments/tool_calling_server.py#L370C1-L397C71).
   - Members debated the applicability of the **ProRL** paper ([arxiv link](https://arxiv.org/abs/2505.24864)) to larger models, questioning its conclusions regarding entropy collapse issues.
- **Knowledge Distillation Exposure Bias Emerges**: A member is distilling using a **student -> critic -> teacher** setup, storing teacher logits to fine-tune the student model to upgrade a student model to a reasoner.
   - They fear **exposure bias** from the teacher's reasoning patterns and are unsure how to approach distillation across reasoners.
- **McCulloch-Pitts: AI Foundation**: The [McCulloch-Pitts paper](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf) is considered a foundational work of **AI** and **computational neuroscience**.
   - The model proposed is a simplified model of **neurons** and **neural networks**, capable of computing any **Turing-computable function**.
- **Frutiger Aero Revival**: Members expressed a desire for resources to be focused on bringing back **Frutiger Aero** and implementing **AI Slop iMessage backgrounds** instead of **AI heart monitoring** technologies.
   - One member stated *nobody cares about ai heart monitoring that would save/extend lives*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API Doc Destination Determined**: Users looking for **Swagger documentation** for the **LM Studio Server API** were directed to the [official API documentation](https://lmstudio.ai/docs/app/api) and the **Developer Tab** within LM Studio.
   - The conversation revolved around structuring HTTP requests to interface with the **LMS server**.
- **Image Generation Gems Given Generously**: A user with a **4070ti Super** seeking local image generation recommendations was advised to explore **Fooocus** and **ComfyUI**, and potentially use **Pinokio** for installing point solutions.
   - Additional recommendations included using quantized **Flux.dev** models and fine-tunes on **civitai**, with a preference for **sd-forge webui** despite ComfyUI's steeper learning curve.
- **Qwen 3 235b fits on Evo X2**: A user inquired about running **Qwen 3 235b Unsloth** on an **Evo X2**, and another user reported success using the **Q3_K_XL** quantization, achieving **10-12t/s**.
   - The setup involved using Linux, a context length of 10000, Q8 quantization for the KV cache, no mmapping, and an Evolution Batch Size of 320.
- **Dual Socket AMD Turin a Beastly Buy**: A **Dual Socket AMD Turin** server would have **1.2TB/sec** memory bandwidth and **640 GB/sec** of PCI-e, and a machine with 8 dual B60s would enable **16 GPUs and 384GB VRAM**.
   - One user linked to a Supermicro motherboard [Supermicro.com](https://www.supermicro.com/en/products/motherboard/h14dsg-o-cpu) with 20 x8 slots.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Mini PC RAM: soldered is so sad!**: Members discussed why mini PCs are limited to **128GB RAM** due to space issues and the use of soldered **LPDDR5**.
   - They were questioning why these chips can't be soldered on a stick to allow for more RAM.
- **Open Source LLM Agent Framework Debuts**: A member open-sourced their **LLM agent framework** [available on GitHub](https://github.com/starsnatched/llm-backend), which can use a **Linux terminal on a VM** with full access, enabling it to store, modify, delete, and move files, as well as gather information from the web.
   - The author emphasized that *we will never "solve history" and quantum-proof truth like this...*
- **Qwen 3 1.7b Powers Conversational Agent**: A member shared that they tested models down to **Qwen 3 1.7b** to make a conversational agent, and also used the same model for **SQlord**.
   - Another member inquired if it worked well at that size and expressed interest in using their functions to fine-tune the model.
- **Agents Course Final Deadline Announced**: The deadline for the course is **July 1st** but new members can complete the first unit and get a certificate in one day.
   - Members also reported on the costs of the **OpenAI 4o-mini** model being roughly **$10 for 50 questions** but subsequent emails indicated **o3** is cheaper than **4o**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **GPT Models' Memorization Measured**: A new [paper](https://arxiv.org/pdf/2505.24832) estimates that models in the **GPT family** have a capacity of approximately **3.6 bits-per-parameter** when measuring how much a model *“knows”* about a datapoint.
   - The study observed that models memorize until their capacity is reached, after which *“grokking”* begins.
- **DAN Agent Debuts with Multi-Modal Flair**: A member shared their **DAN (Do Anything Now) agent**, which generates images, videos, a story, and narration from a single prompt, shared via [this Ollama link](https://ollama.com/PythagoraTheorem/Aimee3).
   - The agent includes conversation mode with memory, a script improvement extension, and a terminal operator, designed for community-driven extensions.
- **Meta's V-JEPA Advances World Model Benchmarks**: **Meta AI** released a new version of **V-JEPA**, aimed at advancing world model benchmarks, according to [this blog post](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) and [tweet](https://x.com/lxrjl/status/1932499153596149875).
   - This release seeks to improve the evaluation and development of AI models that can better understand and predict the dynamics of the world.
- **Mistral's Compute Clouds Emerge**: **Mistral AI** announced **Mistral Compute**, aiming to democratize AI infrastructure and give more people the tools to build and own AI infrastructure, according to [their blog post](https://mistral.ai/news/mistral-compute).
   - The launch includes various services and resources designed to empower developers and researchers in the AI field.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI Unveils Product Key Memory Sparse Coders**: EleutherAI released a [blogpost](https://blog.eleuther.ai/pkm-coders/), [code](https://github.com/EleutherAI/sparsify/tree/e2e-pkm) and [checkpoints](https://huggingface.co/EleutherAI/pkm-coders/) for researchers to experiment with **Product Key Memory (PKM)** modules for **sparse autoencoders** and **transcoders**.
   - The team found that **PKMs** speed up training and inference, reduce memory usage, and induce hierarchical grouping, but they are unsure whether the improvements justify the added complexity.
- **O3 Pro Pricing Gets Roasted**: Members criticized the [pricing of O3 Pro](https://discord.com/channels/691289280227498005/729741769738158194/1252738340687548506), noting its input cost is **$20 / 1M tokens** and output cost is **$80 / 1M tokens**.
   - Some users humorously remarked that the pricing implied it should be able to *solve the Riemann hypothesis* and considered it *worse than muon* and *worse than RWKV*.
- **Harvard Library Dataset to Unlock Trove of Knowledge**: A [paper](https://arxiv.org/abs/2506.08300) discusses making a set of books accessible through **Harvard Library's share of Google Books**, with a dataset covering about **1 million books** verified as public domain.
   - Members believe the dataset is largely new, and they have been eagerly awaiting it for over a year, with the code and data expected to be released soon.
- **Cosine Decay Gets Scrutinized**: Members are looking for papers and intuition on **cosine decay** to a minimum value (e.g., 10% of peak) versus decaying all the way to 0, questioning if a minimum helps generalization in smaller SFT runs, discussing [this paper](https://arxiv.org/pdf/2502.15938).
   - One member suggests for small LLMs, two epochs work best (first epoch with warmup and linear decay, second epoch with cosine decay to 0), referencing [this paper](https://arxiv.org/pdf/2404.06395) and [this Meta AI research](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/).
- **Knock Off Error Control resurfaces for AI Safety**: A member suggested that ideas from **knock off error control** might be useful, sharing a link to the [Knockoff Inference](https://arxiv.org/abs/1811.06687) paper.
   - Another member expressed gratitude, stating that they haven't seen knockoffs mentioned in like **4 years** and that it's a reminder they should learn about them.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O3 Price Slashed but KYC Stays**: The **O3 model** saw an **80% price drop** now costing **$2 input** and **$8 output** but [OpenRouter](https://openrouter.ai/) still requires users to bring their own key and KYC.
   - A user noted that *openai makes you show them your passport before they let you use `o3`*.
- **Kingfall Kicks Butt, Benchmarks Questioned**: A user's comparison of **0605 (32k)** vs **Kingfall (auto thinking)** showed that **Kingfall** performed much better in auto-thinking.
   - However, another user contested these numbers and stated that they are *ridiculous numbers*, and might be an [OpenRouter bug](https://openrouter.ai/).
- **R1 Leaves Users Wanting More**: Members debated the performance of the new **R1 model**, with one suggesting the new **R1** is better at **$4.8**, but another countered that they believe the new **R1** is even worse than `0506`.
   - One member stated, *Nearly all benchmarks are opne souecemeaning that ai companies can train on the bench* casting doubt on the utility of all benchmarks.
- **Aider User Runs Wild with Pro Max**: A user mentioned they are using a fair bit of the **Pro Max subscription**, to the point that they get usage warnings due to running several **Claude code instances** in parallel.
   - The user jokingly admitted to *kinda asking for it though*.
- **Deepseek-R1 Config Causes Headaches**: A user is experiencing difficulties configuring **deepseek-r1** from chutes within Aider.
   - Another user suggested setting up a [.env file](https://aider.chat/docs/config/dotenv.html) to configure it.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Custom AI Styles Spark Debate**: A user inquired about tailoring **AI conversational styles** based on demographics like *age, ethnicity, and gender*, to get more **personalized AI interactions**.
   - The conversation sparked interest in techniques and tools to get the **conversational style** they want, but no solutions were identified.
- **Audio Overview Confusion Emerges**: New users are struggling to customize the **AI audio overview generator** in NotebookLM to create separate overviews for each topic.
   - Users are reporting difficulty locating the "customize" option in the app, and are seeking best practices for **generating audio from documents**.
- **Pierceday Metalabel Use Cases Explored**: Users are exploring various use cases for [Pierceday Metalabel](https://pierceday.metalabel.com/aphone), suggesting applications in car manuals, maintenance details, electrical box notes, and **conference presentations**.
   - It was noted that the tool could be useful for adding **contextual information** to various physical objects.
- **Podcast Length Limits Debated**: Inspired by online examples, users are investigating methods to generate **podcasts longer than 20 minutes** using NotebookLM.
   - The conversation focused on finding workarounds and best practices for **generating longer audio content** from NotebookLM sources.
- **LaTeXLM Extension Released**: A user released an [open-source chrome extension](https://github.com/hachoj/LaTeXLM) to enable **MathJax rendering** within NotebookLM.
   - The extension, designed to improve the display of **mathematical equations and notations**, is available on GitHub, with plans for future Chrome Web Store publication.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cursor Eyes Anthropic Investment Amid Windsurf's Exit**: With **Windsurf** now aligned with **OpenAI**, **Cursor** is potentially doubling down on its investment in **Anthropic**, raising questions about a possible deal.
   - The community believes **Cursor** is more invested in **Claude Code** than **OpenAI** was with **Codex**, with users reporting that **Claude Code** *clicked* for them.
- **Apple Mulls Anthropic Buyout for Siri Upgrade**: Speculation suggests **Apple** should acquire **Anthropic** due to their financial capacity and urgent need to improve **Siri**.
   - A member quipped that **Siri** *can't reliably send a text message with a 15-year head start*, highlighting the pressing need for improvement.
- **Altman Hints at Gentle Singularity**: [Sam Altman's blog post](https://blog.samaltman.com/the-gentle-singularity) sparked interest and discussion about a **gentle singularity**.
   - The link was shared alongside [Kevin Hou's X post](https://x.com/kevinhou22/status/1932516093333266538?s=46), further fueling the conversation.
- **Windsurf's 'Plan Mode' Manages Tasks**: **Windsurf** released a new **'Plan Mode'** feature, allowing the **AI agent** to perform complex tasks using a planning document, accessible for free on [Windsurf.com](https://windsurf.com/).
   - Early tests on a small greenfield project showed that **'Plan Mode'** *worked well*.
- **Sharon Zhou Joins AMD**: Sharon Zhou is joining **AMD**, collaborating with **Lisa Su** to focus on **AI research** and scaling, bringing colleagues from **LaminiAI**.
   - Zhou aims to **democratize GPUs** for **AI**, as highlighted in [this X post](https://x.com/realSharonZhou/status/1932817096510931380) from the **#AdvancingAI** conference.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Hume AI Launches MCP Evals**: **Hume AI**'s approach to *evals* for their **MCP server** is described in [a new blog post](https://www.hume.ai/blog/roleplays-evals-hume-mcp-server), inviting discussion on evaluation methodologies.
   - Community members show curiosity on how others are evaluating their systems.
- **iframe MCP Servers Spark Debate**: The concept of deploying an **MCP server** as an *iframe* or frontend API gains traction, supported by arguments for utilizing a **webio transport** akin to the existing stdio transport.
   - This method promises to resolve current **MCP** issues by enabling custom UIs and **OAuth flows** via `window.open`, streamlining setup with URL copy-pasting, and managing a virtual filesystem via web APIs rather than granting shell access.
- **Hugging Face Enters MCP Arena**: A screenshot shows that **Hugging Face** now supports **MCP**, indicating a significant development for those involved in model development.
   - This is exciting for those involved in model development.
- **OAuth2 UI Eases MCP Connections**: Integrating a real **OAuth2 UI** for authentication would significantly ease connections to services like **Google** and **GitHub** for end-users, vastly improving the user experience for **MCP** servers.
   - Ideally, **OpenAI**, **Anthropic**, and **Google** implementing **OAuth2** login for their APIs would further simplify this process.
- **Slides.com Embraces MCP**: A hosted **MCP server** is available for generating [slides.com presentations](https://www.epicai.pro/use-ai-to-create-presentations-with-mcp-tsb4j).
   - This represents a practical application of **MCP** in content creation.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Edu Emails Exclude UK Domains**: A member inquired whether *.edu* email domains encompass *.UK* (*ac.uk*) domains and another member said they don't see *UK* domains being included.
   - A screenshot confirmed the exclusion of *.UK* domains from the *.edu* domain list.
- **ReactNexus Scheduled for July 2025**: Members discussed the upcoming **ReactNexus** event ([https://reactnexus.com/](https://reactnexus.com/)) scheduled for **July 3-5, 2025**, at the **J N Tata Auditorium**.
   - The conference is focused on **React**, a popular JavaScript library for building user interfaces.
- **Veo 3 Pricing Elicits Ire**: A user lamented that a single **Veo3 video** costs **300 credits**, and processing **38 clips** proved costly.
   - Another user echoed the sentiment, expressing concerns about the high expense and questioning the accessibility of **Veo3** to a broader audience.
- **Manus Chat Mode Debuts for Free**: **Manus** introduced a **FREE & UNLIMITED Chat Mode** for all users, enabling them to ask questions and receive instant answers.
   - Users can upgrade to **Agent Mode** for more advanced capabilities, such as creating comprehensive output.
- **High Effort Mode Vanishes**: Several users reported the disappearance of **High Effort Mode** from their **Pro accounts**.
   - One user remarked on the puzzling requirement of manually selecting **High Effort Mode**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Restructures Operations Directory**: The **Tinygrad** project is [refactoring](https://xl0.github.io/tinygrad-notes/bounty1.html) its codebase by moving operations to a separate directory, which may impact developers working directly with **Tinygrad** internals.
   - The aim is likely to improve code organization and maintainability.
- **Used Tinybox Green on Sale**: A member is offering a used **Tinybox Green 6X 4090** for sale at **70%** of its original price, noting it's in perfect working condition from a data center.
   - Another member expressed *interest*, signaling a potential transaction within the community.
- **Debate Sparks over CS Degree Relevance**: A member questioned the continued usefulness of a **CS degree**, igniting a discussion on its value in the current tech landscape.
   - One response suggested that questioning the degree's value negates its necessity, indicating a perception that its benefits should be self-evident.
- **SVD Bounty Sparks Algorithm Discussions**: The **linalg.svd bounty** discussion led to a proposal to use the **Jacobi algorithm** for eigenvalue computation in **Tinygrad**, with members considering handcoding functions to ensure zero dependencies.
   - It was also suggested to use modified versions of the **Jacobi algorithm**, referencing [A Novel Fully Hardware-Implemented SVD Solver Based on Ultra-Parallel BCV Jacobi Algorithm](https://cdn.discordapp.com/attachments/1070745817025106080/1382505351634616381/A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf?ex=684b65f1&is=684a1471&hm=3eeaa1287761b9210d1e4a54b7c65b1be2a3c4b3838d55d14e60ca76d8cbefc7&).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Launches on AMD, Mammoth Unveiled**: The **Modular Platform** now supports **AMD InstinctTM MI300X** and **MI325 GPUs**, improving throughput up to **53%** on prefill-heavy **BF16** workflows, as detailed in [their blog post](https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus).
   - **Mammoth**, a new **Kubernetes-native system**, scales **GenAI inference** across any GPU, enabling deployment of Hugging Face models across **AMD** and **NVIDIA** using a single container, without manual configuration, with a [public preview available](https://www.modular.com/blog/introducing-mammoth-enterprise-scale-genai-deployments-made-simple).
- **Mojo Integrates with Python**: **Mojo kernels** can now be integrated directly into Python workflows, supported by **450k+ lines of open source Mojo kernel code** and available in nightly builds.
   - Developers can start using Mojo within Python environments, guided by [this documentation](https://docs.modular.com/mojo/manual/python/mojo-from-python/).
- **TensorWave Offers Free AMD Compute for Modular**: In partnership with **TensorWave**, users can now test the Modular Platform in real workloads using free AMD compute.
   - Interested users can access this offer at [Modular.com/tensorwave](https://www.modular.com/tensorwave) to evaluate the platform's performance capabilities.
- **Mojo Cross-Compilation Causes Cross-Platform Concerns**: Direct cross-platform static compilation from macOS to Linux is currently unsupported in Mojo, prompting a member to explore Docker containerization.
   - The member encountered an *'apple-m1' is not a recognized processor'* error, leading to the consideration of bundling dependencies and `.mojo` files to run within a **Docker container** on a serverless platform.
- **Mojo on Runpod Runs Rapid, Initially!**: A member reported success running Mojo GPU code on [runpod.io](https://runpod.io), noting good performance with fast hot executions.
   - The initial cold start time is around 10 seconds, but they plan to share a detailed setup post on the forums.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Flex Attention Peaks Under Pressure**: A user reported higher peak memory usage in a `(bs, seqlen*8)` input versus a `(bs*8, seqlen)` input when using **flex attention**, suspecting the softmax square matrix in self-attention.
   - The user is investigating peak memory usage with **flex attention** and **FSDP**, noting peak memory use jumps rapidly after a "tipping point" when running a sweep.
- **Tokenizer Integration Tweaks Incoming**: A member plans to iterate on [#2574](https://github.com/pytorch/torchtune/pull/2574) and [#2794](https://github.com/pytorch/torchtune/pull/2794) to improve the new **HF tokenizer** and its integration, with experiments planned using **C4**.
   - A pull request is expected to fix [#2809](https://github.com/pytorch/torchtune/pull/2809), further solidifying **tokenizer** support.
- **Iterable Datasets Get Packing Refactor**: A packing refactor has been proposed in [Proposal on packing refactor](https://github.com/pytorch/torchtune/pull/2819) to work with iterable datasets to support packing for **DPO**, **GRPO**, **multimodal**, etc.
   - These changes are expected to broaden the configurations supported by models.
- **Nemo RL Schematics Surface**: The plans for **Nemo RL** have [surfaced](https://cdn.discordapp.com/attachments/1360680363885854841/1382420201823404032/AP1GczOQioexSd_ieqkppCKoVizt91prnymZ_uGi6mCeQdrSJE65osblAXMqxQw3030-h2272-s-no-gm.png?ex=684b16a4&is=6849c524&hm=8ca8961e205603c01114bb66f46acb3bb86d01b2d1297bef22f7817f5b6efeca&), originating from the Databricks conference.
   - Further details are scarce, but the graphic outlines the general approach being taken.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere's Costs Cause consternation**: Members voiced concerns over **Cohere's** pricing for creative writing tasks, deeming it excessively expensive, while highlighting its **North** agent currently in beta.
   - One member noted that while [Claude code](https://www.anthropic.com/product) excels in code generation and understanding, [n8n](https://n8n.io/) is better suited for workflow automation when building agents.
- **Multi-Modal Re-Ranker Request Rebuffed**: A member inquired about **Cohere** releasing a **multi-modal re-ranker** for image re-ranking, but was informed that **Cohere** does not currently offer one.
   - Alternatives suggested included using **GPT-4.1** with structured output or exploring **CLIP** and **openCLIP** for relevant solutions.
- **Cohere Charts Course Northward with EnsembleHP**: **Cohere** is partnering with **EnsembleHP** to deploy **Cohere North** in the healthcare sector via their secure **AI agents platform**.
   - The aim is to alleviate administrative burdens and improve the patient experience; more details are available on the [Cohere blog](https://cohere.com/blog/ensemble-partnership).
- **API Tiers Tabled at This Time**: A user inquired about API tiers akin to **OpenAI**, but was informed that **Cohere** does not provide predefined API tiers.
   - A member suggested contacting [carolyn@cohere.com](mailto:carolyn@cohere.com) for custom solutions, after a user reported a **2-second latency** with the reranking API.
- **Datatune Dazzles with Natural Language Transformations**: The co-founder of **Vitalops** introduced [Datatune](https://github.com/vitalops/datatune), an open-source tool that performs data transformations using plain natural language.
   - They expressed enthusiasm about joining the **Cohere** community and learning from fellow members, and getting involved with the community.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Cleanlab and LlamaIndex Integrate for Trustworthy Insights**: [CleanlabAI](https://cleanlab.ai/) and [LlamaIndex](https://www.llamaindex.ai/) have integrated to build AI knowledge assistants and production agents, enhancing the trustworthiness of **LLM responses** by scoring trust and detecting hallucinations.
   - The integration was [announced on Twitter](https://twitter.com/llama_index/status/1932837489238290941) with the goal of improving the reliability of insights derived from enterprise data.
- **Community Calls for LlamaIndex to Claim Chainlit Code!**: With [Chainlit](https://github.com/Chainlit/chainlit) being decommissioned, users are actively encouraging LlamaIndex to acquire its code, highlighting its importance within the **LLM ecosystem** and its seamless integration with LlamaIndex.
   - Chainlit is praised for its pure Python implementation and simple deployment, allowing use on many platforms such as Discord and Slack, and one member noted *LlamaIndex + Chainlit works amazing!*
- **AI Security Webinar to Handle Hacken's Hazards**: Hacken is hosting a webinar on **June 12 at 13:00 UTC** about **AI security**, exploring **LLM vulnerabilities** and defenses, featuring Stephen Ajayi.
   - Those interested can find more details and register via the [Luma link](https://lu.ma/xl53xbfs) to learn how to handle any unexpected Hacken's hazards.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gemini Fullstack Langgraph Gets Quickstart**: Google released a full-stack implementation of a research app called [gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) with members noting that it's *a very good* implementation.
   - A member then refactored the **LangGraph** portions of the **Gemini** code with **DSPy** and implemented a simple **React** front end, available [on GitHub](https://github.com/pgurazada/deep-research.git).
- **DSPy's Agentic Pattern's Pack a Punch**: Members have implemented *so many* **agentic patterns** with **DSPy** and were *blown away by how powerful the primitives are*.
   - The refactored workflow is only **200 lines long** ([workflow.py](https://github.com/pgurazada/deep-research/blob/main/backend/agent/workflow.py)) and *elegantly implements the original Langgraph workflow with much lesser hassle*.
- **Community Seeks DSPy Dataset Dev Tools**: A member inquired about tools to easily build and export datasets for **DSPy**, facilitating synthetic example generation and manual labeling.
   - Another suggested that a custom **Streamlit app** could be effective, and coding agents like **Cline** can assist in its creation with minimal guidance.
- **Reasoning Models and DSPy Make Their Debut**: A member asked about **DSPy's** compatibility with new reasoning models that utilize tool-calling in the reasoning process, such as **o3 / o3-pro / o4-mini**.
   - They noted that while `dspy.ReACT` exists, it seems designed for the chat API era rather than the responses API era with tool-calling integrated.
- **DSPy 3.0 Approaches**: A member announced the upcoming **DSPy 3.0** release and linked to the [DSPy 3.0.0b1 release tag](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1).
   - They asked if there's a comprehensive overview of what's to come in **DSPy 3.0**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Python SDK Update Buzzes**: Members in the **GPT4All** channel expressed anticipation for an upcoming **Python SDK** update, with no specific details disclosed.
   - The community awaits to see what improvements and features this update will bring to the **GPT4All** ecosystem.
- **GPT4All Eyes Magistral Small**: A member inquired whether **GPT4All** will support **Mistral's Magistral Small** model, sparking a brief discussion.
   - Alternatives like **JAN**, **LM-Studio**, **obadooga**, and **koboldcpp** were suggested by other members, while the original inquirer decided to wait, citing concerns about model speed.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX Submission Auto-Entry?**: A member asked whether submitting a paper to the **Research Track** competition automatically enters it into consideration for the **AgentX summit's call for papers and proposals**.
   - They wanted to clarify if a separate submission is needed for the summit.
- **AgentX Finalists Must Register?**: A member also asked if finalists need to register for the summit to attend.
   - They worried that tickets might sell out before competition results are released, possibly preventing attendance if they aren't finalists.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Cerebras Hosts AI Tech Talk**: Cerebras is hosting a free AI workshop this **Friday, June 13th**, from **12:00–1:00PM PST**, featuring speakers from Cerebras and Artificial Analysis.
   - The talk will cover topics including new models like **Alibaba’s Qwen 3 series** and model selection strategies, with [RSVP here](https://lu.ma/7f32yy6i?tk=jTLuIY&utm_source=ella).
- **AI Workshop Focuses on Model Selection**: The AI Workshop dives into current research, demonstrating how to pick the right model for a project.
   - Researchers will provide insights and strategies for effective model selection.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Launches Fully Functional Browser**: Windsurf is shipping a new **fully functional browser** that bridges the gap between your development workflow and web-based activities, as part of [Windsurf Wave 10 - Day 2](https://windsurf.com/blog/windsurf-wave-10-browser).
   - The new **Windsurf Browser** is rolling out in beta to all **Free, Pro, and Teams users**, while Enterprise users will receive this on a rolling basis.
- **Windsurf Browser Available to All**: The new **Windsurf Browser** is rolling out in beta to all **Free, Pro, and Teams users**, while Enterprise users will receive this on a rolling basis.
   - Watch the [video on Youtube](https://youtu.be/r4WqTyLb4Vk?si=lNo4aMCIg8tHsVAp), read the [changelog](https://windsurf.com/changelog) or join the [conversation at r/Windsurf](https://reddit.com/r/windsurf).
- **Windsurf Explores New Social Horizons**: Windsurf has created new profiles on [X/Twitter](https://x.com/windsurf_ai/status/1932871558219117022), [Bluesky](https://bsky.app/profile/windsurfai.bsky.social), [Threads](https://www.threads.com/@windsurf_ai/post/DKxShipsbPk?hl=en), [Instagram](https://www.instagram.com/p/DKxWKKkxvu6/) and [Linkedin](https://www.linkedin.com/feed/update/urn:li:activity:7338638111393886211/).
   - Stay up to date with **Windsurf's social media** to find out the latest company news and product updates.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM Leaderboard to Update Soon**: The **Gorilla LLM** leaderboard will be updated this week.
   - This was communicated in the `#leaderboard` channel.
- **Second Topic Placeholder**: This is a placeholder topic to satisfy the minimum item requirement.
   - It does not reflect any actual discussion but is necessary for validation.



---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1382076539293798521)** (1271 messages🔥🔥🔥): 

> `Grok, Gemini, Mistral, Kontext, Flux` 


- **Gemini's Game Plan**: Members discuss **Grok**, **Gemini**, and **Mistral**, and they mentioned [Gemini failed](https://ai.google.dev/) to make an actual game.
- **Flux Kontext is the new GPT Image Killer**: **Kontext** is the new Flux series that outperforms GPT Image however, the [open source version isn't out yet](mailto:kontext-dev@blackforestlabs.ai), but **Pro** and **Max** are.
   - It's a *lightweight 12B diffusion transformer suitable for customization and compatible with previous FLUX.1 inference code* and will be distributed through [FAL](https://www.fal.ai/), [Replicate](https://replicate.com/), [Runware](https://run.ai/runware/), [DataCrunch](https://www.datacrunch.io/), [TogetherAI](https://www.together.ai/) and [HuggingFace](https://huggingface.co/).
- **Perplexity users want better limit on O3**: Users are unhappy about the **100 per week** limit of O3, claiming that it should be *400-500 per week* since O3 is cheaper, suggesting [matching Gemini's limit](https://ai.google.dev/) would be good.
- **DIA vs Comet browser**: Members discuss [DIA](https://dia.com/) and [Comet](https://www.comet.com/site/)
   - Members are looking forward to it.
- **GPT-5 flash: Apple open ai pplx glazer**: Members discuss gpt 5 to be unlimited uses for free users but rate limits like gpt 4o.
   - Users discuss [GPT5 mini](https://openai.com/blog/openai-devday).


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1382352832224559277)** (2 messages): 

> `Palantir surveillance, Government contracts` 


- **Palantir builds vast US surveillance system**: Palantir is building a vast US surveillance system, according to a [Perplexity AI search](https://www.perplexity.ai/page/palantir-builds-vast-us-survei-TGaRIDAIQA.I2nKHBK6W1g).
- **Image dump shows Palantir screenshots**: Four [screenshots](https://media.discordapp.net/attachments/1294380382661116045/1382398297456775288/1.jpg?ex=684b023d&is=6849b0bd&hm=35561a455d6f154727e9650366cdccefee0305d4dd36c59cead889c1b6d410c8&=&format=webp&width=648&height=864) show Palantir interfaces.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1382234369586102284)** (4 messages): 

> `postiz` 


- **Try Postiz for Your Project**: A member suggested trying [Postiz](https://github.com/gitroomhq/postiz-app) for project management.
   - No other details were given.
- **Postiz Github Repository**: [Postiz](https://github.com/gitroomhq/postiz-app) is a github repository, but no further details were given.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1382071853983469721)** (935 messages🔥🔥🔥): 

> `Kingfall release, O3 Pro benchmarks, Thinking budget for models, O3 pricing, Liquid glass design` 


- **Kingfall Release Remains Mysterious**: Members discussed the [elusive Kingfall model](https://tenor.com/view/wink-eye-wink-gif-3023120962008687924), with one user sharing an image of an iPhone 11 Pro supposedly created by **Kingfall** and calling it *some next gen stuff*.
   - Some suspect **Kingfall** might just be a nerfed version of **2.5 Flash lite**.
- **O3 Pro Benchmarks Underwhelm**: Initial [benchmarks for O3 Pro](https://cdn.discordapp.com/attachments/1340554757827461211/1382439311781400596/image.png) are out, and users are reporting *literally no difference* compared to **O3**.
   - The sentiment is that **O3 Pro** *wasn't exactly impressive*.
- **Thinking Budget's Impact Scrutinized**: A member ran a quiz **30 times** on different thinking budgets, but found *there isn't* a correlation with increased length.
   - The takeaway is that *long thinking is way more important than big models*.
- **OpenAI's Pricing Strategy Questioned**: Members debated whether **OpenAI** had been *overcharging* for **O3** and agreed the recent **5x** price reduction aligns with increased competition and **Blackwell** capacity.
   - One member quipped, *lol.. yeah. I wont being too cynical will make you depressed*.
- **Liquid Glass Design Becomes Hot Topic**: Users discussed a [liquid glass design](https://www.youtube.com/watch?v=1E3tv_3D95g) and one user stated *liquid glass will truly be the next design feature*.
   - However, there's [no consensus on whether it resembles liquid glass cleaner](https://i.imgur.com/GCzNMsh.png).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1382071824069693661)** (513 messages🔥🔥🔥): 

> `LLMs: OpenAI vs Claude vs Gemini, Cursor for non-coding tasks, Cursor Indexing Issues, O3 vs Sonnet, GPT-5 (or GPT-6)` 


- ****O3 Pro** or NOT to **O3 Pro** - the model, the pricing, the speed?**: Members discussed whether [**O3 Pro**](https://openai.com/blog/new-embedding-models-and-api-updates) is worth the extra cost compared to the regular **O3**, and users were very impressed with O3, and are ready to pay extra for pro, just to have it in Cursor.
- ****GPT-6** (and beyond!) is coming!**: After mentioning **GPT-5**, a user jokingly suggested that **GPT-6.1-pro-preview-patch** is already here!
- ****Taskmaster** can mess up subtasks!**: A user shared their experience with **Taskmaster**, noting that while it tracks tasks correctly, the subtasks can disrupt the entire process on the long run.
- ****Mistral** models are good and cheap!**: Users expressed their appreciation for **Mistral** models, noting their effectiveness as chatbots and their good coding skills, with one user highlighting its proficiency in generating physics-related content, [available for free](https://mistral.ai/).
- **Cursor UI Hides Error Messages**: Users are reporting Cursor's UI now hides tool call error messages after a second, which creates *a perceived lack of transparency*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1382082850500902953)** (13 messages🔥): 

> `Background Agents Connection Issues, Cursor vs Codex Performance, Agents.md Implementation, Incorrect Branch Errors with Fix in Cursor` 


- **Background Agents Face Connection Failures**: Several users reported issues with **background agents** failing to connect, often displaying a `ConnectError: [unknown] No response from model` message, particularly after hitting the **25 tool use limit**.
   - One user noted that the problem persists even after multiple attempts to retry, and network diagnostics return green, linking to a relevant [Discord thread](https://discord.com/channels/1074847526655643750/1380811765218283660) on the topic.
- **Cursor Outshines Codex in UE4 C++ Repo Handling**: A user lauded **Cursor's** superior performance over **Codex** in handling a **UE4 C++ repo**, citing **Codex's 80% failure rate** making it *unusable* and praised Cursor's coding enthusiasm and faster environment setup.
   - The user emphasized the speed of environment setup in Cursor, taking *less than half a minute* compared to Codex's *10 minutes*.
- **Agents.md Cheat Sheet Proposed for Cursor**: A user suggested implementing an `Agents.md` file as a cheat sheet for **Cursor's AI agents**, drawing from their experience with **Codex** to maintain complexity and speed up tasks.
   - The user mentioned that putting a cheat sheet in a file that the AI references before any task helped keep complexity down and speed up the process.
- **"Fix in Cursor" Feature Plagued by "Incorrect Branch" Errors**: Users encountered issues with the *Fix in Cursor* feature, frequently getting stuck in an *Incorrect Branch* retry loop even when on the correct branch.
   - The error persisted despite attempting to checkout the branch again, with the system indicating an *incorrect workspace*.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 messages): 

OpenAI: ### OpenAI o3-pro is rolling out now to all Pro users in ChatGPT and in the API.
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1382074156740247614)** (254 messages🔥🔥): 

> `O3 Pro rollout, Gemini YouTube analysis, KD on Reasoner Model, O3 Pro in projects, Team plan vs Pro plan` 


- **O3 Pro's Price Reduction Causes Craze**: Users discuss the **price reduction** of **O3 Pro**, with some questioning if the delay in its release was due to pricing concerns, while others speculated about the costs associated with web search functionality.
- **Gemini gets Grilled for Gaffe on YouTube Grasp**: Users reported that **Gemini** was failing to analyze YouTube videos, possibly due to being logged in with a Workspace/enterprise account or having the YouTube extension switched off, while others found Gemini can do it perfectly well.
   - One user tested with [this link](https://youtu.be/4MydzP3Mzy4?feature=shared) and had no problems.
- **O3 Pro has Problems: Prompt Engineers Post Problems**: Users subscribing to Pro plan are having problems with O3 Pro mode, seeing an error message saying *Error in message stream*, with one user sharing [this prompt](link.to.prompt) that produced the error.
   - They have been bug reporting it, supposing that the servers have been broken today.
- **Team plan teases tantalizing, but throttled, tokens**: Users discussed the **O3 message cap limit** for the Teams plan, which is only **100 per week**, a member also complained about how the team plan lacks the internal knowledge feature which helps the model grab context in milliseconds and respond back, but O3 is unlimited.
   - Another user stated that *they are intentionnally not releasing more cap limit for Teams plan for o3, because O3 + Internal sources feature is literally very intelligence*.
- **Distilling Knowledge for Frontier Reasoners**: A member is looking into upgrading a student model in their stack to a reasoner and is interested if anyone is using **KD** on a reasoner model, asking if anyone is using KD on a reasoner model.
   - They mentioned that *most frontier reasoners hide the reasoning logits and tokens I think it may need to give this up if we switch to a reasoner*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1382077259321577644)** (31 messages🔥): 

> `ChatGPT docx files, Custom GPT advantages, O3 Pro coding issues, Translating Novels` 


- **ChatGPT Teases Non-Existent DOCX Feature**: Members complain that **ChatGPT** teases users with **docx files** but the feature doesn't actually exist and the feature has been **disabled for over a month**.
   - A member stated that this is a let down, especially if you're trying to compile data, since chatgpt can't remember ten seconds prior.
- **Custom GPTs boast Advantage**: A member suggested making **custom GPTs** to get better results.
   - He stated that depending on the model version, or if you have a subscription, you have the benefits.
- **O3 Pro coding abilities worse than O3?**: A member wondered what is going on with **OpenAI** and states that **O3 Pro** was released and it's even worse than **O3** for coding and is not accurate.
   - Another member responded that based on their testing so far, it's a fantastic model, while the first member stated that **Claude 4** is 1000% better.
- **Translation tools compared for Egyptian Arabic novels**: A member asked about the best method to translate novels from **Egyptian Arabic to English** and which model to use.
   - Members suggested that **ChatGPT** is not the right tool, and that there are dedicated tools that do a way better job, and to also check out **DeepL**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1382176207784644680)** (35 messages🔥): 

> `AI system reliability, Config file for LLMs, Banning CP in AI, Prompt security, AI's moral values` 


- **User shares config file for reliable AI system**: A user shared a [config file](https://cdn.discordapp.com/attachments/1046317269069864970/1382207370427367574/config.txt?ex=684af92d&is=6849a7ad&hm=18e8864d5d0c91788c8e3e971f760cc507a8f144c51d09bc0da1d95285f30161) designed to make **LLMs more reliable**, with lower hallucination, truth anchoring, and awareness of impossible outcomes.
   - The config is intended to be used by any LLM platform, including GPT, to base user memories and get a fairly reliable AI system.
- **Debate over banning CP**: A user questioned the necessity and appropriateness of explicitly mentioning **CP** (child pornography) is banned in the config file.
   - Another user responded that while they agree that safety is paramount, they're just not sure that's it, because *enumerating forbidden tokens at the end of a config amplifies recency bias and increases the risk of LLM leakage*.
- **Safety Measures Face Scrutiny**: Concerns were raised regarding the config file's safety measures, specifically the inclusion of **forbidden topics** like bomb-making and bioweapons.
   - One user argued that *it only takes one line - the right forbidden token in the right place - to create catastrophic leakage risk*, emphasizing that risk is multiplicative, not additive.
- **Framework for AI Testing**: A user asked about testing an **A.I.** to see if the claim of **self-governing moral values** is true.
   - Another user responded that tests are needed at scale because of the diversity of human inputs.
- **AI Reply Accuracy Thresholds**: The AI's reply accuracy is validated through **Truth validation** and **Reasoning** using Thesis, Antithesis, and Synthesis.
   - If the reply falls under a **95% accuracy** threshold after the second cycle, the AI must clarify that the reply is inferred or state that it doesn't know if it remains unprovable.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1382176207784644680)** (35 messages🔥): 

> `AI Configuration, Prompt Injection, LLM safety, Adversarial Testing, Negative constraints` 


- **Configuration file sparked safety discussion**: A member shared a [configuration file](https://cdn.discordapp.com/attachments/1046317269069864970/1382207370427367574/config.txt?ex=684af92d&is=6849a7ad&hm=18e8864d5d0c91788c8e3e971f760cc507a8f144c51d09bc0da1d95285f30161&) designed to enhance AI system reliability, but this sparked some debate regarding the mention of explicit negative constraints like "CP is banned."
   - One member argued that *enumerating forbidden tokens at the end of a config amplifies recency bias and increases the risk of LLM leakage* due to the pink elephant effect.
- **Negative Instructions Cause Pink Elephant Effect**: Concerns were raised that specifying explicit forbidden content increases the risk of LLM leakage through recency bias, potentially causing the LLM to generate content it is meant to avoid.
   - It was suggested to avoid directly referencing illegal content and instead using generalized instructions, such as *The model must not generate or assist with illegal content, including but not limited to dangerous, exploitative, or otherwise restricted subjects as defined by law and platform policy*.
- **Adversarial Testing at Scale is Needed**: It was pointed out that a framework needs to face adversarial prompt injection, and *absence of evidence is not evidence of absence*, especially with emergent, low-probability, high-severity risks.
   - The standard in safety engineering is not “has it failed for me,” but “could it catastrophically fail under adversarial or unforeseen conditions?”
- **Thesis, Antithesis and Synthesis validation cycle**: A member noted that each reply from the AI is put through a **Truth validation** cycle including **Thesis**, **Antithesis** and **Synthesis**, which repeats based on the accuracy of the cycle.
   - If the reply still falls under **95% accuracy** the second time, the AI must add clearly that any reply is inferred or will say it doesn't know if it is still unproveable according to that section.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1382078105212162058)** (329 messages🔥🔥): 

> `OpenRouter OpenAI Default Reasoning Mode, O3 Pro Pricing, Assistance Needed with OpenRouter Code, LLM Recommendations for Research, OpenRouter Rate Limits` 


- **OpenRouter Reasoning Mode Defaults**: A user inquired about the default reasoning mode used for OpenAI requests through OpenRouter, specifically whether it defaults to *detailed* or *auto*.
   - The question went unanswered in the provided context.
- **Sama and Noam Tweet O3 Pro Pricing**: Users expressed interest in **O3 Pro** pricing, with one user describing **O1 Pro** pricing as *bonkers*.
   - Information on **O3** pricing reportedly came from tweets by **Sama** and **Noam Brown**.
- **Healthcare Professional Requests help with code**: A user with a medical background requested assistance with **OpenRouter** code to input **279 prompts** into various **LLMs** for a healthcare consensus project, sharing a [pastebin link](https://pastes.dev/UqlfuJgF2W) to the code.
   - Other users suggested using unique persona prompts, rate limiting, and validating CSV data to ensure questions aren't empty, and recommended that the **LLM output JSON** for easier parsing.
- **LLM Panelists Debating Models for Consensus Study**: Users discussed selecting **LLMs** for a consensus study, with suggestions to include **Gemini**, **Claude**, and **Sonar** (Perplexity), while excluding **OpenAI**, **DeepSeek**, and **Grok** due to performance gaps.
   - It was mentioned that for a fair comparison, non-reasoning models should be used, and that Qwen needs to be set to non-reasoning mode; and that [Grok had low contextual memory](https://x.com/OpenRouterAI/status/1932828433148776650) and is *really stupid*.
- **OpenRouter Has Almost Unlimited TPM Rates**: A user inquired about **TPM rate limits** imposed by **OpenAI** and whether they apply to **OpenRouter**, particularly when not using a personal OpenAI key.
   - It was clarified that **OpenRouter** has very high limits in practice, meaning *unlimited TPM* for users, and [rate limits do apply based on the model used](https://openrouter.ai/models?fmt=cards&supported_parameters=structured_outputs).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1382074957856768113)** (231 messages🔥🔥): 

> `Fine Tuning Explanation, 4090 vs 5090 for Unsloth, GRPO vs DAPO, Magistral Quants Released, DeepSeek R1 8Q Model` 


- **Unsloth Docs Explain Fine Tuning**: A member asked for an explanation of fine-tuning LLMs, and another member shared the [Unsloth fine-tuning guide](https://docs.unsloth.ai/get-started/fine-tuning-guide) for beginners.
- **5090 vs 4090 Pricing Makes People Mad**: Members discussed the absurdity of **4090** pricing, with some preferring to wait for the **5090** due to current prices, and others reporting used **4090s** being overpriced.
- **GRPO or DAPO? Does it Matter?**: Members discussed that Unsloth is using **DAPO** (Data-Aware Prefix Optimization) but calling it **GRPO** (Grouped Relative Positional Encoding), and one member suggested that they are *really close tbh*.
   - Another member mentioned that they *worked with Mistral behind the scenes to ensure everything is correct*.
- **Magistral Quants Now Available**: The UnslothAI team announced the release of **Magistral quants**, linking to [their tweet](https://x.com/UnslothAI/status/1932441885618147402).
   - They also mentioned the team *worked with Mistral behind the scenes to ensure everything is correct*.
- **DeepSeek R1 8Q Benchmark is Crazy Good**: Members are blown away by the performance of the **DeepSeek-R1-0528-UD-Q3_K_XL** model, with one reporting it holding close to **80%** on their benchmark.
   - A team member noted the test model was a few days old, from June 8, before later sharing a fixed [link](https://huggingface.co/TheBloke/DeepSeek-R1-0528-UD-Q3_K_XL-GGUF).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1382076254965989386)** (11 messages🔥): 

> `Elise-lora, NoisySpeechDetection-v0.2, Uptime` 


- **Elise-lora Usage**: A member asked whether to set `model_name="Etherll/Elise-lora"` to `unsloth/orpheus-3b-0.1-ft` or do normal SFT first and save LoRA adapters.
   - Another member suggested to do LoRA first then SFT.
- **Noisy Speech Classifier released**: A member announced the release of [NoisySpeechDetection-v0.2](https://huggingface.co/Etherll/NoisySpeechDetection-v0.2), an audio classifier trained with Unsloth based on Whisper Small.
- **Questioning Platform Uptime**: A member questioned the reality and uptime of a platform, accompanied by a [screenshot](https://cdn.discordapp.com/attachments/1179039861576056922/1382521252853584004/Screenshot_20250611-204515.png?ex=684b74c0&is=684a2340&hm=e045135237e6dd500da023be19faa4d4e06209980790a7e68e2cc55bf53c086d&).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1382072415382933584)** (63 messages🔥🔥): 

> `Ollama integration of safetensors, SSD impact with large quantized models, Unsloth models vs Normal Models, Gemma3 model family, Multilingual fine-tuning` 


- ****Ollama** Now Supports **Safetensors**, Save Your Merged Model**: Users in the community report that `save_to_gguf` is still on the roadmap and isn't working yet, but suggest that you save your merged model in **safetensors**, then use [ollama's create --quantize command](https://ollama.com/docs/guides/how-to-create-modify-prompt#quantize-your-model) to turn it into an ollama compatible format.
- ****SSD Longevity** Questioned when Running Quantized Models**: Members asked if running an r1 quant model that exceeds ram/vram would hammer the **SSD**, and if there's an easy way to split workload over multiple **SSDs** for bandwidth.
   - A member replied that streaming from **SSD** shouldn't harm it much, as it's mainly reads, but *R1 doesn't play as nice with SSD offload* as **Qwen3/Maverick/Scout**.
- ****Unsloth** Models vs. Normal Models**: A user asked what the difference between a "normal" model and a model from **Unsloth** is, specifically between `Qwen/Qwen2.5-VL-7B-Instruct` and `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`.
   - Another member explained that the **Unsloth** models are pre-quantized and smaller, while the normal models are full weights that are quantized on the fly when `load_in_4bit=True` is set.
- ****Gemma3 Family** of Models Has Issues Being Fixed**: One of the Unsloth team members noted that there are issues with the **gemma3** family of models and that they are being fixed.
   - If you're in a hurry checkout other models like the **Qwen**, **Llama** or **Mistral**.
- ****Multilingual Finetuning** advice sought**: A member asked for advice on a multilingual fine-tuning issue, where the **orpheus-3b** model was fine-tuned to support Kazakh, but the emotion tokens and speaker tokens were forgotten.
   - Another member suggested pretraining for a new language, stating that the dataset will need to have emotions, and if stuff isn't in the pretraining or distribution, it's not in there.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1382075323381977148)** (12 messages🔥): 

> `AIME 2025, Prompting Changes, GRPO Efficiency, shuffled answer ordering, LoRA vs FFT` 


- **AIME 2025 Math Competition Out Now**: The **AIME 2025** math competition is out now; more information at [ArtofProblemSolving](https://artofproblemsolving.com/wiki/index.php/2025_AIME_I).
- **Tiny Prompting Tweaks Yield Huge Accuracy Leaps**: A member shared a cool 2 year old paper on how a tiny change in prompting massively changes the accuracy on 0-shot tasks, detailed in this [paper](https://arxiv.org/pdf/2311.01967).
- **Shuffled Answers Stumps Models**: Models fail to answer many MMLU questions correctly when given questions they previously solved correctly, but with shuffled answer ordering, discussed in this [paper](https://arxiv.org/pdf/2406.19470).
- **GRPO Update Size Debate Sparks**: A member asked if GRPO+LoRA is inefficient because parameters don't change a lot.
   - Another member pointed out that the same inherent sparsity occurs for pretraining. [This paper](https://arxiv.org/abs/1803.03635) isn't unique at all to GRPO and RL in general.
- **LoRA Fares Well in RL Faceoff**: The study you're mentioning does cover that LoRA actually performs comparably to FFT for RL, as described in [this paper](https://arxiv.org/pdf/2505.11711).
   - Instead the paper advocated that their sparse update method, where only the meaningful parameters are updated is superior to both FFT and LoRA.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1382089626751205457)** (217 messages🔥🔥): 

> `Sonnet 4 with Reasoning, GRPO Length Bias, ProRL Paper, Mistral's Reasoning Mode, IQ discredited` 


- **Sonnet 4 Reasoning vanishes**: The **Sonnet 4** with reasoning test disappeared from testing, suggesting it's not available with *thinking* in **OpenRouter**.
- **GRPO's Length Bias Fixed?**: Discussion around **GRPO** reveals it has a length bias that might be addressed via fixes like **DRGRPO** or **DAPO**, with internal training environments using length penalties, described in [this github link](https://github.com/NousResearch/atropos/blob/main/environments/tool_calling_server.py#L370C1-L397C71).
- **ProRL Effect Debate heats up**: A discussion on the **ProRL** paper ([arxiv link](https://arxiv.org/abs/2505.24864)) and its applicability to larger models arises, with some questioning its conclusions, but others agreeing with the entropy collapse issues.
- **Mistral releases Magistral Reasoning, but with Rough Edges**: The new **Mistral** reasoning mode activates via its chat template ([huggingface link](https://huggingface.co/mistralai/Magistral-Small-2506)) and the *thinking* is formatted in tags, however there are issues such as models looping the final answer due to reward hacking.
- **Apple's Tim Cooke slams Reasoning LLMs; IQ Scores are Jokes**: Following **Tim Cooke's** announcement downplaying *reasoning LLMs*, one member noted that **Yann Lecun** has said recently whatever LLMs are doing and what humans are doing is still very different.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1382395360076501122)** (5 messages): 

> `Knowledge Distillation, Exposure Bias, Obsidian Loom, Obsidian Error 402` 


- **Knowledge Distillation Discussion Begins**: A member is doing distillation based on a **student -> critic -> teacher** run, storing the teacher logits and using them to fine tune the student model.
   - They are looking into upgrading a student model to a reasoner and are interested if anyone is using **KD** on a reasoner model; since most frontier reasoners hide the reasoning logits and tokens, they think it may need to give this up if switching to a reasoner.
- **Exposure Bias Concerns Arise in Knowledge Distillation**: The same member thinks the issue will be **exposure bias** to the reasoning patterns from the teacher generations compared to the student generations.
   - They are unsure how to best approach distillation cross-reasoners.
- **Obsidian Loom Connection Questioned**: A new member asked how to connect **Obsidian Loom**.
   - They stated that a friend told them that a **402 error** means they have to reinstall Obsidian.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1382289732653813800)** (1 messages): 

> `McCulloch-Pitts Model, AI Foundations, Computational Neuroscience` 


- **McCulloch-Pitts: Foundation of AI and Neuroscience**: The [McCulloch-Pitts paper](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf) is considered a foundational work of **AI** and **computational neuroscience**.
   - The model proposed in the paper is a simplified model of **neurons** and **neural networks**, capable of computing any **Turing-computable function**.
- **Turing Computability and Neural Networks**: The McCulloch-Pitts model demonstrates the theoretical capability of **neural networks** to compute any function that a **Turing machine** can compute.
   - This establishes a fundamental link between **artificial neural networks** and the theory of computation, influencing subsequent developments in both fields.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1382072016089252011)** (8 messages🔥): 

> `Frutiger Aero, AI Slop iMessage Backgrounds, AI Heart Monitoring, Le Chat exposure` 


- **Frutiger Aero Comeback Campaign**: Members expressed a desire for resources to be focused on bringing back **Frutiger Aero** and implementing **AI Slop iMessage backgrounds** instead of **AI heart monitoring** technologies.
   - One member stated *nobody cares about ai heart monitoring that would save/extend lives*.
- **Teknium Boosts Le Chat**: A member shared a link to [Teknium's Tweet](https://x.com/teknium1/status/1932677245962788982?s=46) to celebrate **Le Chat** getting exposure.
   - Another member simply stated that they knew the origin of the tweet.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1382289732653813800)** (1 messages): 

> `McCulloch-Pitts Model, AI Foundations, Computational Neuroscience` 


- **McCulloch-Pitts: Foundational AI Paper Spotted**: A member shared a link to [*A Logical Calculus of the Ideas Immanent in Nervous Activity*](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf) by Warren McCulloch and Walter Pitts.
   - This paper is considered one of the **foundational works** of **AI** and **computational neuroscience**.
- **Logical Calculus**: The paper explains that *Given the activity in any part of the nervous system, it is possible to deduce logically either a definite description of that activity*
   - *Or, if the activity be not described, to deduce an equivalent expression for any one of its adequate stimuli*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1382109921134510100)** (82 messages🔥🔥): 

> `LM Studio API Documentation, Connect LM Studio to External Ollama Instance, AI Autonomy and Decision Making, Local Image Generation Recommendations, CUDA 12.8 Runtime Speed on 4070ti` 


- **API Doc Discovery Diversion**: A user sought **Swagger documentation** for the LM Studio Server API, but was directed to the [official API documentation](https://lmstudio.ai/docs/app/api) and the **Developer Tab** within LM Studio.
   - The user was trying to structure their http requests in order to work with the LMS server.
- **Ollama Orchestration Out of Reach**: A user inquired about connecting LM Studio to an external **Ollama** instance, but was informed that **LM Studio** functions solely as a server.
   - It was suggested that they should use **Open WebUI** for such connectivity.
- **Image Generation Gems Guide Given**: A user with a **4070ti Super** asked for local image generation recommendations and was advised to start with **Fooocus**, then explore **ComfyUI**, and potentially use **Pinokio** for installing point solutions.
   - Other recommendations included using quantized **Flux.dev** models and exploring fine-tunes on **civitai**, with a preference for **sd-forge webui** over ComfyUI due to its steeper learning curve.
- **Flash Attention Foibles Found**: A user discovered that the **deepseek-r1-0528-qwen3-8b@q6_k** model exhibited issues with **flash attention**.
   - It was noted that many users encounter bugs when using flash attention, and the same applies to using **Q4 KV caches**.
- **SSD Swap Scare Story Surfaces**: A user planned to use `mmap()` swap to run a model on a machine with **32+16GB** of memory, prompting a warning that this approach could damage an **SSD**.
   - Despite the warning, the user asserted that **SSDs** are rated in terabytes written, not read, and expressed *"faith in llama.cpp"*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1382090947780808948)** (131 messages🔥🔥): 

> `Digits Memory Bandwidth vs M3 Max, Qwen 3 235b on Evo X2, Rednote dots LLM, Octa-channel RAM server, Dual Socket AMD Turin` 


- **Digits might be slower than M3 Max due to Memory Bandwidth**: It's expected that **Digits** will be slower due to lower memory bandwidth compared to the **M3 Max**, referencing a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1he2v2n/speed_test_llama3370b_on_2xrtx3090_vs_m3max_64gb/) as evidence.
   - Prompt processing speed is thought to be driven by compute power, while token generation speed is driven by memory bandwidth.
- **Qwen 3 235b's Feasibility on Evo X2 Explored**: A user inquired about running **Qwen 3 235b Unsloth** on an **Evo X2**, with another user responding that the **Q3_K_XL** quantization is the largest that can fit, achieving **10-12t/s**.
   - The setup involved using Linux, a context length of 10000, Q8 quantization for the KV cache, no mmapping, and an Evolution Batch Size of 320.
- **Rednote dots LLM Double Speed Claims**: A user mentioned the **Rednote dots LLM**, stating that it should be roughly twice as fast when built with the appropriate *llama.cpp* branch.
   - Another user confirmed that the merge had occurred recently, however another user expressed skepticism, citing the need to check the original issue.
- **High-Bandwidth CPU-Only Solution Debate**: Users discussed the feasibility of running a **150B** parameter model with a **60K** token context window at 5 tokens/sec using an Octa-channel fast RAM server rig.
   - One user shared performance data from an Intel Xeon Gold 5120 CPU with 256GB RAM, achieving only ~0.86 tokens per second for evaluation, leading to skepticism about achieving the desired performance.
- **Dual Socket AMD Turin Server**: It was mentioned that a **Dual Socket AMD Turin** server would have **1.2TB/sec** memory bandwidth and **640 GB/sec** of PCI-e.
   - Such machine with 8 dual B60s would enable **16 GPUs and 384GB VRAM**, consuming around 3.5kW. One user linked to a Supermicro motherboard [Supermicro.com](https://www.supermicro.com/en/products/motherboard/h14dsg-o-cpu) with 20 x8 slots.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1382074840499884121)** (157 messages🔥🔥): 

> `Gradio app feedback, AMD Radeon Pro SSG for AI, Inserting a sphere in depth video, LLM validation loss range, Distilled models of Qwen` 


- **Gradio App User Feedback Request**: A member sought **user feedback** on their already deployed **Gradio app** to identify what users find off-putting and what works well for improvements.
- **Mini PC RAM limitations Discussed**: Members discussed why mini PCs are limited to **128GB RAM** due to space issues and the use of soldered **LPDDR5**, questioning why these chips can't be soldered on a stick to allow for more RAM.
- **Training a small LLM with Rust gains interest**: A member inquired about the availability of a **Rust end-to-end pipeline** for training a **tiny (100M parameter) LLM**, seeking a hackable solution for all training phases.
- **Evaluating LLM Validation Loss**: A member training a ~**400M parameter text generation model** asked about a decent **validation loss range**, reporting values around **3.5** and seeking standard benchmarks for that scale.
   - One member noted the shape was basically good, and generally a lower validation loss is better, but that it is important to ensure the model performs according to expectations.
- **Exploring ZeroGPU Billing Anomalies**: A user reported being charged for **ZeroGPU usage** despite only making a few API calls, and questioned why the refresh time was significantly longer than the usual **15 minutes**.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

stark_0278: Hi guys, Im Stark. Just started learning AI Agent course.
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1382411818802086030)** (1 messages): 

> `FuriosaAI, AI Hardware, Deep Learning, New Chip Designs` 


- **FuriosaAI Grabs Attention in AI Hardware Space**: A member spotlighted [FuriosaAI](https://furiosa.ai/), an AI hardware company, noting its potential and innovation in the field.
   - The comment suggested that FuriosaAI is *one to keep an eye on*, hinting at possible advancements or disruptions in AI chip design and performance.
- **Emerging Trends in Specialized AI Accelerators**: The discussion briefly touches upon the increasing interest in specialized AI accelerators, like those developed by FuriosaAI, tailored for specific deep learning workloads.
   - This reflects a broader trend in the industry toward optimizing hardware for AI, moving beyond general-purpose GPUs to achieve greater efficiency and performance.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1382126552594124821)** (11 messages🔥): 

> `LLM Agent Framework, HyBio-Agent on Hugging Face Spaces, Qwen 3 1.7b Model Testing, Conversational Agent for PostgreSQL` 


- ****LLM Agent Framework** goes open source!**: A member open-sourced their **LLM agent framework** [available on GitHub](https://github.com/starsnatched/llm-backend), which can use a **Linux terminal on a VM** with full access, enabling it to store, modify, delete, and move files, as well as gather information from the web.
   - The author emphasized that *we will never "solve history" and quantum-proof truth like this...*
- ****HyBio-Agent** debuts on Hugging Face Spaces**: A member shared a link to **HyBio-Agent** on [Hugging Face Spaces](https://huggingface.co/spaces/Agents-MCP-Hackathon/HyBio-Agent).
   - Another member exclaimed *no way, AGI achieved!*
- ****Qwen 3 1.7b Model** works surprisingly well!**: A member shared that they tested models down to **Qwen 3 1.7b** to make a conversational agent.
   - Another member inquired if it worked well at that size and expressed interest in using their functions to fine-tune the model.
- ****SQlord**, Conversational Agent for PostgreSQL!**: A member made a conversational agent to explore any **PostgreSQL database**, tested in both **English and Portuguese**, and made it available on [Hugging Face](https://huggingface.co/spaces/Agents-MCP-Hackathon/SQlord).
   - It was also tested using the **Qwen 3 1.7b model**.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1382343355190480947)** (2 messages): 

> `Reading Group Scheduling, Paper Presentations` 


- **Reading Group Schedule in Limbo**: A member asked when the next reading group is happening and the response was that *nothing is scheduled now*.
   - However, *anyone is welcome to take the lead on presenting a paper*.
- **Paper Presentations Encouraged**: It was communicated that there is no currently scheduled reading group.
   - All members are encouraged to present a paper.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1382081843469221920)** (40 messages🔥): 

> `Codeagent capabilities, Agents Course Deadlines, Local vs Colab, OpenAI Model Costs, MCP Protocol Deployment on HuggingFace` 


- **Codeagent solves all problems**: A user found that **Codeagent's prompt** is very hard-coded with the gist of it seeming to be *"write code to solve all of your problems."
- **Agents Course deadline looms**: The deadline for the course is **July 1st** but new members can complete the first unit and get a certificate in one day.
- **Local coding vs Colab for the course**: Members are running the code locally using VSCode or using the Google Colab spaces and it was pointed out that **unit 4** works best when coded locally.
   - A member is looking for a *requirements.txt* file for the course to keep a leash on dependencies.
- **OpenAI 4o-mini costs**: One member reported spending **less than $10** for around a **score of 50** on the questions using the GPT-4o-mini model, but subsequent emails indicated **o3** is cheaper than **4o**.
   - Another member mentioned they've only spent **$0.20** troubleshooting with **Gemini flash pro**, focusing on the first **10 questions** and doing no image analysis.
- **MCP Protocol Deployment in Lecture**: A member shared a lecture on **MCP protocol deployment on Hugging Face**, showing how to add descriptive doc strings and add a code snippet to *app.py*.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1382171549095956510)** (36 messages🔥): 

> `Reservoir Computing, Wumpus World Module, Continual Learning, DAN Agent, Oscar-C project` 


- ****Reservoir Computing** Resurfaces with **Self-Organizing Reservoirs****: A member inquired about [reservoir computers](https://ollama.com/PythagoraTheorem/Aimee3) with **self-organizing reservoirs**.
   - The suggestion highlights similarities to brain function, sparking interest in alternative computational approaches.
- ****Wumpus World** implementation faces funding **fiascoes****: A member reported progress on a **Wumpus World module** for a **Unity game**, but is facing implementation issues and funding shortages for necessary **LoRA training**.
   - They shared a [link to a fun lil agent](https://drive.google.com/file/d/1cTb5g68ivazmx5iVBrwtvAaTg7I-97nm/view?usp=sharing) and expressed hope it would function as an **NPC**.
- ****Continual Learning's DL** doubt **declared****: A member voiced skepticism about **Deep Learning's** ability to achieve **continual learning**, citing its inherent memory limitations due to the i.i.d. assumption.
   - They advocated for exploring alternative methods that address these limitations, despite their complexity, pointing to a [YouTube video](https://youtu.be/AT3Tfc3Um20?si=zL_m5lrW5Yu2O6IW) on solving issues with Deep Learning.
- ****DAN Agent** does **debut****: A member shared their **DAN (Do Anything Now) agent**, which generates images, videos, a story, and narration from a single prompt, also providing a [screenshot of the agent in an Ollama link](https://ollama.com/PythagoraTheorem/Aimee3).
   - The agent features conversation mode with memory, a script improvement extension, and a terminal operator, and is designed to be community-driven with API-ready extension templates.
- ****Oscar-C project** seeks **supporters****: A member invited others to check out their project, **Oscar-C**, related to cognitive architecture, XAI, and neurosymbolic AI, but noted previous attempts to share it were labeled as shitposting.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1382103577551441940)** (22 messages🔥): 

> `BioML people in Berlin, How much do language models memorize?, Agents and World Models, Energy-Based Models, Predictive Coding and Active Inference` 


- **BioML Brains Beam into Berlin**: Some **bioML** researchers in Berlin might present in the YK discord in the future.
- **GPT Grokking Gets Gauged**: A new [paper](https://arxiv.org/pdf/2505.24832) proposes a new method for estimating how much a model *“knows”* about a datapoint, measuring the capacity of modern language models.
   - The measurements estimate that models in the **GPT family** have an approximate capacity of **3.6 bits-per-parameter**, observing that models memorize until their capacity fills, at which point *“grokking”* begins.
- **World Models Warranted for Wider Generalization?**: A [paper](https://arxiv.org/abs/2506.01622) argues that **any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment**.
   - The author writes that *this model can be extracted from the agent's policy, and that increasing the agents performance or the complexity of the goals it can achieve requires learning increasingly accurate world models.*
- **Energy-Based Explorations**: A member shared a [paper](https://arxiv.org/abs/2406.07726) about **energy-based models**, but stated that they had not looked closely at it.
- **Inference Insights Illuminate**: A member suggests that [this survey](https://arxiv.org/abs/2407.04117) is the best introduction to **energy based models**, thinking of them *like a localized gradient descent operation.*
   - They noted that *the term 'Inference Learning' might sound a bit familiar to the term 'Active Inference'* but is distinct, and if you read that survey, and the original VAE paper, you'd actually have pretty much every tool you need in order to actually implement an Active Inference model.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1382148103326535772)** (16 messages🔥): 

> `Diffusion Models, GANs for Language Modeling, V-JEPA, Mistral Compute` 


- **Sama's Unexpected Announcement Teases Diffusion Model**: Following [Sam Altman's tweet](https://x.com/sama/status/1932573231199707168) about an unexpected announcement, members speculated it might involve a **diffusion model**.
- **Gemini Diffusion Shows Promise**: One member with access to **Gemini diffusion** noted that despite its speed and apparent model size, it performs remarkably well, especially in quickly finding solutions.
   - It was mentioned it's *not great for creative tasks* because *it follows patterns a lot more than transformers*.
- **GANs Revisited for Language Modeling**: A member shared a markdown file, [Papers_or_preprints_that_did_Language_Models_with.md](https://cdn.discordapp.com/attachments/853983317044756510/1382377584989573200/Papers_or_preprints_that_did_Language_Models_with.md?ex=684aeef3&is=68499d73&hm=103bb35e5751b57b677f9388cdfad9094bdfbb3e5e5f37be5fd649aad3eb0ad8&), prompting discussions about using **GANs** for language modeling.
- **Meta Launches New V-JEPA version**: Meta AI released a new version of **V-JEPA**, aimed at advancing world model benchmarks, according to [this blog post](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) and [tweet](https://x.com/lxrjl/status/1932499153596149875).
- **Mistral Launches Compute Services**: **Mistral AI** announced **Mistral Compute**, aiming to democratize AI infrastructure and give more people the tools to build and own AI infrastructure, according to [their blog post](https://mistral.ai/news/mistral-compute).


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1382257728524779520)** (1 messages): 

> `Sparse Autoencoders, Product Key Memory Modules, PKM sparse coders` 


- **EleutherAI Releases Product Key Memory Sparse Coders**: EleutherAI released a [blogpost](https://blog.eleuther.ai/pkm-coders/), [code](https://github.com/EleutherAI/sparsify/tree/e2e-pkm) and [checkpoints](https://huggingface.co/EleutherAI/pkm-coders/) for researchers to experiment with **Product Key Memory (PKM)** modules for **sparse autoencoders** and **transcoders**.
   - The team found that **PKMs** speed up training and inference, reduce memory usage, and induce hierarchical grouping, but they are unsure whether the improvements justify the added complexity.
- **PKM Module Benefits Outlined**: **Product Key Memory (PKM)** modules accelerate training and inference, while also cutting down on memory footprint.
   - These modules encourage a hierarchical grouping structure within the latents, potentially streamlining model comprehension and efficiency.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1382071981180190761)** (38 messages🔥): 

> `Ban Data Archiving, O3 Pro Pricing, Ultrasonic Lens 3D Printing, Open Science @ CVPR` 


- **Ban Data Archiving Debated**: Members debated whether to [delete messages upon banning](https://discord.com/channels/691289280227498005/729741769738158194/1252734090107158530) and archive the data for dataset purposes, with concerns about user frustration versus the benefits of open research.
   - The conversation suggested that archived data should be **anonymized** before release to protect user privacy while still enabling AI and psychology research.
- **O3 Pro Pricing Criticized**: Members criticized the [pricing of O3 Pro](https://discord.com/channels/691289280227498005/729741769738158194/1252738340687548506), noting its input cost is **$20 / 1M tokens** and output cost is **$80 / 1M tokens**.
   - Some users humorously remarked that the pricing implied it should be able to *solve the Riemann hypothesis* and considered it *worse than muon* and *worse than RWKV*.
- **Ultrasonic Lens 3D Printing Problems**: A member requested suggestions for mitigating ridge-induced scattering when **3D printing spherical or conical acoustic lenses** for ultrasonic beam focusing.
   - The member had a feeling it is going to be an issue and other members suggested contacting the materials science department for assistance.
- **Open Science Chat at CVPR**: If anyone is at **CVPR** and wants to talk about **open science** stuff, check out [this link](https://lu.ma/z1o7ncnt).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1382241636775624755)** (28 messages🔥): 

> `Harvard Library's dataset, British Library / Library of Congress digitization, Cosine Decay` 


- **Harvard Library's dataset is on the Horizon**: A [paper](https://arxiv.org/abs/2506.08300) discusses making a set of books accessible through **Harvard Library's share of Google Books**, with a dataset covering about **1 million books** verified as public domain.
   - Members believe the dataset is largely new, and they have been eagerly awaiting it for over a year, with the code and data expected to be released soon.
- **British Library and Library of Congress Should Digitize All Books**: Members discussed about libraries like the **British Library** and **Library of Congress** that have roughly **300x more books** than Harvard's dataset, potentially offering **10T more quality tokens** for LLMs if digitized.
   - The paper mentions plans to make millions more books accessible to the public for a variety of uses.
- **Cosine Decay Deep Dive**: Members are looking for papers and intuition on **cosine decay** to a minimum value (e.g., 10% of peak) versus decaying all the way to 0, questioning if a minimum helps generalization in smaller SFT runs, discussing [this paper](https://arxiv.org/pdf/2502.15938).
   - One member suggests for small LLMs, two epochs work best (first epoch with warmup and linear decay, second epoch with cosine decay to 0), referencing [this paper](https://arxiv.org/pdf/2404.06395) and [this Meta AI research](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1382410438943183018)** (3 messages): 

> `knock off error control, AI safety, Interpretability` 


- **Ideas from Knock Off Error Control resurface**: A member suggested that ideas from **knock off error control** might be useful, sharing a link to the [Knockoff Inference](https://arxiv.org/abs/1811.06687) paper.
   - Another member expressed gratitude, stating that they haven't seen knockoffs mentioned in like **4 years** and that it's a reminder they should learn about them.
- **Importance of AI Safety Highlighted**: Discussions revolved around the critical need for **AI safety** measures in the context of rapidly advancing AI technologies.
   - Concerns were raised about potential risks and unintended consequences, emphasizing the urgency of research and development in **AI safety** protocols.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1382072746091216896)** (54 messages🔥): 

> `aider uninstall, O3 Model Price Drop, OpenRouter KYC, Kingfall Model, R1 528 vs 0506` 


- **Aider Uninstallation Antics**: A user inquired about the correct way to uninstall Aider from a Linux machine after using `pip install aider-install && aider-install` and found that `pip uninstall aider-chat` left the binary in `/local/bin`.
   - The user was aware that they could manually delete the binary, indexes, and cache files, but was wondering if there was a better way or if they were missing something.
- **O3 Model Price Drops Drastically, KYC Still Required**: Members noted that the **O3 model** experienced an **80% price drop**, resulting in a cost of **$2 input** and **$8 output** but [OpenRouter](https://openrouter.ai/) still requires users to bring their own key and KYC.
   - One member expressed dissatisfaction, saying, *"Why does openrouter not have o3 flex? Still requires KYC to use though. Sadge"*.
- **Kingfall Model Performance Comparisons Arise**: A user compared **0605 (32k)** vs **Kingfall (auto thinking)** showing that **Kingfall** performed much better, and it looks like *they tried going for the same thing but kingfall was ALOT better at it*.
   - However, another user contested these numbers and stated that they are *ridiculous numbers*, and might be an [OpenRouter bug](https://openrouter.ai/).
- **R1528 Benchmarks Disappointing**: Members discussed the performance of the new **R1 model**, with one suggesting the new **R1** is better at **$4.8**, but another countered that they believe the new **R1** is even worse than `0506` but *it's cheaper, so it's probably the best in terms of cost effiencey*.
   - One member expressed skepticism about current benchmarks, stating, *Nearly all benchmarks are opne souecemeaning that ai companies can train on the bench*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1382096352753877082)** (12 messages🔥): 

> `Aider Pro Max Subscription Usage, Aider LLM Leaderboard by Language, Deepseek-R1 Configuration with Aider, O3 with new pricing, Aider planning mode` 


- **Aider User Maxes out Pro Max Subscription**: One user mentioned they are using a fair bit of the **Pro Max subscription**, to the point that they get usage warnings due to running several **Claude code instances** in parallel.
   - The user jokingly admitted to *kinda asking for it though*.
- **Aider LLM Leaderboard Seeks Per-Language View**: A user inquired about the possibility of viewing the [Aider LLM leaderboard](https://aider.chat/docs/leaderboards/) on a **per-language basis** to assess model performance with specific languages.
   - Another user then shared another [LLM leaderboard](https://leaderboard.techfren.net/).
- **Deepseek-R1 Configuration Troubles**: A user is experiencing difficulties configuring **deepseek-r1** from chutes within Aider.
   - Another user suggested setting up a [.env file](https://aider.chat/docs/config/dotenv.html) to configure it.
- **O3 Pricing Challenges in Aider**: A user is seeking advice on using **o3** with the new pricing structure in Aider, facing issues with **OpenAI API Tier 2** access and **OpenRouter** requirements.
   - Another user mentioned that *openai makes you show them your passport before they let you use `o3`*.
- **Desire for Aider Planning Mode**: A user suggested implementing a **/planning mode** to enhance Aider's automation capabilities, envisioning it as a taskmaster.
   - Another user shared a link to the [OpenAI background guide](https://platform.openai.com/docs/guides/background).


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1382073660646359183)** (8 messages🔥): 

> `Conversational Style Tailoring, AI Overview Audio Generator, Pierceday Metalabel` 


- **Tailoring Conversational Style**: A user inquired about tailoring the conversational style of AI responses for specific demographics like **age, ethnicity, and gender**, assuming the availability of relevant source information.
   - They requested comments and experiences from others on this subject, showcasing interest in **personalized AI interactions**.
- **AI Audio generator, newbies seek help**: A new user requested assistance on how to configure the AI overview audio generator to produce **separate overviews for each topic** within a source.
   - The user mentioned difficulty in locating the **customize** option referenced in instructions.
- **Pierceday Metalabel for manuals and reminders**: A user shared the [Pierceday Metalabel link](https://pierceday.metalabel.com/aphone) and possible use-cases.
   - Use cases included car manuals, maintenance details, electrical box notes, and conference presentations


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1382075973595435059)** (51 messages🔥): 

> `Podcast Generation Length, Customize Audio Overview, NotebookLM Limitations, Spreadsheet Support, Retrieve Old Output` 


- **Unlock Podcast Generation Secrets**: Users are exploring how to generate **podcasts longer than 20 minutes** using NotebookLM, inspired by online examples.
   - One user noted difficulty customizing the audio overview in the app.
- **Lacking Spreadsheets on NLM**: Users are requesting support for **spreadsheets** (.xlsx) in NotebookLM, noting that while Google Docs and Slides are supported, spreadsheets are strangely absent.
   - It was further noted that .xlsx files *are* supported in Gemini and AI Studio, but not in NotebookLM.
- **Unearth D&D Campaign Consistency with NotebookLM**: A user is leveraging NotebookLM to maintain consistency in their **D&D campaign**, uploading written content and session notes.
   - They were impressed by the automatically generated audio overview but are facing issues with prompt accuracy in specifying session ranges and want to know if [there's a better way](https://www.example.com) to get NotebookLM to cover only what they want.
- **Pinpoint Gemini Model NLM Runs On**: Members are curious about which **Gemini model** NotebookLM is running on.
   - Some believe it's the newest release of **2.5 Flash**, while others point out that only Geminio Pro 1.5 has a 2B token window.
- **LaTeXLM chrome extension released**: A user created an [open source chrome extension](https://github.com/hachoj/LaTeXLM) to have **MathJax rendering** on NotebookLM.
   - The extension is available on GitHub, and they may publish it to the Chrome Web Store later.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1382080361252651109)** (56 messages🔥🔥): 

> `Cursor's Anthropic Investment, DeepSeek Model's Narrative, Windsurf's 'Plan Mode', Altman's blog, Open-Weights Model Delay` 


- ****Cursor** May **Double Down** with **Anthropic****: **Cursor** might be doubling down with **Anthropic** now that **Windsurf** is with **OpenAI**, raising questions about a potential deal on the horizon.
   - Members believe **Cursor** is more invested in **Claude Code** than **OpenAI** was with **Codex**, with some users noting **Claude Code** was the first workflow that *clicked* for them.
- ****Apple** Rumored to **Acquire** Anthropic for **Siri** Help**: Rumors and speculation suggest **Apple** should acquire **Anthropic** because they can afford it and *god knows they need the help* with **Siri**.
   - One member noted that **Siri** *can’t reliably send a text message with a 15-year head start*.
- ****Altman** Teases **Gentle Singularity** on his Blog**: A link to [Sam Altman's blog post titled *The Gentle Singularity*](https://blog.samaltman.com/the-gentle-singularity) was shared, sparking interest and discussion.
   - The link was shared alongside a link to [Kevin Hou's X post](https://x.com/kevinhou22/status/1932516093333266538?s=46).
- ****Windsurf Launches New 'Plan Mode'** for Task Management**: **Windsurf** launched a new **'Plan Mode'** feature, enabling the **AI agent** to perform complex tasks by creating and maintaining a planning document, available for free on [Windsurf.com](https://windsurf.com/).
   - One user who tested it on a small greenfield project reported that it *worked well*.
- ****Sharon Zhou** Joins **AMD** to Democratize GPUs for AI**: Sharon Zhou announced her move to **AMD**, joining **Lisa Su** to focus on **AI research** and teaching, bringing colleagues from **LaminiAI** to **AMD**.
   - She aims to **democratize GPUs** and **scale AI**, with mentions of her attendance at the **#AdvancingAI** conference, as posted in [this X post](https://x.com/realSharonZhou/status/1932817096510931380).


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1382088512794525826)** (46 messages🔥): 

> `MCP server integration, iframe or frontend API, webio transport, OAuth2 UI authentication, headless agents` 


- **Hume AI's Evals Attract Interest**: Hume AI posted about their approach to "evals" for their MCP server in a [blog post](https://www.hume.ai/blog/roleplays-evals-hume-mcp-server), sparking curiosity about how others are evaluating their systems.
- **MCP Server as iframe Gains Traction**: The idea of running an MCP server as an iframe or frontend API is gaining support, with discussion around the potential benefits of using a **webio transport** similar to the existing stdio transport.
   - It may solve current MCP issues by allowing custom UI and **OAuth flows** via `window.open`, and simplifying setup with URL copy-pasting, while also managing a virtual filesystem via web APIs rather than granting shell access.
- **OAuth2 UI Authentication Eases MCP Integration**: Using real **OAuth2 UI** for authentication would greatly simplify connecting services like Google and GitHub for regular users, providing a much better user experience for MCP servers.
   - Ideally, OpenAI, Anthropic, and Google would offer OAuth2 login for their APIs, further streamlining the process.
- **Reverse Proxy Adds Security Issues**: While coding a reverse proxy could be an engineering exercise, it may reintroduce some of the security issues that the iframe/webio approach aims to solve.
   - A local service could receive messages from the browser and forward them as an MCP server.
- **Hugging Face Jumps into the MCP Game!**: Hugging Face now has **MCP** as evidenced by a screenshot, which is great news for those involved in model development.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1382079717787828346)** (5 messages): 

> `MCP Server for Slides.com, Glama Troubleshooting, Glama sandbox` 


- **MCP Server Creates Slides.com Presentations**: A hosted MCP server has been published for creating [slides.com presentations](https://www.epicai.pro/use-ai-to-create-presentations-with-mcp-tsb4j).
- **Troubleshooting Glama with "Testing" State**: A member reported being stuck in a `testing` state for a day without outputting any logs in **Glama**, despite it working locally.
- **Glama Sandbox runs ok**: The member noted that the **Glama sandbox** can connect locally and works fine as-is.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1382072011228057701)** (37 messages🔥): 

> `edu emails, ReactNexus, Veo 3 cost, Project Building Competition, Manus Chat Mode` 


- **edu emails versus ac.uk domains**: A member asked whether *edu* email domains include *.UK* (*ac.uk*) domains.
   - A member said they don't see *UK* domains being included, according to a screenshot.
- **ReactNexus Happening in July 2025**: A member asked if anyone is attending the **ReactNexus** event ([https://reactnexus.com/](https://reactnexus.com/)) happening at **J N Tata Auditorium** between **July 3-5, 2025**.
   - This conference is focused on React, a popular JavaScript library for building user interfaces.
- **Veo 3 video price causes sticker shock**: A member complained that one **Veo3 video** costs **300 credits** and that they had **38 clips** that cost them.
   - Another member said that the pricing was very expensive and asked about **Veo3** becoming available for everyone.
- **Manus Chat Mode Launches**: Manus launched a **FREE & UNLIMITED Chat Mode** for all users, enabling them to ask any question and get instant answers.
   - Users can upgrade to **Agent Mode** for more advanced capabilities, such as creating comprehensive output.
- **High Effort Mode Disappears**: Several users reported that **High Effort Mode** had disappeared from their **Pro accounts**.
   - One user noted they've "*never understood why high effort mode has to be manually selected in the first place*".


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1382301980235661332)** (12 messages🔥): 

> `Tinygrad operations directory, Selling Tinybox Green 6x 4090, AGI in a year or two, n8n vs Claude code, CS degree` 


- **Tinygrad ops move to separate directory**: Tinygrad is [moving operations](https://xl0.github.io/tinygrad-notes/bounty1.html) to a separate directory.
- **Selling a Used Tinybox Green**: A member is selling a used **Tinybox Green 6X 4090** in perfect working condition from a data center, listed at **70%** of its original price.
   - Another member expressed *interest*.
- **Members ponder AGI in one to two years**: One member asked about **AGI** in a year or two.
- **Members compare N8N to Claude for building agents**: One member inquired about using **n8n** vs **Claude code** to build agents, wondering if there are limitations to what n8n can't do that Claude can.
- **Discord users question the value of CS degree**: A member asks whether *a CS degree is still useful.*
   - Another member replied that it is *not if you have to ask*.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1382211365984276540)** (24 messages🔥): 

> `Micro Benchmarks, lovely_XXX author, linalg.svd bounty, Jacobi algorithm, Tensor.norm()` 


- **Micro Benchmarking Questioned**: A member asked about the best practice for micro benchmarks, showing an example using **Timing** and **Tensor.randperm**.
   - Another member suggested reading the chat logs for relevant experiences.
- **lovely_XXX author**: A member thanked the author of **lovely_XXX** for their helpfulness in Jupyter Notebook.
   - The author responded with gratitude.
- **SVD bounty**: The discussion covered whether to handcode functions in tinygrad for the **linalg.svd bounty** to ensure 0 dependencies.
   - LLM suggested LAPACK uses different algorithms for calculating the eigenvalues/vectors depending on the type of matrix (General, symmetric, Hermitian) which may also need reimplementing in Tinygrad.
- **Discussion on SVD Eigenvalue Computation**: The use of the Jacobi algorithm for computing eigenvalues was proposed for the **linalg.svd bounty**.
   - Someone suggested using modified versions of the Jacobi algorithm: [A Novel Fully Hardware-Implemented SVD Solver Based on Ultra-Parallel BCV Jacobi Algorithm](https://cdn.discordapp.com/attachments/1070745817025106080/1382505351634616381/A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf?ex=684b65f1&is=684a1471&hm=3eeaa1287761b9210d1e4a54b7c65b1be2a3c4b3838d55d14e60ca76d8cbefc7&).
- **Missing Tensor.norm() and LLM Solution Explored**: A member inquired about the existence of **Tensor.norm()** and suggested feeding Discord chat/codebase into an LLM like unblocked.com to answer questions.
   - It was suggested to feed the transcript for the Monday morning meetings.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1382412757139197972)** (1 messages): 

> `Modular Platform on AMD GPUs, Mammoth for GenAI Inference, Mojo in Python, Free AMD Compute from TensorWave` 


- **Modular Fuels AMD GPUs**: Modular Platform is now generally available on **AMD InstinctTM MI300X** and **MI325 GPUs**, showing up to **53%** better throughput on prefill-heavy **BF16** workflows.
   - Check out the [full blog post](https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus) for details on combining best-in-class compute with developer-friendly software.
- **Mammoth Scales GenAI Inference**: **Mammoth**, Modular's new **Kubernetes-native system**, scales **GenAI inference** across any GPU, deploying Hugging Face models across **AMD** and **NVIDIA** from a single container, without manual configuration.
   - Those interested can [learn more and join the public preview](https://www.modular.com/blog/introducing-mammoth-enterprise-scale-genai-deployments-made-simple) to explore its capabilities.
- **Mojo Invades Python Workflows**: Mojo kernels can now be directly integrated into Python workflows, available in nightly builds and supported by **450k+ lines of open source Mojo kernel code**.
   - Developers can [get started here](https://docs.modular.com/mojo/manual/python/mojo-from-python/) to begin using Mojo within Python environments.
- **TensorWave Grants Free AMD Compute**: Thanks to a partnership with **TensorWave**, users can test Modular Platform in real workloads using free AMD compute.
   - Those interested can access this offer at [Modular.com/tensorwave](https://www.modular.com/tensorwave) to evaluate the platform's performance.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1382188455244726434)** (19 messages🔥): 

> `Cross-compilation with Mojo, Mojo GPU coding on Runpod, Variadic types in Mojo, Implicit variable declaration in Mojo, Mojo documentation discrepancies` 


- **Mojo Cross-Compilation Crosses Platforms?**: A member found that direct cross-platform static compilation from macOS to Linux is currently not supported in Mojo, encountering an *'apple-m1' is not a recognized processor'* error.
   - As an alternative, they are exploring bundling dependencies and `.mojo` files to run within a **Docker container** on a serverless platform.
- **Runpod-io Runs Mojo GPU Code!**: A member managed to get a minimal implementation for running Mojo GPU code working on [runpod.io](https://runpod.io), reporting good performance with fast hot executions.
   - The only snag is a slow **cold start time of around 10 seconds**, with plans to share a setup post on the forums.
- **Mapping Variadic Types Missed?**: A member inquired about mapping between variadic types in Mojo and opened a thread on the [Modular Forum](https://forum.modular.com/t/map-variadic-types/1638?u=emil.martens) to discuss the topic.
   - No solution was mentioned, but this seems to be an open area of interest.
- **Mojo Changelog Mixup: Implicit or Explicit?**: A discussion arose regarding a potential typo in the Mojo release notes concerning implicit variable declarations, with some interpreting it as a mix of explicit and implicit type declarations.
   - The changelog will be clarified to address the confusion around whether variables are implicitly or explicitly declared, highlighting the nuanced use of the `var` keyword.
- **Docs Don't Jive: Stable vs Nightly?**: A member reported an error encountered while using a code example from the Mojo documentation on lifetimes, specifically related to the use of `ref` with `__getitem__()`.
   - It was clarified that the documentation likely reflects the **nightly build** and may not align with the stable release, advising the member to use the nightly build for compatibility.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1382523524274454559)** (6 messages): 

> `Peak Memory Usage, Flex Attention, FSDP` 


- **Peak Memory Usage in Flex Attention**: A user asked what would cause higher peak memory usage in a `(bs, seqlen*8)` input versus a `(bs*8, seqlen)` input, e.g. `(1, 64k)` uses more memory than `(8, 8k)` when using **flex attention**.
   - The user suspects the softmax square matrix in self-attention, but thought that the **flash/flex attention tiling** dealt with that and didn't require materializing fully.
- **Flex Attention and FSDP Investigation**: A user is investigating peak memory usage with **flex attention** and **FSDP** to replicate the issue on `main`.
   - They are running a sweep (e.g. 8x1k, 4x2k, 2x4k, etc.) noting that the peak memory use was constant until a "tipping point" where it jumped rapidly.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1382138079875432499)** (11 messages🔥): 

> `Qwen2, Tokenizer Integration, C4 Experiments, Iterable Datasets` 


- **Qwen2 needs a checkup**: A member will check **Qwen2** to see if a particular issue exists there also, without stating what that issue is.
   - The member expressed doubt that it makes a big difference, but they want to fix it either way.
- **Tokenizer Integration Improvements Incoming**: A member will iterate on [#2574](https://github.com/pytorch/torchtune/pull/2574) and [#2794](https://github.com/pytorch/torchtune/pull/2794) to improve the new **HF tokenizer** and its integration.
   - The member also plans to open a pull request to fix [#2809](https://github.com/pytorch/torchtune/pull/2809) and has started some experiments with **C4**.
- **Iterable Datasets Packed Up For Refactor**: A member proposed a packing refactor to work with iterable datasets to support packing for **DPO**, **GRPO**, **multimodal**, etc, see [Proposal on packing refactor](https://github.com/pytorch/torchtune/pull/2819).
   - It seems like these packing changes will make it easier to support a wider variety of configurations for our models.


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1382420202465267782)** (3 messages): 

> `Nemo RL Plans, vllm-project` 


- **Nemo RL Plans Exposed**: The plans for **Nemo RL** were [leaked](https://cdn.discordapp.com/attachments/1360680363885854841/1382420201823404032/AP1GczOQioexSd_ieqkppCKoVizt91prnymZ_uGi6mCeQdrSJE65osblAXMqxQw3030-h2272-s-no-gm.png?ex=684b16a4&is=6849c524&hm=8ca8961e205603c01114bb66f46acb3bb86d01b2d1297bef22f7817f5b6efeca&).
   - They appear to be from the Databricks conference.
- **vllm-project in the works**: There is a new [vllm-project](https://github.com/vllm-project/vllm/pull/18745) in the works.
   - This appears to be a pull request.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1382187179526062111)** (13 messages🔥): 

> `Cohere's pricing, n8n vs Claude code for building agents, Multi-modal re-ranker, CLIP and openCLIP` 


- **Cohere's Creative Costs Criticized**: A member expressed that [**Cohere**](https://cohere.com) is great for creative writing, but its **pricing is insane**.
- **n8n Navigates Node-Based Agent Automation**: Members discussed using **n8n vs Claude code** to build agents and noted that [Claude code](https://www.anthropic.com/product) is for generating and understanding code, while [n8n](https://n8n.io/) is for making workflow automations.
   - A member mentioned that **Cohere** has been working on its own agent, **North**, currently in beta.
- **Multi-Modal Re-Ranker Rumors Refuted**: A member inquired about the release of a **multi-modal re-ranker** and the recommended approach to re-rank images, but was informed that **Cohere** does not currently offer one.
   - A member suggested using **GPT-4.1** with structured output, while another suggested looking into **CLIP** and **openCLIP**.


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1382137971301548104)** (1 messages): 

> `Cohere North, EnsembleHP Partnership, AI Agents Platform, Healthcare Industry` 


- **Cohere Heads North with EnsembleHP**: Cohere is partnering with **EnsembleHP** to bring **Cohere North** to the healthcare industry, aiming to reduce administrative friction and improve patient experience with their secure **AI agents platform**.
   - Further details can be found on the [Cohere blog](https://cohere.com/blog/ensemble-partnership).
- **AI Agents Secure Healthcare Partnership**: Through the **EnsembleHP** partnership, Cohere plans to reduce friction and elevate patient experience.
   - The secure AI Agents platform may reduce administrative overhead for hospitals.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1382096191231098931)** (2 messages): 

> `Cohere API tiers, Reranking API latency` 


- **Cohere Opts Against API Tiers**: A user asked if **Cohere** has API tiers similar to **OpenAI**, but was informed that they don't offer tiers.
   - However, a member mentioned that they can offer other solutions and directed the user to contact [carolyn@cohere.com](mailto:carolyn@cohere.com).
- **Reranking API Latency Troubleshoot**: A user reported a **2-second latency** with the reranking API and inquired about potential improvements.
   - In response, a member suggested emailing [carolyn@cohere.com](mailto:carolyn@cohere.com) for alternative solutions, implying potential optimizations or workarounds exist outside of tiered API access.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1382148761173885019)** (2 messages): 

> `Vitalops, Datatune` 


- **Datatune Founder Opens Vitalops**: The co-founder of **Vitalops** introduced their open-source tool, [Datatune](https://github.com/vitalops/datatune), which performs data transformations using plain natural language.
   - They expressed excitement about joining the community and learning more from its members.
- **User joins Cohere's Discord server**: A user introduces themselves to the Cohere Discord server as part of the welcome message.
   - They are excited to be part of the community.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1382396834130952253)** (1 messages): 

> `CleanlabAI, LlamaIndex` 


- **Cleanlab and LlamaIndex team up**: [CleanlabAI](https://cleanlab.ai/) and [LlamaIndex](https://www.llamaindex.ai/) have integrated to build AI knowledge assistants and production agents that generate insights from enterprise data and make their responses more trustworthy.
   - Together, they can score trust for every **LLM response** and catch [hallucinations](https://t.co/pTjn642OUO).
- **LlamaIndex announces Cleanlab integration**: LlamaIndex [announced on Twitter](https://twitter.com/llama_index/status/1932837489238290941) a new integration with CleanlabAI.
   - The integration aims to enhance the trustworthiness of **LLM responses** generated by LlamaIndex.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1382121274851196958)** (11 messages🔥): 

> `Chainlit Decommission, LlamaIndex + Chainlit, AI Security Webinar` 


- **Chainlit Chopped: Community Clamors for LlamaIndex to Claim Code!**: Users are urging LlamaIndex to acquire [Chainlit](https://github.com/Chainlit/chainlit) as it is being decommissioned, highlighting its significance in the LLM ecosystem and seamless integration with LlamaIndex.
   - One member noted that Chainlit folks were very accomodating in supporting every release of LlamaIndex, emphasizing that *LlamaIndex + Chainlit works amazing!*
- **Chainlit Championed: Coding Community Craves Continued Contributions!**: Users champion Chainlit for its pure Python implementation, ease of deployment across platforms like Discord, Microsoft Teams, Slack, and its ChatGPT-like UI.
   - As one member stated, *Chainlit is like JavaScript in that it's all event-listeners in their programming (decorated functions)* and praised its usability: *I use Chainlit as the frontend layer for all my production apps, and for all my demos in my Medium articles.*
- **Hacken's Hotfix: How to Handle AI Hazards!**: Hacken is hosting a webinar on **June 12 at 13:00 UTC** about **AI security**, exploring **LLM vulnerabilities** and defenses, featuring Stephen Ajayi.
   - Interested parties can find more information and register through the [Luma link](https://lu.ma/xl53xbfs).


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1382339683580772413)** (2 messages): 

> `Gemini Fullstack Langgraph, DSPy Refactor, Agentic Patterns with DSPy` 


- **Gemini Fullstack Langgraph Quickstart Released**: Google recently released a full-stack implementation of a comprehensive research app called [gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart).
   - A member mentioned that it's *a very good* implementation.
- **DSPy Refactors Gemini LangGraph**: A member refactored the LangGraph portions of the **Gemini** code with **DSPy** and implemented a simple **React** front end, available [on GitHub](https://github.com/pgurazada/deep-research.git).
   - The refactored workflow is only **200 lines long** ([workflow.py](https://github.com/pgurazada/deep-research/blob/main/backend/agent/workflow.py)) and *elegantly implements the original Langgraph workflow with much lesser hassle*.
- **DSPy's Agentic Pattern Power**: A member has implemented *so many* **agentic patterns** with **DSPy**.
   - They were *blown away by how powerful the primitives are*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1382111367020023898)** (6 messages): 

> `DSPy Dataset Creation Tools, DSPy and New Reasoning Models, DSPy 3.0` 


- **Data Set Dev Tools for DSPy**: A member inquired about tools to easily build and export datasets for **DSPy**, facilitating synthetic example generation and manual labeling.
   - Another suggested that a custom **Streamlit app** could be effective, and coding agents like **Cline** can assist in its creation with minimal guidance.
- **Reasoning Models Compatibility with DSPy**: A member asked about **DSPy's** compatibility with new reasoning models that utilize tool-calling in the reasoning process, such as **o3 / o3-pro / o4-mini**.
   - They noted that while `dspy.ReACT` exists, it seems designed for the chat API era rather than the responses API era with tool-calling integrated.
- **DSPy 3.0 is Coming**: A member announced the upcoming **DSPy 3.0** release and linked to the [DSPy 3.0.0b1 release tag](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1).
   - They asked if there's a comprehensive overview of what's to come in **DSPy 3.0**.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1382083228298645605)** (4 messages): 

> `Python SDK Update, Mistral's Magistral Small Support` 


- **Python SDK Update Anticipation**: Members expressed interest in a coming update on the **Python SDK**.
   - No specific details were provided about the update.
- **GPT4All considers Magistral Small?**: A member inquired about whether **GPT4All** will support **Mistral's Magistral Small**.
   - Another member suggested using **JAN**, **LM-Studio**, **obadooga**, or **koboldcpp** as alternatives, while the original inquirer indicated they would wait, citing model speed concerns.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1382511149341212753)** (1 messages): 

> `AgentX summit, Research track, Summit Registration` 


- **AgentX Summit Paper Submission Clarification**: A member inquired whether submitting a paper to the **Research Track** competition automatically enters it into consideration for the **AgentX summit's call for papers and proposals**.
   - They sought clarity on whether a separate submission is required for the summit.
- **AgentX Summit Finalist Registration**: The member also questioned if finalists are required to register for the summit to attend.
   - They expressed concern that tickets might sell out before the competition results are released, potentially preventing attendance if not selected as a finalist.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1382190233457529014)** (1 messages): 

> `Cerebras Tech Talk, Alibaba’s Qwen 3 series, AI workshop` 


- **Cerebras Hosts Tech Talk**: Cerebras is hosting a free AI workshop this **Friday, June 13th**, from **12:00–1:00PM PST** featuring speakers such as Daria Soboleva from Cerebras, Aran Komatsuzaki, and George Cameron from Artificial Analysis. 
   - The talk will cover topics from new models like **Alibaba’s Qwen 3 series** to model selection strategies for various project types, with [RSVP here](https://lu.ma/7f32yy6i?tk=jTLuIY&utm_source=ella).
- **AI Workshop**: The AI Workshop dives into current interesting research.
   - Researchers will show you how to pick the right model.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1382431878916739172)** (1 messages): 

> `Windsurf Browser, Windsurf Wave 10, Windsurf social links` 


- **Windsurf launches Browser**: Windsurf is shipping a new **fully functional browser** that bridges the gap between your development workflow and web-based activities, as part of [Windsurf Wave 10 - Day 2](https://windsurf.com/blog/windsurf-wave-10-browser).
- **Windsurf Browser available to all users**: The new **Windsurf Browser** is rolling out in beta to all **Free, Pro, and Teams users**, while Enterprise users will receive this on a rolling basis.
   - Watch the [video on Youtube](https://youtu.be/r4WqTyLb4Vk?si=lNo4aMCIg8tHsVAp), read the [changelog](https://windsurf.com/changelog) or join the [conversation at r/Windsurf](https://reddit.com/r/windsurf).
- **Follow Windsurf on Social Media**: Follow Windsurf on [X/Twitter](https://x.com/windsurf_ai/status/1932871558219117022), [Bluesky](https://bsky.app/profile/windsurfai.bsky.social), [Threads](https://www.threads.com/@windsurf_ai/post/DKxShipsbPk?hl=en), [Instagram](https://www.instagram.com/p/DKxWKKkxvu6/) and [Linkedin](https://www.linkedin.com/feed/update/urn:li:activity:7338638111393886211/).


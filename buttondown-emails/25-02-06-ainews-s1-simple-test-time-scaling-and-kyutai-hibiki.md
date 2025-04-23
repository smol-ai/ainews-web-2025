---
id: 2d857dce-c18a-4052-b410-8600e1f0a510
title: '[AINews] s1: Simple test-time scaling (and Kyutai Hibiki)'
date: '2025-02-07T03:47:44.376508Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-s1-simple-test-time-scaling-and-kyutai
---

<!-- buttondown-editor-mode: plaintext -->**"Wait" is all you need.**

> AI News for 2/5/2025-2/6/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**210** channels, and **4396** messages) for you. Estimated reading time saved (at 200wpm): **490 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We're regrettably late to covering this paper, but late is better than never. [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) documents a new reasoning model with 2 novel contributions:

- finetuned from Qwen 2.5 32B on **just 1000 questions paired with reasoning traces** distilled from Gemini 2.0 Flash Thinking, filtered for difficulty, diversity, and quality (26 mins of training on 16 H100s)
- controllable test-time compute by either forcefully terminating the model's thinking process or lengthening it by appending "Wait" multiple times to the model's generation when it tries to end.
![image.png](https://assets.buttondown.email/images/614feebc-4fbf-4b51-9b55-5eb06ab593ac.png?w=960&fit=max)

Lead author [Niklas Muennighoff](https://scholar.google.com/citations?user=Me0IoRMAAAAJ&hl=en), who notably worked on Bloom, StarCoder, MTEB, and contributed to BIG-bench, [notes](https://x.com/Muennighoff/status/1886405528777073134) that this second trick reproduces the famous o1 scaling chart:

![image.png](https://assets.buttondown.email/images/bc28620b-478b-4847-bddb-df5360b4c34f.png?w=960&fit=max)

Compared to Bespoke-Stratos ([our coverage here](https://buttondown.com/ainews/archive/ainews-bespoke-stratos-sky-t1-the-vicunaalpaca/)), the filtering is also remarkably sample efficient.

![image.png](https://assets.buttondown.email/images/fbeac362-f99f-494e-8eec-7f8e6521133c.png?w=960&fit=max)

We would also recommend [Simonw](https://simonwillison.net/2025/Feb/5/s1-the-6-r1-competitor/) and [Tim Kellogg](https://timkellogg.me/blog/2025/02/03/s1)'s explainers.

**Honorable mention today:**

**Kyutai Moshi** made a splash last year ([our coverage here](https://buttondown.com/ainews/archive/ainews-o1-destroys-lmsys-arena-qwen-25-kyutai/)) for its realtime voice with inner monologue, and now Hibiki shows [very impressive French-English live translation offline on an iPhone](https://www.reddit.com/r/LocalLLaMA/comments/1ij35u7/hibiki_by_kyutai_a_simultaneous_speechtospeech/). Not bad for an [intern project](https://x.com/kyutai_labs/status/1887495511474573517).

![image.png](https://assets.buttondown.email/images/1667f4f3-6237-4c67-b0db-dab1f5ba9f0a.png?w=960&fit=max)



---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**AI Models and Releases**

- **DeepSeek R1 and R3 Open Source Release**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1887403244046995458) announced that **R1-low-mid-high models are coming soon**, potentially marking **the first real Open Source moment in LLMs** comparable to **nginx, Blender, or even Linux**. This release could **flatten the market owned by a cartel of incumbents with proprietary tech**.

- **Hugging Face Releases SmolLM2**: [@_akhaliq](https://twitter.com/_akhaliq/status/1887371050628903065) shared that **Hugging Face announced SmolLM2**, detailed in the paper **"When Smol Goes Big -- Data-Centric Training of a Small Language Model"**. [@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1887500167055560922) provided a breakdown of the **SmolLM2 paper**, emphasizing that **data is the secret sauce behind strong performance** in small LMs.

- **IBM's Granite-Vision-3.1-2B Model**: [@mervenoyann](https://twitter.com/mervenoyann/status/1887521464292614382) discussed the release of **Granite-Vision-3.1-2B**, a **small vision language model with impressive performance** on various tasks. A **notebook** is available to **test the model**.

**AI Research Papers and Findings**

- **LIMO: Less is More for Reasoning**: [@_akhaliq](https://twitter.com/_akhaliq/status/1887372529112686810) highlighted **LIMO**, showing that **complex reasoning capabilities can emerge through minimal but precisely crafted demonstrations**. [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1887353699644940456) noted that **LIMO achieves 57.1% accuracy on AIME and 94.8% on MATH with only 817 training samples**, significantly outperforming previous approaches.

- **Token-Assisted Reasoning**: [@_akhaliq](https://twitter.com/_akhaliq/status/1887373223152492665) shared insights from the paper **"Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning"**, discussing how combining latent and text tokens can enhance reasoning in language models.

- **Advancements in Long Chains of Thought**: [@gneubig](https://twitter.com/gneubig/status/1887495037820567815) presented research providing insights on **short vs. long chains of thought**, the role of **supervised fine-tuning vs. reinforcement learning**, and methods to **control reasoning length** in language models.

**AI Tools and Platforms**

- **Gradio DualVision App**: [@_akhaliq](https://twitter.com/_akhaliq/status/1887377041634316492) introduced **DualVision**, a **Gradio template app for image processing** featuring **multi-modal predictions**, **GPU support**, and an **examples gallery** for enhanced user experience.

- **Le Chat by Mistral AI Now on Mobile**: [@sophiamyang](https://twitter.com/sophiamyang/status/1887517050697842899) announced the release of **Le Chat**, an AI assistant by **Mistral AI**, now available on **mobile platforms** with features like **code interpreter** and **blazing-fast responses** powered by the **Mistral models**.

- **Canvas Sharing in ChatGPT**: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1887604146515423390) announced that **canvas sharing is now live in ChatGPT**, allowing users to **share, interact with, or edit canvases**, enhancing collaborative capabilities.

**AI Industry News and Events**

- **Applied ML Days Workshops with Google DeepMind**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1887537279306244321) invited participants to two workshops at **Applied ML Days** focused on **Building LLM Applications using Google Gemini** and **Natural Interactions with Foundational Models**.

- **Cerebras Powers Leading AI Lab**: [@draecomino](https://twitter.com/draecomino/status/1887624699351605495) shared that **Cerebras** is now **powering a leading AI lab in production**, showcasing advancements in AI infrastructure and computing capabilities.

- **Keras Community Meeting**: [@fchollet](https://twitter.com/fchollet/status/1887573636082770345) announced a **public Keras team community meeting**, offering updates on **what's new in Keras** and an opportunity for developers to **ask questions**.

**Personal Achievements and Updates**

- **Google Developers India Recognition**: [@RisingSayak](https://twitter.com/RisingSayak/status/1887489752171225137) expressed gratitude for being nominated and thanked **@GoogleDevsIN** for the recognition, highlighting a sense of fulfillment in the community.

- **Philipp Schmid Joins Google DeepMind**: [@osanseviero](https://twitter.com/osanseviero/status/1887520341276098940) welcomed **Philipp Schmid** to **Google DeepMind**, expressing excitement to work with a **dream team** including **@DynamicWebPaige**, **@film_girl**, and others.

**Memes/Humor**

- **Types of Programmers**: [@hyhieu226](https://twitter.com/hyhieu226/status/1887540297103778268) humorously categorized programmers into two types: those who write **verbose type declarations** and those who use **'auto'** for simplicity.

- **Overconfidence Warning**: [@qtnx_](https://twitter.com/qtnx_/status/1887496898484822126) shared a personal reflection reminding that **overconfidence can lead to loss**, advising to **stay humble and work diligently**.

- **AI Lab Grifters**: [@scaling01](https://twitter.com/scaling01/status/1887487264965435629) called out **YouTube grifters** in the AI community, highlighting a shift from dismissing AI advancements to monetizing them, implying a focus on profit over technology.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Hibiki Speech-to-Speech Translation - FR to EN Capability**

- **[Hibiki by kyutai, a simultaneous speech-to-speech translation model, currently supporting FR to EN](https://v.redd.it/gpawbnvlyihe1)** ([Score: 448, Comments: 40](https://reddit.com/r/LocalLLaMA/comments/1ij35u7/hibiki_by_kyutai_a_simultaneous_speechtospeech/)): **Hibiki**, developed by **Kyutai**, is a simultaneous speech-to-speech translation model that currently supports translation from **French (FR) to English (EN)**.
  - **Hibiki's Capabilities**: **Hibiki** is praised for its real-time translation quality, naturalness, and speaker similarity, with resources available on [GitHub](https://github.com/kyutai-labs/hibiki) and [Hugging Face](https://huggingface.co/kyutai). The model's ability to preserve the speaker's voice while adapting its pace to the semantic content is highlighted, and it is noted for outperforming previous systems.
  - **Community Feedback and Requests**: Users express admiration for the model's performance, with some desiring additional language support, particularly **Spanish** and **Chinese**. There is a desire for an on-device version for convenience and travel purposes, especially for non-English speaking regions.
  - **Cultural and Developmental Observations**: There are humorous remarks about the French's proficiency in English and the Japanese-inspired names of the French-developed model. The open-source nature of the project, similar to **Mistral**, is noted, with expectations for future advancements in on-device translation capabilities.


**Theme 2. Challenges with Gemini 2.0 Pro Experimental Model**

- **The New Gemini Pro 2.0 Experimental sucks Donkey Balls.** ([Score: 205, Comments: 83](https://reddit.com/r/LocalLLaMA/comments/1iirej3/the_new_gemini_pro_20_experimental_sucks_donkey/)): The author criticizes the **Gemini 2.0 Pro Experimental** model for its poor performance compared to the previous **1206** model, highlighting issues like frequent mistakes and unwanted code refactoring. They express frustration with Google's pattern of releasing models that regress in quality, contrasting it with the impressive speed and efficiency of the **Flesh light 2.0** for OCR tasks.
  - Many users express dissatisfaction with **Gemini 2.0 Pro Experimental**, noting issues like decreased intelligence and increased speed at the cost of quality, with some preferring the older **1206** model or other models like **Flash 2.0** for better performance in specific tasks like coding and creative writing.
  - **Flash 2.0** and **o1** models are praised for their effectiveness, especially in handling complex queries and maintaining context over longer tasks, while newer models like **o3-mini** are criticized for requiring more verbose input to understand user intent, leading to inefficiencies.
  - The discussion highlights a broader trend where AI models are becoming faster and more efficient but at the expense of depth and consistency, with some users pointing out the limitations of current evaluation metrics and the challenges of balancing speed with quality in real-world applications.


**Theme 3. Open WebUI Releases Code Interpreter and Exa Search Features**

- **Open WebUI drops 3 new releases today. Code Interpreter, Native Tool Calling, Exa Search added** ([Score: 185, Comments: 61](https://reddit.com/r/LocalLLaMA/comments/1iisj7j/open_webui_drops_3_new_releases_today_code/)): **Open WebUI** introduced significant updates in version **0.5.8**, including a **Code Interpreter** that executes code in real-time using **Pyodide**, a redesigned chat input UI, and **Exa Search Engine Integration** for retrieving information within the chat. Additionally, **Native Tool Calling Support** is now available experimentally, promising reduced query latency and improved contextual responses. [Release details](https://github.com/open-webui/open-webui/releases) are available online.
  - **Code Interpreter and Pyodide**: Users appreciate the addition of the code interpreter using **Pyodide**, noting its limitations but recognizing its utility for common use cases. There's a call for improvements such as integrating **Gradio** and enabling result downloads, like plots or processed data.
  - **Community Contributions**: Despite many contributors, **tjbck** is acknowledged as the primary consistent contributor to **Open WebUI**, with suggestions to support them through [GitHub sponsorship](https://github.com/sponsors/tjbck). The project is praised for its rapid feature updates and its competitive edge over proprietary UIs.
  - **Document Handling and RAG**: There are criticisms regarding document handling, particularly the use of simple vector DB RAG for single document references, which often fails on simple queries. Suggestions include moving document, RAG, and search functionalities to separate pipelines to keep up with fast-moving advancements, and disabling RAG by default for better user control.


**Theme 4. Over-Tokenized Transformer Enhances LLM Performance**

- **[Over-Tokenized Transformer - New paper shows massively increasing the input vocabulary (100x larger or more) of a dense LLM significantly enhances model performance for the same training cost](https://www.reddit.com/gallery/1iiwmsq)** ([Score: 324, Comments: 37](https://reddit.com/r/LocalLLaMA/comments/1iiwmsq/overtokenized_transformer_new_paper_shows/)): A new paper demonstrates that massively increasing the input vocabulary of a dense **Large Language Model (LLM)** by 100 times or more significantly boosts model performance without increasing training costs. This finding suggests a potential strategy for improving transformer efficiency by expanding the vocabulary size.
  - **Tokenization and Vocabulary Size**: Increasing the vocabulary size to millions, as opposed to the typical **32k to 128k**, can enhance model performance by using more meaningful, hierarchical tokens. This approach achieves faster convergence by combining multiple tokens into new ones, though it primarily improves training performance rather than final performance in direct proportion.
  - **Potential Challenges and Considerations**: Concerns arise about undertrained tokens due to greedy tokenizers, which might lead to performance issues with misspellings and tasks sensitive to single character mutations, such as arithmetic or algebraic reasoning. There are also questions regarding the impact on memory usage, inference speed, and effective context size when using smaller tokens.
  - **Research and Comparisons**: A similar study from three months ago suggested that models like **Llama 2 70B** should use at least **216k tokens** for optimal compute use, and even larger token counts could be beneficial. The paper's findings are particularly interesting for dense models, but they did not show the same improvement for **Mixture of Experts (MoE)** models, highlighting a potential area for further exploration.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. Altman admits reduced competitive edge for OpenAI**

- **Altman admits OpenAl will no longer be able to maintain big leads in AI** ([Score: 259, Comments: 69](https://reddit.com/r/OpenAI/comments/1ij13ub/altman_admits_openal_will_no_longer_be_able_to/)): **Sam Altman** acknowledged that **OpenAI** will face increased competition and will not maintain its previous lead in AI development. He noted that while **OpenAI** will produce better models, the competitive gap will be smaller, as reported in a **Fortune.com** interview. [Source](https://fortune.com/2025/02/01/sam-altman-openai-open-source-strategy-after-deepseek-shock/).
  - **OpenAI's Competitive Strategy**: Several commenters discussed the idea that **OpenAI** attempted to maintain a monopoly by controlling the release of their discoveries, which gave them an advantage of about **3-4 months** before competitors could replicate their work. This strategy was seen as a temporary measure to stay ahead in the competitive landscape.
  - **Technology Plateau and Model Training**: There is a perception that AI technology may be plateauing, with users noting that **OpenAI** admitted to facing inevitable competition. Commenters highlighted the challenge of preventing others from using larger models' outputs to train their own, indicating that **OpenAI** will have to continue innovating alongside other companies.
  - **Media and Public Interaction**: A commenter's question ended up in a **Fortune** article, leading to discussions about media ethics and the value of such publications. There was also appreciation for **Sam Altman's** openness during an AMA, despite the limitations on what he could disclose.


**Theme 2. Deep Reconstruction using AI tools for complex analysis**

- **Give me a prompt for Deep Research and I'll run it for you!** ([Score: 246, Comments: 111](https://reddit.com/r/OpenAI/comments/1iinuib/give_me_a_prompt_for_deep_research_and_ill_run_it/)): The user has paid **$200** for access to **Deep Research** and is offering to run prompts for the community to evaluate its capabilities. They compare it to **o3-mini-high**, noting that Deep Research supports attachments but doesn't seem significantly better. They invite the community to submit serious prompts and vote on them to prioritize which ones to execute.
  - **Complex Prompt Challenges:** Users are submitting complex, multidisciplinary prompts, such as those involving **particle physics**, **ontological spaces**, and **depression subtypes**. These often require clarification from the AI to proceed with research or analysis, highlighting the need for precise inputs to optimize AI responses.
  - **Investment and Economic Predictions:** There is significant interest in using AI for **stock market predictions** and economic analysis in a post-ASI world. Users are curious about the impact of ASI on stock valuations, GDP growth, and bond markets, emphasizing the speculative nature of these inquiries and the need for AI to consider multiple scenarios and variables.
  - **Agricultural and Environmental Systems:** The discussion includes innovative agricultural methods like the **3 sisters method** and its potential expansion using AI to optimize plant cooperation systems for different climates and soil types. This reflects a broader interest in applying AI to enhance sustainable agricultural practices.


- **Dear OpenAI, if I'm paying $200 per month for Deep Research, the ability to save to PDF/Markdown would be nice!** ([Score: 229, Comments: 40](https://reddit.com/r/OpenAI/comments/1iit2y5/dear_openai_if_im_paying_200_per_month_for_deep/)): The author expresses disappointment that **OpenAI's Deep Research**, despite its cost of **$200 per month**, lacks a feature to save reports directly to PDF or Markdown. They suggest a workaround by using the 'copy' button to obtain raw Markdown, which can then be pasted into **Notion**.
  - Many users express frustration over **OpenAI's Deep Research** lacking a straightforward PDF or Markdown export feature, emphasizing that the AI should reduce busy work and facilitate easier integration with other applications like **Pages** and **Word**. The absence of these features is seen as a significant oversight given the tool's high cost of **$200 per month**.
  - Suggestions for workarounds include using the 'copy' button for Markdown, then pasting into a **Markdown Editor** or using **print > save as PDF**. However, users find these manual processes counterintuitive to the AI's purpose of saving time and simplifying tasks.
  - There is a humorous discussion around the naming conventions of AI tools, with comparisons to **Gemini Deep Research** and anticipation for future tools like 'Microsoft Co-pilot - In to Deep' edition. The conversation highlights a broader dissatisfaction with current AI capabilities and the expectation for more seamless functionalities in premium tiers.


**Theme 3. Open Source AI for Trackable Health Diagnostics**

- **How I Built an Open Source AI Tool to Find My Autoimmune Disease (After $100k and 30+ Hospital Visits) - Now Available for Anyone to Use** ([Score: 195, Comments: 27](https://reddit.com/r/OpenAI/comments/1ij6619/how_i_built_an_open_source_ai_tool_to_find_my/)): The author shares their journey of building an **open-source AI tool** to aid in diagnosing autoimmune diseases after spending **$100k** and visiting **30+ hospitals** without clear answers. The tool allows users to upload and standardize medical records, track lab result changes, and identify patterns using different AI models, including **Deepseek** and **GPT4/Claude**. They provide resources like [Fasten Health](https://github.com/fastenhealth/fasten-onprem) for obtaining medical records and mention plans to migrate document parsing to run locally.
  - **Data Security Concerns**: Several commenters emphasize the critical importance of running the tool locally to avoid data breaches, especially given the sensitivity of **medical records** and the high value of such data on the dark market. **Mithril** was mentioned as a secure AI deployment option for handling medical information, highlighting the need for **certifications** like **FISMA** and **HITRUST**.
  - **Fragmented Diagnosis to Discovery**: The discussion includes a personal account of receiving multiple diagnoses like **herniated disc** and **spinal curvature**, which were later unified into a diagnosis of **Ankylosing Spondylitis** using the tool. A suggestion to consider **EDS (Ehlers-Danlos Syndrome)** was also made, indicating the tool's potential in refining and discovering complex medical conditions.
  - **User Reactions**: Strong reactions from users indicate surprise and concern about the potential for serious data breaches, with multiple comments expressing disbelief and highlighting the legal implications of mishandling sensitive medical data.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. Breakthroughs in Model Capabilities and Performance**

- [**Hibiki Achieves Real-Time Speech Translation Like a Human**](https://x.com/kyutai_labs/status/1887495488997404732): Kyutai's **Hibiki model** delivers *simultaneous speech-to-speech translation* from 🇫🇷 to 🇬🇧, adapting pace to content and preserving speaker's voice. Early reports boast **Hibiki's** superior *quality*, *naturalness*, and *speaker similarity*, rivaling professional human interpreters in real-time communication.
- [**Gemini 2.0 Flash Parses PDFs at Scale for Pennies**](https://x.com/deedydas/status/1887556219080220683): **Gemini 2 Flash** now efficiently parses large PDF documents for approximately **$1 per 6000 tokens**, marking a significant leap in document processing. This cost-effective solution unlocks new possibilities for applications requiring high-volume, high-accuracy text extraction from complex document formats.
- [**Unsloth's GRPO Makes DeepSeek-R1 Reasoning Accessible on 7GB VRAM**](https://x.com/UnslothAI/status/1887562753126408210): Unsloth's latest **GRPO** update slashes memory usage by **80%**, allowing users to reproduce **DeepSeek-R1's** reasoning with just **7GB VRAM**.  This breakthrough democratizes access to advanced reasoning models, enabling local experimentation with models like **Llama 3.1 (8B)** and **Phi-4 (14B)** even on resource-constrained systems.

**Theme 2. Tooling and Framework Enhancements for AI Engineers**

- [**GitHub Copilot Awakens as an Agent, Edits Code Like a Pro**](https://github.blog/news-insights/product-news/github-copilot-the-agent-awakens/): **GitHub Copilot** introduces *agent mode* and general availability of *Copilot Edits*, enhancing developer workflow with smarter AI assistance. This update aims to provide more proactive and effective coding support, transforming **Copilot** into a more integrated and powerful development partner.
- [**Windsurf IDE Supercharges with Gemini 2.0 Flash and Cascade Web Search**](https://x.com/windsurf_ai/status/1887235006374035966): **Windsurf** now supports *blazingly fast* **Gemini 2.0 Flash**, consuming only **0.25** prompt credits, and **Cascade** gains automatic web search via **@web**, costing 1 flow action credit. These enhancements aim to boost developer productivity with faster models and integrated information retrieval within the IDE environment.
- [**Cursor IDE Unveils GitHub Agents and Architect Feature for Productivity Boost**](https://forum.cursor.com/): **Cursor IDE** rolls out new *GitHub agents* and an *architect feature*, aiming to significantly boost developer productivity and streamline complex projects. While users are enthusiastic about these additions, some report potential bugs in command execution within the Composer tool, signaling active development and refinement of these features.

**Theme 3. Navigating Challenges in Model Performance and Infrastructure**

- [**DeepInfra Provider Plagued by 50% Failure Rate, Users Report**](https://discord.com/channels/1091220969173028894): **DeepInfra** provider is currently failing to return responses *50% of the time*, causing zero token completions and significant processing delays for users, particularly in applications like SillyTavern. Community members are actively sharing observations and seeking solutions to these performance issues across different models and providers on OpenRouter.
- [**LM Studio Users Face API Error Avalanche, Seek Debugging Guidance**](https://discord.com/channels/1110598183144399058): **LM Studio** users are reporting a surge of errors like *'unknown error'* and *'exit code: 18446744072635812000'* when loading models, prompting calls for detailed system specs and API insights for effective debugging. State handling issues when connecting via API also highlight the need for clearer documentation and user support for API interactions.
- [**Codeium Jetbrains Plugin Criticized for Unresponsiveness and Frequent Restarts**](https://discord.com/channels/1027685395649015980): Users are voicing frustrations with the **Codeium Jetbrains plugin**, citing frequent failures to respond and the necessity for frequent restarts, impacting developer workflow.  Some users are opting to switch back to Copilot for reliability, while others report specific errors in PhpStorm, indicating persistent instability in the plugin's performance.

**Theme 4.  Community Driven Innovations and Open Source Contributions**

- [**Independent Researchers Leverage JAX and TPUs for Low-Cost AI Research**](https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform): Independent AI researchers are exploring realistic domains for AI/ML research, with recommendations to learn **JAX** for access to **TPU Research Cloud**, enabling resource-efficient experimentation. The community cites the **OpenMoE** GitHub repository as a prime example of impactful research in Mixture-of-Experts models achievable with limited resources.
- [**Y CLI Project Emerges as OpenRouter Terminal Chat Alternative**](https://github.com/luohy15/y-cli): **Y CLI**, a personal project, offers a terminal-based web chat alternative for OpenRouter, storing chat data locally in jsonl files and now supporting **Deepseek-r1** reasoning.  Developers are actively encouraged to contribute to **Y CLI** via its GitHub repository, fostering community-driven development and catering to terminal enthusiasts.
- [**Hugging Face Community Clones DeepResearch for Open Access**](https://huggingface.co/blog/open-deep-research): **HuggingFace** researchers launched an *open-source clone* of **DeepResearch**, emphasizing the importance of agent frameworks and introducing the **GAIA benchmark** to foster community contributions. This initiative promotes transparency and collaborative development in AI agent technology, encouraging broader participation and innovation.

**Theme 5.  Ethical Debates and Business Model Scrutiny in AI**

- [**OpenAI's Profit-First Approach Sparks Community Debate and Skepticism**](https://x.com/OpenAI/status/1887616278661112259): Members are debating the motivations of AI giants like **OpenAI**, criticizing their prioritization of *profit over public good* and questioning the competitiveness of smaller companies. Skepticism surrounds **OpenAI's** updated chain of thought feature, with doubts about its real purpose amidst concerns about corporate agendas dominating AI development.
- [**AI Backlash Echoes Crypto Distrust, Fuels Ethical Concerns**](https://rentry.org/vwa65v85): Public distrust towards **AI** is linked to past negative experiences with **cryptocurrency** and **NFTs**, impacting the perception of AI technology and raising ethical concerns about AI development. Critics point to *unlicensed AI training data* and the potential for AI to disrupt labor markets, fueling broader societal anxieties about AI's ethical implications.
- [**Stability AI's Subscription Costs and 'Private Images' Option Spark Debate**](https://discord.com/channels/1002292111942635562): Members are questioning the 'private images' option in Stability AI's **Max subscription**, debating if it implicitly caters to NSFW content, while others compare cloud service costs to local electricity expenses. These discussions reflect varying user attitudes towards the entry costs and perceived utility of different AI models, highlighting the ongoing debate about the economics of AI services.



---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's GRPO now reasons with vLLM!**: Unsloth's latest update on **GRPO** allows reproducing **DeepSeek-R1's** reasoning with as little as **7GB VRAM**, also supporting models with reduced memory use using [Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb).
   - Users can experiment with the latest features and notebook updates for performance enhancements, as well as training **Llama 3.1 (8B)** and **Phi-4 (14B)** models.
- **Unsloth Fine-Tunes R1 Distill Llama + Qwen!**: Unsloth introduced support for fine-tuning distilled **DeepSeek models**, utilizing **Llama** and **Qwen** architectures and making [model uploads available](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5).
   - Unsloth also supports new models such as **Mistral-Small-24B-2501** and **Qwen2.5**, which can be found in the [Hugging Face collection](https://huggingface.co/collections/unsloth/mistral-small-24b-2501-all-versions-679fe9a4722f40d61cfe627c).
- **Quantization cuts VRAM by 60%!**: Recent discussions highlight effective use of **BitsandBytes quantization**, reducing VRAM usage by approximately **60%** when selectively quantizing layers with further details available in [Unsloth’s blog posts](https://unsloth.ai/blog/dynamic-4bit).
   - Participants discussed using multi-turn conversational datasets with GRPO, emphasizing retaining reasoning contexts during model training and improving AI model reasoning capabilities with well-formatted datasets.
- **OpenAI Prioritizes Profit**: Members debated motivations of major AI players like **OpenAI**, criticizing profit prioritization over public good, along with concerns about smaller companies' competitiveness and potential alliance needs.
   - A user highlighted **OpenAI's** updates to the **chain of thought** feature, linking to the [announcement](https://x.com/OpenAI/status/1887616278661112259), but responses showed skepticism about its real purpose.
- **Indie AI Researchers Tap TPUs with JAX!**: Independent researchers seek realistic domains to start AI/ML research, where a member recommends learning **JAX** for **TPU Research Cloud** access, linking to the [application form](https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform).
   - Members cited the **OpenMoE** GitHub repository as a relevant example of conducting research in Mixture-of-Experts models, and even pretraining small transformers on the **TinyStories dataset**.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability Welcomes New Community Chief**: Maxfield, the new Chief Community Guy at Stability, introduced himself to improve community engagement, previously contributing at **Civitai** since 2022.
   - Acknowledging past engagement was *lackluster*, Maxfield plans to launch a **feature request board** and encourage researchers to share project updates for improved **transparency**.
- **Civitai Plagued by Download Errors**: Users reported encountering **Error 1101** when downloading models from **Civitai**, leading to community frustration over downtime.
   - The issues raised concerns about the accessibility and reliability of accessing models via **Civitai**.
- **Users Dissect Latent Space Intricacies**: A user expressed confusion over the complexity of tools for swapping **latent space parameters**, suggesting a need for more user-friendly solutions.
   - Discussions touched on potential implementations for newer **diffusion models** and the challenges of adapting existing architectures.
- **AI Subscription Costs Spark Debate**: Members questioned the 'private images' option in Stability's **Max subscription**, debating if it catered to NSFW content, while others compared cloud service costs to local electricity expenses.
   - The discussions highlighted varying attitudes towards the entry costs versus the utility of different **AI models**.
- **Engineers Seek AI Prompting Clarity**: A user sought insights into **prompting techniques** for generative models, while others suggested external tools like [brxce/stable-diffusion-prompt-generator](https://ollama.com/brxce/stable-diffusion-prompt-generator) to assist.
   - The conversation underscored the difficulty in adapting to different **AI model** requirements and generating satisfactory prompts, especially across platforms.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Adds Gemini 2.0 Flash**: Windsurf now supports **Gemini 2.0 Flash**, consuming only **0.25** user prompt credits per message and flow action credits per tool call, as announced in [a tweet](https://x.com/windsurf_ai/status/1887235006374035966).
   - While *blazingly fast* and efficient, **Gemini 2.0 Flash** is limited in tool calling ability but excels at answering codebase-related questions.
- **Windsurf Next Beta Arrives**: Users can now access the latest features of **Windsurf Next** by downloading the beta from [this link](https://codeium.com/windsurf/download-next).
   - The beta allows early exploration of new AI capabilities with the flexibility to switch between **Next** and **Stable** versions.
- **Jetbrains Plugin Criticized by Users**: Users reported frustration with the **Codeium Jetbrains plugin**, citing frequent failures to respond and the necessity for frequent restarts.
   - One user switched back to **Copilot** for reliability, while another reported an error in **PhpStorm** related to file access.
- **Users Report Windsurf Performance Issues**: Users reported performance issues with **Windsurf**, particularly with models like **O3-mini** and **Gemini Flash**, which finish prematurely without complete suggestions.
   - One user expressed frustration over the need to continuously prompt the model to *'continue'*, raising concerns about wasted credits.
- **Cascade Learns Web Search**: **Cascade** can now perform web searches automatically or via user commands like **@web** and **@docs**, costing 1 flow action credit, described in the [Windsurf Editor Changelogs](https://codeium.com/changelog).
   - This functionality supports URL input and uses web context to improve responses, aiming to provide more accurate and comprehensive information.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Users Find Port Error Fix**: A user reported an invalid port error with **Aider** when loading model metadata, indicating a potential configuration issue.
   - Another member suggested overriding the default model metadata file as a workaround to resolve this error, ensuring the tool functions correctly.
- **Gemini's Unique Editing Needs**: Users discussed inconsistencies with **DeepSeek** and **Gemini** models, noting **Gemini's** unique editing format (**udiff**) differs from other models.
   - **Aider** automatically uses **udiff** for **Google** models while maintaining different defaults for others, accommodating this variation.
- **Pen Testing AI Profitable, Risky**: A member shared their project for pen testing using LLMs, creating a simulated hacking environment where two models collaborate.
   - Despite high token usage, professional pen tests can be extremely lucrative, suggesting a potential financial benefit.
- **HuggingFace Clones DeepResearch**: **HuggingFace** researchers created an open-source clone of **DeepResearch**, as detailed in their [blog post](https://huggingface.co/blog/open-deep-research).
   - This initiative emphasizes the importance of **agent frameworks** and introduces the **GAIA benchmark**, to foster community contributions.
- **R1 Model Dumps Junk `<think>` Tokens**: A user reported commit messages filled with `` tokens when using **R1** via **Together.ai**, seeking guidance on configuration.
   - Recommendations included configuring model settings to minimize these tokens in commit messages and keep commits clean.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.0 Pro Sparking Excitement**: Users are excited about **Gemini 2.0 Pro** with its **2 million token context**, which facilitates complex interactions, but raised concerns about its usability compared to **Google AI Studio**.
   - Free alternatives offer extensive customization and might give users better results on certain tasks, and the community suggests weighing effort against perceived value of additional features in premium models.
- **DeepSeek Tangles with ChatGPT for Chess Title**: A potential chess match between **DeepSeek** and **ChatGPT** is piquing user's interest given the models' limitations on reasoning, which promises to be highly amusing.
   - Humorous contrasts were drawn between the pricing of DeepSeek’s **$1 chess game** versus OpenAI's **$100 chess game**, suggesting some prefer the cheaper, yet still challenging, game.
- **Gemini Flash 2.0 and Copilot Shine as Coding Tools**: In discussions about coding, members recommended **Gemini Flash 2.0** and **Microsoft Copilot** for their features and cost-effectiveness, particularly for advanced mathematics.
   - Users noted that **Copilot** offers a free trial, making it easier to explore without immediate financial commitment and allows engineers to 'try before they buy'.
- **Deep Research Chat Eagerly Awaited by Plus Users**: Several members expressed eagerness for the **Deep Research** chat feature to be available for **Plus users** soon, noting their **need** for it in the coming days.
   - A member inquired if anyone had shared information about **Deep Research** chats, obviously looking for insights, and prompted others to express similar anticipation regarding the feature coming to Plus subscriptions.
- **Fine Tuning AI with Iterative Editing**: A member suggested using Python to count words and iterate to ensure better response length, but noted this may impact creativity when attempting to control **Response Length** in AI responses.
   - Members also noted the importance of editing inputs using the edit button to sculpt the AI's output effectively by adjusting your input until satisfied before proceeding to ensure coherent context in the conversation.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE Gets GitHub Agents, Architect Feature**: Users are excited about the new GitHub agents and the architect feature in **Cursor IDE**, which aims to boost productivity.
   - However, some users reported a potential bug with running commands within the Composer tool after recent updates, as noted in the [Cursor forum](https://forum.cursor.com/).
- **Gemini 2.0 Self-Learning Solid, But Not Top Dog**: Users find **Gemini 2.0** performs well for self-learning tasks due to its affordability and context management, some discussion mentioned it was solid but not superior to **Sonnet** for coding.
   - The community noted its effective context use makes it appealing for handling large codebases, potentially shaking up **AI testing tools** like [Momentic](https://momentic.ai/).
- **Clipboard Comparison Tool Recommendations**: The community is recommending a **VSCode extension** for clipboard comparisons, which allows users to compare against clipboard content as documented in [Microsoft's VSCode documentation](https://github.com/microsoft/vscode-docs/issues/7284).
   - Users are also drawing comparisons between **VSCode's local history** and **JetBrains' Timeline**, suggesting **Timeline** offers greater efficiency, and the **Partial Diff** extension from the [VSCode Marketplace](https://marketplace.visualstudio.com/items?itemName=ryu1kn.partial-diff).
- **MCP Server Configs Demand Better Context**: A user is seeking assistance with **MCP server configurations** and accessing keys for **Supabase**, noting limited access with some keys and the github repo for [mcp-starter](https://github.com/daniel-lxs/mcp-starter).
   - The community is generally highlighting the need for improved context management within **Cursor**, particularly for managing complex projects, using the releases from [daniel-lxs/mcp-starter](https://github.com/daniel-lxs/mcp-starter/releases/).
- **Cursor's Context Crunch Spurs Debate**: Concerns are surfacing about context limitations in **Cursor**, with some users preferring models like **Cline** or **Google models** for their larger context windows, perhaps because they are reading Andrej Karpathy's tweets on [vibe coding](https://x.com/karpathy/status/1886192184808149383).
   - There's ongoing debate on how context size impacts the effectiveness of **AI models**, specifically how larger context windows could boost performance in broader applications, and the role of model specific rules as discussed in the [cursor forum](https://forum.cursor.com/t/model-specific-rules/47175).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Focus Mode Gets the Chop**: Users noticed the temporary removal of **Focus Mode** in Perplexity AI, sparking debate on the necessity to explicitly mention sources like Reddit in prompts.
   - Some users expressed concerns about the complication this adds to their ability to direct the AI's information sourcing effectively.
- **Decoding Model Use in Perplexity Pro**: Users are trying to clarify if **Pro mode** fully uses models like **Claude 3.5** end-to-end or integrates **R1** for reasoning, suggesting a more complex, multi-model approach.
   - Insights indicate that undisclosed models conduct initial searches before handing off to chosen models for the final answer generation.
- **ByteDance Dives Deep with Deepfakes**: ByteDance's release of new **deepfake technology** has ignited discussions on the ethical implications and potential for misuse within the AI community.
   - Community members are actively speculating on the ramifications of this technology, weighing its innovative possibilities against its capacity for harm.
- **Desire for Model Transparency Swells**: Users are urging **Perplexity AI** for clearer communication about **model specifications** and updates, particularly regarding changes that impact functionality and performance.
   - Greater transparency is expected to diminish user confusion and improve interaction with the platform’s AI functionalities.
- **Sonar Pro Devs on Hot Seat for Security**: An urgent call went out for contact with the **Sonar Pro reasoning developers** due to the discovery of a **security issue**.
   - Users were directed to email api@perplexity.ai to address the vulnerability.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek Insurance gets even deeper**: OpenRouter now insures **DeepSeek R1** requests that receive no completion tokens, so you won't be charged even if the upstream provider does.
   - The completion rate for Standard **DeepSeek R1** has improved from **60%** to **96%** over time, making it a more reliable option.
- **Kluster's Cancellation Catastrophe Corrected**: A **Kluster** integration issue caused delayed completion tokens and unexpected charges due to failure to cancel timed-out requests.
   - This issue has since been addressed, resolving the problem of users being charged despite apparent timeouts on OpenRouter's end.
- **Qwen Quitting Quietly**: Novita is deprecating their **Qwen/Qwen-2-72B-Instruct** model, with OpenRouter set to disable it around the same time.
   - Users should transition away from this model to avoid disruption when the model becomes unavailable.
- **Y CLI yearns for your attention**: **Y CLI**, a personal project and web chat alternative, stores all chat data in single jsonl files and has added **Deepseek-r1** reasoning content support, evidenced in [this asciinema recording](https://asciinema.org/a/701903).
   - Developers are encouraged to contribute to **Y CLI** via its [GitHub repository](https://github.com/luohy15/y-cli), with a call for fellow **terminal fans**.
- **DeepInfra Deeply Inconsistent**: Users reported that **DeepInfra** is currently failing to return responses **50%** of the time due to an increase in processing delays, often causing zero token completions when utilized with applications like SillyTavern.
   - The community is sharing observations about performance differences between models and providers, including suggestions for improvements.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **User face LM Studio API Error Avalanche**: Users reported errors like 'unknown error' and 'exit code: 18446744072635812000' when loading models in **LM Studio**, needing system specs and API details for debugging.
   - One user struggled with **state handling** when connecting to local models via API, indicating the need for better guidance on API interactions.
- **Obsidian's Smart Connections Extension Causes Turmoil**: Users faced errors connecting **Obsidian's Smart Connections extension** to **LM Studio**, citing conflicts with other extensions and missing required fields in API responses.
   - Troubleshooting involved uninstalling conflicting plugins and rebuilding caches, though ongoing errors persisted even after setting up a connection.
- **TheBloke Models Still a Standard**: Members inquired about the safety and reliability of downloading AI models from **TheBloke**, even with his reduced community presence.
   - It was confirmed that **TheBloke's models** remain a standard in the industry, with users encouraged to monitor community channels for availability updates.
- **DDR5 6000 EXPO Timings are Conservative**: A user found their **DDR5 6000 EXPO timings** to be conservative, observing a peak memory bandwidth of **72** during inference.
   - After completing **4 passes of memtest86**, another member suggested trying [TestMem5](https://github.com/CoolCmd/TestMem5) for a more rigorous assessment of stability.
- **DeepSeek R1 Model Support GPU Acceleration?**: Inquiries arose about GPU acceleration for the **DeepSeek R1 Distill Qwen 7B model**, with uncertainty about which models support GPU use.
   - It was clarified that only specific models like **Llama** have known support for acceleration, leaving some ambiguity for the **DeepSeek** model.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Home Assistant Gets Functional MCP Client**: A user released a **Home Assistant** with **MCP client/server support** and plans to add an animated talking head avatar via [met4citizen/TalkingHead](https://github.com/met4citizen/TalkingHead) for better user interaction.
   - The project is still in development, as they balance paid work with open source development. Also, there was curiosity about usage statistics of the **Home Assistant MCP** bridging with tools like **Claude**.
- **Goose MCP Client Honks Loudly**: Users shared positive experiences using the **Goose MCP Client** in testing, highlighting its effectiveness. 
   - A pull request to enhance its logging features, [block/goose@162c4c5](https://github.com/block/goose/actions/runs/13058183345/job/36804119892?pr=947), is in progress, with a fix to include cached tokens in usage count logs in Goose.
- **Claude Grapples Image Display**: A user reported challenges displaying images as tool results on **Claude Desktop**, encountering an input error. 
   - The error has led to speculation that converting image results to embedded resources might be a potential workaround. 
- **PulseMCP Boasts Use Cases**: A new showcase of practical **PulseMCP Use Cases** debuted, featuring instructions and videos for using various client apps and servers, and launched the use-cases on [PulseMCP](https://www.pulsemcp.com/use-cases).
   - It highlights uses of **Gemini voice**, **Claude**, and **Cline** for managing Notion, converting Figma designs, and creating knowledge graphs.
- **Mobile MCP options discussed**: Members suggested that **Sage** supports iPhones, while options for **Android** users may require using web clients like **LibreChat** or **MCP-Bridge**.
   - This conversation underscores interest in extending MCP functionality beyond desktop environments.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Gemini 2.0 Pro Creates SVGs**: Members discussed that **Gemini 2.0 Pro** demonstrates impressive performance in creating SVGs, surpassing models such as **o3-mini** and **R1**, as noted in [Simon Willison's blog](https://simonwillison.net/2025/Feb/5/gemini-2/).
   - Several members also observed its enhanced SQL query capabilities, hinting at significant progress by Google with **Gemini Flash 2.0**.
- **DeepSpeed Dataloader Batch-size Woes**: A user reported confusion regarding the need to manually define **batch_size** in Dataloader while utilizing DeepSpeed's auto batch size configuration.
   - Another member proposed the integration of DeepSpeed tags into the Dataloader for optimization and suggested potential performance modifications for specific nodes.
- **Harmonic Loss Paper Lacks Punch**: Community members expressed skepticism towards the **harmonic loss paper**, deeming it hastily assembled and failing to provide meaningful performance improvements despite its theoretical advantages.
   - One member indicated that the [GitHub repository](https://github.com/simplescaling/s1) associated with the paper offers more valuable information than the paper itself.
- **Gemini 2.0 Flash leaves mark**: Users trying the new **Gemini 2.0 Flash** model through [LlamaIndex](https://openrouter.ai/google/gemini-2.0-flash-001) reported incredible speeds, although not as fast as **Groq**.
   - One user stated that the model **struggled with returning valid JSON formats**, concluding it may not be suitable for tasks needing reliability in output.
- **S1 Model emerges under $50**: The **S1 reasoning model** was discussed, highlighting its performance compared to models like **OpenAI's o1** but at a fraction of the cost, under **$50**.
   - The S1 model and its tools are available on [GitHub](https://github.com/simplescaling/s1) and was developed through distillation from **Gemini 2.0**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Adobe Seeks LLM Agent Research Partners**: A senior ML Engineer at Adobe is looking for collaboration on **LLM agent research projects**.
   - Interested individuals are invited to join discussions to explore potential partnerships.
- **Deepspeed Batch Size still required**: When using **Deepspeed** with auto batch sizing, the **batch_size** needs to be specified for the data loader.
   - This requirement persists despite auto batch sizing configuration.
- **Thematic Generalization Benchmark Emerges**: A member shared a [GitHub repository](https://github.com/lechmazur/generalization) detailing a **thematic generalization benchmark** for evaluating **LLMs** in category inference from examples and anti-examples.
   - The benchmark's correlation with the performance of **SAE autointerp** was questioned.
- **New Architectures are Cookin' at RWKV**: The **RWKV** team is actively developing some new architectures, showing their proactive stance.
   - One user grappling with scaling issues has invited discourse about prospective collaboration.
- **MATS 8.0 Cohort Applications Now Open**: Applications for the **MATS 8.0** cohort are open until **Feb 28**, offering opportunities for paid full-time mechanistic interpretability research, apply [here](https://tinyurl.com/neel-mats-app).
   - Previous mentees have significantly contributed, evidenced by their involvement in **10 top conference papers**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deep Research Excites Users**: Members laud **OpenAI's Deep Research** for efficiently gathering relevant connections and sources, boosting their cognitive bandwidth.
   - One user highlighted its ability to explore obscure online communities and gather unexpected data.
- **AI Backlash Echoes Crypto Concerns**: Public distrust towards **AI** stems from past issues with **cryptocurrency** and **NFTs**, impacting the perception of AI technology, according to some members.
   - Critics are concerned about **AI training data** being unlicensed and the disruptive effects of AI on labor markets, as articulated in [Why Everyone Is Suddenly Mad at AI](https://rentry.org/vwa65v85).
- **Purpose AI Agent in Legal Limbo**: A user aims to develop a purpose-driven **AI agent** within a legal trust framework, aiming to pioneer legal discourse around **AI personhood**.
   - Feedback centered on the engineering complexity, including integrating fiscal management functions while emphasizing the potential for custom software solutions like the ones shown in [I Built the Ultimate Team of AI Agents in n8n With No Code (Free Template)](https://www.youtube.com/watch?v=9FuNtfsnRNo).
- **Model Merging Mania**: Members discussed strategies for merging **AI models**, sharing insights on improving model instruction tuning and reasoning performance.
   - Various **fine-tuning methods** were explored, highlighting the benefits of innovative techniques in **AI training** to enhance model performance with tools like [Unsloth Documentation](https://docs.unsloth.ai/).
- **Synthetic Data Dream**: A member seeks resources on **synthetic data generation**, focusing on seed-based approaches similar to **Self-Instruct**, after facing challenges with **Magpie** outputs.
   - They discovered the [Awesome-LLM-Synthetic-Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data) GitHub repository, offering a list of resources on **LLM-based synthetic data generation**.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Schulman Shuffles from Anthropic**: Leading AI researcher and OpenAI co-founder **John Schulman** has left **Anthropic** after around five months, prompting speculation about his next career steps [link](https://www.bloomberg.com/news/articles/2025-02-06/openai-co-founder-john-schulman-leaves-rival-firm-anthropic?srnd=undefined).
   - Potential destinations mentioned include **Deepseek** and **AI2**, according to sources.
- **Copilot Becomes Agent**: **GitHub Copilot** introduced **agent mode**, enhancing developer assistance along with general availability of **Copilot Edits** [link](https://github.blog/news-insights/product-news/github-copilot-the-agent-awakens/).
   - This update seeks to provide more effective coding support through AI.
- **LRM Test-Time Scaling Terminology Troubles**: Members questioned the term **test-time scaling** for Long-Range Models (LRMs), emphasizing that **models decide their own output** [link](https://discord.com/channels/1179127597926469703/1179208129083363358/1337167338444689502).
   - It was pointed out that scaling occurs during the **training phase**, rendering the term misleading; a member called the whole concept fundamentally flawed.
- **Qwen Achieves Magical Results**: Qwen 2.5 models are showing impressive results with minimal training data, as noted by members discussing their findings [link](https://x.com/lateinteraction/status/1887356471555563839).
   - Aran Komatsuzaki remarked that Qwen models seem to have a *magical quality*, achieving notably good performance with limited data.
- **Scale AI Faces Adaptation Challenge**: Members recognized that adaptation is possible for **Scale AI**, but challenges remain due to current operational models and valuations [link](https://www.turingpost.com/p/fod86).
   - The consensus was a **bleak outlook** without significant changes to their approach amid a shifting landscape.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Mobile Users Limited to One Model**: Users cannot change the model within the mobile version of **NotebookLM**, a limitation causing frustration for those expecting greater flexibility.
   - This restriction hinders the user experience on mobile devices, leading to confusion among users accustomed to managing models on the web platform.
- **Gemini Shines with Sheets, NotebookLM Stumbles**: Members voiced concerns about using **NotebookLM** for analyzing spreadsheet data, suggesting tools like **Gemini** within **Google Sheets** are more suitable.
   - As [Engadget reported](https://www.engadget.com/ai/gemini-can-now-do-more-complex-data-analysis-in-google-sheets-191218214.html), **Gemini** can use **Python code** to generate insights and charts, reinforcing **NotebookLM's** strength as primarily a *text analysis tool*.
- **Sliders Could Refine AI Creativity**: A user proposed integrating sliders for tuning AI's creativity, similar to features in the **Gemini API**, inspired by discovering an *exploit* related to the AI's features.
   - This functionality would allow users to adjust parameters, offering greater control over the creative output of AI models.
- **NotebookLM Summarizes Legal Testimony at NY Budget Hearing**: A user employed **NotebookLM** to capture testimony at the **New York State Legislature’s Budget hearing on Environmental Conservation**.
   - The user highlighted the challenge of sharing this extensive document due to licensing, while the notes are available [here](https://docs.google.com/document/d/1kcUJvQiAwzX1GU4b0HvOUhLV0UtecLvuQSaTmfRFPpg/).
- **Max Headroom Glitches Back, Critiques AI**: The iconic **Max Headroom** makes a return with a new video, showcasing a unique approach to AI interaction.
   - As seen [on Youtube](https://youtu.be/YXgav2-6DsI?feature=shared), the new content humorously critiques corporate AI practices, urging viewers to share and engage.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Fall 2024 MOOC Certificates Finally Drop**: The **Fall 2024 MOOC certificates** were released today at **8am PT**, after the resolution of technical challenges.
   - Some participants were *downgraded to the Trailblazer tier* due to incomplete coursework, with no makeups offered.
- **Certificate Timeline Elusive**: Members expressed uncertainty regarding the certificate issuance timeline, hoping for delivery within a week or two due to *unforeseen technical issues* being resolved.
   - A member noted discrepancies in certificate receipt, indicating a potential *soft bounce* issue affecting communications.
- **Quiz Availability Creates Confusion**: Concerns arose over the availability of answers for Quiz-1 as Quiz-2 was launched, prompting members to seek clarification on the new policy regarding answer releases.
   - Community members clarified that score visibility for Quiz-1 was possible through the original submission link.
- **Certificate Tier Distribution**: It was revealed that there are **301 Trailblazer**, **138 Masters**, **89 Ninjas**, **11 Legends**, and **7 Honorees** amongst the participants.
   - Clarification was provided that only the honorary tier would be noted if both an honorary and a specific tier were achieved.
- **Course Experience Earns Praise**: The community expressed gratitude for the support received during the course, especially acknowledging the team handling grading and certificate queries.
   - Participants shared enthusiasm for the course, with one member reflecting on their learning journey and the significance of their certificate for future endeavors.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA Blackwell gets OpenAI Triton**: The **Triton compiler** now supports the **NVIDIA Blackwell architecture** due to ongoing collaboration between NVIDIA and OpenAI, enhancing performance and programmability via [cuDNN](https://developer.nvidia.com/cudnn) and [CUTLASS](https://github.com/NVIDIA/cutlass).
   - This advancement enables developers to optimize **matrix multiplication** and **attention mechanisms** for modern AI workloads, improving efficiency and capabilities.
- **Minimize AI Research Costs**: Members shared that independent researchers can conduct efficient work on **LLMs and vision** tasks while fine-tuning models on a limited budget, economizing AI research via stability with **low-bit training weights**.
   - The success of **GPT-2 speedruns** with Muon was highlighted as a prime example of impactful research using limited resources.
- **FP8 Attention requires Hadamard**: A member observed that **FP8 Attention** for video models performed significantly better when utilizing the **Hadamard Transform**, drastically reducing error rates; the [Flash Attention 3 paper](https://arxiv.org/pdf/2407.08608) suggests that this approach is crucial for operations in FP8.
   - Another member recommended using the [fast-hadamard-transform repository](https://github.com/Dao-AILab/fast-hadamard-transform/tree/master/csrc) to implement Hadamard before the attention mechanism for enhanced performance.
- **Reasoning Gym Embraces Sokoban Puzzles**: A pull request was submitted to add **Sokoban puzzles** to **reasoning-gym**, demonstrating a new puzzle format for users to solve, including a graphic explanation of the puzzle setup along with example moves.
   - Members are also discussing collaboratively building a basic gym for the **Rush Hour** game integration into the **reasoning-gym** to encourage joint coding efforts.
- **Linear Attention faces Distillation Challenges**: A member attempted to distill a small LLM to a **linear attention model** following a [recipe from Lolcats](https://cdn.discordapp.com/attachments/1300872762163728550/1337017267925291068/distill_linear.ipynb?ex=67a5e9dd&is=67a4985d&hm=1a5dc02fb98a1f89ed72f7481e30459202f9d1de210fa3729663825137211832&) but the model only produced repeating characters.
   - The member reached out for help specifically from the **Lolcats team**, highlighting the community support aspect often relied upon in AI model development.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **O3 Remains Ahead Despite Pricing**: Despite pricing concerns, **O3** continues to outperform other models, with *Llama 4* being anticipated as the next potential challenger, according to discussions in the **general** channel.
   - Links comparing **DeepSeek-R1 vs o3** [are available online](https://llm-stats.com/models/compare/deepseek-r1-vs-o3) and **o3-mini vs DeepSeek-R1** [are also available](https://llm-stats.com/models/compare/o3-mini-vs-deepseek-r1).
- **DeepSeek Constrained in Political Discussions**: Users found that **DeepSeek** has greater limitations than *ChatGPT* and *O3-mini* in sensitive political discussions, often resulting in unexpected deletions or evasions.
   - This highlights potential constraints in language models when prompted with sensitive political topics.
- **DeepSeek's Cutoff Date Raises Questions**: Reportedly, **DeepSeek's** knowledge cutoff date is July **2024**, which raises questions about its current relevance given that we are now in **2025**.
   - The **Time Bandit** method, for extracting information by leveraging temporal context, was discussed in relation to **DeepSeek**, with more details on its system prompt [available online](https://www.knostic.ai/blog/exposing-deepseek-system-prompts).



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **GRPO Implementation Scores Big**: A member reported a successful implementation of **GRPO** training, achieving training scores ranging from **10% to 40%** on GSM8k.
   - While debugging, they noted challenges with deadlocks and memory management, and are planning improvements and opening the project for contributions.
- **Kolo Extends to Torchtune**: **Kolo** officially announced support for **Torchtune** on their [GitHub page](https://github.com/MaxHastings/Kolo).
   - The project delivers a comprehensive solution for fine-tuning and testing LLMs locally using the best available tools.
- **Llama 3.1 and Qwen 2.5 stumble on Configs**: Members identified **FileNotFoundError** issues downloading and fine-tuning **Llama 3.1** and **Qwen 2.5** due to mismatched path configurations.
   - One member created a [GitHub issue](https://github.com/pytorch/torchtune/issues/2352) to address the incorrect default paths and propose fixes.
- **Hugging Face Fast Tokenizers Get Support**: The community discussed the prospect of using **Hugging Face fast tokenizers**, with members indicating current limitations but ongoing progress.
   - A member mentioned that **Evan** is actively enabling support, as detailed in [this GitHub pull request](https://github.com/pytorch/torchtune/pull/2350).
- **Full DPO Distributed PR Faces Hurdles**: A user reported issues with GitHub checks on their [Full DPO Distributed PR](https://github.com/pytorch/torchtune/pull/2275), with specific errors related to GPU and OOM issues.
   - The error, `ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!`, prompted the user to seek assistance from the community.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Pivots from Python, Focuses on GPU**: In a [recent community meeting](https://www.youtube.com/watch?v=XYzp5rzlXqM), **Modular** clarified that **Mojo** is not currently a superset of **Python**, but focuses on leveraging **GPU** and **performance programming**.
   - This shift emphasizes enhancing **Mojo's** efficiency in its designed applications rather than broadening its language framework.
- **Parser Revision Balances Branching Costs**: A member suggested that the **parser** needs adjustment for handling multiple slices of data, weighing the costs of branching and noting that *branching may be cheaper than significant data transfers*.
   - This is a valid consideration for those not focusing on higher performance needs.
- **Msty Simplifies Local Model Access**: A member introduced **Msty**, an OpenAI-compatible client that simplifies local model interactions compared to using Docker and other complex setups, highlighting its ease of use and features for accessing AI models seamlessly with [Msty's Website](https://msty.app).
   - The importance of offline usability and privacy with Msty was emphasized, suggesting it is highly favorable for users who wish to avoid complex configurations.
- **MAX Serve CLI Mimics Ollama's Features**: Members discussed building a CLI similar to **ollama** on top of **MAX Serve**, noting that MAX Serve can already handle many functionalities offered by Ollama with a docker container.
   - The discussion highlighted the hope for better performance running local models compared to Ollama.
- **Community Reports OpenAI API Incompatibilities**: A user reported missing features in the **OpenAI completions API** with **max serve (v24.6)**, such as generation stopping at specified tokens, suggesting that they file issues on the **GitHub repo** to highlight these missing elements.
   - The group acknowledged ongoing issues with OpenAI API compatibility, particularly referencing the **v1/models** endpoint, and other missing functionalities like token stopping and prompt handling in [this GitHub issue](https://github.com/modular/max/issues/292).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Hibiki Champions Real-time Translation**: [Kyutai](https://x.com/kyutai_labs/status/1887495488997404732)'s **Hibiki** model achieves real-time speech-to-speech translation from 🇫🇷 to 🇬🇧, preserving the speaker's voice and adapting to context.
   - Early reports claim **Hibiki** excels in *quality*, *naturalness*, and *speaker similarity*, rivaling human interpreters.
- **Melanie Mitchell Raises Agent Concerns**: @mmitchell_ai's latest [paper](https://huggingface.co/papers/2502.02649) argues against developing **Fully Autonomous Agents**, emphasizing ethical considerations.
   - The piece sparked debates within the AI community, acknowledging her **balanced perspective** amidst fervent discussions.
- **Mistral AI's Le Chat Enters the Scene**: [Mistral AI](https://x.com/MistralAI/status/1887517520040448510) launched **Le Chat**, a versatile AI assistant tailored for daily personal and professional tasks, accessible on web and mobile.
   - The tool is set to redefine user interaction with AI, potentially impacting workflows and personal routines.
- **OpenAI Enhances o3-mini Capabilities**: OpenAI rolled out enhanced **chain of thought** features in **o3-mini** and **o3-mini-high** ([source](https://x.com/openai/status/1887616278661112259?s=46)), benefiting both free and paid subscribers.
   - The updates promise improved performance and a smoother user experience, reaffirming OpenAI's commitment to continuous service evolution.
- **PDF Parsing Achieves Breakthrough**: **PDF parsing** is now efficiently solved at scale; **Gemini 2 Flash** can parse large documents for approximately $1 per 6000 tokens, according to @deedydas.
   - This advancement in processing complex documents unlocks new possibilities for applications needing high-caliber text extraction.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Gemini 2.0 is now generally available**: **Gemini 2.0** from @google launched with day 0 support, developers can install the latest integration package with `pip install llama-index-llms-gemini` and read more in the [announcement blog post](https://t.co/6oBbYpcFAU).
   - The updated **2.0 Flash** is available to all users in the **Gemini app** on desktop and mobile.
- **LlamaParse Tackles Complex Financials**: Hanane D showcased the parsing of **complex financial documents** accurately and cost-effectively using **LlamaParse** 'Auto' mode, using **@OpenAI embeddings** as shared in this [link](https://t.co/UMZXeXJ5pS).
   - Her demonstration highlights the advancements in parsing technology for extracting relevant insights from intricate data, charts and tables.
- **Embedding Print Troubles LlamaIndex**: A member requested deletion of the **embedding print** from the LlamaIndex documentation due to excessive space usage and readability issues, see [GitHub issue](https://github.com/run-llama/llama_index/issues/17735).
   - Another member offered to create a Pull Request (PR) to address the **embedding print removal**.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LLMs Classify Well, But Noise Makes Them Falter**: Members discussed that although **LLMs** are effective for classification, **noisy data** requires additional techniques like dense embeddings and autoencoder rerankers to improve performance.
   - This suggests the necessity of a more intricate strategy when handling challenging data scenarios.
- **Latency Concerns Dampen LLM Enthusiasm**: The discussion revealed that although LLMs classify effectively, their suitability might diminish in scenarios with strict **latency requirements** due to their processing limits.
   - The suitability of LLMs depends on the specific latency constraints of a given application.
- **Business Requirements Highlight ML Misfits**: A member noted that a **missed opportunity** occurred in properly framing the business requirements during the transition to an ML solution.
   - It should have been evident from the onset that if low-latency is paramount, traditional LLMs might not be the ideal choice.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Fine-Tuning Limits Spark Concern**: A user encountered a **BadRequestError** (Status code: 400) in **Cohere**, indicating the training configuration surpassed the maximum of **250 training steps**, with a **batch size** limit of 16.
   - A member questioned if this restricts fine-tuning to **4000 examples**, highlighting that this limitation wasn't previously in place.
- **AIML System Design Interview Questions Requested**: A member inquired about **system design interview questions** specific to **AI/ML** in the Cohere channel.
   - Another member acknowledged the request and indicated it would be collected, implying team collaboration on this topic.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Request for Canonical System Prompts Arises**: A member requested clarification on the **canonical system prompts** for **fine-tuned tool-using models**, noting the **Gorilla paper** lacked this detail.
   - The goal is to ensure the models reliably return responses or JSON for function calls, suggesting a need for standardized prompt engineering practices.
- **Hugging Face Datasets Seek Transformation**: A member aimed to streamline experimentation by transforming data and using `datasets.map` on **Hugging Face**, signalling a move towards more flexible data manipulation.
   - This highlights ongoing efforts to improve the usability and accessibility of datasets for research and development purposes.
- **Dataset Format Issue with Hugging Face**: A member reported a dataset format mismatch within **Hugging Face**, where **.json** files actually contained **jsonl** data, leading to compatibility problems.
   - The suggested solution involves renaming the file suffix to **.jsonl** and adjusting the dataset config files to align with the actual data format.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Paper Posted on DSPy**: A member shared [a link](https://arxiv.org/abs/2502.02508) to a paper on **DSPy**.
   - The paper was shared in the **#papers** channel.
- **Member Asks About Git Repo**: In the **#examples** channel, a member inquired about the availability of a **Git repo** for their work, indicating interest in accessing related code or resources.
   - The member did not specify which project it was referring to.
- **Colab Notebook Surfaces**: In response to the **Git repo** query, a member provided [a link to a Colab notebook](https://colab.research.google.com/drive/1OXmTKexR9gX33DXRNEAe3dNuxkLXnutX?usp=sharing).
   - Accessing the notebook requires **signing in** and it is likely related to the **DSPy** discussion.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1336790336411930707)** (516 messages🔥🔥🔥): 

> `GRPO and vLLM integration, DeepSeek models and fine-tuning, Quantization techniques, Multi-turn conversational datasets, AI ethics and data privacy` 


- **GRPO with vLLM Support**: The latest update on GRPO reasoning allows users to reproduce DeepSeek-R1's reasoning with reduced memory use, now supporting models with as little as 7GB VRAM.
   - Users are encouraged to experiment and test the latest features and notebook updates for enhanced performance.
- **Fine-Tuning DeepSeek Models**: Unsloth has introduced support for fine-tuning distilled DeepSeek models, with future notebooks planned for guiding users in this process.
   - These distilled models utilize Llama and Qwen architectures, allowing compatibility with Unsloth.
- **Quantization Insights**: Recent discussions highlight the effective use of BitsandBytes quantization, with a significant reduction in VRAM usage by approximately 60% when selectively quantizing layers.
   - Users have shown interest in further reading about the details of this quantization technique in Unsloth’s blog posts.
- **Conversational Dataset Handling**: Participants discussed the use of multi-turn conversational datasets with GRPO, emphasizing the nuances of retaining reasoning contexts during model training.
   - There was a consensus that a well-formatted dataset can enhance the reasoning capabilities of AI models.
- **Ethics and AI Development**: A debate emerged about data privacy and the implications of closed-source models versus open-source alternatives in AI development.
   - Users expressed concerns about the direction of AI and the importance of building models aligned with user values rather than corporate agendas.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.downloadmoreram.com/">DownloadMoreRAM.com - CloudRAM 2.0</a>: no description found</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://huggingface.co/collections/unsloth/llama-32-66f46afde4ca573864321a22">Llama 3.2 - a unsloth Collection</a>: no description found</li><li><a href="https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF">unsloth/SmolLM2-135M-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit Quantization</a>: Unsloth&#x27;s Dynamic 4-bit Quants selectively avoids quantizing certain parameters. This greatly increases accuracy while maintaining similar VRAM use to BnB 4bit.</li><li><a href="https://satori-reasoning.github.io/blog/satori/">Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search</a>: PaperGithubHuggingfaceIntroductionSince the release of OpenAI's o1, significant efforts have been made within the research community to enhance open-source LLMs with advanced reasoning capabilities. T...</li><li><a href="https://x.com/UnslothAI/status/1887562753126408210">Tweet from Unsloth AI (@UnslothAI)</a>: You can now reproduce DeepSeek-R1&#39;s reasoning on your own local device!Experience the &#34;Aha&#34; moment with just 7GB VRAM.Unsloth reduces GRPO training memory use by 80%.15GB VRAM can transfor...</li><li><a href="https://tenor.com/view/travis-neil-primrose-neil-ba-dum-tss-ba-dum-gif-11550351308913763721">Travis Neil Primrose GIF - Travis Neil primrose Neil - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/cognitivecomputations/stablemax-orthogonal">GitHub - cognitivecomputations/stablemax-orthogonal</a>: Contribute to cognitivecomputations/stablemax-orthogonal development by creating an account on GitHub.</li><li><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">black-forest-labs/FLUX.1-dev · Hugging Face</a>: no description found</li><li><a href="https://deepnewz.com/ai/stanford-s-s1-32b-model-outperforms-openai-s-o1-preview-27-on-aime24-math-using-bc4ff754">Stanford&#x27;s s1-32B Model Outperforms OpenAI&#x27;s o1-Preview by 27% on AIME24 Math Questions Using 1,000 Diverse Questions | DeepNewz</a>: Stanford researchers have introduced a new approach called Simple Test-Time Scaling (s1), which enhances reasoning performance in language models. The s1-32B model, fine-tuned on a dataset of 1,000 di...</li><li><a href="https://github.com/unslothai/unsloth/issues/1561">[Fixing] More finetuning support · Issue #1561 · unslothai/unsloth</a>: Support sequence classification Flex Attention for Gemma and others Variable sequence length and auto unpadding / padding Tool Calling Refactor and merge xformers, SDPA, flash-attn, flex-attention</li><li><a href="https://github.com/unslothai/unsloth/issues/1376#issuecomment-2632615715">llama.cpp GGUF breaks [FIXED] · Issue #1376 · unslothai/unsloth</a>: As of 3rd December 2024 - fixed. Please update Unsloth via pip install --upgrade --no-deps --no-cache-dir unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1337126988737019974)** (1 messages): 

> `Reasoning in Unsloth, DeepSeek-R1, Model Fine-Tuning, New Model Support` 


- **Unsloth introduces Reasoning with R1**: Unsloth has unveiled reasoning capabilities with the release of R1, which can be trained locally or for free on [Colab](https://unsloth.ai/blog/r1-reasoning). The approach allows users to reproduce R1-Zero's insights with just **7GB of VRAM**.
   - Additionally, the [Colab notebooks](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) provide resources for both **Llama 3.1 (8B)** and **Phi-4 (14B)** models.
- **DeepSeek-R1 boosts accuracy**: The new **R1 Dynamic 1.58-bit** model has been introduced, promising greater accuracy compared to standard bits. More details and a tutorial can be found in the [DeepSeek-R1 blog](https://unsloth.ai/blog/deepseek-r1).
   - Additionally, users can now fine-tune R1 Distill Llama + Qwen models, with [model uploads available](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5).
- **Support for Mistral and Qwen models.**: Unsloth has added support for new models such as **Mistral-Small-24B-2501** and **Qwen2.5**, which can be found in the [Hugging Face collection](https://huggingface.co/collections/unsloth/mistral-small-24b-2501-all-versions-679fe9a4722f40d61cfe627c).
   - Users can also explore models with **1M context** on [Hugging Face](https://huggingface.co/unsloth?sort_models=created&search_models=1m#models).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-GRPO.ipynb)">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device)">Unsloth Documentation</a>: no description found</li><li><a href="https://huggingface.co/unsloth?sort_models=created&search_models=1m#models)">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1336892147567431722)** (38 messages🔥): 

> `Model Merging, DeepSeek V3, User Benefit vs. Corporate Profit, OpenAI developments, Societal Value in AI` 


- **Discussion on Model Merging Limitations**: A user expressed doubt about their ability to reproduce **DeepSeek V3** despite having sufficient funding, stating, *'I am not smart enough.'*
   - Another member shared their commitment to address this challenge, mentioning the need to spearhead conversations around solutions.
- **Concerns About AI Corporations**: There was a debate about the motivations of major AI players like **OpenAI**, with one member criticizing that companies prioritize profit over public good, stating, *'those that put profit over people win at this point'*. 
   - Discussion led to questioning whether smaller companies could compete effectively under such conditions, suggesting that alliances may be necessary for survival.
- **OpenAI's Latest Updates**: A user highlighted an announcement from OpenAI about updates to the **chain of thought** feature for various user tiers, linking to the update [here](https://x.com/OpenAI/status/1887616278661112259).
   - Responses to the update included skepticism about it providing any real purpose, with comments like *'just general ones'*.
- **The Value of AI and User Focus**: A user argued that most competitors fail to focus on the end user’s needs, highlighting that *'no competitor has actually focused on user directly.'*
   - They proposed that emphasizing **societal value** could yield greater benefits than current corporate models.
- **Federated Collective Training Approach**: Members discussed potential paths forward in the AI space, including the idea of *federated collective ways* to train models and challenge existing monopolies.
   - Opinions varied on the feasibility of surpassing larger entities with superior intelligence or through cooperative methodologies, touching on a Marxist perspective of claiming production.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenAI/status/1887616278661112259">Tweet from OpenAI (@OpenAI)</a>: Updated chain of thought in OpenAI o3-mini for free and paid users, and in o3-mini-high for paid users.</li><li><a href="https://github.com/SakanaAI/evolutionary-model-merge">GitHub - SakanaAI/evolutionary-model-merge: Official repository of Evolutionary Optimization of Model Merging Recipes</a>: Official repository of Evolutionary Optimization of Model Merging Recipes - SakanaAI/evolutionary-model-merge
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1336818382820868207)** (188 messages🔥🔥): 

> `Unsloth model training, GRPO and reward functions, Model merging issues, Continued pretraining with LoRA, Adapter performance comparison` 


- **Unsloth model training strategies discussed**: Users shared their experiences with unsloth, emphasizing the impact of pretraining and instruction data on model performance.
   - One user noted that their model performed better before additional finetuning, highlighting the challenges with further training.
- **GRPO and reward functions effectiveness**: There was a discussion on teaching models to follow specific formats using GRPO, with a focus on reward functions for guiding output.
   - Participants suggested that combining a reward function with supervised fine-tuning (SFT) could enhance the training outcome.
- **Challenges with model merging**: A user expressed concerns about merging LoRA adapters with base models, revealing that post-merging training led to poor outputs.
   - It was suggested to keep using LoRA adapters for training instead of merging until fully confident in model performance.
- **Issues with continued pretraining using adapters**: Participants encountered issues where the adapter failed to continue training the lm_head and embed tokens during resumed training.
   - This raised questions about whether merging would solve these issues or if training could proceed effectively with just the adapter.
- **Optimizations and praise for Unsloth**: Users expressed admiration for the optimizations in the Unsloth framework, mentioning the efficiency improvements.
   - The release of RL features was highlighted as a significant enhancement that many users were looking forward to.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF?show_file_info=Qwen2.5-Math-1.5B-Instruct-Q6_K.gguf>">bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/">Hugging Face</a>: The AI community building the future. Hugging Face has 287 repositories available. Follow their code on GitHub.</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks>">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1, Mistral, Phi-4 &amp; Gemma 2 LLMs 2-5x faster with 70% less memory</a>: Finetune Llama 3.3, DeepSeek-R1, Mistral, Phi-4 &amp; Gemma 2 LLMs 2-5x faster with 70% less memory - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

yaska0971: Name strings tooooooooooo long. Please shorten it
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1337099685068013620)** (6 messages): 

> `Realistic AI Research Domains, TPU Research Cloud with JAX, OpenMoE Project, Pretraining Small Transformers` 


- **Exploring Realistic AI Research Avenues**: An independent researcher seeks realistic domains to start AI/ML research without significant funding, citing limitations on pretraining large models.
   - Suggestions for areas of focus include **comparative studies**, creating new data generation models, or improving prompts for **LLMs**.
- **Value of Learning JAX for TPU Access**: A member recommends learning **JAX** for access to the **TPU Research Cloud**, suggesting it could be quite beneficial.
   - They provided a [link to the TPU Research Cloud](https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform) application form.
- **OpenMoE: An Example from the Community**: A member highlighted the **OpenMoE** GitHub repository as a relevant example of someone conducting research in the field of Mixture-of-Experts models.
   - The repository is led by a researcher who has progressed to working at **DeepMind**, demonstrating success in this area.
- **Pretraining Small Transformers on TinyStories**: Another member suggested pretraining small transformers on the **TinyStories dataset** as a potential research option.
   - This could offer new pathways for independent projects without needing extensive resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2311.10770">Paper page - Exponentially Faster Language Modelling</a>: no description found</li><li><a href="https://github.com/XueFuzhao/OpenMoE">GitHub - XueFuzhao/OpenMoE: A family of open-sourced Mixture-of-Experts (MoE) Large Language Models</a>: A family of open-sourced Mixture-of-Experts (MoE) Large Language Models - XueFuzhao/OpenMoE</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform">Accelerate your research with the TPU Research Cloud</a>: Please complete the following questions for your application to be considered.
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1336798894218281130)** (1 messages): 

> `Maxfield Introduction, Community Engagement Initiatives, Feature Request Board, Showcasing Researcher Progress` 


- **Maxfield, the New Community Chief, Introduces Himself**: Maxfield, the new Chief Community Guy at Stability, introduces himself and expresses his commitment to improving community engagement after heavy involvement in AI media generation since 2022.
   - He highlights that he previously contributed at Civitai and acknowledges that community engagement has been **lackluster** lately.
- **Two New Initiatives to Boost Engagement**: Maxfield announces two initiatives aimed at improving communication and sharing community interests with features like a **feature request board** for suggestions on models and tools.
   - He emphasizes that the goal is to ensure that the community's voices are heard and that initiatives will cater to hobbyists and professionals alike.
- **Encouraging Creators to Share Progress**: Maxfield plans to promote **transparency** by encouraging Stability's researchers and creators to share updates on their projects and developments.
   - He believes that the fantastic work being done shouldn't be kept a secret and should be shared more widely with the community.


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1336788750096662539)** (459 messages🔥🔥🔥): 

> `Stability AI Updates, Model Compatibility, AI Prompting Techniques, Community Dynamics, AI Subscriptions and Costs` 


- **Discussion on Stability AI features and subscription costs**: Members discussed the 'private images' option in the Max subscription, questioning if it implied NSFW content, with some sharing their experiences with the service.
   - Others highlighted the costs involved in using cloud services for model training, comparing them to local electricity costs.
- **Issues with downloading from Civitai**: Several users reported encountering Error 1101 when trying to download models from Civitai, indicating possible server issues.
   - The community shared their frustrations over the downtime and difficulties accessing models.
- **Latent space tools and model training**: A user expressed confusion over the complexity of tools for swapping latent space parameters, indicating a need for more intuitive solutions.
   - Discussions included potential implementations for newer diffusion models and challenges with running existing architectures.
- **General attitudes towards different AI models**: Participants shared their experiences and opinions regarding various AI models, including Stability AI and Midjourney, reflecting mixed feelings about subscription models and community dynamics.
   - There was a contemplation of the value of entry costs against the utility of using specific AI models.
- **AI prompting techniques and tools**: A user sought insights into prompting techniques for generative models, while others suggested using external tools to generate prompts.
   - Discussion included the difficulty of adjusting to different AI model requirements and generating satisfactory prompts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ollama.com/brxce/stable-diffusion-prompt-generator">brxce/stable-diffusion-prompt-generator</a>: Stable Diffusion Prompt Generator</li><li><a href="https://tenor.com/view/creep-hands-adventure-time-deer-remove-the-gloves-gif-15274634">Creep Hands GIF - Creep Hands Adventure Time - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lu.ma/fyu8iqnk">Multimodal AI Agents - Hackathon · Luma</a>: Gen AI AgentsCreatorsCorner, collaborating with Google Deepmind, Weights &amp; Biases, Together.ai, Stytch, Senso, LlamaIndex and others enthusiastically…</li><li><a href="https://github.com/NeuralNotW0rk/LoRAW">GitHub - NeuralNotW0rk/LoRAW: Flexible LoRA Implementation to use with stable-audio-tools</a>: Flexible LoRA Implementation to use with stable-audio-tools - NeuralNotW0rk/LoRAW</li><li><a href="https://purplesmart.ai/">Expanding the frontiers of AI creativity - PurpleSmartAI</a>: no description found
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1336794325979238410)** (3 messages): 

> `Gemini 2.0 Flash, Windsurf Next Beta, Windsurf 1.2.6 Patch Fixes, Cascade Web Search` 


- **Gemini 2.0 Flash speeds up coding**: The new **Gemini 2.0 Flash** is now available on Windsurf, consuming only **0.25** user prompt credits per message and flow action credits per tool call.
   - *Blazingly fast* and efficient, it is limited in tool calling ability but excels at answering codebase-related questions.
- **Join Windsurf Next for early features**: Users can gain early access to the latest features of **Windsurf Next** by downloading the beta from [this link](https://codeium.com/windsurf/download-next).
   - The beta enables exploration of new AI capabilities while allowing users to switch between Next and Stable versions as needed.
- **Windsurf 1.2.6 Patch addresses credit issues**: The latest **Windsurf 1.2.6 Patch** fixes problems with partial credits during the transition to flex credits, detailed in the [full changelog](https://www.codeium.com/changelog).
   - This patch enhances user experience by ensuring smoother credit transitions for actions.
- **Cascade's new web search capabilities**: Cascade can now perform web searches automatically or via user commands like **@web** and **@docs**, making it versatile for obtaining real-time information.
   - This functionality includes URL input support and incorporates web context for improved responses, with a web search costing **1 flow action credit**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1887235006374035966">Tweet from Windsurf (@windsurf_ai)</a>: Gemini 2.0 Flash is now available in Windsurf!From our testing, Flash is:⚡ blazingly fast💪 efficient - only consumes 0.25X credits🧠 limited in its tool calling ability, but great for questions about...</li><li><a href="https://codeium.com/windsurf/download-next">Thank you for downloading Windsurf Editor</a>: Tomorrow&#x27;s editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.</li><li><a href="https://codeium.com/blog/windsurf-next">Windsurf Next Launch</a>: Introducing Windsurf Next, our opt-in prerelease version of Windsurf.</li><li><a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1336832439594651759)** (31 messages🔥): 

> `Codeium Jetbrains Plugin Issues, DeepSeek Feature Request, Function Length Display in CodeLens, Educational Email Discounts, Version Updates and Bug Reports` 


- **Codeium Jetbrains Plugin Faces Criticism**: Users expressed frustration with the Codeium Jetbrains plugin, stating that it often fails to respond and requires frequent restarts, with one switching back to Copilot for reliability.
   - An error related to file access was reported in PhpStorm, indicating ongoing problems with the plugin's performance.
- **Request for DeepSeek Feature Implementation**: A user requested the addition of the DeepSeek feature to Codeium, emphasizing its potential benefits.
   - Another member encouraged this request, prompting users to officially submit feature requests through designated channels.
- **Codelens and Function Lengths Inquiry**: A member asked if it's possible to show the length of a function, which led to the identification of the feature as Codelens.
   - However, the specific implementation of adding logic to Codelens within VSCode remains unclear among users.
- **Clarification on Educational Email Discounts**: Discussion arose about educational email eligibility for discounts, with users clarifying that emails must end with .edu to qualify.
   - Concerns were raised about the detection methods used, with some speculating that eligibility is specific to US educational institutions.
- **Credits Usage in Codeium Models**: A user inquired whether using the Codeium Premier model in VSCode requires credits, to which it was confirmed that chat features do not consume credits.
   - It was clarified that none of the models in the extension are connected to credits, ensuring users can utilize them freely.


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1336789037784105073)** (345 messages🔥🔥): 

> `Issues with Windsurf Performance, Gemini Flash vs Sonnet, Usage of Multiple AI Models, Windsurf Installation and Login Problems, User Experience with Cascading Files` 


- **Issues with Windsurf Performance**: Many users reported performance issues with Windsurf, particularly noting problems with calls to models like O3-mini and Gemini Flash that finish prematurely without providing complete suggestions.
   - One user expressed frustration about the need to continuously prompt the model to 'continue', leading to concerns over wasted credits.
- **Gemini Flash vs Sonnet**: Some users are comparing Gemini Flash and Sonnet, with Gemini Flash noted for its faster speeds and lower costs, but still being behind Sonnet in terms of overall quality.
   - As per discussions, Claude remains a preferred option for many due to its higher performance metrics on coding challenges.
- **Usage of Multiple AI Models**: There are discussions about which AI models to use depending on the task, with some users advocating for DeepSeek for debugging and Claude for better quality outputs.
   - Windsurf's agentic capabilities with different models were debated, indicating variability in performance based on the selected model.
- **Windsurf Installation and Login Problems**: A user reported difficulties logging into the Windsurf IDE despite having a Pro subscription, facing issues with trial activation and authentication.
   - Another instance noted error messages about version mismatches after attempting to reinstall the IDE.
- **User Experience with Cascading Files**: Users expressed the tediousness of manually adding multiple files to the Cascade chat in their Angular projects, seeking better methods for integration.
   - Suggestions were made for using right-click options to copy file paths and include them more efficiently in discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://windsurf‑stable.codeiumdata.com/wVxQEIWkwPUEAGf3/apt">no title found</a>: no description found</li><li><a href="https://codeium.com/changelog/windsurf-next">Windsurf Next Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Next extension.</li><li><a href="https://codeium.canny.io/feature-requests">Feature Requests | Codeium</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://open-vsx.org/extension/sr-team/vscode-clangd-cmake">Open VSX Registry</a>: no description found</li><li><a href="https://codeium.canny.io/feature-requests/p/auto-commit-message">Auto commit message | Feature Requests | Codeium</a>: Generate Commit Messages from Committed File Context</li><li><a href="https://codeium.canny.io/feature-requests/p/roll-over-of-pro-credits">Roll-over of Pro Credits | Feature Requests | Codeium</a>: Unused Premium User Prompt Credits and Premium Flow Action Credits roll-over to the next month</li><li><a href="https://x.com/kevinhou22/status/1886827501004931511">Tweet from Kevin Hou (@kevinhou22)</a>: we love docs! 📖 I&#39;m working on improving / adding more @ docs shortcuts to @windsurf_ailmk what you want and I&#39;ll add as many as I can... 🧵also shoutout @mintlify for auto-hosting all docs w...</li><li><a href="https://www.reddit.com/r/Codeium/comments/1ihn6gp/submit_your_docs_suggestions_to_head_of_product/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1336788439122837635)** (337 messages🔥🔥): 

> `Hiring Update, Aider Error Handling, DeepSeek and Gemini Models, LLM Editing Formats, Pen Testing with LLMs` 


- **Congrats on the Job Offer!**: A member announced they received a job offer, highlighting a significant pay increase and less downtime compared to their current role.
   - They expressed excitement about the new opportunity and the potential for increased earnings to fund their AI projects.
- **Common Aider Error Encountered**: A user reported an issue with Aider indicating an invalid port error while trying to load model metadata.
   - Another member suggested overriding the default model metadata file as a workaround to resolve this error.
- **Discussion on DeepSeek and Gemini Models**: Users discussed DeepSeek's recent inconsistencies and the performance of the Gemini models, particularly mentioning difficulties with the 1206 model.
   - Specifically, they noted that Google's models by default use a unique editing format (udiff) which differs from other models.
- **Understanding LLM Editing Formats**: Users spoke about the various edit formats used by Aider, distinguishing between standard diff formats and Aider's own UDIFF syntax.
   - They clarified that Aider automatically uses udiff for Google models while maintaining different defaults for others.
- **Pen Testing Project with LLMs**: One member shared their project creating a pen testing setup using LLMs, highlighting how two models work together in a simulated hacking environment.
   - Despite high token usage, they mentioned the potential financial benefits, as professional pen tests can be extremely lucrative.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers/perplexity">Perplexity AI (pplx-api) | liteLLM</a>: https://www.perplexity.ai</li><li><a href="https://aider.chat/docs/usage/lint-test.html#linting">Linting and testing</a>: Automatically fix linting and testing errors.</li><li><a href="https://aider.chat/docs/more/edit-formats.html">Edit formats</a>: Aider uses various “edit formats” to let LLMs edit source files.</li><li><a href="https://openrouter.ai/perplexity/sonar-reasoning">Sonar Reasoning - API, Providers, Stats</a>: Sonar Reasoning is a reasoning model provided by Perplexity based on [DeepSeek R1](/deepseek/deepseek-r1).It allows developers to utilize long chain of thought with built-in web search. Run Sonar Reas...</li><li><a href="https://deepclaude.com/docs">DeepClaude</a>: no description found</li><li><a href="https://www.litellm.ai/">LiteLLM</a>: LiteLLM handles loadbalancing, fallbacks and spend tracking across 100+ LLMs. all in the OpenAI format</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>: Configuring advanced settings for LLMs.</li><li><a href="https://www.youtube.com/watch?v=MQw_zncxk-E">Vercel&#39;s Guillermo Rauch on AI and the Future of Coding - Ep. 47</a>: Read Dan Shipper&#39;s essay on the allocation economy: https://every.to/chain-of-thought/the-knowledge-economy-is-over-welcome-to-the-allocation-economyGuillerm...</li><li><a href="https://www.youtube.com/watch?v=pb6GtL0WFT8">Autonomous AI in Action 💪 | Live Codestream with Aider &amp; Deepseek v3 🧠</a>: In this experimeny, Deepseek v3 (via Aider) is in charge of building a project with minimal human intervention. The AI is working on a Summarizer app that mo...</li><li><a href="https://github.com/Aider-AI/aider/issues/3159">Add tree-sitter-hcl-tags.scm for terraform repomap generation · Issue #3159 · Aider-AI/aider</a>: This is a first pass at a tree-sitter-hcl-tags.scm to enable repomap for terraform repositories. I&#39;m able to use it locally after adding .tf extensions to grep-ast and identifying them as hcl. Usi...</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider is AI pair programming in your terminal</li><li><a href="https://github.com/getAsterisk/deepclaude/issues/13">Aider&#39;s benchmark is explicitly not about using R1 thinking tokens (and says that using them did worse) · Issue #13 · getAsterisk/deepclaude</a>: Hey deepclaude folks, I&#39;m a bit confused about why you are prominently citing aider&#39;s R1+Sonnet benchmarking results. The blog article and twitter post about these results explicitly state tha...</li><li><a href="https://github.com/Aider-AI/aider/issues/2052">SDK not that good · Issue #2052 · Aider-AI/aider</a>: Hi, I really love your tool—I&#39;m using it, and I think it&#39;s great. However, when I try to wrap it in Python, it&#39;s not as easy as I expected. While the documentation shows how to use coder.r...</li><li><a href="https://github.com/jj-vcs/jj">GitHub - jj-vcs/jj: A Git-compatible VCS that is both simple and powerful</a>: A Git-compatible VCS that is both simple and powerful - jj-vcs/jj</li><li><a href="https://t.co/ss4DAzMi4J">GitHub - lumina-ai-inc/chunkr: Vision model based document ingestion</a>: Vision model based document ingestion. Contribute to lumina-ai-inc/chunkr development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/issues/2879">Bug: Creating files named with the file extension only without the filename. · Issue #2879 · Aider-AI/aider</a>: It suggests the correct filenames but then it will generate a file named php instead of install.php or sql instead of migration.sql</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://github.com/Aider-AI/aider/issues/3139#issue-2832352562">Aider creates files using random strings as filenames · Issue #3139 · Aider-AI/aider</a>: Issue Using o3-mini to prompt and it&#39;s been using very weird filenames like 2. New file for modular integration of the embedding worker New file (empty file) ────────────────────────────── I think...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1336792416727990293)** (23 messages🔥): 

> `Aider Support for Agents, Staging Changes in Aider, Commit Messages with R1, Model Configuration Issues, Architect Mode Functionality` 


- **Aider Might Support Agents Soon**: A user inquired whether Aider will support agents at any point, indicating a potential development or update in the tool's capabilities.
   - This reflects ongoing interest in expanding Aider's functionality to enhance user experience.
- **Staging Changes Not Yet Available**: One user asked if there's a way to stage changes instead of committing them directly in Aider, suggesting a desire for more granular control.
   - This indicates user interest in workflow features that may simplify the versioning process.
- **Troubles with <think> Tokens in Commits**: A user shared concerns about their commit messages filled with `<think>` tokens from using R1 via Together.ai, seeking tips on configuration.
   - Recommendations included configuring model settings appropriately to minimize these tokens in commit messages.
- **Help with Internal OpenWeb UI Instance**: A user requested guidance on using a JWT API key from an internal OpenWeb UI instance with Aider, noting that standard API keys weren't available.
   - This highlights the challenges with internal tools that restrict direct API access, complicating integration efforts.
- **Concerns about Architect Mode**: A user expressed confusion over Architect mode's behavior, stating it doesn't allow them to dictate the next steps as intended.
   - Others noted that using the `/ask` command can achieve desired control without needing to adjust the mode.



**Link mentioned**: <a href="https://aider.chat/docs/config/reasoning.html">Reasoning models</a>: How to configure reasoning model settings from secondary providers.

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1336806433760612363)** (2 messages): 

> `Gemini 2.0, Open Deep Research, HuggingFace, Agent frameworks` 


- **Gemini 2.0 Launches on LMSYS**: The latest model, **Gemini 2.0**, has been featured on [LMSYS](https://lmarena.ai) showcasing its advancements in capabilities.
   - This launch aims to enhance the discussion around next-gen models in the AI community.
- **HuggingFace Unveils Open Deep Research Clone**: HuggingFace researchers have created an open-source clone of **DeepResearch**, detailed in their [blog post](https://huggingface.co/blog/open-deep-research).
   - This initiative emphasizes the importance of **agent frameworks** and sets out the **GAIA benchmark**, paving the way for community contributions.



**Link mentioned**: <a href="https://huggingface.co/blog/open-deep-research">Open-source DeepResearch – Freeing our search agents</a>: no description found

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1336792980035735775)** (276 messages🔥🔥): 

> `Gemini 2.0 Pro, OpenAI vs DeepSeek, AI for Coding, Chatbot Aggregators, AI Model Comparisons` 


- **Gemini 2.0 Pro Excitement**: Users expressed excitement about the capabilities of **Gemini 2.0 Pro**, highlighting its **2 million token context** which allows for complex interactions and creative writing tasks.
   - However, concerns were raised regarding its usability compared to free alternatives like **Google AI Studio**, which offers extensive customization.
- **The Great Chatbot Showdown**: A potential chess match between **DeepSeek** and **ChatGPT** sparked interest, with users speculating on the outcomes given the models' limitations on reasoning.
   - There was a humorous contrast drawn between the pricing of DeepSeek’s **$1 chess game** versus OpenAI's **$100 chess game**.
- **AI for Coding Recommendations**: In discussions about coding, members recommended **Gemini Flash 2.0** and **Microsoft Copilot** for their features and cost-effectiveness, particularly for advanced mathematics.
   - Users noted that **Copilot** offers a free trial, making it easier to explore without immediate financial commitment.
- **AI Automated Coding Solutions**: Conversations shifted towards finding an **automated way to code** with minimal dependencies, seeking solutions that reduce terminal time.
   - The goal was to discover the most agentic approach for coding tasks using AI, emphasizing user-friendly environments.
- **AI Model Comparisons and Experiences**: Users shared mixed experiences with various AI models, noting that **Gemini 2.0** and **Sonnet 3.5** performed better in user tasks but had different features.
   - The consensus was that task requirements greatly influenced model choice, with attention to both capabilities and costs.



**Link mentioned**: <a href="https://tenor.com/view/fire-writing-gif-2088247993237804628">Fire Writing GIF - Fire writing - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1336830335719182388)** (5 messages): 

> `Deep Research chat for Plus users` 


- **Anticipation for Deep Research in Plus Version**: Several members expressed eagerness for the **Deep Research** chat feature to be available for **Plus users** soon.
   - Members specifically noted their **need** for it, hoping for a release in the coming days.
- **Conversation on not knowing details**: A member remarked about a previous comment stating, 'I never knew,' indicating a lack of information on the topic.
   - Another member responded with skepticism, saying, 'It's not, what do you mean?'
- **Request for Deep Research chat sharing**: A member inquired if anyone had shared information about **Deep Research** chats, obviously looking for insights.
   - This inquiry prompted others to express similar anticipation regarding the feature coming to Plus subscriptions.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1336825419885842493)** (6 messages): 

> `Response Length Control, Undesired Behavior in AI Models, Input Influencing Output` 


- **Strategies to Ensure Optimal Response Length**: One member suggested using Python to count words and iterate to ensure better response length, though this may impact creativity.
   - They noted that more input generally results in more output but acknowledged the challenge of character counting without external assistance.
- **Editing for Desired Output**: Another member emphasized the importance of editing inputs using the edit button to sculpt the AI's output effectively.
   - They advised adjusting your input until satisfied before proceeding to ensure coherent context in the conversation.
- **Undesired Behaviors Persist**: A member expressed frustration with AI models that repeat undesired behaviors unless actively controlled.
   - They highlighted the need for strategies to manage and mitigate these behaviors for a better user experience.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1336825419885842493)** (6 messages): 

> `Controlling AI Response Length, Managing Undesired AI Behavior` 


- **Methods to Ensure AI Response Length**: A member suggested using Python to count words and iterate, though warned it might impact **creativity**. They noted that more content typically results in longer responses, but counting **characters** is difficult for AI without assistance.
- **Editing to Sculpt AI Output**: Another member mentioned that using the edit button can help control undesired behavior, allowing users to refine prompts until satisfied. This method sets the stage for the next interaction in the 'context' or 'conversation' chain.


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1336788595117391872)** (282 messages🔥🔥): 

> `Cursor IDE Updates, Gemini 2.0 Performance, Clipboard Comparison Tools, MCP Server Configurations, Context Limitations in AI Models` 


- **Cursor IDE Updates and Features**: Users shared insights on Cursor IDE updates, specifically the introduction of GitHub agents and discussion about the architect feature that could further enhance productivity.
   - There was also mention of challenges with running commands within the Composer tool, indicating a potential bug after recent updates.
- **Gemini 2.0 Performance Evaluation**: Several users expressed satisfaction with Gemini 2.0, noting its solid performance for self-learning tasks, although some feel it isn't superior to Sonnet for coding.
   - There were discussions about the model's affordability and effective context use, contributing to its appeal for large codebases.
- **Clipboard Comparison Tools Suggestions**: Participants provided suggestions for clipboard comparison tools, highlighting the VSCode extension that allows comparisons against the clipboard content.
   - Users compared capabilities of VSCode's local history and suggested using tools like Timeline for better efficiency akin to JetBrains.
- **MCP Server Configurations and Documentation**: A user sought help with MCP server configurations and accessing necessary keys for Supabase, shared that some keys provide limited access.
   - The community discussed configurations and the need for better context management in Cursor, particularly for complex projects.
- **Context Limitations in AI Models**: Concerns were raised about context limitations in Cursor, with a preference for models like Cline or Google models that provide larger contexts.
   - The impact of context size on the effectiveness of AI models was debated, specifically how larger context windows could enhance performance in broader applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://livebench.ai/">LiveBench</a>: no description found</li><li><a href="https://momentic.ai/">AI Testing Tool | Automated AI Testing - Momentic</a>: Supercharge your QA process with Momentic&#x27;s advanced AI tools for automated testing. Ship faster with reliable, AI-driven tests.</li><li><a href="https://marketplace.visualstudio.com/items?itemName=ryu1kn.partial-diff">Partial&#32;Diff&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Compare&#32;(diff)&#32;text&#32;selections&#32;within&#32;a&#32;file,&#32;across&#32;files,&#32;or&#32;to&#32;the&#32;clipboard</li><li><a href="https://docs.fireworks.ai/deepseek/general-deepseek">no title found</a>: no description found</li><li><a href="https://forum.cursor.com/t/o3-mini-is-live-what-version-are-we-getting/46674">O3-mini is LIVE! What version are we getting?</a>: First off, shoutout to the Cursor team for pushing this out so quickly!  The team stating that their devs still prefer Sonnet for most tasks (which surprised them). (source: x.com)  According to the O...</li><li><a href="https://forum.cursor.com/t/model-specific-rules/47175">Model-specific rules</a>: The new cursor rules system is great, it would also be good if we could have rules for specific models.</li><li><a href="https://forum.cursor.com/t/new-diff-is-shown-between-each-line-of-the-previewed-code/36903">New Diff is shown between each line of the previewed code</a>: Describe the Bug What’s new is shown between each line of the previewed code  Steps to Reproduce just ask Cursor  Expected Behavior Normal code preview  Screenshots / Screen Recordings    Operating Sy...</li><li><a href="https://x.com/karpathy/status/1886192184808149383">Tweet from Andrej Karpathy (@karpathy)</a>: There&#39;s a new kind of coding I call &#34;vibe coding&#34;, where you fully give in to the vibes, embrace exponentials, and forget that the code even exists. It&#39;s possible because the LLMs (e.g...</li><li><a href="https://github.com/daniel-lxs/mcp-starter/releases/">Releases · daniel-lxs/mcp-starter</a>: Contribute to daniel-lxs/mcp-starter development by creating an account on GitHub.</li><li><a href="https://github.com/daniel-lxs/mcp-starter/releases/tag/v0.1.0">Release v0.1.0 · daniel-lxs/mcp-starter</a>: Initial Release</li><li><a href="https://github.com/microsoft/vscode-docs/issues/7284">Document &quot;Compare active file with clipboard&quot; functionality · Issue #7284 · microsoft/vscode-docs</a>: Introduced in VS Code 1.19, you can compare the active file with the contents of the clipboard. Command: File: Compare Active File with Clipboard (workbench.files.action.compareWithClipboard) Keybi...</li><li><a href="https://github.com/robert-at-pretension-io/mcp">GitHub - robert-at-pretension-io/mcp: code</a>: code. Contribute to robert-at-pretension-io/mcp development by creating an account on GitHub.</li><li><a href="https://github.com/daniel-lxs/mcp-perplexity">GitHub - daniel-lxs/mcp-perplexity</a>: Contribute to daniel-lxs/mcp-perplexity development by creating an account on GitHub.</li><li><a href="https://github.com/daniel-lxs/mcp-starter">GitHub - daniel-lxs/mcp-starter</a>: Contribute to daniel-lxs/mcp-starter development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1336796638928306320)** (247 messages🔥🔥): 

> `Perplexity AI Focus Mode, Query Handling in Perplexity Pro, R1 vs. Other Models, Performance Issues with Deepseek, User Concerns regarding Model Specifications` 


- **Perplexity AI's Focus Mode Temporarily Removed**: Users discussed the recent removal of the Focus Mode feature in Perplexity AI, with varying reports on whether this change is ongoing or temporary, as stated in some change logs.
   - Some users expressed frustration over this change and noted that it complicates their ability to specify the source of information, such as requiring prompts to mention Reddit explicitly.
- **Clarifications on Model Usage in Pro**: Questions were raised about how Pro mode interacts with model choices like Claude 3.5 and whether it utilizes reasoning from R1, with insights suggesting that Pro may not use models end-to-end.
   - It was noted that the actual processing involves undisclosed models for initial searches before passing to selected models like Claude or R1 for final answers.
- **User Experiences with Performance of R1 and Deepseek**: Users have compared the R1 reasoning capabilities of Perplexity with those on Deepseek, noting that Perplexity's version seems to generate more reliable outputs under certain conditions.
   - Concerns were raised about the speed and quality difference between the available models, particularly with references to the processing power of different configurations.
- **Issues with Stability in AI Applications**: Some users reported experiencing slow performance and stability issues with Perplexity, particularly with the Android app and while using the O3 mini model.
   - Complaints were directed towards muting discrepancies and inefficiencies in model interactions, prompting discussions on user support responsiveness.
- **Need for Clear Communication from AI Developers**: A common sentiment among users was the desire for greater transparency from Perplexity AI regarding model specifications and updates, especially concerning operational changes.
   - Users suggested that clarity about model modifications could enhance user experience and reduce confusion when interacting with multiple AI functionalities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/george-droyd-gif-12399766756846341904">George Droyd GIF - George droyd - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/monnef/status/1887231954543575135">Tweet from mennof (@monnef)</a>: Hey AI fans! 🤖 Just wrapped up some DeepSeek R1 testing with a puzzle prompt (3 runs per service)!The tea: @perplexity_ai is blazing fast at processing, while @cursor_ai takes its time to ponder thin...</li><li><a href="https://by-ai-monnef-9ff5d9c2460ae15d70e737f77eab719c6e8a4c64c2f99ca1c2.gitlab.io/2025/pplx-tech-props/">Perplexity Tech Props</a>: no description found</li><li><a href="https://forms.gle/zYnhGFj3FKACoN27A">Ski Equipment Rental Project Questionnaire </a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1336799036077899857)** (22 messages🔥): 

> `Tesla Robotaxi Launch, AI Skills Development Opportunities, USA vs China AI Race, Deepfake Technology from ByteDance, Trans Athlete Executive Order` 


- **Tesla Robotaxi Launch set for June**: Perplexity AI announced that **Tesla Robotaxi** will launch in **June**, marking a significant advancement in autonomous technology.
   - A YouTube video shared on this topic discussed the implications of this launch for the AI and automotive industries.
- **Explore AI Skills Development**: A comprehensive thread was shared discussing various **opportunities** for developing AI skills suitable for all levels from beginner to master.
   - The discussion provides insights on harnessing one's true potential within the AI field.
- **Detailed Overview of USA vs China AI Race**: An intricate thread on the **USA vs China AI Race** presents collected information, with sources provided for verification.
   - The author highlights the challenges in obtaining openly acknowledged information in this competitive landscape.
- **ByteDance Releases New Deepfake Technology**: A report on **ByteDance's** latest endeavor revealed the release of a deepfake tool, which raises various discussions on ethical implications.
   - As part of this release, the community speculated on potential uses and misuses of deepfake technology.
- **Executive Order Bans Trans Athletes in Sports**: A recent executive order banning **Trans athletes** from competing in certain sports generated substantial debate within the community.
   - Members discussed the broader implications of this order on the sports industry and civil rights.



**Link mentioned**: <a href="https://www.youtube.com/embed/mE1aAZAIX40">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1336791540982616084)** (7 messages): 

> `Perplexity API usage, Sonar Pro Reasoning devs, Image uploading limitations, Monthly cost limits and invoicing` 


- **Questions about Monthly Cost Limits**: A member inquired whether they can set a hard limit on how much money is used per month with the Perplexity API.
   - They also asked if invoices are sent for costs incurred or if charges need to be added manually.
- **Exploring Image Upload Workarounds**: A new user expressed interest in the Perplexity API for an app they are building but noted that the current API seems to lack image uploading capabilities.
   - They proposed a workaround using Claude for detailed descriptions before sending prompts to Perplexity for output.
- **Urgent Request for Sonar Pro Reasoning Devs**: Multiple messages indicated an urgent need to contact the Sonar Pro reasoning developers due to a security issue discovered by a member.
   - Another member directed them to send an email to api@perplexity.ai for assistance.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1336847310683897897)** (14 messages🔥): 

> `DeepSeek Insurance, Kluster integration issues, Qwen model deprecation, Website downtime update` 


- **DeepSeek Insurance Now Covers No Completion Tokens**: OpenRouter will now insure DeepSeek R1 requests that receive no completion tokens, ensuring no charge even if the upstream provider does.
   - The completion rate for Standard DeepSeek R1 has improved from **60%** to **96%** over time.
- **Kluster Integration Issues Resolved**: A user explained a situation where completion tokens were delayed by Kluster, leading to unexpected charges despite apparent timeouts on OpenRouter's end.
   - *They discovered that Kluster was failing to cancel requests when timing out,* but this issue has since been addressed.
- **Qwen Models Being Deprecated by Novita**: Novita will be deprecating their **Qwen/Qwen-2-72B-Instruct** model, with OpenRouter disabling it around the same time.
   - Users should make sure to transition away from this model before the deprecation date.
- **OpenRouter Website Experiences Downtime**: OpenRouter experienced a minor downtime due to their authentication provider being down, affecting website access but not the API.
   - The issue was resolved within approximately **15 minutes** and services were restored.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/deepseek/deepseek-r1">R1 - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:nitro">R1 (nitro) - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1337098797960003678)** (1 messages): 

> `Y CLI Development, Terminal Enthusiasm, Chat Data Management, MCP Client Support, Deepseek-r1 Integration` 


- **Y CLI emerges as an open router chat alternative**: A personal project, **Y CLI**, aims to provide a web chat alternative with all chat data stored in **single jsonl files**.
   - You can check out the project on its [GitHub page](https://github.com/luohy15/y-cli).
- **MCP client support showcased**: The project includes support for an MCP client, demonstrated in this [asciinema recording](https://asciinema.org/a/701901) capturing its functionality on macOS.
   - This recording received **4 views** and showcases **xterm-256color** and **zsh** in action.
- **Deepseek-r1 reasoning support added**: Another feature of **Y CLI** is the **Deepseek-r1** reasoning content support, evidenced in this [asciinema recording](https://asciinema.org/a/701903).
   - This demo also runs on macOS with **2 views** and supports the **xterm-256color** and **zsh** terminal setup.
- **Github encourages contributions**: Developers are invited to contribute to **Y CLI** via its [GitHub repository](https://github.com/luohy15/y-cli).
   - The page highlights the ongoing development efforts and user contributions illustrated in this [GitHub overview](https://opengraph.githubassets.com/fcfbadfea6316b0b3a67649871dbdbbacd8aaa18e7894691e09a340a8a6b914d/luohy15/y-cli).
- **A call for terminal fans**: The developer expressed interest in finding fellow **terminal fans** within the community.
   - The project aims to attract those who appreciate terminal-based tools and configurations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://asciinema.org/a/701901">y-cli mcp client</a>: https://github.com/luohy15/y-cli</li><li><a href="https://asciinema.org/a/701903">y-cli reasoning content</a>: https://github.com/luohy15/y-cli</li><li><a href="https://github.com/luohy15/y-cli">GitHub - luohy15/y-cli</a>: Contribute to luohy15/y-cli development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1336790898289279177)** (242 messages🔥🔥): 

> `DeepInfra issues, Gemini 2.0 Flash readiness, OpenRouter authentication service, Error handling with models, Provider performance discrepancies` 


- **DeepInfra experiencing failures**: Users reported that **DeepInfra** is currently failing to return responses 50% of the time due to an increase in processing delays.
   - Some users are seeing zero token completions when utilizing DeepInfra with applications like SillyTavern.
- **Gemini 2.0 Flash model integration concerns**: There are discussions around issues with the **Gemini 2.0 Flash** model regarding its incompatibility with tool calling.
   - Users are filing issues as they encounter errors stating that tool invocation must have a return result, but it works fine with other models.
- **Authentication service downtime**: OpenRouter experienced downtime due to issues with their **authentication service** provided by Clerk, Inc.
   - Although the website faced challenges, the API remained operational for users, with updates being shared regarding the status.
- **Error identification with models**: Users reported discrepancies and errors when utilizing different models, such as **Mistral** and **Novita AI**.
   - Issues include one model returning unusually high token counts and another causing frequent processing failures.
- **General discussions about provider performances**: The community is sharing observations about performance differences between models and providers, including suggestions for improvements.
   - There is a call for better mechanisms to handle errors and optimize responses to streamline user experiences with AI models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/77762483/the-caller-does-not-have-permission-when-creating-api-key">&quot;The caller does not have permission&quot; when creating API key</a>: I&#x27;m using MakerSuite with Gemini and I deleted an API Key. I went to create a new one, but I&#x27;m getting an error saying the caller does not have permission. What does that mean and how can I ...</li><li><a href="https://openrouter.ai/">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://x.com/OfficialLoganK/status/1887178282950426914">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: @HCSolakoglu Vertex customers tend to skew larger enterprise customers and have flexibility to negotiate things like bulk discounts, etc. This does not apply to the Gemini Developer API, everyone pays...</li><li><a href="https://openrouter.ai/google/gemini-2.0-flash-001">Gemini Flash 2.0 - API, Providers, Stats</a>: Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1. Run Gemini Flash 2.0 with API</li><li><a href="https://openrouter.ai/provider/google-ai-studio">Google AI Studio | OpenRouter</a>: Browse models provided by Google AI Studio</li><li><a href="https://share.cleanshot.com/6rvDHCY5">CleanShot 2025-02-06 at 11 .21.37</a>: Screenshot uploaded to CleanShot Cloud</li><li><a href="https://openrouter.ai/docs/features/provider-routing">Provider Routing — OpenRouter | Documentation</a>: Route requests to the best provider</li><li><a href="https://share.cleanshot.com/jhf5tq3D">CleanShot 2025-02-06 at 10 .58.39</a>: Screenshot uploaded to CleanShot Cloud</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider/issues">OpenRouterTeam/ai-sdk-provider</a>: The OpenRouter provider for the Vercel AI SDK contains support for hundreds of AI models through the OpenRouter chat and completion APIs. - OpenRouterTeam/ai-sdk-provider</li><li><a href="https://ai.google.dev/gemini-api/docs/api-key">no title found</a>: no description found</li><li><a href="https://status.clerk.com/">
Clerk, Inc. status
</a>: no description found</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider">GitHub - OpenRouterTeam/ai-sdk-provider: The OpenRouter provider for the Vercel AI SDK contains support for hundreds of AI models through the OpenRouter chat and completion APIs.</a>: The OpenRouter provider for the Vercel AI SDK contains support for hundreds of AI models through the OpenRouter chat and completion APIs. - OpenRouterTeam/ai-sdk-provider
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1336790225208348673)** (215 messages🔥🔥): 

> `LM Studio API error handling, Model performance inquiries, Obsidian Smart Connections integration, Updating AI models and features, Safety of downloading AI models from TheBloke` 


- **Issues loading models in LM Studio**: Users reported various errors while loading models in LM Studio, including an 'unknown error' and 'exit code: 18446744072635812000'. Recommendations included providing system specs and checking the API for details on the error.
   - One user particularly struggled with state handling when connecting to local models, indicating a need for more guidance on API interactions.
- **Evaluating model performance on specific hardware**: Discussion took place regarding the suitability of the XTX 7900 24GB graphic card for running 30GB AI models, with insights shared about performance capabilities. Users highlighted settings and configurations needed for optimal results.
   - Another user seeking to run deep learning tasks locally expressed concerns about RAM and processing capabilities in relation to model demands.
- **Integrating Obsidian with LM Studio**: Users explored issues with connecting Obsidian's Smart Connections extension to LM Studio, reporting various errors and conflicts with other extensions. Troubleshooting steps included uninstalling conflicting plugins and rebuilding caches.
   - One user managed to set up a connection but still faced ongoing errors related to missing required fields in API responses, asking for further clarifications.
- **Updates on model availability and safety**: Users inquired about the safety and reliability of downloading AI models from TheBloke after noting some models were unavailable elsewhere. It was confirmed that TheBloke's models remain a standard in the industry despite his reduced presence in the community.
   - Users were encouraged to keep an eye on community channels for updates on model availability and potential releases of newer versions.
- **Frequency of model updates in LM Studio**: The frequency of new updates for LM Studio was questioned, with one user anticipating improvements for the Qwen2.5-VL model. Insights were shared that updates often coincide with the release of new models rather than regular software updates.
   - Users expressed excitement over potential enhancements and acknowledged the necessity of closely monitoring community announcements for the latest features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:1234"">no title found</a>: no description found</li><li><a href="https://model.lmstudio.ai/download/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF">Download and run lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF in LM Studio</a>: Use lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF locally in your LM Studio</li><li><a href="https://imgur.com/a/WnPhj6Y">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://huggingface.co/showlab/ShowUI-2B">showlab/ShowUI-2B · Hugging Face</a>: no description found</li><li><a href="https://www.apple.com/shop/buy-mac/macbook-pro/14-inch-m4-max">Buy 14-inch MacBook Pro with M4 Max</a>: Discover the MacBook Pro laptop with the M4 family of chips, built for Apple Intelligence. Get credit when you trade in an eligible Mac. Buy now.</li><li><a href="https://huggingface.co/lmstudio-community/MiniCPM-o-2_6-GGUF">lmstudio-community/MiniCPM-o-2_6-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/UI-TARS-2B-SFT-GGUF">lmstudio-community/UI-TARS-2B-SFT-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#vulkan">llama.cpp/docs/build.md at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/kth8/llama-server-vulkan">GitHub - kth8/llama-server-vulkan: Run llama.cpp server with Vulkan</a>: Run llama.cpp server with Vulkan. Contribute to kth8/llama-server-vulkan development by creating an account on GitHub.</li><li><a href="https://llm-stats.com/models/compare/o3-mini-vs-deepseek-r1">o3-mini vs DeepSeek-R1</a>: In-depth o3-mini vs DeepSeek-R1 comparison: Latest benchmarks, pricing, context window, performance metrics, and technical specifications in 2025.</li><li><a href="https://www.cloudflarestatus.com/">Cloudflare Status</a>: no description found</li><li><a href="https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>: Run Llama, Mistral, Phi-3 locally on your computer.</li><li><a href="https://youtu.be/7xTGNNLPyMI?t=11967">Deep Dive into LLMs like ChatGPT</a>: This is a general audience deep dive into the Large Language Model (LLM) AI technology that powers ChatGPT and related products. It is covers the full traini...</li><li><a href="https://github.com/stackblitz-labs/bolt.diy?tab=readme-ov-file#requested-additions">GitHub - stackblitz-labs/bolt.diy: Prompt, run, edit, and deploy full-stack web applications using any LLM you want!</a>: Prompt, run, edit, and deploy full-stack web applications using any LLM you want! - stackblitz-labs/bolt.diy</li><li><a href="https://github.com/stackblitz-labs/bolt.diy">GitHub - stackblitz-labs/bolt.diy: Prompt, run, edit, and deploy full-stack web applications using any LLM you want!</a>: Prompt, run, edit, and deploy full-stack web applications using any LLM you want! - stackblitz-labs/bolt.diy</li><li><a href="https://github.com/ggerganov/llama.cpp/">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1336812216501932095)** (23 messages🔥): 

> `DDR5 6000 EXPO Performance, Hardware Configuration for LMS, Memory Testing Tools, Multi-GPU Setup on PCIe 3.0` 


- **DDR5 6000 EXPO timings conservative**: A member pointed out that the **EXPO timings** for their DDR5 6000 were likely **super conservative**, noting max memory bandwidth during inference peaked at **72**.
   - They successfully completed **4 passes of memtest86** to ensure stability, although another member recommended trying TestMem5 for a more rigorous assessment.
- **LMS Hardware Configuration Troubles**: Another user raised a question about hardware setup for hosting LMS 0.3.9, mentioning their **32-core** CPU but no GPU, receiving advice on memory usage settings.
   - The recommendation included running in **Developer mode** with the option to keep the entire model in RAM, suggesting tweaks in thread usage for better speed.
- **Exploring Multi-GPU Capabilities**: A new member inquired about running multiple **3090s** on **PCIe 3.0 16x**, seeking experiences from others in the community.
   - Discussion revolved around whether such a setup remains viable, with another user asking for examples of setups that manage to run larger models effectively.
- **Inference Speed Considerations**: Concerns were raised over whether running RAM at **7600** would yield noticeable changes in inference speeds, but initial comparisons showed only a **15% improvement**.
   - Members noted that larger amounts of prompts could affect average speeds, particularly contrasting plain text responses with those generated from Python code.
- **Understanding GPU Acceleration**: Inquiries were made regarding GPU acceleration for the **DeepSeek R1 Distill Qwen 7B model**, with some confusion about which models support GPU use.
   - It was clarified that only specific models like **Llama** have known support for acceleration, leaving some ambiguity for the DeepSeek model.



**Link mentioned**: <a href="https://github.com/CoolCmd/TestMem5">GitHub - CoolCmd/TestMem5: TestMem5 - PC RAM stress test</a>: TestMem5 - PC RAM stress test. Contribute to CoolCmd/TestMem5 development by creating an account on GitHub.

  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1336865347130626182)** (97 messages🔥🔥): 

> `Home Assistant MCP Client/Server, MCP Server Usage, Goose MCP Client, Image Display in Claude, MCP Server Configurations` 


- **Home Assistant MCP Client with New Features**: A user announced the publication of their **Home Assistant** with **MCP client/server support**, describing it as functional but needing a 'wow factor'. They also shared plans to include an animated talking head avatar for better user interaction.
   - The project is still in progress as they balance paid work with development.
- **Curiosity about MCP Server Usage Statistics**: A user expressed curiosity about how many people are using the **Home Assistant** MCP, referring to efforts to bridge it with other tools like **Claude**.
   - This sparked discussions about the functionality of different MCP clients and their wider usage.
- **Discussion on Goose MCP Client**: Users shared their experiences using the **Goose MCP Client**, noting its effectiveness and current use cases in testing setups.
   - One user also mentioned a pending pull request that aims to improve its logging features, highlighting the close collaboration within the community.
- **Challenges with Image Display in Claude Desktop**: A user asked about displaying images as tool results on **Claude Desktop**, pointing out an input error encountered when trying to do so.
   - They speculated that converting image results to embedded resources might be a solution.
- **MCP Server Configuration Insights**: Discussion about designing better **MCP server configurations** ensued, with users sharing thoughts on how to effectively manage multiple servers.
   - One suggested using a multiplexer approach with **bridge** to streamline server management, and others shared their development plans for MCP clients.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://github.com/splendasucks/webperfect-mcp-server">GitHub - splendasucks/webperfect-mcp-server: webperfect-mcp-server</a>: webperfect-mcp-server. Contribute to splendasucks/webperfect-mcp-server development by creating an account on GitHub.</li><li><a href="https://github.com/block/goose/actions/runs/13058183345/job/36804119892?pr=947">fix(anthropic): include cached tokens in usage count logs · block/goose@162c4c5</a>: an open-source, extensible AI agent that goes beyond code suggestions - install, execute, edit, and test with any LLM - fix(anthropic): include cached tokens in usage count logs · block/goose@162c4c5</li><li><a href="https://github.com/block/goose/pull/947">fix(anthropic): include cached tokens in usage count logs by evalstate · Pull Request #947 · block/goose</a>: cache_creation_input_tokens and cache_read_input_tokens were not beingadded to the usage total recorded in goose.log. This fix includesthose categories in the calculation of &amp;quot;input tokens&amp...</li><li><a href="https://github.com/met4citizen/TalkingHead">GitHub - met4citizen/TalkingHead: Talking Head (3D): A JavaScript class for real-time lip-sync using Ready Player Me full-body 3D avatars.</a>: Talking Head (3D): A JavaScript class for real-time lip-sync using Ready Player Me full-body 3D avatars. - met4citizen/TalkingHead
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1336803791445426307)** (54 messages🔥): 

> `PulseMCP Use Cases, MCP Servers, Claude for Research, Web Research Tools, Markdown in Discord` 


- **PulseMCP Use Cases launched**: A new showcase of practical **PulseMCP Use Cases** was announced, featuring detailed instructions and videos for using various client apps and servers effectively.
   - Initial highlights include uses of **Gemini voice**, **Claude**, and **Cline** for managing Notion, converting Figma designs, and creating knowledge graphs, respectively.
- **Claude replicates ChatGPT DeepResearch**: **Claude** demonstrated the ability to replicate **ChatGPT DeepResearch** efficiently using specific MCP servers like *mzxrai's web research MCP* and *Brave web search MCP*.
   - A user noted that with sufficient time, Claude could process up to **100 articles**, highlighting the flexibility of the tool when given proper inputs.
- **Web search concerns and solutions**: Discussions revealed challenges with **Google searches** triggering bot detections and offered alternatives, like using **SearXNG** for searches without captchas.
   - Modifications to **chromedriver** and the suggestion to use **puppeteer** were among tools recommended to overcome these issues.
- **MCP capabilities on mobile devices**: Inquiries about mobile MCP clients suggested that **Sage** supports iPhones, while options for **Android** users may require using web clients like **LibreChat** or **MCP-Bridge**.
   - This reflects ongoing interest in accessing MCP functionality beyond desktop applications.
- **Markdown rendering in Discord**: A conversation emerged around Discord's **Markdown rendering** capabilities, noting its implementation since last year and user surprises at its features.
   - Members shared informal banter about using **Markdown** styles, reflecting a light-hearted community engagement.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.pulsemcp.com/use-cases">Community use-cases of MCP in-action | PulseMCP</a>: Explore all the ways in which the community is putting the Model Context Protocol (MCP) to use.</li><li><a href="https://x.com/tadasayy/status/1887253558749471034">Tweet from Tadas Antanavicius (@tadasayy)</a>: 🎉 Announcing the launch of Use Cases on PulseMCP (follow @pulsemcp to keep up)!There have been a ton of great MCP servers & clients built since its launch by @Anthropic and we built a resource to hig...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1336862981111152702)** (112 messages🔥🔥): 

> `Gemini 2.0 Performance, DeepSpeed with Hugging Face, AI Legislation Impact, Australia's Internet Infrastructure, Open Source AI Models` 


- **Gemini 2.0 outperforms competitors**: Blog discussions mentioned that **Gemini 2.0 Pro** shows impressive performance on tasks like creating SVGs, especially compared to **o3-mini** and **R1**.
   - Several members noted stronger performance in SQL queries, suggesting Google is making significant strides with **Gemini Flash 2.0**.
- **DeepSpeed and Dataloader Confusion**: One user expressed confusion over needing to manually specify **batch_size** in their Dataloader while using DeepSpeed's auto batch size configuration.
   - Another member suggested integrating DeepSpeed tags into the Dataloader for optimization and hinted at potential performance adjustments for various nodes.
- **Concerns About AI Legislation**: A user shared concerns about new legislation in Australia, suggesting it was aimed at restricting free speech and thought, with major implications for society.
   - This was framed in the context of a broader sentiment that such laws could render many discussions illegal, stifling open dialogue and inquiry.
- **Australia's Struggles with Internet Infrastructure**: A participant lamented that despite considerable financial investment, Australia still suffers from slow internet speeds, with some reporting home connectivity as low as **3 Mbps**.
   - Discussions highlighted failures in infrastructure decisions, referencing the poor choice of **copper over fiber optics** made decades ago.
- **The Future of Open Source AI Models**: The dialogue included concerns about the push against **open source AI models**, with discussions centering around limiting competitive and innovative models in favor of proprietary systems.
   - Members expressed frustration over perceived attempts to control discourse surrounding AI by outlawing certain discussions related to technology and free expression.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://simonwillison.net/2025/Feb/5/gemini-2/">Gemini 2.0 is now available to everyone</a>: Big new Gemini 2.0 releases today: - **Gemini 2.0 Pro (Experimental)** is Google&#x27;s &quot;best model yet for coding performance and complex prompts&quot; - currently available as a free preview. -...</li><li><a href="https://x.com/NationFirstAust/status/1887361530955755800">Tweet from George Christensen (@NationFirstAust)</a>: They&#39;re coming for your words. Your thoughts. Your beliefs. 1/24</li><li><a href="https://techcrunch.com/2025/02/05/researchers-created-an-open-rival-to-openais-o1-reasoning-model-for-under-50/">Researchers created an open rival to OpenAI&#039;s o1 &#039;reasoning&#039; model for under $50 | TechCrunch</a>: AI researchers at Stanford and the University of Washington were able to train an AI &quot;reasoning&quot; model for under $50 in cloud compute credits, according
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1336828307668471881)** (10 messages🔥): 

> `Harmonic Loss Paper, VideoJAM Discussion, EU Discussion Hours, DeepSeek Hosting` 


- **Skeptical Reviews on Harmonic Loss**: Some members expressed skepticism about the **harmonic loss paper**, noting it was 'kind of hastily done' and lacked performance increases despite theoretical advantages.
   - Another member mentioned that the GitHub repository for the paper is 'more informative' than the paper itself.
- **Excitement for VideoJAM Paper Review**: A member announced an upcoming review of the [VideoJAM paper](https://hila-chefer.github.io/videojam-paper.github.io/) scheduled for **6 PM EU time**.
   - This timing might be beneficial for US members, although it presents challenges for those in EU time zones.
- **EU Hour Challenges in Discussions**: Concerns were raised about the 'bad EU hours' for daily discussions, prompting a suggestion to move discussions to **6 PM EU time** next week.
   - Another member confirmed the time difference, noting it's **6-9 hours earlier** for US east/west coasts.



**Link mentioned**: <a href="https://hila-chefer.github.io/videojam-paper.github.io/">VideoJAM</a>: VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Model

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1336805898257043466)** (10 messages🔥): 

> `Gemini 2.0 Flash, Flash-lite issues, S1 reasoning model, Inference scaling insights, OpenAI scaling laws` 


- **Gemini 2.0 Flash impresses**: A user reported trying out the new **Gemini 2.0 Flash** model via [LlamaIndex](https://openrouter.ai/google/gemini-2.0-flash-001), noting its incredible speed, although not as fast as **Groq**.
   - This latest addition to the OpenRouter class is generating buzz among users eager to test its capabilities.
- **Flash-lite struggles with structured output**: Another user reported that the **Flash-lite** model struggled with returning valid structured outputs, often generating **invalid JSON** formats.
   - Finding this underwhelming, they suggested that it might not be suitable for tasks needing reliability in output.
- **S1 emerges as a low-cost reasoning alternative**: A recent blog post discussed the **S1 reasoning model**, which demonstrates competent performance similar to models like **OpenAI's o1** but can run on basic machines, highlighting its low cost of under **$50** for training.
   - Developed through distillation from **Gemini 2.0**, the S1 model and its tools are available on [GitHub](https://github.com/simplescaling/s1).
- **Insights on Inference Scaling**: The conversation revealed insights into **inference scaling**, claiming that longer thinking times can enhance LLM performance; however, methods to achieve longer thinking processes were questioned.
   - The **s1 paper** illustrated this with graphs, sparking discussions about how to implement such strategies effectively.
- **Questions on flash capabilities**: Community members posed questions regarding the capabilities of recently released models, including the **2.0 pro experimental version**, and whether there were prompts for testing.
   - The value of newly launched models was debated, referencing potential and past experiences with **distilled** versions of existing models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/google/gemini-2.0-flash-001">Gemini Flash 2.0 - API, Providers, Stats</a>: Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1. Run Gemini Flash 2.0 with API</li><li><a href="https://openrouter.ai/google/gemini-2.0-flash-lite-preview-02-05:free/api">Google: Gemini Flash Lite 2.0 Preview (free) – Run with an API</a>: Sample code and API for Google: Gemini Flash Lite 2.0 Preview (free) - Gemini Flash Lite 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1.5](google/gemini-flash...</li><li><a href="https://x.com/osanseviero/status/1887247587776069957">Tweet from Omar Sanseviero (@osanseviero)</a>: Hey r/LocalLLaMA 👋We&#39;re cooking 🫡 Gemma going brrr</li><li><a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: Language models ranked and analyzed by usage across apps</li><li><a href="https://timkellogg.me/blog/2025/02/03/s1">S1: The $6 R1 Competitor?</a>: no description found</li><li><a href="https://techcrunch.com/2025/02/05/researchers-created-an-open-rival-to-openais-o1-reasoning-model-for-under-50/">Researchers created an open rival to OpenAI&#039;s o1 &#039;reasoning&#039; model for under $50 | TechCrunch</a>: AI researchers at Stanford and the University of Washington were able to train an AI &quot;reasoning&quot; model for under $50 in cloud compute credits, according
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1336866947928821780)** (22 messages🔥): 

> `Collaboration on LLM Research, Deepspeed and Hugging Face, Benchmarking LLMs, Weight Decay in Fine Tuning, RWKV Architecture Development` 


- **Seeking Collaboration on LLM Projects**: A senior ML Engineer at Adobe expressed interest in exploring **research projects related to LLM agents** and invited collaboration with like-minded individuals.
   - *Looking forward to some exciting discussions!*
- **Clarifications on Deepspeed Usage**: A user inquired about the need to specify **batch_size** for the data loader when using **Deepspeed** with auto batch sizing in the config.
   - Another member pointed out that the **batch_size** specified would still be required for the data loader.
- **New Thematic Generalization Benchmark Raised**: A member shared a link to a GitHub repository discussing a **thematic generalization benchmark** designed to evaluate LLMs in inferring categories from examples and anti-examples.
   - They questioned if this benchmark might correlate with the performance of **SAE autointerp**.
- **Weight Decay's Role in Fine Tuning**: A discussion arose regarding the appropriateness of using **weight decay** during fine-tuning, with one member affirming its common use.
   - Another user noted an interesting point from the **OLMo2 paper**, where weight decay wasn't applied to embedding parameters during pretraining.
- **RWKV Team Develops New Architectures**: The RWKV team was mentioned as working on some exciting new architectures, indicating a proactive approach to model development.
   - A user shared their struggles with scaling and resource-intensive designs, inviting further discussion about potential collaboration.



**Link mentioned**: <a href="https://github.com/lechmazur/generalization">GitHub - lechmazur/generalization: Thematic Generalization Benchmark: measures how effectively various LLMs can infer a narrow or specific &quot;theme&quot; (category/rule) from a small set of examples and anti-examples, then detect which item truly fits that theme among a collection of misleading candidates.</a>: Thematic Generalization Benchmark: measures how effectively various LLMs can infer a narrow or specific &amp;quot;theme&amp;quot; (category/rule) from a small set of examples and anti-examples, then d...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1336791726290894992)** (92 messages🔥🔥): 

> `Multi Token Prediction Inference, Independent Research in AI/ML, A/B Testing and Reward Modeling, Quadratic Fitting for Parameter Estimation, DeepSeek MTP Implementation` 


- **Understanding Multi Token Prediction for Inference**: A question arose on how Multi Token Prediction (MTP) works during inference, particularly in relation to generating the initial token and using embeddings to improve speed.
   - Discussion highlighted that MTP serves as an efficient way to generate tokens, with resources shared about its implementation, including a [Github pull request](https://github.com/vllm-project/vllm/pull/12755).
- **Realistic Domains for Independent AI Research**: An independent researcher inquired about feasible areas to explore in AI/ML without the need for extensive funding, given various constraints on computing resources.
   - Responses suggested looking for grant opportunities and joining collaborative research groups, with mention of programs like [Google's TRC](https://sites.research.google/trc/about/).
- **A/B Testing and Parameter Estimation Challenges**: The conversation shifted to the viability of using A/B testing to determine optimal sampler parameters, raising concerns over relying on traditional quadratic fittings.
   - It was suggested that employing a Reward Model could better capture user preferences, while acknowledging the complexity that bandit algorithms introduce into the approach.
- **Quadratic Fitting vs. Arbitrary Function Learning**: A discussion on fitting a quadratic function to A/B data led to exploring concepts of reward modeling, symbolizing a possible avenue to refine the estimating process.
   - Participants pointed out the limitations of only fitting quadratics and discussing alternative methods like using pairwise preference models to optimize sampler parameters.
- **DeepSeek MTP's Surprising Outcomes**: Insights were shared on the performance of the DeepSeek model, highlighting its implementation of MTP and its relative success in user-rated token generation.
   - Participants expressed curiosity about the effectiveness of the model while sharing resources and practical outcomes from their experiences with the underlying methodology.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.03387">LIMO: Less is More for Reasoning</a>: We present a fundamental discovery that challenges our understanding of how complex reasoning emerges in large language models. While conventional wisdom suggests that sophisticated reasoning tasks de...</li><li><a href="https://sites.research.google/trc/about/">TPU Research Cloud - About</a>: no description found</li><li><a href="https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k">bespokelabs/Bespoke-Stratos-17k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B#distilled-model-evaluation">deepseek-ai/DeepSeek-R1-Distill-Qwen-32B · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/auth/endorse?x=XGKX4E">arXiv user login</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/pull/12755">[Model][Speculative Decoding] DeepSeek MTP spec decode by luccafong · Pull Request #12755 · vllm-project/vllm</a>: Implement DeepSeek MTP: #12181 to support DeepSeek MTP layers for next n prediction.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1336893345737478154)** (1 messages): 

> `MATS cohort applications, Mechanistic Interpretability Research, Mentoring in AI research` 


- **Summer MATS Cohort Applications Open!**: Applications for the **MATS 8.0** cohort are now open, with submissions due by **Feb 28**. Interested candidates can apply [here](https://tinyurl.com/neel-mats-app) to participate in paid full-time mechanistic interpretability research.
   - The program welcomes applicants of all experience levels, and previous mentees have contributed to **10 top conference papers** in the field.
- **Access MATS FAQ and Admissions Procedure**: Details about the MATS admissions procedure and FAQ can be found in the linked [document](https://docs.google.com/document/?usp=docs_web). Potential applicants are encouraged to sign in to review the comprehensive guidelines.
   - This resource aims to clarify the process for those interested in applying to the mentorship program.
- **Mentoring Success Stories**: Over the years, Neel has mentored more than **40 mentees**, contributing significantly to mechanistic interpretability research. This experience showcases the program's effectiveness in cultivating talent within the field.
   - Neel expressed pride in his mentees' success, specifically noting their contributions to major conferences, emphasizing that you don't need to be at a big lab to excel in this research area.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tinyurl.com/neel-mats-app">Neel Nanda MATS 8.0 Stream -  Admissions Procedure + FAQ</a>: Neel Nanda MATS 8.0 - Admission Procedure + FAQ Apply here Why Might I Want to Apply? Selection Question What should my application look like? Executive Summary Format Useful Resources What research p...</li><li><a href="https://x.com/NeelNanda5/status/1887274059408548208">Tweet from Neel Nanda (@NeelNanda5)</a>: Apps are open for my MATS stream, where I try to teach how to do great mech interp research. Due Feb 28!I love mentoring and have had 40+ mentees, who’ve made valuable contributions to the field, incl...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1337056905717092394)** (7 messages): 

> `cons@64, majority voting, eval configuration in YAML` 


- **Clarification on cons@64 Terminology**: Members discussed the meaning of **cons@64**, speculating if it refers to majority voting over 64 outputs or an LLM generating answers using those outputs.
   - *Consensus* and *majority voting* were identified as interchangeable terms in this context, as one member shared a link from OpenAI discussing the topic.
- **Expert Inquiry on Eval YAML Configuration**: A member inquired about the possibility of automating the specification of *apply chat template* or *fewshot-as-multiturn* from within an eval .yaml file.
   - They wondered if this should be coded in **utils.py** to incorporate the **mgsm_chat** functionality into various evaluations.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1337066904510271539)** (1 messages): 

> `Sequence parallelism implementation, Model parallelism size issues, AttributeError in Megatron library, Training crash log` 


- **Struggles with Sequence Parallelism and MP size 2**: A user encountered a **training crash** when trying to enable sequence parallelism with a model parallelism size (MP) of **2**, citing a specific error in the Megatron library.
   - The error traceback points to *AttributeError: module 'megatron.mpu' has no attribute 'get_fp32_allreduce'* indicating a potential issue in the implemented function.
- **Confusion over Documentation and Flag Use**: The user expressed confusion over the documentation that suggests sequence parallelism should work with MP greater than **1** by merely turning a flag on.
   - This discrepancy raises questions about whether the documentation is inaccurate or if there is an existing issue in the implementation.



**Link mentioned**: <a href="https://wandb.ai/aflah/hubble-speed-testing/runs/oawmmmpd/overview">aflah</a>: Weights & Biases, developer tools for machine learning

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1336795387784335361)** (100 messages🔥🔥): 

> `Deep Research Feedback, AI Backlash and Crypto, Purpose AI Agent in Trusts, New AI Models and Training Techniques, Fine-tuning Approaches` 


- **Deep Research receives positive reviews**: Members shared their enthusiasm for OpenAI's Deep Research, highlighting its ability to efficiently pull relevant connections and sources, enhancing their cognitive bandwidth.
   - One user pointed out the model's capability to explore obscure online communities and gather unexpected data.
- **AI backlash tied to past tech controversies**: Discussions surfaced regarding the public's distrust towards AI stemming from past negative experiences with cryptocurrency and NFTs, as some members believe this sentiment is impacting perceptions of AI technology.
   - Critics emphasized concerns around AI training data being unlicensed and its disruptive effects on labor markets.
- **Exploration of Purpose AI Agents**: A user outlined their ambition to develop a purpose-driven AI agent within a legal trust framework, aiming to pioneer legal discourse around AI personhood.
   - Feedback focused on the engineering complexity involved, including integrating fiscal management functions while emphasizing the potential for custom software solutions.
- **Advancements in AI model merging and fine-tuning**: Conversations included strategies for merging different AI models, with members sharing insights on improving model instruction tuning and reasoning performance.
   - Various fine-tuning methods were discussed, exploring the benefits of incorporating innovative techniques in AI training to enhance model performance.
- **Concerns over reasoning trace accessibility**: Members expressed skepticism regarding the availability of reasoning traces from models like DeepSeek and feared that OpenAI may leverage them without providing API access.
   - The conversation highlighted the trend of major AI companies limiting access to advanced features and information, potentially to protect proprietary technology.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/teknium/Llama-3.1-AlternateTokenizer">teknium/Llama-3.1-AlternateTokenizer · Hugging Face</a>: no description found</li><li><a href="https://rentry.org/vwa65v85">Why Everyone Is Suddenly Mad at AI</a>: AI Backlash: Another Tech Hype Hangover (and Is Crypto to Blame?)(OpenAI Deep research demo prompt: Write an essay on why there might be backlash to AI, and if it&#39;s related to NFT/crypto visibilit...</li><li><a href="https://huggingface.co/minpeter/Llama-3.2-1B-AlternateTokenizer-chatml/commit/f2528b7382f529d36a224ff04c5a73af3acd4e9c">Upload folder using huggingface_hub · minpeter/Llama-3.2-1B-AlternateTokenizer-chatml at f2528b7</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=9FuNtfsnRNo">I Built the Ultimate Team of AI Agents in n8n With No Code (Free Template)</a>: 📌 Join my free Skool community for the workflows shown in my videos! 👇https://www.skool.com/ai-automation-society/about🌟 Join my paid Skool community if y...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-GRPO.ipynb)">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device)">Unsloth Documentation</a>: no description found</li><li><a href="https://huggingface.co/unsloth?sort_models=created&search_models=1m#models).">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1337134893699694775)** (1 messages): 

> `DeepSeek-R1 training loop, Reward loss vs KL loss sensitivity, Pitfalls of small instruct models, Model size considerations, Hyperparameter importance` 


- **DeepSeek-R1 Training Loop Insights**: A user inquired about the **sensitivity** of the model to the weighting between **reward loss** and **KL loss**, questioning its significance as a hyperparameter.
   - They sought insights into which hyperparameters hold the most importance in optimizing the model's performance.
- **Concerns About Small Instruct Models**: The user expressed interest in potential **pitfalls** when starting with a smaller instruct model like **Qwen2.5 3B**, compared to larger base models.
   - They emphasized a desire to find the smallest model that can still provide reliable testing and development for resource management.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1337077333068222617)** (1 messages): 

> `Synthetic data generation, Seed-based approaches, Magpie output issues, Self-instruct alternatives, Awesome-LLM-Synthetic-Data resource` 


- **Exploring Synthetic Data Generation Techniques**: A member is seeking papers on **synthetic data generation**, particularly focusing on newer **seed-based approaches** similar to Self-Instruct.
   - They mentioned challenges with **Magpie** outputs due to experimenting in a non-English language.
- **Issues with Magpie Outputs**: The member expressed frustration with the quality of outputs from **Magpie** when using seed system prompts, finding them unsatisfactory.
   - *WizardLM has not been helpful* as they require effective seed instructions to proceed.
- **Found Resource on LLM Synthetic Data**: The member discovered a GitHub repository titled [Awesome-LLM-Synthetic-Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data) which offers a list of resources on **LLM-based synthetic data generation**.
   - This resource aims to assist in understanding various techniques and methodologies in the domain, particularly for newer models.



**Link mentioned**: <a href="https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data?tab=readme-ov-file,">GitHub - wasiahmad/Awesome-LLM-Synthetic-Data: A reading list on LLM based Synthetic Data Generation 🔥</a>: A reading list on LLM based Synthetic Data Generation 🔥 - wasiahmad/Awesome-LLM-Synthetic-Data

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1336792312008671306)** (3 messages): 

> `Deep Dive into LLMs, Mina's zkML Library` 


- **Explore AI with Deep Dive into LLMs**: A [YouTube video titled 
- **Mina's zkML Library Developer Guide**: An article on [Mina's Blog](https://minaprotocol.com/blog/minas-zkml-library-developer-guide) discusses the launch of the zkML library, which enables AI models to operate on-chain while maintaining **full privacy** and **verification**. This guide serves as a walkthrough for developers looking to leverage Mina's zkML capabilities for decentralized applications.



**Link mentioned**: <a href="https://m.youtube.com/watch?v=7xTGNNLPyMI&pp=ygUgRGVlcCBkaXZlIGludG8gbGxtcyBsaWtlIGNoYXRncHQ%3D">Deep Dive into LLMs like ChatGPT</a>: This is a general audience deep dive into the Large Language Model (LLM) AI technology that powers ChatGPT and related products. It is covers the full traini...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1337077333068222617)** (1 messages): 

> `Synthetic Data Generation, Self-instruct, Magpie, WizardLM, Awesome LLM Synthetic Data` 


- **Seeking Seed-based Synthetic Data Solutions**: A user is looking for papers or directions on **synthetic data generation**, specifically seed-based methods like **Self-instruct**, to improve their results with **Magpie**.
   - They noted that outputs using a non-English language are not satisfactory and are seeking better seed instructions beyond what **WizardLM** offers.
- **Found Resource on GitHub for Synthetic Data**: The user discovered a [GitHub repository](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data) titled **Awesome-LLM-Synthetic-Data**, which provides a reading list on **LLM-based synthetic data generation**.
   - The repository emphasizes various resources and is now a potential avenue for the user to explore alternatives for their data generation needs.



**Link mentioned**: <a href="https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data?tab=readme-ov-file,">GitHub - wasiahmad/Awesome-LLM-Synthetic-Data: A reading list on LLM based Synthetic Data Generation 🔥</a>: A reading list on LLM based Synthetic Data Generation 🔥 - wasiahmad/Awesome-LLM-Synthetic-Data

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1336902458001199146)** (39 messages🔥): 

> `John Schulman leaves Anthropic, Hibiki speech-to-speech translation model, Le Chat AI sidekick, GitHub Copilot agent mode, OpenAI updated chain of thought` 


- **John Schulman departs Anthropic**: Leading AI researcher and OpenAI co-founder **John Schulman** has left **Anthropic** after around five months, raising questions about his next move [link](https://www.bloomberg.com/news/articles/2025-02-06/openai-co-founder-john-schulman-leaves-rival-firm-anthropic?srnd=undefined).
   - Speculations about his potential next steps included thoughts about positions at organizations like **Deepseek** and **AI2**.
- **Hibiki revolutionizes translation**: The new **Hibiki** model from **Kyutai Labs** supports **simultaneous speech-to-speech translation** and adapts its pace based on content [link](https://x.com/kyutai_labs/status/1887495488997404732).
   - It reportedly outperforms previous systems in **quality**, **naturalness**, and **speaker similarity**, approaching human interpreter capabilities.
- **Le Chat AI officially launched**: **MistralAI** has launched **Le Chat**, marketed as a comprehensive AI sidekick for work and life, now available on both web and mobile [link](https://x.com/MistralAI/status/1887517520040448510).
   - This new tool aims to enhance productivity and personal assistance through AI.
- **GitHub Copilot unleashes agent mode**: GitHub announced the introduction of **agent mode** for **Copilot**, designed to assist developers more effectively with coding [link](https://github.blog/news-insights/product-news/github-copilot-the-agent-awakens/).
   - This update also includes the general availability of **Copilot Edits**, enhancing the developer experience.
- **OpenAI enhances chain of thought**: OpenAI updated the **chain of thought** mechanism in **o3-mini** models, aiming to refine the user experience, though they emphasize these aren't the raw CoTs [link](https://x.com/OpenAI/status/1887616278661112259).
   - The conversation suggests this may lead to **better distillation outputs**, although the significance of certain tokens remains debated.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenAI/status/1887616278661112259">Tweet from OpenAI (@OpenAI)</a>: Updated chain of thought in OpenAI o3-mini for free and paid users, and in o3-mini-high for paid users.</li><li><a href="https://github.blog/news-insights/product-news/github-copilot-the-agent-awakens/">GitHub Copilot: The agent awakens</a>: Introducing agent mode for GitHub Copilot in VS Code, announcing the general availability of Copilot Edits, and providing a first look at our SWE agent.</li><li><a href="https://x.com/TheXeophon/status/1887343884662935894">Tweet from Xeophon (@TheXeophon)</a>: @apples_jimmy I would be so hard if John joins Ai2</li><li><a href="https://x.com/kyutai_labs/status/1887495488997404732">Tweet from kyutai (@kyutai_labs)</a>: Meet Hibiki, our simultaneous speech-to-speech translation model, currently supporting 🇫🇷➡️🇬🇧.Hibiki produces spoken and text translations of the input speech in real-time, while preserving the sp...</li><li><a href="https://fxtwitter.com/polynoamial/status/1887621287616651429">Tweet from Noam Brown (@polynoamial)</a>: When we briefed people on 🍓 before o1-preview&#39;s release,  seeing the CoT live was usually the &#34;aha&#34; moment for them that made it clear this was going to be a big deal. These aren&#39;t th...</li><li><a href="https://x.com/shiringhaffary/status/1887340283916140922?s=61">Tweet from Shirin Ghaffary (@shiringhaffary)</a>: Leading AI researcher and OpenAI co-founder John Schulman has left Anthropic after around five months working at the company https://www.bloomberg.com/news/articles/2025-02-06/openai-co-founder-john-s...</li><li><a href="https://x.com/MistralAI/status/1887517520040448510">Tweet from Mistral AI (@MistralAI)</a>: Introducing the all new Le Chat: your ultimate AI sidekick for life and work! Now live on web and mobile!
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1337167338444689502)** (3 messages): 

> `LRMs test-time scaling, Model decision-making, Training phase scaling` 


- **Confusion over LRMs Test-Time Scaling**: A member questioned the term **test-time scaling** for Long-Range Models (LRMs), noting that **models decide their own output** without external control.
   - They highlighted that the scaling occurs during the **training phase**, prompting a broader discussion about the terminology used.
- **Concerns about LRM Control**: Another member dismissed the entire concept, expressing that the discussions around **test-time computing** for LRMs are fundamentally flawed.
   - This sentiment emphasizes skepticism towards the efficacy and clarity of the term in the context of autonomous model behaviors.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1336791596548620370)** (9 messages🔥): 

> `Crowd-sourced prompts, Jailbreaking models, Open Source Community, Incentives in AI` 


- **Concerns Over Crowd-sourced Expertise**: A prominent member expressed disdain about providing expertise for crowd-sourced prompts that appear to serve investors by promoting safety falsely, stating, *'I’m allergic to money, so don’t bother.'*
   - This raises questions about whether genuine community benefit is prioritized over profit motives in AI development.
- **Skepticism on Achievement in AI Levels**: A member questioned why, if an individual could complete all 8 levels, they hadn't done so yet, suggesting there may be limitations beyond frontend bugs.
   - Another noted the individual's extensive work on jailbreaking existing models, implying they've likely encountered challenges before.
- **Work vs. Free Contribution Debate**: Discussion arose around the idea that contributions to AI should not be compelled as free labor, with a member saying, *'this is a job. you are trying to get me to do work. for free.'*
   - This highlights the tension between community contributions and the expectations placed on individuals within the open-source landscape.
- **Entertainment in the Open Source Community**: A member remarked on the humor found within the open-source community, indicating that open source tends to encourage amusing interactions.
   - They referred to humorous replies received on Bluesky, specifically citing a comment about Europe potentially taking initiatives instead of the US.



**Link mentioned**: <a href="https://fxtwitter.com/elder_plinius/status/1887225319582466125">Tweet from Pliny the Liberator 🐉 (@elder_plinius)</a>: I don’t want to provide my world-class expertise just for you to hoard crowd-sourced prompts and construct elaborate security theater performances to appease investors who are foolish enough to believ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1336798627384918171)** (23 messages🔥): 

> `ChatGPT Fishing Techniques, Long Chain of Thought in LLMs, Qwen Model Discoveries, Deep Research Applications` 


- **Fishing with ChatGPT for Halibut**: The [YouTube video](https://youtu.be/BR_HSUUQDjA?si=hpBcvK6eskCCOhfK) titled **'Fishing for first timers'** showcases using ChatGPT in the pursuit of catching halibut.
   - A humorous note was made about using o3 to catch crabs, referring to the inquiry's lighter side.
- **Understanding Long CoT Reasoning in LLMs**: A discussion on [demystifying Long CoT Reasoning](https://x.com/xiangyue96/status/1887332772198371514) highlighted the mystery behind models like R1, O1, and O3, pursuing insights into their training dynamics.
   - *11 major takeaways* from the thread were noted, suggesting a detailed exploration of the topic.
- **Qwen Models' Surprising Results**: Recent discussions pointed out that Qwen 2.5 models achieved impressive results with minimal training data, as noted by various members discussing their findings.
   - A quote by Aran Komatsuzaki emphasized that Qwen models seem to possess a *magical quality*, achieving notably good performance with limited data.
- **Gary's Praise for Deep Research**: Gary Marcus remarked on the utility of **Deep Research**, pointing out its practicality despite still facing challenges in facts and temporal reasoning.
   - A community consensus emerged recognizing Deep Research's strengths in specific applications while acknowledging its shortcomings in factual accuracy.
- **Accessing O3 Efficiently**: A member shared a technique for utilizing O3 effectively by bypassing the browsing function, calling it a useful coding trick.
   - This method reportedly aids in gathering and implementing solutions accurately in a single attempt, boosting coding efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/BigTechAlert/status/1887363101328117946">Tweet from Big Tech Alert (@BigTechAlert)</a>: 🚫 @allen_ai is no longer following @ehsanik(🤖💭: who has more details?)</li><li><a href="https://x.com/ZeyuanAllenZhu/status/1887419529359237393">Tweet from Zeyuan Allen-Zhu, Sc.D. (@ZeyuanAllenZhu)</a>: (2/8) Just like P vs NP, reviewing paper (L3) requires less intelligence than authoring one (L4). Auditing a review only needs L2; arbitrating author-reviewer dispute needs L1 --- only need to follow ...</li><li><a href="https://x.com/lateinteraction/status/1887356471555563839">Tweet from Omar Khattab (@lateinteraction)</a>: So many &#34;we did almost nothing and now Qwen 2.5 can do everything&#34; results 😆</li><li><a href="https://x.com/s_streichsbier/status/1887341868348023142">Tweet from Stefan Streichsbier (@s_streichsbier)</a>: Thanks, @iruletheworldmo.This is a very useful trick to access the full o3 for coding.It researches a bunch of sources for how to implement the solution and then puts it together correctly in 1 shot.W...</li><li><a href="https://x.com/GaryMarcus/status/1887505877437211134">Tweet from Gary Marcus (@GaryMarcus)</a>: Deep Research is genuinely useful - depending on your application - but crucially  (as anticipated by Rebooting AI in 2019, and by @yudapearl)  facts and temporal reasoning remain problematic for curr...</li><li><a href="https://fxtwitter.com/ZeyuanAllenZhu/status/1882283698239971499)">Tweet from Zeyuan Allen-Zhu, Sc.D. (@ZeyuanAllenZhu)</a>: Don’t let ridiculous ICLR reviewers get you down—this happens even on ACs. Our paper (8,8,6,3) was unilaterally rejected by a meta-reviewer. Thankfully, ICLR is open-review so this misbehavior will be...</li><li><a href="https://x.com/lateinteraction/status/1887355468965945795">Tweet from Omar Khattab (@lateinteraction)</a>: More Qwen. I&#39;m increasingly comfortable saying these papers seem to be a discovery of some sort about Qwen models, not necessarily about reasoning.Quoting Aran Komatsuzaki (@arankomatsuzaki) LIMO:...</li><li><a href="https://x.com/ZeyuanAllenZhu/status/1887419526738014693">Tweet from Zeyuan Allen-Zhu, Sc.D. (@ZeyuanAllenZhu)</a>: (1/8) We classify L1~L5 intelligence, and observe only Gemini-2-FT, DeepSeek-R1, OpenAI-o1 can reach L2; most are only L1 (o3-mini). Yet, one can still use L1-level AIs to arbitrate disputes and ensur...</li><li><a href="https://x.com/xiangyue96/status/1887332772198371514">Tweet from Xiang Yue (@xiangyue96)</a>: Demystifying Long CoT Reasoning in LLMshttps://arxiv.org/pdf/2502.03373Reasoning models like R1 / O1 / O3 have gained massive attention, but their training dynamics remain a mystery. We&#39;re taking ...</li><li><a href="https://x.com/WenhuChen/status/1887371348663579032">Tweet from Wenhu Chen (@WenhuChen)</a>: I agree. It&#39;s pretty much the same discovery as s1, which arrives at the similar results with around 1000 training examples. We have actually tried training the other models with the same data and...</li><li><a href="https://youtu.be/BR_HSUUQDjA?si=hpBcvK6eskCCOhfK">Fishing for first timers</a>: Using ChatGPT to catch halibut
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1336979732322975820)** (2 messages): 

> `Duality of Man, Discussion on X, Post by mcmillen.dev` 


- **Exploring the Duality of Man**: A post shared on [X](https://x.com/distributionat/status/1887410881392427183) discusses the concept of the **duality of man**, highlighting contrasting aspects of human nature.
   - This theme invites deeper reflection on how individuals balance conflicting traits such as light and darkness in their lives.
- **Engagement with mcmillen.dev's Post**: A link to a [post by mcmillen.dev](https://bsky.app/profile/mcmillen.dev/post/3lhjdatt5xk2f) was shared, though no further details were provided about its contents.
   - The lack of context leaves the discussion open to interpretation, prompting curiosity about the ideas presented.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/distributionat/status/1887410881392427183">Tweet from thomas (@distributionat)</a>: the duality of man</li><li><a href="https://bsky.app/profile/mcmillen.dev/post/3lhjdatt5xk2f">Colin McMillen (@mcmillen.dev)</a>: &quot;In the future, we will no longer have aspirational goals&quot; is a masterstroke of Google corporate communicationshttps://www.theverge.com/google/607012/google-dei-hiring-goals-internal-memo
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1336852994783973439)** (11 messages🔥): 

> `RL dataset skepticism, Unsloth GRPO support, Unified memory usage, Training on same GPUs, DM paper on rollouts` 


- **Model developers skeptical of RL datasets**: Members discussed skepticism among model developers regarding RL datasets published by non-model shops, suggesting that such datasets may be viewed as lacking credibility without validation from established organizations.
   - *One member noted*, 'my instinct is that the dataset wouldn't be worth the paper it's printed on' if it lacks the endorsement of a credible source.
- **Unsloth enhances GRPO process**: Unsloth announced support for Group Relative Policy Optimization (GRPO), claiming their enhancements allow users to reduce VRAM usage by **80%** compared to previous methods.
   - This feature lets users reproduce R1-Zero's findings with just **7GB** of VRAM using Qwen2.5 while streamlining dependencies.
- **Unified memory may make async RLHF obsolete**: Discussion surfaced around Unsloth's unified memory usage, potentially decreasing the need for separate GPUs for training and rollouts by allowing both processes to run concurrently.
   - Members speculated this advancement could reduce resource waste, as GPUs wouldn't sit idle during operational switching.
- **DM paper confirms cyclical generation during training**: A member recalled a paper that discussed generating and feeding back data during training, despite it being slightly off policy, which individuals found relevant to their topic on simultaneous processes.
   - Another member confirmed a related paper [here](https://arxiv.org/abs/2410.18252) that supports similar conclusions about training dynamics.
- **Switching costs when using same GPUs**: Participants agreed that using the same GPUs for both training and rollouts is preferable if switching costs are minimal, but uncertainty remains about the actual costs involved.
   - *One member expressed,* 'You have to every time transform the model into vLLM format and idk how long that takes,' indicating potential complexity in implementation.



**Link mentioned**: <a href="https://unsloth.ai/blog/r1-reasoning">Train your own R1 reasoning model locally</a>: You can now reproduce your own DeepSeek-R1 reasoning model with Unsloth 100% locally. Using GRPO.Open-source, free and beginner friendly.

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1336870055522340885)** (8 messages🔥): 

> `Open source AI, DeepSeek's impact on Scale AI, AI's evolving definitions, The importance of human oversight, Dario on Chinatalk` 


- **Open source AI and Freedoms Discussion**: A recent piece discussed the transition of **AI beyond traditional software** confines and elaborated on OpenAI's Four Freedoms, inspired by Sam Altman's thoughts.
   - The post emphasized **Aaron Swartz**'s influence on the open-source movement, linking back to the foundational ideas of genuine open access.
- **Debating DeepSeek's Role in Scale AI's Model**: In response to DeepSeek, **Scale CEO Alexandr Wang** highlighted the misperception of automation in data generation, calling it 'lazy' to assume a fully automated process.
   - Despite transparency issues from DeepSeek, the company reportedly pioneered new automation techniques in creating training data.
- **Adaptation Challenges for Scale AI**: There's recognition that while adaptation is possible for **Scale AI**, challenges remain due to current operational models and valuations.
   - The emphasis was on the **bleak outlook** without significant changes to their approach amid a shifting landscape.
- **Dario's Feature on Chinatalk**: A mention of **Dario's** appearance on Chinatalk triggered interest and discussion among members about his insights.
   - This sparked curiosity and potentially provided a platform for deeper exploration of the topic.



**Link mentioned**: <a href="https://www.turingpost.com/p/fod86">🌁#86: Four Freedoms of Open AI</a>: – what are they? Defining the future

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/)** (1 messages): 

xeophon.: https://x.com/AndrewCurran_/status/1887505463211925557
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1336856625755197515)** (17 messages🔥): 

> `Collaboration and Similarities, Corruption in Religious Leadership, AI Features for Creativity, Uses of NotebookLM in Law, Max Headroom's Comeback` 


- **Finding Similarities in Work**: A member expressed excitement about discovering someone with a similar approach to their work in narratives, noting it helps identify weaker points in their narratives.
   - They mentioned that the hosts’ feedback greatly influenced the development of their narratives.
- **Corruption in Religious Leadership Discourse**: A member noted a discussion where hosts pointed out how historically, **religious leaders** advising on crops and voting are often susceptible to corruption.
   - This led to a realization about the inherent issues within these roles, indicating an obvious yet overlooked truth.
- **Proposed Sliders for AI Creativity**: A member suggested integrating sliders for tuning AI's creativity, similar to features found in **Gemini API** and other services.
   - This idea was sparked after they discovered an exploit related to the AI's features just two days prior.
- **NotebookLM for Legal Summaries**: A user shared experiences using **NotebookLM** to capture testimony at the New York State Legislature’s Budget hearing on Environmental Conservation.
   - They highlighted the challenge of sharing this extensive document due to licensing limitations, proposing it as a compelling demonstration of NotebookLM's capabilities.
- **Max Headroom's Digital Comeback**: A user excitedly announced the return of **Max Headroom** with an edgy video and music, showcasing a unique approach to AI interaction.
   - They encouraged others to watch and share their content, referencing a new video that humorously critiques corporate AI practices.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1kcUJvQiAwzX1GU4b0HvOUhLV0UtecLvuQSaTmfRFPpg/">New York State Legislature Environmental Conservation Budget Hearing 2025 - Notes</a>: Notes by Jon Garfunkel   Joint Legislative Public Hearing on 2025 Executive Budget Proposal: Topic Environmental Conservation | NYSenate.gov   I uploaded the source documents - 45 written testimonies ...</li><li><a href="https://youtu.be/YXgav2-6DsI?feature=shared">Max Headroom 2025 featuring &quot;🎵 &quot;BOT-NOIA&quot;</a>: 🚨 GLITCH ALERT! 🚨Guess who just escaped the mainframe? That’s right, ME! MAX! HEADROOM! Back from the digital graveyard, angrier, glitchier, and more sarca...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1336805129893974106)** (78 messages🔥🔥): 

> `NotebookLM Model Limitations, Audio Overview Customization, Spreadsheet Data Analysis, Sharing Notebooks Issues, Interactive Mode Problems` 


- **NotebookLM lacks model change options on mobile**: A user expressed frustration regarding the inability to change the model within the mobile version of NotebookLM, which another member confirmed is not possible.
   - This limitation seems to hinder user experience, leading to confusion among those expecting more flexibility in model management.
- **Audio Overviews Generation and Customization Tips**: Members discussed the process of customizing audio overviews in NotebookLM, confirming that users must delete and regenerate the audio file to access the customization button.
   - One member suggested using a specific phrasing to differentiate between primary and complementary sources for better outputs.
- **Spreadsheet Compatibility Concerns**: Concerns were raised about using NotebookLM for analyzing spreadsheet data, with suggestions to utilize tools like Gemini within Google Sheets instead.
   - Users highlighted the importance of understanding the capabilities of NotebookLM as primarily a text analysis tool.
- **Sharing Notebooks Functionality**: There were discussions on sharing notebooks among different Google accounts, with confirmation that while sharing is available, some users experienced visibility issues with shared notebooks.
   - Links to shared notebooks were discussed, and it was noted that sharing functionalities are currently being improved by the development team.
- **Issues with Interactive Mode**: A user reported persistent issues with the interactive mode in NotebookLM, noting that it fails to work across both web and mobile platforms.
   - The issue was recognized as potentially affecting both free and plus versions, raising questions about overall accessibility and functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/14276471?hl=en">Notebooks - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15724458?hl=en#:~:text=NotebookLM%20Plus%20gives%20you%20everything,premium%20features%2C%20and%20additional%20sharing">Get started with NotebookLM and NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://www.engadget.com/ai/gemini-can-now-do-more-complex-data-analysis-in-google-sheets-191218214.html">Gemini can now do more complex data analysis in Google Sheets</a>: Gemini in Google Sheets is about to become more powerful. The AI agent can now use Python code to generate insights and charts about your data.
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1336969747555684372)** (1 messages): 

> `Fall 2024 MOOC Certificates, Coursework Submission Challenges, Future MOOC Opportunities` 


- **Fall 2024 MOOC Certificates Released Today!**: All **Fall 2024 MOOC certificates** will be released today at **8am PT**, addressing recent technical challenges that have been resolved.
   - Congratulations to all recipients for their patience and hard work!
- **Some Participants Downgraded**: A few participants were **downgraded to the Trailblazer tier** due to incomplete coursework submissions.
   - Sadly, a minority will not receive a certificate, and no makeups or regrades will be offered.
- **Encouragement for Future MOOCs**: Participants are encouraged to sign up for **Spring 2025 MOOC** even if they faced challenges this time.
   - The team hopes everyone enjoyed the course and is excited for future opportunities!


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1336814789346857111)** (57 messages🔥🔥): 

> `Certificate issuance timeline, Quiz availability and results, Certificate tier breakdown, Communication and support, Feedback on course experience` 


- **Certificate Issuance Timeline Uncertainty**: Some members inquired about the expected timeframe for receiving their certificates, with a member expressing hope for delivery within a week or two due to *unforeseen technical issues* being resolved.
   - Another member noted discrepancies in certificate receipt, indicating a potential *soft bounce* issue affecting communications.
- **Quiz Availability Confusion**: Concerns arose over the availability of answers for Quiz-1 as Quiz-2 was launched, prompting members to seek clarification on the new policy regarding answer releases.
   - Members reassured fellow participants that score visibility for Quiz-1 was possible through the original link used for submission.
- **Certificate Tier Breakdown Revealed**: In response to a query, it was disclosed that there are 301 Trailblazer, 138 Masters, 89 Ninjas, 11 Legends, and 7 Honorees amongst the participants.
   - This prompted interest in how many people received each tier certificate and clarified that only the honorary tier would be noted if both an honorary and a specific tier were achieved.
- **Effective Communication and Support Resolved Issues**: The community expressed gratitude for the support received during the course, especially acknowledging the team handling grading and certificate queries.
   - Members encouraged clearer communications in cases where certificate statuses were pending, with some emails initially caught in spam filters.
- **Positive Feedback on Course Experience**: Participants shared enthusiasm for the course, with one member reflecting on their learning journey and the significance of their certificate for future endeavors.
   - Expressions of appreciation were made for the course organization, highlighting the difficulty of grading numerous submissions while maintaining quality.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1336946208886755341)** (13 messages🔥): 

> `Output vs Input Token Pricing, Independent Research in AI/ML, Niche Fields for Research, Economizing AI Research` 


- **Output Token Pricing Causes Confusion**: Members discussed the disparity in output and input token pricing, noting that **GPT-4o** charges **$10** per million output tokens compared to **$2.5** for inputs, primarily due to LLMs being **autoregressive**.
   - It's suggested that organizations like TogetherAI adopt a more straightforward aggregated pricing model.
- **Niche Research Areas for Independent Investigators**: An independent researcher sought advice on feasible domains in AI/ML, emphasizing the impracticality of pretraining large models without funding and expressing interest in NLP, Audio, and Vision.
   - Members advised focusing on **niche or untapped domains**, with one sharing their success in computational metabolomics, emphasizing the limited competition in that area.
- **Minimalist AI Research Possibilities**: Though niche research is advised, it was shared that independent researchers can also conduct efficient work on **LLMs and vision** tasks while fine-tuning models on a limited budget.
   - A member pointed out that significant progress can be made in **economizing AI research**, citing examples of reduced training times and innovative methodologies.
- **Value in Economizing AI Research**: Discussions pointed out the importance of economizing aspects in AI research, like achieving stability with **low-bit training weights** and reducing environmental impact through efficient training methods.
   - The success of **GPT-2 speedruns** with Muon was highlighted as a prime example of impactful research using limited resources.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1336954761408413696)** (4 messages): 

> `Triton warp specialization, Triton compiler on NVIDIA Blackwell, Installing Triton on RTX 5080, Deepseek fused MLA implementation in Triton` 


- **Triton introduces warp specialization on NVIDIA Hopper**: The recent rollout of **fully automated Triton warp specialization** for **NVIDIA Hopper GPUs** is now available in Triton [3.2](https://github.com/triton-lang/triton/tree/release/3.2.x) and will ship with PyTorch 2.6.
   - Users can take advantage of this feature by [implementing user-defined Triton kernels](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) as part of enhancing GPU capabilities.
- **Triton compiler supports NVIDIA Blackwell architecture**: NVIDIA’s ongoing collaboration with OpenAI has led to the Triton compiler now being compatible with the **NVIDIA Blackwell architecture**, enhancing performance and programmability.
   - This compatibility allows developers to utilize [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) and [CUTLASS](https://github.com/NVIDIA/cutlass) effectively for modern AI workloads.
- **Getting Triton operational on the new RTX 5080**: A user documented challenges encountered while installing Triton on the new **RTX 5080**, including reinstalling drivers and rebuilding machine learning libraries from source.
   - They provided a guide for installing compatible drivers, highlighting the requirement for **NVIDIA open kernel modules** over proprietary ones to resolve device detection issues.
- **Inquiry on Deepseek fused MLA in Triton**: A user raised a question regarding the availability of a **Deepseek fused MLA implementation** in Triton, indicating interest in this specific functionality.
   - Details regarding its support or development were not provided, leaving the inquiry open for further exploration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://webstorms.github.io/2025/02/06/5080-install.html">Running PyTorch and Triton on the RTX 5080</a>: I was beyond stoked at the opportunity of getting my hands on a new RTX 5080 to speed up my machine learning developments! Unfortunately, as soon as I connected the new GPU to my workstation I quickly...</li><li><a href="https://pytorch.org/blog/warp-specialization/?utm_campaign=4079123-PyTorch%20Blog%20Post%20Promotion&utm_content=324019352&utm_medium=social&utm_source=linkedin&hss_channel=lcp-78618366">Enabling advanced GPU features in PyTorch - Warp Specialization</a>: Meta: Hongtao Yu, Manman Ren, Bert Maher, Shane Nay  NVIDIA: Gustav Zhu, Shuhao Jiang</li><li><a href="https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/">OpenAI Triton on NVIDIA Blackwell Boosts AI Performance and Programmability | NVIDIA Technical Blog</a>: Matrix multiplication and attention mechanisms are the computational backbone of modern AI workloads. While libraries like NVIDIA cuDNN provide highly optimized implementations, and frameworks such as...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1336934948057120798)** (3 messages): 

> `CUDA GEMM Implementation, Double Buffering Performance Issues, Register Usage Optimization, Memory Sector Utilization` 


- **Double Buffering Drops Performance in CUDA GEMM**: A user reported that implementing **double buffering** in their single-precision GEMM kernel led to a dramatic drop in performance metrics.
   - They noted that, according to the **NCU profiler**, their register usage per thread decreased significantly, indicating potential inefficiencies.
- **Register Usage and Compiler Challenges**: A user suggested that dropping register usage might indicate the **compiler's struggle** with unrolling the new, more complex code, recommending the use of `#pragma unroll` for the loop.
   - They emphasized that simplifying the kernel could potentially lead to better register allocation.
- **Understanding Memory Sector Usage**: Another member explained that each GPU cache line is divided into **sectors**, and the reported inefficiency means only 1 byte out of 32 is utilized in memory requests.
   - This suggests that the kernel is not efficiently accessing memory, which might be caused by **stride accesses** between threads.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1337116755494965460)** (8 messages🔥): 

> `FP8 Attention, Hadamard Transform, CUDA Elementwise Kernel for Mixed Integer Linear Programming, Grouped GEMM Implementation, Torch Nested Tensor` 


- **FP8 Attention relies on Hadamard Transform**: A member observed that **FP8 Attention** for video models performed significantly better when utilizing the **Hadamard Transform**, drastically reducing error rates.
   - Referencing the [Flash Attention 3 paper](https://arxiv.org/pdf/2407.08608), they suggested that this approach is crucial not just for the attention mechanism, but for all operations in FP8.
- **CUDA for Mixed Integer Linear Programs**: A member is exploring the feasibility of using a **CUDA elementwise kernel** for pairwise kernel operations that involve solving mixed integer linear programs, traditionally handled by CPUs using scipy.optimize.
   - They questioned if offloading the computation to CUDA would yield a significant speedup when handling many diverse computations simultaneously.
- **Grouped GEMM on GPUs and its implementation**: One member inquired about the typical implementation of **grouped GEMM** on GPUs, asking if it simply loops over different group sizes as seen in some examples like Triton.
   - They raised a query regarding whether **torch.nestedtensor** utilizes a grouped GEMM approach in its operations.
- **Hadamard Transform Implementation Repository**: A member recommended using the [fast-hadamard-transform repository](https://github.com/Dao-AILab/fast-hadamard-transform/tree/master/csrc) to implement Hadamard before the attention mechanism.
   - This library offers CUDA implementation with a PyTorch interface that can enhance performance for operations needing the Hadamard Transform.
- **Mixed Integer Programming optimization conversation**: A member expressed skepticism about using **CUDA** for solving mixed integer programming due to its challenges, while exploring if a single thread could allow for a more competitive speedup.
   - Another user chimed in to suggest that the merit of a CUDA approach would depend heavily on the specific workload and kernel design.



**Link mentioned**: <a href="https://github.com/Dao-AILab/fast-hadamard-transform/tree/master/csrc">fast-hadamard-transform/csrc at master · Dao-AILab/fast-hadamard-transform</a>: Fast Hadamard transform in CUDA, with a PyTorch interface - Dao-AILab/fast-hadamard-transform

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: https://www.youtube.com/watch?v=7xTGNNLPyMI
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1336817106162811010)** (2 messages): 

> `PyTorch Team Visibility, User Concerns` 


- **User Shares Frustration**: A user expressed their frustration with the comment 'mega oof'.
   - This sentiment highlights an ongoing concern among members regarding issues that need attention.
- **Raising Issues for Visibility**: Another member suggested that the frustrated user comment on the issue to increase visibility to the **PyTorch team** 😄.
   - This approach aims to ensure that important concerns are addressed by those able to resolve them.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1336899517471133747)** (2 messages): 

> `Japanese government discussions, Text-generation-inference n-gram decoding` 


- **Japanese Government's Involvement Discussed**: The conversation briefly touched on the role of the **Japanese government** in related discussions.
   - Specific details about their actions or position were not provided in the messages.
- **Inquiring about n-gram Speculative Decoding**: A member asked for experiences using [`text-generation-inference`](https://github.com/user/text-generation-inference)'s **n-gram speculative decoding** implementation.
   - No responses with firsthand experiences were reported in this message history.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1337017268286128270)** (1 messages): 

> `Linear Attention Model, Distillation Process, Training Challenges` 


- **Struggles with Linear Attention Model Distillation**: A member attempted to distill a small LLM to a **linear attention model** following a [recipe from Lolcats](https://cdn.discordapp.com/attachments/1300872762163728550/1337017267925291068/distill_linear.ipynb?ex=67a5e9dd&is=67a4985d&hm=1a5dc02fb98a1f89ed72f7481e30459202f9d1de210fa3729663825137211832&) but faced issues.
   - The model only produced repeating characters, prompting a request for assistance from the **Lolcats team**.
- **Request for Help from Lolcats Team**: In response to the training challenges, the member reached out for help specifically from the **Lolcats team**.
   - This plea highlights the community support aspect often relied upon in AI model development.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1336797540343087134)** (18 messages🔥): 

> `Sokoban Puzzles, Rush Hour Puzzle, Reasoning-Gym Integration` 


- **Sokoban Puzzles added to Reasoning Gym**: A pull request was submitted to add **Sokoban puzzles** to **reasoning-gym**, demonstrating a new puzzle format for users to solve.
   - The pull request includes a graphic explanation of the puzzle setup along with example moves as a string of characters like **LDURRUDL**.
- **Rush Hour puzzle scripting ideas**: Members discussed creating a **text-interface** for representing moves in the **Rush Hour** game and shared useful resources for understanding the puzzle's mechanics.
   - A shared link led to a blog detailing how to programmatically solve the **Rush Hour** puzzle, which featured a grid format outline.
- **Local S1 Running for Reasoning Gym Gauntlet**: One member inquired about anyone having **S1 running locally** to test its capabilities on the **reasoning-gym gauntlet**.
   - They expressed eagerness to observe how it performs against known challenges.
- **Rush Hour GitHub Repository Shared**: A member shared a GitHub repository containing a project for **Rush Hour**, indicating the ease of accessing it for practical use.
   - The repository is focused on heuristic strategies and invites contributions to the development.
- **Collaborative Efforts on Rush Hour Game**: Members expressed enthusiasm in collaboratively building a basic gym for the **Rush Hour** game integration into the **reasoning-gym**.
   - This project would encourage collaborative coding efforts to implement the classic puzzle as a feature.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Rush_Hour_(puzzle)">Rush Hour (puzzle) - Wikipedia</a>: no description found</li><li><a href="https://www.michaelfogleman.com/rush/">Michael Fogleman</a>: no description found</li><li><a href="https://github.com/KaKariki02/rushHour">GitHub - KaKariki02/rushHour: heuristieken project</a>: heuristieken project. Contribute to KaKariki02/rushHour development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/66">Add Sokoban Puzzles by Miserlou · Pull Request #66 · open-thought/reasoning-gym</a>: Ex:This is a sokoban puzzle. Please solve it. Your solution must be a string of characters, ex: LDURRUDL+ + + + + + ++ * - @ - X ++ + - @ - + ++ X - - - $ ++ + + + + + +* - The player% - ...
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1336788959652741283)** (48 messages🔥): 

> `Model Performance Comparison, Language Model Constraints, DeepSeek Model Insights` 


- **Discussions on Model Comparisons**: Users discussed the performance of various models, noting that **O3** is still ahead despite concerns about its pricing.
   - *Llama 4* is anticipated as the next potential successor to challenge existing models.
- **Limitations in Political Discussions**: There was a consensus that various languages models have constraints, with **DeepSeek** demonstrating greater limitations compared to *ChatGPT* and *O3-mini*.
   - Members noted that prompts regarding sensitive political topics often lead to unexpected deletions or evasions in responses.
- **DeepSeek's Cutoff Date and Capabilities**: It was noted that **DeepSeek's** knowledge cutoff date is reportedly July **2024**, raising questions about its current relevance.
   - An interesting method called **Time Bandit** has been discussed for extracting information by leveraging temporal context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: no description found</li><li><a href="https://www.knostic.ai/blog/exposing-deepseek-system-prompts#:~:text=DeepSeek's%20knowledge%20cutoff%20is%20July,want%20to%20know%20its%20limitations.">DeepSeek’s cutoff date is July 2024: We extracted DeepSeek’s system prompt</a>: Discover how the &quot;Time Bandit&quot; method reveals DeepSeek's system prompt, exploring its ethical guidelines and neutrality in global contexts. Learn the implications for AI interactions.</li><li><a href="https://llm-stats.com/models/compare/deepseek-r1-vs-o3">DeepSeek-R1 vs o3</a>: In-depth DeepSeek-R1 vs o3 comparison: Latest benchmarks, pricing, context window, performance metrics, and technical specifications in 2025.</li><li><a href="https://llm-stats.com/models/compare/o3-mini-vs-deepseek-r1">o3-mini vs DeepSeek-R1</a>: In-depth o3-mini vs DeepSeek-R1 comparison: Latest benchmarks, pricing, context window, performance metrics, and technical specifications in 2025.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1336790750578475049)** (30 messages🔥): 

> `GRPO implementation success, Kolo support for Torchtune, Config issues with Llama 3.1 and Qwen 2.5, Hugging Face fast tokenizer support` 


- **GRPO implementation sees success**: A member reported a successful implementation of **GRPO** training, achieving training scores from **10% to 40%** on GSM8k.
   - Debugging issues noted include deadlocks and memory management challenges, but plans are being made to improve and open the project to contributions.
- **Kolo now supports Torchtune**: Excitement was shared as **Kolo** officially announced support for **Torchtune** on their [GitHub page](https://github.com/MaxHastings/Kolo).
   - The project provides a comprehensive solution for fine-tuning and testing LLMs locally using the best available tools.
- **Identified config issues with Llama 3.1 and Qwen 2.5**: Several members noted **FileNotFoundError** issues when downloading and fine-tuning **Llama 3.1** and **Qwen 2.5** due to mismatched path configurations.
   - A member created a [GitHub issue](https://github.com/pytorch/torchtune/issues/2352) to address the incorrect default paths and proposed fixes.
- **Future support for Hugging Face fast tokenizers**: The possibility of using **Hugging Face fast tokenizers** was discussed, indicating current limitations but ongoing progress.
   - A member mentioned that **Evan** is working on enabling support, as noted in a [GitHub pull request](https://github.com/pytorch/torchtune/pull/2350).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/MaxHastings/Kolo">GitHub - MaxHastings/Kolo: A one stop shop for fine tuning and testing LLMs locally using the best tools available.</a>: A one stop shop for fine tuning and testing LLMs locally using the best tools available. - MaxHastings/Kolo</li><li><a href="https://github.com/pytorch/torchtune/pull/2183.">Build software better, together</a>: GitHub is where people build software. More than 150 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama3_1">torchtune/recipes/configs/llama3_1 at main · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/issues/2352">Incorrect Default Config File Paths for Llama 3.1 8B and Qwen 2.5 7B Models · Issue #2352 · pytorch/torchtune</a>: I&#39;ve noticed an issue where the downloaded model directories for Llama 3.1 8B and Qwen 2.5 7B do not match the paths expected in their respective default config files. Llama 3.1 8B Issue The downl...</li><li><a href="https://github.com/pytorch/torchtune/blob/a226a58b8c36db5afa123f0885c5337d1ebc91f6/recipes/configs/qwen2_5/7B_lora_single_device.yaml#L33C3-L33C44">torchtune/recipes/configs/qwen2_5/7B_lora_single_device.yaml at a226a58b8c36db5afa123f0885c5337d1ebc91f6 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/issues/2340">Feature request: GRPO support · Issue #2340 · pytorch/torchtune</a>: As you all might have already known by now DeepSeek-R1 with its GRPO training was quite successful, should we consider bringing GRPO into torchtune?</li><li><a href="https://github.com/pytorch/torchtune/pull/2350">HF tokenizers: initial base tokenizer support by ebsmothers · Pull Request #2350 · pytorch/torchtune</a>: Fixes #2212This is an initial PR to support general tokenizers from Hugging Face via a tokenizer.json file. This is just a starting point to parse relevant JSON files, infer BOS and EOS, and defin...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1336835027014516887)** (16 messages🔥): 

> `GitHub Checks on Full DPO Distributed PR, GPU Testing Issues, Recipe Test Failures, VRAM Usage Optimization` 


- **GitHub Checks Fail on Full DPO PR**: A user reported issues with GitHub checks on their [Full DPO Distributed PR](https://github.com/pytorch/torchtune/pull/2275), with specific errors related to GPU and OOM issues.
   - The mentioned error was `ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!` and the user sought assistance from the community.
- **GPU Testing Issues Persist**: There were discussions about re-running workflows after tests failed, and concerns were raised about running software tests on machines with insufficient GPU capacity.
   - One member mentioned that the tests seemed to fail due to running on a CPU runner instead of a GPU runner, exacerbating the OOM issue.
- **Recipe Tests Encounter Compilation Errors**: Multiple failures were noted in the recipe tests, with one user indicating that issues arose from a bad PR that had previously merged.
   - Despite having two GPUs with 8GB VRAM each, users were surprised by OOM errors, prompting suggestions for optimizing resource usage.
- **Optimizing VRAM Usage for Tests**: To mitigate OOM errors, one user suggested enabling activation checkpointing, activation offloading, and using a smaller batch size.
   - Another user confirmed testing showed a peak of ~4 GB VRAM usage per GPU with 2 GPUs, indicating a reasonable usage level.
- **Future Review of PR Commit**: A user expressed hope that their latest commit in the PR would resolve existing issues.
   - Another member reassured them that they would review the PR again the following morning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://github.com/pytorch/torchtune/pull/2275/commits/fb228c6fb1a0c27795999b7811a55deedbd6bab4).">Build software better, together</a>: GitHub is where people build software. More than 150 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/sam-pi/torchtune/blob/add-feature-full-dpo/tests/recipes/test_full_dpo_distributed.py#L72.">torchtune/tests/recipes/test_full_dpo_distributed.py at add-feature-full-dpo · sam-pi/torchtune</a>: PyTorch native post-training library. Contribute to sam-pi/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/a226a58b8c36db5afa123f0885c5337d1ebc91f6/tests/recipes/test_full_finetune_distributed.py#L75">torchtune/tests/recipes/test_full_finetune_distributed.py at a226a58b8c36db5afa123f0885c5337d1ebc91f6 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/2275">Full DPO Distributed by sam-pi · Pull Request #2275 · pytorch/torchtune</a>: ContextAdapted from the great work in #1966What is the purpose of this PR? Is it to add a new featurePlease link to any issues this PR addresses: relates to #2082ChangelogWhat are the chang...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1337071483285147772)** (2 messages): 

> `Mojo language development, 12/18 community meeting insights` 


- **Mojo shifts away from Python Superset**: In the recent [12/18 community meeting](https://www.youtube.com/watch?v=XYzp5rzlXqM), it was clarified that **Mojo** is not currently a superset of **Python**.
   - The focus has now shifted towards leveraging **Mojo's** strengths in **GPU** and **performance programming**.
- **Chris provides insights on Mojo's future**: Chris discussed the **future direction** of **Mojo**, stating that it won’t evolve into an entirely different language but will concentrate on its current capabilities.
   - This approach emphasizes enhancing **Mojo's** efficiency in its designed applications rather than broadening its language framework.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=XYzp5rzlXqM)">Modular milestones: GPUs, 2024 reflections, and the road ahead 🚀</a>: In this extra special community meeting, we reflected on 2024&#39;s progress and shared updates on:🧑‍🚀 MAX 24.6, featuring MAX GPU!🔥 Our overall approach to M...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1336796515745796240)** (24 messages🔥): 

> `Parser Rewriting, Script Functionality, Mojo Open-Source Aspirations, UpdateDOM Function, Production Readiness of Mojo` 


- **Parser Needs Adjustment**: A member noted the need to rewrite the parser for handling multiple slices of data, while weighing the costs of branching.
   - *Branching may be cheaper than significant data transfers*, making it a valid consideration for those not focusing on higher performance needs.
- **Creating Dynamic Scripts**: The `update_dom` function was revised to create dynamic scripts by integrating all changes directly into the `Script` struct.
   - This change allows for returning a modified script copy using the **transfer sigil** (^), improving efficiency and structure.
- **Hopes for Mojo's Open-Source Future**: A user expressed the desire for Mojo to adopt an open-source approach similar to Google's with Go, rather than frameworks like Swift.
   - This sentiment was echoed with references to the open-source development styles of other programming languages, emphasizing community involvement.
- **Building on Mojo's Foundation**: Discussion emerged about the potential of what Modular could create with Mojo's open-source builds, likened to Unix's foundational impact.
   - Members expressed excitement about the possibilities and progress in Mojo's development, suggesting a significant evolution in the programming landscape.
- **Mojo's Production Readiness**: A user inquired about Mojo's current status regarding production readiness, highlighting curiosity among the community.
   - Responses indicated enthusiasm for Mojo's development trajectory, underscoring its promising nature even if not fully realized yet.



**Link mentioned**: <a href="https://stackoverflow.com/questions/21289806/link-to-class-method-in-python-docstring">Link to class method in Python docstring</a>: I want to add a link to a method in my class from within the docstring of another method of the same class. I want the link to work in Sphinx and preferentially also in Spyder and other Python IDEs...

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1336989420964020246)** (16 messages🔥): 

> `MAX Serve CLI, OpenAI Completion API Issues, OpenAI Model Compatibility, Msty client for local models` 


- **Discussion on CLI for MAX Serve**: Members discussed the possibility of building a CLI similar to **ollama** on top of **MAX Serve**. It was mentioned that MAX Serve can already handle many functionalities offered by Ollama with a docker container.
   - Specific features like local model running were highlighted, with a hope for better performance compared to Ollama.
- **Reporting OpenAI API Issues**: A user raised concerns about missing features in the **OpenAI completions API** with **max serve (v24.6)**, such as generation stopping at specified tokens. They were encouraged to file issues on the **GitHub repo** to highlight these missing elements.
   - Discussion ensued on how to report incidents, with a recommendation for multiple smaller issues for easier tracking and resolution.
- **Msty Client for Easier Access**: Bridging the conversation, a member introduced **Msty**, an OpenAI-compatible client that simplifies local model interactions compared to using Docker and other complex setups. Highlighting its ease of use and features, it was noted as a potential solution for accessing AI models seamlessly.
   - The importance of offline usability and privacy with Msty was emphasized, suggesting it is highly favorable for users who wish to avoid complex configurations.
- **Tracking OpenAI API Compatibility Issues**: The group acknowledged ongoing issues with OpenAI API compatibility, particularly referencing the **v1/models** endpoint. Several GitHub issues were highlighted to illustrate specific missing functionalities like token stopping and prompt handling.
   - The members expressed gratitude for the clarity, with developers indicating those issues will be communicated to internal teams for future improvements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modular/max/issues/294">[Feature Request] OpenAI API Compatibility: Models endpoint is missing · Issue #294 · modular/max</a>: What is your request? max serve (v24.6) openai api endpoint is missing the Models endpoint (https://platform.openai.com/docs/api-reference/models). The example below will return a 404: client = Ope...</li><li><a href="https://github.com/modular/max/issues">modular/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - modular/max</li><li><a href="https://msty.app">Msty - Using AI Models made Simple and Easy</a>: AI beyond just plain chat. Private, Offline, Split chats, Branching, Concurrent chats, Web Search, RAG, Prompts Library, Vapor Mode, and more. Perfect LM Studio, Jan AI, and Perplexity alternative. Us...</li><li><a href="https://www.modular.com/blog/use-max-with-open-webui-for-rag-and-web-search">Modular: Use MAX with Open WebUI for RAG and Web Search</a>: Learn how quickly MAX and Open WebUI get you up-and-running with RAG, web search, and Llama 3.1 on GPU</li><li><a href="https://github.com/modular/max/issues/293">[Feature Request] OpenAI API Compatibility: Only the first element of a list of prompts is considered during generation · Issue #293 · modular/max</a>: What is your request? Invoking max serve (v24.6) text generation via openai api endpoint with a list of prompts produces text generated only for the first element in a prompt. This behavior applies...</li><li><a href="https://github.com/modular/max/issues/292">[Feature Request] OpenAI API Compatibility: Text Generation Does not stop at specified `stop` argument. · Issue #292 · modular/max</a>: What is your request? Invoking max serve (v24.6) text generation via openai api endpoint, I encountered cases where text generation is not terminated when the token specified via the stop argument ...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1337063084358893630)** (36 messages🔥): 

> `Hibiki translation model, Melanie Mitchell's AI perspectives, Mistral AI's Le Chat, OpenAI's o3-mini updates, PDF parsing advancements` 


- **Meet Hibiki: Real-time Translation Champion**: Hibiki, the latest simultaneous speech-to-speech translation model from [kyutai](https://x.com/kyutai_labs/status/1887495488997404732), supports real-time translations from 🇫🇷 to 🇬🇧, preserving the speaker's voice and adjusting pace based on context.
   - Reports indicate Hibiki outperforms prior systems in **quality**, **naturalness**, and **speaker similarity**, nearing human interpreter capabilities.
- **Melanie Mitchell Draws Attention on AI**: @mmitchell_ai released a critical piece discussing why **Fully Autonomous Agents** should not be developed, highlighting ethical considerations and dissecting the concept of AI Agents ([paper](https://huggingface.co/papers/2502.02649)).
   - The conversation reflects varied views on her work, with some noting her balanced perspective amid ongoing debates in the AI community.
- **Unveiling Mistral AI's Le Chat**: [Mistral AI](https://x.com/MistralAI/status/1887517520040448510) announced the launch of Le Chat, described as the ultimate AI sidekick for both personal and professional tasks, now available on web and mobile.
   - This new tool aims to enhance user experience in daily activities, potentially changing how people interact with AI at work and in life.
- **Updated Features in OpenAI's o3-mini**: OpenAI shared updates on the chain of thought processes integrated into the o3-mini, available for [users](https://x.com/openai/status/1887616278661112259?s=46), enhancing capabilities for both free and paid subscribers.
   - These enhancements aim to improve performance and user experience, demonstrating OpenAI's commitment to evolving its services.
- **Advancements in PDF Parsing Technology**: @deedydas remarked that PDF parsing is effectively solved at scale, noting that **Gemini 2 Flash** offers parsing abilities for large documents at a minimal cost of $1 per 6000 tokens.
   - This breakthrough illustrates the growing efficiency in processing complex documents, opening new avenues for applications requiring high-quality text extraction.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/kyutai_labs/status/1887495488997404732">Tweet from kyutai (@kyutai_labs)</a>: Meet Hibiki, our simultaneous speech-to-speech translation model, currently supporting 🇫🇷➡️🇬🇧.Hibiki produces spoken and text translations of the input speech in real-time, while preserving the sp...</li><li><a href="https://x.com/MistralAI/status/1887517520040448510">Tweet from Mistral AI (@MistralAI)</a>: Introducing the all new Le Chat: your ultimate AI sidekick for life and work! Now live on web and mobile!</li><li><a href="https://aiguide.substack.com/">AI: A Guide for Thinking Humans | Melanie Mitchell | Substack</a>: I write about interesting new developments in AI. Click to read AI: A Guide for Thinking Humans, by Melanie Mitchell, a Substack publication with tens of thousands of subscribers.</li><li><a href="https://aiguide.substack.com/p/on-the-arc-agi-1-million-reasoning">On the “ARC-AGI” $1 Million Reasoning Challenge</a>: In this post I’m going to go into the weeds, describing how some people are trying to win a big $$$ prize for solving a still-wide-open AI challenge, the “Abstraction and Reasoning Corpus,” and what i...</li><li><a href="https://x.com/mmitchell_ai/status/1887442915602862389">Tweet from MMitchell (@mmitchell_ai)</a>: New piece out!We explain why Fully Autonomous Agents Should Not be Developed, breaking “AI Agent” down into its components & examining through ethical values.https://huggingface.co/papers/2502.02649Wi...</li><li><a href="https://x.com/neilzegh/status/1887498102455869775">Tweet from Neil Zeghidour (@neilzegh)</a>: Today we release Hibiki, real-time speech translation that runs on your phone. Adaptive flow without fancy policy, simple temperature sampling of a multistream audio-text LM. Very proud of @tom_labiau...</li><li><a href="https://x.com/deedydas/status/1887556219080220683">Tweet from Deedy (@deedydas)</a>: PDF parsing is pretty much solved at scale now.Gemini 2 Flash&#39;s $0.40/M tokens and 1M token context means you can now parse 6000 long PDFs at near perfect quality for $1</li><li><a href="https://aiguide.substack.com/p/the-llm-reasoning-debate-heats-up">The LLM Reasoning Debate Heats Up </a>: Three recent papers examine the robustness of reasoning and problem-solving in large language models</li><li><a href="https://x.com/openai/status/1887616278661112259?s=46">Tweet from OpenAI (@OpenAI)</a>: Updated chain of thought in OpenAI o3-mini for free and paid users, and in o3-mini-high for paid users.
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1336834313890693152)** (2 messages): 

> `Gemini 2.0 availability, LlamaParse for financial documents` 


- **Gemini 2.0 Launches with Day 0 Support**: Gemini 2.0 from **@google** is now generally available, providing **day 0 support** and impressive benchmarks. Developers can install the latest integration package with `pip install llama-index-llms-gemini` and check out the announcement [blog post](https://t.co/6oBbYpcFAU).
   - The updated **2.0 Flash** is available to all users in the **Gemini app** on desktop and mobile, promoting collaboration with Gemini's enhanced features and **low latency** capabilities.
- **LlamaParse Simplifies Financial Document Parsing**: Hanane D showcased how to tackle parsing **complex financial documents** accurately and cost-effectively using LlamaParse's 'Auto' mode on **LinkedIn**. She leverages **@OpenAI embeddings** and advanced **LlamaParse** capabilities for effective parsing of charts and tables, as shared in this [link](https://t.co/UMZXeXJ5pS).
   - Her demonstration highlights the **advancements in parsing technology**, making it easier for users to extract relevant insights from intricate data.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/6oBbYpcFAU">Gemini 2.0 is now available to everyone</a>: We’re announcing new updates to Gemini 2.0 Flash, plus introducing Gemini 2.0 Flash-Lite and Gemini 2.0 Pro Experimental.</li><li><a href="https://aistudio.google.com/prompts/new_chat?model=gemini-2.0-flash)">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1337101224331182121)** (4 messages): 

> `Embedding Print Removal, Pull Request Suggestion, Documentation Clarity` 


- **Request to Delete Embedding Print**: A member requested to delete the **embedding print** from the documentation as it takes up excessive space and affects readability.
   - They linked to a [GitHub issue](https://github.com/run-llama/llama_index/issues/17735) highlighting the **documentation issue** and suggested that it should be removed for better clarity.
- **Suggestion for Pull Request**: Another member acknowledged the request and offered to create a Pull Request (PR) to address the embedding print removal.
   - They indicated willingness to handle it if the original requester did not want to proceed with the PR.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/issues/17735">[Documentation]: Postgres Vector Store · Issue #17735 · run-llama/llama_index</a>: Documentation Issue Description Embeddings print takes up excessive space and affects readability. To improve clarity and docs usability, the embedding print should be removed Documentation Link ht...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/">Postgres Vector Store - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1336988389635002400)** (6 messages): 

> `LLMs in Classification, Latency Requirements in ML, Composite Pipeline for Noisy Data` 


- **LLMs excel in classification but struggle with noise**: A member emphasized that while **LLMs** are effective for classification, **noisy data** demands additional techniques like dense embeddings and autoencoder rerankers to enhance performance.
   - This indicates that a more complex approach may be necessary when dealing with challenging data environments.
- **Latency concerns when using LLMs**: Discussion revealed that while LLMs can classify well, it may be impractical to use them in scenarios with strict **latency requirements** due to their processing limits.
   - The conversation concluded that the suitability of LLMs really hinges on the specific latency constraints of a given application.
- **Reframing business requirements for ML solutions**: A member mentioned that there was a **missed opportunity** in properly framing the business requirements when transitioning to an ML problem.
   - They noted that it should have been apparent from the start that if low-latency is critical, traditional LLMs might not be the best fit.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1336934498247376939)** (6 messages): 

> `Fine-tuning Error, System Design Interview Questions` 


- **Fine-tuning Error and Batch Size Limits**: A user reported a **BadRequestError** (Status code: 400) indicating that the current training configuration exceeds the maximum of **250 training steps**, with a limit on **batch size** set to 16.
   - Concerns were raised about whether this means a limit of **4000 examples** for fine-tuning, as a member noted this limitation wasn't present before.
- **Inquiry on AIML System Design Questions**: A member asked if anyone has **system design interview questions** specific to **AI/ML**.
   - Another member acknowledged the inquiry and directed it for collection, signaling collaboration amongst teams.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1336952403538477127)** (4 messages): 

> `Tool-using model system prompts, Hugging Face dataset transformation issues, Dataset file format mismatch` 


- **Need for canonical system prompts**: A member inquired about the **canonical system prompts** for fine-tuned tool-using models to ensure they return responses or JSON for function calls.
   - They noted that the **Gorilla paper** did not include the system prompt used, creating a gap in available resources.
- **Experimenting with Hugging Face datasets**: A member expressed a desire to run experiments more easily by transforming data and utilizing `datasets.map` on **Hugging Face**.
   - This indicates a push towards enhancing the functionality and accessibility of datasets for experimentation.
- **Dataset format issue with Hugging Face**: A member pointed out that the dataset is in **.json** format, but its content is actually in **jsonl** format, which has caused issues with Hugging Face.
   - They suggested changing the file suffix to **.jsonl** and modifying the dataset config files to potentially resolve the issue.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: https://arxiv.org/abs/2502.02508
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1336822412875927653)** (2 messages): 

> `Git Repository, Colab Notebook` 


- **Inquiry about Git Repo**: A member asked if there is a **Git repo** available for their work.
   - The inquiry suggests interest in accessing code or resources related to the project.
- **Colab Notebook Shared**: A member provided a link to a [Colab notebook](https://colab.research.google.com/drive/1OXmTKexR9gX33DXRNEAe3dNuxkLXnutX?usp=sharing) in response to the Git repo query.
   - This notebook is likely related to the discussion and can be accessed by **signing in**.



**Link mentioned**: <a href="https://colab.research.google.com/drive/1OXmTKexR9gX33DXRNEAe3dNuxkLXnutX?usp=sharing">Google Colab</a>: no description found

  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

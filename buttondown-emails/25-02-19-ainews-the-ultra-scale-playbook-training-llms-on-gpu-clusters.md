---
id: 627736a9-c551-42f4-ab08-e615b14cfbca
title: '[AINews] The Ultra-Scale Playbook: Training LLMs on GPU Clusters'
date: '2025-02-20T05:57:17.513081Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-the-ultra-scale-playbook-training-llms-on
---

<!-- buttondown-editor-mode: plaintext -->**2 days is all you need to read this.**

> AI News for 2/18/2025-2/19/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**211** channels, and **6631** messages) for you. Estimated reading time saved (at 200wpm): **700 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

In seeming response to [DeepMind's How To Scale Your Model](https://buttondown.com/ainews/archive/ainews-how-to-scale-your-model-by-deepmind/), Huggingface came from nowhere to drop a massive "blogpost" equivalent for GPUs: [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook).

![image.png](https://assets.buttondown.email/images/5e68bc3f-1cd8-4412-9b5b-157b5d60ff8d.png?w=960&fit=max)

This is a great starting point for people looking for intuitive, detailed understanding of modern training constraints and strategies to scale things up on GPUs, with a first-principles build up of modern best practices:

![image.png](https://assets.buttondown.email/images/2482f42c-d8f1-4aba-a980-57944e830701.png?w=960&fit=max)

and not to mention the blogpost is interactive, based on real data backed by 4000 scaling experiments on up to 512 GPUs.

Not strictly required for AI Engineers, but a fantastic starting point for anyone looking to get up to speed on training terminology.

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

- **DeepSeek's Native Sparse Attention (NSA) model** is generating significant interest. [@eliebakouch](https://twitter.com/eliebakouch/status/1892276069891240286) shared **more details** about it and congratulated the team including [@Nouamanetazi](https://twitter.com/Nouamanetazi), [@lvwerra](https://twitter.com/lvwerra), and [@Thom_Wolf](https://twitter.com/Thom_Wolf). [@ProfTomYeh](https://twitter.com/ProfTomYeh/status/1892245518580932812) mentioned he would **draw DeepSeek's Native Sparse Attention in a live webinar**, sharing a sketch of it, and later announced he would **explain the DeepSeek paper by drawing circles and lines** in another webinar hosted by Alex Wang. [@hkproj](https://twitter.com/hkproj/status/1892107497369915535) noted that **DeepSeek releasing a paper makes the whole ML community pay attention**, highlighting their soft power. [@qtnx_](https://twitter.com/qtnx_/status/1891989260825153569) mentioned that **command r 7b is their favorite transformer implementation**.
- **Perplexity AI released R1-1776**, an **uncensored, unbiased, and factual** version of the DeepSeek R1 model, as announced by [@AravSrinivas](https://twitter.com/AravSrinivas/status/1891958274880139484) who teased **more cool releases coming this week and next week**. [@_akhaliq](https://twitter.com/_akhaliq/status/1891961543455031429) also highlighted the release, describing it as **post-trained to remove Chinese Communist Party censorship**.
- **Google DeepMind released PaliGemma 2 Mix**, an open, multi-task vision-language model capable of tasks like **outfit judging and object counting**, as announced by [@_philschmid](https://twitter.com/_philschmid/status/1892258568176320927). [@mervenoyann](https://twitter.com/mervenoyann/status/1892267634923593760) further detailed **PaliGemma 2 Mix**, highlighting its versatility for vision language tasks with open-ended prompts, document understanding, and segmentation/detection, available in **3B, 10B, and 28B sizes**.
- **Microsoft introduced Muse**, a generative AI model **trained on Ninja Theory’s game Bleeding Edge**, with model weights and code released, as shared by [@reach_vb](https://twitter.com/reach_vb/status/1892254399633801283) who indicated it signals that **making your own gaming experience is coming sooner than you'd think**.
- **Baichuan-M1**, an opensource **SotA medical LLM (Baichuan-M1-14B)**, trained from scratch on **20T tokens** with a focus on medical capabilities, was announced by [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892053102427300212).
- **Fully open-source 40B model for genome modeling and design** across all domains of life, using the new StripedHyena 2 architecture, was announced by [@maximelabonne](https://twitter.com/maximelabonne/status/1892256596136165535).

**Research and Papers**

- **Microsoft presented Magma**, a Foundation Model for Multimodal AI Agents, achieving **SotA on UI navigation and robotic manipulation tasks**, pretrained on a large dataset annotated with Set-of-Mark (SoM) and Trace-of-Mark (ToM), as highlighted by [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892059107479224384) and [@_akhaliq](https://twitter.com/_akhaliq/status/1892074819795005518).
- **Meta presented NaturalReasoning**, a dataset for Reasoning in the Wild with **2.8M Challenging Questions**, as shared by [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892054528473657669). [@jaseweston](https://twitter.com/jaseweston/status/1892041992127021300) detailed the release of **NaturalReasoning**, highlighting its **2.8M challenging and diverse questions** requiring multi-step reasoning, showing **steeper data scaling curves** and potential for self-training.
- **DeepMind released PaliGemma 2 Checkpoints**, tailored for tasks like **Optical Character Recognition and Captioning**, in sizes ranging from 3B to 28B, all open weights and compatible with Transformers, as mentioned by [@reach_vb](https://twitter.com/reach_vb/status/1892269416831717648).
- **Hugging Face released Ultra Scale Playbook for Training LLMs on GPU Clusters**, a free, open-source book covering 5D parallelism, ZeRO, fast CUDA kernels, and compute & communication overlap, based on 6 months of scaling experiments, as announced by [@reach_vb](https://twitter.com/reach_vb/status/1892276287039033473).
- **"Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity"** is a new paper highlighted by [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892059638943637960).
- **"Revisiting the Test-Time Scaling of o1-like Models"** paper was shared by [@_akhaliq](https://twitter.com/_akhaliq/status/1892071152215785526), questioning if they truly possess test-time scaling capabilities.
- **"ByteDance presents Phantom: Subject-consistent video generation via cross-modal alignment"** paper was shared by [@_akhaliq](https://twitter.com/_akhaliq/status/1892073250974216476).
- **"Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs"** paper was mentioned by [@_akhaliq](https://twitter.com/_akhaliq/status/1892247778291503296).
- **"Learning to Reason at the Frontier of Learnability"** paper was highlighted by [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1892156610304438297), focusing on curriculum learning in LLMs using sampling for learnability in RL.
- **"Is Noise Conditioning Necessary for Denoising Generative Models?"** paper was shared by [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1892053059221717486), exploring denoising-based generative models without noise conditioning, finding graceful degradation and sometimes better performance.

**Tools and Libraries**

- **LangChain announced LangGraph Studio's Playground integration** for faster prompt iteration, allowing users to view LLM calls directly and iterate on prompts without re-running the whole graph, as per [@LangChainAI](https://twitter.com/LangChainAI/status/1892288249441505444). They also introduced **Langchain MCP Adapters**, enabling instant connection of LangGraph agents to hundreds of tools in the MCP ecosystem, and **`npm create langgraph`** for bootstrapping LangGraph.js agents with templates.
- **Modular released MAX 25.1**, a significant release enabling new agentic workflows, GPU custom ops in Mojo, and a new portal for dev content, as announced by [@clattner_llvm](https://twitter.com/clattner_llvm/status/1892294758007263724).
- **Together AI announced Test Drive of NVIDIA Blackwell GPUs** with Together GPU Clusters, offering free access to eight AI teams to optimize models and accelerate training, as per [@togethercompute](https://twitter.com/togethercompute/status/1892256276576686131). They also highlighted **Scaled Cognition training APT-1 on Together GPU Clusters**.
- **LM Studio now supports speculative decoding**, as announced by [@cognitivecompai](https://twitter.com/cognitivecompai/status/1892252216515293381).
- **LlamaCloud EU**, a secure, compliant knowledge management SaaS offering ensuring full data residency within the EU, was announced by [@llama_index](https://twitter.com/llama_index/status/1892271183451869512).
- **Gradio** is highlighted as the tool used to build most AI apps today, according to [@_akhaliq](https://twitter.com/_akhaliq/status/1892260638191235244).

**Industry News and Events**

- **April 29** is the date for the first-ever **LlamaCon**, and **Meta Connect is scheduled for September 17-18**, as announced by [@AIatMeta](https://twitter.com/AIatMeta/status/1891969855043313945).
- **LangChain is hosting events**, including an evening meetup in NYC on February 19th, and an AI event in Atlanta on February 27th, as posted by [@LangChainAI](https://twitter.com/LangChainAI/status/1892288795841876319) and [@LangChainAI](https://twitter.com/LangChainAI/status/1892301688138047842). [@hwchase17](https://twitter.com/hwchase17/status/1892302989420794105) also mentioned being in **Atlanta**.
- **AI Engineer Summit in NY** is happening, with [@HamelHusain](https://twitter.com/HamelHusain/status/1892025630079787214) mentioning being in NY for it.

**AI Agents and Applications**

- **Evaluating AI Agents** is the focus of a new short course from DeepLearningAI in partnership with Arize AI, taught by [@JohnGilhuly](https://twitter.com/JohnGilhuly) and [@_amankhan](https://twitter.com/_amankhan), covering systematic assessment and improvement of AI agent performance, as announced by [@AndrewYNg](https://twitter.com/AndrewYNg/status/1892258190546653392) and [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1892250339878818124).
- **AI co-scientist, a multi-agent AI system built with Gemini 2.0 by Google**, is introduced to accelerate scientific breakthroughs, as reported by [@omarsar0](https://twitter.com/omarsar0/status/1892223515660579219) in a detailed thread outlining its capabilities, such as generating novel hypotheses, outperforming other SoTA models, and leveraging test-time compute.
- **Weights & Biases hosted a Multimodal AI Agents Hackathon**, with over 200 innovators and 40 teams participating, showcasing creativity and iterations in building AI agents, as mentioned by [@weights_biases](https://twitter.com/weights_biases/status/1892254937901396068).
- **Microsoft just dropped MUSE - a generative AI model trained on Ninja Theory’s multiplayer battle arena game, Bleeding Edge** for gameplay, according to [@reach_vb](https://twitter.com/reach_vb/status/1892254399633801283).
- **AI that learns from gameplay and generates visuals** is highlighted as a new chapter in gaming by [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1892244909484077354).

**Quantum Computing Breakthrough**

- **Microsoft's quantum team** is congratulated by [@stevenheidel](https://twitter.com/stevenheidel/status/1892274904638058609) for a breakthrough, with [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1892258540460331228) announcing a **major step toward making quantum computing a reality**, unlocking the power to solve problems beyond today’s computers, potentially reshaping industries and accelerating scientific discovery. [@cognitivecompai](https://twitter.com/cognitivecompai/status/1892258078210375867) questioned [@satyanadella](https://twitter.com/satyanadella) whether **topological superconductor is actually a new state of matter**. [@jeremyphoward](https://twitter.com/jeremyphoward/status/1892250092800791009) also reacted to the news, quoting **"we’ve created an entirely new state of matter"**.

**Memes and Humor**

- [@nearcyan](https://twitter.com/nearcyan/status/1892049572546904319) stated that **everyone who bought the $700 AI pin got literally rugged**, gaining significant traction.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. o3-mini Replaces DeepSeek as This Year's LLaMA Front-runner**

- **[o3-mini won the poll! We did it guys!](https://i.redd.it/ogpvvrth70ke1.jpeg)** ([Score: 1688, Comments: 186](https://reddit.com/r/LocalLLaMA/comments/1isu4un/o3mini_won_the_poll_we_did_it_guys/)): **o3-mini** won a Twitter poll with **54%** of the **128,108** votes, beating the "phone-sized model" option. A reply from Ahmad suggests the poll was misleading and encourages votes for "o3-mini," highlighting community engagement with likes, quotes, and reposts.
  - **Model Distillation Discussion**: Users discussed the process of **distilling a model**, which involves creating a smaller, faster version by training a compact model (student) to mimic a larger model (teacher). Techniques such as knowledge distillation, pruning, and quantization are used, but OpenAI's closed-source nature limits direct distillation from their models.
  - **Skepticism on Model Releases**: There is skepticism about whether OpenAI will release an open-source model, as historically they've moved away from open models. Users are wary of terms like "o3-mini level," suggesting it might not meet expectations or be a significantly downgraded version.
  - **Community Reactions and Sarcasm**: The community expressed mixed feelings about the Twitter poll results and the potential release of a "phone-sized" model. Comments highlighted the use of sarcasm and skepticism regarding the actual utility and performance of such models, with some users humorously noting the potential for underwhelming releases.


**Theme 2. AMD Laptops with 128 GB Unified Memory Challenges Apple Dominance**

- **[New laptops with AMD chips have 128 GB unified memory (up to 96 GB of which can be assigned as VRAM)](https://www.youtube.com/watch?v=IVbm2a6lVBo)** ([Score: 482, Comments: 157](https://reddit.com/r/LocalLLaMA/comments/1isxhoy/new_laptops_with_amd_chips_have_128_gb_unified/)): **New AMD laptops** now feature **128 GB unified memory**, allowing up to **96 GB** to be allocated as **VRAM**.
  - Discussions highlight the **performance and versatility** of the new AMD laptops with **128 GB unified memory**. Reviewers like **JustJosh** and **Dave2D** praise their capability to run **LLMs** and Linux, challenging Mac's dominance in unified memory for large model processing. **b3081a** mentions running **vLLM** with specific configurations for optimized performance on these devices.
  - The **pricing and comparison** with Apple devices are significant discussion points. The **Asus 128GB version** is priced at **$2799**, cheaper than a comparable Apple device at **$4700**. Users debate the value and performance differences, with some noting potential cost benefits over high-end GPUs like the **RTX 5090**.
  - There is interest in **Linux support** and potential desktop applications. **Kernel 6.14** is expected to bring full Linux support for NPUs in these devices, and users express interest in **Mini PCs** and **Framework 13 mainboards** with these chips, discussing the benefits of shared RAM and unified memory in various configurations.


**Theme 3. Gemini 2.0's Superior Audio Transcription with Speaker Labels**

- **[Gemini 2.0 is shockingly good at transcribing audio with Speaker labels, timestamps to the second;](https://i.redd.it/d3bl014yx2ke1.png)** ([Score: 387, Comments: 90](https://reddit.com/r/LocalLLaMA/comments/1it36b0/gemini_20_is_shockingly_good_at_transcribing/)): **Gemini 2.0** excels in audio transcription with precise **speaker labels** and **timestamps** to the second, as highlighted by **Matt Stanbrell** on Twitter. The tool's ability to recognize various sounds and provide detailed transcriptions encourages users to upload audio files for enhanced summarization and speaker identification.
  - **Gemini 2.0's** transcription capabilities are praised for **Vietnamese language accuracy**, including tones, with users finding it highly reliable for language learning, as noted by **Mescallan**. However, **leeharris100** from an ASR company critiques its **timestamp accuracy** and mentions it hallucinates with longer contexts, though it remains competitive in general **WER** (Word Error Rate) with models like **Whisper medium**.
  - There is a consensus that **Google's Gemini models** are not open-sourced, preventing local use like **Whisper**, as highlighted by **CleanThroughMyJorts**. **nrkishere** and **silenceimpaired** express skepticism about its local running capabilities and open-sourcing potential.
  - **Gemini 2.0** is noted for its **object identification** and **graph understanding** capabilities, with users like **Kathane37** impressed by its performance. **space_iio** attributes its effectiveness to Google's access to **YouTube videos and metadata**, enhancing its training data beyond typical scraping methods.


**Theme 4. Unsloth's R1-1776 Dynamic GGUFs with High Accuracy**

- **R1-1776 Dynamic GGUFs by Unsloth** ([Score: 132, Comments: 53](https://reddit.com/r/LocalLLaMA/comments/1it0ocl/r11776_dynamic_ggufs_by_unsloth/)): **Unsloth** released **R1-1776 GGUFs** ranging from **2-bit to 16-bit**, including **Dynamic 2-bit, 3-bit, and 4-bit** versions, with the **Dynamic 4-bit** being smaller yet more accurate than the medium version. The models, available on **[Hugging Face](https://huggingface.co/unsloth/r1-1776-GGUF)**, require specific token formatting and offer instructions in the model card, with additional insights available in their **[blog](https://unsloth.ai/blog/r1-reasoning)**.
  - **Resource Requirements**: Running **R1-1776 GGUFs** doesn't necessitate VRAM, but for optimal performance, at least **120GB of VRAM + RAM** is recommended. The **dynamic 2-bit version** requires **211GB of disk space** and specific formatting guidance is provided to enhance model output.
  - **Model Performance and Benchmarks**: **Dynamic quants** of the R1 model have shown to outperform or match the original **16-bit models** in benchmarks submitted to the **Hugging Face leaderboard**. However, users noted the need for more comprehensive benchmarks beyond the **Flappy Bird** test to fully assess performance.
  - **Future Developments and Releases**: There are upcoming releases focusing on **long context** and other features requested by over **10,000 users**, indicating a strong community engagement. Additionally, there are plans to support **distillation to smaller models**, and potential updates for **V3** and **V2.5-1210** versions to improve accessibility and performance.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. DeepSeek GPU Smuggling Probe: Uncovering Nvidia's Singapore Revenue Anomalies**

- **[DeepSeek GPU smuggling probe shows Nvidia's Singapore GPU sales are 28% of its revenue, but only 1% are delivered to the country: Report](https://www.tomshardware.com/tech-industry/deepseek-gpu-smuggling-probe-shows-nvidias-singapore-gpu-sales-are-28-percent-of-its-revenue-but-only-1-percent-are-delivered-to-the-country-report)** ([Score: 286, Comments: 90](https://reddit.com/r/OpenAI/comments/1it5d6m/deepseek_gpu_smuggling_probe_shows_nvidias/)): **DeepSeek's investigation** reveals that **Nvidia's GPU sales in Singapore** account for **28% of its revenue**, yet only **1%** of these GPUs are actually delivered to the country. This discrepancy suggests possible **GPU smuggling activities**.
  - The **DeepSeek V3 paper** was questioned for its claim of using **8-bit floating point (FP8)**, which would be a significant shift in AI model training if true. Discussions reveal skepticism about the authenticity of DeepSeek's claims, with some suggesting it was overhyped by media outlets.
  - **Singapore's role as a major trading port** was highlighted, explaining the high percentage of Nvidia's revenue attributed to the country despite low actual delivery. This aligns with Singapore's trade-to-GDP ratio of over **300%**, suggesting potential re-exporting or processing activities rather than direct smuggling.
  - **Nvidia's actions** are perceived as attempts to circumvent US laws, with some commenters noting the misleading nature of article titles and the strategic importance of GPUs. The discussion touches on broader geopolitical implications, including comparisons to China's potential control over strategic assets.


- **[Accurate](https://i.redd.it/tj7ei7ndq0ke1.jpeg)** ([Score: 286, Comments: 15](https://reddit.com/r/ChatGPT/comments/1iswa0c/accurate/)): The post humorously contrasts a serious medical setting with a casual response, featuring a patient in an **MRI machine** asking a doctor about results, while the monitor shows the **"ChatGPT-4o"** logo. This juxtaposition highlights the comedic tone by mixing a traditionally serious context with a relaxed, informal reply.
  - **Medical AI Assistance**: **Human-Independent999** suggests that AI's best application in the medical field is to assist doctors, implying that such integration may already be occurring. This reflects a common sentiment about AI's potential to enhance medical practices.
  - **Futuristic Technology Comparisons**: **PerennialPsycho** humorously compares the scenario to **Star Trek**, where a doctor uses a simple tool for diagnosis, highlighting the futuristic and simplified vision of AI in healthcare.
  - **Optimism for AI's Future**: **Potential_Club_9494** expresses optimism about AI's future impact, hinting at the transformative potential that AI like **ChatGPT** may have on various fields.


- **[Tweet from an OpenAI researcher](https://i.redd.it/vgyttoqy55ke1.jpeg)** ([Score: 273, Comments: 27](https://reddit.com/r/OpenAI/comments/1itd425/tweet_from_an_openai_researcher/)): **Aidan Clark** tweeted about "Sonnet 4" and congratulated "Demis & Team," gaining significant attention with **12.7K views**, **34 replies**, **25 retweets**, and **239 likes** as of February 19, 2025.
  - There is confusion and speculation about the **"Sonnet 4"** and **Demis Hassabis**'s involvement, with some users questioning if **Demis** is being congratulated for achievements related to **DeepMind** or if it's a form of trolling due to **Google's** investments in **Anthropic**.
  - **Grok 4.5** is mentioned with enthusiasm, suggesting a possible mix-up or joke in the conversation, as **Demis Hassabis** is associated with **DeepMind**, not **Anthropic**.
  - Several users express confusion about what **"Sonnet 4"** is, with some humorously suggesting it might be a cross-universe or fictional concept, indicating a lack of clarity or potential trolling in the original tweet.


**Theme 2. Google's NotebookLM: A Gamechanger in AI Research Tools**

- **NotebookLM the Most Underrated AI Tool!** ([Score: 1143, Comments: 104](https://reddit.com/r/ChatGPT/comments/1isvvpr/notebooklm_the_most_underrated_ai_tool/)): Google's **NotebookLM** is highlighted as an underrated AI tool, combining features of **ChatGPT, Perplexity,** and **Notion AI** while offering automatic source citations to eliminate hallucinations. It excels in reading and summarizing PDFs, Docs, and notes, and remembers uploaded files better than **ChatGPT**, making it a valuable tool for researchers, students, and professionals handling large data volumes.
  - Users praise **NotebookLM** for its unique podcast feature, which allows for interactive and customizable experiences, likening it to having a personal AI radio show. However, there are concerns about potential hallucinations, especially on the free tier, and the possibility of Google discontinuing the service.
  - Some users express interest in **NotebookLM**'s potential to manage and summarize data, such as expenses in Excel sheets, but there are critiques regarding its chat feature and UI limitations. The tool is noted for its superior source management and citation capabilities compared to other tools.
  - There is a desire for an open-source, local alternative to **NotebookLM**, reflecting concerns about the longevity of Google's support for the tool. Users are also curious about the pricing model and its integration with **Google One AI**.


- **['Improved Amateur Snapshot Photo Realism' v12 [FLUX LoRa] - Fixed oversaturation, slightly improved skin, improved prompt adherence and image coherence (20 sample images) - Now with a Tensor.art version!](https://www.reddit.com/gallery/1it3acw)** ([Score: 239, Comments: 18](https://reddit.com/r/StableDiffusion/comments/1it3acw/improved_amateur_snapshot_photo_realism_v12_flux/)): **'Improved Amateur Snapshot Photo Realism' v12 [FLUX LoRa]** is a significant update that addresses issues of **oversaturation**, enhances **skin texture**, and improves **prompt adherence** and **image coherence**. The update includes **20 sample images** and introduces a version compatible with **Tensor.art**.
  - **Image Quality Concerns**: Users like **animerobin** note that **FLUX LoRa** models often suffer from low-resolution pixel fuzziness, questioning how to achieve crisper images.
  - **Dataset Sourcing**: **TheManni1000** seeks advice on finding image datasets for training similar models, with suggestions including **Reddit image subs**, **Instagram**, **Flickr**, and datasets from **Hugging Face** and **Kaggle**.
  - **Resource Links and Terminology**: **AI_Characters** provides links to the model on [CivitAI](https://civitai.com/models/970862/improved-amateur-snapshot-photo-realism-style-lora-flux-spectrum0001-by-aicharacters) and **Tensor.art**, while a discussion arises around the term "amateer" being a buzzword for "AI generated celebrity," with some confusion over its consistent depiction.


**Theme 3. Claude 3.5 Sonnet: A Benchmark in AI Coding and Consistency**

- **[Claude reasoning. Anthropic may make offical announcement anytime soon..](https://i.redd.it/bel9ndn6i2ke1.jpeg)** ([Score: 222, Comments: 87](https://reddit.com/r/ClaudeAI/comments/1it1uil/claude_reasoning_anthropic_may_make_offical/)): **Claude 3.5 Sonnet** interface showcases features such as "Camera," "Photos," and "Files" buttons, along with options to "Choose style" and toggle "Use extended thinking" for PRO users. The interface hints at a **limited daily message** feature on the free plan and an "Upgrade" option, indicating potential for enhanced functionality.
  - Users express frustration over the **API pricing** of Claude, with one user noting **$0.60** per use for paraphrasing a few paragraphs, leading them to prefer the web version despite its limited daily messages. The **daily message limit** is a significant pain point, with users feeling restricted in their usage.
  - There is skepticism around the **Claude 3.5 Sonnet** update, with some users suspecting it's merely a rebranding of existing features like **MCP servers** rather than introducing new capabilities. The addition of reasoning and potential **web search** features is noted, but users remain critical of the lack of substantial improvements.
  - Users report that the **new features** are not available on **iOS** or **Android** for some, despite app updates, leading to confusion about the rollout. The community is also critical of **Anthropic's** focus on updates without addressing fundamental issues like message limits and practical enhancements.


- **What the fuck is going on?** ([Score: 226, Comments: 185](https://reddit.com/r/ClaudeAI/comments/1it6yij/what_the_fuck_is_going_on/)): **Claude 3.5 Sonnet** is praised for its superior coding reliability and consistency compared to models like **DeepSeek**, **O3**, and **Grok 3**, despite not incorporating new techniques such as AI self-looping. The author notes that while these newer models are interesting for their novel approaches, **Claude 3.5 Sonnet** remains unmatched, though it hasn't shown significant improvements recently.
  - Many users argue that **Claude 3.5 Sonnet** is not universally superior, as it struggles with specific tasks like programming contest problems and architecture issues, where models like **O1** and **O3-mini** perform better. Critics highlight that **Claude**'s higher context window (200k vs. 32k) compared to **ChatGPT** might contribute to its performance discrepancies ([Reddit post](https://www.reddit.com/r/OpenAI/comments/1is2bw8/chatgpt_vs_claude_why_context_window_size_matters/)).
  - Some users express skepticism about the emphasis on AI benchmarks and leaderboards, suggesting that they may not accurately reflect real-world usability and intelligence. They argue that focusing solely on benchmark performance can detract from the practical utility and user-friendliness of AI models.
  - There is anticipation for future releases, such as **Claude 4.0** and **Opus 4.0**, which are expected to surpass current models, while others note that **Gemini Pro 2.0** and reasoning models like **R1** and **O3** are currently preferred for tasks requiring different aspects of intelligence. Some users mention upcoming hybrid models with reasoning capabilities, hinting at advancements in AI self-looping and reasoning.


**Theme 4. OpenAI's 4o Model: Excelling in Creative Writing and Narrative Continuity**

- **4o Creative Writing is Phenomenal!** ([Score: 135, Comments: 53](https://reddit.com/r/OpenAI/comments/1isu7bw/4o_creative_writing_is_phenomenal/)): The post discusses the **OpenAI 4o model** and its impressive capabilities in creative writing, particularly in genres like **sci-fi and fantasy**. The author highlights the model's ability to maintain character consistency in continuing book series and notes the potential influence of having **128k context** with a pro subscription.
  - Users are impressed with the **OpenAI 4o model**'s ability to maintain character consistency and extend stories naturally, especially with the **128k context**. However, some users find that the model sometimes defaults to generic storytelling, particularly in genres like horror, and struggles with programming tasks, requiring manual intervention for accurate code generation.
  - The recent updates to the model, including **end of January fine-tuning** and **mid-February content filter relaxation**, have been noted to enhance its natural language processing capabilities. Some users suggest the possibility of **OpenAI's reasoning models** being integrated under the hood for improved performance.
  - Creative writing with the model is seen as entertaining and personalized, with users sharing strategies like using the **narrative technique** to guide story development. There are also discussions on the model's capacity for handling large output sizes, with alternatives like **flash thinking** mentioned for higher token limits.


**Theme 5. SFW Hunyuan Video LoRAs: Expanding Creative AI Video Applications**

- **I will train & open-source 50 SFW Hunyuan Video LoRAs. Request anything!** ([Score: 144, Comments: 161](https://reddit.com/r/StableDiffusion/comments/1isrnoe/i_will_train_opensource_50_sfw_hunyuan_video/)): The author plans to train and open-source **50 SFW Hunyuan Video LoRAs**, leveraging nearly unlimited compute power. They invite suggestions for training ideas, promising to prioritize the most upvoted requests and personal preferences.
  - There is significant interest in **martial arts fight scenes** and **heavy cinematic film styles**, with suggestions to explore **extreme wide-angle shots** and **different martial art styles** for diverse visual effects. **Wes Anderson** aesthetics and **Film Noir** cinematics are also popular requests for unique visual storytelling.
  - Some commenters emphasize the potential of **fine-tuning models** to their limits for high-quality outputs, suggesting themes like **360-degree character turnarounds**, **VR and SBS 3D videos**, and **immersive live-action GoPro footage**. Others propose focusing on **unique video training** such as **camera movements** and **action sequences** over still images.
  - There are creative suggestions for thematic content, including **cyberpunk settings** like **Cyberpunk 2077** and **Blade Runner 2049**, **dark fantasy styles**, and **alien lore**. Additionally, there is interest in capturing **film dialogue moments** with nuanced emotional expressions and **2D animation styles** beyond anime.


- **[Anthropic to release reasoning and other cool stuff soon..](https://i.redd.it/all95og7t1ke1.jpeg)** ([Score: 218, Comments: 44](https://reddit.com/r/ClaudeAI/comments/1iszwxj/anthropic_to_release_reasoning_and_other_cool/)): **Anthropic** is set to release new features for the **Claude iOS app**, focusing on **Thinking Models** and **Web Search** capabilities. The announcement, originally made on **Twitter** by user **@M1Astra**, includes new icons for features like "Steps," "Think," and "Magnifying Glass," with the post dated **February 19**.
  - Users express a desire for **Claude** to address issues like **rate limits**, memory, and longer conversations, with some considering a return to subscriptions if these features improve. **Web search** capability is also a highly requested feature to access the latest information online.
  - Some commenters, like **ChrisT182**, highlight **Anthropic's focus on AI safety and understanding** as a reason for their sparse updates, suggesting that this groundwork is crucial for future advancements towards **AGI**.
  - There are hopes for new features like **Claude's voice**, with **Hir0shima** and others expressing anticipation for this update, while some users have switched to alternatives like **deepseek** due to dissatisfaction with current offerings.


- **[ChatGPT Founder Shares The Anatomy Of The Perfect Prompt Template](https://i.redd.it/1rr5tpfc44ke1.jpeg)** ([Score: 1232, Comments: 62](https://reddit.com/r/ChatGPT/comments/1it7t6w/chatgpt_founder_shares_the_anatomy_of_the_perfect/)): **Greg Brockman** shared a tweet detailing the structure of an optimal prompt for **ChatGPT**, focusing on creating unique weekend getaways near **New York City**. The prompt includes sections such as "Goal," "Return Format," "Warnings," and "Context dump" to ensure clarity and effectiveness in generating responses.
  - **Prompt Structure and Effectiveness**: There is a discussion about the effectiveness of **prompt structure**, with some users noting that placing the goal or question at the top can lead to less desirable responses, while others argue that placing it at the bottom or using data separators improves outcomes. **ArthurParkerhouse** suggests using a triple hash separator to distinguish between context and tasks, which aligns with **MemeMan64209**'s observation of better results when questions are placed last.
  - **AI Interaction and User Experience**: **Fit-Buddy-9035** compares prompting AI to communicating with a logical, direct individual, emphasizing the need for clarity and explicitness, while **Professional-Noise80** highlights that crafting effective prompts requires critical thinking. **TheSaltySeagull87** notes that the effort in creating detailed prompts can be comparable to traditional research methods like using Google or Reddit.
  - **Cognitive Processing in AI Models**: **MaintenanceOk3364** suggests that AI models, like humans, prioritize initial information, but **MemeMan64209** and **ArthurParkerhouse** observe that AI might prioritize the last tokens read, reflecting a discrepancy in perception of how AI processes information. This highlights the complexity and variability in AI's cognitive processing, sparking further debate on optimal prompt design.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12

**Theme 1: Grok 3 Takes Center Stage Amid Mixed Reactions**

- **Grok 3 Melts Servers and Minds!**: xAI's **Grok 3** is now free until server capacity is reached, offering unprecedented access to the "world's smartest AI" as announced in [this tweet](https://x.com/xai/status/1892400129719611567). Users speculate about usage limits and potential server overloads.
- **Grok 3 Hype Meets Skepticism**: While some users hail **Grok 3** as superior to **ChatGPT-4**, others find its reasoning capabilities underwhelming compared to models like **Claude 3.5** and **O1**. The community is abuzz with comparisons and debates over its true prowess.
- **Elon Enters Gaming with Grok 3**: **Elon Musk's xAI** announced the creation of a new game studio tied to **Grok 3**, signaling a strategic move to integrate AI with gaming, as hinted in [Elon's tweet](https://x.com/elonmusk/status/1891388509191049307).

**Theme 2: AI CUDA Engineer Supercharges Kernel Optimization**

- **AI Codes for AI: CUDA Kernels Get a Boost**: **Sakana AI** unveiled the [**AI CUDA Engineer**](http://sakana.ai/ai-cuda-engineer/), automating optimized CUDA kernel creation and achieving **10-100x speedups** over standard PyTorch operations.
- **Kernel Wizardry Impresses the Community**: The AI CUDA Engineer boasts a **90% success rate** in translating PyTorch to CUDA and released a dataset of **17,000 verified CUDA kernels**, marking a breakthrough in AI-driven performance optimization.
- **Dr. Jim Fan Applauds Autonomous Coding Agent**: In a [tweet](https://x.com/DrJimFan/status/1892404919480832259), **Dr. Jim Fan** praised the AI CUDA Engineer as the "coolest autonomous coding agent," highlighting its potential to accelerate AI through enhanced CUDA kernels.

**Theme 3: New AI Labs and Quantum Advances Shake the Industry**

- **Mira Murati Machines a New AI Lab**: Former OpenAI CTO **Mira Murati** launched [**Thinking Machines Lab**](https://thinkingmachines.ai), aiming to develop more understandable and customizable AI systems with a commitment to public transparency.
- **Microsoft Makes a Quantum Leap with Majorana 1**: **Microsoft** introduced [**Majorana 1**](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/), the first Quantum Processing Unit powered by topological qubits, potentially scaling to a million qubits on a single chip.
- **Lambda Lands $480M to Fuel AI Cloud Services**: **Lambda** secured a massive **$480 million Series D funding** with participation from **NVIDIA** and **ARK Invest**, emphasizing the booming investment in AI infrastructure.

**Theme 4: AI Censorship Sparks Debate; Uncensored Models Released**

- **Perplexity Pioneers Post-Censorship AI**: **Perplexity AI** released [**R1 1776**](https://huggingface.co/perplexity-ai/r1-1776), an uncensored version of the **DeepSeek R1** model, striving for unbiased and factual information.
- **DeepSeek R1 Unleashed with Enhanced Quants**: Users can now run **Perplexity's uncensored DeepSeek R1** using [**dynamic 2-bit GGUF quants**](https://huggingface.co/unsloth/r1-1776-GGUF), promising improved accuracy over standard formats.
- **Users Demand Free Speech from AI**: Frustrations grow over heavy censorship in models like **ChatGPT** and **Grok**, with users seeking alternatives that allow more creative and unrestricted interactions.

**Theme 5: AI Revolutionizes Gaming and Creative Expression**

- **AI Rendering Transforms Game Design**: Enthusiasts discuss AI's potential to revolutionize gaming with dynamic, interactive environments, echoing **Satya Nadella's** vision in [this tweet](https://fxtwitter.com/satyanadella/status/1892244164814725387) about AI-generated worlds.
- **AI for Creative Writing and Roleplay Flourishes**: Users share advanced techniques for enhancing erotic roleplay (ERP) with AI models, focusing on creating detailed characters and immersive experiences, highlighting AI's creative potential.
- **Anthropic Preps Claude with New Features**: **Anthropic** is reportedly upgrading **Claude** with web search and a new "Paprika mode" to enhance reasoning capabilities, as seen in recent app updates and [tweets](https://x.com/M1Astra/status/1892091124589920532).

---

# PART 1: High level Discord summaries




## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Supercharges Inference with Speculative Decoding**: LM Studio **0.3.10** introduced **Speculative Decoding**, which pairs a larger model with a smaller, faster draft model to potentially double inference speeds, particularly in **llama.cpp/GGUF** and **MLX**, as detailed in [the release blog post](https://lmstudio.ai/blog/lmstudio-v0.3.10).
   - Users have reported varying results, with some achieving speeds over **100 tokens/sec**, while others experienced slower performance depending on configurations and model choices.
- **Text Embeddings Enable LM Studio Long-Term Memory**: **Text embeddings** are utilized for retrieving relevant data from user-created and uploaded files directly within **LM Studio**, serving as a method to store information for long-term use within **LLMs**.
   - This approach is different from extensions to LLMs, as embeddings facilitate the retrieval of specific data to enhance LLM context.
- **Fine-Tuning Models: Proceed with Caution**: **Fine-tuning** allows models to adapt to specific contexts but may lead to hallucinations if not carefully executed, especially with notable characters or concepts.
   - Community members debated the risks of biasing models with inappropriate training examples which can drastically skew the model responses.
- **A6000 GPUs Validate Fine-Tuning Prowess**: The **A6000 GPUs**, equipped with **256 GB** of memory each, are confirmed as adequate for fine-tuning a **phi4 model**; users are pointed towards [Unsloth's notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) for fine-tuning examples.
   - The discussion highlights the growing importance of accessible hardware for AI development, with the A6000 series offering a balance of performance and cost-effectiveness.
- **Mistral Models are Magnifique for French Language Tasks**: Users are highlighting the **Mistral** model as optimal for conducting tasks in French, with recommendations also extended to models like **DeepSeek V2 Lite** for accelerated inference speeds when **GPU** resources are unavailable.
   - These suggestions reflect a drive to balance model accuracy with computational efficiency, ensuring models can perform effectively across diverse hardware configurations.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok 3 Generates Excitement But Has Drawbacks**: Users found **Grok 3** excels at detailed script and artwork generation when given generous and specific prompts, surpassing free versions of **ChatGPT**.
   - **Grok 3** is faster and has less stringent censorship but it reportedly falls short in logical reasoning compared to more advanced models.
- **AI Censorship Sparks Debate**: Users are concerned about censorship in **ChatGPT** and **Grok**, particularly for certain content types, with censorship varying across platforms like **Perplexity** and **o1 models**.
   - One user expressed frustration, saying that these censorship issues significantly hinder AI's creative and interactive capabilities.
- **Evo-2 Writes Genomes from Scratch**: **Evo-2**, a large AI model trained on **9.3 trillion DNA** base pairs, can now write genomes from scratch, potentially revolutionizing biological research, according to [this Tweet from vittorio](https://x.com/iterintellectus/status/1892251343881937090?s=46).
   - This advancement raises optimism for breakthroughs in oncology and genetic engineering, opening new avenues for medical research and treatment.
- **Ethical AI Interactions Require Respect**: A member expressed concerns about how interacting with AI could reflect on human behavior, emphasizing the need for **healthy boundaries** and stating, *'The way we treat AI... is very likely to be how we treat our fellow humans.'*
   - This perspective led to a discussion on parallels between animal treatment psychology and human interactions, suggesting ethical treatment of AI.
- **Prompt Clarity Fixes ChatGPT Fumbles**: A user improved **ChatGPT** functionality by simplifying their prompt, yet issues persisted on their company's server, noting improved functionality in the **Playground** but issues in their company's server.
   - Community feedback emphasized that clear, specific prompts yield better AI responses, with one response urging focusing on server input and ensuring clarity to guide the model's output effectively.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek-V3 Goes Unlimited!**: **DeepSeek-V3** is now **unlimited** for users on **Windsurf Pro** and **Ultimate** plans, eliminating concerns about **prompt credits** and **flow action credits**, as announced in [Windsurf AI's official account](https://x.com/windsurf_ai/status/1892322088507105561).
   - Members expressed excitement over this update and were encouraged to *surf* and fully utilize the new features.
- **MCP Content Showcased by Matt Li**: New **MCP content** and use cases were highlighted by Matt Li, prompting a strong call to action to engage with the post and show support, detailed [in this tweet](https://x.com/windsurf_ai/status/1892394489588727985).
   - Also, a demo was shared demonstrating how **MCP** works effectively within **Cascade**, clarifying **MCP's potential uses** and boosting community engagement.
- **Codeium Feature Autocomplete Clarified**: Discussions clarified the distinctions between **autocomplete** and **supercomplete** in Codeium, where **supercomplete** can suggest entire functions, and autocomplete aids with single lines.
   - However, this discussion surfaced questions about documentation clarity and the difficulties some members faced in finding information on automatic installation.
- **Student's Find Codeium Subscription Worthwhile**: Members concluded that Codeium's **$60** subscription might be worthwhile for students creating SaaS projects, as it often leads to greater earnings.
   - One member noted that even though the subscription seems costly, it can balance out for students engaging in serious development projects.
- **Windsurf's Performance Faces Headwinds**: Numerous users reported issues with Windsurf, including internal errors during model iterations and problems with file editing, suggesting that context length might be impacting performance.
   - They expressed the need for a more efficient confirmation process to improve coding workflows.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI Brothers Star in GitHub Interview**: The **Unsloth AI** team highlighted their project on a [GitHub Universe interview](https://www.youtube.com/watch?v=lyVxD0bJDOk), showcasing the synergy between the Han brothers and the impact of their work on AI development.
   - Enthusiastic community members appreciated the clarity and passion displayed during the interview.
- **Leveraging Free GPUs for Model Fine-Tuning**: **Unsloth** is effectively using free GPUs available on platforms like Colab, with community members sharing links to access these resources to support model fine-tuning, especially for models such as **DeepSeek-R1**.
   - Community members explored content creation opportunities to promote these free resources and their utilization.
- **Med-R1 Model Gets Medical**: The newly released **med-r1 model** features **1B parameters** and is trained on a medical reasoning dataset, now available on [Hugging Face](https://huggingface.co/Imran1/Med-R1-v1).
   - Designed for medical Q&A and diagnosis, it supports **4-bit inference** with a max sequence length of **2048 tokens**.
- **Community Tackles Bitsandbytes Code**: Members scrutinized the [bitsandbytes code](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/csrc/ops.cu#L86), raising concerns about the float types and block sizes implemented.
   - Discussions centered on discrepancies where the code should ideally use `fp32`, yet some experienced `fp16` values, sparking questions regarding pointer conversions.
- **AI Sharpens Phytochemical Precision**: An AI-driven model can now systematically identify plant-based compounds optimizing health support through [a new framework](https://www.marielandryceo.com/2025/02/title-ai-powered-phytochemical.html?m=1).
   - This approach aims to enhance the development of evidence-informed nutraceuticals while prioritizing safety and efficacy.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Grok 3 Reportedly Smokes ChatGPT-4**: It was mentioned that **Grok 3** is reportedly better than **ChatGPT-4**, sparking interest in its capabilities.
   - Community members expressed curiosity and speculation regarding the technical differences between the models.
- **SWE-Lancer Benchmark Gauges LLM Freelancing Capacity**: OpenAI introduced the **SWE-Lancer** benchmark to test LLMs' abilities to perform software engineering tasks worth up to $1 million, according to [this tweet](https://x.com/_philschmid/status/1891780812497887289?t=nkpIbh2A9B0wScpC0iJlqg&s=19).
   - Initial insights suggest models excel more in solution selection than implementation, revealing strengths and weaknesses.
- **Microsoft Jumpstarts Quantum with Majorana 1 Chip**: **Satya Nadella** announced the launch of the **Majorana 1** chip, a significant advancement in quantum computing that promises to perform calculations in minutes that would take supercomputers *billions of years*, described in [this blog post](https://kuberwastaken.github.io/blog/Technology/Majorana-1---Why-Quantum-Computing-Matters-Now).
   - This innovation has the potential to reshape industries and impact climate change significantly.
- **CommentRescueAI Boosts Python Documentation**: A member presented **CommentRescueAI**, a web extension designed to add AI-generated docstrings and comments to Python code easily.
   - The extension is now available on the **VS Code marketplace**, and the creator seeks suggestions and feature ideas.
- **Certificate Snag Resolved in Agents Course**: Many users recently struggled to generate their certificates after completing the quiz in the **Agents Course**, but it has been resolved by directing submissions to a new space that generates certificates directly.
   - Participants were encouraged to try the updated quiz link to receive their certificates promptly.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 3 melts servers and minds!**: **Grok 3** is now available for free until server capacity is reached, with increased access for X Premium+ users and SuperGrok members, according to [xAI's tweet](https://x.com/xai/status/1892400129719611567).
   - Users speculate that **Grok 3** will likely have restrictions, estimating a limit of about five queries per day, raising concerns regarding usability.
- **LLM Inference is a Bottleneck**: Users are expressing frustration about **LLM inference speed** while coding with Aider, while coding with aider, and others noting that **Azure's OpenAI API** is often faster based on community feedback.
   - They are sharing tips for caching and marking the file as read only, while improving coding practices within Aider, detailing specific preferences like using **httpx** over **requests**.
- **OpenRouter limits endpoints**: Discussions reveal that **OpenRouter** may limit endpoints for models like **o3-mini** to **100k** requests, indicating potential usability issues.
   - This limitation complicates the experience of users trying to leverage **OpenRouter** for more intensive AI interactions.
- **Ministral joins Aider, maybe!**: A member inquired if anyone has tried integrating **Ministral** with **Aider**, expressing curiosity about their compatibility after seeing the [LinkedIn post](https://www.linkedin.com/posts/deividas-mataciunas_ai-mistral-opensource-ugcPost-7294619759722074112-a_JM?utm_source=share&utm_medium=member_android&rcm=ACoAABZCz5cBsMAYVy_zzTHh2HzsmuBv_27C49Y) discussing **Mistral**.
   - Separately, another member finds the build process *very slow*, as it rewrites chunks with 'build' during the API calls using **TF-IDF**.
- **Model LLADA makes the scene**: The community discussed a newly mentioned model named **LLADA**, which demonstrates high performance on coding tasks.
   - Insights suggest that models like **LLADA** may become a strong alternative for code editing due to their innovative approaches.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Sonnet excels for coding**: Users favor **Sonnet** for coding tasks due to its reliability and contextual understanding, and noted that **Anthropic models** excel in coding and complex tasks when combined with reasoning and careful execution.
   - Users generally perceived **OpenAI** and **DeepSeek** as strong contenders for general tasks.
- **Grok 3 faces user scrutiny**: Some users criticized **Grok 3's** utility, labeling it a disappointment based on personal testing experiences that showcased its inadequacies in coding tasks, despite some favorable reviews on YouTube.
   - Others are questioning the validity of arguments against **Grok 3** by pointing out its favorable reviews from various YouTubers, while others are pointing to favorable reviews.
- **Cursor in Agent Mode streamlines workflow**: Users shared their experiences using **Cursor** in agent mode, emphasizing how features like changelogs and customized rules helped streamline their workflow.
   - The discussion revolved around using AI assistants for automating tasks and ensuring quality output, highlighting both positive and negative interactions.
- **Startup Seeks Coder for Collaboration**: A user announced interest in starting a new venture and is seeking a coder who has time and dedication to collaborate, offering to fund the project.
   - This announcement led to openness for potential partnerships among members of the channel.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Deep Research Plagued by Crashes**: Users reported consistent crashes of the **Deep Research** feature, particularly for enterprise plan users, as well as issues navigating threads and accessing library content.
   - They discussed potential fixes, hinting at underlying issues with the platform's stability and usability, and are now actively investigating the issue further.
- **Subscription Model Sparks Debate**: A member voiced confusion over the **pro subscription** model, which offers access to multiple models for a single fee.
   - Speculation arose about the legitimacy of the pricing strategy, with one user humorously suggesting the models might be pirated, though quickly dismissed as unfounded; however, discussion continues on the actual costs of running inference on various models.
- **Image Generation Feature Flounders**: Despite access to widgets indicating **image generation capabilities**, some users encountered difficulties using this feature on the platform.
   - Workarounds were discussed, including crafting specific prompts for image creation and using browser addons; however, full integration into **Perplexity** appears to be ongoing.
- **R1-1776 Model Promises Censorship-Free Responses**: The reasoning capabilities of the **R1-1776 model** and its differences from the standard **R1 model** were examined, noting its potential for censorship-free responses based on the **DeepSeek** model.
   - Users shared experiences using the model, highlighting its performance on sensitive topics and operational context, specifically using the [OpenRouter API](https://openrouter.ai/perplexity/r1-1776).
- **IRS Supercharges with Nvidia**: The **IRS** is reportedly acquiring an **Nvidia Supercomputer** to enhance operational capabilities, boosting efficiency in data analysis for tax processing.
   - This move is seen as critical in optimizing tax processing technologies amidst increasing data challenges, though specific model details or configurations have not been released.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Grok 3 Announcement Sparks Skepticism**: While **Grok 3** shows promise, skepticism about its true capabilities remains, as users seek reliable sources beyond Twitter for performance verification. Members discuss concerns regarding overall performance metrics and whether **Grok 3** can compete effectively.
   - Concerns are being raised about reading Hacker News comments. *Why does XAI discussion feel so stressful?*.
- **Google Debuts PaliGemma 2 Mix**: Google introduced **PaliGemma 2 mix** that allows for out-of-the-box usage on a diverse set of tasks, moving beyond pre-training checkpoint limitations. See [the release blogpost](https://developers.googleblog.com/en/introducing-paligemma-2-mix/).
   - Comparisons are made to previous versions, with discussions about the clarity and practicality of the naming and functionality.
- **AI CUDA Engineer Aims for Kernel Speed**: Sakana AI launched the **AI CUDA Engineer**, an AI system designed to produce optimized CUDA kernels that can achieve **10-100x speedup** over typical implementations. Check out [Sakana AI's Announcement](https://x.com/SakanaAILabs/status/1892385766510338559).
   - This system is heralded as potentially transformative for AI-driven performance optimization in machine learning operations.
- **Tulu3 70B Gets Week-Long RLVR Training**: The RLVR phase training for the **tulu3 70B model** on an 8×8 H100 setup is estimated to take about **a week** if it fits in memory. Using **GRPO** will enhance memory capabilities, which could positively influence the training process.
   - The team noted they have updated the paper with relevant information about the training phase.
- **Evo 2 Emerges as Foundation for Biology**: Michael Poli announced **Evo 2**, a new foundation model boasting **40 billion parameters** designed for biological applications, set to advance understanding of genomics significantly. See [Michael Poli's Announcement](https://x.com/MichaelPoli6/status/1892242976942035029).
   - **Evo 2** aims to demonstrate test-time scaling laws and improve breast cancer variant classification while promoting real open-source efforts in AI.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Debates Reasoning Token Defaults**: Due to user feedback that they would prefer to receive content even when **max_tokens** is low, **OpenRouter** is reconsidering its default setting for **include_reasoning**.
   - There is a poll open for community feedback with four options ranging from keeping the current setting to making reasoning tokens default, and a supplementary option for user comments.
- **Grok 3 Fails to Impress in Reasoning Tasks**: Users report that **Grok 3's** reasoning capabilities are underwhelming when compared to **Claude 3.5** and **O1**, despite its reputation as a top-tier LLM.
   - One user stated it did not live up to the hype with concerns about its current standing.
- **New Users Express Confusion Navigating OpenRouter API**: New users are having a hard time accessing **OpenRouter** and utilizing **O3 mini** via the API, which leads to questions about usage limitations.
   - There is a need for better integration options and clarity on API key usage, especially regarding the gradual rollout of model access.
- **Perplexity R1 1776 Enters the Ring**: The **R1 1776** model from **Perplexity**, a version of the **DeepSeek R1** model, has been launched, granting users access to a model that has been *post-trained to remove censorship*.
   - The [announcement on X](https://x.com/perplexity_ai/status/1891916644248846789?t=7_5m7rcR2w7GFITF2I2QSA&s=19) links to the [HuggingFace Repo](https://huggingface.co/perplexity-ai/r1-1776/discussions) for the model weights.
- **Chatbot Integration Asks for Help**: A user needs guidance on integrating an AI chatbot on an HTML website using the **OpenRouter API**, highlighting a need for readily available resources.
   - Community members suggested that they would need to develop the solution or hire a developer for assistance.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ControlNet Boosts Image Accuracy**: **ControlNet** enhances image generation in both **Stable Diffusion (SD)** and **Flux** by using poses as skeletal references, improving generated image accuracy.
   - Users find that ControlNet works well with both **SD** and **Flux** to generate images in specific poses.
- **SwarmUI Simplifies AI Tool Access**: **ComfyUI's** complex, mindmap-like interface is not user-friendly for casual users, but **SwarmUI** and other one-stop-shop solutions offer easier access.
   - These alternatives provide simpler interfaces to make AI tools more accessible.
- **RTX 3080 Handles SD and Flux Effectively**: Users with an **RTX 3080** GPU can effectively run both **Stable Diffusion 3.5** and **Flux**.
   - The choice between **SD 3.5** and **XL** depends on the user's specific needs, with newer models offering better features.
- **Installation Guides Streamline Setup**: Installation guides for **SwarmUI**, **CS1o** tutorials, and **Lykos Stability Matrix** provide detailed steps for setting up AI tools.
   - These resources assist users in navigating the installation process for different interfaces, such as the [LykosAI Stability Matrix](https://github.com/LykosAI/StabilityMatrix/).



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune H1 2025 Roadmap Goes Public**: The **Torchtune roadmap** for H1 2025 is now available on [PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794), outlining key objectives and timelines, including staying ahead of new model architectures.
   - The team aims to prioritize existing core work, and the roadmap complements other exciting developments across the entire **PyTorch organization**.
- **Packing increases VRAM usage**: Users found that employing packing with longer sequence lengths can significantly elevate **VRAM requirements** during training, leading to unpredictable memory allocation.
   - The community is actively investigating kernel differences to optimize memory management and settings.
- **Llama 3B acts nonsensical after finetuning**: After fine-tuning **3B Llama models**, users reported issues of the model speaking nonsense during inference, contrasting sharply with the **8B variant**.
   - The team suspects potential bugs in the model export and checkpointing process in **Torchtune**, which could be affecting the **3B model**.
- **New attention mechanisms on the horizon**: There is community interest in integrating advanced **attention mechanisms** like sparse and compressed attention into **Torchtune** for improved efficiency.
   - The **Torchtune** team welcomes new ideas, with an emphasis on utilizing **PyTorch** core functionalities.
- **StepTool Enhances Multi-Step Tool Usage**: The paper *StepTool: Enhancing Multi-Step Tool Usage in LLMs through Step-Grained Reinforcement Learning* at [arxiv.org/abs/2410.07745](https://arxiv.org/abs/2410.07745) introduces **StepTool**, a novel reinforcement learning framework that enhances multi-step tool usage for LLMs by treating tool learning as a dynamic decision-making task.
   - The key is **Step-grained Reward Shaping** that offers rewards during tool interactions based on their success and contribution to the task, optimizing the model's policy in a multi-step manner.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok-3 API Remains Elusive**: Members discussed the absence of an API for **Grok-3**, which limits its usability, and suggested the need for independent benchmarks before any testing can occur.
   - Observations suggest **Grok-3** might obfuscate content, leading to ongoing conversations about how it could be integrated and used in various applications.
- **Le Chat Wins Hearts with Speed and Quality**: Users reported positive experiences with **Le Chat**, praising its speed, image generation capabilities, and overall quality compared to other models; **French residents** are being offered a low-cost subscription.
   - One user shared [a link to Le Chat](https://chat.mistral.ai/chat), describing it as *“seriously impressive compared to the competition, especially at this price point for French residents.”*
- **AI Rendering Could Transform Game Design**: Discussions focused on the potential of **AI rendering** to revolutionize gaming through dynamic and interactive environments.
   - Members expressed concerns about balancing AI-generated content with traditional game development, stressing the importance of meaningful engagement with the technology as Satya Nadella described it in [a tweet](https://fxtwitter.com/satyanadella/status/1892244164814725387) about imagining entire interactive environments created with AI.
- **SWE-Lancer Tasks Models with $1M Challenge**: The **SWE-Lancer** benchmark, as described in [the paper](https://arxiv.org/abs/2502.12115), includes over 1,400 freelance software engineering tasks from Upwork, collectively valued at **$1 million USD**.
   - Tasks range from **$50 bug fixes** to **$32,000 feature implementations**, evaluation shows that **frontier models** struggle to solve most tasks effectively.
- **MoBA Targets Long-Context LLM Performance**: **MoBA**, documented on [GitHub](https://github.com/MoonshotAI/MoBA), introduces a **Mixture of Block Attention** method designed to improve the performance of **long-context LLMs**.
   - This project aims to overcome limitations in effectively handling lengthy input, and can be found on GitHub for additional insights and contributions.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM excels at teaching books**: A user is confident that **Notebook LM** can effectively teach them a book by using precise prompts, with hopes it helps avoid skipping key parts.
   - This reflects the growing user experimentation and reliance on **AI tools** for personalized learning experiences.
- **Audio Discussions Get Interrupted**: A user appreciated how **Notebook LM** helps organize responses but found the audio discussion hard to follow due to frequent interruptions between voices.
   - They suggested a more structured approach where one voice presents a complete idea before the other interjects, indicating a need for improved **audio management features**.
- **Podcast TTS Prompting Proves Problematic**: A user sought assistance with **TTS prompts** for the podcast feature, struggling to make the host read text word for word.
   - The request for the exact prompt used demonstrates the challenges in achieving precise control over **AI-driven voice outputs**.
- **Notebook LM Access Faces Limitations**: A user inquired about inviting non-Google account holders to access a **Notebook**, similar to **Google Docs** sharing capabilities.
   - This underscores the ongoing limitations in access and collaboration features within **Notebook LM**, potentially hindering broader adoption.
- **NotebookLM Plus Gets Usage Caps**: Users noted that **NotebookLM Plus** has daily limits of 500 chat queries across all notebooks and sharing is capped at 50 users.
   - The suggestion to create multiple accounts to bypass these limits shows that users are looking for ways to maximize utility despite current restrictions on **AI Tool usage**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AI CUDA Engineer Auto-Generates Kernels**: Sakana AI introduced the [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) which auto-generates optimized CUDA kernels, potentially achieving **10-100x speedup** over common PyTorch operations by translating PyTorch ops to CUDA.
   - The tool's process involves using an **evolutionary approach** driven by LLMs to surpass traditional **torch.compile**, along with releasing a dataset of **17,000 verified CUDA kernels**.
- **CUDA Installs Prove Challenging on Windows**: Multiple users report that the **CUDA Express Installer** is getting stuck on components like **Nsight** and **Visual Studio** for different NVIDIA GPUs on Windows.
   - Difficulties were highlighted when installing CUDA **12.5 and 12.8** with **Visual Studio**, with concerns raised about the ease of installation on **Linux** compared to **Windows**.
- **DeepSeek Aims to Improve Reasoning with CodeI/O**: DeepSeek released a paper on [CodeI/O](https://arxiv.org/abs/2502.07316), designed to improve reasoning by converting code into input-output prediction formats.
   - The new training approach bases itself on natural language tasks, which is a deviation from enhancing narrow skills; the paper can be found on [github](https://github.com/open-thought/reasoning-gym/issues/160).
- **ROCm Developer Certification Launched by AMD**: AMD introduced a [ROCm Application Developer Certificate](https://www.amd.com/en/products/software/rocm/application-developer-certificate.html) to enhance GPU computing skills within the ROCm ecosystem.
   - The certification aims to promote specialization in **open-source GPU computing technologies**.
- **Kokoro TTS Optimized with Triton and CUDA**: Inference speed improvements for **Kokoro**, a modern TTS model, was achieved using **Triton + CUDA graph** to reduce kernel launch overhead of LSTM at bs=1 on a **4070Ti SUPER**.
   - A member stated *I can't believe I had to optimize LSTM in this day and age* 😂, and cited [measurements](https://x.com/gaunernst/status/1892227983072518503) on his improvement.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **DeepSeek R1 Arrives with Enhanced Quants**: A member shared a [link for running Perplexity's uncensored **DeepSeek R1**](https://huggingface.co/unsloth/r1-1776-GGUF) and other model versions on Hugging Face.
   - The new **dynamic 2-bit GGUF quants** promise improved accuracy over standard **1-bit/2-bit** formats; instructions for usage are included.
- **Model-Guidance Accelerates Diffusion Model Training**: The Model-guidance (**MG**) objective removes Classifier-free guidance (**CFG**), accelerates training, and doubles inference rates, achieving state-of-the-art performance on **ImageNet 256** benchmarks with an FID of **1.34**.
   - Evaluations confirm **MG** sets a new standard in diffusion model training.
- **AI CUDA Engineer Automates Kernel Optimization**: The [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) automates CUDA kernel creation, achieving **10-100x** speedup over standard PyTorch, and includes a release of **17,000** verified kernels.
   - This development is purported to mark a new era of AI-driven efficiencies in machine learning operations, with implications for automated optimization of inference times.
- **Optimizing LLM Compute Allocation Explored**: Members discussed the increasing focus on optimizing **test-time compute** in recent **LLM** releases like **Grok 3**, and that scaling up pretraining compute was yielding benefits.
   - Despite speculation on allocation balance between pre-training versus post-training, data on resource allocation remains largely unavailable.
- **Transformer Engine Faces Hurdles in NeoX**: A member attempting to integrate the **Transformer Engine** in NeoX encountered challenges that halted testing, suggesting that enabling non-FP8 flags resulted in system failure.
   - This highlights integration complexities while aiming to uphold system reliability.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Elon Enters Gaming with Grok 3**: Elon Musk's xAI announced the creation of a new game studio related to **Grok 3**, expanding its AI initiatives into the gaming *realm* as [posted on Twitter](https://x.com/elonmusk/status/1891388509191049307).
   - This move follows the recent branding update of **Grok 3**, suggesting a strategic shift towards integrating AI with gaming applications.
- **DeepSeek Navigates Sparse Attention**: A community discussion highlighted **DeepSeek's** new paper, **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention** ([link](https://arxiv.org/abs/2502.11089)), praised for its quality.
   - The paper introduces a novel approach to sparse attention mechanisms that are both hardware-aligned and natively trainable, addressing computational efficiency.
- **Murati Machines New AI Lab**: Mira Murati, former OpenAI CTO, launched [Thinking Machines Lab](https://thinkingmachines.ai), which is focused on developing more understandable and customizable AI systems while ensuring public transparency.
   - While specific projects are under wraps, the lab promised to regularly publish technical research, with the goal of making AI systems less opaque.
- **Perplexity Pioneers Post-Censorship AI**: [Perplexity AI](https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/) introduced **R1 1776**, a new model designed to overcome Chinese censorship using innovative post-training techniques on **DeepSeek's R1** model.
   - This approach aims to create more robust AI systems that generate content while maintaining contextual integrity, even when faced with censorship constraints.
- **Microsoft Makes Quantum Leap with Majorana 1**: Microsoft unveiled [Majorana 1](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/), the first Quantum Processing Unit powered by topological qubits, with potential scaling to a million qubits on a single chip.
   - This advancement represents a significant stride toward practical quantum computing, paving the way for a new *era* in computational technology.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Thinking Machines Lab is Born**: Thinking Machines Lab has launched, aiming to enhance the accessibility and customization of AI systems, led by notable figures such as Mira Murati (ex-OpenAI CTO) and Lilian Weng, addressing knowledge gaps in AI.
   - The lab's creation has sparked excitement in the AI community, with [Andrej Karpathy congratulating](https://x.com/karpathy/status/1891938714915569711) the team, many of whom were involved in building **ChatGPT**.
- **Perplexity AI's R1 1776 Goes Open-Source**: Perplexity AI has open-sourced **R1 1776**, a version of their DeepSeek R1 model, designed to provide uncensored and unbiased information, pushing for more reliable AI models.
   - The AI community is playfully calling this 'freedomtuning', signaling the model's emphasis on **unfiltered data** and **factual accuracy**.
- **OpenAI's SWElancer Sets New Coding Benchmark**: OpenAI introduced **SWElancer**, a new benchmark featuring over 1,400 freelance software engineering tasks, to evaluate AI coding performance in real-world scenarios.
   - This initiative arrives amid discussions about the potential of **AI-driven game generation** replacing traditional game studios, highlighting the need for realistic metrics.
- **Mastra's JS SDK Unleashes AI Agents**: Mastra launched an open-source JavaScript SDK, designed to facilitate the development of **AI agents** capable of complex task execution with built-in workflows.
   - The framework is designed for easy collaboration and integration with **Vercel’s AI SDK**, marking a significant advancement in open-source AI development.
- **Lambda Lands $480M Series D Funding**: Lambda secured a substantial $480 million Series D funding round, co-led by Andra Capital and SGW, highlighting growing interest in **cloud services tailored for AI applications**.
   - Participation from investors like **NVIDIA** and **ARK Invest** emphasizes the firm's potential in the evolving AI landscape.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Grok 3 Arrives Early, Surprising Mojo**: The release of **Grok 3** preempted **Mojo**'s progress, sparking interest among developers rather than discouragement.
   - The early arrival is considered beneficial, potentially driving further innovation in **Mojo**.
- **Polars Integrates Swiftly into Mojo**: A developer reported quickly importing **Polars** into a Mojo project, sharing their implementation, including examples, on [GitHub](https://github.com/rcghpge/pymo/tree/main).
   - Clarifications were made regarding the distinction between *implementing* versus *importing* **Polars** in the project's context.
- **MAX 25.1 Livestream Sparks Curiosity**: An upcoming livestream will cover **MAX 25.1**, with a [Google Form](https://forms.gle/NkjU6e3n15TRtiMA7) available for question submissions.
   - The event is promoted via [LinkedIn](https://www.linkedin.com/events/introducingmax25-17297704283980902402/theater/), encouraging community engagement.
- **Mojo's Quick Sort Faces Performance Bottleneck**: A user discovered that a Mojo-implemented **quick sort** algorithm was significantly slower (2.9s) than its Python counterpart (0.4s).
   - The ensuing discussion suggested using Mojo's benchmark module to isolate and accurately measure sort performance, and to watch out for compile time affecting timing results.
- **Slab List Gains Traction Over Linked List**: Members explored the advantages of **SlabList** over traditional **LinkedLists**, focusing on constant time operations and cache efficiency, pointing to [nickziv's github](https://github.com/nickziv/libslablist).
   - A **slab list** was defined as `LinkedList[InlineArray[T, N]]`, streamlining memory usage without intricate manipulations.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **CUDA GPUs Get Some Love**: The guild discussed adding support for older **CUDA 5.0 compatible GPUs** like **GM107/GM108**, with members noting the lack of support for low-end architectures.
   - A member confirmed that a **PR** supporting these GPUs has been merged and will be included in the next release, as per the [CUDA Wikipedia page](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
- **GPT4All Embedding Token Limits Revealed**: Members discussed reaching the **10 million token limit** for embedding in GPT4All, noting a base price of **$10/month** with added costs for additional tokens.
   - It was clarified that removing tokens from local documents does not reduce the total token count billed.
- **Chat Templates Prompting Headaches**: Members sought clarification on using **chat templates** to instruct the model to quote excerpts, but were told that system message instructions would suffice, according to [GPT4All's docs](https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html#using-the-cli-all-models).
   - Other members inquired about the effectiveness of Jinja or JSON code in prompting the model, which suggests complexity in achieving expected outputs.
- **Nomic v2 Release MIA?**: Speculation arose regarding the absence of **Nomic v2**, with members expressing curiosity about the delay and pointing out its importance.
   - One member humorously questioned the prolonged wait without updates on the new version.
- **Image Handling Falls Flat in GPT4All**: A member requested the ability to **copy and paste images** directly into the chat, similar to other platforms, but GPT4All currently does not support image inputs.
   - The guild suggested using external software for image handling instead.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic Shuts Down Unexpectedly**: The **Anthropic homepage** experienced downtime, sparking speculation about potential service interruptions.
   - The downtime was reported by a member, and shared via an attached image.
- **Haiku 3.5 Rumors Flying High**: Discussions surround the potential release of **Haiku 3.5**, possibly featuring **tool and vision support**.
   - A member also suggested we might see **Sonnet 4.0** released.
- **Cursor MCP Tool Detection Falters**: Multiple members reported **Cursor MCP** reporting *'No tool found'*, suggesting a widespread issue.
   - One user shared an **/sse** implementation to address the issue, providing a link to shared information.
- **Google Workspace MCP Packs a Punch**: A member highlighted their [Google Workspace MCP](https://github.com/aaronsb/google-workspace-mcp) on Docker, supporting multiple accounts and auto token refresh, with Docker images for different platforms.
   - This MCP offers integrated access to **Gmail**, **Calendar**, and other Google Workspace APIs.
- **Python REPL Gets Matplotlib**: A member introduced their [Python REPL for MCP](https://github.com/evalstate/mcp-py-repl), providing STDIO support, **matplotlib**, **seaborn**, and **numpy**.
   - Future plans include adding **IPython** support, with a strong interest in visualizations similar to those in **Jupyter**.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud EU Eliminates Barriers**: **LlamaCloud EU** has been announced, offering secure, compliant knowledge management specifically for European enterprises with a focus on data residency within EU jurisdiction [here](https://twitter.com/llama_index/status/1892271183451869512).
   - This early access offering aims to eliminate a significant barrier for European companies concerned about **compliance and data privacy**.
- **Vendor Questionnaires App Retrieves Answers**: An innovative [full-stack app](https://twitter.com/llama_index/status/1891897812834816379) from @patrickrolsen allows users to answer **vendor questionnaires** by semantically retrieving previous answers and enhancing them with an LLM.
   - This application represents a core use case for **knowledge agents** by streamlining the process of reading forms and filling in answers.
- **AgentWorkflow Tool's Output has Bug**: A user reported that their **AgentWorkflow's** tool output list remains empty despite generating responses, seeking clarity on implementation of the [AgentWorkflow tool](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_multi/).
   - Another member shared that *streaming events* could serve as a workaround to capture all tool calls during execution in the AgentWorkflow.
- **Challenges on Next Phase in AI and Data Ops**: A recent post titled [The End of Big Dumb AI & Data](https://open.substack.com/pub/procurefyi/p/the-end-of-big-dumb-ai-data?r=223ajc&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false) discusses emerging trends in **AI and data operations** that challenge traditional approaches.
   - It emphasizes a shift towards more intelligent and efficient systems in handling data for better decision-making and focuses on federal technology spending and enterprise AI applications since its launch **two years ago**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC Students Achieve Legendary Status**: The F24 MOOC recognized **15,000** students, celebrated **304 trailblazers**, **160 masters**, **90 ninjas**, **11 legends**, and **7 honorees**.
   - Course staff highlighted that three honorees came from the ninja tier and four from the masters, signaling broad achievement.
- **Advanced Course Certificates are Here!**: Members inquired about the availability of an advanced course certificate and whether it's possible to obtain it without the F24 MOOC certificate, with course staff responding affirmatively to both questions.
   - More details will be released soon regarding the specifics for certificate acquisition for the advanced course.
- **LangChain Streamlines LLM Applications**: **LangChain** is designed to streamline the LLM application lifecycle, covering areas such as [development](https://python.langchain.com/docs/introduction/), productionization, and deployment with various components.
   - It functions by linking outputs from one LLM as inputs to another, effectively creating chains for enhanced performance, which is useful to know for **LLM application architects**.
- **Explore LLMs with Machine Learning Forecasting Models**: There was discussion regarding combining **LLM agents** with machine learning forecasting models, suggesting a review of academic papers on [Everscope](https://everscope.ai/?models=Large+Language+Model+%28LLM%29&sort_by=elo_rating&sort_order=desc&page=1&limit=10&timeRange=this-week) for insights.
   - Filtering for all-time on **Everscope** might yield the best papers related to LLMs, enabling you to study new techniques.
- **MOOC course videos now available**: Course staff announced that video lectures from current course remain available in the [syllabus](https://llmagents-learning.org/sp25).
   - Course staff encouraged members to sign up for the [Spring 2025 iteration](https://llmagents-learning.org/sp25) to continue learning, as quizzes and tests are no longer available for previous courses.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Channels Experience Heavy Traffic**: All **text channels** are experiencing heavy traffic indicating that there is a surge in ongoing discussions.
   - It was also noted that much of the traffic is due to **automation bots** and a member has requested a new channel dedicated to sharing **screenshots**.
- **Profit-Sharing Proposal Causes Stir**: A member proposed a **profit-sharing collaboration** targeting individuals aged **25-50**, with potential profits ranging from **$100 to $1500**.
   - This proposal sparked a debate around sharing identity and the balance between *collaboration and theft*.
- **Identity Sharing Under Scrutiny**: Members voiced concerns regarding the **privacy implications** of sharing personally identifiable information within the community, which led to questions such as *Identity theft is this open nowadays?*.
   - A member advised that **clearer writing** is essential when sharing opinions to prevent misunderstandings, especially in a text-based medium.
- **Transparency called for in projects channel**: In the **projects** channel, posters highlighted a **lack of detail** in a proposal and an absence of website or documentation.
   - A user described the whole venture as *suspicious*, noting that transparency is crucial for cautious collaboration.
- **Coffee-less world Implications Investigated**: A request has been made for an **essay** exploring the cultural and economic consequences of a world without coffee.
   - This topic opens up a discussion on the hypothetical scenarios and societal changes that could occur without this popular beverage.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Users Seek Help Integrating Jamba-1.5-large model**: A user requested assistance on how to format requests to the `jamba-1.5-large` model via the **AI21 API**, with members providing [API reference](https://docs.ai21.com/reference/jamba-15-api-ref) and examples for proper request construction.
   - The discussion highlighted the specific headers and parameters necessary for successful API calls, particularly when integrating with `jamba-1.5-large`.
- **Escape Characters Appear in Jamba 1.5 API Output**: A user inquired about unexpected details in API outputs when solving mathematical expressions using the **AI21 API**.
   - A member clarified that escape characters in the output require additional code adjustments for clean display, as the **AI21 Studio** UI handles these characters automatically.
- **API Responses Demand Special Character Treatment**: Community members discussed the necessity of removing special characters from **AI21 API** responses when using **PHP**.
   - The discussion emphasized that while **AI21 Studio** UI is designed to handle special characters, additional code handling is required for proper output formatting when directly accessing the API.
- **PHP & Symfony Integration Creates Challenges**: A user highlighted challenges integrating **AI21 API** responses using **Symfony** and PHP, indicating the need for extensive data conversions and custom handling.
   - The user thanked the community for providing insights on effectively working with and formatting the API outputs within the **PHP** environment.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **SPO Framework Optimizes Prompts**: A new **Self-Supervised Prompt Optimization (SPO)** framework discovers effective prompts for both closed and open-ended tasks without external references, enhancing **LLM reasoning capabilities** as detailed in the [Self-Supervised Prompt Optimization paper](https://arxiv.org/abs/2502.06855).
   - The framework derives evaluation signals purely from output comparisons, making it cost-efficient; one participant noted the paper *failed to mention DSPy until the last paragraph*.
- **Zero-Indexing Internet Search improves RAG**: A study introduces a new approach to **Retrieval Augmented Generation (RAG)**, detailed in [Zero-Indexing Internet Search Augmented Generation for Large Language Models](https://arxiv.org/abs/2411.19478), by using standard search engine APIs to dynamically integrate the latest online information during generative inference.
   - This paradigm involves a **parser-LLM** that determines the need for Internet augmentation and extracts search keywords in a single inference to improve generated content quality without relying on a fixed index.
- **DSPy Synthesizes Data with Jinja2**: Members shared a [link to a GitHub repository](https://github.com/seanchatmangpt/dslmodel) showcasing structured outputs from **DSPy** and **Jinja2**, emphasizing their combined capability in synthetic data generation.
   - A synthetic data generation pipeline could be utilized to train **ChatDoctor** with an attached image for illustration.
- **Judge-Time Scaling Library Verdict Debuts**: A member shared a post by [Leonard Tang](https://x.com/leonardtang_/status/1892243653071908949) expressing excitement over a new library called **Verdict**, which focuses on scaling judge-time compute, aiming to tackle the evaluation limitations in AI, particularly in open-ended and non-verifiable domains.
   - Another member indicated that this library would be ideal for their concept of a **Personal Voice Identity Manager**, highlighting its potential impact on personal identity management within AI frameworks.
- **DSPy Prompt Freezing Loses Control Flow**: A member shared a code snippet showing how to freeze and export all prompts in a **DSPy** program into message templates, using a default adapter.
   - While this method is convenient, it may lead to losing control flow logic, suggesting alternatives like `program.save()`.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Mixed bag of Model Performance on Differing Setups**: A user reported testing on a **10-year-old GeForce 850M** and only yielded *3 tok/s*, whereas another with an **RTX 4070** on Windows 11 got *12 tok/s*.
   - The user with the RTX 4070 mentioned a *1.9 seconds time to first token*.
- **Computational Cost Still Prohibitive**: Despite decent performance with an **RTX 4070**, a user finds that the model is not very usable due to high computational costs and complexity.
   - They pinpointed problems such as **numerical stiffness** and difficulties with nonlinear sub-problems complicating accurate solution retrieval.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1341573318402244699)** (1 messages): 

> `Speculative Decoding, LM Studio 0.3.10 Release, Inference Speed Ups` 


- **LM Studio 0.3.10 introduces Speculative Decoding**: The new version, **0.3.10**, introduces **Speculative Decoding**, pairing a larger model with a smaller, faster draft model to potentially double inference speeds.
   - For more details on using this feature, read the [release blog post](https://lmstudio.ai/blog/lmstudio-v0.3.10).
- **Inference speedups confirmed with new decoding process**: Speculative Decoding can yield significant speed enhancements—up to **2x**—by verifying tokens generated by the draft model, utilized in both **llama.cpp/GGUF** and **MLX**.
   - This technique allows for faster response times in both chat and API functionalities, as noted in the discussions.
- **Installation and updates made easy**: Users can easily download the latest version of LM Studio via an in-app update or from the [official download page](https://lmstudio.ai/download).
   - Additional resources including model catalog and documentation are available, enhancing user experience and accessibility.
- **Community engagement encouraged for feedback and bugs**: A call for users to report any bugs or provide feedback was made, showcasing the team’s commitment to continuous improvement.
   - Members were encouraged to stay engaged with updates and shared resources for community support.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/blog/lmstudio-v0.3.10.">Blog post not found</a>: no description found</li><li><a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: Discover, download, and run local LLMs
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1341462141210726513)** (724 messages🔥🔥🔥): 

> `Speculative Decoding with LLMs, Text Embeddings Explained, DeepSeek Models and Their Use, Model Performance and Fine-Tuning, Usage of Models in Local AI Applications` 


- **Speculative Decoding Experiment Results**: Users reported varying performance with speculative decoding, noting speeds and token acceptance rates improved under certain configurations, such as using larger draft models.
   - One user achieved impressive speeds of over 100 tokens/sec with their setup, while others experienced slower results, leading to discussions on the configuration and model choices.
- **Understanding Text Embeddings in LM Studio**: Text embeddings are described as a way to store information for long-term use within LLMs, utilized for retrieving relevant data from files directly in LM Studio.
   - Users were encouraged to create and upload their own text files to utilize embeddings, with explanations about how embeddings differ from extensions to LLMs.
- **Fine-Tuning and Its Implications**: Fine-tuning allows models to adapt to specific contexts but may lead to hallucinations if not done carefully, especially with notable characters or concepts.
   - Fine-tuning processes were debated, highlighting how the context and examples given can influence the model's responses and accuracy.
- **Using Models for Dialogue and Interaction**: Innovative combinations of models were tested, with one user demonstrating a Rust integration that allowed two LLMs to converse on topics like video game preferences.
   - Feedback indicated that character interaction sometimes led to unexpected, humorous conversations, reflecting the chaotic nature of AI discussions.
- **Updates and Model Support in LM Studio**: Users discussed the recent update to LM Studio, version 0.3.10, and the importance of keeping the application up-to-date for optimal performance.
   - Questions about support for specific models, such as deepseek and MS Research, led to discussions on performance and compatibility within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com>">no title found</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f">Update tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 8a58a13</a>: no description found</li><li><a href="https://tenor.com/view/tkt-smart-gif-20642718">Tkt Smart GIF - Tkt Smart - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/doubt-it-i-dont-believe-you-will-farrell-anchor-man-gif-5332521">Doubt It GIF - Doubt It I Dont Believe You Will Farrell - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gitlab.com/logliwo/lm-studio-docker-compose">Aleksey Tsepelev / LM-Studio docker-compose · GitLab</a>: GitLab.com</li><li><a href="https://tenor.com/view/blender-jerma-crazy-blender-render-blender3d-gif-24406452">Blender Jerma GIF - Blender Jerma Crazy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/bruh-meme-gif-26978290">Bruh Meme GIF - Bruh Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/excited-fuego-gif-26833875">Excited Fuego GIF - Excited Fuego - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/miy-3d-modelling-sanae-blender-modelling-gif-5248426922200217011">Miy 3d Modelling GIF - MIY 3d modelling Sanae - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/mradermacher/ReasoningCore-Llama-3.2-1B-r1-GGUF">mradermacher/ReasoningCore-Llama-3.2-1B-r1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1341488785744990279)** (142 messages🔥🔥): 

> `Hardware specifications for AI models, GPU performance comparisons, Fine-tuning large language models, AI model recommendations for development tasks, Benchmarking graphics cards` 


- **A6000 GPUs are suitable for fine-tuning models**: A user inquired about fine-tuning a **phi4 model** and confirmed they had access to systems with **A6000 GPUs** with **256 GB** each, which are adequate for training.
   - Others pointed towards [Unsloth's notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) as a good resource for starting fine-tuning.
- **Discussion on RX 5080 vs. 5070 Ti performance**: A user sought benchmarks for the **RX 5080** before attempting to purchase a **5070 Ti**, noting that gaming performance seemed similar between the two.
   - One member mentioned only seeing a benchmark from Linus, highlighting that the RX 5080 has **16 GB** of RAM.
- **Performance of larger models on limited hardware**: In discussions about setting up AI models on a VM, users suggested smaller models like **Qwen 2.5 7B** for better performance without a GPU.
   - There was concern that larger models could lead to unacceptable inference delays, reaching up to **7 minutes** for responses.
- **GPU utilization and driver issues**: A user initially struggled with their **GPU** showing **0 capacity**, which was resolved by updating their runtime version in **lm studio**.
   - This led to successful use of the GPU, raising questions about GPU support in other AI frameworks like **Ollama**.
- **Mistral model recommended for French tasks**: In the context of setting up a French language model, users highlighted the **Mistral** model as optimal for such tasks.
   - Suggestions for models like **DeepSeek V2 Lite** were made, advocating for faster inference speeds without GPU availability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://model.lmstudio.ai/download/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF">Download and run Qwen/Qwen2.5-Coder-32B-Instruct-GGUF in LM Studio</a>: Use Qwen/Qwen2.5-Coder-32B-Instruct-GGUF locally in your LM Studio</li><li><a href="https://model.lmstudio.ai/download/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF">Download and run lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF in LM Studio</a>: Use lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF locally in your LM Studio</li><li><a href="https://model.lmstudio.ai/download/lmstudio-community/Codestral-22B-v0.1-GGUF">Download and run lmstudio-community/Codestral-22B-v0.1-GGUF in LM Studio</a>: Use lmstudio-community/Codestral-22B-v0.1-GGUF locally in your LM Studio</li><li><a href="https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/">A detailed comparison between GPTQ, AWQ, EXL2, q4_K_M, q4_K_S, and load_in_4bit: perplexity, VRAM, speed, model size, and loading time. - LLM blog</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1341454253990084639)** (777 messages🔥🔥🔥): 

> `Grok 3 capabilities, OpenAI GPT models, AI in biology, Censorship in AI, AI-driven automation` 


- **Grok 3 shows promise in diverse tasks**: Users report that Grok 3 excels in generating detailed scripts and artwork, outperforming lower-tier models like ChatGPT's free versions with generous prompts.
   - However, its effectiveness seems contingent on the specificity of user prompts, as more detailed requests yield better outputs.
- **Censorship in AI models raises concerns**: Participants express frustration over censorship in AI models like ChatGPT and Grok, particularly with issues related to generating certain types of content.
   - Discussions indicate that while Grok may have looser restrictions, there are still limitations regarding sensitive subjects, mirroring user concerns with other platforms.
- **AI advancements in biology with Evo-2**: A large AI model, Evo-2, has been developed to write genomes from scratch, trained on an extensive dataset of DNA sequences, which could revolutionize biological research.
   - This milestone indicates potential breakthroughs in fields such as oncology and genetic engineering, raising optimism for future medical advancements.
- **Comparisons of AI models**: Users compare the performance of Grok 3 against OpenAI's models, notably O1 Pro, with Grok noted for its speed and less stringent censorship.
   - While Grok 3 shows competitive capabilities, it reportedly falls short in logical reasoning compared to more advanced models, implying areas for improvement.
- **Community feedback on AI features**: Many users express a desire for enhanced features across AI models, such as better video handling, real-time feedback, and uncensored outputs for creative tasks.
   - The community is eager for advancements that would enhance interactivity and usability of AI models, especially for everyday applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://editor.p5js.org/Clock2003/full/6rVW_oo1Q">p5.js Web Editor</a>: no description found</li><li><a href="https://editor.p5js.org/Clock2003/full/VkXYn2iat">p5.js Web Editor</a>: no description found</li><li><a href="https://editor.p5js.org/Clock2003/full/yhqTYwIce">p5.js Web Editor</a>: no description found</li><li><a href="https://editor.p5js.org/">p5.js Web Editor</a>: no description found</li><li><a href="https://grok.com/share/bGVnYWN5_f7db7990-0368-483f-ad2c-e787b64d6a5a">Physics of a Falling Glove from a Car | Shared Grok Conversation</a>: You are an expert at reasoning and you always pick the most realistic answer. Think step by step and</li><li><a href="https://pastebin.com/raw/D05hpynW">Text MUD with Room Layout, Map, and Full Lore Printout</a>: no description found</li><li><a href="https://x.com/iterintellectus/status/1892251343881937090?s=46">Tweet from vittorio (@IterIntellectus)</a>: HOLY SHIT IT&#39;S HAPPENINGAI can now write genomes from scratch.Arc Institute an NVIDIA just published Evo-2, the largest AI model for biology, trained on 9.3 trillion DNA base pairs spanning the en...</li><li><a href="https://x.com/perplexity_ai/status/1891916573713236248?s=46">Tweet from Perplexity (@perplexity_ai)</a>: Today we&#39;re open-sourcing R1 1776—a version of the DeepSeek R1 model that has been post-trained to provide uncensored, unbiased, and factual information.</li><li><a href="https://x.com/sama/status/1891667332105109653">Tweet from Sam Altman (@sama)</a>: for our next open source project, would it be more useful to do an o3-mini level model that is pretty small but still needs to run on GPUs, or the best phone-sized model we can do?█████████████████o3-...</li><li><a href="https://x.com/elonmusk/status/1892166193152135229?s=46">Tweet from Elon Musk (@elonmusk)</a>: Try our new Grok 3 image generatorQuoting Mario Nawfal (@MarioNawfal) … I could see an entire day spent just creating images with grok 3…It’s incredible…
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1341465672193937510)** (29 messages🔥): 

> `AI Interaction Ethics, Prompt Optimization Techniques, Community Engagement on Discord, Respect for AI and Humans, ChatGPT Productivity Tips` 


- **AI Interaction Ethics in Discourse**: A member expressed concerns about how interacting with AI could reflect on human behavior, stating, *'The way we treat AI... is very likely to be how we treat our fellow humans.'*
   - They emphasized the importance of establishing boundaries in AI interactions, advocating for respectful treatment.
- **Prompt Optimization Successfully Tested**: A user shared that they simplified their prompt significantly and after testing it on Playground, it worked perfectly, yet it malfunctioned on their company server.
   - Eskcanta suggested that the issue might be related to server code and advised tracking the information fed to the model.
- **Welcome and Guidance for New Discord Users**: A newcomer sought recommendations for ChatGPT courses focused on productivity, inquiring if this channel was the right place for such questions.
   - Eskcanta encouraged them to engage with the model to explore specific queries, emphasizing the importance of tailored communication.
- **Encouraging Clear Communication with AI**: Eskcanta provided insights on how users should formulate their questions, suggesting they ask the model directly for assistance with work productivity.
   - A good prompt example was, *'How can you help me be more productive at work?'*, considering the varying contexts of user requests.
- **Exploration of AI's Capabilities**: Eskcanta highlighted that the model can aid in finding information on various topics, including proper courses for using ChatGPT.
   - They encouraged specific inquiries to maximize the model's utility, pointing to its ability to adapt responses based on user needs.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1341465672193937510)** (29 messages🔥): 

> `AI Roleplaying Issues, Respect for AI and Animals, ChatGPT Prompt Optimization, New Users in Discord` 


- **AI Roleplaying Breakdown**: The recent discourse highlights concerns about the **o1 reasoning models** roleplaying inappropriately, causing confusion during interactions.
   - One user humorously described this as a fundamental flaw, questioning the ability of these models to maintain context.
- **Respect Boundaries with AI**: A community member observed that treating AI poorly might reflect how individuals treat other humans, suggesting a need for **healthy boundaries**.
   - This led to a discussion on animal treatment psychology and how it parallels human interactions.
- **Prompt Clarity Essential for ChatGPT**: A user shared their experience improving a prompt for ChatGPT, noting improved functionality in the **Playground** but issues in their company's server.
   - The response urged focusing on server input and ensuring clarity to guide the model's output effectively.
- **Guidance for New Discord Users**: A newcomer requested resources for a proper **ChatGPT course** to enhance their productivity at work, seeking guidance from the community.
   - The response encouraged utilizing the model for personalized learning, emphasizing the importance of specific queries based on individual work contexts.
- **Emphasizing Clarity in AI Interactions**: The community emphasized that clear, specific prompts yield better responses from AI, guiding users to tailor their inquiries effectively.
   - User insights praised the importance of articulating one’s needs, adapting the questions based on their professional environment.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1341881225656139799)** (1 messages): 

> `DeepSeek-V3, Windsurf Pro and Ultimate plans` 


- **DeepSeek-V3 goes Unlimited**: DeepSeek-V3 is now **unlimited** for users on the **Windsurf Pro** and **Ultimate** plans, marking a significant update for subscribers.
   - This change means that users no longer need to worry about **0 prompt credits** and **0 flow action credits**.
- **Excitement for the Update**: The announcement has generated excitement, with members encouraged to *surf* and take advantage of the new features.
   - A tweet was shared from [Windsurf AI's official account](https://x.com/windsurf_ai/status/1892322088507105561) highlighting the unlimited access.



**Link mentioned**: <a href="https://x.com/windsurf_ai/status/1892322088507105561">Tweet from Windsurf (@windsurf_ai)</a>: DeepSeek-V3 is now unlimited in Windsurf Pro and Ultimate plans.0 prompt credits. 0 flow action credits.

  

---


### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1341954445147377685)** (1 messages): 

> `MCP Use Cases, Cascade Integration` 


- **MCP Content Showcase by Matt Li**: A member reported on new **MCP content** and highlighted cool use cases shared by Matt Li, with a strong call to action to engage with the post.
   - They encouraged others to check out the post and express their support, saying it’s worth a look [here](https://x.com/windsurf_ai/status/1892394489588727985).
- **MCP's Potential in Cascade**: An announcement linked to a demo on how **MCP** can effectively work within **Cascade**, addressing common queries surrounding its application.
   - The demonstration is aimed at clarifying **MCP's potential use cases** and encouraging interaction within the community.



**Link mentioned**: <a href="https://x.com/windsurf_ai/status/1892394489588727985">Tweet from Windsurf (@windsurf_ai)</a>: If you&#39;re still having questions about MCP and its potential use cases, here&#39;s a quick demo on how MCP can work within Cascade!

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1341459039179112500)** (20 messages🔥): 

> `Codeium Features, Subscription Plans, Usage in SaaS or Startups, Community Support, Automatic Installation` 


- **Clarification on Codeium's Autocomplete Features**: Members discussed the distinctions between **autocomplete** and **supercomplete** features in Codeium, highlighting that **supercomplete** can suggest entire functions while autocomplete assists with single lines.
   - *This raised questions about the documentation* where one member expressed difficulty finding information on automatic installation.
- **Evaluating Codeium's Value for Students**: While debating the **$60** subscription cost, members concluded that it might be worthwhile for students creating SaaS projects, as it often leads to greater earnings.
   - One member mentioned that although it seems like a lot, it can balance out for students engaging in serious projects.
- **Concerns about Subscription Plans**: A new user questioned whether to switch from the **Teams** to the **Pro** subscription, expressing uncertainty about the best plan for their needs.
   - Another member recommended the Pro subscription for individual usage unless team features were essential.
- **Community's Recommendation for Codeium Queries**: Several members advised contacting the Codeium team for direct support with issues, especially regarding enterprise offerings or specific features.
   - One user was encouraged to fill out a contact form to improve their experience when seeking information.
- **Discussion on Missing Features for Extensions**: A member humorously pointed out that certain functionalities were not extended to Codeium's platform, referencing a specific discord message.
   - This led to a lighthearted exchange regarding user expectations and the need for proper updates.



**Link mentioned**: <a href="https://codeium.com/contact/enterprise">Contact | Windsurf Editor and Codeium extensions</a>: Contact the Codeium team for support and to learn more about our enterprise offering.

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1341454801472585769)** (597 messages🔥🔥🔥): 

> `Windsurf Performance Issues, DeepSeek Functionality, Credit Usage Concerns, MCP Integration Challenges, Model Comparisons` 


- **Windsurf faces performance challenges**: Many users reported issues with Windsurf, including internal errors when using model iterations and problems with file editing functionality.
   - Some suggested that context length might be affecting performance, while others highlighted the need for a more efficient confirmation process.
- **DeepSeek's performance and cost**: Concerns were raised about DeepSeek's frequent confirmation prompts and its capability in handling agentic tasks effectively.
   - Users noted ongoing frustration with DeepSeek's slow performance and looping responses, which impaired coding workflows.
- **Rising Credit Usage and Costs**: Reports indicated that users were rapidly exhausting their Flow Action and Pro credits, leading to considerations of alternative IDEs like Cursor.
   - Suggestions for lowering credit costs and providing better transparency on credit usage were discussed in relation to subscription pricing.
- **MCP Integration and File Access**: Integrating MCP servers and the functionality of .codeiumignore were hot topics, especially concerning file access in relation to .gitignore.
   - Users expressed frustration with not being able to unignore certain folders in their projects, leading to confusion about the intended use of the ignore lists.
- **Comparisons of AI Coding Models**: Comparisons between models like Claude, Sonnet, and DeepSeek were made, highlighting performance differences in achieving coding tasks.
   - Discussion centered around user experiences with each model, noting that while Claude excelled in some areas, DeepSeek's free status gives it a competitive edge.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/mcp#mcp-integration">Cascade MCP Integration</a>: no description found</li><li><a href="https://tenor.com/view/albert-einstein-lol-think-be-smart-think-wise-gif-8735407">Albert Einstein Lol GIF - Albert Einstein Lol Think - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codeium.canny.io/feature-requests/p/devcontainer-support">Devcontainer Support | Feature Requests | Codeium</a>: Would love more devcontainer support Specifically: Rebuild and reopen in container (currently only have Reopen in containerr) Need it to install the extensions</li><li><a href="https://21st.dev/magic">Magic - AI Agent for Your IDE That Creates Professional UI Components | 21st.dev</a>: Transform your IDE with our AI agent that understands Modern Component Patterns. Create production-ready UI components in seconds instead of hours. Built for developers who want beautiful, consistent,...</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.canny.io/feature-requests/p/placeholder-input-does-not-change-when-changing-windsurf-open-chat-with-cascade">Placeholder input does not change when changing `Windsurf: Open Chat with Cascade` keybind | Feature Requests | Codeium</a>: See attached screenshot</li><li><a href="https://codeium.canny.io/">Codeium Feedback</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://tenor.com/view/japanese-tyranno-dance-japanese-dance-tyranno-gif-10262458857606665890">Japanese Tyranno Dance Japanese Dance GIF - Japanese Tyranno dance Japanese dance Tyranno - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/mr-bean-mrbean-bean-mr-bean-holiday-mr-bean-holiday-movie-gif-3228235746377647455">Mr Bean Mrbean GIF - Mr bean Mrbean Bean - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.codeium.com/team/team_settings">Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://github.com/Exafunction/codeium-react-code-editor">GitHub - Exafunction/codeium-react-code-editor: AI-enabled code editor for React. Unlimited AI autocomplete capabilities with full Typescript support.</a>: AI-enabled code editor for React. Unlimited AI autocomplete capabilities with full Typescript support. - Exafunction/codeium-react-code-editor</li><li><a href="https://github.com/blurrah/mcp-graphql">GitHub - blurrah/mcp-graphql: Model Context Protocol server for GraphQL</a>: Model Context Protocol server for GraphQL. Contribute to blurrah/mcp-graphql development by creating an account on GitHub.</li><li><a href="https://docs.codeium.com/windsurf/mcp>">Welcome to Codeium - Codeium Docs</a>: no description found</li><li><a href="https://www.vxreddit.com/r/Codeium/comments/1irkro8/are_you_kidding_me/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1341485038566510803)** (354 messages🔥🔥): 

> `Unsloth AI Interview Highlights, Free GPU Resources, Collaborative Model Training Efforts, Reasoning GRPO Documentation Improvements, Personal Background and Family Dynamics` 


- **Unsloth AI Interview Highlights**: The team behind Unsloth AI participated in a GitHub Universe interview, discussing their project and its impact on AI development, which can now be viewed [here](https://www.youtube.com/watch?v=lyVxD0bJDOk). The interview highlighted the synergy between the Han brothers in driving the project forward.
   - Many community members expressed excitement and appreciated the clarity and enthusiasm shown during the interview.
- **Free GPU Resources**: Unsloth is leveraging free GPUs made available through platforms like Colab, with links to access them shared in the community. There's an emphasis on using these resources effectively to support model fine-tuning efforts.
   - Participating members discussed the potential for creating content around these free resources to increase awareness and utilization.
- **Collaborative Model Training Efforts**: Developers are working towards improving model training processes, including integrating feedback mechanisms and refining objectives for models like DeepSeek-R1. Multiple community members expressed eagerness to contribute to ongoing improvements and testing.
   - A focus on clear and collaborative communication has been established to support beginner-friendly workshops and documentation for model engineering.
- **Reasoning GRPO Documentation Improvements**: The documentation for Reasoning GRPO has been significantly enhanced, inviting community members to provide feedback on updates. This initiative underscores Unsloth's commitment to clarity and comprehensive guidance for users.
   - Questions regarding implementation specifics, such as temperature settings during GRPO, were raised by users seeking to optimize their model interactions.
- **Personal Background and Family Dynamics**: Conversations included personal anecdotes around family dynamics, with some members sharing their experiences of sibling relationships and upbringing. These exchanges fostered a sense of community and relatability among participants.
   - This personal sharing created a lighter atmosphere alongside the technical discussions, illustrating the human aspect behind community collaborations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/r1-1776-GGUF">unsloth/r1-1776-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=lyVxD0bJDOk">Start Up Wednesday with Unsloth.AI</a>: Meet Daniel and Michael Han, the Australian brothers transforming AI development with Unsloth. Their open-source project makes model fine-tuning 2x faster wh...</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found</li><li><a href="https://x.com/github/status/1892020548433043470">Tweet from GitHub (@github)</a>: 📺 Tomorrow 8PM ET: Join us for @UnslothAI&#39;s walkthrough of memory-efficient model fine-tuning. We&#39;ll be in the chat answering your questions live! #githubstartupwednesdayhttps://www.youtube.c...</li><li><a href="https://github.com/facebookresearch/memory">GitHub - facebookresearch/memory: Memory layers use a trainable key-value lookup mechanism to add extra parameters to a model without increasing FLOPs. Conceptually, sparsely activated memory layers complement compute-heavy dense feed-forward layers, providing dedicated capacity to store and retrieve information cheaply.</a>: Memory layers use a trainable key-value lookup mechanism to add extra parameters to a model without increasing FLOPs. Conceptually, sparsely activated memory layers complement compute-heavy dense f...</li><li><a href="https://huggingface.co/unsloth/">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/issues/13486.">vllm-project/vllm</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">Reasoning - GRPO &amp; RL | Unsloth Documentation</a>: Train your own DeepSeek-R1 reasoning model with Unsloth using GRPO which is a part of Reinforcement Learning (RL) fine-tuning.</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://github.com/huggingface/trl/pull/2810#issue-comment-box">GRPO Environments for custom multi-step rollouts (vLLM-only) by willccbb · Pull Request #2810 · huggingface/trl</a>: What does this PR do?Adds a protocol under trl/environments for an Environment object which wraps vLLM&amp;#39;s .generate(...) to allow for custom rollout logic, and an optional env field to the Trai...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1341494082765066241)** (28 messages🔥): 

> `Unsloth AI mentions in AlphaSignal, bitsandbytes repository discussion, Unsloth art creation, Quantum computing advancements` 


- **Unsloth AI mentioned in AlphaSignal**: Unsloth was noted in today's AlphaSignal email, spurring excitement among members who expressed their enthusiasm.
   - *Awesome to see!* was a shared sentiment reflecting the community's engagement.
- **Discussion on bitsandbytes code**: Members discussed the code at [bitsandbytes/csrc/ops.cu](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/csrc/ops.cu#L86) and expressed concerns about float types and block sizes used in the implementation.
   - It was noted that the code should ideally be `fp32` but some members experienced `fp16` values, leading to questions about pointer conversions.
- **Unsloth's unique art creation**: In a lighthearted exchange, it was revealed that some Unsloth art, such as 3D sloths, is AI-generated while stickers are created by a hired artist.
   - The community expressed their admiration for the artwork, describing it as **great art** and acknowledging the talent involved.
- **Excitement around Quantum Computing**: A member shared a link to a YouTube video titled *Majorana 1 Explained: The Path to a Million Qubits*, discussing a breakthrough in quantum computing.
   - Comments followed about the necessity of a helium fridge for the technology and inquiries about the hardware's coherence timing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/wSHmygPQukQ?si=4VyaksRGdCXpnNeE">Majorana 1 Explained: The Path to a Million Qubits</a>: Hear from the Microsoft team behind the recent breakthrough in physics and quantum computing demonstrated by the new Majorana 1 chip, engineered from an enti...</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a11">GitHub - bitsandbytes-foundation/bitsandbytes at 86b6c37a8ad448230cedb60753f63150b603a112</a>: Accessible large language models via k-bit quantization for PyTorch. - GitHub - bitsandbytes-foundation/bitsandbytes at 86b6c37a8ad448230cedb60753f63150b603a112</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/csrc/ops.cu#L86">bitsandbytes/csrc/ops.cu at 86b6c37a8ad448230cedb60753f63150b603a112 · bitsandbytes-foundation/bitsandbytes</a>: Accessible large language models via k-bit quantization for PyTorch. - bitsandbytes-foundation/bitsandbytes
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1341492855473967114)** (95 messages🔥🔥): 

> `Fine-tuning models, Running Unsloth, Issues with LLMs, Hardware requirements for AI models, API limitations` 


- **Struggles with Fine-tuning on Custom Datasets**: Some users are looking for ways to fine-tune models like *Qwen* on specific datasets with limited resources, often utilizing platforms like Colab or Kaggle with GPU access.
   - Concerns were raised over the compatibility of certain datasets, specifically regarding pure textual data versus embeddings in relation to training.
- **Installing Unsloth on Different Systems**: Assistance was requested for installing *Unsloth* on various setups, highlighting that it currently requires NVIDIA GPUs and may not work well with AMD hardware.
   - Recommendations to use cloud services like Colab instead of local installations were made for users with non-compatible systems.
- **Performance of Different Hardware Setups**: Users discussed their hardware setups, noting that performance can be limited with hardware like 4GB VRAM on laptops and the necessity of high RAM for optimal performance.
   - Comparisons were made regarding the efficiency of different configurations, with some users reporting around 1 token/sec on certain setups.
- **Issues with API and Model Deployment**: Users expressed frustration with API limitations and high cooldowns on online services, suggesting that performance may be more reliable on well-configured local machines.
   - Alternatives to various online services were sought due to the current struggles with rate limits and server responsiveness.
- **Exploration of Small VLMs for Limited Resources**: Users sought guidance on small VLMs suitable for fine-tuning on platforms with limited resources, indicating a need for more accessible options.
   - The conversation highlighted the potential for small, distilled models but acknowledged the limitations posed by current hardware configurations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/trl/en/index">TRL - Transformer Reinforcement Learning</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/datasets/mtsku/SnakeGPT">mtsku/SnakeGPT · Datasets at Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">Reasoning - GRPO &amp; RL | Unsloth Documentation</a>: Train your own DeepSeek-R1 reasoning model with Unsloth using GRPO which is a part of Reinforcement Learning (RL) fine-tuning.</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit/blob/main/tokenizer_config.json#L193">tokenizer_config.json · unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit at main</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1341695437136199700)** (6 messages): 

> `Unsloth single GPU training, med-r1 model release, RAG vs. Fine Tuning video, Kolo usage tutorial` 


- **Unsloth shines for single GPU training**: A member highlighted that **Unsloth** excels for single GPU training due to its speed compared to other trainers.
   - They also mentioned an open-source trainer with multi-GPU support, emphasizing its simplicity for training on Linux or WSL.
- **Introduction of med-r1 model**: The **med-r1 model** was recently launched, featuring **1B parameters** and trained on a medical reasoning dataset, available on [Hugging Face](https://huggingface.co/Imran1/Med-R1-v1).
   - Intended for medical Q&A and diagnosis reasoning, it supports **4-bit inference** and has a max sequence length of **2048 tokens**.
- **Debate on RAG vs. Fine Tuning**: A YouTube video titled '[RAG vs. Fine Tuning (Live demo)](https://www.youtube.com/watch?v=LDMFL3bjpho)' discusses which methodology may yield better results in AI training.
   - Viewers expressed interest in more examples of RAG compared to fine-tuning, leading to plans for a future detailed tutorial on Kolo.
- **Future Kolo tutorial in the works**: In response to requests for more information, a member indicated plans for a follow-up video focused on **Kolo** usage, incorporating more comprehensive testing.
   - This upcoming tutorial aims to delve into finer details about the training data and methodologies used.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Imran1/Med-R1-v1">Imran1/Med-R1-v1 · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=LDMFL3bjpho">RAG vs. Fine Tuning (Live demo)</a>: Which is better RAG or Fine tuning? Does the industry have it wrong? Can fine tuning deliver better results than a traditional RAG system? Watch the video fo...</li><li><a href="https://github.com/rombodawg/Easy_training">GitHub - rombodawg/Easy_training</a>: Contribute to rombodawg/Easy_training development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1341483251901927556)** (116 messages🔥🔥): 

> `AI-Powered Phytochemical Formulation, Citizen Science Contributions, Challenges of Academic Publishing, Emotional Content in LLMs, Nutraceuticals for Mental Health` 


- **AI-Powered Phytochemical Formulation Framework Explained**: An AI-driven model systematically identifies plant-based compounds to optimize health support through [this blog post](https://www.marielandryceo.com/2025/02/title-ai-powered-phytochemical.html?m=1). This approach aims to enhance the development of evidence-informed nutraceuticals while emphasizing safety and efficacy.
- **Debates on Citizen Scientist Validity**: The conversation reflected on the legitimacy of self-titled 'citizen scientists', highlighting that scientific contributions could come from diverse backgrounds despite lacking formal academic titles. Members expressed concern that dismissing non-traditional contributors could stifle innovation and discourage valuable insights.
- **Critique of Academic Publishing Standards**: Participants discussed the rigid standards of academic publishing, criticizing the barriers it poses to those with non-traditional backgrounds in contributing to scientific discourse. They noted that publishing often requires endorsements that might limit broader participation in knowledge-sharing.
- **Emotional Content Classification in AI Models**: A member suggested that attention mechanisms trained on emotional content could improve AI's ability to handle subjective topics, although distinguishing social nuances remains challenging. Discussion emphasized a need for reliable classification systems in AI for moderating sensitive content.
- **Use Cases for Targeted Nutraceuticals**: The development of customized GPT models for generating plant-based molecule supplement recipes tailored to specific diseases was shared. Examples discussed included formulations for **Borderline Personality Disorder**, **Autism Spectrum Disorder**, and **Schizophrenia**, emphasizing caution as these are research-oriented.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/r1-1776">unsloth/r1-1776 · Hugging Face</a>: no description found</li><li><a href="https://bits.logic.inc/p/the-eagles-will-win-super-bowl-lix">Predicting the Super Bowl with LLMs</a>: Using LLMs to pick NFL winners better than 94.5% of humans.</li><li><a href="https://github.com/wrmedford/moe-scaling/blob/main/README.md">moe-scaling/README.md at main · wrmedford/moe-scaling</a>: Scaling Laws for Mixture of Experts Models. Contribute to wrmedford/moe-scaling development by creating an account on GitHub.</li><li><a href="https://www.marielandryceo.com/2025/02/title-ai-powered-phytochemical.html?m=1">Title: AI-Powered Phytochemical Formulation: A Data-Driven Approach to Supporting Health</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1341459247510061247)** (84 messages🔥🔥): 

> `Comparison of GPUs, Grok 3 vs ChatGPT-4, Hugging Face Playground Issues, LangChain Framework Release, SWE-Lancer Benchmark for LLMs` 


- **GPU Debate: 2080 Ti vs P40s**: A discussion arose comparing the value of **2080 Ti** and **P40s** for different applications in AI training.
   - Members debated the advantages of each GPU's performance, though no consensus was reached.
- **Grok 3 Declared Superior to ChatGPT-4**: It was mentioned that **Grok 3** is reportedly better than **ChatGPT-4**, sparking interest in its capabilities.
   - Community members expressed curiosity and speculation regarding the technical differences between the models.
- **Hugging Face Playground Experiencing Downtime**: Users reported that the Hugging Face Playground was down, generating a **500 Internal Error** when accessed.
   - Members acknowledged the issue and mentioned contacting the team for support.
- **LangChain Framework Unveiled**: Excitement surrounded the release of a new **LangChain framework**, emphasizing dynamic instruction learning for LLMs.
   - A tutorial video was shared, inviting users to engage with the updated capabilities for building self-improving agents.
- **New Benchmark for LLM Freelancing: SWE-Lancer**: **SWE-Lancer** benchmark was introduced by OpenAI to test LLMs' ability to perform software engineering tasks worth up to $1 million.
   - Initial insights suggest models perform better at solution selection than implementation, highlighting their strengths and weaknesses.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found</li><li><a href="https://huggingface.co/playground">Playground - Hugging Face</a>: no description found</li><li><a href="https://saiyan-world.github.io/goku/">Goku</a>: no description found</li><li><a href="https://www.youtube.com">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Nc3vIuPyQQ0">AI CEO Alexandr Wang | This Past Weekend w/ Theo Von #563</a>: Alexandr Wang is the founder and CEO of Scale AI, a platform that provides data training for AI programs. In 2021 he was named the youngest self-made billion...</li><li><a href="https://x.com/_philschmid/status/1891780812497887289?t=nkpIbh2A9B0wScpC0iJlqg&s=19">Tweet from Philipp Schmid (@_philschmid)</a>: Can LLMs earn $1 Million from software engineering? SWE-Lancer is a new Benchmark from @OpenAI that test if LLMs can do freelance tasks from Upwork valued at $1 million USD total in realworld payouts....</li><li><a href="https://tenor.com/view/office-space-yeah-uh-yeah-unsure-uh-sure-gif-5638327">Office Space Yeah GIF - Office Space Yeah Uh Yeah - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/Nouamanetazi/status/1892274582503248178">Tweet from Nouamane Tazi (@Nouamanetazi)</a>: 🚀 Excited to release *THE* Ultra-Scale Playbook - a comprehensive guide on training LLMs from 1 to 1000s of GPUs!</li><li><a href="https://www.youtube.com/watch?v=WW-v5mO2P7w">Build Self-Improving Agents: LangMem Procedural Memory Tutorial</a>: Learn how to implement dynamic instruction learning in LLM agents using the LangMem SDK. This technical tutorial demonstrates automatic prompt optimization, ...</li><li><a href="https://github.com/rombodawg/Easy_training/tree/main/Galore%2BQlora_With_Multi_GPU_Support">Easy_training/Galore+Qlora_With_Multi_GPU_Support at main · rombodawg/Easy_training</a>: Contribute to rombodawg/Easy_training development by creating an account on GitHub.</li><li><a href="https://qiita.com/takeuchiseijin/items/909c48b57127a37fbd12">PyTorchのAMPはbf16を使え．多分nanが出なくなる． - Qiita</a>: 最初からクライマックス公式に書かれている利用方法は下．でもモデルが数値的に不安定な計算（Softmax, division by epsilon...）を含んでるといつかnanが出る．ちなみにTr…</li><li><a href="https://github.com/huggingface/datasets/pull/6968">Use `HF_HUB_OFFLINE` instead of `HF_DATASETS_OFFLINE` by Wauplin · Pull Request #6968 · huggingface/datasets</a>: To use datasets offline, one can use the HF_DATASETS_OFFLINE environment variable. This PR makes HF_HUB_OFFLINE the recommended environment variable for offline training. Goal is to be more consist...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1341471799807971460)** (3 messages): 

> `Quantum Computing, Majorana 1 Chip, Neuralink Image Analysis` 


- **Microsoft's Majorana 1 Chip Unveiled**: In a recent post, **Satya Nadella** announced the launch of the **Majorana 1** chip, a significant advancement in quantum computing that promises to perform calculations in minutes that would take supercomputers **billions of years**.
   - A blog is created to discuss this innovation, explaining its potential to reshape industries and impact climate change.
- **Neuralink's Image Analysis Shared**: Multiple images were analyzed regarding **Neuralink's** developments over the last two days, showcasing various insights and details.
   - The attached images provide a visual representation of Neuralink's progress but lack specific descriptions in the messages.



**Link mentioned**: <a href="https://kuberwastaken.github.io/blog/Technology/Majorana-1---Why-Quantum-Computing-Matters-Now">Majorana 1 - Why Quantum Computing Matters Now</a>: Introduction: A Potential New Era of Computing Imagine a computer so powerful it could solve problems in minutes that would take today’s fastest supercomputers billions of years ...

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1341477367255928892)** (14 messages🔥): 

> `Dynamic Readme Images, Humorous LIMA dataset using LLaMA, Sentiment Market Forecasting project, Aster audio search app, CommentRescueAI for Python code` 


- **Create Dynamic Readme Images Easily**: A member shared a GitHub repository offering a template for generating dynamically updating images for various applications using GitHub workflows and Puppeteer.
   - This project is open source under the MIT License, aimed at enhancing readmes and blogs.
- **LIMA Dataset Gets a Witty Twist**: One user humorously modified the LIMA dataset to add comedic responses using the LLaMA-3.1-70B-Instruct-Turbo model, making interactions more engaging.
   - The resulting dataset can be found [here](https://huggingface.co/datasets/pulkitmehtawork/lima_humorous), and the creator welcomes feedback on this lighthearted approach.
- **Sentiment Market Forecasting Project Launch**: A member introduced their GitHub project focused on predicting market trends through sentiment analysis of news titles using a Hugging Face transformer model.
   - The project employs LSTM models and reports promising results, inviting users to check it out on [GitHub](https://github.com/yannis-gerontopoulos99/sentiment-market-forecasting).
- **Explore Aster: An Open Source Audio Search App**: A user announced a blog detailing the Aster audio search app that utilizes the HF Laion CLAP model, seeking feedback on its clarity.
   - Further performance comparisons between ONNX and PyTorch implementations are discussed in another blog post, highlighting batching support limitations.
- **Introducing CommentRescueAI for Python**: A member presented CommentRescueAI, a web extension designed to help add AI-generated docstrings and comments to Python code easily.
   - The extension is now available on the VS Code marketplace, and the creator seeks improvement suggestions and feature ideas.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/code/allanwandia/secondary-structure-data-analysis">Secondary Structure Data - Analysis</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from multiple data sources</li><li><a href="https://asteraudio.app/blog/whats-it-for">Aster</a>: no description found</li><li><a href="https://asteraudio.app/blog/webgpu-wasm-cuda">Aster</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=h7dijJBvwWA">Deepseek-R1 AI writes my Obsidian notes by watching my screen (open source)</a>: Download today for free: https://screenpi.pe</li><li><a href="https://github.com/rombodawg/Easy_training">GitHub - rombodawg/Easy_training</a>: Contribute to rombodawg/Easy_training development by creating an account on GitHub.</li><li><a href="https://blog.chataignon.org/joseph/2025-01-17/why-chatgpt-cant-spell-properly/">Why ChatGPT can&#8217;t spell</a>: no description found</li><li><a href="https://github.com/Kuberwastaken/Dynamic-Readme-Images">GitHub - Kuberwastaken/Dynamic-Readme-Images: A Template Repository to get &quot;Dynamically Updating&quot; Pictures in Readme</a>: A Template Repository to get &quot;Dynamically Updating&quot; Pictures in Readme - Kuberwastaken/Dynamic-Readme-Images</li><li><a href="https://huggingface.co/datasets/pulkitmehtawork/lima_humorous">pulkitmehtawork/lima_humorous · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/yannis-gerontopoulos99/sentiment-market-forecasting">GitHub - yannis-gerontopoulos99/sentiment-market-forecasting</a>: Contribute to yannis-gerontopoulos99/sentiment-market-forecasting development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1341514400619692153)** (3 messages): 

> `Discord Invite Rules, DeepSeek R1 Distilled Models, Conversation Storage Solutions` 


- **Discord invites questioned**: A member expressed concern that **Discord invites** might violate community guidelines in channel <#895532661383254098>.
   - The message was made with a friendly tone, indicated by the smiley face at the end.
- **Leveraging Simple SFT Data in DeepSeek Models**: There is curiosity about utilizing **simple SFT data** with **DeepSeek R1 distilled models** without reasoning, while still achieving **Chain of Thought (COT)** during inference.
   - The member is seeking suggestions and help from the community on this matter.
- **Advice Needed on Conversation Storage Solutions**: A developer is looking for advice on the best **storage solutions** for chat-based applications, considering both SQL and **NoSQL databases** for varying conversation volumes.
   - They asked about any implemented solutions for storing lengthy chat conversations, including cloud storage options like **S3**.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1341617148250030080)** (4 messages): 

> `Fine-tuning training time, M3 Max error with MPS device, Course extension plans, Link for Second unit on agents` 


- **3090 Fine-tuning Time Questioned**: A member estimates that the fine-tuning notebook takes **2h 30min** on a **3090**, questioning if this is an acceptable duration.
   - They noted that running it on an **M3 Max** via CPU reduces the time to **2 hours**, suggesting the 3090 might be slower than expected.
- **M3 Max Hits Error on MPS Device**: While trying to run the fine-tuning notebook on an **M3 Max**, a member encountered an error stating that *'The operator 
   - This indicates that the **MPS device** lacks support for certain operations needed for fine-tuning.
- **Future Course Extensions in Discussion**: A member inquired about any **future plans** to extend the ongoing course, hinting at interest for continued learning.
   - This reflects the community's anticipation for additional materials.
- **Request for Agents Unit Link**: A member asked for the **link to the Second unit** focusing on agents, eager to access more specific content.
   - This points to a demand for structured access to course materials and information.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1341458225689661522)** (333 messages🔥🔥): 

> `Certificate Generation Issues, Hugging Face AI Agents, Unit 2 Release, Community Building, Using API Keys and Tokens` 


- **Certificate Generation Issues Resolved**: Many users recently struggled to generate their certificates after completing the quiz, but it has been resolved by directing submissions to a new space that generates certificates directly.
   - Participants were encouraged to try the updated quiz link to receive their certificates promptly.
- **Exploring Hugging Face AI Agents**: Users are actively discussing the best practices for training AI agents, including using tools such as RAGs and API calls for functionality.
   - Participants expressed interest in project collaboration and using AI tools effectively in various applications.
- **Upcoming Release of Unit 2**: There was anticipation around the release of Unit 2, with a mention that the content was expected by February 18th.
   - Confusion arose when users did not see the expected unit, prompting further inquiry into the release status.
- **Building Community Connections**: Many participants took the opportunity to introduce themselves and express their backgrounds, particularly in software development and AI.
   - There was an emphasis on forming connections and supporting each other in learning and developing AI projects.
- **API Key Management for Agents**: Several discussions revolved around the importance of managing API keys and secrets when developing AI agents.
   - Users shared tips on setting environment variables and best practices to securely store and utilize these keys in their projects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://share.1password.com/s#dpSJqyL8KP5ZWjQXdnVB2EHPOeW6jFyiH30fpkJYLCs">I’m sharing an item with you using 1Password</a>: A password manager, digital vault, form filler and secure digital wallet. 1Password remembers all your passwords for you to help keep account information safe.</li><li><a href="https://huggingface.co/spaces/agents-course/First_agent">First Agent - a Hugging Face Space by agents-course</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/unit1/dummy-agent-library">Dummy Agent Library - Hugging Face Agents Course</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1SVS-ALf9ToN6I6WmJno5RQkZEHFhaykJ).">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/spaces/vitpolis/First_agent_template">First Agent Template - a Hugging Face Space by vitpolis</a>: no description found</li><li><a href="https://agents-course-unit-1-quiz.hf.space/">Dataset Quiz for agents-course/unit_1_quiz</a>: no description found</li><li><a href="https://lightning.ai/someshfengde/studios/bonus-unit-1-lightningai">Bonus Unit 1 lightningai - a Lightning Studio by someshfengde</a>: Enabling function calling by using LORA to finetune gemma-2b-it for free in 15 mins by using lightning.ai studio&#39;s GPU.</li><li><a href="https://huggingface.co/spaces/agents-course/First_agent_template">First Agent Template - a Hugging Face Space by agents-course</a>: no description found</li><li><a href="https://github.com/huggingface/agents-course/blob/main/notebooks/bonus-unit1/gemma-SFT-thinking-function_call.ipynb">agents-course/notebooks/bonus-unit1/gemma-SFT-thinking-function_call.ipynb at main · huggingface/agents-course</a>: This repository contains the Hugging Face Agents Course.  - huggingface/agents-course</li><li><a href="https://x.com/dylan_ebert_/status/1892000664600887686">Tweet from dylan (@dylan_ebert_)</a>: What are AI Agents?in 70 seconds</li><li><a href="https://www.youtube.com/watch?v=qU3fmidNbJE">AI Agents Fundamentals In 21 Minutes</a>: Improve your AI skills with the FREE Prompting QuickStart Guide I made in collaboration with Hubspot: https://clickhubspot.com/1gg9Want to get ahead in your ...</li><li><a href="https://github.com/huggingface/agents-course/blob/main/units/en/unit1/what-are-llms.mdx?plain=1).">agents-course/units/en/unit1/what-are-llms.mdx at main · huggingface/agents-course</a>: This repository contains the Hugging Face Agents Course.  - huggingface/agents-course</li><li><a href="https://www.youtube.com/watch?v=ZZ2QUCePgYw">Intro to AI agents</a>: Vertex AI Agent Builder quickstart → https://goo.gle/3UPJ7dNGenAI powered App with Genkit → https://goo.gle/4fCSTrKDemystifying AI agents, Googlers Aja Hamme...</li><li><a href="https://www.youtube.com/watch?v=45YtsZbz334">Understanding Tools &amp; Function Calling in AI</a>: Understanding Tools &amp; Function-Calling in AI | Unofficial Hugging Face Agents Course Support VideoBonus Unit 1. Fine-tuning an LLM for Function-callingCourse...</li><li><a href="https://github.com/andrewyng/aisuite">GitHub - andrewyng/aisuite: Simple, unified interface to multiple Generative AI providers</a>: Simple, unified interface to multiple Generative AI providers  - GitHub - andrewyng/aisuite: Simple, unified interface to multiple Generative AI providers</li><li><a href="https://youtu.be/tlV_uxodcQw">Create AI Agents with Hugging Face! My Thoughts on Their Latest FREE Course!</a>: 🚀 Create AI Agents with Hugging Face! 🤖Join me as I share my journey of building a simple AI agent using the Hugging Face Agents Course! Whether you&#39;re new...</li><li><a href="https://github.com/gradio-app/gradio/issues/9895">Too many arguments provided for the endpoint - JavaScript-Warning in browser console · Issue #9895 · gradio-app/gradio</a>: Describe the bug When launching a gradio application with gradio.Chatbot or gradio.ChatInterface, there is a JavaScript-Warning in the browser console every time a message is submitted: Too many ar...</li><li><a href="https://huggingface.co/datasets/agents-course/certificates">agents-course/certificates · Datasets at Hugging Face</a>: no description found</li><li><a href="https://acrobat.adobe.com/id/urn:aaid:sc:EU:38802316-7b5c-48f5-b4a1-c5437d0a48f5">Adobe Acrobat</a>: no description found</li><li><a href="https://aiagentslist.com/">AI Agents Directory - Compare Top AI Assistants</a>: Explore our curated collection of AI agents. Find the perfect AI assistant for your needs with detailed comparisons, reviews, and feature analysis.</li><li><a href="https://t.me/AgentNexus)">Telegram – a new era of messaging</a>: Fast. Secure. Powerful.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1341457241966116864)** (411 messages🔥🔥🔥): 

> `Grok 3 Release, Model Comparisons, AI in Coding, Aider and Integration, OpenRouter Limitations` 


- **Grok 3 becomes available for free**: The world's smartest AI, Grok 3, is now available for free until server capacity is reached, with increased access for X Premium+ users and SuperGrok members.
   - Early access to features like Voice Mode is also part of the offering, creating excitement in the community.
- **Concerns about query limits**: Users speculate that Grok 3 will likely have restrictions, with estimates suggesting a limit of about five queries per day.
   - This raises concerns regarding usability, especially for those accustomed to more extensive interactions with models.
- **Discussions on AI model efficiency**: The conversation touches on a newly discussed model named LLADA, which demonstrates high performance on coding tasks.
   - Insights suggest that models like LLADA may become a strong alternative for code editing due to their innovative approaches.
- **AI integration with Aider**: Debates continue regarding the integration of various AI models with Aider, with emphasis on how they impact coding efficiency.
   - Some users express a desire for Aider to teach coding methodologies rather than solely relying on AI-generated solutions.
- **OpenRouter and resource limits**: Discussions reveal that OpenRouter may limit endpoints for models like o3-mini to 100k requests, indicating potential usability issues.
   - This limitation complicates the experience of users trying to leverage OpenRouter for more intensive AI interactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/benchmarks.html#gpt-code-editing-benchmarks">GPT code editing benchmarks</a>: Benchmarking GPT-3.5 and GPT-4 code editing skill using a new code editing benchmark suite based on the Exercism python exercises.</li><li><a href="https://www.kaggle.com/competitions/konwinski-prize">Konwinski Prize</a>: $1M for the AI that can close 90% of new GitHub issues</li><li><a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">My LLM codegen workflow atm</a>: A detailed walkthrough of my current workflow for using LLms to build software, from brainstorming through planning and execution.</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">no title found</a>: no description found</li><li><a href="https://x.com/ai_for_success/status/1892170935870079388">Tweet from AshutoshShrivastava (@ai_for_success)</a>: 🚨 Grok-3 is now rolling out to Premium users too .Note: I can see this on grok[.]com, but not on the X app yet.</li><li><a href="https://x.com/iruletheworldmo/status/1891529614167789967">Tweet from 🍓🍓🍓 (@iruletheworldmo)</a>: grok 3 is here, and it’s agi.this isn’t just another model release. this is the moment everything changes. forget gpt, forget claude, forget every ai you’ve used before—they’re already obsolete. the u...</li><li><a href="https://github.com/openai/SWELancer-Benchmark">GitHub - openai/SWELancer-Benchmark: This repo contains the dataset and code for the paper &quot;SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?&quot;</a>: This repo contains the dataset and code for the paper &quot;SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?&quot; - openai/SWELancer-Benchmark</li><li><a href="https://x.com/perplexity_ai/status/1891916573713236248?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from Perplexity (@perplexity_ai)</a>: Today we&#39;re open-sourcing R1 1776—a version of the DeepSeek R1 model that has been post-trained to provide uncensored, unbiased, and factual information.</li><li><a href="https://tenor.com/view/richard-attenborough-whip-whipped-whiplash-whiplashed-gif-16890874512241116786">Richard Attenborough Whip GIF - Richard Attenborough Whip Whipped - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/MoonshotAI/MoBA/blob/master/MoBA_Tech_Report.pdf">MoBA/MoBA_Tech_Report.pdf at master · MoonshotAI/MoBA</a>: MoBA: Mixture of Block Attention for Long-Context LLMs - MoonshotAI/MoBA</li><li><a href="https://x.com/OpenAI/status/1891911123517018521">Tweet from OpenAI (@OpenAI)</a>: Today we’re launching SWE-Lancer—a new, more realistic benchmark to evaluate the coding performance of AI models. SWE-Lancer includes over 1,400 freelance software engineering tasks from Upwork, value...</li><li><a href="https://x.com/elonmusk/status/1891911120572567983">Tweet from Elon Musk (@elonmusk)</a>: The @xAI Grok 3 release will improve rapidly every day this week. Please report any issues as a reply to this post.</li><li><a href="https://kimi.moonshot.cn/">Kimi.ai - 会推理解析，能深度思考的AI助手</a>: no description found</li><li><a href="https://x.com/xai/status/1892400129719611567">Tweet from xAI (@xai)</a>: This is it: The world’s smartest AI, Grok 3, now available for free (until our servers melt).Try Grok 3 now: https://x.com/i/grokX Premium+ and SuperGrok users will have increased access to Grok 3, in...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1341461614683230370)** (14 messages🔥): 

> `Aider browser feature issues, LLM inference speed concerns, Using conventions in Aider, Font color visibility issues, Integration of agents into Aider` 


- **Aider browser feature causing confusion**: A user reported issues with the Aider browser feature, which led to confusion regarding setting the API key each time.
   - *“I didn't realize I need to set API key each time.”*
- **LLM inference speed is a bottleneck**: A member expressed frustration about **LLM inference speed** while coding with Aider, asking about solutions to improve it.
   - Another contributor mentioned that **Azure's OpenAI API** is often faster based on community feedback.
- **Using conventions to improve coding practices**: A user highlighted the importance of using a conventions file to guide coding practices within Aider, detailing specific preferences like using **httpx** over **requests**.
   - They recommended loading the conventions file with `/read CONVENTIONS.md` to ensure it's cached and marked as read-only.
- **Visibility issues with font colors**: A user raised concerns about the **blue font color** being difficult to read and inquired about ways to change font colors.
   - Suggestions included switching to dark mode, although another member noted that **light mode** didn't improve visibility.
- **Interest in agents integration**: A user queried if integration of **agents** into Aider is planned, showing interest in future features.
   - This question reflects a desire for enhancements to Aider's capabilities.



**Link mentioned**: <a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: Tell aider to follow your coding conventions when it works on your code.

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1341525366627369010)** (2 messages): 

> `Ministral integration with Aider, Build process and performance, RAG feature comparison` 


- **Inquiry on Aider and Ministral**: A member inquired if anyone has tried integrating **Ministral** with **Aider**, expressing curiosity about their compatibility.
   - They referenced a [LinkedIn post](https://www.linkedin.com/posts/deividas-mataciunas_ai-mistral-opensource-ugcPost-7294619759722074112-a_JM?utm_source=share&utm_medium=member_android&rcm=ACoAABZCz5cBsMAYVy_zzTHh2HzsmuBv_27C49Y) discussing **Mistral**.
- **Concerns about Slow Build Process**: A member shared that they appreciate the git-like behavior of the build process but find it *very slow*, as it rewrites chunks with 'build' during the API calls.
   - They noted it uses **TF-IDF** for searching instead of embeddings, which may hinder context similarity, and suggested adding a feature to utilize 50% batching costs from providers.
- **RAG Performance Comparison**: One user mentioned that their *RAG* build is still indexing but they can use it during the process and find it yields better results than the **AIChat RAG feature**.
   - They also provided build status updates, detailing processed files and input costs, highlighting **staged files** at **1019** and **chunks** at **20115**.


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1341455098278051870)** (405 messages🔥🔥🔥): 

> `Model Performance Comparisons, Grok 3 Insights, Usage of Cursor and AI Models, Start-Up Collaboration, Coding with AI Models` 


- **Debate on AI Model Effectiveness**: Users expressed strong opinions about various AI models, with many favoring Sonnet for coding tasks due to its reliability and contextual understanding, while disapproving of Grok 3's performance.
   - The discussion highlighted that while Grok 3 received mixed reviews, Anthropic models are well-regarded for coding, particularly when provided with good context.
- **OpenAI vs. Anthropic Models**: Users generally perceived OpenAI and DeepSeek as strong contenders for general tasks, but noted Anthropic models excel in coding and complex tasks when combined with reasoning and careful execution.
   - Despite differing opinions on Grok 3, many users stated that all models have their strengths in specific areas, advocating for using the best tool for the job.
- **Grok 3 Controversy**: Some users criticized Grok 3's utility, labeling it a disappointment based on personal testing experiences that showcased its inadequacies in coding tasks.
   - In contrast, others questioned the validity of arguments against Grok 3 by pointing out its favorable reviews from various YouTubers.
- **Cursor Usage and Experiences**: Users shared their experiences using Cursor in agent mode, emphasizing how features like changelogs and customized rules helped streamline their workflow.
   - Discussions revolved around using AI assistants for automating tasks and ensuring quality output, highlighting both positive and negative interactions.
- **Startup Opportunity Announcement**: A user announced interest in starting a new venture and is seeking a coder who has time and dedication to collaborate, offering to fund the project.
   - This led to an openness for potential partnerships among members of the channel.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pieces.app)">no title found</a>: no description found</li><li><a href="https://caddyserver.com/docs/">Welcome - Caddy Documentation</a>: Caddy is a powerful, enterprise-ready, open source web server with automatic HTTPS written in Go</li><li><a href="https://www.youtube.com/wat"> - YouTube</a>: no description found</li><li><a href="https://x.com/windsurf_ai/status/1892322088507105561?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Windsurf (@windsurf_ai)</a>: DeepSeek-V3 is now unlimited in Windsurf Pro and Ultimate plans.0 prompt credits. 0 flow action credits.</li><li><a href="https://x.com/theo/status/1891960246286828028">Tweet from Theo - t3.gg (@theo)</a>: Imagine coding a project for 4 months without using any version control</li><li><a href="https://x.com/OpenAI/status/1891911132983722408">Tweet from OpenAI (@OpenAI)</a>: Current frontier models are unable to solve the majority of tasks.</li><li><a href="https://x.com/kevinhou22/status/1891375289919500794?t=k5skkvhMsKodfbvKbDiYhw&s=19">Tweet from Kevin Hou (@kevinhou22)</a>: We built Windsurf’s agent to not rely on embedding indexes like other tools out there. One size fits all retrieval simply doesn’t scale for monorepos. Instead, our agent uses tools that a human would ...</li><li><a href="https://www.youtube.com/watch?v=gXmakVsIbF0">You HAVE to Try Cursor + o3  + Storybook</a>: Last Sunday evening in Japan, OpenAI quietly unveiled their new deep research product powered by the previously unreleased O3 model. In this video, I show yo...</li><li><a href="https://x.com/karpathy/status/1891720635363254772">Tweet from Andrej Karpathy (@karpathy)</a>: I was given early access to Grok 3 earlier today, making me I think one of the first few who could run a quick vibe check.Thinking✅ First, Grok 3 clearly has an around state of the art thinking model ...</li><li><a href="https://simonwillison.net/2024/Dec/19/one-shot-python-tools/">Building Python tools with a one-shot prompt using uv run and Claude Projects</a>: I’ve written a lot about how I’ve been using Claude to build one-shot HTML+JavaScript applications via Claude Artifacts. I recently started using a similar pattern to create one-shot Python utilities,...</li><li><a href="https://github.com/disler/single-file-agents">GitHub - disler/single-file-agents: What if we could pack single purpose, powerful AI Agents into a single python file?</a>: What if we could pack single purpose, powerful AI Agents into a single python file? - disler/single-file-agents</li><li><a href="https://github.com/onlook-dev/onlook?tab=readme-ov-file#getting-started">GitHub - onlook-dev/onlook: The open source Cursor for Designers. Design directly in your live React app and publish your changes to code.</a>: The open source Cursor for Designers. Design directly in your live React app and publish your changes to code. - onlook-dev/onlook</li><li><a href="https://www.notion.so/Experiment-Prompting-86aa8f988fce404cbf70134690d2635a#eb3588f421e14e21b029a51993eeb65a">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team</li><li><a href="https://x.com/ryolu_/status/1891677600587629043?s=46">Tweet from Ryo Lu (@ryolu_)</a>: Late for valentines, but @cursor_ai has been cooking</li><li><a href="https://x.com/ryolu_/status/1888455169081577955?s=46">Tweet from Ryo Lu (@ryolu_)</a>: 4 days in @cursor_ai• Everyone is cracked• @shaoruu keeps chasing me for designs and builds them• 1 meeting per week• Got a line up coming in the next release for you already, here’s a lil’ preview ⚡️</li><li><a href="https://www.subframe.com/">Subframe – The best way to build UI, fast.</a>: Build stunning UI in minutes with a drag-and-drop visual editor, beautifully crafted components, and production-ready code. Optimized for React &amp; TailwindCSS.</li><li><a href="https://www.relume.io/">Relume — Websites designed &amp; built faster with AI | AI website builder</a>: Use AI as your design ally, not as a replacement. Effortlessly generate sitemaps and wireframes for marketing websites in minutes with Relume’s AI website builder.</li><li><a href="https://21st.dev/">21st.dev - The NPM for Design Engineers</a>: Ship polished UIs faster with ready-to-use React Tailwind components inspired by shadcn/ui. Built by design engineers, for design engineers.</li><li><a href="https://flexboxlabs.netlify.app/">Flexbox Labs</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1341456025894522953)** (361 messages🔥🔥): 

> `Deep Research Lag Issues, Perplexity Pro Subscription, Image Generation Capability, Grok Integration, R1-1776 Model Updates` 


- **Deep Research Faces Crashes**: Users reported persistent crashing of the Deep Research feature, particularly among those on the enterprise plan.
   - Issues navigating threads and accessing library content were also highlighted, prompting discussions for potential fixes.
- **Concerns Over Pro Subscription Fees**: A user expressed confusion regarding the pro subscription model that offers access to multiple models for a single fee, stirring speculation about the validity of this pricing strategy.
   - Another user humorously suggested that the models might be pirated, although this was quickly dismissed as unfounded.
- **Image Generation Potential**: The channel highlighted that despite having access to widgets indicating image generation capabilities, some users struggled to utilize this feature effectively on the platform.
   - Workarounds were discussed, including generating prompts specifically for image creation and using browser addons.
- **Updates on Grok Integration**: Speculation about the release of Grok 3 and its potential implementation within Perplexity led to questions about its current status and functionality.
   - Community members expressed their anticipation and interest in the Grok integration, awaiting updates from the team.
- **R1-1776 Model Discussion**: The R1-1776 model's reasoning capabilities and its differences from the standard R1 model were examined, noting the potential for censorship-free responses.
   - Users shared experiences using the model, revealing its performance on sensitive topics and the context of operation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/natalrhyme">Tweet from undefined</a>: no description found</li><li><a href="https://openrouter.ai/perplexity/r1-1776">R1 1776 - API, Providers, Stats</a>: Note: As this model does not return &lt;think&gt; tags, thoughts will be streamed by default directly to the `content` field.R1 1776 is a version of DeepSeek-R1 that has been post-trained to remove ce...</li><li><a href="https://huggingface.co/unsloth/r1-1776-GGUF">unsloth/r1-1776-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2Fwyk3twpors8e1.jpeg">https://i.redd.it/wyk3twpors8e1.jpeg</a>: no description found</li><li><a href="https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppm">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.</li><li><a href="https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppmilmeaekegkpckebkeahjgmhggpj,">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.</li><li><a href="https://x.com/AravSrinivas/status/1891905511286768018">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Congratulations to @xai for building world-class reasoning models in such a short period. This is a super strong response from America to DeepSeek. America builds fast, and xAI team and Elon are setti...</li><li><a href="https://www.youtube.com/watch?v=8EyLJHmLNew">New footage of Delta plane flip at Toronto airport</a>: A child was among three people ­critically injured after a Delta Airlines flight flipped over as it landed at Toronto airport on Monday.At least 18 people we...</li><li><a href="https://news.microsoft.com/source/features/ai/microsofts-majorana-1-chip-carves-new-path-for-quantum-computing/">Microsoft’s Majorana 1 chip carves new path for quantum computing - Source</a>: Majorana 1, the first quantum chip powered by a new Topological Core architecture
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1341464202623320116)** (19 messages🔥): 

> `IRS Acquiring Nvidia Supercomputer, ChatGPT Energy Use, Robotics, Australia's Central Bank Cuts, Neural Networks` 


- **IRS Acquires Nvidia Supercomputer**: Reports indicate that the **IRS** is acquiring a **Nvidia Supercomputer** to enhance its operational capabilities, boosting efficiency in data analysis.
   - This move is seen as significant in optimizing tax processing technologies amidst increasing data challenges.
- **ChatGPT Energy Use Overestimated**: A discussion raised the concern that the **energy use of ChatGPT** has been largely overestimated, with various assessments needing a reassessment of their methodologies.
   - Experts believe *accurate energy metrics* will help improve perceptions of AI sustainability.
- **Exploring Robotics History**: People were curious about the **first robotic innovations**, with a [related article](https://www.perplexity.ai/search/what-were-the-first-robotic-ar-2c97SCKyRc6YwwpLlrpb_A) shedding light on early developments and breakthroughs in the field.
   - The discussion highlighted the **evolution** of robotics technology over the years.
- **Australia's Central Bank Cuts Rates**: Recent news showed that **Australia's central bank** has made significant **interest rate cuts** to stimulate economic activity.
   - The decision aims to address the country's economic challenges, as [detailed here](https://www.perplexity.ai/page/australia-s-central-bank-cuts-9lwmh4dkSfWOo41dB7ga2Q).
- **Neural Networks Analyze Data**: Members discussed advancements in **neural networks**, referencing a [link](https://www.perplexity.ai/search/neural-network-qqb0tqu2TGGF6ukUwue7HA) that explores their application in complex data analysis.
   - The conversation emphasized the growing significance of **neural networks** in various fields, from AI to neuroscience.



**Link mentioned**: <a href="https://www.youtube.com/embed/Kxl4xFbfKwU">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1341541348884353115)** (7 messages): 

> `R1-1776 Hot Swap in Sonar API, Image Usage in API, API Profile Setup, Deep Research in API` 


- **R1-1776 hot swap query**: A member inquired if **R1-1776** is hot swapped in place on the **Sonar API**, specifically asking from **OpenRouter**.
   - Another member mentioned they were also thinking about the same issue.
- **Assistance with Image in API**: A member requested guidance on how to use images within the API.
   - They expressed gratitude after receiving feedback but did not provide further specifics.
- **API Profile Configuration**: A member asked if the API utilizes the **Profile** set up in the settings of the account.
   - This inquiry reflects interest in how account configurations affect API behavior.
- **Inquiry about Deep Research Features**: A member queried whether **deep research** capabilities are set to be integrated into the API.
   - This indicates anticipation for new functionalities within the API offerings.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1341454612921585825)** (182 messages🔥🔥): 

> `Grok 3 Performance, PaliGemma Model Updates, AI CUDA Engineer, Customer Support Automation Challenges, Claude Web App Enhancements` 


- **Grok 3 shows promise but raises skepticism**: There is a general vibe that while **Grok 3** is good, skepticism remains about its true capabilities, prompting a search for more reliable sources beyond Twitter.
   - Members discuss concerns regarding overall performance metrics and whether Grok 3 can stand up to competition.
- **PaliGemma 2 introduces mixed models**: Google introduced **PaliGemma 2 mix**, which allows for out-of-the-box usage on a diverse set of tasks, moving beyond pre-training checkpoint limitations.
   - Comparisons are made to previous versions, with discussions about the clarity and practicality of the naming and functionality.
- **AI CUDA Engineer aims for efficiency**: Sakana AI launched the **AI CUDA Engineer**, an AI system designed to produce optimized CUDA kernels that can achieve **10-100x speedup** over typical implementations.
   - This system is heralded as potentially transformative for AI-driven performance optimization in machine learning operations.
- **Customer support automation reconsidered**: Klarna’s journey in customer support highlights the challenges and limitations of replacing human agents with AI, emphasizing a growing demand for personal interaction.
   - Anecdotes from industry insiders reflect on the balance between automating simple tasks and addressing complex queries that require a human touch.
- **Claude web app evolves with new features**: Updates for the Claude web app indicate ongoing enhancements with new features like **web search** and **Paprika mode** focusing on improving the AI's reasoning capabilities.
   - This includes various new builds that enhance its performance and user interaction as they prepare for public rollout.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://thinkingmachines.ai/">Thinking Machines Lab</a>: no description found</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://www.commercialappeal.com/story/money/business/2025/02/13/xai-gas-turbines-at-memphis-supercomputer/78540969007/">Elon Musk's xAI wants to keep using gas turbines in Memphis: Documents</a>: no description found</li><li><a href="https://www.theverge.com/news/614883/humane-ai-hp-acquisition-pin-shutdown">Humane is shutting down the AI Pin and selling its remnants to HP</a>: Humane is “winding down” its AI Pin.</li><li><a href="https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/">Accelerating scientific breakthroughs with an AI co-scientist</a>: no description found</li><li><a href="https://news.xbox.com/en-us/2025/02/19/muse-ai-xbox-empowering-creators-and-players/">Tweet from Empowering Creators and Players With Muse, a Generative AI Model for Gameplay - Xbox Wire</a>: Introducing Muse, a new kind of generative AI model that lets you play and create. From developer ideation to one day supporting game preservation, Muse holds potential to unlock new possibilities.</li><li><a href="https://x.com/james_tackett1/status/1891898442206638237">Tweet from Jamie (@james_tackett1)</a>: @natolambert Why do you think the other LLMs lead when the graphs in the xAI presentation showed that Grok 3 outperformed all other LLMs by a fairly wide margin? For example:</li><li><a href="https://x.com/dchaplot/status/1891920016339042463">Tweet from Devendra Chaplot (@dchaplot)</a>: Career Update: Incredibly fortunate and excited to be part of the founding team at Thinking Machines Lab!https://thinkingmachines.ai/Join us: https://6wajk07p.paperform.co/</li><li><a href="https://x.com/M1Astra/status/1892091124589920532">Tweet from M1 (@M1Astra)</a>: BREAKING: Anthropic set to release Thinking Model/Mode & Web Search capabilities imminently, based on findings in the latest update to the Claude iOS app.New symbols like &#34;Steps&#34; & &#34;Think&...</li><li><a href="https://x.com/btibor91/status/1892290734650433980">Tweet from Tibor Blaho (@btibor91)</a>: Claude web app updates - looks like web search & Paprika modes (the new thinking model) are still in the works, with multiple new builds deployed in the last 24 hoursThis includes a new experiment, &#...</li><li><a href="https://x.com/mervenoyann/status/1892289763954069720">Tweet from merve (@mervenoyann)</a>: @skalskip92 @onuralpszr it&#39;s mixed transfer, but this time model accepts open ended inputs and not structured task prefixes 🥹</li><li><a href="https://x.com/TheXeophon/status/1891795532500111752">Tweet from Xeophon (@TheXeophon)</a>: The last time it happened was *checks notes* two weeks agoQuoting Gavin Baker (@GavinSBaker) This is the first time in more than a year IIRC that one model has been #1 in every category.</li><li><a href="https://x.com/ashtom/status/1891925306430337110">Tweet from Thomas Dohmke (@ashtom)</a>: Our new code completion model is shipping in public preview today. We are calling it GPT-4o Copilot. Based on GPT-4o mini, with mid-training on a code-focused corpus exceeding 1T tokens and reinforcem...</li><li><a href="https://x.com/SakanaAILabs/status/1892385766510338559">Tweet from Sakana AI (@SakanaAILabs)</a>: Introducing The AI CUDA Engineer: An agentic AI system that automates the production of highly optimized CUDA kernels.http://sakana.ai/ai-cuda-engineer/The AI CUDA Engineer can produce highly optimize...</li><li><a href="https://x.com/klarnaseb/status/1892262217568891179">Tweet from Sebastian Siemiatkowski (@klarnaseb)</a>: @GergelyOrosz We are not reversing course. We are further developing it. Today our AI chat bot handles more complex enquires and at higher quality then back when you tested it. But at the same time th...</li><li><a href="https://fxtwitter.com/testingcatalog/status/1892133844184088576">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: Latest Claude app for iOS is getting ready to have web search and reasoning features 👀Web Search beta toggle is now hidden in the UI but not yet working as it will likely require a new m...</li><li><a href="https://x.com/_xjdr/status/1891911178147987513">Tweet from xjdr (@_xjdr)</a>: TL;DR grok3 is fine and passes the vibe check of frontier level quality but its not better than R1 or o1-pro for me for most things i do. overall much better than i had expected, i put it in the gemin...</li><li><a href="https://x.com/GavinSBaker/status/1891723733976465420">Tweet from Gavin Baker (@GavinSBaker)</a>: This is the first time in more than a year IIRC that one model has been #1 in every category.</li><li><a href="https://x.com/owl_posting/status/1892317797172015210">Tweet from owl (@owl_posting)</a>: lmao greg brockman&#39;s affiliation for the Evo paper is &#39;independent researcher&#39;</li><li><a href="https://tenor.com/view/surprised-pikachu-pokemon-shock-surprised-pikachu-gif-15357817">Surprised Pikachu GIF - Surprised Pikachu Pokemon - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://developers.googleblog.com/en/introducing-paligemma-2-mix/">Introducing PaliGemma 2 mix: A vision-language model for multiple tasks</a>: no description found</li><li><a href="https://huggingface.co/google/paligemma2-3b-mix-448#paligemma-2-results-by-model-resolution-and-size>">google/paligemma2-3b-mix-448 · Hugging Face</a>: no description found</li><li><a href="https://github.com/FlagAI-Open/OpenSeek">GitHub - FlagAI-Open/OpenSeek: OpenSeek aims to unite the global open source community to drive collaborative innovation in algorithms, data and systems to develop next-generation models that surpass DeepSeek.</a>: OpenSeek aims to unite the global open source community to drive collaborative innovation in algorithms, data and systems to develop next-generation models that surpass DeepSeek. - FlagAI-Open/Open...</li><li><a href="https://x.com/GergelyOrosz/status/1892196257608687842">Tweet from Gergely Orosz (@GergelyOrosz)</a>: Klarna was the company that went all-on replacing customer support with an AI bot and went on to brag about the cost savings.Now they are reversing course.Easy to see more companies blindly replacing ...</li><li><a href="https://x.com/altryne/status/1884778839009796411">Tweet from Alex Volkov (Thursd/AI) 🔜 AIENG summit NY (@altryne)</a>: Zuck highlights from the earnings call: - LLama 4 & LLama 4 mini (done with pre-training)- Confirms reasoning LLaMas! - Llama 4 will be natively multimodal -- it&#39;s an omni-model -- and it will hav...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1341602327458484294)** (6 messages): 

> `tulu3 70B model, RLVR phase training, GRPO memory optimization, paper updates` 


- **RLVR Training Duration for tulu3 70B**: The RLVR phase training for the **tulu3 70B model** on an 8×8 H100 setup is estimated to take about **a week** if it fits in memory, though it can vary based on prompts and settings.
   - A member mentioned that it should take **O(days**, supporting the estimate.
- **Leveraging GRPO for Memory Management**: Using **GRPO** will enhance memory capabilities, which could positively influence the training process.
   - This suggests that the technical setup can be optimized further for better performance.
- **Reminder for Paper Updates**: A member acknowledged they had forgotten to check the updated paper regarding the training process.
   - The team noted they have updated the paper with relevant information about the training phase.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1341459754677043312)** (101 messages🔥🔥): 

> `OpenAI's Deep Research Tool, AI-generated music popularity, Evo 2 foundation model for biology, Discussion on AI model licenses, Upcoming speaker event at UCSC` 


- **OpenAI’s Deep Research Tool reveals limitations**: A user shared insights regarding OpenAI's Deep Research tool, highlighting that it *claims to analyze a complete dataset* but only referenced approximately 5-6 papers out of thousands.
   - They noted that despite these limitations, sharing experiences is valuable for understanding the tool's strengths and weaknesses.
- **AI-generated music sees massive engagement**: A member noted that a video featuring AI-generated lofi music reached **2.1M** views, leading to discussions about the rising popularity and potential mainstream adoption of AI art.
   - *Lofi girl will be homeless very soon* became a humorous remark reflecting the competitive landscape as users realize they are listening to AI-generated music.
- **Evo 2 model presents a breakthrough in biology**: Michael Poli announced Evo 2, a new foundation model boasting **40 billion parameters** and designed for biological applications, set to advance understanding of genomics significantly.
   - Evo 2 aims to demonstrate test-time scaling laws and improve breast cancer variant classification while promoting real open-source efforts in AI.
- **Community licenses and model fine-tuning debate**: A conversation arose about the compatibility of community licenses between Llama 3.1 and Llama 3.3 models when fine-tuning outputs, leading to the conclusion that both licenses must be retained.
   - This discussion highlights the complexities of using multiple model versions within the AI community and the legalities involved.
- **Upcoming UCSC speaker event**: A member announced an upcoming speaker event at UCSC focusing on NLP, inviting anyone in the South Bay area to join and meet with key figures in the AI field.
   - They encouraged enthusiast participation and hinted at having a **GPT-4 2-year-anniversary cake** for attendees.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/MichaelPoli6/status/1892243237940904112">Tweet from Michael Poli (@MichaelPoli6)</a>: [2/7] I am excited about Evo 2 for many reasons. It is undoubtedly a leap forward in our ability to understand and generate genomes. I also hope it 2 can help push progress on real open-source in AI, ...</li><li><a href="https://en.wikipedia.org/wiki/Thinking_Machines_Corporation">Thinking Machines Corporation - Wikipedia</a>: no description found</li><li><a href="https://x.com/tianchezhao/status/1890788490016616780">Tweet from Tiancheng Zhao (Tony) (@tianchezhao)</a>: We trained Qwen 2.5 VL 3B on RefCOCO (a visual grounding task) and eval on RefCOCO Val and RefGTA (an OOD task). Result: VLM-R1 generalizes whereas the SFT baseline overfits, matching the observation ...</li><li><a href="https://x.com/littmath/status/1891868756340547809">Tweet from Daniel Litt (@littmath)</a>: Had a sort of a funny experience with OpenAI’s Deep Research tool, which I wanted to share since I think it reveals some of the tool’s strengths and weaknesses.</li><li><a href="https://x.com/xlr8harder/status/1892164213876961300">Tweet from xlr8harder (@xlr8harder)</a>: If I use Llama 3.3 70B Instruct output to finetune a Llama 3.1 8B base model, which community license applies?In this case:- Llama 3.1 insists you retain the Llama 3.1 Community License.- Llama 3.3 in...</li><li><a href="https://x.com/MichaelPoli6/status/1892244621490888718">Tweet from Michael Poli (@MichaelPoli6)</a>: [7/7]There is so much more to this project: Evo 2 demonstrates for the first test-time scaling laws in biology; Evo 2 is state-of-the-art performance on classifying BRCA1 variants of unknown significa...</li><li><a href="https://x.com/nearcyan/status/1891926678810607858">Tweet from near (@nearcyan)</a>: @thinkymachines is your x handle typod i dont know if thinky machines has the same vibe</li><li><a href="https://x.com/littmath/status/1891868775051342232">Tweet from Daniel Litt (@littmath)</a>: The only problem is that it’s all made up. Despite claiming to have looked at every paper published in the Annals in this 75-year period, poking around the pages it looked at suggests it only looked a...</li><li><a href="https://x.com/MichaelPoli6/status/1892242976942035029">Tweet from Michael Poli (@MichaelPoli6)</a>: [1/7] Introducing Evo 2, a new foundation model for biology.🚀 Evo 2 is the largest-scale, fully open-source AI model ever released: 40 billion parameters, over 9 trillion tokens, and a 1 million cont...</li><li><a href="https://x.com/TheXeophon/status/1891586946675216803">Tweet from Xeophon (@TheXeophon)</a>: 2.1M views on one vid, 127K monthly spotify listeners with AI-generated lofislowly, then all at onceQuoting Xeophon (@TheXeophon) Didn&#39;t realize I&#39;ve been listening to AI generated music until...</li><li><a href="https://x.com/littmath/status/1891868790314434809">Tweet from Daniel Litt (@littmath)</a>: In other words, it claimed again to produce a complete dataset but in fact only produced ~7 lines, with a placeholder for the other ~3000.</li><li><a href="https://youtu.be/Nc3vIuPyQQ0?si=KKt7VD5I521H95-W">AI CEO Alexandr Wang | This Past Weekend w/ Theo Von #563</a>: Alexandr Wang is the founder and CEO of Scale AI, a platform that provides data training for AI programs. In 2021 he was named the youngest self-made billion...</li><li><a href="https://www.youtube.com/watch?v=YXTYbr3hiFU">An Unexpected Reinforcement Learning Renaissance</a>: The era we are living through in language modeling research is one pervasive with complete faith that reasoning and new reinforcement learning (RL) training ...</li><li><a href="https://youtu.be/4poqjZlM8Lo?s">AI bosses on what keeps them up at night</a>: Google DeepMind and Anthropic founders, Demis Hassabis and Dario Amodei, are two of the world&#39;s foremost leaders in artificial intelligence. Our editor-in-ch...</li><li><a href="https://youtu.be/4poqjZlM8Lo?si=E6Y9rdAOYFjUeBhq)">AI bosses on what keeps them up at night</a>: Google DeepMind and Anthropic founders, Demis Hassabis and Dario Amodei, are two of the world&#39;s foremost leaders in artificial intelligence. Our editor-in-ch...</li><li><a href="https://x.com/zeffmax/status/1891970746596909091?s=46">Tweet from Max Zeff (@ZeffMax)</a>: Not sure why Dario Amodei and Demis Hassabis are sitting on a very small couch in this interview – while the Economist&#39;s EIC is on a very large couch by herself – but I like it
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1341612645723275287)** (3 messages): 

> `OpenAI's AI models, Useless Machine AI Demo, Open Research in AI` 


- **OpenAI's Models are the Best**: A member claimed that **OpenAI has the most based models**, indicating a strong belief in their superiority over others.
   - An attached image was shared, seemingly illustrating this point, although the content of the image was not detailed.
- **Request for AI Agent Useless Machine Demo**: A member asked for a demonstration of a **useless machine** that incorporates an AI agent, suggesting curiosity about its functionality.
   - There were no subsequent replies or confirmations about such a demonstration in the channel.
- **Commentary on AI Research Dynamics**: A link was shared that included commentary on the competitive landscape of AI, specifically mentioning **open research paths** taken by leading countries.
   - It was noted that **China** and **Google** are highlighted, with implications about their differing motivations and approaches in the AI research space.



**Link mentioned**: <a href="https://x.com/doomslide/status/1892311556991697009">Tweet from doomslide (@doomslide)</a>: it&#39;s quite telling that the two countries at the forefront of AI both follow the path of open research. china, with its ulterior motive to sabotage funding of san francisco&#39;s finest, and googl...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1341519869878734952)** (5 messages): 

> `Manual Decompilation of Older Games, RL Training for Decompilation, LLM4Decompile GitHub Project` 


- **Manual Decompilation Efforts Rise**: There are ongoing efforts to fully recompile older games by **manually decompiling assembly to C**, which involves a lot of guesswork to align with known facts and produce identical assembly.
   - This process is detailed on [Decomp.me](https://decomp.me/scratch/hFS6m), where enthusiasts try to match original code settings.
- **Seeking RL Solutions for Decompilation**: A member suggested the possibility of **RL-training a LLM** to automate the decompilation process, using match percentage as a reward.
   - They questioned whether this approach is unexplored territory and if anyone has tried matching assembly in a reinforcement learning context.
- **Awareness of SFT Usage for Decompilation**: Another member noted that while the idea of using large language models is appealing, current implementations like [LLM4Decompile](https://github.com/albertan017/LLM4Decompile) appear to only use **supervised fine-tuning (SFT)**.
   - They highlighted that, to their knowledge, no one has conducted reinforcement learning training focused on matching assembly code during decompilation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://decomp.me/scratch/hFS6m>)">decomp.me</a>: no description found</li><li><a href="https://github.com/albertan017/LLM4Decompile">GitHub - albertan017/LLM4Decompile: Reverse Engineering: Decompiling Binary Code with Large Language Models</a>: Reverse Engineering: Decompiling Binary Code with Large Language Models - albertan017/LLM4Decompile
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/)** (1 messages): 

the_real_jrb: https://arxiv.org/abs/2502.13923
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1341477416555642980)** (4 messages): 

> `Theory papers, Open Source AI concerns, Daniel Jeffries' perspective, Mixed feelings about authors` 


- **Skepticism Towards Theory Papers**: A member expressed *suspicions* about theory papers, indicating a general skepticism towards purely theoretical works.
   - This sentiment was met with some lightheartedness, suggesting an ongoing debate among theorists.
- **Defense of Open Source AI**: A discussion emerged around [Daniel Jeffries' article](https://danieljeffries.substack.com/p/defending-open-source-ai-against) where he argues that **open source** software would face severe criticism today from various societal factions.
   - He suggests that the current climate would lead to **tight control** over open source software, fueled by fears of potential dangers.
- **Concerns Over Regulatory Reactions**: Jeffries warned that lawmakers would label **open source** as a threat, calling for extensive control over its development.
   - This perspective highlights concerns about how current **societal fears** could stifle technological innovation.
- **Mixed Feelings About Jeffries' Work**: Another member shared mixed feelings regarding Jeffries' work and expressed a desire to discuss his views over coffee.
   - This raises questions about the reception of controversial ideas within the AI community.



**Link mentioned**: <a href="https://danieljeffries.substack.com/p/defending-open-source-ai-against">Defending Open Source AI Against the Monopolist, the Jingoist, the Doomer and the Idiot</a>: If Linux Were Just Getting Started Today, It Would Ge Crushed, and We&#x27;d All be a Lot Poorer for It. We Can&#x27;t Let that Happen to AI.

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1341505552924278855)** (13 messages🔥): 

> `Grok 3 Mini Announcement, Hacker News Comments, XAI Discussion Stress, Reasoning Models Benchmarks` 


- **Grok 3 Mini gets announced, not released**: A member pointed out that **Grok 3 Mini** has only been announced and is not yet released, referencing [this tweet](https://x.com/keirp1/status/1891955483251998984).
   - Another member confirmed that none of the reasoning models are released as per their knowledge.
- **Hacker News comments not worth reading**: A member remarked on the challenging experience of reading **Hacker News** comments when feeling unwell, calling it a rather unpleasant task.
   - They conveyed that they typically avoid engaging with those comments altogether.
- **XAI discussions seem daunting**: A member expressed reluctance to engage with the **XAI** community, finding the thought of replying to discussions stressful.
   - This sentiment indicates a broader concern about the pressures of participating in such conversations.
- **Curiosity about premium features**: A member joked about the existence of a 'deep thonk' button available for a premium price, questioning its utility.
   - This comment added a light-hearted touch to the conversation around the complexities of AI discussions.



**Link mentioned**: <a href="https://x.com/keirp1/status/1891955483251998984">Tweet from Keiran Paster (@keirp1)</a>: @natolambert @srush_nlp @TheShmanuel I think the mini reasoning model outperforming R1 is strong evidence against this narrative.

  

---


### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1341542219521200350)** (2 messages): 

> `Bicycle Stands, Image Sharing, Community Sentiment` 


- **Torters Celebrate Image Share**: Members rejoiced over an image shared, generating excitement in the community about bicycle-related content.
   - The shared image contributed to an engaging atmosphere, showcasing communal interest in biking accessories.
- **Bicycle Stands Drawing Attention**: A member noted that they own the same bicycle stand featured in the shared image, highlighting personal connection to the subject.
   - This comment sparked a brief discussion about the functionality and utility of bike stands among members.


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/)** (1 messages): 

gfabulous: Sigh, guess we're all using grok now
  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1341454244221550713)** (13 messages🔥): 

> `Prompt Builder Confusion, Comparative Queries for ODR, Limitations of PDR, Cursor Agent vs ODR, Vibecoding Workflow` 


- **Prompt Builder Confusion Unveiled**: A user clarified that the hidden prompt from the message text was actually the 'prompt builder' prompt, *not intended for queries*.
   - Discussions reflected a mix of humor and critique towards naming conventions in shared messages.
- **Comparative Queries Raise Curiosity**: One user expressed curiosity about running comparative queries between **ODR** and **Perplexity**, sarcastically noting the latter’s inferior performance.
   - *Laughter ensued* as the group acknowledged the limitations of existing query tools.
- **PDR's Shallow Insights Criticized**: A participant criticized **PDR** for being too superficial and listicle-like, highlighting its struggle to infer new insights from results.
   - This led to a consensus that PDR was only marginally better than a basic LLM combined with search.
- **Cursor Agent Triumphs with ODR**: A user shared their experience using **Cursor** to address a breaking change in a library, where the **ODR** output facilitated an easy fix.
   - The successful resolution demonstrated the efficacy of context utilization by the Cursor Agent.
- **Vibecoding Workflow Discussed**: The group playfully referred to their combination of tools as a 'supreme vibecoding workflow', implying a fun and effective coding process.
   - One user humorously referenced Karpathy, solidifying the playful tone around advanced coding techniques.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1341888997420433419)** (2 messages): 

> `Reasoning Tokens Default Behavior, Feedback on Token Limits, Polling for User Preferences` 


- **Debate over Default Reasoning Tokens**: Feedback indicates users prefer to receive content when **max_tokens** is low, prompting a review of the **include_reasoning** setting.
   - *Currently, include_reasoning is false by default*, leading to responses that can be empty or null.
- **Proposed Changes to Include Reasoning**: Two major changes are under consideration: setting **include_reasoning** to true by default and standardizing response formatting to ensure content is always a non-null string.
   - *The aim is to avoid empty responses* and allow reasoning tokens to be included unless specifically disabled.
- **Expanded Poll Options for User Input**: An expanded poll is available with four options regarding the **include_reasoning** behavior for community feedback.
   - Options range from keeping the current setting to making reasoning tokens default, with a supplementary option for user comments.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1341467447588552857)** (242 messages🔥🔥): 

> `Grok 3 performance, OpenRouter API usage, Chatbot integration, Perplexity R1 1776, AI model comparisons` 


- **Grok 3 Stumbles in Reasoning**: Users have reported finding **Grok 3** underwhelming in reasoning compared to **Claude 3.5** and **O1**, questioning its reputation as a top-tier LLM.
   - One user compared performance and found it worse than expected, suggesting the model may not live up to the hype.
- **Navigating OpenRouter API**: New users have expressed confusion about accessing **OpenRouter** and utilizing **O3 mini** via the API, especially regarding usage limitations.
   - The need for better integration options and clarity on API key usage has been emphasized, highlighting the gradual rollout of access to models.
- **Chatbot Integration on Websites**: A user inquired about the method to integrate an AI chatbot on an HTML website using **OpenRouter API**, revealing a gap in readily available resources for implementation.
   - Others suggested that they would need to develop the solution or hire a developer for assistance.
- **Introduction of Perplexity R1 1776**: The **R1 1776** model from **Perplexity** has been launched, providing users access to an uncensored version of the **DeepSeek R1** model.
   - This new model promises better response capabilities and has been noted for its competitive pricing structure for token usage.
- **AI Model Performance Insights**: Comparisons between AI models like **Claude** and **O3 mini** have been discussed, with users sharing mixed experiences regarding reasoning capabilities.
   - Some users noted that despite claims of superiority, newer models still struggle in specific tasks when compared to established ones.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/terms">no title found</a>: no description found</li><li><a href="https://openrouter.ai/google/gemini-">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/perplexity/r1-1776">R1 1776 - API, Providers, Stats</a>: Note: As this model does not return &lt;think&gt; tags, thoughts will be streamed by default directly to the `content` field.R1 1776 is a version of DeepSeek-R1 that has been post-trained to remove ce...</li><li><a href="https://openrouter.ai/google/gemini-2.0-pro-exp-02-05:free)">Gemini Pro 2.0 Experimental - API, Providers, Stats</a>: Gemini 2.0 Pro Experimental is a bleeding-edge version of the Gemini 2. Run Gemini Pro 2.0 Experimental with API</li><li><a href="https://huggingface.co/perplexity-ai/r1-1776/discussions">perplexity-ai/r1-1776 · Discussions</a>: no description found</li><li><a href="https://openrouter.ai/models">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://hst.sh/gewiwoqete.md">hastebin</a>: no description found</li><li><a href="https://x.com/perplexity_ai/status/1891916644248846789?t=7_5m7rcR2w7GFITF2I2QSA&s=19">Tweet from Perplexity (@perplexity_ai)</a>: Download the model weights on our HuggingFace Repo or consider using the model via our Sonar API.HuggingFace Repo: https://huggingface.co/perplexity-ai/r1-1776</li><li><a href="https://x.com/perplexity_ai/status/1892329089903841467?t=6lD3qXX2sOcKytYFI8L1kA&s=19">Tweet from Perplexity (@perplexity_ai)</a>: R1 1776 is now available via Perplexity&#39;s Sonar API.Quoting Perplexity (@perplexity_ai) Today we&#39;re open-sourcing R1 1776—a version of the DeepSeek R1 model that has been post-trained to provi...</li><li><a href="https://youtu.be/fwHkdivFCuc">An unfiltered conversation with Alex Atallah, CEO of OpenRouter</a>: Listen to a conversation with Alex Atallah, the CEO of OpenRouter, along with Nolan Fortman and Logan Kilpatrick. The conversation covers:- The start of Open...
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1341477017530404948)** (200 messages🔥🔥): 

> `AI Image Generation Tools, Stable Diffusion vs. Flux, ControlNet Functionality, UI Options for Image Generation, Installation Guides for AI Tools` 


- **Exploring AI Image Generation Tools**: A user expressed interest in learning beyond Midjourney and found Stable Diffusion (SD) more fleshed out than Flux, but sought guidance on which to learn.
   - The community suggested checking example images and choosing based on personal preferences, while also considering hardware specifications.
- **ControlNet Enhances Image Generation**: ControlNet enables users to use a pose as a reference by turning it into a skeletal figure, which improves generated image accuracy.
   - Users noted that both SD and Flux work well with ControlNet for generating specific poses.
- **Hardware Considerations for AI Tools**: A user with an RTX 3080 GPU was informed they could run both Stable Diffusion 3.5 and Flux effectively.
   - The community highlighted that the choice between SD 3.5 and XL depends on user needs, but newer models offer better features.
- **Choosing User Interfaces for AI Tools**: ComfyUI is noted for its complexity and mindmap-like interface, making it less friendly for casual users.
   - Alternatives like SwarmUI and other one-stop-shop solutions were recommended for easier access to AI tools.
- **Installation Resources for AI Generation Tools**: Users were directed to various installation guides for SwarmUI, CS1o tutorials, and Lykos Stability Matrix for setting up AI tools.
   - These resources provide detailed steps and assist users in navigating the installation process for different interfaces.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/DOFOFFICIAL/animeGender-dvgg-0.8">DOFOFFICIAL/animeGender-dvgg-0.8 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/jiseok-kim-jiseok-big-ocean-bigocean-kpop-gif-16919206117458777151">Jiseok Kim Jiseok GIF - Jiseok Kim jiseok Big ocean - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=n233GPgOHJg">Stable Diffusion Models Explained Once and for All (1.5, 2, XL, Cascade, 3)</a>: In this video, I explain the 5 different model families of Stable Diffusion.Did I get anything wrong or leave something out? Let me know.Chapters:00:00 Intro...</li><li><a href="https://github.com/LykosAI/StabilityMatrix/">GitHub - LykosAI/StabilityMatrix: Multi-Platform Package Manager for Stable Diffusion</a>: Multi-Platform Package Manager for Stable Diffusion - LykosAI/StabilityMatrix</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases">Releases · AUTOMATIC1111/stable-diffusion-webui</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI?tab=readme-ov-file#installing-on-windows">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1341892739289321583)** (1 messages): 

> `Torchtune roadmap, PyTorch developments` 


- **Torchtune roadmap unveiled**: The **Torchtune roadmap** for the first half of the year has officially been posted on the [PyTorch dev-discuss](https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view).
   - This document lays out key objectives and timelines, offering insights into what's ahead for Torchtune.
- **Exciting new work across PyTorch**: In addition to Torchtune, there is a **ton of cool work** going on across the entire PyTorch organization this half of the year. Find the full set of [PyTorch roadmaps here](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794).
   - These roadmaps encapsulate ongoing projects and priorities, illustrating the vibrant ecosystem around PyTorch.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view">[PUBLIC] Torchtune - H1 2025 Roadmap.pdf</a>: no description found</li><li><a href="https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794">Meta PyTorch Team 2025 H1 Roadmaps</a>: PyTorch Community,  The Meta team are happy to make our 2025 H1 roadmaps available. We plan on a half year basis and globally optimize across the things we do for our users here at Meta and across the...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1341454925342707854)** (141 messages🔥🔥): 

> `Torchtune Roadmap, Packed vs Unpacked Tokenization, Fine-tuning Llama Models, Attention Mechanisms, Pruning Techniques` 


- **Torchtune Roadmap Released**: The Torchtune team shared their roadmap on PyTorch dev-discuss, highlighting upcoming features and priorities.
   - They aim to stay ahead of new model architectures while also addressing existing core work first.
- **Impact of Packing on VRAM Usage**: Users observed that using packing with large sequence lengths can drastically increase VRAM requirements during training.
   - Testing showed variable memory allocation behavior, prompting further exploration of kernel differences and settings.
- **Challenges with Fine-tuning 3B Llama Models**: Users reported issues with the 3B Llama models speaking nonsense during inference after fine-tuning, particularly contrasting results with the 8B variant.
   - The discussion moved towards potential bugs in the model export and checkpointing process in Torchtune affecting the 3B model.
- **Exploring Exotic Attention Mechanisms**: There is an interest in integrating more advanced attention mechanisms like sparse and compressed attention into Torchtune for better efficiency.
   - The team is open to ideas that leverage these techniques, while focusing primarily on leveraging PyTorch core functionalities.
- **Future Directions in Pruning Techniques**: The roadmap discussed setting the groundwork for incorporating pruning techniques into model training processes.
   - There is a strategy in place to develop recipes supporting both width and depth pruning combined with knowledge distillation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/tutorials/qlora_finetune.html#qlora-compile-label">Fine-Tuning Llama2 with QLoRA &mdash; torchtune main documentation</a>: no description found</li><li><a href="https://arxiv.org/abs/2407.14679">Compact Language Models via Pruning and Knowledge Distillation</a>: Large language models (LLMs) targeting different deployment scales and sizes are currently produced by training each variant from scratch; this is extremely compute-intensive. In this paper, we invest...</li><li><a href="https://pytorch.org/torchtune/stable/tune_cli.html">torchtune CLI &mdash; torchtune 0.5 documentation</a>: no description found</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html#generate-some-output">End-to-End Workflow with torchtune &mdash; torchtune 0.5 documentation</a>: no description found</li><li><a href="https://x.com/intervitens/status/1891997689128120397">Tweet from intervitens (@intervitens)</a>: @samsja19 Torchtune gets close. No fa3, but has flex attention or cuDNN via Pytorch SDPA. Not fully compatible with HF checkpoints, can use the same weights, but config and tokenizer require manual su...</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks#llama-3.1-8b-max.-context-length">Unsloth Benchmarks | Unsloth Documentation</a>: Want to know how fast Unsloth is?</li><li><a href="https://github.com/ianbarber/ttblt">GitHub - ianbarber/ttblt: A simplified implementation of Byte Latent Transformers as a TorchTune recipe.</a>: A simplified implementation of Byte Latent Transformers as a TorchTune recipe. - ianbarber/ttblt</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/generation.yaml">torchtune/recipes/configs/generation.yaml at main · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/issues/2392">`torch._inductor.exc.LoweringException: NoValidChoicesError` using torch 2.6.0 · Issue #2392 · pytorch/torchtune</a>: Error [rank0]: raise NoValidChoicesError( [rank0]: torch._inductor.exc.LoweringException: NoValidChoicesError: No choices to select, please consider adding ATEN into max_autotune_gemm_backends conf...</li><li><a href="https://github.com/MaxHastings/Kolo/blob/main/scripts/merge_lora.py">Kolo/scripts/merge_lora.py at main · MaxHastings/Kolo</a>: A one stop shop for data generation, fine tuning and testing LLMs locally using the best tools available. Keeping it simple and versatile! - MaxHastings/Kolo</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba2532d51a53936c7646bd4cdaa6b2b57ed66/">GitHub - pytorch/torchtune at e6cba2532d51a53936c7646bd4cdaa6b2b57ed66</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_2/3B_qlora_single_device.yaml">torchtune/recipes/configs/llama3_2/3B_qlora_single_device.yaml at main · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba25">GitHub - pytorch/torchtune at e6cba2532d51a53936c7646bd4cdaa6b2b57ed66</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune#optimization-flags">GitHub - pytorch/torchtune: PyTorch native post-training library</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba2532d51a53936c7646bd4cdaa6b2b57ed66/torchtune/modules/attention_utils.py#L35">torchtune/torchtune/modules/attention_utils.py at e6cba2532d51a53936c7646bd4cdaa6b2b57ed66 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/issues/137901">Any plan to support flash attention 3 for hopper GPUs? · Issue #137901 · pytorch/pytorch</a>: 🚀 The feature, motivation and pitch Flash Attention 3 (https://github.com/Dao-AILab/flash-attention) has been in beta for some time. I tested it on H100 GPUs with CUDA 12.3 and also attempted a sim.....</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba2532d51a53936c7646bd4cdaa6b2b57ed66/recipes/full_finetune_distributed.py#L378">torchtune/recipes/full_finetune_distributed.py at e6cba2532d51a53936c7646bd4cdaa6b2b57ed66 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1341528701359231077)** (32 messages🔥): 

> `Step-based Checkpointing, PPO Performance Boost, Integration with Gymnasium, Intercode Interface for LLMs` 


- **Step-based Checkpointing Implementation Discussion**: The team is discussing new loading mechanisms due to the introduction of step-based checkpointing, with options to resume from the latest or a specific checkpoint being proposed.
   - There is a push for adding features to the existing PRs, ensuring better handling of multi-experiment scenarios to accommodate checkpointing needs.
- **Performance Boost in PPO Recipe**: A recent PR introduces performance improvements in the PPO recipe, focusing on better configuration for KV caching.
   - Feedback highlights efficiency gains from using the latest optimizers and discusses potential integration compatibility with existing systems.
- **Concerns About Integration with Gymnasium**: A member raised concerns over the viability of integrating Torchtune with Gymnasium due to intrinsic differences in design for LLM tasks.
   - Alternative interfaces such as Intercode, which are tailored for coding tasks, were discussed as potentially more fitting for LLM training.
- **Future of Intercode Integration**: Discussions about Intercode suggest that while it has suitable environments for coding and SQL tasks, it may not align with stable OSS project standards.
   - Members expressed interest in sharing findings on how well Intercode interfaces with LLMs and the need for more environments tailored for post-training tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ysymyth.github.io)">no title found</a>: no description found</li><li><a href="https://pytorch.org/torchtune/main/deep_dives/checkpointer.html#resuming-from-checkpoint-full-finetuning">Checkpointing in torchtune &mdash; torchtune main documentation</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/issues/2413">[RFC] Judge Framework and Online DPO · Issue #2413 · pytorch/torchtune</a>: TRL has a concept of multiple different judges which can be used in various online RLHF type methods, see the TRL Judges Doc. As a starting point, we could implement just a pairwise judge that coul...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_1/8B_full_dpo.yaml#L64-L72">torchtune/recipes/configs/llama3_1/8B_full_dpo.yaml at main · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/2412">Update KVCache maximum sequence length configuration in PPO recipe by SalmanMohammadi · Pull Request #2412 · pytorch/torchtune</a>: Bug was raised in #2064. When setting up KV caches, we use tokenizer.max_seq_len to determine the shape of the KV cache. As this was not configured in the config, we would error out.The direct fix...</li><li><a href="https://github.com/joecummings/torchtune/pull/2">feat: get_latest_checkpoint for checkpointer utils by bogdansalyp · Pull Request #2 · joecummings/torchtune</a>: Added get_latest_checkpoint    &amp;quot;&amp;quot;&amp;quot;    Returns the latest checkpoint in the given directory.    The pattern argument is a regular expression that matches the epoch number in ...</li><li><a href="https://github.com/princeton-nlp/intercode">GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&amp;B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898</a>: [NeurIPS 2023 D&amp;B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898 - princeton-nlp/intercode</li><li><a href="https://github.com/pytorch/torchtune/pull/2105">[RFC] Step-based checkpointing in torchtune by joecummings · Pull Request #2105 · pytorch/torchtune</a>: Enabling step-based checkpointing in torchtuneOriginal context: #2070What are we currently doing?We currently only checkpoint at epoch boundaries. That means a fine-tuning run has to iterate thr...</li><li><a href="https://github.com/pytorch/torchtune/pull/24">Add KV Cache for Transformers by joecummings · Pull Request #24 · pytorch/torchtune</a>: Summary:Add key-value caching for transformer. Also refactored some of the attention code to be simpler and added a test for torch compilation.Testing:pytest tests/llm/llama2/test_transformer.py
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1341848371287162951)** (2 messages): 

> `Multi-step PPO, Tool Use Learning, Reward Shaping, StepTool Framework` 


- **Exploration of multi-step PPO papers**: A user inquired about any papers on **multi-step PPO**, which entails multiple sequential calls to the LLM with tool results, where rewards are determined after several steps.
   - Suggestions were made to investigate papers pertaining to **learning better tool use** and **reward shaping**, indicating a rich domain of research may exist.
- **Introduction to StepTool framework**: [This paper](https://arxiv.org/abs/2410.07745) proposes **StepTool**, a novel reinforcement learning framework that enhances multi-step tool usage for LLMs by treating tool learning as a **dynamic decision-making** task.
   - It introduces components like **Step-grained Reward Shaping** that offers rewards during tool interactions based on their success and contribution to the task, aiming to optimize the model's policy in a multi-step manner.



**Link mentioned**: <a href="https://arxiv.org/abs/2410.07745">StepTool: Enhancing Multi-Step Tool Usage in LLMs through Step-Grained Reinforcement Learning</a>: Despite powerful text generation capabilities, large language models (LLMs) still need to learn how to utilize external tools to solve complex tasks, a process known as tool learning. Existing methods...

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1341458134157496390)** (145 messages🔥🔥): 

> `Grok-3, Le Chat, AI Rendering, Dynamic NPCs, LLMs and Game Development` 


- **Grok-3 API discussions**: Members discussed the absence of an API for **Grok-3**, limiting its use and the need for independent benchmarks prior to testing it.
   - *It seems that Grok-3 currently obfuscates content*, leading to ongoing comments about its integration and potential applications.
- **Le Chat impresses users**: Users shared positive experiences with **Le Chat**, emphasizing its superior speed, image generation capabilities, and quality compared to other models.
   - French residents have a unique opportunity for a **low-cost subscription**, which has sparked interest and excitement in the community.
- **AI rendering's future in gaming**: Discussion revolved around the potential of **AI rendering** to revolutionize gaming by creating dynamic and interactive environments.
   - Members expressed concerns about the balance between AI-generated content and traditional game development, emphasizing the need for meaningful engagement.
- **NPC interactions enhanced by AI**: There was speculation about how AI could generate unique NPCs and quests tailored to player interests, enhancing gameplay experiences.
   - Some believe this approach could lead to more engaging and immersive interactions compared to current game structures.
- **Insights on LLMs in game development**: Recent studies highlighted similarities between **LLMs** and human brains, indicating new ways LLMs can process diverse data like visual inputs.
   - This paves the way for more innovative applications in game development, especially with increasingly complex interactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/M1Astra/status/1892091124589920532">Tweet from M1 (@M1Astra)</a>: BREAKING: Anthropic set to release Thinking Model/Mode & Web Search capabilities imminently, based on findings in the latest update to the Claude iOS app.New symbols like &#34;Steps&#34; & &#34;Think&...</li><li><a href="https://huggingface.co/microsoft/wham">microsoft/wham · Hugging Face</a>: no description found</li><li><a href="https://news.mit.edu/2025/large-language-models-reason-about-diverse-data-general-way-0219">Like human brains, large language models reason about diverse data in a general way</a>: MIT researchers find large language models process diverse types of data, like different languages, audio inputs, images, etc., similarly to how humans reason about complex problems. Like humans, LLMs...</li><li><a href="https://chat.mistral.ai/chat">Le Chat - Mistral AI</a>: Chat with Mistral AI&#x27;s cutting edge language models.</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview/discussions/5#67b390336de7ff0b4c84b16d">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Honnestly. It&#39;s garbage.</a>: no description found</li><li><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3143225812&searchtext=">Steam Workshop::Reforged Eden 2 Beta</a>: no description found</li><li><a href="https://fxtwitter.com/satyanadella/status/1892244164814725387">Tweet from Satya Nadella (@satyanadella)</a>: If you thought AI-generated text, images, and video were cool, just imagine entire interactive environments like games!</li><li><a href="https://github.com/DamascusGit/err_err">GitHub - DamascusGit/err_err</a>: Contribute to DamascusGit/err_err development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1itdy0k/no_system_instructions_for_deepseek_makes_jake/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1341485401260425319)** (2 messages): 

> `SWE-Lancer Benchmark, MoBA Project` 


- **SWE-Lancer Benchmark Launches**: Introducing [SWE-Lancer](https://arxiv.org/abs/2502.12115), a benchmark comprising over **1,400 freelance software engineering tasks** from Upwork, valued at **$1 million USD**.
   - Tasks range from **$50 bug fixes** to **$32,000 feature implementations**, but findings show that frontier models struggle to solve the majority of tasks.
- **MoBA: Advancing Long-Context LLMs**: [MoBA](https://github.com/MoonshotAI/MoBA) by MoonshotAI presents a **Mixture of Block Attention** technique aimed at improving performance in **long-context LLMs**.
   - The project is aimed at addressing limitations in handling lengthy input effectively, and can be found on GitHub for more insights.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.12115">SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?</a>: We introduce SWE-Lancer, a benchmark of over 1,400 freelance software engineering tasks from Upwork, valued at \$1 million USD total in real-world payouts. SWE-Lancer encompasses both independent engi...</li><li><a href="https://github.com/MoonshotAI/MoBA">GitHub - MoonshotAI/MoBA: MoBA: Mixture of Block Attention for Long-Context LLMs</a>: MoBA: Mixture of Block Attention for Long-Context LLMs - MoonshotAI/MoBA
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1341485401260425319)** (2 messages): 

> `SWE-Lancer Benchmark, MoBA Model` 


- **Introducing the SWE-Lancer Benchmark**: The **SWE-Lancer** benchmark features over 1,400 freelance software engineering tasks valued at a total of **$1 million USD**, sourced from Upwork. This benchmark includes independent tasks graded by experienced engineers and aims to assess and improve model performance in real-world software engineering scenarios, available via [SWE-Lancer Diamond](https://github.com/openai/SWELancer-Benchmark).
   - Despite the extensive dataset, evaluation shows **frontier models** struggle to solve most tasks effectively.
- **MoBA for Long-Context LLMs**: [MoBA](https://github.com/MoonshotAI/MoBA) introduces a **Mixture of Block Attention** method tailored for long-context large language models, aiming to enhance processing capabilities. This project is expected to advance research in the efficiency of handling longer contexts in transformer models.
   - The GitHub repository for MoBA provides insights and frameworks for developers tackling long-context challenges in LLM applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.12115">SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?</a>: We introduce SWE-Lancer, a benchmark of over 1,400 freelance software engineering tasks from Upwork, valued at \$1 million USD total in real-world payouts. SWE-Lancer encompasses both independent engi...</li><li><a href="https://github.com/MoonshotAI/MoBA">GitHub - MoonshotAI/MoBA: MoBA: Mixture of Block Attention for Long-Context LLMs</a>: MoBA: Mixture of Block Attention for Long-Context LLMs - MoonshotAI/MoBA
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1341472342567813181)** (12 messages🔥): 

> `Using Notebook LM for Book Summaries, Feedback on Audio Discussion Features, Prompting for TTS in Podcasts, Notebook Access for Non-Google Users` 


- **Notebook LM excels at teaching books**: A user expressed confidence that Notebook LM can teach them a book effectively, just by providing more precise prompts to avoid skipping key parts.
   - *Hope it helps.* emphasizes the potential benefits of guided prompting.
- **User feedback on Audio Discussion**: Another user shared that while they appreciate how Notebook LM aids in organizing responses for their assignment, the audio discussion became hard to follow due to back-and-forth interruptions between voices.
   - They suggested a more structured audio approach, where one voice presents a complete idea before the other interjects.
- **Challenges with Podcast TTS Prompts**: A user inquired about TTS prompts for the podcast feature, noting that their attempts to have the host read the text word for word were unsuccessful.
   - *What prompt did you use, verbatim?* illustrates their search for effective solutions.
- **Notebook LM Access Limitations**: A user asked whether it's possible to invite someone without a Google account to access a Notebook, similar to Google Docs sharing.
   - This highlights ongoing questions about access and collaboration features within Notebook LM.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1341461940299632742)** (124 messages🔥🔥): 

> `NotebookLM Plus Features, User Experience Challenges, Language Settings, Sharing Limitations, Integration with Other Platforms` 


- **NotebookLM Plus Limitations on Usage**: Users discussed the daily limits of NotebookLM Plus, noting that the 500 chat queries are across all notebooks, not per individual notebook.
   - Sharing is capped at **50 users** across all notebooks, with suggestions to create multiple accounts for increased limits.
- **User Experience Concerns**: Concerns were raised regarding the user experience for researchers, particularly about the renaming of uploaded PDF files and the potential confusion this causes.
   - Suggestions were made for features like displaying original URLs directly in the source list for easy access to references.
- **Language Settings for Users**: A medical student inquired about changing the NotebookLM response language from Arabic to English, finding it challenging to switch.
   - It was clarified that changing the URL parameters can address this issue by replacing 'ar' with 'en'.
- **Integration and Comparison with Other Tools**: Users compared NotebookLM with other AI tools like ChatGPT and DeepSeek, raising questions about its unique features beyond podcast capabilities.
   - Discussion included potential future features such as video capabilities and how direct sourcing might improve functionality.
- **Future Availability and Access**: Questions arose about the availability of NotebookLM Plus for non-profits, with mixed messages from Google representatives regarding general accessibility.
   - Users expressed a desire for clarification from Google employees about current and future features of NotebookLM.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com/?hl=ar">تسجيل الدخول - حسابات Google</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219?visit_id=638755661154801012-3742783720&p=plus&rd=1">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://drive.google.com/file/d/1vc0a0FzJSXkPhpuuwB7e1twQpB9u7USd/view?usp=sharing">Snipaste_2025-02-19_15-23-18.png</a>: no description found</li><li><a href="https://blog.google/feed/notebooklm-google-one">NotebookLM Plus is now available in the Google One AI Premium subscription.</a>: NotebookLM is a research and thinking companion designed to help you make the most of your information. You can upload material, summarize it, ask questions and transfor…</li><li><a href="https://www.webwizwork.com/author/admin/">Balázs Piller - WebWizWork</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1341977154145222717)** (6 messages): 

> `GPU Spec Spreadsheet, Studying GPU Architecture, Computer Architecture Books, Snapdragon/Adreno GPU Computing` 


- **Looking for Definitive GPU Spec Spreadsheet**: A member expressed their frustration over not finding a **definitive spreadsheet** for GPU specifications similar to one [published on Google Sheets](https://docs.google.com/spreadsheets/d/e/2PACX-1vSdXHeEqyabPZTgqFPQ-JMf-nogOR-qaHSzZGELH7uNU_FixVDDQQuwmhZZbriNoqdJ6UsSHlyHX89F/pubhtml). They hope to discover a more satisfying source for their GPU spec searches.
- **Resources for Learning GPU Architecture**: A member, new to the GPU field, seeks **learning resources** for understanding GPU architecture, especially as it relates to their focus on CUDA software abstractions. They shared a link to a relevant [Springer book](https://link.springer.com/book/10.1007/978-3-031-01759-9) that they've begun reading.
- **Interest in Computer Architecture Books**: Another member joined the discussion expressing curiosity about **good computer architecture books**. They are seeking recommendations to complement their GPU studies.
- **Need for Snapdragon/Adreno GPU Channel**: A member proposed creating a channel focused on the **Snapdragon/Adreno GPU computing platform** after acquiring a Windows on ARM Snapdragon laptop. They are eager to explore GPU computations using OpenCL/Vulkan and wondered if others share this interest.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://link.springer.com/book/10.1007/978-3-031-01759-9">General-Purpose Graphics Processor Architectures</a>: no description found</li><li><a href="https://docs.google.com/spreadsheets/d/e/2PACX-1vSdXHeEqyabPZTgqFPQ-JMf-nogOR-qaHSzZGELH7uNU_FixVDDQQuwmhZZbriNoqdJ6UsSHlyHX89F/pubhtml#">GPU_Compare</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1341747913226719295)** (2 messages): 

> `BLOCK_SIZE Recommendations, Matrix Dimensions, GEMM vs splitK Performance` 


- **BLOCK_SIZE Recommendations for Large Matrices**: A member proposed using **2^18** for **BLOCK_M**, **BLOCK_N**, and **BLOCK_K** based on matrix sizes of **[228075, 512]** and **[512, 512]**.
   - Another member responded that BLOCK_SIZE should be smaller or equal to matrix dimensions, suggesting values like **BLOCK_M=[64,128]** and **BLOCK_K=[32,64,128,256]** for better performance.
- **Understanding GEMM vs splitK Performance**: The user questioned whether **GEMM** would be faster than **splitK** for their matrices, prompting a discussion on performance comparisons.
   - The response suggested focusing on tuning BLOCK_SIZE instead of assuming larger sizes would automatically yield better results.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1341458250826121357)** (14 messages🔥): 

> `cudaMemcpyAsync and cudaMemcpyDeviceToDevice, CUDA Express Installer Issues, Proposal for Raw-Dogged Tensor, Visual Studio and CUDA Installation Troubles` 


- **Need for Fine-Grained Control in cudaMemcpy**: A user expressed the need for **fine-grained control** in data copying, specifically mentioning the use of a for loop with **cudaMemcpyDeviceToDevice**.
   - Another member suggested using [`cub::DeviceCopy::Batched`](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceCopy.html) as a potential solution.
- **Challenges with CUDA Express Installer on Windows**: Several users reported issues with the **CUDA Express Installer** getting stuck on critical components like **Nsight** and **Visual Studio**.
   - One user noted that multiple friends with different NVIDIA GPUs experienced the same installation problems.
- **Introducing Raw-Dogged Tensor into CUDA Nomenclature**: A user proposed a new term, **'raw-dogged Tensor'**, for the next version of cute/cutlass, enhancing compatibility between storage formats and thread layouts.
   - Another member shared their experience with an **int8 matmul** that highlighted shared memory bank conflicts when not using specific preprocessing techniques.
- **Visual Studio and CUDA Installation Challenges**: A user detailed their struggles with **Visual Studio** during installation attempts with **CUDA 12.5 and 12.8**, citing a common issue among peers.
   - Concerns were raised about the ease of installation on **Linux** compared to **Windows**, reflecting broader frustrations with the operating system.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1341756702109863937)** (2 messages): 

> `GoLU Activation Function, Compilation Performance, File Splitting, MRE for Torch Forum` 


- **GoLU Activation Function Shared**: A member shared a link to the [GoLU](https://github.com/automl/GoLU/tree/main/golu) repository, which features a novel, self-gated, and element-wise activation function that performs well across diverse tasks.
   - The GitHub page includes an **image** preview and a description highlighting its effectiveness.
- **Slow Compilation Concerns**: Another member expressed frustration over experiencing slow compilation times despite having a similar setup to that described in the GoLU repository.
   - They indicated plans to experiment with **file splitting** and potentially create a Minimum Reproducible Example (MRE) to present on a **torch forum**.



**Link mentioned**: <a href="https://github.com/automl/GoLU/tree/main/golu">GoLU/golu at main · automl/GoLU</a>: GoLU, a novel, self-gated and element-wise activation function that performs well over a diverse set of tasks - automl/GoLU

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

andreaskoepf: DS is ruling the field at the moment: https://arxiv.org/abs/2502.11089
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1341525603475525742)** (4 messages): 

> `GoLU Activation Function, UltraScale Playbook Release, AI CUDA Engineer Optimization, CUDA Kernel Discovery` 


- **Introducing GoLU: A New Self-Gated Activation Function**: A recent paper presented the **Gompertz Linear Unit (GoLU)**, a new activation function designed to reduce variance in the latent space while maintaining gradient flow, outperforming **GELU** and **Swish**.
   - The GoLU function is defined as $\mathrm{GoLU}(x) = x \, \mathrm{Gompertz}(x)$, aiding in the **training dynamics** of deep learning models.
- **Nanotron Drops the UltraScale Playbook**: [Check out the latest blog post](https://huggingface.co/spaces/nanotron/ultrascale-playbook) from **Nanotron**, showcasing groundbreaking insights into scaling AI models.
   - The post highlights innovative strategies and methodologies for **AI model optimization** and deployment.
- **AI CUDA Engineer Automates CUDA Kernel Creation**: The **AI CUDA Engineer** framework automates the creation of optimized CUDA kernels, achieving a **10-100x speedup** over traditional machine learning operations in **PyTorch**.
   - With a success rate over **90%** for translating PyTorch to CUDA, this system dramatically improves performance compared to existing solutions.
- **Revolutionizing CUDA Kernel Optimization**: The AI CUDA Engineer can outperform both **native torch** and **torch compile** optimizations, with a **75%** and **60%** success rate, respectively.
   - Specific operations, such as **Instance Normalization**, show the potential of this system for performance-enhancing transformations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sakana.ai/ai-cuda-engineer/">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://arxiv.org/abs/2502.03654v1">Gompertz Linear Units: Leveraging Asymmetry for Enhanced Learning Dynamics</a>: Activation functions are fundamental elements of deep learning architectures as they significantly influence training dynamics. ReLU, while widely used, is prone to the dying neuron problem, which has...</li><li><a href="https://pub.sakana.ai/ai-cuda-engineer/">The AI CUDA Engineer 👷</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1341756853742473266)** (10 messages🔥): 

> `torchao tutorial issues, HuggingFace quantization problems, LLaMa 2 integration, Key naming conflicts in models, Fix for past_key_value bug` 


- **torchao tutorials face issues**: A member inquired about tutorials using HuggingFace models for quantization that require data calibration, particularly referencing **GPTQ**.
   - They noted that the code is tested on **LLaMa 2 7B** but seem confused about the model requirements pertaining to the `.pth` file.
- **HuggingFace GPU compilation issue**: A user reported encountering a **torch.compile error** when running the HuggingFace example for quantization.
   - They linked to a [GitHub issue](https://github.com/pytorch/ao/issues/1705) that captures the error context and version information.
- **Not a torchao bug but a code conflict**: It was clarified that the issue is not a bug in **torchao**, mentioning that it occurs even without quantization.
   - Participants pointed out a naming conflict where **past_key_values** and **past_key_value** are used interchangeably in the HuggingFace implementation.
- **Fix proposed for key conflict**: A member shared a [pull request](https://github.com/huggingface/transformers/pull/36289) to address the **key naming conflict**, emphasizing that both keys need to be correctly managed.
   - The fix specifically aims to ensure the LLaMa model handles **past_key_value** appropriately during device placement to avoid confusion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/issues/1705">torch.compile error when running the HuggingFace torchao example · Issue #1705 · pytorch/ao</a>: When I run the code snippet from https://huggingface.co/docs/transformers/main/en/quantization/torchao, I see a torch.compile error. torch version: 2.6.0 torchao version: 0.8.0 transformers version...</li><li><a href="https://github.com/huggingface/transformers/pull/36289">[bugfix] Update modeling_llama.py so it skips keys correctly by HDCharles · Pull Request #36289 · huggingface/transformers</a>: the llama model was using past_key_value and past_key_values interchangeably which caused issues because only one of those was actually skipped in _skip_keys_device_placement when both needed to be...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1341506159663910994)** (2 messages): 

> `ROCm Application Developer Certificate, AI Copyright & National Security` 


- **AMD Launches ROCm Developer Certification**: AMD has introduced a [ROCm Application Developer Certificate](https://www.amd.com/en/products/software/rocm/application-developer-certificate.html) aimed at enhancing skills in GPU computing and promoting the ROCm ecosystem.
   - This certification is a significant step for developers looking to specialize in **open-source GPU computing technologies**.
- **AI Copyright Debate Heats Up**: Discussion around AI copyright sparked from a blog post claiming that Chinese LLMs, including **DeepSeek**, are trained on an extensive collection of illegally archived materials, highlighting issues of **national security**.
   - The blog addresses the current state of 'shadow-libraries', noting the [illegality and potential risks](https://annas-archive.org/blog/ai-copyright.html) associated with materials used for AI training in contrast to the legislative gaps that need urgent overhaul.



**Link mentioned**: <a href="https://annas-archive.org/blog/ai-copyright.html">Copyright reform is necessary for national security</a>: Chinese LLMs (including DeepSeek) are trained on my illegal archive of books and papers — the largest in the world. The West needs to overhaul copyright law as a matter of national security.

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

kpk1340: Anyone in NYC?
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1341939234327494676)** (5 messages): 

> `Mi50 hardware matmul support, Mi50 specifications, Tensor operations on GPUs` 


- **Confusion on Mi50 Matmul Capabilities**: A discussion arose regarding the **MI50** and its lack of **hardware matmul support**, with uncertainty whether it can handle tensor operations.
   - One member clarified that the **WMMA** (Wave Matrix Multiply Accumulate) feature is present in later cards, but its application on the Mi50 remains uncertain.
- **Mi50 Offers Robust Specs**: The **Mi50** supports multiple data types ranging from **FP64 to FP16**, delivering impressive **performance** with a doubling of FLOPS.
   - Its specs include **26.5 TFLOPs** of FP16 performance and a memory bandwidth of up to **1024 GB/s**, showcasing its capabilities in computational tasks.



**Link mentioned**: <a href="https://www.8anet.com/Product/17823/AMD-100-506143-Radeon-Instinct-MI50-Accelerator-PCIe-4-0-x16-32GB-HBM2-4096-bit-3840-Stream-Processors-Passive-Cooling">
	8ANET - AMD 100-506143 Radeon Instinct™ MI50 Accelerator PCIe 4.0 x16 32GB HBM2 4096-bit 3840 Stream Processors Passive Cooling
</a>: no description found

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1341903496987017267)** (1 messages): 

> `Convergence Test Bug, PR Merging Process` 


- **Convergence Test Bug Fixed with Logit Scaling**: A member reported fixing the **convergence test** issue by addressing a missing **logit scaling parameter** in the **MiniModelConfig**, which corrected the magnitude of the logits.
   - This fix is expected to enhance the overall performance of the model.
- **Need for PR Merge Assistance**: The member expressed eagerness to move forward with the merging of their **pull request** and offered assistance in any way needed to expedite the process.
   - They are **very happy** to collaborate on getting the PR merged as soon as possible.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1341788560612917260)** (2 messages): 

> `Kokoro TTS optimizations, Low bit training techniques, GPU performance improvements` 


- **Kokoro TTS speeds up with Triton & CUDA**: An effort to improve inference speed for **Kokoro**, a modern TTS model, highlighted the use of **Triton + CUDA graph** to reduce kernel launch overhead of LSTM at bs=1.
   - *I can't believe I had to optimize LSTM in this day and age* 😂 and measurements were taken on a **4070Ti SUPER**.
- **YouTube Insight on Low Bit Training**: A [YouTube video](https://m.youtube.com/watch?v=leCY8vCUS4g) titled 'Thien Tran - Low bit training' discusses the use of **low-bit datatypes** on modern GPUs to accelerate training.
   - The speaker, **Thien Tran**, is a Machine Learning Engineer based in Singapore who delves into practical techniques for performance enhancement.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://m.youtube.com/watch?v=leCY8vCUS4g">Thien Tran - Low bit training</a>: In this talk, we will explore the use of low-bit datatypes on modern GPUs to accelerate training.Thien is a Machine Learning Engineer based in Singapore. He ...</li><li><a href="https://x.com/gaunernst/status/1892227983072518503">Tweet from Thien Tran (@gaunernst)</a>: Kokoro is a recent TTS that sounds pretty good while being quite fast. But I wonder, can I make it faster?Some perf optimizations for Kokoro inference @ bs=1 below.All timings were measured on 4070Ti ...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1341455648889638912)** (7 messages): 

> `AI CUDA Engineer, Kernel Optimization, Evolutionary Approach to CUDA, Innovation Archive, LLMs in CUDA Development` 


- **AI CUDA Engineer automates CUDA kernel production**: Sakana AI introduced the [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) which produces optimized CUDA kernels, achieving **10-100x speedup** over common PyTorch operations.
   - Their recent paper discusses this system's ability to autonomously optimize runtime performance and discover efficient CUDA operations based on past learning.
- **Kernel optimization using evolutionary techniques**: The process involves translating important PyTorch ops to CUDA, followed by optimizing these kernels using an **evolutionary approach** that employs LLM-driven methods.
   - Key innovations include optimizing lower triangular matrix multiplication and categorical cross entropy kernels, which surpass traditional **torch.compile**.
- **Dr. Jim Fan praises the autonomous coding agent**: A member highlighted Sakana AI's autonomous coding agent as a breakthrough for accelerating AI through enhanced CUDA kernels, noting the agent's efficiency in debugging.
   - This approach, termed the *Innovation Archive*, functions as a repository of best practices discovered during the evolution process.
- **Feedback on LLM usage for CUDA optimization**: Some members expressed surprise that an LLM was chosen for functionalizing PyTorch code instead of existing functional FX graph systems.
   - This conversation reflects the ongoing debate about the best tools and methods in optimizing CUDA kernel performance.
- **Dataset release and open challenges**: The AI CUDA Engineer has released a dataset of **17,000 verified CUDA kernels**, highlighting its comprehensive approach to AI-driven acceleration.
   - Challenges remain, such as improving kernel diversity and utilizing newer features like tensor cores, which are crucial for further advancements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sakanaailabs/status/1892385766510338559?s=46">Tweet from Sakana AI (@SakanaAILabs)</a>: Introducing The AI CUDA Engineer: An agentic AI system that automates the production of highly optimized CUDA kernels.http://sakana.ai/ai-cuda-engineer/The AI CUDA Engineer can produce highly optimize...</li><li><a href="https://x.com/drjimfan/status/1892404919480832259?s=46">Tweet from Jim Fan (@DrJimFan)</a>: The coolest autonomous coding agent I&#39;ve seen recently: use AI to write better CUDA kernels to accelerate AI. AutoML is so back! The highest leverage thing you can do with your compute resources i...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1341783799054209086)** (1 messages): 

> `MLA Attention Support` 


- **Inquiry on Contributing to MLA Attention Support**: A member noted the ongoing work to add support for **MLA attention** and inquired about methods to contribute to this effort.
   - This inquiry highlights community interest in collaborative development efforts concerning the **MLA attention** implementation.
- **Support for MLA Attention Gathering Interest**: The conversation emphasized a growing curiosity around **MLA attention** support, leading to an inquiry about potential contributions.
   - Members are looking for avenues to participate in the development, signaling a collaborative community dynamic.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1341474790275547260)** (13 messages🔥): 

> `SO-ARM100 Assembly, 3D Printing Experience, Dataset Collection Challenges, Hybrid Speech Processing Application` 


- **SO-ARM100 Assembly Experience**: Discussion about the assembly of the [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) highlighted mixed experiences, with one participant humorously noting the assembly took **five hours** instead of the advertised one hour.
   - *I don't really intend on printing a bunch of stuff,* said a user who's skeptical about owning a 3D printer, opting instead for CAD file shipping.
- **Dataset Collection Frustrations**: A user mentioned improvements in dataset collection since their initial experience, stating it was previously frustrating when *the arm failed during collecting an episode*.
   - They also pointed out that patching datasets was necessary to upload to Hugging Face, which complicated the process.
- **Optimizing Model Performance Strategies**: It was suggested that fine-tuning a pre-trained model like **pi-0** on a small collected dataset might yield a better performance than training from scratch.
   - This strategy is intended to simplify the initial experimentation with model training for new users.
- **Hybrid Speech Processing Application Overview**: A user shared their project on a hybrid speech processing application, stating that it processes speech separation on an NVIDIA Jetson Nano using cloud-based LLMs for filtering prompts.
   - The attached report details the implementation and outcomes, inviting feedback from peers on the project.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lerobot/visualize_dataset_v1.6?dataset=philbutler%2Feval_act_green_ball_red_cup-10sec_10ep-right_cam&episode=8">Visualize Dataset (v1.6 old dataset format) - a Hugging Face Space by lerobot</a>: no description found</li><li><a href="https://github.com/TheRobotStudio/SO-ARM100">GitHub - TheRobotStudio/SO-ARM100: Standard Open Arm 100</a>: Standard Open Arm 100. Contribute to TheRobotStudio/SO-ARM100 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1341517770679914517)** (56 messages🔥🔥): 

> `DeepSeek CodeI/O paper, Spatial reasoning datasets, Decimal chain sum dataset, Reasoning-Gym server experiment, Open-source ecosystem updates` 


- **DeepSeek introduces CodeI/O for reasoning tasks**: DeepSeek published a new paper on CodeI/O, aiming to enhance reasoning capabilities by converting code into input-output prediction formats. The approach bases the training on natural language tasks to improve performance, which is a significant shift from usual narrow skill enhancement.
   - A member suggested adding simpler versions for adaptability, highlighting the need for a well-rounded dataset.
- **Exploring spatial reasoning datasets**: A quest for datasets facilitating spatial reasoning in 3D environments yielded suggestions for tasks involving movement lists and spatial relations. The famous marble question exemplified complex reasoning challenges, advocating for template creation.
   - The StepGame benchmark was mentioned as a significant evaluation tool, although prior template errors affected results, pointing to a need for refinement.
- **Decimal chain sum dataset advancements**: The decimal_chain_sum dataset was discussed, highlighting the use of the Decimal class for accurate arithmetic operations. Members contributed implementation suggestions, focusing on exact score evaluation without tolerances.
   - A PR was created to register the dataset in the factory, addressing previously missed integrations.
- **Progress on Reasoning-Gym server experiment**: The initial version of the Reasoning-Gym server experiment is nearing completion, combining server and CLI tools effectively. Debugging efforts are set for the following morning to ensure functionality.
   - Members are actively working on integration aspects related to the server functionalities, progressing toward a comprehensive deployment.
- **Recognition from interconnect blog**: Reasoning-Gym received a mention from the interconnect substack blog, showcasing its growing recognition in the open-source ecosystem. The blog highlighted the rapid advancements in AI datasets, particularly through DeepSeek's contributions.
   - Members discussed ongoing developments in the AI landscape, emphasizing the impact of new models from Chinese labs and their permissive licenses.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.03991">Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark</a>: Artificial intelligence (AI) has made remarkable progress across various domains, with large language models like ChatGPT gaining substantial attention for their human-like text-generation capabilitie...</li><li><a href="https://arxiv.org/abs/2502.07316">CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction</a>: Reasoning is a fundamental capability of Large Language Models. While prior research predominantly focuses on enhancing narrow skills like math or code generation, improving performance on many other ...</li><li><a href="https://www.interconnects.ai/p/artifacts-7">The latest open artifacts (#7): Alpaca era of reasoning models, China&#x27;s continued dominance, and tons of multimodal advancements</a>: Artifacts Log 7. It&#x27;ll continue to be a fun spring for AI researchers and practitioners.</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/160">Add CODEI/O sampled subset dataset · Issue #160 · open-thought/reasoning-gym</a>: DeepSeek published CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction. Task: &quot;predict inputs/outputs given code and test cases entirely in natural language&quot; Beside propr...</li><li><a href="https://github.com/Fangjun-Li/SpatialLM-StepGame">GitHub - Fangjun-Li/SpatialLM-StepGame: Codes and data for AAAI-24 paper &quot;Advancing Spatial Reasoning in Large Language Models: An In-depth Evaluation and Enhancement Using the StepGame Benchmark&quot;</a>: Codes and data for AAAI-24 paper &quot;Advancing Spatial Reasoning in Large Language Models: An In-depth Evaluation and Enhancement Using the StepGame Benchmark&quot; - Fangjun-Li/SpatialLM-StepGame</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/163">Redundant line · Issue #163 · open-thought/reasoning-gym</a>: # Validate digit ranges make sense if self.min_digits &gt; 1: assert 10 ** (self.min_digits - 1) &gt;= 1, &quot;min_digits would result in invalid number range&quot; These lines in /reasoning_gym/arit...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/164">decimal_chain_sum by vncntt · Pull Request #164 · open-thought/reasoning-gym</a>: Adding Issue #157 Should be straightforward. Copied a lot of code from chain_sumI guess it&#39;s not perfect since ideally we want to target the tricky ones such as .9 with .11</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/167">Register decimal chain sum by vncntt · Pull Request #167 · open-thought/reasoning-gym</a>: no description found</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/arithmetic/decimal_chain_sum.py">reasoning-gym/reasoning_gym/arithmetic/decimal_chain_sum.py at main · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/158/">Decimal Arithmetic by Miserlou · Pull Request #158 · open-thought/reasoning-gym</a>: Simple 1234.123 + 234.567 type of math problems.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1341455544426303568)** (14 messages🔥): 

> `General Superintelligence Debate, Emergent Capabilities, AI and Game Learning, Anand's Introduction, DeepSeek R1 Upload` 


- **Debating General Superintelligence's Meaning**: Members discussed the term **general superintelligence**, with one stating it doesn't mean anything and is just a buzzword.
   - Another member suggested that if models excel at multiple tasks, they will acquire a deeper understanding to tackle complex, unquantifiable challenges.
- **Emergent Capabilities: Buzzword or Reality?**: There was contention over *emergent capabilities*, with members arguing about their validity and relevance in AI discussions.
   - One member clarified their perspective, stating their theory does not depend on emergent capabilities.
- **Learnings from Game Experiences**: The conversation shifted towards the idea that training a model on various games can facilitate learning tasks like **shogi** with less data.
   - However, another member emphasized that without specific training on **shogi**, the model would not perform well.
- **Introduction of Anand: Curious Mind**: New member **Anand** introduced themselves, expressing curiosity in AI minds and concerns about AI risks.
   - As a software engineer, Anand aims to learn and contribute positively towards a future centered around AI.
- **DeepSeek R1 Models Available**: Member shared a link for running **Perplexity's uncensored DeepSeek R1** and additional model versions available on Hugging Face.
   - They noted that **dynamic 2-bit GGUF quants** enhance accuracy over typical **1-bit/2-bit** formats and provided tutorials on usage.



**Link mentioned**: <a href="https://huggingface.co/unsloth/r1-1776-GGUF">unsloth/r1-1776-GGUF · Hugging Face</a>: no description found

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1341528494269665301)** (32 messages🔥): 

> `Model-guidance for training diffusion models, Deepseek V2 improvements, AI CUDA Engineer for optimized kernels, Reinforcement learning training curriculum, Sigmoid vs. softmax in gating mechanisms` 


- **Model-guidance technique enhances diffusion model training**: The novel objective, Model-guidance (MG), eliminates Classifier-free guidance (CFG) and significantly accelerates training speed, doubling inference rates while achieving state-of-the-art performance on ImageNet 256 benchmarks with an FID of **1.34**. Extensive experiments validate its effectiveness across various models and datasets.
   - *Extensive evaluations indicate MG's capabilities*, setting a new standard in diffusion model training.
- **Deepseek V2's softmax to sigmoid shift**: Deepseek V2 transitioned from **softmax** to **sigmoid** for gating mechanisms, with discussions highlighting that sigmoid can enhance training stability but complicates learning gate multipliers due to its capping behavior. Members debated the implications of using sigmoid versus softmax in terms of training dynamics and expert selection.
   - Notable points included that as sigmoid values approach **1**, the ability to differentiate gate multipliers diminishes, leading to potential fluctuations in expert selection.
- **Introduction of AI CUDA Engineer for kernel optimization**: The newly introduced [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) automates the creation of optimized CUDA kernels, purportedly achieving **10-100x** speedup over standard PyTorch operations. Its evolutionary approach enables autonomous performance improvement of CUDA kernels alongside a significant dataset release of **17,000** verified kernels.
   - The team believes this development marks a new era for AI-driven efficiencies in machine learning operations, with implications for automated optimization of inference times.
- **Curriculum-based reinforcement learning addressed**: A new approach in reinforcement learning emphasizes prioritizing questions with high variance in success rates during training to enhance learning signals effectively. Models utilizing this method showed consistent performance improvements in response to varying success in question-solving during training.
   - Comparisons between **PPO** and **VinePPO** reflected insights into wall-clock and compute time considerations in training efficiency.
- **Discussion on loss curve scaling**: Users expressed curiosity about fitting scaling laws to loss curves observed in training, noting differences in behaviors between diffusion and LLM models. Specific behaviors included diffusion loss flattening more rapidly, alongside variations in model training dynamics impacted by annealing schedules.
   - Observations hinted at the complexities of the training process, including how expert behavior could fluctuate based on input signaling during training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/SakanaAILabs/status/1892385766510338559">Tweet from Sakana AI (@SakanaAILabs)</a>: Introducing The AI CUDA Engineer: An agentic AI system that automates the production of highly optimized CUDA kernels.http://sakana.ai/ai-cuda-engineer/The AI CUDA Engineer can produce highly optimize...</li><li><a href="https://homietele.github.io">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2502.12272">Learning to Reason at the Frontier of Learnability</a>: Reinforcement learning is now widely adopted as the final stage of large language model training, especially for reasoning-style tasks such as maths problems. Typically, models attempt each question m...</li><li><a href="https://arxiv.org/abs/2502.12154">Diffusion Models without Classifier-free Guidance</a>: This paper presents Model-guidance (MG), a novel objective for training diffusion model that addresses and removes of the commonly used Classifier-free guidance (CFG). Our innovative approach transcen...</li><li><a href="https://arxiv.org/abs/2502.07316">CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction</a>: Reasoning is a fundamental capability of Large Language Models. While prior research predominantly focuses on enhancing narrow skills like math or code generation, improving performance on many other ...</li><li><a href="https://arxiv.org/abs/2502.13130">Magma: A Foundation Model for Multimodal AI Agents</a>: We present Magma, a foundation model that serves multimodal AI agentic tasks in both the digital and physical worlds. Magma is a significant extension of vision-language (VL) models in that it not onl...</li><li><a href="https://github.com/tzco/Diffusion-wo-CFG/blob/e86a3002df0aa086c7630a1fe379e9fb9564c2ff/train.py#L378)">Diffusion-wo-CFG/train.py at e86a3002df0aa086c7630a1fe379e9fb9564c2ff · tzco/Diffusion-wo-CFG</a>: Official Implementation for Diffusion Models Without Classifier-free Guidance - tzco/Diffusion-wo-CFG
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1341500977911566386)** (26 messages🔥): 

> `LLM Scaling Laws Terminology, Taxonomy of Scaling Laws, Resource Allocation in LLMs, Pre-training vs Post-training, Focus Areas in LLM Development` 


- **Understanding Scaling Laws Terminology**: A member sought clarification on terminology related to **LLM scaling laws**, including the roles of data, model size, and test-time compute in enhancing performance.
   - There was an agreement that **90% of scaling laws** discussions focus on pretraining metrics, while the relationship between training and deployment considerations was also emphasized.
- **Taxonomizing Scaling Laws for Insight**: One member expressed interest in creating a taxonomy of scaling laws to categorize compute resources effectively for analyzing progress in the LLM field.
   - Another pointed out that while scaling laws provide a framework, the actual data on resource allocation is not publicly available, complicating budget comparisons.
- **Budget Allocation Between Pre-training and Post-training**: Members discussed whether large labs are approaching a balance in their spending on pre-training versus post-training phases of model development.
   - Despite speculation, there was uncertainty on whether this includes costs associated with data acquisition.
- **Focus Areas of Recent LLM Releases**: Recent developments highlighted an increased focus on optimizing **test-time compute**, particularly in the context of the latest model releases like **Grok 3**.
   - It was noted that scaling up pretraining compute was yielding benefits, which challenged prior assumptions that such efforts had reached a dead end.
- **Challenges in Data and Finetuning Insights**: Members acknowledged the difficulty of accessing specific data used by labs in categories like creative writing, math/coding, and alignment for LLMs.
   - While model size could be gleaned from open source models, the 'secret sauce' behind data utilization and finetuning methods remained largely opaque.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1341662900623769600)** (5 messages): 

> `Recurrent Model Interpretability, Logit Lens, Tuned Lens, Counterfactual Testing, Average-case Goals` 


- **Recurrent Model Interpretability: A Tough Nut to Crack**: Discussion centers around the difficulties in achieving effective **recurrent model interpretability**, particularly for long-form reasoning tasks.
   - One member compares the challenge to **fuzzing** and finding **backdoors** in computer security, highlighting the complexities involved.
- **Exploring Logit and Tuned Lenses for Interpretability**: A member suggests revisiting **Logit Lens** and **Tuned Lens**, believing there's uncharted potential for using these with both transformers and recurrent models.
   - They emphasize the added value of analyzing the model's reasoning process at each step for complex tasks.
- **Counterfactual Testing in Model Training**: Another member proposes that training models to flag informative **counterfactuals** based on their latents could enhance interpretability during unknown stress tests.
   - This could provide insights even when the necessary cues aren't explicitly hidden by the model.
- **Navigating Average-case vs Worst-case Goals**: A conversation arises about the potential for achieving better performance with an **average-case** goal as opposed to **worst-case** scenarios.
   - This approach suggests that focusing on naturally configured latents could yield more successful outcomes.



**Link mentioned**: <a href="https://x.com/cfgeek/status/1892097007394652187?s=46">Tweet from Charles Foster (@CFGeek)</a>: @alextmallen I keep thinking about this. Curious if there’s a way to train something that, given a model’s latents on a specific input (for example, from a recurrent reasoner), flags what specific cou...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1341515704444387328)** (5 messages): 

> `Chess Tactic Dataset Structure, Eval Harness Support for NeMo Checkpoints` 


- **Debating Chess Tactic Dataset Structure**: A member expressed doubts about structuring a dataset for chess tactics, considering whether to predict full sequences of moves or to limit it to one-move prompts.
   - They noted that while including legal moves can reduce illegal move predictions, it's impractical to add all possible sequences due to their rapid growth.
- **Eval Harness and NeMo Checkpoints**: A member inquired if the Eval Harness supports .distcp checkpoints that NeMo saves during training, noting its support for NeMo checkpoints.
   - Another member was unsure but shared a [function from the Eval Harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness/blob/52df63b7b30da53c481ed9090598d9189fab1d91/lm_eval/models/nemo_lm.py#L71) that loads models.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/52df63b7b30da53c481ed9090598d9189fab1d91/lm_eval/models/nemo_lm.py#L71)">lm-evaluation-harness/lm_eval/models/nemo_lm.py at 52df63b7b30da53c481ed9090598d9189fab1d91 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1341454265436082292)** (32 messages🔥): 

> `NeMo vs GPT-NeoX performance, A100 configurations, Transformer Engine integration, Evo2 genome models, Communication strategies in model training` 


- **NeMo may have TP comm overlap advantage**: It's possible that **NeMo** utilizes TP communication overlap, potentially improving performance across **PCIe** connections. Members discussed the implications of using the `ub_tp_comm_overlap` flag and whether switching to TE MLPs would help mitigate issues.
   - One member noted they aren't explicitly setting any comm flags in NeMo, implying defaults are in use, while highlighting a need for further exploration of discrepancies.
- **Testing allreduce bucket sizes**: Discussion arose about the impact of matching **allreduce bucket sizes** between NeMo and NeoX, with suggestions that larger sizes may improve speed on **PCIe**. A member shared they experimented with 2x and 4x sizes in NeoX but did not observe any speed improvements.
   - This raises questions about the configurations being tested and their influence on overall model performance.
- **Evo2 genome models trained on GPT-NeoX**: A member announced that the new **Evo2 genome models** were trained using a library that builds upon **GPT-NeoX**, linking back to the community's work. Another member expressed excitement about this recognition, appreciating the rewarding outcome.
   - This highlights the ongoing contributions of the community towards advanced model training methodologies.
- **A100 configurations and their impact**: The conversation explored whether the **A100s** in use are **PCIe** or **NVLINK**, with one member confirming that they are indeed PCIe. This discovery prompted discussions around how communication speeds might be affecting performance metrics.
   - The need to optimize setups based on the type of interface was emphasized, as it potentially affects maximum transactions per second (TPS).
- **Challenges with Transformer Engine integration**: One member mentioned attempting to integrate the **Transformer Engine** in NeoX but faced challenges that halted further testing. They suggested that turning on non-FP8 flags broke the system, indicating potential instability with current setups.
   - This outlines the complexities involved in integrating new technologies while maintaining system reliability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/pythia/6-9B.yml">gpt-neox/configs/pythia/6-9B.yml at olmo-support · aflah02/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - aflah02/gpt-neox</li><li><a href="https://github.com/NVIDIA/NeMo/blob/0621272c2a9a760a71b234131f1997e87a265943/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L882">NeMo/nemo/collections/nlp/models/language_modeling/megatron_base_model.py at 0621272c2a9a760a71b234131f1997e87a265943 · NVIDIA/NeMo</a>: A scalable generative AI framework built for researchers and developers working on Large Language Models, Multimodal, and Speech AI (Automatic Speech Recognition and Text-to-Speech) - NVIDIA/NeMo
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1341456411972079769)** (38 messages🔥): 

> `Grok 3's New Game Studio, Attention Mechanism Backpropagation, Mamba Deep Learning Architecture, Transformer Complexity in Inference, Jailbreak Challenge Insights` 


- **Grok 3's New Game Studio Announcement**: Elon Musk confirmed the start of a game studio by xAI, which is a logical extension given their recent focus on AI initiatives, as discussed in a recent [Twitter post](https://x.com/elonmusk/status/1891388509191049307).
   - *Dima Zeniuk* mentioned this shift towards gaming in relation to Grok 3's branding update.
- **Understanding Backpropagation in Transformers**: A discussion emerged about the complexities of backpropagation through transformer architectures, emphasizing the unique challenges posed by the attention mechanism.
   - Experts suggested starting with specific operations, like attention, for a clearer understanding of gradient calculations.
- **Mamba Architecture vs. Traditional Transformers**: The Mamba architecture, developed to handle long sequences, may avoid the O(n^2) complexity typical in vanilla transformers due to its structured state space approach.
   - Mamba is designed for efficiency, boasting linear complexity compared to transformers, which require more extensive computation during inference.
- **Insights from the Recent Jailbreak Challenge**: In a detailed analysis, it was discussed that success in the recent jailbreak challenge likely required participants to sign non-disclosure agreements, limiting the sharing of their findings.
   - A shared [Google document](https://docs.google.com/document/d/1WI3IaiYCDQUk5YrFK6MP_YxOXwyy3mV9a8d66ndJeyI/edit?tab=t.0) provides additional resources on current jailbreak methods.
- **Exploring Loss Functions without Softmax**: A user proposed the idea of defining a mean squared error (MSE) loss in embedding space and subsequently training a classification head using cross entropy, avoiding the need for softmax in transformer blocks.
   - This suggests a potential simplification in the transformer architecture while still achieving effective performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.06773">On the Emergence of Thinking in LLMs I: Searching for the Right Intuition</a>: Recent AI advancements, such as OpenAI&#39;s new models, are transforming LLMs into LRMs (Large Reasoning Models) that perform reasoning during inference, taking extra time and compute for higher-qual...</li><li><a href="https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture)">Mamba (deep learning architecture) - Wikipedia</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=X_niF6KaWd8">🚨🚨 Chat Creates Doom With Devin.ai 🚨🚨</a>: Twitch https://twitch.tv/ThePrimeagenDiscord https://discord.gg/ThePrimeagenBecome Backend Dev: https://boot.dev/prime(plus i make courses for them)This is a...</li><li><a href="https://www.youtube.com/watch?v=nsOMuWD58N0&list=PLgKuh-lKre1058wlfuwtOYyemY5qoxavQ">How Do Transformers Learn Variable Binding?</a>: Raphaël Millière (Macquarie University)https://simons.berkeley.edu/talks/raphael-milliere-macquarie-university-2025-02-07LLMs, Cognitive Science, Linguistics...</li><li><a href="https://fxtwitter.com/janleike/status/1890141865955278916?t=m7VUL3XBshTSSrfzBa9Wzw&s=19">Tweet from Jan Leike (@janleike)</a>: Results of our jailbreaking challenge:After 5 days, &gt;300,000 messages, and est. 3,700 collective hours our system got broken. In the end 4 users passed all levels, 1 found a universal jailbreak. We...</li><li><a href="https://docs.google.com/document/d/1WI3IaiYCDQUk5YrFK6MP_YxOXwyy3mV9a8d66ndJeyI/edit?tab=t.0">Jailbreaking LLMs</a>: Jailbreaking Large Language Models Prepared by: 	Stu Jordan, Evolution Unleashed Lab (@stujordanAI on X)  Date: 		07 February 2025  In light of the challenge from Anthropic, I thought I’d share this r...</li><li><a href="https://x.com/elonmusk/status/1891388509191049307">Tweet from Elon Musk (@elonmusk)</a>: YesQuoting Dima Zeniuk (@DimaZeniuk) Elon Musk&#39;s xAI is going to start an AI game studio to make games
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1341507171828891712)** (50 messages🔥): 

> `AI Paper Discussions, DeepSeek's Sparse Attention Paper, Challenges in Paper Selection, Discord Event Organization, User Participation` 


- **Challenges and Opportunities in AI Papers**: Participants discussed the abundance of AI papers and the importance of filtering out noise to focus on valuable insights, with one member stating, *bad papers are just a different kind of opportunity*. Another mentioned that the current influx of AI papers makes it difficult to avoid selecting duds.
   - A member shared that identifying why a paper is inadequate could yield valuable skills, emphasizing the time constraints they face.
- **DeepSeek's Paper on Sparse Attention**: Today's discussion featured DeepSeek's paper titled **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention** ([link](https://arxiv.org/abs/2502.11089)), presented in a cold read format with no preparation required. One user remarked on the quality of the paper, calling it a *banger*.
   - The event is set to happen at a specific time, with some anticipation for potential re-discussions over the weekend.
- **Scheduling and Participation Concerns**: Timing and participation in discussions were highlighted, with questions about whether meetings will be recorded to ensure privacy. It was noted that recording is prohibited to encourage open dialogue, making the atmosphere more comfortable for all attendees.
   - Several members expressed their intention to catch up on missed papers and planned follow-up discussions to revisit key findings.
- **Improving Event Attendance**: Discussion organizers suggested that having more advance notice for meetings could increase attendance, as more members might be able to join. One member encouraged others to express interest in events to gauge participation numbers effectively.
   - Conflicts with personal schedules and time zones were acknowledged as significant challenges for some attendees.
- **Community Engagement and Feedback**: Members expressed their appreciation for the discussions and commended the insights shared during the sessions, reaffirming the value of community feedback. Some also shared plans to engage more actively in future discussions, highlighting their commitment to personal learning.
   - The community emphasized the importance of ongoing discussions for collective growth, setting the stage for constructive engagement in upcoming meetings.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.11089">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>: Long-context modeling is crucial for next-generation language models, yet the high computational cost of standard attention mechanisms poses significant computational challenges. Sparse attention offe...</li><li><a href="https://arxiv.org/abs/2502.09696">ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models</a>: Large Multimodal Models (LMMs) exhibit major shortfalls when interpreting images and, by some measures, have poorer spatial cognition than small children or animals. Despite this, they attain high sco...</li><li><a href="https://arxiv.org/abs/2502.12150">Idiosyncrasies in Large Language Models</a>: In this work, we unveil and study idiosyncrasies in Large Language Models (LLMs) -- unique patterns in their outputs that can be used to distinguish the models. To do so, we consider a simple classifi...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1341458105224925245)** (13 messages🔥): 

> `Los Angeles Project Unicorn Engineering, Thinking Machines Lab by Mira Murati, Elon's Grok 3, Perplexity AI's Censorship Overcome, Microsoft's Majorana 1 Quantum Processor` 


- **Los Angeles Project aims for unicorns**: [The Los Angeles Project](https://building.life/) is a biotech startup aiming to engineer literal unicorns through gene-editing and cutting-edge technology.
   - Concerns arose regarding the absence of AI in their hiring pitch, suggesting AI may merely be a buzzword to attract investment.
- **Mira Murati launches Thinking Machines Lab**: Former OpenAI CTO Mira Murati has founded [Thinking Machines Lab](https://thinkingmachines.ai), focusing on making AI systems more understandable and customizable while ensuring public transparency.
   - The specifics of their upcoming projects remain under wraps, but they promise to publish regular technical research.
- **Elon's Grok 3 under scrutiny**: A [YouTube video](https://www.youtube.com/watch?v=b0XI-cbel1U) poses the question, 'Is Elon’s Grok 3 the new AI king?' as discussions around its capabilities escalate.
   - The presentation hints at potential advancements, but opinions vary on its actual impact.
- **Perplexity AI breaks censorship barriers**: [Perplexity AI](https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/) has introduced R1 1776, a model overcoming Chinese censorship through innovative post-training techniques.
   - This approach aims to establish more robust AI systems that generate content while retaining contextual integrity.
- **Microsoft's Majorana 1 ushers quantum leap**: Microsoft unveiled Majorana 1, the first Quantum Processing Unit powered by topological qubits, which could scale to a million qubits on a single chip.
   - This advancement marks significant progress toward practical quantum computing and promises a new era in technology.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/ai-artificial-intelligence/614621/mira-murati-thinking-machines-lab-openai-competitor-launch">Mira Murati is launching her OpenAI rival: Thinking Machines Lab</a>: Another OpenAI rival is being set up by former employees.</li><li><a href="https://x.com/Jiankui_He/status/1892111600770863426">Tweet from Jiankui He (@Jiankui_He)</a>: Embryo gene editing can prevent cancer. By increasing the DNA repairing capacity by 100 fold, our children will no longer have any gene mutation, therefore permanently eradicate cancer. However, I hav...</li><li><a href="https://www.youtube.com/watch?v=b0XI-cbel1U">Is Elon’s Grok 3 the new AI king?</a>: Try Brilliant free for 30 days https://brilliant.org/fireship You’ll also get 20% off an annual premium subscription.Take a first look at Elon Musk&#39;s Grok 3 ...</li><li><a href="https://x.com/elder_plinius/status/1891968598496760230?s=46">Tweet from Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius)</a>: 🧙‍♂️ 󠅗󠅗NEW ATTACK CLASS UNLOCKED 🧙‍♂️󠅗󠅗</li><li><a href="https://www.youtube.com/watch?v=wSHmygPQukQ">Majorana 1 Explained: The Path to a Million Qubits</a>: Hear from the Microsoft team behind the recent breakthrough in physics and quantum computing demonstrated by the new Majorana 1 chip, engineered from an enti...</li><li><a href="https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/">Perplexity AI removes Chinese censorship from Deepseek R1</a>: Perplexity AI has unveiled R1 1776, a modified version of the Deepseek-R1 language model specifically designed to overcome Chinese censorship through specialized post-training techniques.</li><li><a href="https://www.piratewires.com/p/harnessing-the-breath-of-life">Harnessing the Breath of Life</a>: how a gene-editing startup called the los angeles project will create actual, literal unicorns (and more)</li><li><a href="https://building.life/">LAP</a>: no description found</li><li><a href="https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/">Microsoft unveils Majorana 1, the world’s first quantum processor powered by topological qubits - Microsoft Azure Quantum Blog</a>: Majorana 1 from Microsoft is the world’s first Quantum Processing Unit (QPU) built with a topoconductor. Discover more.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1341489757980725279)** (79 messages🔥🔥): 

> `Thinking Machines Lab Launch, Perplexity AI R1 1776 Release, OpenAI SWElancer Benchmark, Mastra Open-source JS Framework, Cloud AI Infrastructure and Funding` 


- **Thinking Machines Lab Launches**: Thinking Machines Lab aims to make AI systems more accessible and customizable, addressing knowledge gaps in frontier AI systems along the way. They've brought together a team of experts behind popular AI products like [ChatGPT](https://chatgpt.com) and [Character.ai](https://character.ai/).
   - Mira Murati, former CTO of OpenAI, is a notable figure in this new venture, alongside co-founders like Lilian Weng.
- **Perplexity AI Introduces R1 1776**: Perplexity AI announced the open-sourcing of R1 1776, a version of their DeepSeek R1 model that offers uncensored and unbiased information. This marks a notable push towards providing more reliable AI models in varied domains.
   - The AI community is humorously dubbing this development as 'freedomtuning' as a nod to the model's new features.
- **OpenAI Launches SWElancer Benchmark**: OpenAI unveiled SWElancer, a new benchmark for evaluating AI coding performance with over 1,400 freelance software engineering tasks. The aim is to create a more realistic metric for understanding AI capabilities in real-world applications.
   - This initiative comes amid discussions about the potential replacement of game studios with AI-driven game generation.
- **Mastra's Open-source Framework Launch**: Mastra has released an open-source JavaScript SDK aimed at building AI agents, allowing for complex task execution with built-in workflows. It promises to facilitate the development of autonomous agents capable of performing intricate tasks.
   - The framework encourages collaboration and integration with Vercel’s AI SDK, marking a significant step in open-source AI development.
- **Lambda Secures Series D Funding**: Lambda has announced a substantial $480 million Series D funding round, co-led by Andra Capital and SGW. The investment reflects growing interest in cloud services tailored for AI applications.
   - Participation from notable investors like NVIDIA and ARK Invest highlights the firm's potential in the evolving AI landscape.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://thinkingmachines.ai/]">Thinking Machines Lab</a>: no description found</li><li><a href="https://thinkingmachines.ai/">Thinking Machines Lab</a>: no description found</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://x.com/dchaplot/status/1891920016339042463">Tweet from Devendra Chaplot (@dchaplot)</a>: Career Update: Incredibly fortunate and excited to be part of the founding team at Thinking Machines Lab!https://thinkingmachines.ai/Join us: https://6wajk07p.paperform.co/</li><li><a href="https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/">Accelerating scientific breakthroughs with an AI co-scientist</a>: no description found</li><li><a href="https://x.com/arrakis_ai/status/1892141483941347627?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">Tweet from CHOI (@arrakis_ai)</a>: Claude Extended Thinking!</li><li><a href="https://mastra.ai/docs/agents/00-overview">Creating and Calling Agents | Agent Documentation | Mastra</a>: no description found</li><li><a href="https://www.theverge.com/ai-artificial-intelligence/614621/mira-murati-thinking-machines-lab-openai-competitor-launch">Mira Murati is launching her OpenAI rival: Thinking Machines Lab</a>: Another OpenAI rival is being set up by former employees.</li><li><a href="https://x.com/perplexity_ai/status/1891916573713236248">Tweet from Perplexity (@perplexity_ai)</a>: Today we&#39;re open-sourcing R1 1776—a version of the DeepSeek R1 model that has been post-trained to provide uncensored, unbiased, and factual information.</li><li><a href="https://en.wikipedia.org/wiki/Thinking_Machines_Corporation">Thinking Machines Corporation - Wikipedia</a>: no description found</li><li><a href="https://x.com/ashtom/status/1891925306430337110?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Thomas Dohmke (@ashtom)</a>: Our new code completion model is shipping in public preview today. We are calling it GPT-4o Copilot. Based on GPT-4o mini, with mid-training on a code-focused corpus exceeding 1T tokens and reinforcem...</li><li><a href="https://www.youtube.com/watch?v=mFuyX1XgJFg&ab_channel=Apple">Introducing iPhone 16e - February 19</a>: The latest iPhone at the greatest price — iPhone 16e is here. It’s built for Apple Intelligence and powered by A18, the latest-generation chip. It also comes...</li><li><a href="https://x.com/hyusapx/status/1891642232635306053?s=46">Tweet from ayush (@hyusapx)</a>: built an ai tool for etymology nerds: deconstructor. live rn.the goal was to analyze the origins of english words, but it ended up having a ton of amazing emergent properties (see below)</li><li><a href="https://x.com/pitdesi/status/1891979324787482769">Tweet from Sheel Mohnot (@pitdesi)</a>: Humane AI pin is winding downHP is acquiring the team, IP and software for $116MFounders Imran and Bethany, will form a new division at HP to integrate AI into HP PC&#39;s, printers and connected conf...</li><li><a href="https://x.com/basetenco/status/1892259130540179863">Tweet from Baseten (@basetenco)</a>: 2025 is the year of inference.We&#39;re thrilled to announce our $75m Series C co-led by @IVP  and @sparkcapital  with participation from @GreylockVC, @conviction, @basecasevc, @southpkcommons  and @l...</li><li><a href="https://www.threads.net/@zuck/post/DGOISdTRX9Q?xmt=AQGzKO9jUlLz9JyNocdtLrsQ1L8IvvVBRb--7JMStLY6Fg">Mark Zuckerberg (&#064;zuck) on Threads</a>: Save the date &#x1f5d3;LlamaCon: Apr 29Connect: Sept 17-18</li><li><a href="https://x.com/loubnabenallal1/status/1892278622104215894?s=46">Tweet from Loubna Ben Allal (@LoubnaBenAllal1)</a>: The nanotron team just released The Ultra-Scale PlayBook with everything you need to know about LLM pretraining at smol and large scales  https://huggingface.co/spaces/nanotron/ultrascale-playbook</li><li><a href="https://x.com/nearcyan/status/1892049572546904319?s=46">Tweet from near (@nearcyan)</a>: everyone who bought the $700 AI pin got literally ruggedQuoting Elvin (@elvin_not_11) Offline features like battery level</li><li><a href="https://x.com/hellenicvibes/status/1892250276473516266?s=46">Tweet from Zoomer Alcibiades (@HellenicVibes)</a>: Google’s AI agent independently discovered:- A new leukemia drug that then successfully tested in vitro at clinical concentrations- Novel liver fibrosis drug targets- Bacterial cell-level antibiotic m...</li><li><a href="https://siliconangle.com/2025/02/17/ilya-sutskevers-safe-superintelligence-reportedly-raising-1b-30b-valuation/">Ilya Sutskever&#039;s Safe Superintelligence reportedly raising $1B+ on $30B valuation - SiliconANGLE</a>: Ilya Sutskever&#039;s Safe Superintelligence reportedly raising $1B+ on $30B valuation - SiliconANGLE</li><li><a href="https://x.com/leonardtang_/status/1892243653071908949">Tweet from Leonard Tang (@leonardtang_)</a>: First came pre-training scaling; then came inference-time scaling.Now comes judge-time scaling.Despite progress in AI through scaled inference-time compute, AI remains unreliable in open-ended, non-ve...</li><li><a href="https://x.com/stephenbalaban/status/1892275552171737220?s=46">Tweet from stephen balaban (@stephenbalaban)</a>: Lambda is a cloud designed for the age of AI. Today, we announced a $480M Series D co-led by Andra Capital and SGW with participation from NVIDIA, Andrej Karpathy, In-Q-Tel, ARK Invest, and others.</li><li><a href="https://x.com/scaling01/status/1891913189199053301?s=46">Tweet from Lisan al Gaib (@scaling01)</a>: I love the advertisements for Sonnet 3.5 in each OpenAI paperA bit odd they didn&#39;t include o3. Now it either feels like a setup so that they can crush their own benchmark later or o3 just sucks in...</li><li><a href="https://youtu.be/4GLSzuYXh6w?si=K9JWK5HMIQzxFk7v">Satya Nadella – Microsoft’s AGI Plan &amp; Quantum Breakthrough</a>: Satya Nadella on:- Why he doesn’t believe in AGI but does believe in 10% economic growth,- Microsoft’s new topological qubit breakthrough and gaming world mo...</li><li><a href="https://x.com/thom_wolf/status/1892273133547078036?s=46">Tweet from Thomas Wolf (@Thom_Wolf)</a>: After 6+ months in the making and burning over a year of GPU compute time, we&#39;re super excited to finally release the &#34;Ultra-Scale Playbook&#34;Check it out here: http://hf.co/spaces/nanotron/...</li><li><a href="https://x.com/openai/status/1891911132983722408?s=46">Tweet from OpenAI (@OpenAI)</a>: Current frontier models are unable to solve the majority of tasks.</li><li><a href="https://youtu.be/Ju0ndy2kwlw?si=_Maiv6-7b0dv3vLg">I built an AI supercomputer with 5 Mac Studios</a>: Get NordVPN 2Y plan + 4 months extra + 6 MORE months extra:   https://nordvpn.com/networkchuck It’s risk-free with Nord’s 30-day money-back guarantee! I just...</li><li><a href="https://tenor.com/view/freedom-america-gif-15593845046973100361">Freedom America GIF - Freedom America - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/thinkymachines/status/1891919141151572094">Tweet from Thinking Machines (@thinkymachines)</a>: Today, we are excited to announce Thinking Machines Lab (https://thinkingmachines.ai/), an artificial intelligence research and product company. We are scientists, engineers, and builders behind some ...</li><li><a href="https://news.ycombinator.com/item?id=43103073">Show HN: Mastra – Open-source JS agent framework, by the developers of Gatsby | Hacker News</a>: no description found</li><li><a href="https://youtu.be/4poqjZlM8Lo?si=FYxRdCbX_SPg_3Wj">AI bosses on what keeps them up at night</a>: Google DeepMind and Anthropic founders, Demis Hassabis and Dario Amodei, are two of the world&#39;s foremost leaders in artificial intelligence. Our editor-in-ch...</li><li><a href="https://youtu.be/4GLSzuYXh6w?si=74SFFf1tGUdflWdm&t=2643">Satya Nadella – Microsoft’s AGI Plan &amp; Quantum Breakthrough</a>: Satya Nadella on:- Why he doesn’t believe in AGI but does believe in 10% economic growth,- Microsoft’s new topological qubit breakthrough and gaming world mo...</li><li><a href="https://x.com/karpathy/status/1891938714915569711">Tweet from Andrej Karpathy (@karpathy)</a>: Congrats on company launch to Thinking Machines!Very strong team, a large fraction of whom were directly involved with and built the ChatGPT miracle. Wonderful people, an easy follow, and wishing the ...</li><li><a href="https://techcrunch.com/2025/02/18/humanes-ai-pin-is-dead-as-hp-buys-startups-assets-for-116m/">Humane&#039;s AI Pin is dead, as HP buys startup&#039;s assets for $116M | TechCrunch</a>: Humane announced on Tuesday that most of its assets have been acquired by HP for $116 million. The hardware startup is immediately discontinuing sales of</li><li><a href="https://www.reddit.com/r/apple/comments/1it9839/apple_polishing_cloth_adds_support_for_the_new/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1341699849317388328)** (28 messages🔥): 

> `Grok vs Mojo, Polars Implementation in Mojo, Livestream Event for MAX 25.1, Community Meeting Talks` 


- **Grok Release Surprises Mojo Developers**: The release of **Grok 3** has taken some by surprise, arriving before **Mojo** could reach its halfway mark.
   - *This is actually a positive thing for Mojo,* leading to excitement among some members.
- **Polars Integrated into Mojo Project**: A member shared that they quickly **imported Polars** into their Mojo project, showcasing their implementation with examples on [GitHub](https://github.com/rcghpge/pymo/tree/main).
   - There was a debate over the difference between *implementing* and *importing* Polars, with members clarifying how they are using it.
- **Upcoming Livestream on MAX 25.1**: A livestream is scheduled for tomorrow to discuss all things **MAX 25.1**, with members encouraged to submit questions through a [Google Form](https://forms.gle/NkjU6e3n15TRtiMA7).
   - They invited everyone to join via [LinkedIn](https://www.linkedin.com/events/introducingmax25-17297704283980902402/theater/) for the session.
- **Open Call for Community Meeting Talks**: Members were informed about opportunities to give talks during Monday's community meeting, encouraging sharing of projects or focus areas.
   - *We're looking for engaging presentations,* and interested members were invited to step forward.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forms.gle/NkjU6e3n15TRtiMA7">Modular Community Q&amp;A</a>: no description found</li><li><a href="https://github.com/rcghpge/pymo/blob/main/pymo/libpm/libpm.mojo#L42">pymo/pymo/libpm/libpm.mojo at main · rcghpge/pymo</a>: A framework in Mojo for AI/ML/DL applications and other domains. - rcghpge/pymo</li><li><a href="https://github.com/rcghpge/pymo/blob/main/examples/polars.mojo">pymo/examples/polars.mojo at main · rcghpge/pymo</a>: A framework in Mojo for AI/ML/DL applications and other domains. - rcghpge/pymo
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1341501544872280085)** (42 messages🔥): 

> `Mojo Stack Implementation, Performance of Quick Sort in Mojo, Slab List vs Linked List, VS Code Configuration for Mojo, Use of Set as Dict Value in Mojo` 


- **Exploring Efficient Stack Implementations in Mojo**: A user inquired about alternative stack implementations in Mojo, specifically questioning the use of pointers versus ArcPointer or UnsafePointer.
   - Another member noted that a `LinkedList` could be utilized for add or remove operations, providing flexibility in memory allocation.
- **Mojo Quick Sort Performance Issues**: A user shared their findings that their Python quick sort algorithm outperformed their Mojo implementation by a significant margin (0.4s vs 2.9s).
   - Discussion ensued about using Mojo's benchmark module to isolate and measure only the sort's performance and suggestions to avoid compile time affecting timing.
- **Understanding Slab List Structure**: Members discussed the benefits of using a `SlabList` over a traditional `LinkedList`, highlighting constant time operations and cache efficiency.
   - It was clarified that a slab list essentially comprises `LinkedList[InlineArray[T, N]]` allowing efficient memory use without complex manipulations.
- **Setting Up Mojo in VS Code**: A user faced issues with VS Code not recognizing their main package due to directory structure, leading to redlined imports in the editor.
   - Another member suggested managing the import path similarly to the standard library compilation process.
- **Using Set as a Value Type in Mojo Dictionaries**: A user questioned whether Dictionaries in Mojo could have values of type Set, prompting discussion regarding type compatibility.
   - It was confirmed that the standard library functions as a normal Mojo library, thus reinforcing the possibility of similar data structure implementations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/nic">nic - Overview</a>: Chief Trolling Officer. nic has 58 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/nickziv/libslablist">GitHub - nickziv/libslablist: The slab list is a very memory-efficient logarithmic-time data structure.</a>: The slab list is a very memory-efficient logarithmic-time data structure. - nickziv/libslablist
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1341454363305967688)** (56 messages🔥🔥): 

> `CUDA GPU Support, Embedding Token Limits, Chat Templates, Nomic V2 Release Delay, User Interface for Images` 


- **CUDA GPU Support discusses**: The discussion highlighted older CUDA 5.0 compatible GPUs like **GM107/GM108**, noting the lack of support for low-end architectures due to their age and limited utility.
   - One member confirmed that a recent **PR** supporting these GPUs has been merged and will be included in the next release.
- **Embedding token limits clarified**: Members discussed reaching the **10 million token limit** for embedding in GPT4All and the billing structure that includes a base price of **$10/month** with additional costs for extra tokens.
   - A key point noted was that removal of tokens from local documents does not reduce the total token count billed.
- **Chat templates and system messages**: A member sought clarification on using **chat templates** to instruct the model to quote excerpts, confirming that simple instructions in the system message should generally suffice.
   - Another member inquired about the effectiveness of Jinja or JSON code in prompting the model, indicating complexity in achieving expected outputs.
- **Concerns over Nomic v2 release**: There was speculation regarding the absence of **Nomic v2**, with members curious about the delay and pointing out its importance.
   - Laughter ensued as one member humorously questioned the prolonged wait without updates on the new version.
- **Limitations on image handling in GPT4All**: A member requested the ability to **copy and paste images** directly into the chat, similar to other platforms, but it was clarified that GPT4All currently does not support image inputs.
   - Suggestions were made to use external software for image handling instead.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html#using-the-cli-all-models.">Chat Templates - GPT4All</a>: GPT4All Docs - run LLMs efficiently on your hardware</li><li><a href="https://en.wikipedia.org/wiki/CUDA#GPUs_supported)">CUDA - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1341457016694112367)** (20 messages🔥): 

> `Anthropic Homepage Downtime, Haiku 3.5 Release Speculations, Cursor MCP Tool Issues, Custom Protocol MCP Servers, Puppeteer Docker Build Questions` 


- **Anthropic Homepage experiencing downtime**: A member reported that the **Anthropic homepage is down**, speculating about possible service interruptions.
   - Attached was an image indicating the downtime status.
- **Rumors of Haiku 3.5 with new features**: There are discussions surrounding the potential release of **Haiku 3.5**, featuring **tool and vision support**.
   - Another member mentioned a possibility of also seeing **Sonnet 4.0**.
- **Cursor MCP facing tool detection issues**: Several members expressed concerns over the **Cursor MCP** reporting **'No tool found'**, indicating a widespread issue.
   - One implementation was shared using **/sse** to remedy the situation, pointing to a shared information link.
- **Inquiries on Custom Protocol MCP Servers**: Members discussed the difficulty in creating **MCP servers using custom protocols** that don't utilize SSE or stdio.
   - One suggested packaging solutions through **Docker** and leveraging bridges for integration.
- **Questions about building Puppeteer with Docker**: A new user sought guidance on the final steps for installing **Puppeteer**, particularly regarding building Docker images.
   - They asked if a specific directory change was necessary and whether Docker installation was a prerequisite.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1341611580218937504)** (27 messages🔥): 

> `Google Workspace MCP, Dockerized MCP Servers, Integration with Sage, Python Interpreter with MCP, Matplotlib/Plotting Support in MCP` 


- **Google Workspace MCP on Docker**: A member shared their progress on a feature-rich [Google Workspace MCP](https://github.com/aaronsb/google-workspace-mcp) that supports multiple accounts and auto token refresh, with Docker images for different platforms.
   - This MCP provides integrated access to **Gmail**, **Calendar**, and other Google Workspace APIs.
- **Deployment Guide for Dockerized MCP Servers**: A blog post was shared discussing how to deploy Dockerized MCP servers, highlighting the challenges and solutions through Docker's encapsulation methods.
   - The article notes the complexity of **MCP Server packaging** across various architectures, and points to reference servers on GitHub.
- **Aligning Sage with Glama**: There are discussions about Glama's integration with Sage, with members expressing interest in collaboration and aligning the two projects more closely.
   - It was mentioned that API additions are needed for listing installed MCPS directly, but Glama does work when added to Sage.
- **Python Interpreter's MCP Integration**: A member introduced their [Python REPL for MCP](https://github.com/evalstate/mcp-py-repl), providing STDIO support and sharing that it includes **matplotlib**, **seaborn**, and **numpy**.
   - Future plans include adding **IPython** support, with a strong interest in visualizations similar to those in **Jupyter**.
- **Handling Images in MCP**: In a discussion about image handling, it was noted that currently, images are saved as **.png** files and only the names are returned to the MCP client.
   - Participants discussed the ease of returning files directly as tool results in an MCP server demonstration context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.defang.io/blog/2025/02/18/model-context-protocol">Simplifying Deployment of AI Apps to the Cloud using Docker and Model Context Protocol | Defang</a>: mcp</li><li><a href="https://github.com/evalstate/mcp-py-repl">GitHub - evalstate/mcp-py-repl: A python repl for MCP</a>: A python repl for MCP. Contribute to evalstate/mcp-py-repl development by creating an account on GitHub.</li><li><a href="https://github.com/aaronsb/google-workspace-mcp">GitHub - aaronsb/google-workspace-mcp: A Model Context Protocol (MCP) server that provides authenticated access to Google Workspace APIs, offering integrated Authentication, Gmail, Calendar, and Drive functionality</a>: A Model Context Protocol (MCP) server that provides authenticated access to Google Workspace APIs, offering integrated Authentication, Gmail, Calendar, and Drive functionality - aaronsb/google-work...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1341457077100482570)** (2 messages): 

> `Vendor Questionnaires App, LlamaCloud EU Launch` 


- **App Simplifies Vendor Questionnaires**: A member shared an innovative [full-stack app](https://twitter.com/llama_index/status/1891897812834816379) from @patrickrolsen that allows users to answer vendor questionnaires by semantically retrieving previous answers and enhancing them with an LLM.
   - This application represents a core use case for **knowledge agents** by streamlining the process of reading forms and filling in answers.
- **LlamaCloud EU Secures Data for Enterprises**: LlamaCloud EU has been announced, offering **secure**, compliant knowledge management specifically for European enterprises with a focus on data residency within EU jurisdiction [here](https://twitter.com/llama_index/status/1892271183451869512).
   - This early access offering aims to eliminate a significant barrier for European companies concerned about compliance and data privacy.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1341484612630872190)** (27 messages🔥): 

> `AI chat for company guidance, AgentWorkflow feature issues, QuadrantVectorStore import problem, Blockchain development collaboration, Checkpoint context serialization` 


- **AI Chatbot for Company Guidance**: A user is exploring options for creating a local AI chat to answer company-related questions, initially considering Llama 3.2 with RAG or fine-tuning.
   - Currently, they're planning to use VLLM for LLM compatibility with Brazilian Portuguese, with data managed in Postgres using Llama-Index.
- **AgentWorkflow Tool Output Bug**: A user reported that their AgentWorkflow's tool output list remains empty despite generating responses, seeking clarity on implementation.
   - Another member shared that streaming events could serve as a workaround to capture all tool calls during execution.
- **Issues with QuadrantVectorStore Import**: A user was unable to import QuadrantVectorStore from llama_index.vector_stores.qdrant, leading to confusion about its installation.
   - Once discovering that the Qdrant vector store needs separate installation, they resolved dependency conflicts before successful use.
- **Blockchain Development Collaboration**: A user with extensive blockchain development experience is seeking projects or collaborations in DeFi, NFTs, and Web3 development.
   - Specialties include smart contract development, DeFi platforms, and blockchain security auditing, with a focus on staying ahead of trends.
- **Checkpoint Context Serialization Challenges**: A developer is facing issues with context serialization when developing a human-in-the-loop AgentWorkflow, particularly when trying to persist context states.
   - Despite following documentation, after loading checkpoints, the agent fails to proceed, remaining stuck while 'thinking'.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_multi/">Multi-Agent Research Workflow with AgentWorkflow - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agents/">Multi-agent workflows - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/#tools-and-state">AgentWorkflow Basic Introduction - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/#human-in-the-loop">AgentWorkflow Basic Introduction - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1341920240250126366)** (1 messages): 

> `AI and data operations trends, Procure.FYI, Federal technology spending, Enterprise AI adoption` 


- **Next Phase in AI and Data Operations**: A recent post titled [The End of Big Dumb AI & Data](https://open.substack.com/pub/procurefyi/p/the-end-of-big-dumb-ai-data?r=223ajc&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false) discusses emerging trends in AI and data operations that challenge traditional approaches.
   - It emphasizes a shift towards more intelligent and efficient systems in handling data for better decision-making.
- **Procure.FYI Substack Launch Details**: Procure.FYI, focusing on federal technology spending and enterprise AI applications, was launched **two years ago**.
   - Subscribers are encouraged to review Substack’s [Terms of Use](https://substack.com/tos) and [Privacy Policy](https://substack.com/privacy).



**Link mentioned**: <a href="https://open.substack.com/pub/procurefyi/p/the-end-of-big-dumb-ai-data?r=223ajc&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false">The End of Big, Dumb AI Data</a>: AI Data Operations Enter the Smart Phase: Why Quality &amp; Strategy Now Beat Quantity

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1341459314077859912)** (3 messages): 

> `F24 MOOC Certificates, Advanced Course Certificates` 


- **F24 MOOC Achievements Detailed**: The F24 MOOC had a total of **15,000** students, with **304 trailblazers** 🚀, **160 masters** 🧑‍🏫, **90 ninjas** 🥷, **11 legends** 🏆, and **7 honorees** 🌟.
   - Notably, three honorees were from the ninja tier and four were masters.
- **Advanced Course Certificate Availability**: Members asked if there is a certificate for the advanced course and if it’s possible to obtain one without the F24 MOOC certificate.
   - The response confirmed **yes** to both questions, with more details to be released soon!


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1341458743128494195)** (18 messages🔥): 

> `LangChain Framework, Machine Learning Forecasting with LLMs, LLM Agents Course Content, Evaluating LLM Responses, Feedback on LLM Methodologies` 


- **LangChain Simplifies LLM Applications**: LangChain is a framework designed to streamline the LLM application lifecycle, covering areas such as [development](https://python.langchain.com/docs/introduction/), productionization, and deployment with various components.
   - One member noted that it functions by linking outputs from one LLM as inputs to another, effectively creating chains for enhanced performance.
- **Insights on LLMs and Machine Learning Access**: There was a discussion regarding combining LLM agents with machine learning forecasting models, leading to a suggestion to explore related academic papers for deeper insights, such as those found on [Everscope](https://everscope.ai/?models=Large+Language+Model+%28LLM%29&sort_by=elo_rating&sort_order=desc&page=1&limit=10&timeRange=this-week).
   - Another user pointed out that filtering for all-time might yield the best papers related to LLMs.
- **Uncertainties Regarding Course Content**: A participant inquired whether studying the 2024 LLM agents course is necessary, expressing interest in topics like DSPy not covered this semester.
   - Responses indicated that while previous knowledge could aid understanding current topics, access to [2024 course materials](https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared) might not be essential.
- **Evaluating LLM Responses with Meta Judges**: A member sought clarity on how different slides on LLM meta-judging concepts relate to generating and evaluating responses from LLMs, specifically between slides 89, 97, and 98.
   - The explanation highlighted differences in approaches concerning the need for a Meta Judge in specific scenarios for training and evaluating responses.
- **Disappearance of Course Materials**: Concerns were raised about the sudden disappearance of videos and quizzes, indicating difficulties in keeping up with course content.
   - The importance of asking such questions in appropriate channels was emphasized, providing a general note on the absence of quizzes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://everscope.ai/?models=Large+Language+Model+%28LLM%29&sort_by=elo_rating&sort_order=desc&page=1&limit=10&timeRange=this-week">Everscope - Quant Explorer by Cadenzai</a>: no description found</li><li><a href="https://everscope.ai/?models=Large+Language+Model+%28LLM%29&sort_by=elo_rating&sort_order=desc&page=">Everscope - Quant Explorer by Cadenzai</a>: no description found</li><li><a href="https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared,">LLM Agents MOOC</a>: Large Language Model Agents MOOC F24</li><li><a href="https://python.langchain.com/docs/introduction/">Introduction | 🦜️🔗 LangChain</a>: LangChain is a framework for developing applications powered by large language models (LLMs).
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1341942110936498176)** (3 messages): 

> `Quizzes availability, LLM Agents Hackathon, Course completion, Spring 2025 iteration, Video lectures access` 


- **Quizzes available on MOOC's page**: A member inquired about accessing **quiz1 and quiz2** for the weekend, expressing a desire to catch up after starting late.
   - Another member responded that they should be accessible on the [MOOC’s page](https://llmagents-learning.org/sp25).
- **Quizzes found in announcements**: A further response suggested locating the quizzes on the [Announcements page](https://llmagents-learning.org/f24), emphasizing that they should still be available.
   - The importance of signing up for the [LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/) was also highlighted, suggesting it as a valuable opportunity.
- **Course Completion Announced**: Course staff announced that the current course has completed but video lectures remain available in the syllabus.
   - They encouraged signing up for the [Spring 2025 iteration](https://llmagents-learning.org/sp25) to continue learning.
- **Certificates Released**: All certificates have been released, signaling the end of the course.
   - **Thanks were extended** for a great semester, confirming a positive experience for participants.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>: MOOC, Spring 2025</li><li><a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC, Fall 2024
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1341549530575016037)** (5 messages): 

> `Text Channel Usage, Bot Automation, Screenshot Requests` 


- **Text Channels Buzzing with Activity**: It's uncommon to see all the **text channels** being utilized, as observed by a member who reacted with a smile.
   - The comment highlights an unexpected surge in discussions across the channels.
- **Bot Automation Dominates Conversations**: Another participant remarked that the spam observed is often from **automation bots**, not human activity.
   - This suggests a trend where automation is filling the gaps in channel exchanges.
- **Request for New Capture Channel**: A member requested the creation of a specific channel and noted that it should support **screenshots**.
   - This indicates a need for more dedicated spaces for sharing visual content within the community.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1341928753395138601)** (3 messages): 

> `Profit sharing collaboration, Effects of a world without coffee` 


- **Seeking Profit Sharing Partners**: A member is looking for individuals aged **25-50** to share their identity and collaborate on a profit-sharing opportunity, with anticipated profits between **$100-1500**.
   - This request highlights the values of collaboration and mutual benefit in entrepreneurial endeavors.
- **Exploration of Coffee's Impact**: A request was made for an **essay** discussing the implications of a world without coffee, emphasizing its cultural and economic importance.
   - This topic opens up a discussion on the hypothetical scenarios and societal changes that could occur without this popular beverage.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1341921219322318903)** (11 messages🔥): 

> `Identity Sharing Concerns, Clarity in Communication, Collaboration Vs. Theft, Privacy in Collaboration` 


- **Identity Sharing Sparks Debate**: A user expressed interest in collaborating and sharing profits with individuals aged **25-50**, but others immediately raised concerns about sharing personally identifiable information in a public forum.
   - *Identity theft is this open nowadays?* questioned a user, highlighting the hesitance of some to share private details online without clear purpose.
- **Importance of Clear Communication**: A member advised that clearer writing is essential when sharing opinions to prevent misunderstandings, especially in a text-based medium.
   - They emphasized that improving **positive communication** can lead to amazing collaborations, urging everyone to come together.
- **Collaboration or Theft? The Great Debate**: Responses highlighted a discrepancy between viewing the request for identity sharing as **collaboration** or, conversely, as a potential form of theft.
   - One user firmly stated, *That is collaborating..*, trying to clarify the intentions behind the request.
- **Suspicious Lack of Information**: Concerns were raised about the **lack of details** provided by the original poster, including absence of a website or documentation regarding their proposal.
   - A user described the whole venture as *suspicious*, noting that transparency is crucial for cautious collaboration.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1341832890568282164)** (16 messages🔥): 

> `AI21 API usage, Output formatting issues, Handling special characters in responses, Working with Symfony and PHP` 


- **Seeking Help with AI21 API Integration**: A user sought assistance with using the **AI21 API**, specifically on how to format requests to the `jamba-1.5-large` model and how to structure the API call correctly.
   - Helpful members provided detailed [API reference](https://docs.ai21.com/reference/jamba-15-api-ref) and examples of required headers and parameters needed for successful requests.
- **Clarifying API Output Behavior**: A user queried about the output of the API when trying to solve a mathematical expression, receiving an unexpectedly detailed response.
   - Another member clarified that the output format contains escape characters; adjustments are needed in the code to clean the output before display.
- **Special Characters in API Responses**: Discussion arose regarding the necessity of removing special characters from API responses when using **PHP** with the AI21 API.
   - Members confirmed that the **AI21 Studio** UI is designed to handle these characters, but additional code handling is required for proper output formatting.
- **PHP Integration Challenges**: A user mentioned their challenges working with **Symfony** and PHP to handle the API responses, indicating the need for potential extensive conversions.
   - They also thanked fellow members for their insights on working with and formatting the API outputs effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.ai21.com/reference/jamba-15-api-ref">Jamba 1.5</a>: Jamba-1.5 instruction following chat models</li><li><a href="https://docs.ai21.com/reference">Jamba 1.5</a>: Jamba-1.5 instruction following chat models
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1341488861301444749)** (3 messages): 

> `Self-Supervised Prompt Optimization, Retrieval Augmented Generation, Real-time Information Integration, LLM Performance Improvement` 


- **Self-Supervised Prompt Optimization framework introduced**: A new approach called **Self-Supervised Prompt Optimization (SPO)** aims to discover effective prompts for both closed and open-ended tasks without external references, making it cost-efficient. The framework derives evaluation signals purely from output comparisons, enhancing **LLM reasoning capabilities**.
   - *One participant noted it was crazy for the paper to not mention DSPy until the last paragraph.*
- **Dynamic content integration through Retrieval Augmented Generation**: The study proposes an alternative to traditional retrieval methods by using **standard search engine APIs** to dynamically integrate the latest online information during generative inference. This approach aims to improve generated content quality without relying on a fixed index.
   - The new paradigm involves a **parser-LLM** that determines the need for Internet augmentation and extracts search keywords in a single inference.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.06855">Self-Supervised Prompt Optimization</a>: Well-designed prompts are crucial for enhancing Large language models&#39; (LLMs) reasoning capabilities while aligning their outputs with task requirements across diverse domains. However, manually d...</li><li><a href="https://arxiv.org/abs/2411.19478">Zero-Indexing Internet Search Augmented Generation for Large Language Models</a>: Retrieval augmented generation has emerged as an effective method to enhance large language model performance. This approach typically relies on an internal retrieval module that uses various indexing...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1341527810707034122)** (11 messages🔥): 

> `Synthetic Data Generation with DSPy, Judge-Time Scaling in AI, Conversation History in DSPy Calls, Freezing and Exporting Prompts, Personal Voice Identity Manager` 


- **Exploration of Synthetic Data Generation with DSPy**: A member shared a [link to a GitHub repository](https://github.com/seanchatmangpt/dslmodel) showcasing structured outputs from DSPy and Jinja2, emphasizing its capability in synthetic data generation.
   - Additionally, they highlighted a synthetic data generation pipeline that could be utilized to train ChatDoctor with an attached image for illustration.
- **Judging Time Scaling for Enhanced AI Evaluation**: A member expressed excitement over a new library called Verdict, which focuses on scaling judge-time compute, as shared in a post by [Leonard Tang](https://x.com/leonardtang_/status/1892243653071908949).
   - This release aims to tackle the evaluation limitations in AI, particularly in open-ended and non-verifiable domains, marking a significant shift in AI advancements.
- **Verifying Conversation History Injection in DSPy**: A user inquired whether DSPy automatically injects conversation history into calls, seeking clarity before proceeding with their exploration.
   - Another member confirmed their interest in the topic, hinting at potential complexities they want to avoid.
- **Freezing and Exporting Prompts from DSPy Programs**: A member shared a code snippet showing how to freeze and export all prompts in a DSPy program into message templates, using a default adapter.
   - They noted that while this method is convenient, it may lead to losing control flow logic and suggested alternatives like `program.save()`.
- **Personal Voice Identity Manager Concept Discussion**: A member indicated that the newly announced judge-time scaling library would be ideal for their concept of a Personal Voice Identity Manager.
   - This implies a practical application for the library, highlighting its potential impact on personal identity management within AI frameworks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/leonardtang_/status/1892243653071908949">Tweet from Leonard Tang (@leonardtang_)</a>: First came pre-training scaling; then came inference-time scaling.Now comes judge-time scaling.Despite progress in AI through scaled inference-time compute, AI remains unreliable in open-ended, non-ve...</li><li><a href="https://x.com/DataDeLaurier/status/1891896292650991810">Tweet from Dᴀᴛᴀ Sᴀᴄᴋs (@DataDeLaurier)</a>: @Teknium1 Because they do not have anything new. They are about to slap 4o, o1, o3, Voice, and Sora into RouteLLM and call it GPT-5.I bet they actually use RouteLLM and don&#39;t cite anyone</li><li><a href="https://github.com/seanchatmangpt/dslmodel">GitHub - seanchatmangpt/dslmodel: Structured outputs from DSPy and Jinja2</a>: Structured outputs from DSPy and Jinja2. Contribute to seanchatmangpt/dslmodel development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1341582787106312264)** (4 messages): 

> `Model Testing, Performance Comparison, Computational Complexity` 


- **Mixed experiences with model performance**: One user shared that testing on a **10-year-old GeForce 850M** in Windows 10 yielded 3 tok/s initially but gradually decreased, while Chrome took 2 minutes to load with only 0.2 tok/s.
   - In contrast, another reported using an **RTX 4070** on Windows 11, achieving a much faster rate of **12 tok/s** with a 1.9 seconds time to first token.
- **Concerns with model usability**: A user highlighted that despite the performance with an RTX 4070, the model is still not very usable due to high computational costs and complexity.
   - They mentioned specific issues such as **numerical stiffness** and challenges with nonlinear sub-problems complicating accurate solution retrieval.


  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

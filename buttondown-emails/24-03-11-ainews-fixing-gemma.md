---
id: 35ce32a7-d9d2-418c-b544-6195895b01ea
title: '[AINews] Fixing Gemma'
date: '2024-03-12T00:03:26.597451Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-fixing-gemma
---

<!-- buttondown-editor-mode: plaintext -->> AI News for 3/7/2024-3/11/2024. We checked [**356** Twitters](https://twitter.com/i/lists/1585430245762441216) and **21** Discords (**335** channels, and **6154** messages) for you. Estimated reading time saved (at 200wpm): **734 minutes**. We [added Unsloth AI today](https://news.ycombinator.com/item?id=39671146).


Google's recently released Gemma model was widely known to be unstable for finetuning. Last week, [Daniel Han from Unsloth got some love](https://twitter.com/danielhanchen/status/1765446273661075609) for finding and fixing 8 bugs in the implementation, some of which are being upstreamed. There is a [thread](https://twitter.com/danielhanchen/status/1765446273661075609), [blogpost](https://unsloth.ai/blog/gemma-bugs), and today [Hacker News commentary and Google Colab](https://news.ycombinator.com/item?id=39671146) to follow along, with some [deserved community love](https://twitter.com/karpathy/status/1765473722985771335).

 ![image.png](https://assets.buttondown.email/images/2476c160-45fc-48be-96fc-afc0fbb17dc2.png?w=960&fit=max) 

It is full of extremely subtle numerical precision issues like this:
 ![image.png](https://assets.buttondown.email/images/f5f406fb-cf44-4e2c-bea1-bc6cb5a9c4e4.png?w=960&fit=max) 

Which takes extreme attention to detail to notice. Kudos!

---


**Table of Contents**

[TOC] 


# PART X: AI Twitter Recap

> all recaps done by Claude 3 Opus. Today's output is lightly swyx edited. We are working on antihallucination, NER, and context addition pipelines.

<div><p>Here is a summary of the key topics and themes from the provided tweets, with relevant tweets organized under each category:</p>
<p><strong>Technical Deep Dives</strong></p>
<ul>
<li><a href="https://twitter.com/ylecun/status/1766855439570886798" target="_blank" rel="noopener noreferrer">Yann LeCun explains</a> the technical details of a pseudo-random bit sequence used to pre-train an adaptive equalizer, which is a linear classifier trained with least squares and a descendant of the Adaline (competitor of the Perceptron).</li>
<li>Subtweeting <a href="https://twitter.com/ylecun/status/1766498677751787723?t=90xQ8sGy63D2OtiaoGJuww">a Yann tweet</a>, <a href="https://twitter.com/fchollet/status/1766909709288976630" target="_blank" rel="noopener noreferrer">François Chollet argues</a> that the information bandwidth of the human visual system is much lower than 20MB/s, despite having 1 million optic nerve fibers. He estimates the actual information input is under 1MB/s, and the information extracted by the visual cortex and incorporated into the world model is even lower, measured in bytes per second.</li>
<li><a href="https://twitter.com/nearcyan/status/1766961996451418502" target="_blank" rel="noopener noreferrer">NearCyan feels that</a> search engines provide monotonous sludge with zero actual information, so he now uses LLMs as his primary conduit of information with any semblance of reality.</li>
</ul>
<p><strong>New AI Model Releases &amp; Benchmarks</strong></p>
<ul>
<li><a href="https://twitter.com/AravSrinivas/status/1766931722015531263" target="_blank" rel="noopener noreferrer">Arav Srinivas reports</a> that after 100s of queries on Perplexity with Claude 3 (Opus and Sonnet) as the default model, he has yet to see a hallucination, unlike his experience with GPT-4. Similar reports from others who are switching.</li>
<li><a href="https://twitter.com/Hacubu/status/1766867651165667461" target="_blank" rel="noopener noreferrer">Hacubu benchmarked</a> Anthropic's new Claude-3 models on structured data extraction using LangSmith. The high-end Opus model had no errors over 42 examples and slightly outperformed the previous non-GPT-4 contender, Mistral-Large.</li>
</ul>
<p><strong>Emerging Trends &amp; Reflections</strong></p>
<ul>
<li><a href="https://twitter.com/ylecun/status/1766849709488959911" target="_blank" rel="noopener noreferrer">Yann LeCun reflects on</a> the history of AI, noting that generations of researchers thought the latest paradigm would lead to human-level AI, but it's always harder than expected with no single magic bullet. However, progress is definitely being made and human-level AI is merely a matter of time.</li>
<li><a href="https://twitter.com/Teknium1/status/1766883224876458437" target="_blank" rel="noopener noreferrer">Teknium predicts that</a> people will start breaking down every GPT-based pipeline and rebuild it to work well with Claude instead.</li>
<li><a href="https://twitter.com/_aidan_clark_/status/1766917995098763310" target="_blank" rel="noopener noreferrer">Aidan Clark experiences</a> the emotional rollercoaster of hitting a bug and loving/hating machine learning in quick succession when working on ML projects.</li>
</ul>
<p><strong>Tutorials &amp; How-To Guides</strong></p>
<ul>
<li><a href="https://twitter.com/svpino/status/1766811259901739473" target="_blank" rel="noopener noreferrer">Santiago Valdarrama recorded</a> a 1-hour video on building a RAG application using open-source models (Llama2 and Mixtral 8x7B) to answer questions from a PDF.</li>
<li><a href="https://twitter.com/jerryjliu0/status/1766913924006613329" target="_blank" rel="noopener noreferrer">Jerry Liu demonstrates</a> receipt processing with @llama_index + local models + PaddleOCR.</li>
<li><a href="https://twitter.com/LangChainAI/status/1766885157590675543" target="_blank" rel="noopener noreferrer">Langchain published</a> in-depth documentation on how to customize all aspects of Chat LangChain, in both Python and JS/TS, including core concepts, modifications, local runs, and production deployment.</li>
</ul>
<p><strong>Memes &amp; Humor</strong></p>
<ul>
<li><a href="https://twitter.com/drjwrae/status/1766803741414699286" target="_blank" rel="noopener noreferrer">A meme jokes that</a> it's "happy deep learning is hitting a wall day" for those who celebrate it.</li>
<li><a href="https://twitter.com/fchollet/status/1766865282902151482" target="_blank" rel="noopener noreferrer">François Chollet finds it amusing that</a> the more problems you solve, the more problems you have.</li>
<li><a href="https://twitter.com/Teknium1/status/1766907410982256994" target="_blank" rel="noopener noreferrer">Teknium jokes that</a> Nvidia hates us regarding the challenges of working with their hardware/software for AI workloads.</li>
</ul></div>

---

# PART 0: Summary of Summaries of Summaries

## Claude 3 Sonnet (14B?)

1. **Model Finetuning and Performance Optimization**:
   - [Unsloth AI](https://discord.com/channels/1179035537009545276) discussions centered around **finetuning Gemma** models, dealing with special tokens, and addressing issues like OOM errors. Solutions included updating Unsloth, using `pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"`, and exploring **Gradient Low-Rank Projection (GaLore)** ([ArXiv paper](https://arxiv.org/abs/2403.03507)) for reducing memory usage during LLM training.
   - The **CUDA MODE** community explored techniques like **thread coarsening**, **vectorized memory access**, and **CUDA profiling tools** to optimize performance. Projects like [ring-attention](https://github.com/cuda-mode/ring-attention) and flash decoding were discussed.
   - **Answer.AI** announced the ability to train **70B models locally** using **FSDP + QLoRA** on standard GPUs like RTX 3090 ([blog post](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html)).

2. **AI Model Comparisons and Benchmarking**:
   - Discussions compared models like **Claude Opus**, **GPT-4**, and **Mistral** for coding prowess, with Claude Opus often outperforming GPT-4 in areas like SQL and Rust. Users also anticipated the release of **GPT-4.5/5** and its potential improvements.
   - The **DiscoResearch** community explored using **GPT-4** and **Claude3** as judges for creative writing, developing benchmarks, and comparing models like **Brezn3** and **Dpo** on German datasets.
   - **Gemini** was highlighted for its impressive performance, with a [YouTube video](https://youtu.be/IuehDA1M_Lw) comparing it to Claude Opus and GPT-4 Turbo, noting its superior speed and lower costs.

3. **AI Ethics, Regulation, and Societal Impact**:
   - Concerns were raised about **censorship** and restrictions creeping into AI models like the "Claude 2 self-moderated versions." Discussions touched on balancing free expression with content moderation.
   - The impact of AI on **creativity** and **employment** was debated, with some believing AI will assist rather than replace human creativity, while others anticipated job market shifts.
   - A [Slashdot article](https://yro.slashdot.org/story/24/03/11/185217/us-must-move-decisively-to-avert-extinction-level-threat-from-ai-govt-commissioned-report-say) highlighted U.S. government concerns about frontier AI posing an **extinction-level threat**, suggesting potential regulatory measures.

4. **Open-Source AI Models and Community Contributions**:
   - Anticipation grew around the open-sourcing of models like **Grok** by `@xAI`, as announced by [Elon Musk's tweet](https://x.com/elonmusk/status/1767108624038449405?s=46).
   - **Cohere** introduced **Command-R**, a new retrieval augmented model with a 128k context window and public weight release for research ([blog post](https://txt.cohere.com/command-r/)).
   - Community members shared projects like [Prompt Mixer](https://www.promptmixer.dev/) for building AI prompts, an [open-source AI chatbot](https://github.com/Haste171/langchain-chatbot) using LangChain, and tools like [claudetools](https://github.com/vatsalsaglani/claudetools) for function calling with Claude 3.


## Claude 3 Opus (8x220B?)

<p><br class="Apple-interchange-newline"><span style="color: rgb(242, 242, 241); font-family: -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, Roboto, Oxygen, Ubuntu, Cantarell, &quot;Fira Sans&quot;, &quot;Droid Sans&quot;, &quot;Helvetica Neue&quot;, sans-serif; font-size: medium; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: pre-wrap; background-color: rgb(26, 26, 26); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;"></p>
<ul>
<li><p><strong>Claude Outperforms GPT-4 in Coding Tasks</strong>: Engineers have observed that <strong>Claude Opus</strong> consistently delivers more complete and effective code outputs compared to <strong>GPT-4</strong>, particularly excelling in languages like SQL and Rust, as discussed in the <a href="https://discord.com/channels/974519864045756446/998381918976479273">OpenAI Discord</a>.</p>
</li>
<li><p><strong>Perplexity AI&#39;s Context Retention Struggles</strong>: Users have expressed frustration with <strong>Perplexity AI</strong>&#39;s inability to retain context effectively, often defaulting to base knowledge responses, leading to refund requests and bug reports, as seen in the <a href="https://discord.com/channels/1047197230748151888/1047649527299055688">Perplexity AI Discord</a>. The removal of the 32k context length feature from the roadmap has also raised transparency concerns.</p>
</li>
<li><p><strong>Gemma Models Gain Traction Despite Issues</strong>: While <strong>Gemma models</strong> have shown promise, such as the release of <strong>Ghost 7B v0.9.1</strong> which ranked 3rd on <a href="https://huggingface.co/lamhieu/ghost-7b-v0.9.1">VMLU&#39;s leaderboard</a>, users in the <a href="https://discord.com/channels/1110598183144399058/1111649100518133842">LM Studio Discord</a> have reported technical issues with Gemma models in <strong>LM Studio</strong>, even after the release of custom quantized versions.</p>
</li>
<li><p><strong>Efficiency Breakthroughs in LLM Training and Inference</strong>: Researchers have made significant strides in reducing memory requirements and accelerating LLM training and inference. <strong>GaLore</strong> (<a href="https://arxiv.org/abs/2403.03507">arXiv paper</a>) reduces memory usage by up to 65.5%, while <strong>Answer.AI</strong>&#39;s system using FSDP and QLoRA (<a href="https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html">blog post</a>) enables training 70B models on consumer GPUs. For inference, techniques like <strong>ToDo</strong> (<a href="https://arxiv.org/abs/2402.13573">arXiv paper</a>) can increase Stable Diffusion speeds by 2-4.5x through token downsampling.</span></p>
</li>
</ul>


## ChatGPT (GPT4T)

<div><p>I apologize for that oversight. Here's a revised version incorporating the relevant links inline:</p><ul><li><p><strong>Finetuning Challenges and Solutions in AI Modeling</strong>: The Unsloth AI community tackled <strong>finetuning Gemma</strong>, highlighting issues with special tokens and adapter precision. Recommendations include reinstalling <code>xformers</code> to address errors, suggested via command <code>pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"</code>. The integration of <strong>multi-GPU support</strong> and a <strong>FSDP + QLoRA</strong> system by Answer.AI for training 70B models on gaming GPUs marked significant advancements (<a target="_new" href="https://github.com/oKatanaaa/unsloth">oKatanaaa/unsloth</a>). <strong>Ghost 7B v0.9.1</strong> showcased advancements in reasoning and language, accessible on <a target="_new" href="https://huggingface.co/lamhieu/ghost-7b-v0.9.1">huggingface.co</a>, highlighting Unsloth AI's efficiency improvements during LLM fine-tuning.</p></li><li><p><strong>Emerging AI Technologies and Community Engagement</strong>: OpenAI Discord highlighted <strong>Claude Opus</strong>' superior performance over GPT-4 in coding tasks, spurring discussions on AI consciousness and Claude's capabilities. Technical solutions for GPT-4 bugs and strategies to improve ChatGPT's memory recall were shared, emphasizing the use of an <strong>output template</strong> for achieving consistency in custom models.</p></li><li><p><strong>Model Compatibility and Efficiency in Coding</strong>: LM Studio's discourse revolved around model selection for coding and cybersecurity, noting <strong>Mistral 7B</strong> and <strong>Mixtral</strong>'s compatibility with various hardware. Persistent issues with Gemma models prompted suggestions for alternatives like <strong>Yi-34b</strong>, available on <a target="_new" href="https://arxiv.org/abs/2402.17764">arXiv</a>. Discussions on power efficiency and <strong>ROCM</strong> compatibility underscored the ongoing search for optimal LLM setups, with detailed hardware discussions available at their <a target="_new" href="https://discord.com/channels/1110598183144399058/1153759714082033735/1215624964154331137">hardware discussion channel</a>.</p></li><li><p><strong>Innovative Tools and Techniques for AI Development</strong>: CUDA MODE Discord provided insights into merging CUDA with image and language processing. The community also engaged in self-teaching CUDA and exploring <strong>Triton</strong> for performance improvements. Techniques like <strong>GaLore</strong> and <strong>FSDP with QLoRA</strong> for large model training were discussed, along with shared resources for CUDA learning, including CUDA Training Series on <a target="_new">YouTube</a> and lecture announcements for CUDA-MODE Reductions.</p></li></ul><p>These summaries more accurately reflect the discussions and technical explorations across AI communities, showcasing challenges, innovative solutions, and the collaborative spirit driving advancements in the field, with relevant links provided inline for deeper exploration.</p></div>

---


# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord Summary

**Finetuning Frustrations and Triumphs**: Discussions focused on **finetuning Gemma** created challenges with special tokens and the efficacy of model loading after finetuning, suggesting potential versioning issues and the impact of adapter precision. A recommendation included reinstalling `xformers` with `pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"` to address errors and updating Unsloth as a possible fix for **OOM errors**.

**Unsloth Giveaways and Growth**: The Unsloth community celebrated the **implementation of multi-GPU support** ([oKatanaaa/unsloth](https://github.com/oKatanaaa/unsloth)) and the release of a new **FSDP + QLoRA** system by **Answer.AI** for training 70B models on gaming GPUs. A knowledge sharing exercise for **Unsloth finetuned models on Kaggle** identified key bugs and fixes, and the community also recognized contributors' support on [Ko-fi](https://ko-fi.com/unsloth).

**Boosting Productivity with Unsloth AI**: **Ghost 7B v0.9.1** advanced in reasoning and language, ranking 3rd on VMLU's leaderboard and accessible on [huggingface.co](https://huggingface.co/lamhieu/ghost-7b-v0.9.1). Another significant achievement was reported by `@lee0099`, demonstrating **Unsloth AI's optimizations** resulting in a 2x speedup and 40% memory reduction during LLM **fine-tuning** with no loss in accuracy.

**Celebrating AI Contributions and Cutting-edge Updates**: The Unsloth AI community shared updates and insights, including a **new 0.43.0 release of bitsandbytes** for FSDP support, contributing to the existing finesse of framework operations. AI2 Incubator’s provision of $200 million in AI compute to startups was highlighted, and discussions around **OpenAI’s transparency** surfaced as consequential.

**Welcoming Winds and Gear for Growth**: New Unsloth community members were directed to essential information channels, while suggestions for Unsloth advancements involved integrating features from **Llama-factory** into Unsloth. The prominence of the **Galore** thread was acknowledged, and a GitHub project named **GEAR** was shared, showcasing an efficient cache compression recipe for generative inference ([GEAR on GitHub](https://github.com/HaoKang-Timmy/GEAR)).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Claude Edges Out GPT-4 in Coding Prowess**: Engineers have noted that **Claude Opus** appears to outperform **GPT-4** in providing coding solutions, exhibiting strengths in SQL and Rust. The community has cited Claude's ability to offer more complete code outputs.

- **AI's Existential Question: Consciousness on the Table**: The guild has engaged in debates concerning the potential consciousness of AI, specifically **Claude**. Papers and philosophical views on *universal consciousness* have been referenced, revealing a profound interest in the metaphysical aspects of AI technology.

- **AI Hiccups: Workarounds for GPT-4 Bugs**: Users across the guild have reported **GPT-4** outages and language setting bugs. A widely agreed solution is to switch the language to *Auto-detect* and refresh the browser, which has helped alleviate the issues for many users.

- **Transforming Prompts into Reliable AI Memories**: Discussions have revolved around optimizing ChatGPT's memory recall with prompt structuring. The approach includes formatting advice, like avoiding grammar mistakes and ensuring clarity, and using summaries to cue AI memory.

- **Maximizing Output Consistency across Custom Models**: For achieving consistent outputs from custom GPT models, it's been suggested to use an **output template**. The template should contain variable names that encode summary instructions, aligning well with an engineer's need for standardized results.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Model Selection for Coding and Cybersecurity**: Engineers are exchanging experiences using various models like **Mistral 7B** and **Mixtral** on different systems, including Mac M1 and PCs with Nvidia GPUs. For more detailed hardware and model compatibility discussions, such as running 70B models on a 64GB M2 MacBook with slow response times, engineers are advised to consult the [hardware discussion channel](https://discord.com/channels/1110598183144399058/1153759714082033735/1215624964154331137).

- **GEMMA Models’ Quirks Confirmed**: Technical issues persist with Gemma models in **LM Studio**, even after the release of custom quantized versions. [Yi-34b](https://arxiv.org/abs/2402.17764) with a 200k context window was suggested as a feasible alternative.

- **Explorations in Power Efficiency for LLM Setups**: Community members are actively discussing the power consumption of high-end GPUs like **7900 XTX** and CPU performance, especially AMD 3D cache models. The importance of efficient RAM setups and cooling systems, like Arctic P12 fans, is also noted. For system configuration recommendations, [hardware discussion chat](https://discord.com/channels/1110598183144399058/1153759714082033735/1215624964154331137) is a valuable resource.

- **Desire for Improved LM Studio Features**: Users are requesting enhancements in **LM Studio**, including the ability to view recent models easily and more sophisticated filter capabilities to select models by size, type, and performance. Solutions like using Hugging Face to view recent models with a specific search are being shared while waiting for the platform to expand its features. An example search link is [here](https://huggingface.co/models?sort=created&search=GGUF).

- **ROCM Readiness for Diverse Operating Systems**: Compatibility concerns with **ROCM** on various operating systems, including Windows and non-Ubuntu Linux distributions, have been raised. ROCm's performance on Debian has been described as challenging due to Python version conflicts and AMD's predominant Ubuntu support. Users successfully running models on Windows with **ROCM** have suggested using `koboldcpp` and the override `HSA_OVERRIDE_GFX_VERSION=10.3.0`. 

- **CrewAi vs AutoGen Evaluation for Bot Integration**: As users navigate the complex landscape of bot integrations, with options like **AutoGen** and **CrewAi**, there's active discussion on structural design and compatibility. CrewAi is characterized by its intuitive logic, while AutoGen offers a graphical user interface. Concerns over token costs due to agent loops and API calls are noted for those integrating these systems with GPT.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

**Perplexity's Context Retention Struggles**: Users expressed frustrations over **Perplexity AI's** context handling ability, with complaints about it defaulting to base knowledge responses and subsequent requests for refunds. Concerns were raised about transparency after the removal of the 32k context length from the roadmap.

**Confusion Around API Token Limits**: Queries on the maximum output token length for new models and the absence of the expected 32k context length feature on the roadmap sparked discussions, amidst concerns of documentation inconsistencies and how they might affect API usage and development of projects like an Alexa-like personal assistant.

**New Users Navigate the Pro Plan**: New **Perplexity Pro** users were confused about redeeming promo subscriptions and using the API conservatively to avoid depleting credits, leading to requests for clear guidance on usage tracking.

**Legal, Health, and Tech Discussions on Sharing Channel**: Insightful conversation threads from the **sharing** channel touched on **Apple's legal actions against Epic**, life expectancy concerns, the merits of a specific Super Bowl halftime show, Google's payments to publishers, and discussions on nootropic efficiencies recommending caffeine, L-theanine, and creatine stack.

**Comparative Analysis and Learning**: The community exchanged thoughts on diverse AI services, comparing **Perplexity** to others like **Copilot Pro** and **ChatGPT Pro**, with **Perplexity** drawing praise specifically for its image generation capabilities.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Decoding the Decoder Models**: An inquiry by `@mattlawhon` regarding the implications of using longer sequences during inference with decoder models trained without Positional Encoding was raised. `@vatsadev` clarified that feeding more tokens is possible, though it may lead to errors or nonsensical output, and the question's specificity caused some puzzlement among peers.

- **Creative AI Unleashed**: A new multiplayer game **[Doodle Wars](https://doodlewars.netlify.app)** ventured into neural network-scoring doodles, while discussions on enabling party games with fewer players via multi-modal LLMs took place. The announcement of **Command-R from Cohere** as a new generation model optimized for RAG and multilingual generation was also shared through **[Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01)**.

- **Benchmarks and AI Analysis Expansion**: The **Gemini** AI, designed to understand entire books and movies, was introduced alongside the **[WildBench](https://huggingface.co/spaces/allenai/WildBench)** and **[Bonito](https://huggingface.co/BatsResearch/bonito-v1)** models, proposing new approaches to benchmarking and dataset creation. Discussions also highlighted Lex Fridman's tweet addressing the intersection of AI with power dynamics, although the exact content wasn't provided.

- **Model Parallelism and GPT-next**: The complexities of model parallelism were dissected, with insights on the limitations of current methods and anticipation for GPT-5's release stirring debates. Meanwhile, Cohere's new model release and practical assistance with Genstruct were also hot topics.

- **LLMs at the Forefront**: The ability to train effective chatbots with a curated set of 10k training examples was discussed, referencing insights from the Yi paper found on **[Reddit]( https://www.reddit.com/r/LocalLLaMA/comments/1b9kq9v/01ai_paper_is_a_gem_for_model_trainers/)**. XML tagging was highlighted as an evolving method for precise function call generation, and **[open-webui](https://github.com/open-webui/open-webui)** was recommended as a user-friendly GUI for Claude 3.

- **Quality Data and Quirky Model Responses**: Within **Project Obsidian**, the challenge of maintaining **data quality** was acknowledged. Language models reflecting user-provided assumptions — even whimsical events like a fictional *squirrel uprise* — point to inherent model behaviors worth considering.

- **Focused Discussions for Bittensor**: A prompt reminder was issued to keep the discussion on Bittensor topics, following a scam alert. Questions about primary insights from models produced by the subnet and the mention of an enhanced data generation pipeline, aimed at increasing diversity, indicated ongoing improvements.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Innovative Code Splitting with CodeHierarchyNodeParser**: Users in the LlamaIndex guild discussed the use of `CodeHierarchyNodeParser` for splitting large code files into hierarchies, potentially enhancing RAG/agent performance. The approach has been shared on [Twitter](https://twitter.com/llama_index/status/1766152269874266170).

- **AI Chatbot Challenges and Cosine Similarity Clarifications**: A user sought advice on creating a RAG chatbot using LlamaIndex, citing the [Ensemble Retriever document](https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval.html), while another user clarified the range of cosine similarity, which includes negative values and its implication for similarity score cutoffs in a query engine, referencing [Wikipedia](https://en.m.wikipedia.org/wiki/Cosine_similarity).

- **Handling Ingestion Pipeline Duplication and Conda Install Issues**: Discussions highlighted solutions for ingestion pipelines processing duplicates, solved by using `filename_as_id=True`, while another user reported on and sought help with resolving Conda installation conflicts involving version mismatches and modules not found post-upgrade. 

- **Query Pipeline Storage Queries and PDF Parsing with LlamaParse**: One user inquired about saving pipeline outputs, questioning the feasibility of using Pydantic objects, and another shared informational resources on PDF parsing using LlamaIndex's LlamaParse service through a [YouTube video](https://youtu.be/wRMnHbiz5ck).

- **Engaging Community with User Surveys and AI-enhanced Browser Automation**: LlamaIndex is conducting a **3-minute user survey**, found [here](https://www.surveymonkey.com/r/PNSP3P9), to gather user feedback for improvements while also discussing LaVague, a project by @dhuynh95 utilizing RAG and MistralAI to aid in creating Selenium code from user queries, detailed in [this post](https://twitter.com/llama_index/status/1766508631825235968).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Artefacts Troubling Engineers**: Technical discussions highlighted issues with **high-resolution AI models**, such as discernible artefacts at large resolutions and constraints in smaller models like the 600m. Engineers like `@marianbasti` and `@thejonasbrothers` indicated a shared concern that these limitations might prevent full realization of the models' capabilities.

- **Constructing Advanced Video Scripting Tools**: `@spirit_from_germany` proposed an advanced two-model system for video scripting capable of analyzing and predicting video and audio, recommending concentration on the most popular videos to ensure data quality. The idea was shared through a [Twitter post](https://twitter.com/laion_ai/status/1766596812347941234).

- **Generated Datasets Under Microscope**: `@pseudoterminalx` mentioned the limitations of generated datasets, underlining the potential for being trapped within a specific knowledge corpus and the automated descriptions being constrained by the training of the generating model.

- **CogView3 vs. Pixart - An Incomplete Picture**: The exploration of CogView3's framework, a 3-billion parameter text-to-image diffusion model, was discussed with reference to its [arXiv paper](https://arxiv.org/pdf/2403.05121.pdf). The absence of comparative data with Pixart was noted, bringing into question the assessments of CogView3's capabilities.

- **Loss Spike Dilemmas on MacBooks**: MacBook Pro M1 Max users like `@keda4337` are facing challenges with overheating while training diffusion models, resulting in erratic loss spikes from 0.01 - 0.9 to 500 when resuming training across epochs. Such issues underscore the practical challenges of model training on certain hardware configurations.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Inference API Performance Inquiry**: `@hari4626` reported possible performance issues with Hugging Face's Inference API, expressing concerns about receiving incomplete responses which might affect production suitability.

- **Collaborative Learning on Generative AI**: `@umbreenh` and `@yasirali1149` showed interest in collaborative learning on generative AI for development purposes, while `@wukong7752` looked for guidance on calculating KL-divergence in latent-DM.

- **Algorithm Optimization & AI Advancements**: Discussions about AI models for programming optimization included GitHub Co-Pilot and DeepSeek-Coder instruct. Important resources include discussions about strategic reasoning with LLMs using few-shot examples ([arXiv paper](https://arxiv.org/abs/2305.19165)) and the scope of NLP covered by a [deep learning article](https://www.deeplearning.ai/resources/natural-language-processing/).

- **AI-Created Entertainment and Legal Datasets Released**: **Doodle Wars**, a neural network-scored doodling multiplayer game, was introduced at [Doodle Wars](https://doodlewars.netlify.app), and **Caselaw Access Project** with Harvard Library released over 6.6 million U.S. court decisions data set, accessible via [Enrico Shippole's Tweet](https://x.com/EnricoShippole/status/1766157358672359862?s=20).

- **Mistral Model Bluescreens and Image-to-Text With Problems**: User `@elmatero6` sought advice on CPU-optimizing Mistral to prevent system bluescreens, and `@ninamani` searched for high-performing, accurate open-source models for uncensored image captioning, with **cogvlm** as a suggested option, albeit with noted quantization stability issues.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **The Great BOS Debate**: The use of the **Beginning of Sentence (BOS) token** was under scrutiny, with a consensus that its application varies across different models; no uniform standard exists. [HFLM code](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L716) was discussed regarding the incorporation of 'self.add_bos_token'.

- **Efficiency Leap in Image Diffusion**: **ToDo**, a method to up the ante in Stable Diffusion speeds by up to 2-4.5x through token downsampling, piqued interest, with [related repository](https://github.com/ethansmith2000/ImprovedTokenMerge) and discussion spanning potential implications for AI residency in hardware.

- **Zero-Shot Wonders Overtaking Few-Shots**: Counterintuitive results on MMLU benchmarks showed zero-shot outperforming few-shot, sparking theories on context distraction and an idea to curve test with varying shots.

- **Dependencies and Developments in NeoX Land**: GPT-NeoX development touched on the challenges of dependency management and the necessity of Apex, amid a climate of container complexity and Flash Attention updates.

- **Resources for AI Interpretability Aspirants**: ARENA 3.0 was hailed as a "gem" for embarking on interpretability research, with a juicy link to riches: [ARENA 3.0 Landing Page](https://mango-ambulance-93a.notion.site/ARENA-3-0-Landing-Page-virtual-8f7193af31b445c586efed03e995fb74).

- **On the AI Existential Radar**: A chilling [Slashdot article](https://yro.slashdot.org/story/24/03/11/185217/us-must-move-decisively-to-avert-extinction-level-threat-from-ai-govt-commissioned-report-say) spotlights U.S. government concerns over frontier AI as an extinction-level threat, nudging towards heavy-handed regulatory steps.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **AI Biographer Raises Security Eyebrows**: **@swyxio** recommended trying out [Emma, the AI Biographer](https://getlifestory.com/) but advises caution on privacy, opting for *fake details* during trials. 

- **Leadership Restructured at OpenAI**: After internal turmoil, [OpenAI reinstates Sam Altman](https://openai.com/blog/review-completed-altman-brockman-to-continue-to-lead-openai) as leader and welcomes three new board members, concluding a governance review.

- **Ideogram 1.0's Quiet Entrance**: The potential of [Ideogram 1.0](https://x.com/ideogram_ai/status/1762881284899008564?s=20), a new text rendering tool, is noted by **@swyxio** but seems to have slipped under the radar.

- **Microsoft Research Seeks LLM Interface Feedback**: A new interface standardization proposal from Microsoft, AICI, is currently up for community feedback, particularly on its Rust runtime, as shared in a [Hacker News post](https://news.ycombinator.com/item?id=39670665).

- **State Space Model Could Rival Transformers**: **@swyxio** spotlights "Mamba," a State Space Model, as a Transformer alternative for LLMs, guiding interested AI Engineers to a [visual guide](https://maartengrootendorst.substack.com/p/a-visual-guide-to-mamba-and-state) and [research paper](https://arxiv.org/abs/2312.00752).

- **Latent Space Paper Clubs Activate!**: In various time zones, members are gearing up for **GPT-focused discussions** with preparatory [notes](https://www.gaohongnan.com/transformer/decoder/concept.html) shared and real-time responses to queries during sessions, such as clarifying "causal attention."

- **AI-strategy Sessions Spark Community Sharing**: From tips on **workflow optimization using AI** to sharing **AI-enhanced CLI tools** like `asciinema`, AI in Action Club members are not just engaging but also advocating for future topics like decentralized AI applications. 

- **Asia Engages with GPT-2 Knowledge**: A call to join **Asia's paper-club** members for an **EPIC presentation** on the GPT-2 paper was made by **@ivanleomk**. Further engagement is seen with the recent release of a [Latent Space pod](https://x.com/latentspacepod/status/1766600314419806350?s=20).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord Summary

- **New Roles to Distinguish Discord Members**: Nathan Lambert introduced **new roles** within the Discord guild to separate manually added close friends from subscribers, inviting feedback on the change.

- **GPT-4's Doom Playing Capabilities Published**: GPT-4 demonstrated its ability to play the 1993 first-person shooter game Doom, as described in a paper shared by Phil Pax ([GPT-4 Plays Doom](https://arxiv.org/abs/2403.05468)). The model's complex prompting is highlighted as a key factor in its reasoning and navigation skills.

- **Musk and Open Models Stir Debate**: A tweet by Elon Musk about OpenAI's **Grok** being open-sourced led to discussions around market reactions and the use of "open source," with concerns over OpenAI's ongoing commitment to open models also mentioned. Separately, Cohere's new model **Command-R** sparked anticipation among engineers due to its long context window and public weight release, potentially impacting startups and academia ([Command-R: Retrieval Augmented Generation at Production Scale](https://txt.cohere.com/command-r/)).

- **AI-Centric Dune Casting Game Unfolds**: Discord members humorously cast prominent figures from the AI industry as characters from *Dune*, with suggestions including Sam Altman as the Kwisatz Haderach and Elon Musk as Baron Harkonnen.

- **Reinforcement Learning Podcast and Papers Touted**: Ian Osband's TalkRL podcast episode on **information theory and RL** was recommended ([Ian Osband's episode on Spotify](https://open.spotify.com/episode/0FuKEjteM0cGzy7pznCkAj)), and discussions emerged around a paper on RLHF, PPO, and Expert Iteration applied to LLM reasoning ([Teaching Large Language Models to Reason with Reinforcement Learning](https://arxiv.org/abs/2403.04642)). The theme of consistent quality in RL content was echoed across discussions.

- **Inflection AI Model Integrity Questioned**: After similar outputs were noted between Inflection AI's bot and OpenAI's **Claude-3-Sonnet**, debates ensued over possible A/B testing or model wrappers, exacerbated by Inflection AI's response about its bot Pi remembering previous inputs ([Inflection AI's Clarification](https://fxtwitter.com/inflectionai/status/1766173427441049684?s=46)).

- **Costs and Approaches to Model Training Examined**: The affordability of less than $1,000 for pretraining models like **GPT-2** and the potential for deals on compute, such as Stability AI's speculated sub-$100,000 expenditure for their model's compute, were hot topics. Fine-tuning with books and articles using a masking strategy was also discussed.

- **Sam Altman's Return to OpenAI and Light-Hearted Role Queries**: Sam Altman's return to the OpenAI board prompted discussions and a sprinkling of humor about leadership. Discord roles, including self-nominated goose roles, were jestingly proposed as subscribers' stakes became a topic of amusement.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord Summary

- **Mistral 7b 0.2 Takes the Stage with Speed**: The newly introduced **Mistral 7b 0.2** model is making waves with a **10x performance increase for short outputs and 20x for longer outputs**, and boasts a **32k token context window**. A performance demo can be viewed in a [tweet by OpenRouterAI](https://twitter.com/OpenRouterAI/status/1766147110443909184).

- **Gemma Nitro Offers Efficiency and Economy**: OpenRouter announces a new model, **Gemma Nitro**, with over **600+ tokens per second** speed and pricing set at an affordable **$0.1 per million tokens**. Details are outlined on [OpenRouter's model page](https://openrouter.ai/models/google/gemma-7b-it:nitro).

- **Conversations Heat Up Around AI Censorship**: User concerns rise about censorship potentially affecting AI models, like Claude 2's self-moderated versions, prompting discussions about free expression and the need for uncensored platforms, alongside technical inquiries regarding message formatting and system parameters.

- **Community Innovates with Claude 3 Library**: `@thevatsalsagalni` presents **claudetools**, a library that facilitates function calling with **Claude 3 models**, promoting ease-of-use for developers with Pydantic support. The library is available for community contribution on [GitHub](https://github.com/vatsalsaglani/claudetools).

- **Technical Discussions Abound on Model Limits and Usage**: Users discuss the technical aspects of AI models, delving into topics like GPT-4's token output limitation, the intricacies of Claude API's role message handling, and the utilization of Chat Markup Language (ChatML) in prompt customization. Community-created tools, like a Google Sheets connection app, demonstrate growing engagement and address model accessibility concerns.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **Combining CUDA with Image and Language Processing**: Engineers discussed challenges in concatenating image features with caption layers and the use of linear layers to project image features to the shape of NLP embeddings. Further insights included CUDA's potential for improving machine learning model operations by employing techniques like vectorized additions.

- **Exploring CUDA and Triton Development**: The community is engaging in self-teaching CUDA and exploring tools for performance improvement, such as the Triton language. There's an interest in comparing CUDA's performance to higher-level tools like libtorch and understanding the compilation process involved in `torch.compile`.

- **Advancements in Large Model Training**: Techniques like GaLore and FSDP with QLoRA are discussed for their contribution to reducing memory requirements and enabling the training of large models on standard GPUs. An ArXiv paper covers Gradient Low-Rank Projection, and Answer.AI's blog post provides insights on training a 70b model at home.

- **CUDA Knowledge Sharing and Lecture Announcements**: A YouTube playlist and GitHub repository for the CUDA Training Series were shared, while a call for participation in a CUDA-MODE Reductions lecture was announced, with resources for the lecture available online. Moreover, CUDA novices discussed compilation differences and performance observations across PyTorch versions.
  
- **Job Opportunities and Project Development in CUDA**: A developer is sought to design a custom CUDA kernel, offering a remuneration between $2,000 and $3,000 USD, with prerequisites including experience in algorithmic development and CUDA programming. Conversations also highlighted user projects like building a custom tensor library and the importance of depth in knowledge for CUDA's practical applications.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Innovative Prompt Crafting Desktop App**: User `@tomatyss` introduced [Prompt Mixer](https://www.promptmixer.dev/), a new tool for building, testing, and iterating on AI prompts, offering features like connecting to various models, prompt version tracking, and a guide for [creating custom connectors](https://docs.promptmixer.dev/tutorial-extras/create-a-custom-connector).

- **Enhancements on Langchain**: Users discussed multiple aspects of Langchain such as PDF extraction issues, handling complex logic in templates, wrappers for ChatOllama functions such as [Ollama Functions](https://js.langchain.com/docs/integrations/chat/ollama_functions), execution locations for Langchain Serve, and capturing outputs from routes. Meanwhile, Claude3 support enhancement is in progress as indicated by `@baytaew`, referencing a [Pull Request #18630 on GitHub](https://github.com/langchain-ai/langchain/pull/18630).

- **RAG Tutorial Resources Shared**: Tutorials on improving and utilizing Retrieval Augmented Generation (RAG) were shared by `@mehulgupta7991` and `@infoslack`, providing [videos](https://youtu.be/TlZ5BFx_m3M?si=tVfbYMUQhOVCV8x_) on enhancing RAG with LangGraph and building a chatbot with RAG and LangChain respectively.

- **Open Source Tools for Chatbots and Data Analytics**: An open-source AI Chatbot for conversational data analysis was shared by `@haste171` on [GitHub](https://github.com/Haste171/langchain-chatbot), while `@appstormer_25583` released Data GPTs in Appstorm 1.5.0 for data exploration and visualization with sample GPTs for various industries.

- **Automated Lead Generation & Generation Tools**: `@robinsayar` is developing an automated tool for generating leads using public company information, sparking interest from `@baytaew` who is anticipating the potential impact of such innovation.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **AI Judging Creative Writing**: Skepticism was raised by `.calytrix` about the feasibility of AI models judging creative writing due to parameter limitations. Despite this, **GPT-4** and **Claude3** are being tested with detailed scoring criteria for such a task, a benchmark is being developed by `.calytrix`, and **Mistral large** has been suggested as a potential candidate for an ensemble of AI judges by `bjoernp`.

- **Evo Tackles Genomic Scale**: **Evo**, featuring the StripedHyena architecture, was released by Together AI and the Arc Institute for handling sequences ranging from DNA, RNA, to proteins and supports over 650k tokens. Interest was shown by `johannhartmann` in **AutoMerger** for automatic model merging, though it's currently non-operational.

- **Benchmarking Tools and Strategies Discussed**: `johannhartmann` shared the [tinyBenchmarks dataset](https://huggingface.co/tinyBenchmarks/tinyWinogrande) for efficient AI benchmarking and expressed intent to translate it for broader usability. Insights on benchmarking with the Hellaswag dataset suggested that using 100 data points might be insufficient for detailed comparisons.

- **Advancements and Challenges in German AI Research**: `johannhartmann` provided insights into training models like **Mistral** using the German Orca dataset and addressed technical issues encountered by `crispstrobe` in model merging through a [GitHub commit fix](https://github.com/mayflower/mergekit/commit/cca4a8d91c213b6e5e4ac34b151955187ceff8a4). Additionally, **Brezn3** showed promising improvements over its predecessor given benchmark results, while **Dpo (Domain Prediction Override)** was noted as in progress. Consideration was being given to **DiscoLM** for better benchmarking consistency over previous base models.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **AI Hallucination Challenges Spark Debate**: Engineers explored strategies to minimize AI hallucinations, discussing Yi's report without a consensus on a definition and considering methods like using RAG (Retrieval-Augmented Generation) or employing a manual rewrite of repetitive responses in fine-tuning datasets to mitigate hallucinations. No consensus emerged from the discussion.

- **Mermaid Magic for Code Diagrams**: The use of **Claude** to create [mermaid graphs](https://github.com/mermaid-js/mermaid) from code bases up to 96k tokens was presented as an innovative approach to visualizing code architecture, sparking interest in potential applications for such visualization techniques.

- **Gemma-7b Arrives with a Bang**: The introduction of **Gemma-7b**, enhanced with C-RLFT and fine-tuned using 6T tokens, was heralded as a significant achievement, almost matching the performance of Mistral-based models. The first usable fine-tune is available on [HuggingFace](https://huggingface.co/openchat/openchat-3.5-0106-gemma) and was celebrated in a tweet by [OpenChatDev](https://fxtwitter.com/openchatdev/status/1766516456034861237).

- **Balancing Act Between Gemma and Mistral Models**: A conversation highlighted why **Gemma 7B** was released even though it doesn't outperform **Mistral 7B**, with agreement that each model represents a distinct experiment and Gemma's potential was yet to be fully explored, especially in areas like NSFW content moderation.

- **Community Collaboration in Coding**: Users shared experiences and extended calls for collaboration, particularly around setting up a Docker environment to facilitate development. The tone was comradely, emphasizing the value of collective input in overcoming technical hurdles.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Free AI Tools for Vercel Pro Subscribers**: **Claude 3 Opus** and **GPT-4 vanilla** are now accessible for free to those with Vercel Pro. More information and tools can be found at the [Vercel AI SDK](https://sdk.vercel.ai/).

- **Migrating from OpenAI to Azure SDK**: Transitioning from **OpenAI's SDK** to an Azure-based solution has been a topic of interest for users like `@pantsforbirds`, who are seeking advice on potential migration challenges.

- **XML Enhances Function Calls in Claude**: Users, notably `@res6969`, have noted improved function call performance when using **XML tags** with **Claude**. Conversely, `@pantsforbirds` pointed out that embedding XML complicates sharing prompt generators.

- **Opus Rises Above GPT-4**: Discussions led by users `@jeffreyw128`, `@nosa_.`, and `@vgel` highlighted **Opus** prevails over GPT-4 in delivering smart responses. `@potrock` preferred **Claude's** straightforward prose over GPT's more verbose explanations. Users are eagerly anticipating **GPT-4.5** and **GPT-5** releases, curious about enhancements over current models.

- **Speculations on Google's Potential AI Dominance**: `@jeffreyw128` theorizes Google could dominate in general AI use due to its capability to integrate AI into its existing platforms, like search and Chrome, and offer it at lower costs, possibly introducing a **Generative Search Experience**. However, they suggest that OpenAI may maintain a competitive lead with specialized applications, while Google might prioritize a blend of generative and extractive AI solutions.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **A Groundbreaking Claim in AI Training**: `@baptistelqt` has announced a substantial methodological breakthrough, asserting the ability to **accelerate convergence by a factor of 100,000** by training models from scratch each round. The details of the methodology or verification of these claims have not been provided. 




---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Shout-out to Symbex**: `@bdexter` expressed gratitude for regular usage of **symbex**, with `@simonw` acknowledging the project's fun aspect.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

**Mysterious Mention of InterconnectAI**: A user named .zhipeng appears to have referenced a blog post from Nathan's InterconnectAI, but no specific details or context were provided.

**AI Video Deep Dive Incoming**: An event has been announced focusing on Gen AI Video and the 'World Model', featuring speakers such as [Lijun Yu](https://www.linkedin.com/in/lijun-yu/) from Google and [Ethan He](https://twitter.com/EthanHe_42) from Nvidia, set for **March 16, 2024**, in San Francisco and available on Zoom. Those interested can [RSVP here](https://lu.ma/b0zrw3q3).



---

# PART 2: Detailed by-Channel summaries and links



### Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1215587192231956490) (368 messages🔥🔥): 

- **Inquiry about Finetuning Gemma**: `@kaleina_nyan` and `@starsupernova` discussed issues with finetuning Gemma using the ChatML template, with concerns about whether special tokens like `<start_of_turn>` and `<end_of_turn>` are trained for vanilla pre-trained models. They explore potential fixes and workarounds, such as unfreezing the embedding matrix ([Unsloth Wiki](https://github.com/unslothai/unsloth/wiki#chat-templates)).
  
- **Multi-GPU Support for Unsloth**: `@kaleina_nyan` shared a fork she made on GitHub implementing multi-GPU support to Unsloth ([oKatanaaa/unsloth](https://github.com/oKatanaaa/unsloth)) and further discussed potential issues with numerical results and memory distribution.

- **New FSDP + QLoRA Training System**: `@dreamgen` highlighted a new system released by Answer.AI, capable of training 70B models locally on typical gaming GPUs, not yet sure how it differs from existing methods involving DeepSpeed and QLoRA. 

- **Experiences Sharing Unsloth Finetuned Models on Kaggle**: `@simon_vtr` shared experiences attempting to use Unsloth finetuned models in a Kaggle competition, dealing with issues related to offline packages and inference bugs. A notebook with bug fixes for Gemma models was mentioned for inference use on Kaggle by `@starsupernova`.

- **Thanking Supporters**: `@theyruinedelise` and `@starsupernova` expressed gratitude towards the Unsloth community members for their support on Ko-fi, thanking individual contributors like `@1121304629490221146` and `@690209623902650427` for their donations.

- **Gemma Token Mapping and `generate` Method**: `@kaleina_nyan` and `@starsupernova` engaged in a technical discussion about the function of `map_eos_token` and its implications for the `.generate` method of Gemma models. They identified a potential issue with `generate` not stopping after creating `

**Links mentioned**:

- [Answer.AI - You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html#how-to-use-fsdpqlora): We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.
- [Answer.AI - You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html#how-to-use-fsdpqlo): We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.
- [Google Colaboratory](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing): no description found
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507): Training Large Language Models (LLMs) presents significant memory challenges, predominantly due to the growing size of weights and optimizer states. Common memory-reduction approaches, such as low-ran...
- [Kaggle Mistral 7b Unsloth notebook](https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook/notebook): Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources
- [Kaggle Mistral 7b Unsloth notebook Error](https://www.kaggle.com/code/simonveitner/kaggle-mistral-7b-unsloth-notebook-error?scriptVersionId=166454847): Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources
- [Kaggle Mistral 7b Unsloth notebook Error](https://www.kaggle.com/code/simonveitner/kaggle-mistral-7b-unsloth-notebook-error?scriptVersionId=166450550): Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources
- [Support Unsloth AI on Ko-fi! ❤️. ko-fi.com/unsloth](https://ko-fi.com/unsloth): Support Unsloth AI On Ko-fi. Ko-fi lets you support the people and causes you love with small donations
- [tokenizer_config.json · unsloth/gemma-7b at main](https://huggingface.co/unsloth/gemma-7b/blob/main/tokenizer_config.json): no description found
- [4 apps incroyables qui utilisent l&#39;IA](https://www.youtube.com/watch?v=gGquFWBY5cs): vous allez kiffer (lien vers les apps 👇)👀 À ne pas manquer, Cet OS va vous faire courir acheter un Mac : https://youtu.be/UfrsyoFUXmULes apps présentées da...
- [Home](https://github.com/unslothai/unsloth/wiki#chat-templates): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.
- [GitHub - stanford-crfm/helm: Holistic Evaluation of Language Models (HELM), a framework to increase the transparency of language models (https://arxiv.org/abs/2211.09110). This framework is also used to evaluate text-to-image models in Holistic Evaluation of Text-to-Image Models (HEIM) (https://arxiv.org/abs/2311.04287).](https://github.com/stanford-crfm/helm): Holistic Evaluation of Language Models (HELM), a framework to increase the transparency of language models (https://arxiv.org/abs/2211.09110). This framework is also used to evaluate text-to-image ...
- [Tensor on cuda device 1 cannot be accessed from Triton (cpu tensor?) · Issue #2441 · openai/triton](https://github.com/openai/triton/issues/2441): The code of softmax below is coppied from tutorials to demonstrate that we cannot pass tensors on devices other than &quot;cuda:0&quot; to triton kernel. Errors are: ValueError: Pointer argument (at 0...
- [GitHub - EleutherAI/cookbook: Deep learning for dummies. All the practical details and useful utilities that go into working with real models.](https://github.com/EleutherAI/cookbook): Deep learning for dummies. All the practical details and useful utilities that go into working with real models. - EleutherAI/cookbook
- [GitHub - oKatanaaa/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/oKatanaaa/unsloth): 5X faster 60% less memory QLoRA finetuning. Contribute to oKatanaaa/unsloth development by creating an account on GitHub.

  

---


### Unsloth AI (Daniel Han) ▷ #[welcome](https://discord.com/channels/1179035537009545276/1179039724355211325/1215701340211249212) (4 messages): 

- **A Warm Welcome and Handy Reminders**: `@theyruinedelise` greeted new members with a hearty welcome in multiple messages and encouraged everyone to check out important channels. Members are specifically reminded to read information in channel 1179040220717522974 and to select their roles in channel 1179050286980006030.
  

---


### Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1215651255997702184) (19 messages🔥): 

- **CUDA Conundrums**: User `@maxtensor` reports **Bootstrap CUDA exceptions** in certain scripts within the same environment where others work perfectly, wondering if it's an OS script limitation. Troubleshooting with `@starsupernova` leads to potential GPU visibility issues.

- **Praise for the Framework**: `@maxtensor` expresses admiration for a framework they find innovative, stating it "opens a lot of new doors."

- **New bitsandbytes Version Released**: `@maxtensor` shares a [link to the new 0.43.0 release of bitsandbytes](https://github.com/TimDettmers/bitsandbytes/releases/tag/0.43.0), notable for FSDP support and officially documented Windows installation, but remains cautious about updating their working environment.

- **AI2 Incubator's Massive Compute Giveaway**: `@mister_poodle` shares [news about the AI2 Incubator](https://www.geekwire.com/2024/ai2-incubator-secures-200m-in-ai-compute-resources-for-portfolio-companies/), which has secured $200 million in AI compute resources for its portfolio companies, offering significant support for startups in the AI space.

- **Questions Around OpenAI's AGI Tactics**: `@iron_bound` and `@theyruinedelise` discuss concerns and implications of OpenAI's approach to AI development, particularly in relation to sharing scientific advancements and Elon Musk's stance on OpenAI's alleged shift in openness.

**Links mentioned**:

- [AI2 Incubator secures $200M in AI compute resources for portfolio companies](https://www.geekwire.com/2024/ai2-incubator-secures-200m-in-ai-compute-resources-for-portfolio-companies/): (AI2 Incubator Image) Companies building artificial intelligence models into their software products need a lot of computation power, also known as
- [[ML News] Elon sues OpenAI | Mistral Large | More Gemini Drama](https://www.youtube.com/watch?v=YOyr9Bhhaq0): #mlnews #ainews #openai OUTLINE:0:00 - Intro0:20 - Elon sues OpenAI14:00 - Mistral Large16:40 - ML Espionage18:30 - More Gemini Drama24:00 - Copilot generate...
- [Release 0.43.0: FSDP support, Official documentation, Cross-compilation on Linux and CI, Windows support · TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes/releases/tag/0.43.0): Improvements and New Features:  QLoRA + FSDP official support is now live! #970 by @warner-benjamin and team - with FSDP you can train very large models (70b scale) on multiple 24GB consumer-type G...

  

---


### Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1215592049794093096) (514 messages🔥🔥🔥): 

- **Xformers Installation Issues**: A user `@fjefo` encountered errors related to `xformers` while attempting to use Unsloth AI with Gemma models. They were advised by `@starsupernova` to reinstall `xformers`, and later to use the python package installation command `pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"`.
- **Gemma Model Load and Fine-tuning Challenges**: [Gemma Loading Difficulty] `@patleeman` faced troubles loading a finetuned Gemma 2B model using Unsloth on the vLLM server, getting a KeyError for `lm_head.weight`. After a workaround to skip the key, the model loaded fine, suggesting a potential issue on vLLM's end, as discussed in [this Github issue](https://github.com/vllm-project/vllm/issues/3323).
- **Using HF_HOME Environment Variable with Jupyter**: [HF_HOME Troubles] `@hyperleash` struggled with setting the `HF_HOME` environment variable in Jupyter notebooks for Unsloth. They managed to successfully set it for .py scripts but hit a snag with notebooks, stating no logs were generated for troubleshooting. `@starsupernova` acknowledged the issue, confirmed there are no logs, and provided advice on trying to set the environment variable correctly.
- **Discussions on Finetuned Model Performance**: Users discussed the performance of finetuned models. `@mlashcorp` observed a performance discrepancy with the merged model versus when loading the adapter directly. `@starsupernova` suggested trying `"merged_4bit_forced"` and mentioned precision issues when merging adapters.
- **Downloading and Finetuning Gemma 7B Issues**: `@fjefo` reported issues with downloading and finetuning Gemma 7B but was later able to initiate training successfully. They mentioned OOM errors compared to Mistral 7B and were guided by `@starsupernova` to update Unsloth and consider redownloading via transformers.

**Links mentioned**:

- [Kaggle Mistral 7b Unsloth notebook](https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook): Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources
- [Repetition Improves Language Model Embeddings](https://arxiv.org/abs/2402.15449): Recent approaches to improving the extraction of text embeddings from autoregressive large language models (LLMs) have largely focused on improvements to data, backbone pretrained language models, or ...
- [Gemma models do not work when converted to gguf format after training · Issue #213 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/213): When Gemma is converted to gguf format after training, it does not work in software that uses llama cpp, such as lm studio. llama_model_load: error loading model: create_tensor: tensor &#39;output.wei...
- [KeyError: lm_head.weight in GemmaForCausalLM.load_weights when loading finetuned Gemma 2B · Issue #3323 · vllm-project/vllm](https://github.com/vllm-project/vllm/issues/3323): Hello, I finetuned Gemma 2B with Unsloth. It uses LoRA and merges the weights back into the base model. When I try to load this model, it gives me the following error: ... File &quot;/home/ubuntu/proj...
- [Faster Inference &amp; Training Roadmap · Issue #226 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/226): @danielhanchen In the unsloth Gemma intro blogpost, you mention VRAM increase due to larger MLP size in Gemma compared to Llama and Mistral, and show a graph demonstrating decreased memory usage wh...
- [VLLM Multi-Lora with embed_tokens and lm_head in adapter weights  · Issue #2816 · vllm-project/vllm](https://github.com/vllm-project/vllm/issues/2816): Hi there! I&#39;ve encountered an issue with the adatpter_model.safetensors in my project, and I&#39;m seeking guidance on how to handle lm_head and embed_tokens within the specified modules. Here&#39...
- [Conda installation detailed instructions · Issue #73 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/73): I&#39;m trying to follow the instructions for installing unsloth in a conda environment, the problem is that the conda gets stuck when running the install lines. I&#39;ve tried running it twice, both ...
- [Google Colaboratory](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing): no description found
- [Hastebin](https://hastebin.com/share/olipibuwez.bash): no description found
- [Google Colaboratory](https://colab.research.google.com/drive/1EOa_X5GwKAkPv5a2keJupGowePkHpq-0?usp=sharing): no description found
- [Tutorial: How to convert HuggingFace model to GGUF format · ggerganov/llama.cpp · Discussion #2948](https://github.com/ggerganov/llama.cpp/discussions/2948): Source: https://www.substratus.ai/blog/converting-hf-model-gguf-model/ I published this on our blog but though others here might benefit as well, so sharing the raw blog here on Github too. Hope it...
- [Home](https://github.com/unslothai/unsloth/wiki): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.
- [Hastebin](https://hastebin.com/share/oterufowit.yaml): no description found
- [LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4 - Predibase](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4): LoRA Land is a collection of 25+ fine-tuned Mistral-7b models that outperform GPT-4 in task-specific applications. This collection of fine-tuned OSS models offers a blueprint for teams seeking to effi...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b6723c/comment/ku5r7d3/): no description found
- [Merging QLoRA weights with quantized model](https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930): Merging QLoRA weights with quantized model. GitHub Gist: instantly share code, notes, and snippets.
- [py : add Gemma conversion from HF models by ggerganov · Pull Request #5647 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5647): # gemma-2b python3 convert-hf-to-gguf.py ~/Data/huggingface/gemma-2b/ --outfile models/gemma-2b/ggml-model-f16.gguf --outtype f16  # gemma-7b python3 convert-hf-to-gguf.py ~/Data/huggingface/gemma-...
- [Build software better, together](https://github.com/huggingface/peft/pull/1474.): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [Third-party benchmark · Issue #6 · jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore/issues/6): Hello, thank you very much for such excellent work. We have conducted some experiments using Llama-Factory, and the results indicate that Galore can significantly reduce memory usage during full pa...
- [unsloth/unsloth/save.py at main · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth/save.py#L706): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.

  

---


### Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1216405262638911508) (8 messages🔥): 

- **Ghost 7B v0.9.1 Takes Flight**: User `@lh0x00` announced the release of **Ghost 7B v0.9.1**, touting improvements in reasoning and language capabilities in both Vietnamese and English. It's available for online use and app applications at [huggingface.co](https://huggingface.co/lamhieu/ghost-7b-v0.9.1).
- **Ghost 7B Secures Top Rank**: In a subsequent message, `@lh0x00` mentioned that Ghost 7B v0.9.1 scored high enough to rank **3rd in VMLU's "Leaderboard of fine-tuned models"**.
- **Community Cheers for Ghost 7B**: Users `@starsupernova` and `@lh0x00` exchanged congratulations on the successful launch and high performance of the Ghost 7B model.
- **French AI app insight**: User `@theyruinedelise` shared a **YouTube** video titled "4 apps incroyables qui utilisent l'IA" offering insights into impressive AI apps: [Watch here](https://www.youtube.com/watch?v=gGquFWBY5cs).
- **Unsloth AI Accelerates Fine-tuning**: `@lee0099` discussed finetuning `yam-peleg/Experiment26-7B` on a [NeuralNovel dataset](https://huggingface.co/datasets/NeuralNovel/Unsloth-DPO), highlighting **Unsloth AI**'s optimizations that lead to **2x speedup, 40% memory reduction**, and **0% accuracy degradation during LLM fine-tuning**.

**Links mentioned**:

- [4 apps incroyables qui utilisent l&#39;IA](https://www.youtube.com/watch?v=gGquFWBY5cs): vous allez kiffer (lien vers les apps 👇)👀 À ne pas manquer, Cet OS va vous faire courir acheter un Mac : https://youtu.be/UfrsyoFUXmULes apps présentées da...
- [NeuralNovel/Unsloth-DPO · Datasets at Hugging Face](https://huggingface.co/datasets/NeuralNovel/Unsloth-DPO): no description found

  

---


### Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1216409155074658484) (5 messages): 

- **Suggestion for Unsloth Integration**: User `@imranullah` suggested implementing features from **Llama-factory** into Unsloth AI, implying that such features have proven to be good in their current application.
- **Agreement on Galore's Usefulness:** User `@starsupernova` agreed on the usefulness of the Galore thread, endorsing its potential application.
- **Implementation Ease**: User `@remek1972` humorously remarked on the ease of implementing a certain feature, tagging `@160322114274983936` in the conversation.
- **GitHub Project Shared**: `@remek1972` shared a link to a GitHub repository named **GEAR**, which relates to an *efficient KV cache compression recipe for generative inference of large language models*. [View the GEAR project on GitHub](https://github.com/HaoKang-Timmy/GEAR).

**Links mentioned**:

[GitHub - opengear-project/GEAR: GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM](https://github.com/HaoKang-Timmy/GEAR): GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM - opengear-project/GEAR

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1215677327703539752) (611 messages🔥🔥🔥): 

- **AI-Assisted Coding Comparison**: Users like `@askejm` and `@sangam_k` shared experiences comparing the coding capabilities of **Claude Opus** and **GPT-4**. The consensus seems to be that Claude Opus is better for coding, offering more complete code outputs and performing well in languages like SQL and Rust.

- **Exploring AI's Consciousness**: A discussion led by `@sotiris.b` touched on the belief by some that **Claude** might be conscious. Debates included different views on universal consciousness and whether AI can be considered conscious, with users like `@metaldrgn` and `@dezuzel` discussing papers on the topic.

- **GPT-4's Cutoff and Performance**: User `@webhead` confirmed using test queries that GPT-4's knowledge cut-off is in April 2023 and that while ChatGPT's conversations may be slower, the recall abilities of various models vary, with Google's 1.5 preview showing impressive recall but potential shortcomings in specific tasks.

- **International Access to AI Products**: There were several mentions of difficulties accessing **Claude 3 Opus internationally**, with users `@lightpictures` and `@lazybones3` discussing workarounds. User `@webhead` recommended using **openrouter** for testing different models.

- **Subscription Issues with OpenAI**: User `@arxsenal` described a problem with their **ChatGPT Plus** subscription not being recognized. Others, including `@eskcanta`, suggested ways to resolve it, including clearing cache, using different devices/browsers, and contacting support through the OpenAI help site.

**Links mentioned**:

- [Skm](https://skm.ai/): no description found
- [LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard): no description found
- [Prompt-based image generative AI tool for editing specific details](https://genai.stackexchange.com/questions/1731/prompt-based-image-generative-ai-tool-for-editing-specific-details): I am trying to make some spritesheets using DALLE3, and while the initial generation of spritesheets by DALLE3 are fascinating, I have encountered these problems:&#xA;&#xA;Inconsistent art style(multi...
- [How can we Improve Democracy?](https://medium.com/@h.a.papageorgiou/autoregression-b8cf7aa561d7): Introduction
- [Bland Web](https://Chat.bland.ai): no description found
- [Tweet from Bland.ai (@usebland)](https://x.com/usebland/status/1766250122277712122?s=61): Introducing Bland web. An AI that sounds human and can do anything. 📢  Add voice AI to your website, mobile apps, phone calls, video games, & even your apple vision pro. ⚡️  Talk to the future right ...
- [GitHub - Kiddu77/Train_Anything: A repo to get you cracking with Neural Nets .](https://github.com/Kiddu77/Train_Anything): A repo to get you cracking with Neural Nets . Contribute to Kiddu77/Train_Anything development by creating an account on GitHub.
- [Literal Labs - Cambridge Future Tech](https://camfuturetech.com/portfolio/literal-labs/): Accelerating The New Generation of Energy Efficient AI Literal Labs applies a streamlined and more efficient approach to AI that is faster, explainable, and up to 10,000X more energy efficient than to...

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1215560402021392384) (78 messages🔥🔥): 

- **GPT Outage and Language Setting Bugs**: Multiple users, including `@kor_apucard`, `@dxrkunknown`, `@snolpix`, and `@alessid_55753`, reported issues with GPT not responding. A common fix found by users like `@pteromaple` and confirmed by others such as `@katai5plate` and `@hccren`, was to switch the language preview in settings to *Auto-detect* and refresh the browser.

- **Chat Functionality Troubles and Workarounds**: Issues were not limited to a single browser, as `@dxrkunknown` and `@macy7272` had problems on both web and mobile. Solutions varied with `@pteromaple` suggesting language setting changes, whereas `@winter9149` found deleting old chats could help resume normal operation.

- **Discussions around AI Competitors**: Several users, including `@tsanva`, `@1kio1`, and `@zeriouszhit`, discussed possibly switching to competitor models like Claude, citing context window limitations and confusion in responses from GPT. Concerns were also raised about the lack of comparable features to support Claude compared to those available for GPT.

- **Help and Status Updates**: User `@openheroes` shared a link to OpenAI's status page indicating no current outages, suggesting users ensure they are not on a VPN or blocking connections and referencing the help center for additional support.

- **Payment Queries for GPT Creators**: User `@ar888` inquired about payment for GPT creators, to which `@elektronisade` responded by noting that the official word from OpenAI suggested payments would start in Q1 for US creators, as stated in a blog post.

**Links mentioned**:

[OpenAI Status](https://status.openai.com/): no description found

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1215613812846100533) (90 messages🔥🔥): 

- **In Search of Enhanced ChatGPT Memory**: User `@youri_k` was troubleshooting ChatGPT's ability to recall chat history for context in responses and received advice from `@eskcanta` on how to improve the prompt structure to handle memory, including the suggestion to ask for a summary before ending conversations.
- **ChatGPT Struggles to Sketch for Beginners**: `@marijanarukavina` encountered issues getting ChatGPT to create a simple sketch explaining Boundary Value Analysis; `@eskcanta` suggested using Python tool for better results and provided a step-by-step approach to tweaking the model's output.
- **Delving into GPT-Based UI Generation**: `@dellgenius` probed into how GPT-4 could be used for creating Figma plugins or generating UI elements, with `@eskcanta` sharing a link showcasing GPT-4's potential capabilities in this area.
- **GPT for Homework Assistance? Not Quite**: `@levidog` enquired about extracting questions from an assignment document using chatGPT, but `@darthgustav.` cautioned about the limitations and the ethical considerations of using GPT for homework-related tasks.
- **Achieving Consistent Output in Custom GPTs**: `@iloveh8` sought advice on ensuring consistent responses from custom GPT models, and `@darthgustav.` recommended using an output template with variable names that encode summary instructions.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1215613812846100533) (90 messages🔥🔥): 

- **Efficient Prompt Engineering with GPT**: `@eskcanta` articulated the basic steps for creating efficient prompts, outlining the importance of clarity, language proficiency, and instructing the model with specifics. They advised to **avoid typos, grammar mistakes**, and to communicate in any language well understood by both the user and the AI.

- **Keeping Custom GPT Outputs Consistent**: According to `@darthgustav.`, employing an **output template** with variable names that encode a summary of the instructions can help maintain consistent output from custom GPT prompts.

- **Professional Vocabulary Expansion Challenge**: `@ericplayz` sought assistance in rewriting a paragraph with professional vocabulary while keeping the word count; `@eskcanta` shared an attempted solution and prompted for feedback to assess if the needs were met. The guidance included ensuring that the rewritten text in Romanian maintains length, details, and appropriate tone.

- **JSON Formatting in GPT-4 Discussions**: `@dellgenius` inquired about the use of JSON formatting for organizing responses; `@aminelg` confirmed its utility for structured data, and `@eskcanta` answered questions about creating UI elements and the varying capabilities of the AI model. There was a focus on how GPT models can aid in designing UI elements, provided the AI has been trained on the relevant data or tools.

- **Requests for Assistance Using ChatGPT API**: Users `@youri_k` and `@levidog` requested help with making ChatGPT remember chat history and extracting questions from an assignment document, respectively. They received guidance from `@eskcanta`, who suggested using summaries for history retention and cautioned that the models are not designed to aid with homework, which might lead to inconsistent results.
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1215611692587810876) (407 messages🔥🔥🔥): 

- **Exploring LLM Capabilities**: Users are discussing the capabilities of different models and seeking advice on model choices for specific purposes, such as coding and cybersecurity. They are sharing experiences using models like Mistral 7B and Mixtral on various systems, including Mac M1 and PCs with Nvidia GPUs.

- **Technical Troubleshooting in LM Studio**: Some users, such as `@amir0717`, have encountered errors when trying to load models in LM Studio and are seeking help to resolve issues like "Model operation failed" or Models "did not load properly." Others are offering solutions such as running LM Studio as an administrator or adjusting GPU offload settings.

- **Hardware Limitations and Model Performance**: Users with different hardware specs are asking about the best models to run on their systems. For example, `@mintsukuu` with an 8GB Mac M1 is advised by `@yagilb` to try out 7B models with conservative layer settings, while `@dbenn8` reports running 70B models on a 64GB M2 Macbook, albeit with slow response times.

- **Interest in New and Alternative Models**: There are queries about support for newer models like Starcoder2 and Deepseek-vl in LM Studio. Some users, like `@real5301`, are looking for models with large context windows upwards of 80k tokens, and `@heyitsyorkie` suggests Yi-34b with a 200k context window.

- **Development of LM Studio**: A user mentions the development pace of LM Studio in relation to llama.cpp builds and `@yagilb` confirms an upcoming beta, acknowledging that updates have been slower than desired. It was noted that the development team has expanded from one to three members.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/beta-releases.html)): Find, download, and experiment with local LLMs
- [deepseek-ai/deepseek-vl-7b-chat · Discussions](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat/discussions): no description found
- [Big Code Models Leaderboard - a Hugging Face Space by bigcode](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard): no description found
- [deepse (DeepSE)](https://huggingface.co/deepse): no description found
- [The Muppet Show Headless Man GIF - The Muppet Show Headless Man Scooter - Discover &amp; Share GIFs](https://tenor.com/view/the-muppet-show-headless-man-scooter-george-the-janitor-headless-gif-26660609): Click to view the GIF
- [How to run a Large Language Model (LLM) on your AMD Ryzen™ AI PC or Radeon Graphics Card](https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709): Did you know that you can run your very own instance of a GPT based LLM-powered AI chatbot on your Ryzen™ AI PC or Radeon™ 7000 series graphics card? AI assistants are quickly becoming essential resou...
- [AMD explains how easy it is to run local AI chat powered by Ryzen CPUs and Radeon GPUs - VideoCardz.com](https://videocardz.com/newz/amd-explains-how-easy-it-is-to-run-local-ai-chat-powered-by-ryzen-cpus-and-radeon-gpus): “Chat with Ryzen/Radeon” AMD guides how to run local AI chats with their hardware.  AMD does not have its own tool, like NVIDIA Chat with RTX. NVIDIA came up with a simple tool that can be used to run...
- [GitHub - joaomdmoura/crewAI: Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.](https://github.com/joaomdmoura/crewAI): Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks. - joaomdmoura/cr...
- [AMD explains how easy it is to run local AI chat powered by Ryzen CPUs and Radeon GPUs - VideoCardz.com](https://videocardz.com/newz/amd-explains-how-easy-it-is-to-run-local-ai-chat-powered-by-ryzen-cpus-a): “Chat with Ryzen/Radeon” AMD guides how to run local AI chats with their hardware.  AMD does not have its own tool, like NVIDIA Chat with RTX. NVIDIA came up with a simple tool that can be used to run...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1215557805524918302) (110 messages🔥🔥): 

- **GEMMA Models Puzzlement**: `@boting_0215` encountered issues with all Gemma models not being usable. `@fabguy` confirmed that only a few Gemma quants work, and these are custom quantized versions by the team, pinpointing a potential issue either with LM Studio or the Gemma model. 

- **Troubleshooting Gemma Load Error**: `@honeylaker_62748_43426` received an error when loading a 7B Gemma model and `@heyitsyorkie` affirmed that Gemma models frequently encounter issues, with some quants known to be broken.

- **Searching for the Elusive Slider**: `@jo_vii` sought advice for models suitable for an M2 Max Apple Metal and `@fabguy` suggested using a DeepSeek Coder Q4 or Q5 to leave room for other processes.

- **Model Upload Confusion**: `@anand_04625` couldn't find the file upload button for the Phi model in LM Studio, and `@heyitsyorkie` clarified that model file uploads are not supported.

- **Awaiting Starcoder 2 Update**: `@rexeh` was looking for alternatives to Starcoder 2 on lm studio for ROCm users, and `@heyitsyorkie` indicated that support for Starcoder 2 will come in the future, while currently recommend building llama.cpp independently.

**Links mentioned**:

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [What is retrieval-augmented generation? | IBM Research Blog](https://research.ibm.com/blog/retrieval-augmented-generation-RAG): RAG is an AI framework for retrieving facts to ground LLMs on the most accurate information and to give users insight into AI’s decisionmaking process.
- [Ternary Hashing](https://arxiv.org/abs/2103.09173): This paper proposes a novel ternary hash encoding for learning to hash methods, which provides a principled more efficient coding scheme with performances better than those of the state-of-the-art bin...

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1215955685129453578) (7 messages): 

- **Users Crave a "New Model" Section**: `@justmarky` expressed a wish for an option in LM Studio to view recent models without needing to search, to easily discover what's new.
- **Desire for More Sort Options**: `@purplemelbourne` echoed the sentiment, suggesting additional sort functions like filtering by the model's release date or specific ranges such as the last 6 or 8 months.
- **Hugging Face Workaround Shared**: `@epicureus` shared a workaround by using Hugging Face to view recent models with a [specific search link](https://huggingface.co/models?sort=created&search=GGUF).
- **Existing Channels as Interim Solutions**: `@yagilb` pointed to existing Discord channels `#1111649100518133842` and `#1185646847721742336` as current places to discuss and find information about the latest models.
- **Feature Refinement & Selection Criteria Wishlist**: `@purplemelbourne` requested advanced filtering capabilities in LM Studio to select models by size, type, and performance, specifying a desire to search based on VRAM requirements and ratings.

**Links mentioned**:

[Models - Hugging Face](https://huggingface.co/models?sort=created&search=GGUF): no description found

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1215624964154331137) (147 messages🔥🔥): 

- **Taming Power Consumption with GPUs**: `@666siegfried666` noted that even high-end GPUs like the 7900 XTX don't always reach their Total Board Power (TBP) limit, staying around 140W in their setup, and sought details on real-time TBP draw for the 4060 Ti in LLM. They also highlighted the importance of CPUs, especially AMD 3D cache models, and RAM setups in power efficiency, and advocated for Arctic P12 fans due to their low power draw. 

- **The Race for Efficiency in LLM Systems**: Users discussed balancing price, power, and performance when building LLM systems. `@nink1` talked about the profitability of Apple M3 processors running LLM tasks on a single battery, while `@666siegfried666` brought up regional variations in hardware pricing.

- **Exploring GPU Underclocking & Overclocking**: `@666siegfried666` shared insights into effective undervolting without underclocking, mentioning optimal performance per watt for the 7900 XTX at 2400-2500MHz. `@nink1` considered dynamic underclocking/overclocking in response to workload changes.

- **LLM Performance Enthusiasts Share Configurations**: `@goldensun3ds` related their experience with a substantial load time for a 189K context LLM on their system, and users exchanged advice on hardware setups for LLM, including the efficient operation of AMD GPUs with LLM, and the use of dual GPUs to improve performance.

- **Practical Advice for New LLM Hardware Entrants**: A new user, `@purplemelbourne`, engaged with the community to understand if they could run multiple LLMs on their newly acquired RTX2080Ti GPUs. The conversation evolved into a general discussion about hardware configurations and potential upgrades involving V100 cards and NVLink for running high-memory models.

**Links mentioned**:

- [nvidia 4060 16gb - Shopping and Price Comparison Australia - Buy Cheap](https://www.staticice.com.au/cgi-bin/search.cgi?q=nvidia+4060+16gb&spos=3): no description found
- [Amazon.com: StarTech.com PCI Express X1 to X16 Low Profile Slot Extension Adapter - PCIe x1 to x16 Adapter (PEX1TO162) : Electronics](https://www.amazon.com/gp/aw/d/B0039XPS5W/): no description found

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1216591709971025931) (7 messages): 

- **Token Overflow Troubles**: `@jarod997` experienced gibberish responses in **Win Beta 4 (0.2.10)** when the chat reaches a multiple of the token overflow amount such as 2048, 4096, etc.
- **Context Overflow Policy Check**: `@jedd1` suggested checking the **Context Overflow Policy** settings and also mentioned changes might not be prominent but do occur semi-regularly.
- **Upgrade Recommendation Discussion**: `@jedd1` and `@fabguy` both recommended upgrading to the newer **0.2.16** version which might resolve the issue noted by `@jarod997`.
- **Beta vs. Stable Release Confusion**: `@jarod997` couldn't find the suggested version on LMStudio.ai, before clarifying they need to use the **Beta** due to their machine's support for AVX and not AVX2.
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1216368336506454058) (1 messages): 

- **Debating the Best Bot Integration**: `@purplemelbourne` is seeking advice on which integration to commit to between **AutoGen, CrewAi, ChatDev**, or any other options. They have **AutoGen installed** but have not executed their first run yet.
  

---


### LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1216369052424077402) (3 messages): 

- **MemGPT Shared Knowledge Base Query**: `@purplemelbourne` asked if MemGPT can have a shared knowledge base across different programming models for tasks like bug fixing, considering using KeyMate for integration.
- **Practicality of Integrating GPT-4 with MemGPT**: `@nahfam_` replied that while it's theoretically possible, the cost associated with using the GPT-4 API would be prohibitive. They suggest cleaning up MemGPT outputs with BeautifulSoup4 and Python to make it more manageable.
- **Cost Concerns with KeyMate Integration**: `@nahfam_` expresses skepticism about the sustainability of KeyMate’s business model, costing $60 a month for a GPT-4 128k powered chat, given the per-request token cost and potential rapid depletion of token allowance. 
- **TOS Disapproval for KeyMate**: `@purplemelbourne` comments on the harshness of KeyMate's Terms of Service, providing a rather grim analogy to highlight their broad power of account termination.
  

---


### LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1215770551386112061) (91 messages🔥🔥): 

- **ROCM on Debian vs Ubuntu**: `@quickdive.` discussed the challenges of using ROCm on non-Ubuntu distros like Debian, highlighting **Python version conflicts** and installation hurdles. The user finds dual-booting necessary due to **AMD's official support** being mainly for Ubuntu.
- **Windows Shows Promise for ROCm**: `@omgitsprovidence` mentioned successfully running language models on Windows with an AMD GPU through `koboldcpp`, while `@ominata_` shared a workaround using `'HSA_OVERRIDE_GFX_VERSION=10.3.0'` for the RX 6600XT, suggesting users are finding creative solutions for ROCm on Windows.
- **Performance Inquiries and Comparisons**: In discussions about performance, `@sadmonstaa` reported that their **6950XT** was slower than their **5900x** when offloading with ROCm. Others like `@666siegfried666` had success with older AMD models, hinting at varying experiences among users.
- **Stable Diffusion on AMD**: `@aryanembered` boasted about the capabilities of ROCm, mentioning it was possible to run **Stable Diffusion** on AMD hardware without DirectML, posing a significant ease-of-use advancement.
- **Dual-Booting Due to Compatibility Issues**: Several users, including `@sadmonstaa`, lamented over the necessity of dual-booting due to the compatibility issues of certain software with Linux, even while preferring it. They discussed the implications of ROCm's performance and occasional system crashes across different operating systems and setups.

**Links mentioned**:

- [GPU and OS Support (Windows) — ROCm 5.5.1 Documentation Home](https://rocm.docs.amd.com/en/docs-5.5.1/release/windows_support.html): no description found
- [Docker image support matrix — ROCm installation (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker-image-support-matrix.html): no description found
- [Arch Linux - gperftools 2.15-1 (x86_64)](https://archlinux.org/packages/extra/x86_64/gperftools/): no description found

  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1215693927278186597) (4 messages): 

- **Innovating with a Multi-Agent Framework**: `@pefortin` is developing a complex framework where a front-facing agent clarifies user tasks, a project manager agent breaks down tasks into atomic units, HR expert agents create specialized personas for each task, and an executor runs the operation. The system also includes evaluators to ensure task resolution and fit, but it is currently running slowly and underperforming.
  
- **Soliciting Structure Feedback**: `@wolfspyre` reached out to `@pefortin` to offer feedback on the structural design of the multi-agent framework being developed.

- **Seeking Compatibility Between Agent Systems**: `@purplemelbourne` inquired about the compatibility between AutoGen and CrewAi, expressing a desire to understand which system would be optimal for use without significant time investment.

- **Contrasting AutoGen and CrewAi**: `@jg27_korny` pointed out that AutoGen and CrewAi have different setups, with CrewAi having an easy and intuitive logic, while AutoGen offers a graphical interface. They advised using these systems with the GPT API for best performance and cautioned about the token cost due to potential agent loops.
  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1215563637515751445) (595 messages🔥🔥🔥): 

- **Perplexity's Context Window Woes**: Users like `@layi__` and `@sebastyan5218` expressed frustration with how Perplexity AI handles context, stating that the service struggles to retain awareness and often defaults to base knowledge responses, leading to requests for a refund and bug report submissions.
- **Pro Subscription Puzzles**: New Perplexity Pro users like `@lu.ciry` encountered confusion around redeeming a promo subscription, prompting exchanges with `@icelavaman` for clarification on why their discount code was not appearing during the checkout process.
- **AI Chatbot Curiosity**: Users such as `@nihal_57646` inquired about creating their own AI chatbots and possibly sharing them with Perplexity, to which `@icelavaman` explained that Perplexity is not a chatbot provider, first suggesting using Collections as an alternative.
- **Translation Trials**: `@reborn09` discussed the challenge of translating a large Korean text file to English with Perplexity, with `@codelicious` advising on how to maintain context over multiple chapters and mention of a possible API use for automation in the translation process.
- **Discussions on AI Comparisons**: There were mixed reviews about different AI services, with users like `@13376666666666666666666666666669` criticizing Copilot Pro and praising Perplexity for image generation, while `@twelsh37` provided more comprehensive comparisons across various platforms like ChatGPT Pro, Gemini, and Copilot.

**Links mentioned**:

- [Jeff Bezos&#x27;s investment in Perplexity AI has nearly doubled in value in a few months as Google challenger nears $1B unicorn status](https://fortune.com/2024/03/10/jeff-bezos-perplexity-ai-tech-investment/): The Amazon founder joined a funding round in January that valued the AI search startup at over $500 million.
- [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions): no description found
- [Waiting Waiting Patiently GIF - Waiting Waiting patiently Waiting for you - Discover &amp; Share GIFs](https://tenor.com/view/waiting-waiting-patiently-waiting-for-you-waiting-on-you-gif-15489516379864441176): Click to view the GIF
- [Stonks Chart GIF - Stonks Chart Stocks - Discover &amp; Share GIFs](https://tenor.com/view/stonks-chart-stocks-going-up-gif-15813050): Click to view the GIF
- [The Internet Goes EXTINCT as Gen AI Takes Over | The Dark Forest Internet &amp; Proving Your &quot;Humanness&quot;](https://www.youtube.com/watch?v=3NN5L-f0cDo): Get on my daily AI newsletter 🔥https://natural20.beehiiv.com/subscribe[News, Research and Tutorials on AI]See more at:https://maggieappleton.com/forest-talk...
- [Bloon AI](https://bloon.ai): Redefining Intelligent Learning
- [Loom | Free Screen &amp; Video Recording Software](https://www.loom.com/share/6be018c033f8466184ea3903a15e63aa): Use Loom to record quick videos of your screen and cam. Explain anything clearly and easily – and skip the meeting. An essential tool for hybrid workplaces.
- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/1b9aa5a/comment/ktut7h2/?utm_source=sh): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/1b9aa5a/comment/ktv6xhn/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/1b9aa5a/perplexity_is_consistently_challenged_by/): no description found
- [Tweet from Elon Musk (@elonmusk)](https://fxtwitter.com/elonmusk/status/1767108624038449405?t=HqsmcmViAZl6L-U8AtO9FQ&s=19): This week, @xAI will open source Grok
- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/1b9aa5a/comment/ktut7h2/?utm_source=share&utm_medium=mweb3x&utm_name=mweb3xcss&utm_term=1&utm_content=share_button): no description found
- [GitHub - brendenlake/MLC: Meta-Learning for Compositionality (MLC) for modeling human behavior](https://github.com/brendenlake/MLC): Meta-Learning for Compositionality (MLC) for modeling human behavior - brendenlake/MLC
- [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350): We investigate the ability of language models to perform compositional reasoning tasks where the overall solution depends on correctly composing the answers to sub-problems. We measure how often model...

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1215644530309603370) (38 messages🔥): 

- **Epic vs. Apple Legal Battle Update**: `@jeffreyhammer` shared insights on the legal spat involving **Apple terminating Epic's developer account**. You can read more about the developments [here](https://www.perplexity.ai/search/Apple-terminates-Epic-BaeX6c9hQ0u9hd9jAan_tw).
  
- **Life Expectancy Concerns**: `@nippy_lovelace` delved into the topic of **life-span** and its contributing factors. Dive into the conversation [here](https://www.perplexity.ai/search/Life-span-of-rfNTuVklS3e3PuMXaZdTDw).

- **Super Bowl Showdown**: According to `@johnmooredesign`, a particular Super Bowl halftime show stands out as the greatest. Opinions or the name of the show? Check [here](https://www.perplexity.ai/search/The-greatest-Super-VfLhIQtrRGGfec5GaJXUYg).

- **Monetization in the Tech World**: `@pintobean8071` delved into the issue of **Google paying publishers** for the content. Details of the arrangement can be found [here](https://www.perplexity.ai/search/Google-pays-publishers-RUm4WAH_SbOIdUxFe1_Uww).

- **Nootropics Efficiency Discussed**: `@sevonade4` introduced Claude 3 Opus, discussing nootropics that work including a stack of **caffeine, L-theanine, and creatine**. Interested in cognitive enhancement? Start [here for nootropics](https://www.perplexity.ai/search/Nootropics-that-works-egohN6BzQ96akat9VsV1BA#0) and [here for the stack](https://www.perplexity.ai/search/Caffeine-Ltheanine-and-VP0cM97PQMyZvXPAkiHjEg#0).
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1215563200360349696) (10 messages🔥): 

- **Confusion Over Max Output Token Length**: User `@hex8529` asked about the maximum output in tokens for new models, noting that only context length is visible. `@brknclock1215` responded, suggesting that the context window length minus the query and search results is effectively the max output.

- **Missing 32k Context Length on Roadmap**: `@dogemeat_` inquired about the apparent removal of the 32k context length feature from the roadmap, expressing concern over the lack of acknowledgement about this change.

- **New API User Seeking Guidance**: `@thbk_32074`, a newcomer to the API, questioned whether light use through Raycast would deplete the $5 credit and asked if there's a way to track usage.

- **Clarification on Model Output Limitations**: `@leoesq` clarified that many models have maximum output limits of 3-8k tokens despite larger context windows, which are further influenced by finetune behavior, to which `@brknclock1215` acknowledged possible documentation inconsistencies.

- **Seeking Assistance for Personal Assistant Project**: User `@shine0252` sought help to improve an Alexa-like personal assistant project using the pplx API for more concise and memory-capable interactions, and `@dogemeat_` provided suggestions, mentioning `sonar` models for concise replies and advising on storing conversations for memory.
  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1216389570279505921) (4 messages): 

- **Curiosity About Decoder Models**: `@mattlawhon` asked for insights regarding the use of longer sequences during inference when the decoder model was trained without Positional Encoding (PE).
- **Open-Ended Question Leaves Peers Puzzled**: `@vatsadev` sought clarification on what `@mattlawhon` meant by referring to the use of longer sequences in decoder models.
- **Clarification on Decoder Constraints**: `@vatsadev` confirmed that it is possible to feed more tokens to a decoder model at inference, but warned that it may lead to errors or nonsensical output.
  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1215605074747723796) (39 messages🔥): 

- **Doodle Wars Game Announcement**: `@om7059` shared a new venture called [Doodle Wars](https://doodlewars.netlify.app), a *multiplayer game* where players' doodles are scored by a neural network. 
- **AI-assisted Party Games**: `@denovich` discussed how a multi-modal LLM could potentially allow playing the party game *Telestrations* with fewer than 4 players. 
- **Physics Data with Genstruct**: `@ee.dd` mentioned working on physics data using Genstruct and pondered the amount of data needed before attempting a training run.
- **Convergence Acceleration Method for Neural Networks**: `@baptistelqt` announced a new method that could purportedly accelerate the convergence of any neural network by a factor of 10000. 
- **Introduction of Cohere's New Generative Model**: `@1vnzh` shared a link to Hugging Face, presenting Command-R from Cohere as a 35 billion parameter model optimized for RAG and multilingual generation, [C4AI Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01).

**Links mentioned**:

- [Doodle Wars](https://doodlewars.netlify.app): no description found
- [CohereForAI/c4ai-command-r-v01 · Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01): no description found
- [Command-R: Retrieval Augmented Generation at Production Scale](https://txt.cohere.com/command-r/): Command-R is a scalable generative model targeting RAG and Tool Use to enable production-scale AI for enterprise.  Today, we are introducing Command-R, a new LLM aimed at large-scale production worklo...
- [Mystic.ai](https://www.mystic.ai/): Enterprise-grade auto-ops for machine learning
- [Genstruct 7B Instruction Generation Model](https://www.youtube.com/watch?v=H6xon8K4Ius): Genstruct 7B is an instruction-generation model, designed to create valid instructions given a raw text corpus. This enables the creation of new, partially s...
- [Tweet from Andrej Karpathy (@karpathy)](https://x.com/karpathy/status/1766509149297189274?s=46): Reading a tweet is a bit like downloading an (attacker-controlled) executable that you instantly run on your brain. Each one elicits emotions, suggests knowledge, nudges world-view.  In the future it ...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1215679696189132840) (13 messages🔥): 

- **Gemini Unlocks Book-Level Reasoning**: `@shashank.f1` highlighted a discussion with the Hugging Face community about sparse mixture of models and introduced [**Gemini**](https://youtu.be/IuehDA1M_Lw), which is an AI capable of processing the content of entire books and movies in a single prompt. The linked YouTube video discusses Gemini's capabilities and its comparison to other large language models, including being 20x cheaper than GPT-4.

- **WildBench Benchmark for Instruction Generation**: `@mister_poodle` shared a link to the [**WildBench**](https://huggingface.co/spaces/allenai/WildBench) benchmark on Hugging Face, which could be seen as a call for a new type of benchmark to assess instruction generation in AI.

- **Bonito for Synthetic Dataset Creation**: Continuing the benchmark theme, `@mister_poodle` also introduced [**Bonito**](https://huggingface.co/BatsResearch/bonito-v1), a model for converting unannotated text into task-specific training datasets, which has implications for both pretrained and instruction-tuned language models.

- **Lex Fridman Tweets About AI and Power**: `@mautonomy` brought to attention [a tweet by Lex Fridman](https://twitter.com/lexfridman/status/1766497567909585104), which potentially covers AI's intersection with power and social dynamics (specific content of the tweet was not provided).

- **A Philosophically Optimistic AI Server**: `@norabelrose` shared an invitation to a Discord server dedicated to discussions on AI, philosophy, technology, open source, and an optimistic future, also aiming to critique AI pessimism. The link to join is [here](https://discord.gg/Ss4Bwkvd), and `@max_paperclips` acknowledged the invitation with thanks.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/Ss4Bwkvd): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Is Cosine-Similarity of Embeddings Really About Similarity?](https://arxiv.org/abs/2403.05440): Cosine-similarity is the cosine of the angle between two vectors, or equivalently the dot product between their normalizations. A popular application is to quantify semantic similarity between high-di...
- [BatsResearch/bonito-v1 · Hugging Face](https://huggingface.co/BatsResearch/bonito-v1): no description found
- [How can we Improve Democracy?](https://medium.com/@h.a.papageorgiou/autoregression-b8cf7aa561d7): Introduction
- [Gemini supports 1M+ tokens and 20x cheaper than GPT4 😮 ~ Unlock ideas from the technical paper](https://youtu.be/IuehDA1M_Lw): Here is a quick summary comparing Gemini, Claude Opus and GPT-4 Turbo to find out why you should be interested in Gemini 1.5 Pro♦️On speed 💨 ~ It takes 1 se...
- [AI2 WildBench Leaderboard - a Hugging Face Space by allenai](https://huggingface.co/spaces/allenai/WildBench): no description found

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1215567662441173042) (395 messages🔥🔥): 

- **Model Parallelism Confusion**: `@mihai4256` shared a link about model parallelism that initially caused confusion, but `@teknium` clarified that Qlora has only worked model serial via device map auto and that Deepspeed has its own quant format. The discussion included comments from `@rtyax`, `@stefangliga`, and others.

- **Claude Conscious Project Plans Amidst Python Woes**: Various users, including `@mihai4256`, `@teknium`, `@gabriel_syme`, and `@fred.bliss`, discussed their plans and experiences with the Claude Conscious project, with `@mihai4256` expressing frustration with Python dependencies and `@gabriel_syme` creating a web page frontend in 25 minutes without web dev knowledge.

- **Big Plans for GPT-5's Release**: Users speculated on GPT-5's potential release date, with predictions ranging from within 56 hours by `@mautonomy` to after the U.S. elections as per `@ee.dd`. `@night_w0lf` mentioned a new model, Deepseek-VL, that is flying under the radar.

- **New Releases and Tools**: `@gabriel_syme` announced that Cohere released a new RAG/tool use model with weights on Hugging Face. `@euclaise` helped `@tonic_1` fix a prompt format for Genstruct, and `@.interstellarninja` teased a new recursive function-calling LLM for local GPUs.

- **Deepseek Making Strides**: `@night_w0lf` highlighted Deepseek-VL, a 7B model with promising performance, even beating and matching larger models on certain benchmarks. They also endorsed the academic knowledge benchmark MMMU and shared a paper link.

**Links mentioned**:

- [Tweet from interstellarninja (@intrstllrninja)](https://fxtwitter.com/intrstllrninja/status/1767296447756828953?s=20): recursive function-calling LLM dropping to your local GPU very soon...
- [Errors in the MMLU: The Deep Learning Benchmark is Wrong Surprisingly Often](https://derenrich.medium.com/errors-in-the-mmlu-the-deep-learning-benchmark-is-wrong-surprisingly-often-7258bb045859): Datasets used to asses the quality of large language models have errors. How big a deal is this?
- [Unlimited Power - Star Wars GIF - Power Unlimited Power Emperor Palpetine - Discover &amp; Share GIFs](https://tenor.com/view/power-unlimited-power-emperor-palpetine-revenge-of-the-sith-gif-5266473): Click to view the GIF
- [CohereForAI/c4ai-command-r-v01 · Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01): no description found
- [Rate limits](https://docs.anthropic.com/claude/reference/rate-limits): no description found
- [We Dont Do That Here Black Panther GIF - We Dont Do That Here Black Panther Tchalla - Discover &amp; Share GIFs](https://tenor.com/view/we-dont-do-that-here-black-panther-tchalla-bruce-gif-16558003): Click to view the GIF
- [Tonic/Genstruct · Fix attempt 2](https://huggingface.co/spaces/Tonic/Genstruct/discussions/2): no description found
- [emozilla/LWM-Text-1M-GGUF · Hugging Face](https://huggingface.co/emozilla/LWM-Text-1M-GGUF): no description found
- [emozilla/LWM-Text-1M-mpe64k · Hugging Face](https://huggingface.co/emozilla/LWM-Text-1M-mpe64k): no description found
- [Tweet from bayes (@bayeslord)](https://x.com/bayeslord/status/1765865268595593707?s=46): like truthfully the last time I used a 7B param model was 2009
- [Tweet from Sam Altman (@sama)](https://fxtwitter.com/sama/status/1766311274089185323): patience jimmy. it will be worth the wait.  ↘️ Quoting Jimmy Apples 🍎/acc (@apples_jimmy)   Openai is nothing without its drama.  Now this is out of the way let’s move the fuck on to the release alre...
- [Copium Cat GIF - Copium Cat - Discover &amp; Share GIFs](https://tenor.com/view/copium-cat-gif-27161395): Click to view the GIF
- [Lipsync](https://github.com/DenchiSoft/VTubeStudio/wiki/Lipsync): VTube Studio API Development Page. Contribute to DenchiSoft/VTubeStudio development by creating an account on GitHub.
- [Genstruct - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/Genstruct): no description found
- [Power Level 800o. Wait, It&#039;S Over 9..Gif GIF - Power level 800o. wait It&#039;s over 9. Label - Discover &amp; Share GIFs](https://tenor.com/view/power-level-800o.-wait-it%27s-over-9.-label-text-poster-gif-6048690988960107821): Click to view the GIF
- [GitHub - teknium1/ShareGPT-Builder](https://github.com/teknium1/ShareGPT-Builder): Contribute to teknium1/ShareGPT-Builder development by creating an account on GitHub.
- [Yann Lecun: Meta AI, Open Source, Limits of LLMs, AGI &amp; the Future of AI | Lex Fridman Podcast #416](https://youtu.be/5t1vTLU7s40?si=HS3WrupXGw_xBvmb): Yann LeCun is the Chief AI Scientist at Meta, professor at NYU, Turing Award winner, and one of the most influential researchers in the history of AI. Please...
- [mlx-lm-notebooks/mlx_genstruct_notebook_dataset_pipeline.ipynb at main · fblissjr/mlx-lm-notebooks](https://github.com/fblissjr/mlx-lm-notebooks/blob/main/mlx_genstruct_notebook_dataset_pipeline.ipynb): Apple MLX language model (mlx-lm) notebooks, exploration and tinkering - fblissjr/mlx-lm-notebooks
- [no title found](https://derenrich.medium.com/errors-in-the-mmlu-the-): no description found
- [Gen AI Video Breakout and World Model by EntreConnect - #Sora #Genie #VideoPoet #V-JEPA #LTXStudio #AnimateDiff · Luma](https://lu.ma/b0zrw3q3): Join us for a groundbreaking event that dives deep into the heart of Gen AI Video! This isn&#x27;t just another tech talk; it&#x27;s a journey into the future. We will also provide dial-in options, wh...
- [Add support for control vectors by vgel · Pull Request #5970 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5970): Many thanks to Nous Research, whose support and collaboration made this work possible! This PR introduces a new activations hacking technique, control vectors (also known as steering vectors, conce...

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1215558107388710912) (175 messages🔥🔥): 

- **On the Hunt for AI Papers**: `@main.ai` and `@atgctg` sparked a discussion about the contents of the Yi paper, highlighting that 10k well-curated training examples could suffice for effective chatbot finetuning, according to a Reddit post detailing the paper's takeaways ([source]( https://www.reddit.com/r/LocalLLaMA/comments/1b9kq9v/01ai_paper_is_a_gem_for_model_trainers/)).

- **Tokenizer Troubles**: `@stoicbatman` brought up a conundrum about the feasibility of replacing or adding a language-specific tokenizer to a pre-trained GPT-2 model. `@teknium` and `@stefangliga` contributed to the idea that while tokens could be added, outright replacing the tokenizer would negate prior learning and possibly necessitate retraining from scratch.

- **XML Magic for Function Calls**: The conversation around inducing LLMs to output function calls enclosed in XML tags was animated, with the team of `@.interstellarninja` and `@teknium` sharing their success in the precise generation of function calls and discussing the use of `ufghfigchv`'s tool sampler for increased output trustworthiness.

- **Guided Model Inference with Libraries**: A discussion led by `@sundar_99385`, `@.interstellarninja`, and `@ufghfigchv` delved into the utility of libraries like outlines and SG-lang for guiding model inference. The collective insight pointed towards the benefits of precompiling grammars and using schemas derived from function signatures to improve reliability.

- **Query on LLM GUI Frontends**: `@vodros` seeks recommendations for open-source GUI/frontend compatible with Claude 3, and `@quicksort` suggests trying out the [open-webui](https://github.com/open-webui/open-webui) which offers user-friendly WebUI for LLMs.

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/github/unaidedelf8777/function-sampler): no description found
- [Answer.AI - You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html): We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.
- [Use XML tags](https://docs.anthropic.com/claude/docs/use-xml-tags): no description found
- [Poor Man GIF - Poor Man - Discover &amp; Share GIFs](https://tenor.com/view/poor-man-gif-23343928): Click to view the GIF
- [Trendyol/Trendyol-LLM-7b-base-v0.1 · Hugging Face](https://huggingface.co/Trendyol/Trendyol-LLM-7b-base-v0.1): no description found
- [emozilla/LWM-Text-1M-GGUF · Hugging Face](https://huggingface.co/emozilla/LWM-Text-1M-GGUF): no description found
- [emozilla/LWM-Text-1M-mpe64k · Hugging Face](https://huggingface.co/emozilla/LWM-Text-1M-mpe64k): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b9kq9v/01ai_paper_is_a_gem_for_model_trainers/): no description found
- [no title found](https://medium.com/@gilinachum/cost-of-llm-continued-pre-training-): no description found
- [scratchTHOUGHTS/selfgenREFLECT.py at main · EveryOneIsGross/scratchTHOUGHTS](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/selfgenREFLECT.py): 2nd brain scratchmemory to avoid overrun errors with self. - EveryOneIsGross/scratchTHOUGHTS
- [$ Cost of LLM continued pre-training](https://medium.com/@gilinachum/cost-of-llm-continued-pre-training-0c1998cb44ec): How much will it cost you to do continued pre-training for a small (7B) LLM?
- [GitHub - edmundman/OllamaGenstruct](https://github.com/edmundman/OllamaGenstruct): Contribute to edmundman/OllamaGenstruct development by creating an account on GitHub.
- [Google Colaboratory](https://colab.research.google.com/github/unaidedelf8777/function-sampler/blob/main/notebooks/Tool_Call_Sampler_demo.ipynb): no description found
- [GitHub - unaidedelf8777/function-sampler: Logit Sampler for Function calling LM&#39;s. Making it probabilistically impossible to generate incorrect function calls!](https://github.com/unaidedelf8777/function-sampler.git): Logit Sampler for Function calling LM&#39;s. Making it probabilistically impossible to generate incorrect function calls! - unaidedelf8777/function-sampler
- [GitHub - open-webui/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)](https://github.com/open-webui/open-webui): User-friendly WebUI for LLMs (Formerly Ollama WebUI) - open-webui/open-webui

  

---


### Nous Research AI ▷ #[collective-cognition](https://discord.com/channels/1053877538025386074/1154961277748256831/1216721515312320574) (3 messages): 

- **Flash Attn Query Redirected**: `@pradeep1148` inquired about how to disable flash attention in Axolotl. `@teknium` informed that the channel is archived and suggested to ask the question in another specified channel `<#1154120232051408927>`.
  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1215742206086873138) (3 messages): 

- **Acknowledging Data Quality Concerns**: `@gabriel_syme` noted that **data quality** is a significant challenge.
- **Models Echo Provided Assumptions**: `@kainan_e` pointed out that language models often simply **"agree"** with the sentiment or assumption provided by the user, potentially fabricating events like a fictional "uprising of the squirrels" in Namibia.
  

---


### Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1216090871456333904) (3 messages): 

- **Scam Alert in Bittensor Channel**: User `@teknium` warned `<@930423397366792202>` that their recent post is considered a scam and this channel should only be used to discuss Bittensor related topics.
- **In Search of Insights on Bittensor's Subnet Outputs**: `@vincentweisser` inquired about the primary insights from the models produced by the subnet.
- **Enhancements in Bittensor Data Generation Pipeline**: `@teknium` responded that there is an elaborate data generation pipeline under development which aims to improve upon the current models, highlighting that the existing pipeline isn't providing the necessary diversity.
  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1215712405515141173) (10 messages🔥): 

- **Hierarchical Code Splitting Innovation**: `@ryanpeach` was recognized for their `CodeHierarchyNodeParser`, which splits large code files into hierarchies, enhancing **RAG/agents**. This approach is discussed in a [tweet](https://twitter.com/llama_index/status/1766152269874266170).
- **Live QA over Dynamic File Systems**: Anup Surendran and Berke Can Rizai are featured for their **@streamlit** blogpost showcasing how to build a QA system on a dynamic Google Drive/Sharepoint using **@pathway_com**. Learn about the live ETL pipeline in the complete [tweet](https://twitter.com/llama_index/status/1766265545236848975).
- **AI-Powered Browser Automation**: @dhuynh95's project, LaVague, makes use of **RAG** and **local embeddings + Mixtral** from **@MistralAI** and **@huggingface**, aiming to produce Selenium code from user queries. The agent, functioning as a browser copilot, is discussed [here](https://twitter.com/llama_index/status/1766508631825235968).
- **User Survey Call-to-Action**: LlamaIndex is conducting a **3-minute user survey** to gather valuable feedback and input supported by a reminder [tweet](https://twitter.com/llama_index/status/1766536043258642833).
- **Enhanced RAG with Tree-Structures**: @parthsarthi03 offers insights on using tree-structures to improve RAG pipeline functionality for complex questions, as highlighted in their [latest webinar](https://twitter.com/llama_index/status/1766632206301294830).
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1215564322059583579) (376 messages🔥🔥): 

- **Chatbot Creation Query**: `@o3omoomin` asked how to create a RAG chatbot using the Llama index, specifically looking for frameworks and examples of already implemented RAG chatbots for deployment purposes. They referenced the Ensemble Retriever document and highlighted challenges faced when questions unrelated to the document content are asked ([issue link](https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval.html)).
- **Cosine Similarity Confusion**: `@icsy7867` discussed the range of cosine similarity, questioning whether it's 0-1 or could include negative values and sought clarification for implementing a similarity score cutoff in a query engine ([cosine similarity background](https://en.m.wikipedia.org/wiki/Cosine_similarity)).
- **Ingestion Pipeline Duplicates**: `@mato8792` raised issues with repeated document processing by ingestion pipelines despite using the same data, which was eventually resolved by correctly including `filename_as_id=True` to manage document duplicates effectively.
- **Conda Install Conflicts**: `@rachel_001.` reported a problem with version conflicts during conda installation and encountered issues with modules not being found post-upgrade, which led to troubleshooting including the use of a fresh virtual environment.
- **Saving Pipeline Outputs**: `@node_0` inquired about saving intermediate or final outputs from a Query Pipeline to a local directory and specifically asked if a Pydantic object can be used as part of the pipeline, which led to `@cheesyfishes` clarifying this wasn’t possible yet but is planned for future development.

**Links mentioned**:

- [Prefill Claude's response](https://docs.anthropic.com/claude/docs/prefill-claudes-response): no description found
- [no title found](https://www.secinsights.ai/): no description found
- [Google Colaboratory](https://colab.research.google.com/github/Arize-ai/phoenix/blob/main/tutorials/tracing/llama_index_tracing_tutorial.ipynb): no description found
- [no title found](https://llamahub.ai/l/readers/llama-index-readers-snowflake?from=): no description found
- [no title found](https://llamahub.ai/l/llama-packs/llama-index-packs-snowflake-query-engine?from=): no description found
- [Ingestion Pipeline + Document Management - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline.html): no description found
- [Query Pipeline over Pandas DataFrames - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_pandas.html#download-data): no description found
- [Customizing LLMs within LlamaIndex Abstractions - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html#example-using-a-huggingface-llm): no description found
- [Cosine similarity - Wikipedia](https://en.m.wikipedia.org/wiki/Cosine_similarity): no description found
- [Starter Tutorial - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html): no description found
- [gist:7f54b5ae756b5362b3ec0871b845eeac](https://gist.github.com/thoraxe/7f54b5ae756b5362b3ec0871b845eeac): GitHub Gist: instantly share code, notes, and snippets.
- [OrdalieTech/Solon-embeddings-large-0.1 · Hugging Face](https://huggingface.co/OrdalieTech/Solon-embeddings-large-0.1): no description found
- [Sentence Embedding Optimizer - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/OptimizerDemo.html#sentence-embedding-optimizer): no description found
- [LlamaIndex user survey](https://www.surveymonkey.com/r/PNSP3P9): Take this survey powered by surveymonkey.com. Create your own surveys for free.
- [Observability - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/module_guides/observability/observability.html): no description found
- [no title found](https://llamahub.ai/l/llama-packs/llama-index-packs-fuzzy-citation?from=): no description found
- [llama_index/llama-index-packs/llama-index-packs-fuzzy-citation/llama_index/packs/fuzzy_citation/base.py at 3e5d0a146fcda01a984818d381f31a19287aead8 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/3e5d0a146fcda01a984818d381f31a19287aead8/llama-index-packs/llama-index-packs-fuzzy-citation/llama_index/packs/fuzzy_citation/base.py#L29): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [Node Postprocessor Modules - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors.html#similaritypostprocessor): no description found
- [Joaquin Dominguez / discord_bot · GitLab](https://gitlab.com/j-dominguez9/discord_bot): GitLab.com
- [kapa.ai - Instant AI Answers to Technical Questions](https://www.kapa.ai/): kapa.ai makes it easy for developer-facing companies to build LLM-powered support and onboarding bots for their community. Teams at OpenAI, Airbyte and NextJS use kapa to level up their developer expe...
- [Node Postprocessor - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/root.html#id2): no description found
- [Ensemble Retrieval Guide - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval.html): no description found
- [Document Stores - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/module_guides/storing/docstores.html): no description found
- [LlamaIndex Sessions: 12 RAG Pain Points and Solutions](https://www.youtube.com/watch?v=EBpT_cscTis): We’re excited to feature Wenqi Glantz for a personal walkthrough video of her popular “12 RAG Pain Points and Solutions” blog post, which is the most compreh...
- [llama_index/llama-index-integrations/llms/llama-index-llms-mistralai/llama_index/llms/mistralai/base.py at d63fec1c69a2e1e51bf884a805b9fd31ad8d1ee9 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/d63fec1c69a2e1e51bf884a805b9fd31ad8d1ee9/llama-index-integrations/llms/llama-index-llms-mistralai/llama_index/llms/mistralai/base.py#L72): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [[Bug]: Intermittent 400 - Invalid parameter Error for Messages with Role tool · Issue #10493 · run-llama/llama_index](https://github.com/run-llama/llama_index/issues/10493): Bug Description We are encountering intermittent 400 errors with the message &quot;Invalid parameter: messages with role &#39;tool&#39; must be a response to a preceding message with &#39;tool_calls&#...
- [GitHub - run-llama/sec-insights: A real world full-stack application using LlamaIndex](https://github.com/run-llama/sec-insights): A real world full-stack application using LlamaIndex - run-llama/sec-insights
- [Vector Stores - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html#vector-store-options-feature-support): no description found
- [GitHub - Arize-ai/phoenix: AI Observability &amp; Evaluation - Evaluate, troubleshoot, and fine tune your LLM, CV, and NLP models in a notebook.](https://github.com/Arize-ai/phoenix?tab=readme-ov-file#tracing-with-llamaindex): AI Observability &amp; Evaluation - Evaluate, troubleshoot, and fine tune your LLM, CV, and NLP models in a notebook. - Arize-ai/phoenix
- [GitHub - NVIDIA/NeMo-Guardrails: NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational systems.](https://github.com/NVIDIA/NeMo-Guardrails): NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational systems. - NVIDIA/NeMo-Guardrails
- [NeMo Guardrails, the Ultimate Open-Source LLM Security Toolkit](https://medium.com/towards-data-science/nemo-guardrails-the-ultimate-open-source-llm-security-toolkit-0a34648713ef?sk=836ead39623dab0015420de2740eccc2): Exploring NeMo Guardrails’ practical use cases

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1216131938990034994) (4 messages): 

- **PDF Parsing Simplified**: `@datasciencebasics` shared a [YouTube video](https://youtu.be/wRMnHbiz5ck) titled **"Super Easy Way To Parse PDF | LlamaParse From LlamaIndex | LlamaCloud"**, providing an overview of LlamaParse and LlamaCloud services for easy PDF parsing.
- **Exploring Code with LlamaIndex**: `@andysingal` posted a [blog post](https://medium.com/ai-advances/unleashing-the-power-of-code-a-journey-with-llamaindex-and-code-hierarchy-node-parser-d8ac5fcced8d) titled **"Unleashing the Power of Code: A Journey with LlamaIndex and Code Hierarchy Node Parser"**, discussing the benefits of organizing extensive code files.
- **Matryoshka Learning Paper Discussion Invite**: `@lien_61024` extended an invitation for a paper discussion on **[Matryoshka Representation Learning](https://lu.ma/wmiqcr8t)**, featuring experts Aditya Kusupati and Aniket Rege, hosted by Jina AI.
- **Searching for an Open Source GUI**: `@vodros` inquired about recommended open source GUI/frontends that are compatible with **Claude 3**, expressing a desire to move away from Chatbox for something more user-friendly.

**Links mentioned**:

- [Super Easy Way To Parse PDF | LlamaParse From LlamaIndex | LlamaCloud](https://youtu.be/wRMnHbiz5ck): In this video, I will first briefly explain what LlamaParse is all about. I will also talk about LlamaCloud from LlamaIndex. LlamaParse is a state-of-the-art...
- [Matryoshka Representation Learning: Paper discussion · Zoom · Luma](https://lu.ma/wmiqcr8t): Join us for an insightful hour as we delve into the fascinating world of Matryoshka Representation Learning. Presented by the knowledgeable Aditya Kusupati and the astute Aniket Rege, and...
- [Unleashing the Power of Code: A Journey with LlamaIndex and Code Hierarchy Node Parser](https://medium.com/ai-advances/unleashing-the-power-of-code-a-journey-with-llamaindex-and-code-hierarchy-node-parser-d8ac5fcced8d): Ankush k Singal

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1215592659075473471) (302 messages🔥🔥): 

- **AI Tech Jargon and Disdain for Ineffective Tools**: In the flurry of discussions, `@ignizherz`, `@pseudoterminalx`, and `@nodja` shared disdain for ineffective adversarial tools like Glaze and Nightshade, suggesting they’re not practically safeguarding content as claimed. The conversation turned to speculations on why this ineffectiveness might persist, with a focus on the misguided yet genuine intentions of these tools’ creators.
- **Debating Artist Infringement and 'Worm' Threats**: Discussions by `@vrus0188`, `@astropulse`, and others focused on the exaggerated threat posed by an AI 'worm' and misleading articles about AI's negative impact on industry and environment. This content often includes hyperbolic language and recycles a set of doom-oriented topics.
- **Creative LLMs, Publishing Ethics, and OpenAI's SD3 Anticipation**: A diverse set of topics emerged, such as the effectiveness of LLMs in creative writing (`@nodja` and `@chad_in_the_house`), ethics of publishing (`@drhead` and `.undeleted` discussing Glaze's submission strategy), and excited anticipation for OpenAI's SD3 release expressed by `.undeleted`.
- **Misinformation on LLMs and Technological Advancements**: The conversation included critiques of misinformation in academic journals (`@progamergov` and `.undeleted` lamenting poor peer review standards) and mentions of technological advancements, including an ultra-low power AI chip (`@chad_in_the_house`) and a 'Complementary-Transformer' from KAIST.
- **Discussing AI's Impact on Creativity and Employment**: The chat touched on the impact of AI on creative processes and employment, with `@ignizherz`, `@astropulse`, and `@nodja` expressing thoughts on AI opening doors for non-artists and the changing job market, as well as sharing the belief that AI will not replace human creativity but assist in it.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39643168): no description found
- [Korean researchers power-shame Nvidia with new neural AI chip &mdash; claim 625 times less power draw, 41 times smaller](https://www.tomshardware.com/tech-industry/artificial-intelligence/korean-researchers-power-shame-nvidia-with-new-neural-ai-chip-claim-625-times-less-power-41-times-smaller): Claim Samsung-fabbed chip is the first ultra-low power LLM processor.
- [360° Panorama Viewer Online](https://renderstuff.com/tools/360-panorama-web-viewer/): Online Panorama 360 Viewer. An easy way to View &amp; Share 360-degree pictures for free. VR ready. 360 image viewer instantly creates interactive full-screen immersive VR spherical 360 3d panoramas i...
- [SDXL MS Paint Portraits - v1.0 | Stable Diffusion LoRA | Civitai](https://civitai.com/models/183354/sdxl-ms-paint-portraits>): Do you love M S P a i n t ? Do you love crappy paintings and portraits? The MS Paint Portrait LoRA might be something for you! It is a bit hard to ...
- [MS Paint LoRA (Pony Diffusion V6 XL) - v1.0 | Stable Diffusion LoRA | Civitai](https://civitai.com/models/323771/ms-paint-lora-pony-diffusion-v6-xl>): ⚠DO NOT USE SCORE TAGS IN YOUR POSITIVE PROMPT⚠ I only bothered to generate 1 example image (with an unpublished earlier version of the model) so I...
- [Tech has graduated from the Star Trek era to the Douglas Adams age](https://interconnected.org/home/2024/02/21/adams): Posted on Wednesday 21 Feb 2024. 1,196 words, 13 links. By Matt Webb.
- [What Luddites can teach us about resisting an automated future](https://www.technologyreview.com/2024/02/28/1088262/luddites-resisting-automated-future-technology/): Opposing technology isn’t antithetical to progress.
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/1bb8uwx/an_ai_worm_has_been_developed_to_burrow_its_way/): no description found
- [Snapshot K-12 Art 2020-21 School Year](https://docs.google.com/presentation/d/1BAbiqX0t7Zbl0NkQ8l6jAEKdazyKEK0qqj_kjmMHmEg/edit?pli=1#slide=id.p): Snapshot K-12 Artwork (in-person, remote, hybrid) 2020-2021
- [Some new SD 3.0 Images.](https://old.reddit.com/r/StableDiffusion/comments/1bbdxg6/some_new_sd_30_images/): Posted in r/StableDiffusion by u/protector111 • 840 points and 231 comments

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1215665446997459035) (75 messages🔥🔥): 

- **Model Resolution and Detail Concern**: Users including `@marianbasti` and `@thejonasbrothers` expressed concerns about the quality of high-resolution models, noting artifacts at large resolutions and the limitations of smaller models like the 600m discussed. There's a shared sentiment that the full potential of these models may not be reached due to these issues.
  
- **Potential for Advanced Video Scripting**: User `@spirit_from_germany` proposed a two-model system for advanced video scripting that could analyze and predict video and audio content, sharing this concept via a [Twitter link](https://twitter.com/laion_ai/status/1766596812347941234). `@louie1943` suggested that focusing such training on the most popular videos in a category might ensure the use of quality data.

- **Concerns of Quality in Generated Datasets**: User `@pseudoterminalx` raised concerns about the limitations of generated datasets, mentioning that they keep you trapped within a certain knowledge corpus and that automated descriptions are limited to what the generating model was trained on.

- **Exploring the CogView3 Framework**: `@twoabove` and `@thejonasbrothers` discussed the 3-billion parameter text-to-image diffusion model detailed in CogView3's [arXiv paper](https://arxiv.org/pdf/2403.05121.pdf). While recognizing improvements, `@thejonasbrothers` noted that comparisons with Pixart were absent, limiting understanding of CogView3's full potential relative to other models.

- **Discussion on Efficient Models**: Conversations around *Efficient Large Language Model Adapters* (ELLA) and its comparison with other models like SD3 were touched upon by `@chad_in_the_house`, `@vrus0188`, and `@thejonasbrothers`. They speculated about performance and scalability, with `@thejonasbrothers` indicating that SD3's linear approach could make it the defining model for text-to-image generation.

**Links mentioned**:

- [ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment](https://ella-diffusion.github.io/): no description found
- [Breaking News: Run Large LLMs Locally with Less RAM and Higher Speed through llama.cpp with QuIP#](https://medium.com/@andreask_75652/breaking-news-run-large-llms-locally-with-less-ram-and-higher-speed-through-llama-cpp-with-quip-5d2a450d58d0): A recent update to llama.cpp enables a new “crazy-sounding, but useable” 2-bit quantization for LLMs — QuIP: Quantization with Incoherence…

  

---


### LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1215978801557798982) (2 messages): 

- **Diffusion Model Training Troubles on Mac**: User `@keda4337` is experiencing issues while training a diffusion model on their MacBook Pro M1 Max as the laptop overheats. They mentioned that when resuming training from saving every epoch, the **training loss spikes unprecedentedly from 0.01 - 0.9 to 500**.
  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1215575727831777300) (168 messages🔥🔥): 

- **Incomplete Responses and Inference API**: `@hari4626` asked if the Inference API always provides incomplete responses, a concern suggesting potential performance issues with the models when used in production.
- **Guidance on Fine-tuning Models**: `@bohaska` seeks advice on a user-friendly way to fine-tune a small GPT model for laptop use, leading to suggestions such as checking out "Ollama," but still needing assistance on the fine-tuning aspect.
- **Optimizing Code with AI**: `@techintermezzo` inquired about the best AI model to optimize shader programming for a beginner, prompting a detailed discussion about using models such as GitHub Co-Pilot and DeepSeek-Coder instruct, as well as references to several AI coding benchmarks and literature.
- **DM Permission Settings in Discord**: Users `@chongdashu` and `@lunarflu` discussed how to enable and disable direct messaging permissions on Discord for bot interactions, with `@lunarflu` clarifying that one can disable DMs after obtaining a Verified role without affecting functionality.
- **IPFS as a Model Backup Solution**: `@endomorphosis` debated the merits of hosting AI models on IPFS to mitigate potential government regulations, discussing with `@lunarflu` about backup strategies and mirroring Hugging Face repositories without explicit approval for domain names or usage.

**Links mentioned**:

- [Wikipedia, the free encyclopedia](https://en.wikipedia-on-ipfs.org/wiki/): no description found
- [OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement](https://arxiv.org/abs/2402.14658): The introduction of large language models has significantly advanced code generation. However, open-source models often lack the execution capabilities and iterative refinement of advanced systems lik...
- [Logging in to HuggingFace from Jupyter notebook without interactive prompt](https://medium.com/@yashsk8/logging-in-to-huggingface-from-jupyter-notebook-without-interactive-prompt-2cb945b4905c): In a recent project, I came across a troubling setup problem. Being a student who wants to learn and contribute, but who is short of funds…
- [Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code](https://arxiv.org/abs/2311.07989): In this work we systematically review the recent advancements in code processing with language models, covering 50+ models, 30+ evaluation tasks, 170+ datasets, and 700+ related works. We break down c...
- [US Must Move 'Decisively' To Avert 'Extinction-Level' Threat From AI, Gov't-Commissioned Report Says - Slashdot](https://yro.slashdot.org/story/24/03/11/185217/us-must-move-decisively-to-avert-extinction-level-threat-from-ai-govt-commissioned-report-says): The U.S. government must move &#34;quickly and decisively&#34; to avert substantial national security risks stemming from artificial intelligence (AI) which could, in the worst case, cause an &#34;ext...
- [Hugging Face – The AI community building the future.](https://huggingface.co/settings/tokens): no description found
- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview#managing-secrets): no description found
- [google/gemma-7b · Hugging Face](https://huggingface.co/google/gemma-7b): no description found
- [social-post-explorers/README · Feature Request : Add Posts to Collections](https://huggingface.co/spaces/social-post-explorers/README/discussions/30): no description found
- [DoNotPay&#039;s AI lawyer stunt cancelled after multiple state bar associations object](https://mashable.com/article/donotpay-artificial-intelligence-lawyer-experiment): The robot lawyer was swiftly deactivated by real lawyers.
- [CohereForAI/c4ai-command-r-v01 · Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01): no description found
- [Command-R: Retrieval Augmented Generation at Production Scale](https://txt.cohere.com/command-r/): Command-R is a scalable generative model targeting RAG and Tool Use to enable production-scale AI for enterprise.  Today, we are introducing Command-R, a new LLM aimed at large-scale production worklo...
- [Adding a Sign-In with HF button to your Space](https://huggingface.co/docs/hub/spaces-oauth): no description found
- [Sign in with Hugging Face](https://huggingface.co/docs/hub/oauth): no description found
- [no title found](https://imagepipeline.io/): no description found
- [wav2vec2-codebook-indices/scripts/helpers/w2v2_codebook.py at master · fauxneticien/wav2vec2-codebook-indices](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42): Contribute to fauxneticien/wav2vec2-codebook-indices development by creating an account on GitHub.
- [notebooks/diffusers at main · huggingface/notebooks](https://github.com/huggingface/notebooks/tree/main/diffusers): Notebooks using the Hugging Face libraries 🤗. Contribute to huggingface/notebooks development by creating an account on GitHub.
- [My views on “doom” — LessWrong](https://www.lesswrong.com/posts/xWMqsvHapP3nwdSW8/my-views-on-doom): I’m often asked: “what’s the probability of a really bad outcome from AI?” …

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1215571025186521118) (8 messages🔥): 

- **Warming Up to Generative AI**: User `@umbreenh` articulated a keen interest in using **generative AI** for data analytics development and is open to suggestions and assistance.
- **Let's Learn Together**: In response to `@umbreenh`, `@yasirali1149` expressed a desire to join forces in the journey of learning about **generative AI**.
- **Searching for KL-Divergence Guidance**: `@wukong7752` inquired about any available tutorials specifically for calculating **KL-divergence in latent-DM (LDM)**.
- **Discussing Optimization Strategies**: `@sajjadrahman56` mentioned diving into **optimization techniques for ML models** with `@refik0727` showing interest in learning from his experiences.
- **ML Newbie Seeks Script Usage Help**: `@210924_aniketlrs02` sought help with understanding how to utilize a particular [GitHub script](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42) for extracting quantized states from the **Wav2Vec2 model**.

**Links mentioned**:

[wav2vec2-codebook-indices/scripts/helpers/w2v2_codebook.py at master · fauxneticien/wav2vec2-codebook-indices](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42): Contribute to fauxneticien/wav2vec2-codebook-indices development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1215666563462733915) (15 messages🔥): 

- **Hugging Face Task Page Discovery**: `@andysingal` revealed their recent find of the Hugging Face [Task Page](https://huggingface.co/tasks), showcasing a comprehensive resource for ML tasks, featuring model counts for various applications like **Image Classification**, **Object Detection**, and **Text-to-Image**.
- **Machine Learning Integrations**: The user shared about [Optimum](https://haystack.deepset.ai/integrations/optimum) by Hugging Face, enhancing **model efficiency on targeted hardware**.
- **Enhancing AI with Few-Shot Examples**: `@epicx` provided a link to an [arXiv paper](https://arxiv.org/abs/2305.19165) discussing a method for strategic reasoning in AI agents, using pretrained LLMs with few-shot examples.
- **NLP Insights**: `@zaidday` highlighted an [article](https://www.deeplearning.ai/resources/natural-language-processing/) discussing the scope and advancements in **Natural Language Processing** (NLP).

**Links mentioned**:

- [The Lucid Dream Project - a Hugging Face Space by ilumine-AI](https://huggingface.co/spaces/ilumine-AI/The-Lucid-Dream-Project): no description found
- [Strategic Reasoning with Language Models](https://arxiv.org/abs/2305.19165): Strategic reasoning enables agents to cooperate, communicate, and compete with other agents in diverse situations. Existing approaches to solving strategic games rely on extensive training, yielding s...
- [Same seed across different gpus in multiple workers](https://discuss.huggingface.co/t/same-seed-across-different-gpus-in-multiple-workers/76535): This is more of a discussion choice question because I have a workaround to suit my use case.  My understanding was if I am training my model on multiple GPUs (say n GPUs) in a DistributedDataParallel...
- [Tasks - Hugging Face](https://huggingface.co/tasks): no description found
- [ML Lecture 23-1: Deep Reinforcement Learning](https://www.youtube.com/watch?v=W8XF3ME8G2I): no description found
- [wav2vec2-codebook-indices/scripts/helpers/w2v2_codebook.py at master · fauxneticien/wav2vec2-codebook-indices](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42): Contribute to fauxneticien/wav2vec2-codebook-indices development by creating an account on GitHub.
- [llm-course/awesome-repos.md at main · andysingal/llm-course](https://github.com/andysingal/llm-course/blob/main/awesome-repos.md): Contribute to andysingal/llm-course development by creating an account on GitHub.
- [Natural Language Processing (NLP) - A Complete Guide](https://www.deeplearning.ai/resources/natural-language-processing/): Natural Language Processing is the discipline of building machines that can manipulate language in the way that it is written, spoken, and organized
- [Optimum | Haystack](https://haystack.deepset.ai/integrations/optimum):      High-performance inference using Hugging Face Optimum   

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1215642735575441418) (18 messages🔥): 

- **Doodle your way to victory**: `@om7059` introduced **Doodle Wars**, a multiplayer game where players doodle objects in 15 seconds which are then scored by a neural network to crown a winner. Play the game [here](https://doodlewars.netlify.app).
- **Legal Precedents go digital with Caselaw Access Project**: `@conceptofmind` shared their support in releasing over 6.6 million U.S. court decisions in collaboration with the **Caselaw Access Project** and **Harvard Library Innovation Lab**. The data is accessible [here](https://x.com/EnricoShippole/status/1766157358672359862?s=20).
- **Soft Prompting Papers Compiled**: `@sauravmaheshkar` is diving into soft prompting as a method for fine-tuning LLMs and has documented relevant papers in a **HuggingFace collection** which can be explored [here](https://huggingface.co/collections/SauravMaheshkar/soft-prompts-65eb62cee008ea6205dee178).
- **Portuguese LLM enters the chat**: `@dominguesm` pre-trained a small LLM, **Mambarim-110m**, with data entirely in Portuguese using the Mamba architecture, available on [HuggingFace](https://huggingface.co/dominguesm/mambarim-110m).
- **BERT Embeds Long Text**: `@pszemraj` fine-tuned a 4k context BERT model, **bert-plus-L8-v1.0-syntheticSTS-4k**, with capabilities for long-text similarity, emphasizing its training on 4k context length and smaller size. The model is up for grabs on [HuggingFace](https://huggingface.co/BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k).

**Links mentioned**:

- [Tweet from DAO Jones (@HungryDAOJones)](https://x.com/HungryDAOJones/status/1766590849494732877): About us: https://youtu.be/E_yThvV6c_I
- [MrOvkill/gemma-2-inference-endpoint-GGUF · Hugging Face](https://huggingface.co/MrOvkill/gemma-2-inference-endpoint-GGUF): no description found
- [Doodle Wars](https://doodlewars.netlify.app): no description found
- [Soft Prompts - a SauravMaheshkar Collection](https://huggingface.co/collections/SauravMaheshkar/soft-prompts-65eb62cee008ea6205dee178): no description found
- [Genstruct - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/Genstruct): no description found
- [HTPP Endpoints — Large Action Model Integration](https://medium.com/@visrow/htpp-endpoints-large-action-model-integration-27e216028b3f): Introducing SwaggerPredictionLoader from Tools4AI
- [KY Open Records Assistant - a Hugging Face Space by jscotthorn](https://huggingface.co/spaces/jscotthorn/kora-assistant): no description found
- [BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k · Hugging Face](https://huggingface.co/BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k): no description found
- [SauravMaheshkar/FewGLUE · Datasets at Hugging Face](https://huggingface.co/datasets/SauravMaheshkar/FewGLUE): no description found
- [dominguesm/mambarim-110m · Hugging Face](https://huggingface.co/dominguesm/mambarim-110m): no description found
- [Soft prompts](https://huggingface.co/docs/peft/en/conceptual_guides/prompting): no description found
- [Tweet from Enrico Shippole (@EnricoShippole)](https://x.com/EnricoShippole/status/1766157358672359862?s=20): @TeraflopAI is excited to help support the @caselawaccess and @HarvardLIL, in the release of over 6.6 million state and federal court decisions published throughout U.S. history.
- [Portfolio – javascript](https://pachinkomachine.quarto.pub/pachinkomachinequartopub/javascript.html): no description found

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1215561220908781598) (35 messages🔥): 

- **Gemini's Impressive Performance**: `@shashank.f1` shared a [YouTube video](https://youtu.be/IuehDA1M_Lw) comparing Gemini, Claude Opus, and GPT-4 Turbo, highlighting Gemini's superior speed and cheaper costs. `@chad_in_the_house` reflected on the benefits of Gemini 1.5 Pro, stating its context length is five times greater than its competitors and it also showcases better multimodal understanding.
- **Mixture of Experts and finetuning challenges**: `@shashank.f1` and `@chad_in_the_house` discussed limitations with Mixture of Experts (MoE) models, revealing that customization like LoRA finetuning is challenging due to increased VRAM requirements, which makes MoE inefficient on single GPU setups.
- **Exploring Long Contexts in LLMs**: `@chad_in_the_house` pointed out **attention sinks** as an interesting technology for handling long contexts in large language models (LLMs), referring to a HuggingFace blog post by thomwolf's collaborator `<@274244546605613056>` [here](https://huggingface.co/blog/tomaarsen/attention-sinks).
- **Video Understanding State of the Art (SOTA)**: `@chad_in_the_house` directed to a benchmark for video understanding technology, highlighting VideoChat2 as a lead contender and providing a [link to the source](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) for further exploration.
- **The Potential of Consistency Distillation and Diffusers**: `@riteshrm` inquired about the availability of a standalone script for consistency models even though it is available in the diffusers library, prompting further discussion on practical implementation.

**Links mentioned**:

- [[Deleted Topic] | Kaggle](https://www.kaggle.com/discussions/questions-and-answers/483264): [Deleted Topic].
- [Ask-Anything/video_chat2/MVBENCH.md at main · OpenGVLab/Ask-Anything](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md): [CVPR2024][VideoChatGPT] ChatGPT with video understanding! And many more supported LMs such as miniGPT4, StableLM, and MOSS. - OpenGVLab/Ask-Anything
- [Gemini supports 1M+ tokens and 20x cheaper than GPT4 😮 ~ Unlock ideas from the technical paper](https://youtu.be/IuehDA1M_Lw): Here is a quick summary comparing Gemini, Claude Opus and GPT-4 Turbo to find out why you should be interested in Gemini 1.5 Pro♦️On speed 💨 ~ It takes 1 se...
- [wav2vec2-codebook-indices/scripts/helpers/w2v2_codebook.py at master · fauxneticien/wav2vec2-codebook-indices](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42): Contribute to fauxneticien/wav2vec2-codebook-indices development by creating an account on GitHub.
- [GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP](https://t.co/qcyEa7EGGY): Training LLMs with QLoRA + FSDP. Contribute to AnswerDotAI/fsdp_qlora development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1215914120600227892) (7 messages): 

- **Seeking Optimal Mistral Settings**: `@elmatero6` asked for advice on running **Mistral** predominantly on CPU to avoid bluescreening, given their system specs include an Intel Core I5-9300H, 32GB DDR4 RAM, and an Nvidia Geforce GTX 1650.
- **Too Fast for Comfort**: `@HuggingMod` reminded `@1097592228714659912` to slow down their message pace, indicating a possible flood in the **diffusion-discussions** channel.
- **Scaling Woes for Chatbot Deployment**: `@rajveerrathod` enquired about scaling an enterprise level chatbot application capable of handling 15-20 queries simultaneously using **LLama 7b** and **Mistral 7b** on a Google Cloud GPU, experiencing crashes with concurrent users.
- **In Search of the Finest Image Captioning Model**: `@ninamani` sought recommendations for the best open-source model for precise uncensored "Image to text" captioning, with `@chad_in_the_house` suggesting **cogvlm** though noting issues with models becoming unstable at 4 bit quantization.
- **Guidance Request for Wav2Vec2 Script Usage**: `@210924_aniketlrs02` requested help on how to use a [specific GitHub script](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42) to extract quantized states from the Wav2Vec2 model as they are new to machine learning.

**Links mentioned**:

[wav2vec2-codebook-indices/scripts/helpers/w2v2_codebook.py at master · fauxneticien/wav2vec2-codebook-indices](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42): Contribute to fauxneticien/wav2vec2-codebook-indices development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1215593282059509760) (41 messages🔥): 

- **Commercial Use of YOLOv4 Clarified**: User `@toni_alright` informed that the **YOLOv4** license is commercially friendly; `@prod.dopamine` responded seeking an implementation as easy-to-use as Ultralytics but suitable for commercial applications.
- **Troubleshooting TensorFlow ImportError**: `@crown_16` encountered an `ImportError` with TensorFlow; `@cursorop` advised testing the code in Google Colab and considering reinstalling TensorFlow if successful there.
- **Learning Journey for GANs and Beyond**: After `@noir_bd` expressed interest in starting with GANs, multiple users including `_homoludens` and `@mikonvergence` provided resources and suggested an inclusive approach that also features diffusion models, VAEs, and more. Links to courses and repositories on [Coursera](https://www.coursera.org/specializations/generative-adversarial-networks-gans), [Github for Diffusion models](https://github.com/mikonvergence/DiffusionFastForward), and a general course on generative models from [Jakub Tomczak](https://github.com/jmtomczak/intro_dgm) were shared.
- **Fast.ai Course Recommended**: `_homoludens` shared a [link to a free course](https://course.fast.ai) covering practical applications of deep learning, with the second part encompassing diffusion models and Hugging Face's Diffusers library.
- **Inpainting Feature Question on Stable Diffusion**: `@okan1962` inquired about the availability and documentation for inpainting and image variations features in Stable Diffusion using HuggingFace's inference API, noting a lack of clear information and closed model endpoints.



**Links mentioned**:

- [Practical Deep Learning for Coders - Practical Deep Learning](https://course.fast.ai): A free course designed for people with some coding experience, who want to learn how to apply deep learning and machine learning to practical problems.
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233): We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models. We achieve this on unconditional image synthesis by finding a better architec...
- [GitHub: Let’s build from here](https://github.com/): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...
- [GitHub - jmtomczak/intro_dgm: &quot;Deep Generative Modeling&quot;: Introductory Examples](https://github.com/jmtomczak/intro_dgm): &quot;Deep Generative Modeling&quot;: Introductory Examples. Contribute to jmtomczak/intro_dgm development by creating an account on GitHub.
- [GitHub - mikonvergence/DiffusionFastForward: DiffusionFastForward: a free course and experimental framework for diffusion-based generative models](https://github.com/mikonvergence/DiffusionFastForward): DiffusionFastForward: a free course and experimental framework for diffusion-based generative models - mikonvergence/DiffusionFastForward
- [Generative Adversarial Networks (GANs)](https://www.coursera.org/specializations/generative-adversarial-networks-gans): Offered by DeepLearning.AI. Break into the GANs space. Master cutting-edge GANs techniques through three hands-on courses! Enroll for free.

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1215682006911033354) (38 messages🔥): 

- **Import Troubles with trl**: User `@solanao64` encountered an `ImportError` while trying to import `SFTTrainer` and `DPOTrainer` from `trl` due to an issue with `topktop_p_filtering` from `transformers`.
- **Deberta-based Classifier Example**: `@darwinanim8or` shared an example of using `Deberta-based` classifiers, providing a [code snippet for text classification using HuggingFace's pipeline](https://github.com).
- **Fine-tuning Mistral 7B**: `@plbjt` inquired about fine-tuning `Mistral 7B` for specific tasks using GPT-4 formatted prompts, sparking a discussion about model suitability for complex tasks.
- **C++ Deployment for BERT**: `@smartguy_41719` is seeking guidance on deploying a trained BERT model for inference in a C++ environment, with `@merve3234` suggesting to use ONNX Runtime and Hugging Face's Optimum.
- **LLMs for Translation Tasks**: `@ninamani` asked for recommendations on optimized and accurate models for NSFW uncensored translation tasks, requiring more precision than older models or overly large LLMs can offer.


**Links mentioned**:

- [PyTorch: Fine Tuning GPT2 For QuestionAnswering](https://www.kaggle.com/code/dsmeena/pytorch-fine-tuning-gpt2-for-questionanswering): Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources
- [Overview](https://huggingface.co/docs/optimum/onnxruntime/overview): no description found
- [wav2vec2-codebook-indices/scripts/helpers/w2v2_codebook.py at master · fauxneticien/wav2vec2-codebook-indices](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42): Contribute to fauxneticien/wav2vec2-codebook-indices development by creating an account on GitHub.
- [GitHub - huggingface/lighteval: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron.](https://github.com/huggingface/lighteval?): LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron. - hug...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1215914120600227892) (7 messages): 

- **CPU Over GPU for Mistral**: User `@elmatero6` seeks advice on running **Mistral** efficiently on a CPU given their system specs (Intel Core i5, 32GB RAM, Nvidia GTX 1650) to avoid bluescreening their PC, suggesting a preference for RAM utilization over GPU.
- **Speedy Poster Gets a Nudge**: `@HuggingMod` gently reminded `@elmatero6` to post more slowly in the chat.
- **Scaling Chatbots for Enterprise**: `@rajveerrathod` is developing a customer success chatbot with **LLama 7b and Mistral 7b**; however, the app crashes under concurrent usage on Google Cloud's GPU. They seek solutions for scaling up to handle 20 users simultaneously with the models quantized to 4 and 8 bits.
- **Quality Image Captioning Model**: User `@ninamani` inquired about the best open-source option for precise uncensored image-to-text or image captioning models. `@chad_in_the_house` recommended **cogvlm** and noted stability at 8 bit quantization.
- **Newcomer Requesting Wav2Vec2 Guidance**: `@210924_aniketlrs02` asked for assistance in using a [GitHub script](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42) to extract the quantized states of the Wav2Vec2 model, indicating they are new to machine learning.

**Links mentioned**:

[wav2vec2-codebook-indices/scripts/helpers/w2v2_codebook.py at master · fauxneticien/wav2vec2-codebook-indices](https://github.com/fauxneticien/wav2vec2-codebook-indices/blob/master/scripts/helpers/w2v2_codebook.py#L42): Contribute to fauxneticien/wav2vec2-codebook-indices development by creating an account on GitHub.

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1215676466973376553) (101 messages🔥🔥): 

- **A Warm Welcome and Direction for Beginners**: `@shida3916` expressed excitement about joining the community to discuss everyday AI and ask simple questions. `@stellaathena` redirected them to other servers given this server's research-level focus.
- **LLMs Seek Home, But Nobody's There**: A deep discussion was sparked by `@faron1111` about the concept of self-awareness within LLMs, specifically talking about self-preservation mechanisms. `@wonkothesensible` argued that while models may have implicit notions of agency, they lack any conscious home to occupy.
- **Persistent State in LLMs and AGI Potential**: The conversation about LLMs continued with a focus on architecture for potential AGI, including 1-bit variants mentioned in a [posted link](https://arxiv.org/abs/2402.17764) by `@wonkothesensible`. The need for advanced planning and awareness within LLMs was debated, suggesting necessary breakthroughs before reaching AGI.
- **Discussion on Training Small Models**: `@biiter` inquired about strategies to pre-train models effectively on limited VRAM and discussed potential issues with AliBi embedding. `@hailey_schoelkopf` addressed the technical problem and agreed to provide a fix.
- **AI Extinction-Level Concern**: `@conceptron` shared a [Slashdot article](https://yro.slashdot.org/story/24/03/11/185217/us-must-move-decisively-to-avert-extinction-level-threat-from-ai-govt-commissioned-report-say) reporting U.S. government concerns that frontier AI poses an extinction-level threat to humanity, suggesting regulatory measures such as restrictions on model weights publication.

**Links mentioned**:

- [US Must Move 'Decisively' To Avert 'Extinction-Level' Threat From AI, Gov't-Commissioned Report Says - Slashdot](https://yro.slashdot.org/story/24/03/11/185217/us-must-move-decisively-to-avert-extinction-level-threat-from-ai-govt-commissioned-report-say): The U.S. government must move &#34;quickly and decisively&#34; to avert substantial national security risks stemming from artificial intelligence (AI) which could, in the worst case, cause an &#34;ext...
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [The orthogonality thesis &amp; AI optimism](https://youtu.be/8H3dblxkLhY): Timestamps:0:00 - Start of video7:39 - Outline of Bostrom’s argument9:25 - Decisive strategic advantage13:26 - Arguments for slow takeoff23:13 - Definition o...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1215572558150377512) (75 messages🔥🔥): 

- **Exploring Efficient Attention for Image Diffusion**: `@nostalgiahurts` discussed [a paper on arXiv](https://arxiv.org/abs/2402.13573) that proposes **ToDo**, a novel method that accelerates Stable Diffusion inference by employing token downsampling, increasing speeds by up to 2x to 4.5x. The conversation included a GitHub link to [a related repository](https://github.com/ethansmith2000/ImprovedTokenMerge).
  
- **Few-Shot Versus Zero-Shot Performance Anomalies**: `@paganpegasus` noted an interesting phenomenon where zero-shot performance on the MMLU benchmark was comparable or superior to few-shot performance for various models they were testing. Several hypotheses were discussed, including the potential distraction of additional context for smaller models and the idea of testing performance with varying numbers of shots.
  
- **Frontiers in Optical Digital Computing**: The paper mentioned by `@ai_waifu` explores the potential of all-optical digital computing and memory ([link to arXiv abstract](https://arxiv.org/abs/2403.00045v1)). Topics such as semiconductors, electronic communication inefficiencies, and the paper's implications on manufacturing were briefly discussed.
  
- **Gemini 1.5 Report Overview Provided**: `@xylthixlm` indicated the release of the Gemini 1.5 report with no substantial technical details ([link to report on arXiv](http://arxiv.org/abs/2403.05530)). `@main.ai` followed up by providing insights into the new content in the report.
  
- **Yi Tech Report's Double Wikipedia Filtering Approach**: `@maxmatical` brought up a question about the Yi tech report's approach that effectively filters Wikipedia content twice ([link to arXiv abstract](https://arxiv.org/abs/2403.04652)). `@thedeviouspanda` suggested that this might be similar to the use of light and heavy rankers in ranking pipelines, with each step filtering progressively more intensively.

**Links mentioned**:

- [Fan Group at Stanford University - Software](https://web.stanford.edu/group/fan/software.html): no description found
- [DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models](https://arxiv.org/abs/2403.00818): Large language models (LLMs) face a daunting challenge due to the excessive computational and memory requirements of the commonly used Transformer architecture. While state space model (SSM) is a new ...
- [Stacking as Accelerated Gradient Descent](https://arxiv.org/abs/2403.04978): Stacking, a heuristic technique for training deep residual networks by progressively increasing the number of layers and initializing new layers by copying parameters from older layers, has proven qui...
- [Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652): We introduce the Yi model family, a series of language and multimodal models that demonstrate strong multi-dimensional capabilities. The Yi model family is based on 6B and 34B pretrained language mode...
- [An All-Optical General-Purpose CPU and Optical Computer Architecture](https://arxiv.org/abs/2403.00045v1): Energy efficiency of electronic digital processors is primarily limited by the energy consumption of electronic communication and interconnects. The industry is almost unanimously pushing towards repl...
- [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](http://arxiv.org/abs/2403.05530): In this report, we present the latest model of the Gemini family, Gemini 1.5 Pro, a highly compute-efficient multimodal mixture-of-experts model capable of recalling and reasoning over fine-grained in...
- [A Theoretical Analysis of Nash Learning from Human Feedback under General KL-Regularized Preference](http://arxiv.org/abs/2402.07314): Reinforcement Learning from Human Feedback (RLHF) learns from the preference signal provided by a probabilistic preference model, which takes a prompt and two responses as input, and produces a score ...
- [Making Large Language Models Better Reasoners with Step-Aware Verifier](https://arxiv.org/abs/2206.02336): Few-shot learning is a challenging task that requires language models to generalize from limited examples. Large language models like GPT-3 and PaLM have made impressive progress in this area, but the...
- [ToDo: Token Downsampling for Efficient Generation of High-Resolution Images](https://arxiv.org/abs/2402.13573): Attention mechanism has been crucial for image diffusion models, however, their quadratic computational complexity limits the sizes of images we can process within reasonable time and memory constrain...
- [Pretrained-Language-Model/CAME/came.py at master · huawei-noah/Pretrained-Language-Model](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/CAME/came.py): Pretrained language model and its related optimization techniques developed by Huawei Noah&#39;s Ark Lab. - huawei-noah/Pretrained-Language-Model
- [CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/abs/2307.02047): Adaptive gradient methods, such as Adam and LAMB, have demonstrated excellent performance in the training of large language models. Nevertheless, the need for adaptivity requires maintaining second-mo...

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1216064284036169768) (3 messages): 

- **Newcomer Seeks Interpretability Insights**: User `@xcodevn` expressed an interest in getting started with interpretability research and asked for resource recommendations.
- **A Useful Resource Shared**: In response, `@wendlerc` shared a link to the ARENA 3.0 Landing Page, [mango-ambulance-93a.notion.site](https://mango-ambulance-93a.notion.site/ARENA-3-0-Landing-Page-virtual-8f7193af31b445c586efed03e995fb74), describing it as a "gem" for those interested in the field.

**Links mentioned**:

[Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://mango-ambulance-93a.notion.site/ARENA-3-0-Landing-Page-virtual-8f7193af31b445c586efed03e995fb74): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1215644760954634270) (12 messages🔥): 

- **Query on BOS token usage for models**: `@jwngx` questioned the standard for using the **BOS (Beginning of Sentence) token** and its inconsistent application across different repos. `@stellaathena` confirmed that usage depends on the model, and there isn't a consolidated resource detailing which models perform better with it.
- **Seeking BOS Token Insights**: `@jwngx` inquired if there is documentation on model performance with the BOS token, but `@hailey_schoelkopf` noted that such details are typically internal and model-dependent; the odd behavior of Gemma with BOS tokens is unprecedented.
- **Adjusting HFLM Decoding for BOS Token**: In light of a commit adding the BOS token flag for Gemma, `@jwngx` shared the [HFLM code link](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L716) and asked whether decoding should consider the `self.add_bos_token` setting. `@hailey_schoelkopf` clarified that `tok_decode` should only be called on continuation text without the BOS token or input text, suggesting the current implementation is correct.

**Links mentioned**:

[lm-evaluation-harness/lm_eval/models/huggingface.py at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L716)): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1215686529695883364) (2 messages): 

- **Warm Welcome to a New Community Member**: User `@shida3916` expressed excitement about joining the community to discuss everyday uses of AI and to ask simple questions. They inquired if this was the right place for such discussions.
- **Clarifying Transformer and Diffusion Concepts**: `@yoavhacohen` provided a clarification stating that **"Transformer is an architecture, while diffusion is a training and inference method."** They also mentioned that diffusion was used with transformers prior to SD3, citing several examples like DALL-E 2, DiT, and PixArt.
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1215557258386079744) (75 messages🔥🔥): 

- **Torch Container Ambiguity**: `@catboy_slim_` highlighted unclear documentation about which commit of apex is being used in the torch development container, suggesting the setup could be more straightforward.

- **Dependency Management Challenges**: Multiple users, including `@catboy_slim_`, `@tfidia`, and `@hailey_schoelkopf` discussed difficulties with managing dependencies in the GPT-NeoX project, mentioning the complexities introduced by a default NGC container that might contain both necessary and extraneous packages.

- **Flash Attention Dependencies**: `@hailey_schoelkopf` clarified that Triton is used both for sparse and flash attention, and the group also discussed how Flash attention's update to 2.5.6 potentially affects compatibility with the NGC PyTorch container.

- **Apex Usage in Question**: Users including `@biiter` and `@catboy_slim_` debated the necessity of Apex, as some of its functionality may now be built into PyTorch, except for specific features like `fusedAdam`.

- **Evaluation and Conversion Queries**: `@tejas.inferq` sought help with the evaluation process for a trained 125M parameter GPT-NeoX model, while `@aphoh` inquired about converting Pythia/NeoX checkpoints to upstream megatron-lm, facing issues with matching weight layouts and losses.

**Links mentioned**:

- [Cleaner dockerfile: Remove already installed deps by tf-nv · Pull Request #1175 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1175/files): Cleaning up the Dockerfile after the ngc pytorch switch (#1170):  Eliminate already installed apt packages sparse attn requirement lead to a triton downgrade flash attn is already part of the ngc c...
- [PyTorch Release 24.02 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-02.html#rel-24-02): no description found
- [PyTorch Release 24.02 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-02.html): no description found

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1215570094059429958) (39 messages🔥): 

- **AI Biographer Trials and Privacy Concerns**: `@swyxio` has been trialing [Emma, the AI Biographer](https://getlifestory.com/) and finds the call experience good enough to recommend trying at least once. However, they caution users about potential security concerns by mentioning their use of *fake name and fake biographical details*.
- **OpenAI Drama Concludes**: `@guardiang` shares a [New York Times article](https://www.nytimes.com/2024/03/07/technology/openai-executives-role-in-sam-altman-ouster.html) detailing internal issues at OpenAI. The OpenAI board has completed a review into the firings, confidently reinstating Sam Altman and announcing the addition of three new board members, as noted in [OpenAI's announcement](https://openai.com/blog/review-completed-altman-brockman-to-continue-to-lead-openai).
- **Ideogram 1.0 Launch Under the Radar**: `@swyxio` mentions that the launch of [Ideogram 1.0](https://x.com/ideogram_ai/status/1762881284899008564?s=20), a new text rendering method, has not received much attention despite its potential.
- **LLM Interface Development by Microsoft Research**: `@swizec` brings up a [Hacker News post](https://news.ycombinator.com/item?id=39670665) discussing AICI, an interface proposed by Microsoft Research to standardize constraints and control mechanisms across various LLM inference engines, seeking feedback for the Rust AICI runtime.
- **A Glance at Transition Beyond Transformers**: `@swyxio` discusses "Mamba," a State Space Model presented as a potential alternative to the Transformer architecture for LLMs. They refer to a [visual guide](https://maartengrootendorst.substack.com/p/a-visual-guide-to-mamba-and-state) and the original [research paper](https://arxiv.org/abs/2312.00752) for those interested in understanding the architecture.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39482428): no description found
- [no title found](https://news.ycombinator.com/item?id=39643894): no description found
- [Show HN: Prompts as WASM Programs | Hacker News](https://news.ycombinator.com/item?id=39670665): no description found
- [Tweet from OpenAI (@OpenAI)](https://x.com/openai/status/1766235980170706967?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): “We have unanimously concluded that Sam and Greg are the right leaders for OpenAI.” — Bret Taylor, Chair of the OpenAI Board  The Special Committee of the OpenAI Board has announced the completion of ...
- [Join Slido: Enter #code to vote and ask questions](https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions): Participate in a live poll, quiz or Q&A. No login required.
- [Review completed & Altman, Brockman to continue to lead OpenAI](https://openai.com/blog/review-completed-altman-brockman-to-continue-to-lead-openai): New board members named and enhancements to the governance structure introduced 
- [OpenAI announces new members to board of directors](https://openai.com/blog/openai-announces-new-members-to-board-of-directors): Dr. Sue Desmond-Hellmann, Nicole Seligman, Fidji Simo join; Sam Altman rejoins board
- [Which Quantization Method is Right for You? (GPTQ vs. GGUF vs. AWQ) ](https://maartengrootendorst.substack.com/p/which-quantization-method-is-right): Exploring Pre-Quantized Large Language Models
- [Tweet from Corry Wang (@corry_wang)](https://x.com/corry_wang/status/1766949316394897851?s=20): Wait, Amazon just revealed that they finished training a 200B parameter LLM… and nobody noticed  This is from SVP James Hamilton’s Jan 15 talk at CIDR 2024. Trained on 5x more compute than Facebook’s ...
- [Life Story](https://getlifestory.com/): Capture life, one story at a time. 
- [Tweet from Ideogram (@ideogram_ai)](https://x.com/ideogram_ai/status/1762881284899008564?s=20): Ideogram 1.0 presents a leap forward in text rendering accuracy. Unleash the flavors of your imagination with Ideogram!  Prompt: “A vibrant and colorful advertisement for &#34;Ideogram&#34; burgers, f...
- [Tweet from Mira Murati (@miramurati)](https://x.com/miramurati/status/1766247920242929913?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Governance of an institution is critical for oversight, stability, and continuity. I am happy that the independent review has concluded and we can all move forward united.  It has been disheartening t...
- [A Visual Guide to Mamba and State Space Models](https://maartengrootendorst.substack.com/p/a-visual-guide-to-mamba-and-state): An Alternative to Transformers for Language Modeling
- [Tweet from cohere (@cohere)](https://x.com/cohere/status/1767275128813928611?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Today, we’re excited to release Command-R, a new RAG-optimized LLM aimed at large-scale production workloads.  Command-R fits into the emerging “scalable” category of models that balance high efficien...
- [Tweet from Sam Altman (@sama)](https://x.com/sama/status/1766291001134715207?s=46&t=90xQ8sGy63D2OtiaoGJuww): i&#39;m very happy to welcome our new board members: fidji simo, sue desmond-hellmann, and nicole seligman, and to continue to work with bret, larry, and adam.  i&#39;m thankful to everyone on our tea...
- [Tweet from Teknium (e/λ) (@Teknium1)](https://x.com/teknium1/status/1766721588244918774?s=46&t=90xQ8sGy63D2OtiaoGJuww): Anyone working on Llama.cpp can you give our PR a look:  https://github.com/ggerganov/llama.cpp/pull/5970
- [Tweet from swyx (@swyx)](https://x.com/swyx/status/1765995892107317407?s=20): I&#39;ve now had multiple &gt;20min phone calls with AI therapists and it feels completely natural. Every AI Engineer should be building their own therapist rn, and voice is the right medium.     forg...

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1215596118767964201) (5 messages): 

- **Asia Gets Schooled on GPT-2**: `@ivanleomk` announced a presentation on the GPT-2 paper, urging the **Asia** `@paper-club` members to join in. The event, said to be **EPIC**, was scheduled on [https://discord.gg/8sYsGc83](https://discord.gg/8sYsGc83).
- **Weekend Podcast Drop Teaser**: `@swyxio` posted excitement over a weekend podcast drop. The **Latent Space pod** covers January and February recap and can be listened to [here](https://x.com/latentspacepod/status/1766600314419806350?s=20).
- **Paper Enthusiast Laments Timezone Woes**: `@420gunna` expressed appreciation for the paper selections at `@paper-club` but humorously mentioned forgetting to set an alarm for the 2 AM meetings. A mix of enthusiasm and timezone-woes marking their engagement with the community.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/8sYsGc83): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Tweet from Latent Space Podcast (@latentspacepod)](https://x.com/latentspacepod/status/1766600314419806350?s=20): 🆕 Weekend pod: Jan+Feb recap + 1 Yr of Latent Space!  https://latent.space/p/jan-feb-2024-recap-audio  Our 2023 recap pod was very well received, so here&#39;s the next in this new series of research...

  

---


### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1215598769534533675) (30 messages🔥): 

- **Prep for LLM Paper Club Discussion**: `@ivanleomk` shared [notes](https://www.gaohongnan.com/transformer/decoder/concept.html) as preparation for `@1123457263638683770`'s sharing in the upcoming club discussion, focusing on Generative Pre-trained Transformers (GPT).
- **Starting Time Updates**: `@ivanleomk` provided multiple updates about the starting time of the session, indicating a commencement within 5-10 minutes, followed by another message indicating a 5-minute start time.
- **Community Support for Newcomers**: `@healthymonkey` expressed their newcomer status to NLP, seeking corrections on any potential mistakes, with `@bryanblackbee` offering support and encouragement.
- **Technical Clarifications in Real Time**: `@kishore.reddy` clarified terminologies used by `@1123457263638683770`, like "causal attention" and correcting a reference to "-inf" during the live club session.
- **LLM Visualization Tools Shared**: `@fx2y` provided a [link](https://bbycroft.net/llm) to a visualization tool for GPT family models and offered commendations to `@1123457263638683770` for their work.

**Links mentioned**:

- [LLM Visualization](https://bbycroft.net/llm): no description found
- [The Concept of Generative Pre-trained Transformers (GPT) &#8212; Omniverse](https://www.gaohongnan.com/transformer/decoder/concept.html): no description found
- [The Implementation of Generative Pre-trained Transformers (GPT) &#8212; Omniverse](https://www.gaohongnan.com/transformer/decoder/implementation.html): no description found

  

---


### Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1215766430230774022) (162 messages🔥🔥): 

- **Workflow Optimization with AI Discussion**: `@kbal11` introduced the AI-in-Action session led by `@363877777977376768`, focusing on using AI to improve workflow. Participants such as `@yikesawjeez` expressed enthusiasm for the topic, while others shared insights and tips on enhancing output through AI.
  
- **Meta Prompts and CLI Tools Breakthroughs**: `@yikesawjeez` highlighted the use of AI to create prompts that lead to progressively better outputs and demonstrated interest in deploying projects on AWS. Resources for AI-driven tools that can assist in these efforts were shared and discussed amongst the community.

- **Importance of Documentation in AI Workflow**: The discussion turned to best practices for recording work and making detailed notes. `@slono` shared the use of `asciinema` for recording terminal sessions, while others like `@yikesawjeez` committed to sharing open-source tools they utilize.

- **Community Engagement and Sharing**: The channel actively engaged in sharing tips, tools, and tricks for improving use of AI in personal workflows. Users like `@yikesawjeez` and `@markredito` shared excitement for collaborative learning and crowdsourcing knowledge within the AI space.

- **Request for Future Session on Decentralized/Distributed AI**: Amidst discussions of workflow and tools, `@yikesawjeez` proposed a future session topic on decentralized and distributed AI applications that move beyond cryptocurrency-focused projects.

**Links mentioned**:

- [Genius](https://1906.shop/products/genius): Tap into your genius with the only edible developed specifically to enhance focus. The combination of plant medicines Rhodiola, Theobromine, Galangal, Bacopa, L-Theanine with cannabis promotes cogniti...
- [Getting started - asciinema docs](https://docs.asciinema.org/getting-started/): no description found
- [AI-enhanced development makes me more ambitious with my projects](https://simonwillison.net/2023/Mar/27/ai-enhanced-development/): The thing I’m most excited about in our weird new AI-enhanced reality is the way it allows me to be more ambitious with my projects. As an experienced developer, ChatGPT …
- [AI In Action: Weekly Jam Sessions](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0): 2024  Topic,Date,Facilitator,Resources UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-structure&lt;/a&...
- [GitHub - JoinTheAlliance/bgent: Flexible, scalable and customizable agents to do your bidding.](https://github.com/JoinTheAlliance/bgent): Flexible, scalable and customizable agents to do your bidding. - JoinTheAlliance/bgent
- [GitHub - bazed-ai/bazed-af: 😎 Bazed.ai Agent Framework - Bazed.ai is a unified platform for building, running and scaling autonomous agents.](https://github.com/bazed-ai/bazed-af): 😎 Bazed.ai Agent Framework - Bazed.ai is a unified platform for building, running and scaling autonomous agents. - bazed-ai/bazed-af

  

---



### Interconnects (Nathan Lambert) ▷ #[announcements](https://discord.com/channels/1179127597926469703/1179127598442348726/1216536442361348116) (1 messages): 

- **New Roles for Better Differentiation**: `@natolambert` implemented **new roles** within the Discord to distinguish between manually added close friends and subscribers. Feedback on this update is welcomed.
  

---


### Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1216681631570595890) (51 messages🔥): 

- **Elon Musk's Grok Tease Sparks Debate**: `@xeophon.` shared a [tweet from Elon Musk](https://x.com/elonmusk/status/1767108624038449405?s=46) announcing that `@xAI` will open source Grok, while `@natolambert` questioned the accurate use of "open source" by Musk.
- **Cohere Introduces Command-R**: `@xeophon.` highlighted the introduction of [Command-R](https://txt.cohere.com/command-r/), a new retrieval augmented model by Cohere, notable for its 128k context window and public release of weights for research purposes.
- **Anticipation for Open Models**: `@xeophon.` and `@natolambert` discussed the potential of Cohere's Command-R, especially for startups, academia, and its usability in multiple European languages as well as its importance beyond the hype for future models like "llama3".
- **Market Reaction to Elon's Announcement**: `@natolambert` expressed that people might be overreacting to Elon's announcement, giving credit prematurely before any model is released, and `@eugenevinitsky` drew an interesting parallel with Twitter's open-source move but with a twist as "Weights without code instead of code without weights."
- **Questioning OpenAI's Commitment to Open Models**: `@dangf91` inquired about the open-source status of Mistral, with `@xeophon.` clarifying that there's still a commitment to open models in the future, and `@natolambert` adding that the environment is ever-changing.

**Links mentioned**:

- [Tweet from Xeophon (@TheXeophon)](https://x.com/thexeophon/status/1765797558696165645?s=46): GPT 5 tonite  GPT 5 tonite queen
- [Tweet from Elon Musk (@elonmusk)](https://x.com/elonmusk/status/1767108624038449405?s=46): This week, @xAI will open source Grok
- [Command-R: Retrieval Augmented Generation at Production Scale](https://txt.cohere.com/command-r/): Command-R is a scalable generative model targeting RAG and Tool Use to enable production-scale AI for enterprise.  Today, we are introducing Command-R, a new LLM aimed at large-scale production worklo...

  

---


### Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1216589316612952245) (2 messages): 

- **GPT-4 Takes On Doom**: `@philpax` shared a paper which demonstrates **GPT-4's ability to play the 1993 first-person shooter game Doom**, using its reasoning and planning capabilities without any game-specific training. The model managed to manipulate doors, fight enemies, and navigate the game world, and the paper suggests that complex prompting could further enhance its performance. Find the paper here: [GPT-4 Plays Doom](https://arxiv.org/abs/2403.05468).

- **Call for a Blog Post**: `@natolambert` reacted to the title of the paper on GPT-4 playing Doom, implying that the content sounds intriguing enough to merit a standalone **blog post**.

**Links mentioned**:

[Will GPT-4 Run DOOM?](https://arxiv.org/abs/2403.05468): We show that GPT-4&#39;s reasoning and planning capabilities extend to the 1993 first-person shooter Doom. This large language model (LLM) is able to run and play the game with only a few instructions...

  

---


### Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1216109634423885904) (36 messages🔥): 

- **Paper Cutoff Date for olmo/dolma clarified**: `@mike.lambert` inquired about the data range for olmo/dolma papers, wondering if it was intentional leading up to May 2023. `@natolambert` responded, clarifying that the cutoff is due to publication timelines rather than an intention to avoid the ChatGPT era, also mentioning the lack of a scraper.

- **Cheap Pretraining of GPT2 Models**: `@natolambert` asked about the current costs of pretraining a model like GPT2. `@xeophon.` responded with a price of **less than $1,000**, an estimate he considered "pretty wild."

- **Pretraining Costs and Feasibility**: `@xeophon.` shared details on training costs from September 2022, GPT-2 document counts, and Databricks pretraining costs for reference. `@natolambert` and `@philpax` discussed how surprisingly quick and cheap it is to train models now, and `@natolambert` mentioned that for Stability AI's model, they **probably paid less than $100,000 for the compute**.

- **Stability AI Compute Deal Speculation**: In the discussion about costs, `@xeophon.` mentioned that Stability AI possibly received free or discounted compute in exchange for using Gaudi2 and promoting it, as suggested by their partnership ad.

- **Fine-Tuning Practices Suggestions**: When `@dangf91` asked about fine-tuning models with more books/articles and whether to use a masking strategy, `@natolambert` and `@vj256` agreed that adding books to the dataset with some pretraining mixture and continuing training is the typical practice.

  

---


### Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1215601465838534686) (12 messages🔥): 

- **Drama Over Inflection AI's Model**: `@xeophon.` highlighted a tweet from `@seshubon` questioning if Inflection AI's chatbot was merely a wrapper for Claude-3-Sonnet, given identical responses to custom queries. Inflection AI boasted their own model, Inflection-2.5, which supposedly rivaled GPT-4. [Inflection AI's Tweet](https://fxtwitter.com/seshubon/status/1765870717844050221).
- **Potential A/B Test at Inflection?**: `@xeophon.` speculated that the observed behavior could be from an A/B test comparing Inflection-2.5 with Claude Opus, considering the unlikely chance of two models generating word-for-word matches for a lengthy and specific prompt.
- **Revoked API Key**: `@natolambert` humorously remarked that someone's API key might get revoked in light of the unfolding drama.
- **Claude's Temperature Setting Noted**: `@mike.lambert` mentioned that claude.ai typically uses a non-zero temperature for its responses, contributing to the ongoing discussion.
- **Inflection AI Clarifies**: `@xeophon.` shared a tweet from Inflection AI explaining that their chatbot, Pi, remembers prior conversations and repeated a message from Claude because it was earlier included in the conversation. The plot thickens as questions about Pi's capabilities and independence arise. [Inflection AI's Clarification](https://fxtwitter.com/inflectionai/status/1766173427441049684?s=46).
- **Caution Advised on Official Replies**: When `@xeophon.` shared Inflection AI's response, `@natolambert` advised never to reply on official accounts, suggesting the company might encounter negative consequences. `@natolambert` implicitly affirmed the trouble with "fd," meaning "f***ed."

**Links mentioned**:

- [Tweet from Inflection AI (@inflectionAI)](https://fxtwitter.com/inflectionai/status/1766173427441049684?s=46): Pi’s responses are always generated by our own models, built in-house. On investigation, it appears the user prompted this particular response from Pi after copy-pasting the output from Claude earlier...
- [Tweet from seshu bonam (@seshubon)](https://fxtwitter.com/seshubon/status/1765870717844050221): WHAT?  @inflectionAI is just a claude-3-sonnet wrapper?   care to explain?   🐒  Produces the exact same answer word to word for a custom query i asked 🤯  ↘️ Quoting Inflection AI (@inflectionAI)   P...

  

---


### Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1215718929402695720) (19 messages🔥): 

- **Lambert Cabal Expands**: `@philpax` joked about the growing presence of Lamberts in AI labs, humorously equating their network to **the Lambert backchannels**, while `@420gunna` referenced the concept of a [mycorrhizal network](https://en.wikipedia.org/wiki/Mycorrhizal_network) to liken it to an underground communication system.
- **Sam Altman Rejoins OpenAI Board**: `@xeophon.` shared a [news article](https://www.theinformation.com/articles/sam-altman-to-return-to-openai-board-of-directors) discussing **Sam Altman's return to the board of directors** at OpenAI, and `@420gunna` quoted Bret Taylor, adding a dash of humor on leadership with a Civ4 Cultural Victory reference.
- **Mocking Self-Review Leadership**: `@natolambert` mocked the idea of self-evaluation in leadership with a tongue-in-cheek quote: "*I have done an internal review and decided I am still the king*".
- **Canadian Geese as a Discord Rite**: `@philpax` humorously pondered the number of Canadian geese needed for a "ritual" to obtain a Friend role on Discord, referring to them as a trusty emblem, while `@natolambert` acknowledged that geese are intimidating.
- **The Quest for the Goose Role**: `@natolambert` proposed the creation of a self-nominated goose role on Discord, suggesting to add a **boost** or icon to make it noticeable, as stakes of subscriber roles come under light-hearted scrutiny.

**Links mentioned**:

[Mycorrhizal network - Wikipedia](https://en.wikipedia.org/wiki/Mycorrhizal_network): no description found

  

---


### Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1215722082491441233) (15 messages🔥): 

- **Dune Casts Reimagined with AI Celebrities**: `@420gunna` started a creative game by imaging prominent figures in AI as characters from *Dune*, designating Sama as the **Kwisatz Haderach**, Alec Radford as **Thufir Hawat**, and several others, with some roles left open for suggestions.
- **Naming the Matriarchs of AI's Dune**: Shortly after, `@natolambert` proposed that the **CTO** is definitely **Lady Jessica**, while also assigning the role of **Baron Harkonnen** to **Elon Musk**.
- **Yann Lecun's Brain Sauce Analogy**: `@420gunna` shared a quirky remark about **Yann Lecun**, referencing how he often talks about his brain turning to "white sauce" with age, and humorously coined the term **Brain to Béchamel pipeline**.
- **Suggestions for Dune’s Key Players**: `@twkillian` brought up Peter Thiel as a candidate for **Stilgar**, however, `@natolambert` disagreed, suggesting **Marc Andreessen** as a more fitting choice for the role. Later, they had a laugh about placing **Gary Marcus** as an equivalent to the maladies introduced in later *Dune* books.
- **The Weirdness of Dune Beyond Book One**: A discussion ensued about whether it's worth reading beyond the first *Dune* book. While `@twkillian` was advised to stop at the first, both `@eugenevinitsky` and `@natolambert` agreed that **continuing the series** is worthwhile due to its intriguing oddity.
  

---


### Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1215600990321905725) (8 messages🔥): 

- **Exploration of Reinforcement Learning**: User `@chygao` shared an [episode link](https://open.spotify.com/episode/0FuKEjteM0cGzy7pznCkAj?si=ds-7rY8yT0emOLgaV7rgcA&context=spotify%3A) from Spotify featuring Ian Osband from OpenAI discussing **information theory and reinforcement learning (RL), exploration, epistemic uncertainty, and scaling to large language models (LLMs)**.
- **Nostalgia for 'Talk RL' Podcast**: `@natolambert` expressed intent to listen to the shared episode, reminiscing about being a fan of the 'Talk RL' podcast.
- **Diminishing Activity on a Favored Podcast**: `@twkillian` lamented that the 'Talk RL' podcast posts less frequently now, which has somewhat diluted their fandom.
- **Selective Listening Based on Guest**: `@natolambert` disclosed that while they still check 'Talk RL', they don't listen to every episode and their engagement depends on the guest speaker.
- **Consistency in Quality Matters**: `@twkillian` acknowledged that the quality of 'Talk RL' episodes has varied, resulting in a more selective approach to the podcast content.

**Links mentioned**:

- [Ian Osband](https://open.spotify.com/episode/0FuKEjteM0cGzy7pznCkAj?si=ds-7rY8yT0emOLgaV7rgcA&context=spotify%3A): Listen to this episode from TalkRL: The Reinforcement Learning Podcast on Spotify. Ian Osband is a Research scientist at OpenAI (ex DeepMind, Stanford) working on decision making under uncertainty.  W...
- [Ian Osband](https://open.spotify.com/episode/0FuKEjteM0cGzy7pznCkAj?si=ds-7rY8yT0emOLgaV7rgcA&context=spotify%3Aepisode%3A0FuKEjteM0cGzy7pznCkAj): Listen to this episode from TalkRL: The Reinforcement Learning Podcast on Spotify. Ian Osband is a Research scientist at OpenAI (ex DeepMind, Stanford) working on decision making under uncertainty.  W...

  

---


### Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1215614054916427786) (7 messages): 

- **Exploring RLHF for LLM Alignment**: User `@xeophon.` shared an [arXiv paper](https://arxiv.org/abs/2403.04642) studying the performance of **RLHF, PPO, and Expert Iteration** on improving LLM reasoning capabilities. `@natolambert` acknowledged the paper stating it looks good.
- **Inquiry into Claude's Synthetic Tasks**: User `@eugenevinitsky` expressed interest in details regarding synthetic tasks used in **Claude** for creating uncertainty-aware LLMs.
- **Seeking Crux of Synthetic Task Issue**: `@natolambert` responded to `@eugenevinitsky`, mentioning that they, along with `<@304671004599255043>`, are exploring theories on the core issue of creating effective synthetic tasks.
- **Envisioning Advanced Methods for Synthetic Tasks**: `@natolambert` speculated about using methods like **CAI (Counterfactual AI)** with improvements for diversity in generating synthetic tasks.
- **Pretraining Data Vs. Instructions/Preferences**: `@natolambert` suggested running CAI on **pretraining data**, contrasting it with the standard focus on instructions or preferences.

**Links mentioned**:

[Teaching Large Language Models to Reason with Reinforcement Learning](https://arxiv.org/abs/2403.04642): Reinforcement Learning from Human Feedback (\textbf{RLHF}) has emerged as a dominant approach for aligning LLM outputs with human preferences. Inspired by the success of RLHF, we study the performance...

  

---



### OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1215718408012963870) (4 messages): 

- **New Speed Champion Mistral 7b 0.2**: @alexatallah proudly introduced **Mistral 7b 0.2**, boasting about a substantial speed boost—**10x faster for short outputs and 20x faster for long ones**, as well as a generous **32k** context window. The performance was showcased in a [demo tweet](https://twitter.com/OpenRouterAI/status/1766147110443909184).
  
- **Gemma Nitro hits the market**: A new cost-effective and high-speed model called **Gemma Nitro** was announced by @alexatallah, featuring impressive speeds of over **600+ tokens per second** and offering an economical rate of **$0.1 per million tokens**. More details can be found on [OpenRouter's website](https://openrouter.ai/models/google/gemma-7b-it:nitro).

- **Sneak peek tweet?**: @alexatallah shared a [mysterious Twitter link](https://twitter.com/OpenRouterAI/status/1766916892755706020) without additional context or comments.

- **OpenRouter flaunts no spending limits**: @alexatallah revealed a user-friendly policy on OpenRouter, stating that *there are no $ usage limits* on the platform, potentially inviting users to utilize their services more freely.

**Links mentioned**:

[Google: Gemma 7B (nitro) by google | OpenRouter](https://openrouter.ai/models/google/gemma-7b-it:nitro): Gemma by Google is an advanced, open-source language model family, leveraging the latest in decoder-only, text-to-text technology. It offers English language capabilities across text generation tasks ...

  

---


### OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1216286555270942760) (1 messages): 

- **Claude 3 Function Calling Made Easy**: User `@thevatsalsagalni` introduced a function calling library tailored for the **Claude 3** model family. The library supports Pydantic function schemas and is open for exploration and contribution at [claudetools on GitHub](https://github.com/vatsalsaglani/claudetools).

**Links mentioned**:

[GitHub - vatsalsaglani/claudetools: Claudetools is a Python library that enables function calling with the Claude 3 family of language models from Anthropic.](https://github.com/vatsalsaglani/claudetools): Claudetools is a Python library that enables function calling with the Claude 3 family of language models from Anthropic. - vatsalsaglani/claudetools

  

---


### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1215563520591142933) (120 messages🔥🔥): 

- **Censorship in AI Models Becomes a Hot Topic**: Multiple users, including `@.toppa` and `@lemmyle`, expressed concerns about censorship creeping into AI models, such as the "Claude 2 self moderated versions," and potential new restrictions related to copyright or AI responses. Conversations touched on how AI models, like Claude 3, are responding to user inputs and the desire for less censored options.

- **Querying AI Format Support and Parameter Functionality**: In a technical discussion, `@cupidbot.ai` and `@spaceemotion` questioned the formatting of messages for various AI models and the functionality of system parameters such as `json_object` and `add_generation_prompt=True`. `@alexatallah` clarified some documentation points, including the removal of `schema` until it sees more support.

- **Model Output Limits Spark Curiosity and Friction**: Users like `@zulfiqaar` and `@.wingedsheep` explored the output length limitations of various models, with specific mention of GPT-4's 4096 token output cap. Despite users like `@lemmyle` showing dissatisfaction with current limitations, `@alexatallah` mentioned that longer completions could significantly increase memory usage.

- **Technical Assistance Sought and Offered Among Users**: Users sought clarification and assistance on model intricacies, ranging from Claude API's handling of system role messages (`@njbbaer`, with a response by `@alexatallah`) to adapting model files for personal use (`@mikef0x.`). Insights included OpenRouter's facilitation of prompt customization using ChatML and direct prompts.

- **User Engagement with OpenRouter and Model Accessibility Issues**: Conversations highlighted user engagement with OpenRouter, as shown by the creation of a Google Sheets connection app by `@mostlystable`, and addressed accessibility issues with models like Nous Hermes 70B. Updates on model status and functionality were given by users such as `@louisgv` and `@spaceemotion`, with official responses from `@alexatallah`.

**Links mentioned**:

- [TOGETHER](https://api.together.xyz): no description found
- [NeverSleep/Noromaid-20b-v0.1.1 · Hugging Face](https://huggingface.co/NeverSleep/Noromaid-20b-v0.1.1#custom-format): no description found
- [The Introduction Of Chat Markup Language (ChatML) Is Important For A Number Of Reasons](https://cobusgreyling.medium.com/the-introduction-of-chat-markup-language-chatml-is-important-for-a-number-of-reasons-5061f6fe2a85): On 1 March 2023 OpenAI introduced the ChatGPT and Whisper APIs. Part of this announcement was Chat Markup Langauge which seems to have gone…
- [openchat/openchat-3.5-0106-gemma · Hugging Face](https://huggingface.co/openchat/openchat-3.5-0106-gemma): no description found
- [FreeGPT for Google Sheets (Full Tutorial + Template)](https://youtu.be/wtKMvbCamlw): How to on using Openrouter inside Google Sheets with custom template. 📍 Unlock the power of AI in Google Spreadsheets for free! 🚀 In this video, we&#39;ll walk...
- [Reddit - Dive into anything](https://www.reddit.com/r/SillyTavernAI/comments/188a3dx/this_is_why_i_love_noromaid20b/): no description found

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1215929549997604875) (30 messages🔥): 

- **Request for Image Caption Generator Assistance**: User `@madhurr` sought help with concatenating image features and captions layers due to mismatching shapes in a remote image caption generator project.
- **CUDA as Factorio**: User `@artificial_anteligence` made an analogy comparing parallel computing to the video game Factorio, discussing components like memory operations paralleling game mechanics.
- **Advice on Image and NLP Embedding**: In response to `@madhurr`, `@andreaskoepf` suggested using a linear layer to project image features to match the shape of NLP embeddings and shared a link to a relevant [Visual Instruction Tuning project](https://llava-vl.github.io/).
- **Learning CUDA and Triton**: User `@umerha` offered to give a community talk on Triton or prefix sum and provided a link to his [personal blog](https://umerha.github.io/) for background on his expertise.
- **Decoding Efficiency in LLMs Explored**: `@andreaskoepf` linked to a [PyTorch blog](https://pytorch.org/blog/flash-decoding/) discussing strategies to make large language model inference more efficient.

**Links mentioned**:

- [UmerHA’s blog](https://umerha.github.io/): This is where the description of your site will go. You should change it by editing the _config.yml file. It can be as long as you like! Happy blogging… ❤
- [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/): Motivation  
- [LLaVA](https://llava-vl.github.io/): no description found
- [CUDA MODE](https://www.youtube.com/@CUDAMODE): A CUDA reading group and community https://discord.gg/cuda-mode-1189498204333543425 Supplementary content here https://github.com/cuda-mode Created by Mark Saroufim and Andreas Köpf    
- [lectures/lecture9 at main · cuda-mode/lectures](https://github.com/cuda-mode/lectures/tree/main/lecture9): Material for cuda-mode lectures. Contribute to cuda-mode/lectures development by creating an account on GitHub.
- [Lecture 21 - Pinned Memory and Streams](https://youtu.be/aNchuoFCgSs?si=noG-T-QSPImfqzBs&t=1988): GPU Computing, Spring 2021, Izzat El HajjDepartment of Computer ScienceAmerican University of BeirutBased on the textbook:Programming Massively Parallel Proc...

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/) (1 messages): 

iron_bound: early tho sounds cool https://github.com/Deep-Learning-Profiling-Tools/triton-viz
  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1215576628520943656) (21 messages🔥): 

- **Thread Coarsening in CUDA**: `@cudawarped` describes **thread coarsening** as similar to loop unrolling, expecting performance improvements only when not fully utilizing memory throughput. They note no performance gain from vectorized loads depending on the workload, and recommend half precision to be read as floats due to cache line size.

- **Optimizing Memory Operations**: `@zippika` suggests using `int4` or `float4` to reduce the number of memory reads/writes, and discusses the potential benefits of vectorizing additions with the `__hadd2` operator in CUDA.

- **Performance Insights from NVIDIA CUDA**: `@zippika` shares a CUDA code snippet for a vectorized addition using `__hadd8` and hints at performance improvements based on observations from the NVIDIA Compute Unified Device Architecture (NCUD) tool.

- **Path to CUDA Proficiency**: Users discuss self-teaching methods for mastering CUDA, including searching through the CUDA toolkit, examining the C++ code generated by `nvcc`, and exploring repositories on GitHub. 

- **Discussion on NVIDIA Magnum IO**: `@joseph_en` brings up NVIDIA Magnum IO, a system meant for high-performance computing and machine learning, highlighting its ability to handle complex simulations and reduce negative performance impacts in multi-tenant environments.

**Links mentioned**:

[NVIDIA Magnum IO](https://www.nvidia.com/en-us/data-center/magnum-io/): IO Subsystem for Modern, GPU-Accelerated Data Centers

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1216307475037552690) (5 messages): 

- **In Search of Torch Compile Limits**: `@andreaskoepf` inquired if there is a cap on the kernel size produced by `torch.compile` and suggested adding a readme to document how to print Triton compile results. They also expressed an interest in delving into the PyTorch source to understand this better.
- **Kernel Launch Uncertainty**: `@marksaroufim` shared uncertainty regarding kernel size limits in `torch.compile` and mentioned the concept of using persistent kernels to model entire networks, without a clear understanding of the underlying trade-offs.
- **Processor vs Language & Compilation Engine**: `@mr.osophy` commented that there is work on creating a new language and compilation engine that compiles efficiently to all types of processors, emphasizing that the focus is not on processor design.
- **Performance Comparison Inquiry**: `@ingiing` asked whether libtorch performs faster than using `load_inline` in PyTorch, seeking to compare the two methods.
  

---


### CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1216112121667129465) (1 messages): 

- **CUDA-MODE Lecture on Reduction Trees**: `@andreaskoepf` announced **CUDA-MODE Lecture 9: Reductions**, informing `@everyone` that it would start in approximately 5 minutes. The lecture, presented by `<@325883680419610631>`, will cover topics such as minimizing control and memory divergence, reducing global memory access, and thread coarsening, as found in chapter 10 of the PMPP book.
  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1215615128741871626) (2 messages): 

- **Fine-tuning Memory Requirements Reduced**: `@iron_bound` discussed a novel approach called Gradient Low-Rank Projection (GaLore), which reduces memory usage by up to 65.5% and can train large language models more efficiently on a single GPU setup. Research details can be found in this [ArXiv paper](https://arxiv.org/abs/2403.03507).
- **70b Model Fine-tuning on Standard GPUs**: `@iron_bound` shared that using FSDP and QLoRA, a 70b language model can now be fine-tuned on a desktop computer with standard gaming GPUs like RTX 3090 or 4090. The full announcement and summary are available at [Answer.AI's blogpost](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html).

**Links mentioned**:

- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507): Training Large Language Models (LLMs) presents significant memory challenges, predominantly due to the growing size of weights and optimizer states. Common memory-reduction approaches, such as low-ran...
- [Answer.AI - You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html): We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.

  

---


### CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1216789025776472205) (1 messages): 

- **CUDA Training Series Resources Shared**: `@w0rlord` provided a [YouTube playlist](https://www.youtube.com/playlist?list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj) titled "cuda-training-series" with lectures linked to [GitHub](https://github.com/olcf/cuda-training-series) and [Oak Ridge National Laboratory's series](https://www.olcf.ornl.gov/cuda-training-series/).
- **Find Your CUDA Homework Here**: Accompanying the lectures, `@w0rlord` also shared the [homework repository on GitHub](https://github.com/olcf/cuda-training-series/tree/master) for the CUDA Training Series, containing training materials associated with NVIDIA's series.

**Links mentioned**:

- [cuda-training-series](https://www.youtube.com/playlist?list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj): from https://github.com/olcf/cuda-training-series and https://www.olcf.ornl.gov/cuda-training-series/
- [GitHub - olcf/cuda-training-series: Training materials associated with NVIDIA&#39;s CUDA Training Series (www.olcf.ornl.gov/cuda-training-series/)](https://github.com/olcf/cuda-training-series/tree/master): Training materials associated with NVIDIA&#39;s CUDA Training Series (www.olcf.ornl.gov/cuda-training-series/) - olcf/cuda-training-series

  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1216442240432345259) (3 messages): 

- **Custom CUDA Kernel Gig Alert**: `@jaivikramaditya` is seeking a developer to design a custom CUDA kernel, specifically a variant of **flashattention**, for machine learning applications. They are offering **$2,000 - $3,000 USD** for the project.
- **Skills Required for CUDA Kernel Project**: The right candidate must have experience in **algorithmic development**, especially in Transformers, and have some prior exposure to CUDA programming.
- **Open Invitation for Direct Messaging**: Interested developers can directly **DM `@jaivikramaditya`** to express their interest or to get more details about the custom CUDA kernel job opportunity.
  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1215585862268755988) (9 messages🔥): 

- **Multitasking with Learning Resources**: `@mertbozkir` advises watching videos and reading the accompanying book for a better learning experience, specifically mentioning that the book they are reading is **very informative**.
  
- **CUDA Tensor Library Adventures**: `@siddharth4570` is building their own tensor library in CUDA and inquires about whether others **write their own backpropagation implementations** or use autodiff libraries.

- **CUDA vs. PyTorch 2.0 + Triton for ML Performance**: `@jsm9036` questions the benefits of learning and using CUDA over high-level tools like PyTorch and Triton, which may offer most of the performance gains with less development time. `@iron_bound` suggests using high-level tools for fast prototyping and resorting to CUDA **when performance needs further improvement**.

- **CUDA C/C++ Differences Sought**: `@apaz` seeks *an official list of differences between CUDA C/C++ and standard C/C++*, specifically during the initial phase of compilation from CUDA source to C/C++ source files.

- **Performance Observations with PyTorch Version Changes**: `@poppingtonic` reports a **speed difference between PyTorch 2.1.2 and 2.2.1** while running matmul_dyn on a 2080 Ti with CUDA 12.1, with the newer version being slower. They also express a new goal to understand tinygrad operations and kernels.
  

---


### CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1216812392802156754) (12 messages🔥): 

- **CUDA Profiling Discussed in Lecture**: `@marksaroufim` mentioned that while the **pmpp book** may not extensively cover profiling tools for CUDA, they do cover profiling in **lecture 1** in detail.
- **Book Authors Share Profiling Guidance on YouTube**: In response to an inquiry about profiling in CUDA, `@marksaroufim` informed `@dasher519` that the book authors discuss profiling in their [YouTube videos](https://www.youtube.com).
- **Syntax Nuance for CUDA Kernels Called Out in the Book**: `@alexanderrgriffing` raised a question regarding the use of **spacing inside triple angle brackets** (e.g., `<< < ... >> >`) in exercise code from the CUDA book.
- **Historical C++ Quirk Explains Spacing in Kernel Launch Syntax**: `@stefangliga` clarified that the spacing was once mandatory in **C++03/98** to avoid confusion with the shift operator, a requirement fixed in **C++11**; however, it's not clear if the same applied to CUDA C++ or if it was always a stylistic choice.
- **Searching for Exercise Solutions in the Latest Book Edition**: `@dasher519` inquired about the location of exercise solutions for the 2023 edition of the **pmpp book**, wondering if they might be in a separate solutions book.
  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1216216405607710790) (3 messages): 

- **CUDA Lecture on Reductions**: `@marksaroufim` shared a [YouTube video](https://www.youtube.com/watch?v=09wntC6BT5o) titled "Lecture 9 Reductions" along with [slide materials](https://docs.google.com/presentation/d/1s8lRU8xuDn-R05p1aSP6P7T5kk9VYnDOCyN5bWKeg3U/edit?usp=sharing) and [code examples](https://github.com/cuda-mode/lectures/tree/master).
- **Quality Check on Video Upload**: `@alexeyzaytsev` noticed that the video is only available in 360p quality one hour after upload, raising concerns about whether it was uploaded correctly. `@marksaroufim` confirmed that the upload is fine but it needs more time for processing.

**Links mentioned**:

[Lecture 9 Reductions](https://www.youtube.com/watch?v=09wntC6BT5o): Slides https://docs.google.com/presentation/d/1s8lRU8xuDn-R05p1aSP6P7T5kk9VYnDOCyN5bWKeg3U/edit?usp=sharingCode https://github.com/cuda-mode/lectures/tree/ma...

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1215681559001174016) (27 messages🔥): 

- **Sync Up Confirmation**: `@jamesmel` confirmed an upcoming sync, mentioning they will join the next day after missing the current one.
- **Troubleshooting Ring-Attention**: `@jamesmel` experienced issues while running ring-attention from **zhuzilin**, getting stuck at `_flash_attn_forward` and noted that subprocesses are paused.
- **Ring-Attention Discussion**: `@iron_bound` and `@andreaskoepf` also engaged in the conversation, with `@iron_bound` finding the pause odd and `@andreaskoepf` suggesting to look at allocations used in the ring-attn implementation.
- **Planning Flash Decoding Sketch**: `@jamesmel` proposed a high-level sketch for Flash decoding, listing steps including a prefill phase and a decoding phase, with `@andreaskoepf` mentioning a planned explanation as part of a lecture.
- **Meeting and Time Adjustments**: `@iron_bound` and `@jamesmel` commented on meeting schedules and adjustments due to daylight saving changes, clarifying timings for future syncs.
  

---


### CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1215782422818324580) (5 messages): 

- **Bing's Unforgettable Swear Word Memory**: `@iron_bound` shared a [Reddit post](https://www.reddit.com/r/ChatGPT/comments/1b8nyur/i_asked_bing_to_swear_at_me_weeks_ago_it_wont/) describing a humorous problem where Bing continues to swear at them despite requests to stop. The issue persists across conversations, as if the system only remembers this single preference.
  
- **Threading Anthems**: `@mr.osophy` made a light-hearted comment about a Spotify track, imagining that the song is about the programming concept of threading, and shared the [link to the song](https://open.spotify.com/track/4BGlpgcdytfO6x5Drgqxh7?si=oiDrfXj6Rlu1k-tTuY62zw).
  
- **Rumor Squashed on Inflection AI and Claude-3**: `@f_michael` inquired about a rumor regarding Inflection AI and their supposed use of Claude-3 with a link to a tweet, which was later clarified as just a rumor through a separate [response tweet](https://twitter.com/inflectionAI/status/1766173427441049684?s=20) shared by `@itali4no`.

- **Acknowledgment for Clarification**: `@f_michael` expressed gratitude to `@itali4no` for providing clarification on the previously mentioned rumor.

**Links mentioned**:

- [Reddit - Dive into anything](https://www.reddit.com/r/ChatGPT/comments/1b8nyur/i_asked_bing_to_swear_at_me_weeks_ago_it_wont/): no description found
- [At The Same Time](https://open.spotify.com/track/4BGlpgcdytfO6x5Drgqxh7?si=oiDrfXj6Rlu1k-tTuY62zw): Oliverse · Song · 2023

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1215610445029965844) (68 messages🔥🔥): 

- **PDF Extraction Assistance Request**: `@dazzling_puppy_08816` inquired about the possibility of sending an entire PDF to vision models for text extraction. `@smashah` suggested using an unstructured API to convert it into documents.
  
- **Template Limitations in Langchain Hub**: `@louis030195` expressed concerns about the limitations of template engines in handling conditional logic for prompts creation. `@baytaew` acknowledged the feedback and mentioned handling complex logic in code before passing to the template, recognizing the inconvenience for non-technical users.
  
- **ChatOllama Functionality Clarification**: `@tmetzger71` experienced an issue with binding functions in ChatOllama, with an error indicating 'tools' do not exist in type 'Partial<ChatOllamaCallOptions>'. Subsequent messages from the same user point to an experimental wrapper [Ollama Functions](https://js.langchain.com/docs/integrations/chat/ollama_functions) as a workaround.

- **Claude3 Support Query in Bedrock**: In a discussion initiated by `@j.barney` about Claude3 support, `@baytaew` indicated that `@761046695722877016` is working on enhancing the Bedrock chat class and ensuring first-class support for Claude3, noting unique API management for each model hosted on the Bedrock service.

- **Langchain Testing Strategy Concern**: `@sharrajesh` asked about best practices for maintaining application response quality in langchain/llm applications, given their non-deterministic nature. `@baytaew` responded with recommendations, such as using Langsmith for evaluations/benchmarking and focusing on metrics like system correctness, faithfulness, and retrieval performance.

**Links mentioned**:

- [Ollama Functions | 🦜️🔗 Langchain](https://js.langchain.com/docs/integrations/chat/ollama_functions): LangChain offers an experimental wrapper around open source models run locally via Ollama
- [Retrieving metadata from vector store · langchain-ai/langchain · Discussion #10306](https://github.com/langchain-ai/langchain/discussions/10306): Hi all I have created a chatbot which uses multiple tools to get an answer for a question. One of the tools queries a Pinecone index to get an answer. The structure of the chain is as follows: def ...
- [讯飞星火认知大模型-AI大语言模型-星火大模型-科大讯飞](https://xinghuo.xfyun.cn/sparkapi): no description found
- [iFLYTEK Open Platform Documents](https://global.xfyun.cn/doc/platform/pricing.html#billing-items): no description found
- [Support for claude v3 models. by 3coins · Pull Request #18630 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/pull/18630): Fixes #18513. Description This PR attempts to fix the support for Anthropic Claude v3 models in BedrockChat LLM. The changes here has updated the payload to use the messages format instead of the f...

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1216104812362469546) (2 messages): 

- **Seeking Clarity on LangChain Serve Code Execution**: `zql_flo` asked about the execution location of code when using Langchain Serve and how file uploads by users that agents need to access are managed. They inquired if Docker is the method for implementation for these processes.
  
- **Harnessing Output from Langserve Route**: `problem9069` is looking for guidance on using Langserve, specifically wanting to know how to capture the output from a route in a variable when adding routes using the ChatOpenAI function.
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1215696905967771678) (14 messages🔥): 

- **Revolutionize Prompting with Prompt Mixer**: User `@tomatyss` introduced a new tool called [Prompt Mixer](https://www.promptmixer.dev/), ideal for building, testing, and iterating on AI prompts. This desktop application allows connecting various models, tracking prompt versions, and it even has a [guide on adding custom connectors](https://docs.promptmixer.dev/tutorial-extras/create-a-custom-connector).

- **Lead Generation Automation in the Works**: User `@robinsayar` is developing an automated lead generation tool to categorize and qualify up-to-date public company information, aiming to streamline the process currently done manually for their clients.

- **Excitement for Automated Lead Generation**: User `@baytaew` expressed enthusiasm about `@robinsayar`'s project on automated lead generation and is looking forward to seeing the results.

- **Open Source Langchain Chatbot**: User `@haste171` shared an [open-source AI Chatbot](https://github.com/Haste171/langchain-chatbot), built on Langchain and RAG, which is designed for analyzing and extracting information from data in conversational format, featuring an easy setup and interactive UI.

- **Appstorm Launches Data GPTs in Version 1.5.0**: User `@appstormer_25583` announced the release of Data GPTs on Appstorm 1.5.0, a feature for exploring, analyzing, and visualizing data, complete with sample GPTs for various applications such as e-sports performance reports and healthcare stats infographics.

**Links mentioned**:

- [GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.](https://github.com/Haste171/langchain-chatbot): AI Chatbot for analyzing/extracting information from data in conversational format. - Haste171/langchain-chatbot
- [Prompt Mixer — Prompt IDE and LLMOps tool](https://www.promptmixer.dev/): PromptMixer – the innovative Prompt IDE for crafting, testing, and deploying prompts with unparalleled ease.
- [Create a Custom Connector | Prompt Mixer Docs](https://docs.promptmixer.dev/tutorial-extras/create-a-custom-connector): Step 1: Copy the Sample Connector
- [EStatGenie](https://beta.appstorm.ai/apps/905abbaf): no description found
- [HealthVizGPT](https://beta.appstorm.ai/apps/dcd2b5c9): no description found
- [DataVizGenie](https://beta.appstorm.ai/apps/093ae941): no description found
- [FootGraphix](https://beta.appstorm.ai/apps/435c088f): no description found
- [Watch Better Movies](https://www.watchbettermovies.com): no description found

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1216638358948810782) (2 messages): 

- **Exploring RAG with LangGraph**: `@mehulgupta7991` shared a [YouTube video](https://youtu.be/TlZ5BFx_m3M?si=tVfbYMUQhOVCV8x_) titled *"Improving RAG using LangGraph and LangChain"*, which demonstrates **how LangGraph can create cycles to enhance RAG retrieval** in external contexts.
  
- **Building a Chatbot with RAG and LangChain**: `@infoslack` provided a [YouTube link](https://www.youtube.com/watch?v=O60-KuZZeQA) for learning how to **build a chatbot using Retrieval Augmented Generation (RAG)**, utilizing OpenAI's gpt-3.5-turbo LLM, as part of the LangChain tutorial series.

**Links mentioned**:

- [Improving RAG using LangGraph and LangChain](https://youtu.be/TlZ5BFx_m3M?si=tVfbYMUQhOVCV8x_): This video demonstrates what LangGraph is and how it can be used to create cycles and improve RAG retrieval in an external context.LangChain in your Pocket: ...
- [Chatbot with RAG, using LangChain and OpenAI](https://www.youtube.com/watch?v=O60-KuZZeQA): In this video, I will guide you on how to build a chatbot using Retrieval Augmented Generation (RAG) from scratch. We will use OpenAI&#39;s gpt-3.5-turbo LLM, wh...

  

---



### DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1216193322398388244) (12 messages🔥): 

- **Judging Creative Writing with AI**: `@.calytrix` expressed skepticism about effectively training a model to judge creative writing, suspecting it requires more parameters than currently feasible. They are instead testing prompts with **GPT-4** and **Claude3** as judges using a detailed scoring criteria.
- **Exploring AI Judges**: `@johannhartmann` showed interest in seeing differences in creative writing judgements made by **GPT-4 and Claude3** and later joked about reading an "open-source outperforms gpt-4" conclusion.
- **Benchmarking AI Models in German**: `@johannhartmann` mentioned integrating Vago solutions into FastEval for **German benchmarks**, observing that GPT-4 still scores better with long detailed answers.
- **Ensemble of AI Judges**: `@bjoernp` discussed the advantage of using an ensemble of AI judges to reduce bias and enhance accuracy, asking if **Mistral large** had been used for judging.
- **Benchmark Development for AI Judges**: `@.calytrix` is creating a benchmark with multiple questions to test several AI judges, indicating that **GPT-4** might be a comparable state-of-the-art (SOTA) judge for this aspect. They are also considering using FastEval, as `@johannhartmann` suggested, which might be more suitable than EQ-Bench.
  

---


### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1215581805646839868) (4 messages): 

- **Evo Unveils Striped Hyena Architecture**: `@rasdani` highlighted the release of **Evo**, a biological foundation model using the StripedHyena architecture, meant for tasks ranging from molecular to whole genome scale. The model, developed by Together AI and the Arc Institute, can handle over 650k tokens and is specialized in DNA, RNA, and protein sequences. Find more in their [blog post](https://www.together.ai/blog/evo).

- **AutoMerger Faces Technical Issues**: `@johannhartmann` expressed interest in **AutoMerger**, an automatic model merger with benchmarks on Hugging Face, though noted that it is currently non-functional. His interest remains despite the tool being broken, as indicated in the link [Hugging Face’s AutoMerger](https://huggingface.co/spaces/mlabonne/AutoMerger).

- **Slerp vs. Dare_ties Merges**: Further commenting, `@johannhartmann` observed that there doesn't seem to be a significant difference between dare_ties and slerp merge strategies in the context of **AutoMerger**.

- **Mixture-of-LoRAs Architecture for LLMs**: `@johannhartmann` shared a link to an [arXiv paper](https://arxiv.org/abs/2403.03432) discussing **Mixture-of-LoRAs (MoA)**, a method designed for enhancing multi-task learning with Large Language Models (LLMs) and mitigating issues like catastrophic forgetting and task interference.

**Links mentioned**:

- [Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models](https://arxiv.org/abs/2403.03432): Instruction Tuning has the potential to stimulate or enhance specific capabilities of large language models (LLMs). However, achieving the right balance of data is crucial to prevent catastrophic forg...
- [AutoMerger - a Hugging Face Space by mlabonne](https://huggingface.co/spaces/mlabonne/AutoMerger): no description found
- [Evo: Long-context modeling from molecular to genome scale](https://www.together.ai/blog/evo): no description found

  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1216799679925190656) (3 messages): 

- **Introducing tinyBenchmarks**: User `@johannhartmann` shared a [link to tinyBenchmarks on Hugging Face](https://huggingface.co/tinyBenchmarks/tinyWinogrande), a dataset designed for efficient benchmarking.
- **Exploring Translation Possibilities**: `@johannhartmann` expressed an interest in potentially translating the [tinyBenchmarks/tinyWinogrande](https://huggingface.co/tinyBenchmarks/tinyWinogrande) dataset, planning to examine its feasibility the following day.
- **Benchmarking Insights from Hellaswag**: `@_chromix_` detailed their testing experience with the Hellaswag dataset, noting score fluctuation within a range of 2.5 after 1000 data points, and a more stable score variation of +/- 0.2 after 9000 data points. They suggested that choosing only 100 datapoints is likely inadequate for anything beyond a rough comparison.

**Links mentioned**:

[tinyBenchmarks (tinyBenchmarks)](https://huggingface.co/tinyBenchmarks): no description found

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1215570910300602368) (14 messages🔥): 

- **Innovative Training with the German Orca Dataset**: `@johannhartmann` explained their method of using a German translation of the **slim orca dataset** to train and merge models like **Mistral**. They use the SPIN-like method - taking the output of one model as input for the next - and track the relationships between models through the dataset, monitoring how training affects verbosity and answer quality.

- **Brezn3 Model Outshines its Predecessor**: `@crispstrobe` noted that **Brezn3** scored significantly higher than **Brezn-7b** on the EQ-Bench (v2) (de) benchmark, asking `@johannhartmann` if the improvement was due to changes in the model and tokenizer settings.

- **Awaiting the Final Push of Dpo**: `@johannhartmann` informed `@crispstrobe` that the **Dpo (Domain Prediction Override)** process was still in progress with approximately 13 hours remaining until completion.

- **Technical Troubleshooting in Model Merging**: `@crispstrobe` sought help from `@johannhartmann` regarding a TypeError encountered during model merging, which `@johannhartmann` addressed by sharing a fix via a [GitHub commit link](https://github.com/mayflower/mergekit/commit/cca4a8d91c213b6e5e4ac34b151955187ceff8a4).

- **Consistency Issues in Base Model Selection**: `@johannhartmann`, prompted by `@bjoernp`, discussed the inconsistencies when using **LeoLM/leo-mistral-hessianai-7b-chat** as a base model due to differences in the chatml and eos token settings, and planned to switch to **DiscoLM** as the base for better results in benchmarking.

**Links mentioned**:

- [SanjiWatsuki/Lelantos-7B · Hugging Face](https://huggingface.co/SanjiWatsuki/Lelantos-7B): no description found
- [Allow tokenizer_source: union with dare_ties · mayflower/mergekit@cca4a8d](https://github.com/mayflower/mergekit/commit/cca4a8d91c213b6e5e4ac34b151955187ceff8a4): no description found

  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1215755362024685658) (7 messages): 

- **Tackling Hallucinations in AI**: `@tirmizi7715` mentioned Yi's technical report which discusses reducing hallucinations. Various interpretations of this statement were discussed but not conclusively defined.
- **Strategies for Reducing AI Hallucinations**: `@rusch` speculated that reducing hallucinations might involve externalizing the knowledge base through RAG (Retrieval-Augmented Generation) or ensuring the fine-tuning data contains only new facts.
- **Fine-Tuning Data's Role in Minimizing Hallucinations**: `@scrungle.tech` considered the possibility of using a validation set of facts and manually rewriting repetitive responses for fine-tuning as a method to reduce hallucinations.
- **Latest from LAION**: `@spirit_from_germany` shared a [Twitter link](https://twitter.com/laion_ai/status/1766596812347941234) but provided no context or explanation in the message.
- **Search for Efficient Small Embedding Models**: `@joshxt` inquired about the best small embedding model that supports 1024+ max input length and can be run locally with minimal RAM. No answers were provided in the summarized messages.
  

---


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1216458189441335528) (8 messages🔥): 

- **Claude's Proficiency at Diagramming Code**: `@joshxt` highlighted the potential for using **Claude** to convert entire code bases into [mermaid graphs](https://mermaid.live/), mentioning successful trials on code bases ranging from 10k to 96k tokens.
- **Mermaid Graphs Explained**: `@lightningralf` clarified for `@teknium` what a mermaid graph is, describing it as a syntax for creating diagrams from text, and shared the [GitHub repository](https://github.com/mermaid-js/mermaid) for mermaid.
- **Visualizing Code with Mermaid**: `@joshxt` provided a practical example of a mermaid graph syntax to visualize a code base's architecture, showing components like `app.py`, `FASTAPI`, and various API endpoints.



**Links mentioned**:

[GitHub - mermaid-js/mermaid: Generation of diagrams like flowcharts or sequence diagrams from text in a similar manner as markdown](https://github.com/mermaid-js/mermaid): Generation of diagrams like flowcharts or sequence diagrams from text in a similar manner as markdown - mermaid-js/mermaid

  

---


### Alignment Lab AI ▷ #[alignment-lab-announcements](https://discord.com/channels/1087862276448595968/1124055853218136175/1216302354563858452) (1 messages): 

- **Gemma-7b Enhanced with C-RLFT**: `@imonenext` announced the first usable Gemma-7b fine-tune based on openchat-3.5-0106 data and methods, achieving nearly the same performance as Mistral-based versions. The fine-tuning leveraged 6T tokens, hinted as the "secret recipe", and the model is available on [HuggingFace](https://huggingface.co/openchat/openchat-3.5-0106-gemma).
- **Gemma's New Milestone Tweeted**: A tweet from `@openchatdev` celebrated the **World's First Gemma fine-tune** using C-RLFT and its comparable performance to Mistral. The fine-tune potentially involves 6T pre-training tokens among other factors, as indicated in the [Twitter post](https://fxtwitter.com/openchatdev/status/1766516456034861237).

**Links mentioned**:

- [Tweet from OpenChat (@openchatdev)](https://fxtwitter.com/openchatdev/status/1766516456034861237): 🚀 The World&#39;s First Gemma fine-tune based on openchat-3.5-0106 data and method (C-RLFT). Almost the same performance as the Mistral-based version.  6T tokens = secret recipe?  HuggingFace: https:...
- [openchat/openchat-3.5-0106-gemma · Hugging Face](https://huggingface.co/openchat/openchat-3.5-0106-gemma): no description found

  

---


### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1133673143064596644/1216307576824664105) (5 messages): 

- **Gemma 7B vs. Mistral 7B - Why release an underperformer?**: `@chatbooapp` inquired as to why **Gemma 7B** was released if it doesn't surpass **Mistral 7B**. `@joshxt` responded, stating that **each model is an experiment**, and Gemma may excel in ways not yet known.
- **Gemma's potential moderation performance**: `@chatbooapp` speculated whether **Gemma 7B** might underperform in NSFW content moderation compared to **Mistral**, due to Google's strict moderation policies. This aspect seemed to resonate with `@joshxt`'s experiences where Gemma failed to impress in tasks they use LLMs for, yet they hadn't tried any fine-tuned models.
- **Mistral's NSFW allowance noted**: Despite moderation concerns, `@chatbooapp` mentioned that even the **Mistral endpoint** doesn't shy away from NSFW content, applauding its capabilities. They also highlighted that **Mistral Large**, when combined with a well-crafted system prompt, can be incredibly helpful.
  

---


### Alignment Lab AI ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/1216458278674894849) (5 messages): 

- **Rumors of User Demise Exaggerated**: `@teknium` humorously expressed concern, using a '<:sad_cat:905917016517521419>' emote, that another user might have "died or somethin".
- **Alive and Coding**: `@autometa` quashed rumors of their demise, stating they are simply "buried in petty coding tasks atm".
- **Docker Development Dilemmas**: `@autometa` mentioned the challenge of setting up a Docker environment to streamline collaborative efforts, eliminate the need for manual sampling, and optimize their development processes.
- **Call for Collaborative Coding**: `@autometa` made a plea for assistance with Docker environment setup, emphasizing that help would be "phenomenal to get moving" with their work.
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1216455617603043398) (1 messages): 

- **Free Access to Claude 3 Opus with Vercel Pro**: User `@jeffreyw128` shared that those who have Vercel Pro can use **Claude 3 Opus** and **GPT-4 vanilla** for free. They provided a link to the Vercel SDK: [sdk.vercel.ai](https://sdk.vercel.ai/).

**Links mentioned**:

[Vercel AI SDK](https://sdk.vercel.ai/): Build AI-powered applications with the latest AI language models

  

---


### LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1216778435091759205) (1 messages): 

- **Transition Inquiry from OpenAI to Azure**: User `@pantsforbirds` is seeking insights on moving from **OpenAI's SDK** to the Azure-based approach for their project. They are interested in understanding potential challenges faced during this migration process.
  

---


### LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1215687345974812672) (15 messages🔥): 

- **Function Calling Praised with XML Tags**: `@res6969` confirmed that function calling works well, although the use of **XML tags** enhances its effectiveness.
- **XML's Impact on Sharing Prompt Generators**: `@pantsforbirds` highlighted that the necessity of **XML** makes sharing a prompt generator more difficult.
- **General Superiority of Opus Over GPT-4**: Multiple users, including `@jeffreyw128`, `@nosa_.`, and `@vgel`, remarked on the overall better performance of **Opus** compared to **GPT-4**, with specific mention of its more "insightful/smart answers" and effectiveness in handling a complex graph BFS algorithm.
- **Claude's Prose Preferred Over GPT's Style**: `@potrock` expressed a preference for **Claude's** prose, noting it avoids the **condescending** explanations often preceding GPT's answers.
- **Anticipation for GPT-4.5's Release and Performance**: `@jeffreyw128` and `@res6969` are looking forward to the potential release of **GPT-4.5** or **GPT-5**, speculating on its capabilities compared to **Claude**, alongside excitement for the **Starship launch**.
  

---


### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 messages): 

res6969: https://x.com/elonmusk/status/1767108624038449405?s=20
  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1216598957350981732) (1 messages): 

- **Google's Potential Dominance in AI**: `@jeffreyw128` discusses two key reasons Google could dominate **general AI adoption**: lack of a long-term moat in foundation models and Google's ability to integrate and serve AI cost-effectively within its search and Chrome platforms.
- **Affordable AI Integration with Google**: With the current revenue from search queries, Google has the potential to offer **AI services within searches** at a negligible cost, potentially rolling out a **Generative Search Experience** widely within the year.
- **OpenAI's Lead May Foster Mass Adoption**: Despite the competition, `@jeffreyw128` believes **OpenAI** will stay ahead for a few years, fostering significant adoption and specialized applications like code generation.
- **Google's Dynamic AI Deployment**: Google's advantage, as noted by `@jeffreyw128`, is in its intelligent choice between text generation and supplying extractive answers, potentially outperforming other online LLM experiences.
- **The Future of AI Integration and Premium Experiences**: Moving beyond browsers and search integrations, deeper hardware integrations may emerge. However, while consumer AI applications will be economically viable, there will still be a market for **premium AI experiences** in areas like coding or writing.
  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1216843016376156260) (1 messages): 

- **Quantum Leap in Convergence Acceleration**: `@baptistelqt` claimed to have developed a method to accelerate convergence by a factor of 100,000. Each "round" involves training **from scratch**.
  

---


### Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/) (1 messages): 

henkdevries_starbound: math quuestions are hard
  

---


### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=H6xon8K4Ius
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/) (1 messages): 

dbreunig: Earhart
  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1216512369966973129) (2 messages): 

- **Praise for Symbex**: User `@bdexter` expressed gratitude for **symbex**, noting frequent use of the project.
- **SimonW Acknowledges Symbex Fun**: `@simonw` responded with enthusiasm, describing **symbex** as a "really fun project".
  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/) (1 messages): 

.zhipeng: from nathan's interconnectai blogpost right ?
  

---


### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1216561269537116250) (1 messages): 

- **Gen AI Video Event Announced**: `@sylviatong` invites all to a deep dive on Gen AI Video and the 'World Model'. The lineup features [Lijun Yu](https://www.linkedin.com/in/lijun-yu/) from Google, [Ethan He](https://twitter.com/EthanHe_42) of Nvidia, Shan Jin from Goodby Silverstein & Partners, and Justin Hackney of Eleven Labs; the event is moderated by Cindy Le and will dispel myths around #Sora, #Genie, and #WorldModel. **March 16, 2024**, in San Francisco & on Zoom. [RSVP link](https://lu.ma/b0zrw3q3).
- **Conversation with AI Video Pioneers**: The event offers a platform for learning from top-tier researchers and promises real, unfiltered conversations. Expect insights into Google's VideoPoet, Nvidia's Sora description, and more creative technologies in AI Video.

**Links mentioned**:

[Gen AI Video Breakout and World Model by EntreConnect - #Sora #Genie #VideoPoet #V-JEPA #LTXStudio #AnimateDiff · Luma](https://lu.ma/b0zrw3q3): Join us for a groundbreaking event that dives deep into the heart of Gen AI Video! This isn&#x27;t just another tech talk; it&#x27;s a journey into the future. We will also provide dial-in options, wh...

  


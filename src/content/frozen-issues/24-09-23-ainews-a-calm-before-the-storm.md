---
id: dd694420-bc14-4229-b025-f0f76213db66
title: a calm before the storm
date: '2024-09-23T23:33:49.803194Z'
original_slug: ainews-sxxx
description: >-
  **Anthropic** is raising funds at a valuation up to **$40 billion** ahead of
  anticipated major releases. **OpenAI** launched new reasoning models **o1**
  and **o1-mini**, with increased rate limits and a multilingual MMLU benchmark.
  **Alibaba** released the open-source **Qwen2.5** model supporting 29+
  languages, showing competitive performance to **gpt-4** at lower cost.
  **Microsoft** and **Blackrock** plan to invest **$30 billion** in AI data
  centers, with **Groq** partnering with Aramco to build the world's largest AI
  inference center. Robotics advances include Disney Research and ETH Zurich's
  diffusion-based motion generation for robots and Pudu Robotics' semi-humanoid
  robot. Slack and Microsoft introduced AI-powered agents integrated into their
  platforms. Research highlights include long-context scaling for
  **llama-2-70b** using Dual Chunk Attention and KV cache quantization enabling
  1 million token context on **llama-7b** models.
companies:
  - anthropic
  - openai
  - alibaba
  - microsoft
  - blackrock
  - groq
  - aramco
  - disney
  - eth-zurich
  - pudu-robotics
  - slack
models:
  - o1
  - o1-mini
  - qwen2.5
  - gpt-4
  - llama-2-70b
  - llama-7b
topics:
  - long-context
  - kv-cache-quantization
  - diffusion-models
  - reinforcement-learning
  - robotics
  - ai-integration
  - multilinguality
  - model-benchmarking
  - model-performance
  - model-optimization
people:
  - adcock_brett
  - philschmid
  - rohanpaul_ai
  - jvnixon
  - kateclarktweets
  - sama
---


<!-- buttondown-editor-mode: plaintext -->**Peace is all you need.**

> AI News for 9/20/2024-9/23/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**221** channels, and **6206** messages) for you. Estimated reading time saved (at 200wpm): **719 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

No clear headline story, but lots of minor notables ahead of anticipated big drops from Anthropic and Meta this week:

- [CUDA MODE](https://x.com/swyx/status/1837577267259887702) and [Weights and Biases](https://x.com/morgymcg/status/1838062480926368013) (sponsor of this month's inference) hosted successful hackathons this weekend. CUDA MODE celebrated with [a rebrand to GPU MODE](https://x.com/jeremyphoward/status/1838341110344880637).
- Berkeley Function Calling Leaderboard [shipped V3](https://x.com/shishirpatil_/status/1837205152132153803)  (yes, [v2 was only last month](https://buttondown.com/ainews/archive/ainews-ideogram-2-berkeley-function-calling/)) focusing on multi-turn/step function calling. O1 mini does surprisingly poorly.
- a couple more notable o1 evals - on [test time budget](https://x.com/hughbzhang/status/1838288923656941860) and [a formal paper exploring its planning](https://x.com/polynoamial/status/1838251987009183775?s=46)
- Anthropic [raising again at up to a $40b valuation](https://x.com/KateClarkTweets/status/1838319202798538974)
- OpenAI shipped [multilingual MMLU (MMMLU)](https://x.com/_philschmid/status/1838230108072476951?s=46).
- Sama calls this [the Intelligence Age](https://ia.samaltman.com/).
- the [Jony Ive phone was confirmed by the NYT](https://x.com/8teapi/status/1837979330867351626?s=46) and [Scale AI deals with a minor crisis](https://x.com/natolambert/status/1837996707780624631?s=46).


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

**AI Developments and Industry Updates**

- **OpenAI's New Models**: [@adcock_brett](https://twitter.com/adcock_brett/status/1837885345972605182) reported on OpenAI's release of new reasoning models, o1 and o1-mini, designed for complex tasks in science, coding, and math. [@JvNixon](https://twitter.com/JvNixon/status/1837884523092283599) noted subjective improvements in output quality with these models. OpenAI also [increased rate limits](https://twitter.com/adcock_brett/status/1837885561203224595) for o1-mini to 50 messages per day and o1-preview to 50 messages per week.

- **Qwen2.5 Model**: Alibaba released [Qwen2.5](https://twitter.com/adcock_brett/status/1837885606384312457), an open-source model with versions for general use, coding, and math, supporting 29+ languages. [@_philschmid](https://twitter.com/_philschmid/status/1837932334823145535) compared its performance to GPT-4, noting similar results at a fraction of the cost.

- **AI Infrastructure**: Microsoft and Blackrock are [raising $30 billion](https://twitter.com/adcock_brett/status/1837885460120547541) to invest in new and existing AI data centers, with potential for $100 billion total investment. Groq partnered with Aramco to build ["the world's largest AI inference center"](https://twitter.com/adcock_brett/status/1837885437651677217) with 19,000 LPUs, eventually growing to 200,000.

- **AI in Robotics**: Disney Research and ETH Zurich presented ['RobotMDM'](https://twitter.com/adcock_brett/status/1837885482669162795), combining diffusion-based motion generation with RL for robot movement. Pudu Robotics announced their [first generation 'semi-humanoid'](https://twitter.com/adcock_brett/status/1837885392097358135) robot.

- **AI Integration in Tech Products**: Slack announced [new AI-powered features](https://twitter.com/adcock_brett/status/1837885415161794902), including AI agents within channels. Microsoft introduced [agents coming to Microsoft 365 Copilot](https://twitter.com/adcock_brett/status/1837885369053831582), working across various Microsoft products.

**AI Research and Techniques**

- **Long Context Models**: A paper on ["Training-Free Long-Context Scaling of Large Language Models"](https://twitter.com/rohanpaul_ai/status/1837853153246470414) introduced Dual Chunk Attention (DCA), enabling Llama2 70B to support context windows of more than 100k tokens without continual training.

- **KV Cache Quantization**: The ["KVQuant" paper](https://twitter.com/rohanpaul_ai/status/1837852496364023831) proposed techniques for quantizing cached KV activations, allowing a LLaMA-7B model to be served with a context length of up to 1 million on a single A100-80GB GPU.

- **Retrieval Techniques**: [@_philschmid](https://twitter.com/_philschmid/status/1837752035501858975) discussed SFR-RAG, a fine-tuned 9B LLM for RAG that matches larger models in performance on academic benchmarks.

- **Synthetic Data**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1837982057693131055) highlighted the crucial role of synthetic data in training Qwen2.5-Coder, detailing the generation process, validation, and integration with open-source datasets.

**AI Tools and Applications**

- **GitHub File Organizer**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1837985813935641024) shared a GitHub repo for a file organizer that uses local LLMs to understand and sort files based on their content.

- **Financial Research Assistant**: [@virattt](https://twitter.com/virattt/status/1837878341405167778) is building an open-source financial research assistant using LangChain, with powerful search tools for financial and web data.

- **Perplexity-like Experience**: [@LangChainAI](https://twitter.com/LangChainAI/status/1837899668103352700) shared an open-source repo using LangGraph, FastHTML, and Tavily to create a Perplexity-like experience, supporting different models including GPT-4 and Llama3.

**AI Ethics and Regulation**

- **California AI Bill SB 1047**: There's ongoing debate about the California AI Bill SB 1047. [@JJitsev](https://twitter.com/JJitsev/status/1837905422415540373) argued that the bill is deeply flawed, regulating general-purpose technology rather than its applications. Several AI researchers and institutions have expressed concerns about the bill's potential impact on AI research and development.

**Miscellaneous**

- **AI Contributions on GitHub**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1837829123625853259) noted that AI contributions on GitHub have surged 230% since OpenAI released ChatGPT.

- **AI Data Centers**: [@ylecun](https://twitter.com/ylecun/status/1837875035270263014) suggested that future AI data centers will be built next to energy production sites, particularly nuclear power plants, for efficient, low-cost, and low-emission electricity.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Qwen2.5 Emerges as New Open Source SOTA, Replacing Larger Models**


- **Who replaced a model with Qwen2.5 for a daily setup? If so, which model did you replace?** ([Score: 42, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1fmoa14/who_replaced_a_model_with_qwen25_for_a_daily/)): **Qwen2.5** is reported to achieve **state-of-the-art (SOTA)** performance across a wide range of tasks, with model sizes ranging from **0.5B to 72B parameters**. The post author is inquiring about users who have integrated Qwen2.5 into their daily workflows, asking which specific models they replaced and for what tasks.
  - **Professional-Bear857** replaced **Llama 3.1 70B IQ2_M** with **Qwen2.5 32B IQ4_XS** for code editing/correction and general queries, citing lower GPU power usage and comparable performance to **Mistral Large**.
  - Users are experimenting with **Qwen2.5** for various tasks, including **article and YouTube video summarization**. **Matteogeniaccio** uses a custom Python setup with **llama.cpp server** to process different content types and extract key information.
  - While some users praise Qwen2.5's instruction-following capabilities, others report mixed results. **Frequent_Valuable_47** found **Gemma2 2B** superior to **Qwen2.5 1.5B** for YouTube transcript summaries, despite Qwen2.5's larger **120k token context** compared to Gemma's **8k**.


**Theme 2. Safe Code Execution in Open WebUI Using gVisor Sandboxing**



- **[Safe code execution in Open WebUI](https://github.com/EtiennePerot/open-webui-code-execution)** ([Score: 324, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1fnbimz/safe_code_execution_in_open_webui/)): Open WebUI has implemented **safe code execution** using **Docker containers** for enhanced security. This feature allows users to run code snippets within isolated environments, preventing potential harm to the host system while enabling interactive coding experiences. The implementation utilizes **Docker SDK** for container management and includes a **timeout mechanism** to automatically terminate long-running processes.
  - The code execution feature is available on [GitHub](https://github.com/EtiennePerot/open-webui-code-execution) and uses **gVisor** for sandboxing. It offers two modes: **"Function"** for running code blocks in LLM messages and **"Tool"** for allowing LLMs to autonomously execute code.
  - Users discussed extending support to other languages like **Go**, with the developer explaining that modifications to the `Sandbox` class and interpreter selection code would be necessary. The tool currently works with **Ollama** backend and models tagged for tool calling.
  - Concerns were raised about handling missing dependencies and the need for more robust features like artifacts and increased concurrent requests. The developer confirmed that **Open WebUI v0.3.22** includes necessary fixes for the tool to function properly.


**Theme 3. NSFW AI Models Optimized for Roleplay Scenarios**



- **Favorite small NSFW RP models (under 20B)?** ([Score: 180, Comments: 156](https://reddit.com//r/LocalLLaMA/comments/1fmqdct/favorite_small_nsfw_rp_models_under_20b/)): The post compares various **small NSFW RP models under 20B parameters**, categorizing them as "Good," "Great," and "ABSOLUTELY FANTASTIC." The author exclusively uses **EXL2** models, with top picks including **MN-12b-ArliAI-RPMax-EXL2-4bpw**, **estopia-13b-llama-2-4bpw-exl2**, and **Mistral-Nemo-Instruct-2407-exl2-4bpw**. Most models listed are **4-4.5bpw** (bits per weight) variants, with sizes ranging from **7B to 13B** parameters.
  - Users discussed various **NSFW RP models**, with **L3-Nymeria-Maid-8B-exl2** and **Cydonia 22B** highlighted as particularly impressive. **Nicholas_Matt_Quail** provided extensive insights on model evolution, noting that **Cydonia 22B** feels like a significant upgrade over 12B models.
  - The community shared recommendations for different VRAM capacities, including **Sao10K_L3-8B-Stheno** for 4GB and **L3-Super-Nova-RP-8B** for higher capacities. Users emphasized the importance of proper **sampling techniques** and **instruct templates** for optimal model performance.
  - Discussions touched on the use cases for uncensored models, including explicit sexual content and non-sexual scenarios involving violence or dark themes. The **chub.ai** website was mentioned as a resource for character cards and RP scenarios.


**Theme 4. Jailbreaking and Censorship Testing of Qwen2.5 Models**



- **Qwen2.5 is able to be jailbroken, but it's not perfect.** ([Score: 49, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1fmvj0n/qwen25_is_able_to_be_jailbroken_but_its_not/)): **Qwen2.5 models** (72b, 32b, 14b) were tested for censorship using Ollama and Open-webui, with initial attempts to ask about **Uyghur persecution** resulting in 100% rejection. A **custom system prompt** was developed to encourage unbiased, detailed responses, which successfully bypassed censorship for questions about Uyghurs and Hong Kong, achieving **100% uncensored answers** in 20 tests. However, the method proved **ineffective for direct questions about the Chinese government**, suggesting a persistent "block" on such topics, while questions about other governments (e.g., American) received more critical responses.
  - Users discussed the model's responses, with some noting it gave a **"well-worded gut punch"** about political greed in America while being more restrained on Chinese topics. The **32b model** was praised for its performance, with mentions of **128k context** capability.
  - Debate arose over whether the model's responses indicate **censorship or bias** from training data. Some argued that the model's pro-China stance might reflect its training rather than deliberate censorship, while others suggested potential **"ablation"** of certain topics.
  - A user tested the **14b model** with a prompt about **Tiananmen Square**, receiving a surprisingly detailed response covering key events and aftermath. This sparked discussion about the model's ability to address sensitive topics and the influence of prompt wording on responses.


**Theme 5. Limited Excitement for New Command-R Model Updates**



- **no love for new command r ?** ([Score: 33, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1fmt93e/no_love_for_new_command_r/)): The post discusses the recent **improvements to the Command-R model** by **Cohere**, noting a lack of public enthusiasm compared to its initial release **about six months ago**. Despite Cohere's claims of enhanced capabilities in **reasoning, RAG, math, and coding**, the author observes a notable absence of **benchmarks, blog posts, LocalLLaMA adaptations, or YouTube reviews** for the updated model. The post concludes by asking if anyone is using the new Command-R and invites users to share their experiences.
  - Users compared **Command-R** to other models like **Qwen2.5-32B**, **Mistral 123b**, and **Magnum 123b**, with mixed opinions on performance. Some found Command-R better for specific tasks like **storytelling** and **document chatting**, while others preferred alternative models.
  - The **non-commercial license** of Command-R was cited as a significant factor limiting interest and adoption. Users expressed frustration with the restrictive terms, particularly the prohibition on commercial use of outputs, which some viewed as hypocritical given Cohere's data collection practices.
  - The new Command-R was noted to be **worse for RP/ERP** compared to the original release, which had accidentally excelled in this area. However, improvements in **GQA** allow for better performance with **large context lengths up to 128k**, potentially benefiting **RAG and tool use** applications.

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Research and Techniques**

- **Google Deepmind advances multimodal learning**: A [paper on joint example selection](https://arxiv.org/html/2406.17711v1) demonstrates how data curation can accelerate multimodal learning. (/r/MachineLearning)

- **Microsoft's MInference speeds up long-context inference**: [MInference](https://arxiv.org/abs/2407.02490) enables inference of up to millions of tokens for long-context tasks while maintaining accuracy. (/r/MachineLearning)

- **Scaling synthetic data creation with 1 billion web-curated personas**: A [paper on scaling synthetic data creation](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) leverages diverse perspectives within large language models to generate data from web-curated personas. (/r/MachineLearning)

**AI Model Releases and Improvements**

- **Salesforce releases xLAM-1b model**: The 1 billion parameter model [achieves 70% accuracy in function calling, surpassing GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/). (/r/LocalLLaMA)

- **Phi-3 Mini updated with function calling**: Rubra AI released an updated Phi-3 Mini model [with function calling capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/), competitive with Mistral-7b v3. (/r/LocalLLaMA)

- **Alibaba launches over 100 new open-source AI models**: Alibaba [released numerous AI models and a text-to-video generation tool](https://www.cnbc.com/2024/09/19/alibaba-launches-over-100-new-ai-models-releases-text-to-video-generation.html). (/r/singularity)

**AI Applications and Experiments**

- **Flux: Iterative image transformation**: An experiment showing [what happens when repeatedly feeding an output image back into a transformer block](https://www.reddit.com/r/StableDiffusion/comments/1fmu7eb/flux_what_happens_if_you_keep_feeding_the_output/). (/r/StableDiffusion)

- **Simple Vector Flux LoRA**: A demonstration of [vector-based image transformations using LoRA](https://www.reddit.com/r/StableDiffusion/comments/1fn465p/simple_vector_flux_lora/). (/r/StableDiffusion)

- **AI-generated desktop icons**: Discussion on [using AI to create custom desktop icons](https://www.reddit.com/r/StableDiffusion/comments/1fn2i9e/do_you_use_ai_to_make_custom_icons_for_your/). (/r/StableDiffusion)

**AI Ethics and Societal Impact**

- **Pope calls for Universal Basic Income**: The Pope [repeated his call for Universal Basic Income](https://www.indcatholicnews.com/news/50680), sparking discussions on AI's impact on employment. (/r/singularity)

- **Worldcoin's iris scanning for UBI**: Sam Altman's Worldcoin project [uses iris scanning for identity verification](https://www.businessinsider.com/worldcoin-sam-altman-iris-scanning-face-auth-tools-humanity-ubi-2024-8) in a proposed UBI system, raising privacy concerns. (/r/singularity)

**AI Humor and Memes**

- **Circuit board spear**: A humorous image of a [spear made with a circuit board tip](https://i.redd.it/nz2560p8mgqd1.jpeg), sparking discussions on post-apocalyptic scenarios and AI's role. (/r/singularity)

- **AI's perspective on evil**: A [ChatGPT conversation](https://i.redd.it/rwgnu6itobqd1.jpeg) where the AI identifies "humanity" as the source of evil, generating debate on AI ethics and human nature. (/r/OpenAI)


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1: New AI Model Releases and Updates**

- [**OpenAI Introduces O1 Models: A Leap in Reasoning**](https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-five): The **O1 models** showcase significant improvements in reasoning, jumping from *0% to 52.8%* on challenging benchmarks, hinting at potential synthetic data training.
- [**Aider v0.57.0 Enhances AI Pair Programming**](https://aider.chat/HISTORY.html): **Aider v0.57.0** now supports **OpenAI O1 models**, improves Windows compatibility, and integrates new **Cohere models**, with **70%** of the release coded by Aider itself.
- [**Gradio 5 Beta Released with Performance Boosts**](https://5-0-dev.gradio-website.pages.dev/playground): The **Gradio 5 Beta** introduces major performance enhancements, modern design updates, and an experimental **AI Playground** for quick app testing.

**Theme 2: Challenges and Issues with AI Tools and Models**

- **Perplexity Pro Users Face Subscription Woes**: Users reported intermittent loss of **Perplexity Pro** status, experiencing *'Query rate limit exceeded'* errors; temporary fixes like logging out were only partially effective.
- **LM Studio Models Hit Loading Snags After Updates**: After updating to **LM Studio**, users faced challenges loading models, with some resorting to rolling back versions to restore functionality.
- **OpenRouter Disables Middle-Out Transform by Default**: **OpenRouter** has disabled the **middle-out transform**, impacting users' workflows and causing confusion over prompt handling.

**Theme 3: AI in Creative Fields**

- [**AI-Powered RPG Development Underway**](https://github.com/slangerosuna/space_cowboy_rpg): A developer is creating an RPG game integrating **AI agents** with memory and networking, seeking community contributions due to the complexity of the system.
- [**Music Production AI Struggles with Music Theory**](https://www.reddit.com/r/LocalLLaMA/comments/15fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook): Discussions reveal that AI models in music production struggle with basic music theory tasks like transposing chords, highlighting limitations due to limited training data.
- [**Podcast Generation Technology Excites Users**](https://huggingface.co/spaces/saq1b/podcastgen): **PodcastGen** utilizes advanced techniques inspired by Google's NotebookLM to generate podcasts, though some users noted issues with content repetition.

**Theme 4: Developments in AI Research and Practices**

- [**μ-Parameterization Guide Simplifies Model Training**](https://blog.eleuther.ai/mutransfer/): **EleutherAI** and **Cerebras** released a joint guide to improve the accessibility of **μ-parameterization (μP)**, including step-by-step instructions and a simple implementation in [nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup).
- [**BFCL V3 Evaluates Multi-Turn Function Calling in LLMs**](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html): The **Berkeley Function-Calling Leaderboard V3** introduces a new evaluation for **multi-turn** and **multi-step function calling**, critical for assessing LLM performance in complex tasks.
- [**SetFit v1.1.0 Released with Enhanced Training Capabilities**](https://huggingface.co/posts/tomaarsen/875775738519407): **SetFit v1.1.0** now uses the **Sentence Transformers Trainer** for efficient classifier training on both **CPU and GPU**, with support for **MultiGPU** and Python **3.11** and **3.12**.

**Theme 5: Community Events and Collaborations**

- **Hackathon Showcases Innovative Projects at CUDA MODE**: The hackathon saw over **40 projects** created in a day, with teams selected for pitches focused on commercial viability and innovation, highlighting the community's collaborative spirit.
- **Participants Seek AI Internship Opportunities**: Members are actively seeking suggestions on where to find **AI internships**, reflecting the community's interest in advancing careers within the AI field.
- [**Open Interpreter Module Proposed for Smart Furniture**](https://www.kickstarter.com/projects/kequel/kequel-modular-customizable-bedside-table): A member proposed creating an **Open Interpreter** module for the **Kequel Modular Customizable Bedside Table**, seeking collaboration from the community.


---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingFace Spaces are down**: Users reported significant issues with **HuggingFace Spaces**, experiencing '500 Internal Error' and file upload failures that lasted several hours.
   - This downtime frustrated users who rely on the platform for model access and content uploads, highlighting its impact on productivity.
- **Fine-Tuning Models Simplified**: A user sought help for fine-tuning a model on a dataset of **350 records** concerning OS and hardware issues, finding support through shared resources like [SimpleTuner](https://github.com/bghira/SimpleTuner).
   - Various users discussed tools for model training, discovering effective solutions, including YouTube video recommendations and community insights.
- **3D Content Creation in Seconds**: A member shared the [threestudio GitHub repo](https://github.com/threestudio-project/threestudio), claiming 3D objects can be generated in under **10 seconds**.
   - Another participant recommended using 'stable fast 3D', which reportedly generates objects from images in less than one second, available in Hugging Face space.
- **Gradio 5 Beta Released**: **Gradio 5 (Beta)** is officially here, addressing developer feedback with enhancements in performance, design updates, and an experimental **AI Playground** for quick app testing.
   - This beta version promises major performance boosts, especially in server-side rendering, while ensuring improved security through a third-party audit.
- **Developing an AI-Powered RPG**: A developer is working on an RPG that integrates AI agents with memory and networking, facing complexities in system construction.
   - They reached out to the community for contributions, emphasizing the significant challenges in implementing such a sophisticated gaming structure.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.57.0 Brings Exciting Updates**: The launch of **Aider v0.57.0** enhances performance with various updates, including support for **OpenAI o1 models**, improved **Windows compatibility**, and integration of new **Cohere models**.
   - It also addresses multiple bugs, and users can access the full **change log** [here](https://aider.chat/HISTORY.html).
- **Aider and OpenRouter Ready but Bumpy**: Users shared mixed experiences using **Aider** with **OpenRouter** and **Claude models**, often facing 'overloaded' errors and confusion.
   - Some members accessed **Anthropic** models successfully, while others printed concerns about the reliability of service during current high traffic.
- **Doubts on Embeddings Highlighted**: A member expressed skepticism about the value of **embeddings**, advocating for a DIY method instead, which mimics a tree structure approach as seen in **llama index**.
   - This discussion points to broader trends in the AI landscape, with some attributing the surge in RAG tools to **VC funding** rather than genuine demand.
- **Creative Solutions for Aider Optimization**: To streamline workflows, a quick search tool using **ripgrep** was suggested for better integration with Aider, emphasizing the importance of speed in development.
   - Users also discussed using lower token counts in Aider's setting to enhance clarity and reduce confusion, particularly when dealing with extensive repositories.
- **Enhancements to Git and Chat Handling**: Aider’s repository mapping facilitates tracking code changes and interactions, though some configurations prompted users to turn off auto-refresh to maintain efficient search capabilities.
   - Integration of **HuggingFace models** and the use of **.env** files for managing environment settings enhance Aider's usability for AI pair programming.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Joint μ-Parameterization Guide with Cerebras**: Today, we're excited to drop a joint blog on [The Practitioner's Guide to the Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/), aiming to improve the accessibility of **μ-parameterization** (μP) for the training community.
   - This guide includes **step-by-step implementation instructions** and a simple implementation at [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup), addressing common accessibility issues found in the original materials.
- **Using Cosine Similarity with GPT-4**: A user is evaluating GPT-4 for a classification task without fine-tuning, considering dynamically selecting examples based on cosine similarity from a test set for improved in-context learning.
   - Concerns were raised about the potential for test set leakage by including similar test examples in the prompt, ensuring that the test question itself is not included.
- **Debate on Curriculum Learning Effectiveness**: There is ongoing discussion about the effectiveness of curriculum learning (CL) in AI, with skepticism about significant improvements over traditional training methods.
   - Members pointed out the absence of guaranteed best practices for filtering data, impacting the real-world application of CL.
- **MMLU_PRO sampling logic needs attention**: The `./leaderboard/mmlu_pro` task differs from its original implementation as it ignores question categories for few-shot sampling, as can be seen in [this code](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/47b9891aacb8bd7cda29d5c5ba17b9434dd333bc/evaluate_from_local.py#L228).
   - Another user suggested an updated sampling logic to improve accuracy based on question categories, available [here](https://github.com/rimashahbazyan/lm-evaluation-harness/blob/f117e6c09e32c553df0ab8cf8964a8b16636832e/lm_eval/api/samplers.py#L186).
- **Activation Functions Documentation Out of Sync**: A member pointed out that the available activation functions listed in the documentation do not reflect the full range present in the code, particularly with [Swiglu](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/activations.py).
   - Another member confirmed that the documentation had not been updated, referencing a [specific line in the code](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/neox_arguments/neox_args.py#L295) where these functions are defined.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **KTO Trainer Needs a Reference Model**: Members clarified that the **KTO trainer** requires a reference model to calculate rewards, suggesting using the untouched base model for comparison during fine-tuning.
   - *Pre-generating responses* from the reference model was suggested to save memory during training.
- **Qwen Model Bug Reports Surface**: **Users** noted unexpected behavior from the **Qwen 2.5 model** post-updates, particularly issues with prompt templates generating incorrect responses.
   - It was confirmed that the smaller model is sensitive to prompt formatting, which led to these problems.
- **RAG Implementation Catching Attention**: Participants discussed using **Retrieval-Augmented Generation (RAG)** to improve model responses and enhance knowledge retention during analysis.
   - One user suggested effectively using existing datasets in RAG to avoid knowledge loss during training.
- **SetFit v1.1.0 Out with Enhanced Training Capabilities**: The release of **SetFit v1.1.0** now employs the Sentence Transformers Trainer for efficient classifier training on both **CPU and GPU**, addressing previous issues.
   - Key updates include **MultiGPU support** and deprecating 'evaluation_strategy' in favor of 'eval_strategy', alongside new support for **Python 3.11** and **3.12**.
- **Training Classifiers Receives Structured Approach**: Training a **SetFit classifier model** involves two phases: finetuning a Sentence Transformer embedding model followed by mapping embeddings to classes.
   - This structured methodology enhances performance and efficiency, particularly with the features in version 1.1.0.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Subscription Woes**: Several users of **Perplexity** reported losing their **Pro** status intermittently, facing error messages like 'Query rate limit exceeded'. Temporary fixes like logging out and back in sparsely resolved the issue but highlighted system-wide lag issues post updates.
   - Concerns lingered over ongoing bugs which users fear could severely impact their experience on the platform.
- **AI Model Showdown: Llama vs. Perplexity**: Discussions revealed that **llama-3.1-sonar-large-128k-online** underperformed compared to the **Perplexity web app**, with users noting incomplete responses and inconsistent formatting. Suggestions to improve output were made, emphasizing capturing source references.
   - The discrepancy in performance has raised questions about model reliability in practical applications.
- **Chemistry of Chain of Thought Reasoning**: Members engaged with resources on **Chain of Thought reasoning**, aimed at boosting AI logic and reasoning skills. A guide detailing implementation was shared, enhancing the toolkit for developing complex AI models.
   - Further threads emphasized the ongoing application of this reasoning style in improving AI's functional abilities in real-world scenarios.
- **Frustration with Perplexity API Citations**: Users expressed disappointment regarding the **Perplexity API**'s erratic citation feature, often failing to deliver consistent references despite explicit requests. The criticisms pointed out how the API's reliability hinges heavily on accurate citation provision.
   - This inconsistency risks diminishing the API's reputation within the developer community focused on serious applications.
- **Potential Azure Deployment for OCR Services**: Curiosity emerged about the feasibility of deploying **Perplexity API** on **Azure** for OCR services, reflecting a growing interest in practical applications of APIs in cloud environments. This could open new avenues for integrating OCR capabilities using the API's features.
   - The volume of inquiries about Azure deployment indicates an evolving trend towards cloud-based AI solutions.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Team Coordination at Hackathon**: Participants set up collaboration strategies for the hackathon, recommending self-organization and communication via designated channels to optimize teamwork.
   - Members suggested using Uber for transport due to limited parking, emphasizing the importance of logistical planning for a successful event.
- **CUDA Mode Event Highlights**: The hackathon kicked off with positive feedback, showcasing notable projects and collaborative efforts, inspiring participants regarding future endeavors.
   - Ten teams were selected for pitches, with the judges focusing on commercial viability and innovation, reminding teams to finalize their submissions on time.
- **KLDivLoss and Kernel Issues**: Concerns over the **KLDivLoss** backward kernel prompted discussions regarding its formula accuracy and potential loop unrolling problems related to larger vocab sizes.
   - Participants suggested investigating the relationship between KLDivLoss and Cross-Entropy implementations to enhance model performance and reduce discrepancies.
- **WebGPU vs. MPS Performance**: Members noted that while **MPS** outperforms **WebGPU** on **macOS**, WebGPU is still in development and hasn't reached peak performance, indicating areas for improvement.
   - There’s a collaborative push to optimize kernel comparisons between MPS and WebGPU, with calls for community input on enhancing implementations.
- **Compute Credits and Support Needs**: Participants clarified how to claim **compute credits**, confirming that no confirmation emails are sent, but funds are credited shortly after sign-up.
   - Support for installing Python packages was confirmed successful across nodes, reflecting the community's resource-sharing mentality in problem-solving.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Facilitates Cloud-Based Testing**: Subscribers can now test **OpenRouter** services directly in the cloud without local installations; a smaller demo is available featuring a Loom video.
   - *This setup makes it easy for users to explore features quickly and efficiently.*
- **Webinar on Advanced OpenRouter Usage Incoming**: An upcoming live webinar is set for **12pm EST**, focusing on scaling to thousands of **parallel agents and proxies**.
   - *Find more details by checking the Live tab on the associated YouTube channel.*
- **Middle-Out Transform Disabled as Default**: **OpenRouter** has officially disabled the middle-out transform by default, which affects many users' workflows.
   - *This change has raised concerns, highlighting the importance of the feature for various frontend and backend systems.*
- **Speculations Rise Around New Anthropic Model Launch**: Rumors suggest an impending launch of a new model from **Anthropic**, with hints indicating an announcement during a Google event.
   - *This announcement may coincide with extensive free token offers, stirring discussion among developers.*
- **Exploration of Private LLM Servers**: A member raised questions about whether participants are running **private LLM servers** themselves or utilizing third-party services.
   - *The inquiry sparked engagement regarding the management and operation of these servers.*



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Music Production AI struggles with music theory**: Discussions revealed that large models in **music production** face challenges with **basic music theory tasks** like transposing chords, with experimentation ongoing using a feline AI to generate MIDI files.
   - Participants agreed that **music notation** remains a significant barrier due to limited training examples.
- **Bittensor raises ethics concerns**: Members voiced concerns regarding **Bittensor** seemingly replicating **Nous Research’s distributed training algorithm** without proper acknowledgment, calling into question ethical practices in AI.
   - The dialogue suggested that **innovation** in distributed training must be prioritized over simply increasing parameter counts.
- **New Medical LLMs on the scene**: Several new models have been introduced, including **HuatuoGPT-II** and **Apollo**, aimed at enhancing medical AI capabilities, particularly in gene-phenotype mapping and multilingual applications.
   - **HuatuoGPT-Vision** was also showcased for its multimodal processing strength, enhancing accessibility in medical data handling.
- **LLMs Transform Clinical Trials**: LLMs are being utilized to improve clinical trials, particularly seen with **AlpaPICO** which generates PICO frames, streamlining the process for clinical reporting.
   - These advancements aim to enhance the quality of **medical documentation** and improve workflows in clinical settings.
- **Exploring RL environments for reasoning**: There are ongoing discussions about creating specialized **RL environments** tailored for reasoning tasks, emphasizing the need for diverse setups similar to open source fine-tuning.
   - Members indicated that successful training depends heavily on the selection of quality datasets and environments.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AI's Role in Mental Health Support**: Members discussed that people with mental health issues may prefer talking to chatbots due to stigma, making ethical AI usage crucial in healthcare.
   - While AI can aid in mental health diagnostics, it must comply with **data privacy regulations** and not replace professional care.
- **Addressing Bias in AI Systems**: The group emphasized the importance of teaching motivated reasoning and confirmation bias to improve critical thinking in AI usage.
   - They agreed that AI recommendations should be grounded in **scientific advice** with strong ethical standards.
- **Cohere's Research Focus is Diverse**: Cohere works on various topics including language models, efficiency, safety, and AI policy, with resources available on their [research papers page](https://cohere.com/research/papers).
   - Members were encouraged to explore these topics as part of their ongoing professional development.
- **Embedding Call Parameter Update**: A user encountered errors with the embedding call stating '`embedding_types parameter is required`,' indicating a recent requirement change.
   - This prompted clarification from the **Cohere team**, as the documentation previously stated it was optional.
- **AI-Telegram-Chatbot Project Launch**: A member shared their [AI-Telegram-Chatbot](https://github.com/derssen/AI-Telegram-Chatbot) GitHub repository demonstrating **Cohere AI** in action.
   - The bot aims to enhance user interaction through **AI-driven responses**, reflecting broader interest in practical applications of Cohere technologies.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Last Call for Mojo Feedback**: Join a quick 30-minute call to share your thoughts about **Magic**; participants receive exclusive swag for input. You can book your slot [here](https://modul.ar/user-feedback).
   - *Engagement is vital* to improve Magic and gather a broader range of experiences from the community.
- **Mojo's Python Integration Woes**: Members debate the feasibility of integrating **Python libraries** into **Mojo**, expressing concerns over potential GIL conflicts impacting performance. They ponder whether creating direct Mojo files for Python classes could simplify usage.
   - The community remains cautious, highlighting that while integration is beneficial, it may affect Mojo's efficiency and objectives.
- **MAX Custom Ops Need Clarity**: A query on the status of **MAX custom ops** sparked concern regarding changes noted on the [modular documentation](https://docs.modular.com/max/api/mojo/register/register/op). Members are looking for updates on recent alterations or function removals.
   - Community members are eager for clearer documentation, expressing a pressing need for guidance on properly utilizing MAX operations.
- **Bit Packing and Structs in Mojo**: Discussion revolved around the absence of native **bit packing** in **Mojo**, with members considering alternatives like manual packing and variable width types to optimize struct sizes. Concerns regarding struct alignment's impact on performance surfaced during this conversation.
   - The potential for **LLVM** enhancements to manage varying bit widths was mentioned, indicating a route to address these efficiency issues.
- **Mojo Evolves Towards General Purpose**: Users express optimism about **Mojo** becoming a full-fledged **general-purpose language**, asserting its capability extends beyond mere AI applications. Integration with platforms like MAX is viewed as essential for broader usability.
   - This sentiment shows a collective eagerness to see Mojo evolve while keeping its performance snappy and competitive.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Models Hit Loading Snags**: Users face challenges loading models after updating to LM Studio, especially post the CUDA Llama.cpp v1.1.9 update, triggering various fixes such as clearing cache.
   - Many resorted to rolling back versions, sharing solutions that reinstated functionality amidst ongoing frustrations.
- **Image Generation Models Not Supported**: Discussions revealed that LM Studio does not support image generation models like **Flux**, resulting in 'unknown model architecture' errors.
   - Users clarified that these models are meant for other platforms, specifying clear usage boundaries for LM Studio.
- **DDR6 Release Timeline Uncertainty**: Concerns about the availability of **DDR6** surfaced, with users speculating that broad adoption might not happen until late next year.
   - Ongoing discussions reflect a waiting period for clear specifications before consumer hardware can adequately utilize this technology.
- **Mixed Results with RTX 4090 Performance**: Mixed performance metrics for **RTX 4090** emerged, with test results jumping from less than **20t/s** to disputed claims of **60t/s**.
   - Inconsistencies indicated challenges in setup and measurement in relation to different model configurations, raising questions about performance consistency.
- **ROCm Support Streamlined**: Users interested in ROCm support learned that the latest LM Studio version simplifies the process by auto-detecting ROCm installations.
   - This update is expected to facilitate easier installations for users relying on AMD GPU setups.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Exploring Stable Diffusion Features**: Users discussed various aspects of **Stable Diffusion**, including [Dalle3 functionality](https://x.com/LikeToasters/status/1836632745075736913) and limitations of **Flux** in terms of VRAM utilization.
   - The conversation highlighted specific tools, like boorutag autocompletion, aimed at enhancing prompts.
- **FLUX Model Utilization Faces VRAM Challenges**: Members shared experiences with **FLUX models**, detailing the challenges of using **LoRAs** and managing VRAM during image generation.
   - Techniques such as keeping text encoders on DRAM were suggested to optimize model performance.
- **Training LoRAs for Character Consistency**: Discussion focused on the need for precise prompts and training **LoRAs** to maintain consistent character generation in projects like comics.
   - Participants mentioned using IP adapters for improved character coherence during image creation.
- **Inpainting Techniques for Image Completion**: Users sought advice on **inpainting techniques** to effectively fill missing parts of images while preserving style and coherence.
   - Tools like **Fooocus** and **RuinedFooocus UI** were recommended to enhance the inpainting process.
- **Consistency in AI Art Generations**: Conversations revolved around ensuring consistency in **AI art** by using the same prompts and settings.
   - Maintaining consistent seeds and settings was emphasized, along with tools that aid in maintaining style across generated images.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **o1-mini flounders in creative writing**: **o1-mini** struggles with clichés and predictable structures in poetry, making it less suitable for creative depth compared to **Claude Opus 3**. Users agree that prompt specificity could enhance results.
   - *Improved prompting could potentially unlock better creativity*, but current performance limitations remain a setback.
- **Efficient embedding storage practices shared**: A member discussed efficient storage solutions for embeddings from a **12-13k text collection**, highlighting **S3** and OpenAI's vector store as key options. The goal is effective clustering and retrieval.
   - This conversation reflects ongoing interest in optimizing AI data management methodologies.
- **AI tools tackling PDF analysis**: A user requested tools that can analyze PDFs, including converting images to text for AI knowledge bases, with many RAG solutions noted for supporting PDF integration. Yet, there remains a gap in converting images accurately.
   - The community acknowledges the necessity of advancing multimodal models to handle such tasks more effectively.
- **Examining AI chatbot model performance**: Participating members compared AI chat models, emphasizing how **o1-mini** falls short against **Claude Opus 3** in creative writing tasks. The discussions highlighted the critical role of prompting in maximizing model output.
   - There's a strong interest in upcoming models promising improved performance in creative endeavors.
- **Insights on gpt-o1-preview quota for enterprises**: Discussion revealed speculation that the **gpt-o1-preview quota** for enterprise accounts may align with **tier 5 limits**, as cited in a [rate limits guide](https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-five).
   - *Members look for clearer documentation to unlock these enterprise features*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Device Development Confirmed**: Jony Ive confirmed the creation of an **OpenAI AI device**, with Sam Altman securing a distribution deal with Apple to potentially reshuffle the smartphone market.
   - The community reacted mixedly to rumored subscription models linked to this forthcoming device.
- **AI SDK 3.4 Enhances Tool Execution**: The release of **AI SDK 3.4** introduces automatic multi-step tool executions, facilitating backend developments in various programming languages.
   - Noteworthy applications utilizing the SDK include **postgres.new** for SQL translation and a versatile web development agent, **v0**.
- **Elicit.org Wins Accolades for Research**: **Elicit.org** earned praise among members for its capabilities in streamlining academic literature reviews, making research processes more efficient.
   - Users emphasized the importance of community recommendations in discovering relevant AI tools and developments.
- **Gorilla Leaderboard V3 Challenges LLMs**: The rollout of **BFCL V3** aims to evaluate how LLMs manage multi-turn workflows and function calling, critical for complex AI tasks.
   - This leaderboard addresses performance metrics crucial for real-world AI applications.
- **Anthropic Poised for Significant Funding**: Anthropic is engaging in discussions that could value the company between **$30 billion and $40 billion**, potentially doubling its previous valuation.
   - This funding maneuver occurs in a competitive AI market, reflecting substantial investor confidence.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **O1 model's reasoning leap**: Recent discussions unveiled that **O1's** improved reasoning capabilities saw a jump from **0% to 52.8%** on a challenging benchmark, hinting at potential synthetic data training.
   - This suggests significant advancements, possibly tied to utilizing effective training methodologies for complex tasks.
- **Anthropic aims for valuation boost**: News surfaced that **Anthropic** seeks to raise capital that could propel its valuation to **$30 billion to $40 billion**, potentially double its previous worth.
   - This reflects rising investor enthusiasm in the AI startup ecosystem amidst fierce competition.
- **Shampoo trains Gemini, sparks gatekeeping talks**: It was confirmed that **Shampoo** was utilized for training **Gemini**, which raised conversations about information gatekeeping within the community.
   - Despite the paper's availability, many expressed surprise at the implications of Shampoo's role in this context.
- **GameGen diffusion model makes a sudden exit**: Discussions focused on the rapid rise and unexpected disappearance of the **GameGen diffusion model** from GitHub, causing confusion among users.
   - This incident echoed concerns about 'rug pulls' within the AI game development space.
- **Twitter security woes escalate**: Numerous Twitter accounts have recently been hacked, leading to meme coin scams impacting high-profile users, as reported in a [community alert](https://x.com/zachxbt/status/1836473279479189916).
   - Questions were raised whether the security issues stemmed from **SIM swapping** or inherent vulnerabilities, especially when accounts with 2FA security still faced compromises.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Building RAG Applications with NVIDIA NIM**: A great tutorial on [NVIDIA NIM](https://t.co/zFC0DorIMW) guides users in creating a full-stack RAG application, connecting **Llama 3**, an **ArXiv dataset**, **Milvus** as the vector database, and **Gradio** for the app interface.
   - This project showcases effective integration of key components necessary for robust RAG functionalities.
- **Nudge Fine-Tuning Improves Embeddings**: [NUDGE](https://t.co/FT1C2x3Iov) offers a non-parametric method for embedding fine-tuning that accelerates the process from **hours to minutes**.
   - This innovation highlights a significant boost in operational efficiency for model finetuning.
- **Multimodal RAG Tackles Product Manuals**: Discussion centered on the construction of multimodal RAG systems to simplify the understanding of **complex product manuals**, like those for IKEA furniture assembly.
   - The approach signifies a need for intricate setups to efficiently index, search, and retrieve data, enhancing the user experience.
- **Cleanlab's TLM Enhances Trust**: An article discusses how **Cleanlab's TLM** improves **RAG systems** in **LlamaIndex**, focusing on enhancing AI output reliability in critical applications like law.
   - It emphasizes the importance of dependable AI systems that yield accurate responses, combating prevalent issues of incomplete and overconfident outputs.
- **Local Model Serving with LitServe**: [LitServe](https://t.co/Xikqk20peW) from **LightningAI** provides a framework to serve and scale LLM models using FastAPI, as shown in a demo with LlamaIndex.
   - This framework allows users to build efficient RAG servers and host them locally, improving operational workflows.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 2.5.0 Launches Quietly**: The **long-awaited DSPy 2.5.0** has been released, streamlining the migration process and deprecating all pre-2.4 LM clients, encouraging users to transition to supported providers through `dspy.LM(model_name, **kwargs)`.
   - Feedback is actively sought as users adapt to the new version, with documentation and support readily available to assist in the transition.
- **Chat Adapter Improvements Address Repetitive Responses**: Members discussed the need for custom chat adapters due to lower LLM models (<7B) producing repetitive responses in 'chat complete' mode, a solution now in testing.
   - This enhancement is aimed at improving user experience, and feedback from early adopters is crucial to fine-tuning the new architecture.
- **Synthetic Data Generation Speeds Surge**: A report highlighted impressive improvements in synthetic data generation speeds after fine-tuning a lower model, achieving **from 30 to 2500 tokens per second**.
   - This improvement positions DSPy as a promising tool for generating large volumes of synthetic training data efficiently.
- **TrueLaw Makes Waves with DSPy Insights**: In a recent episode of the [MLOps Podcast #260](https://youtu.be/O0F3RAWZNfM?si=ckG2DWkwop8zu-ZA), CTO of **TrueLaw Inc.**, Shiva Bhattacharjee, discussed leveraging **DSPy** for specialized domain problems.
   - The conversation underscored the importance of **domain-specific models** to enhance performance, particularly in the legal sector.
- **Text Classification Challenges and Inquiries**: A member raised questions about the possibility of extending docstrings for complex text classification tasks, seeking ways to improve LLM understanding.
   - There was also a request for available **Chain of Thought (COT)** methods with Groq, indicating active interest in expanding testing capabilities.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Curious Minds at the CUDA Hackathon**: One member inquired if anyone was attending the upcoming **CUDA Mode IRL hackathon**, prompting interest in gathering insights from the event.
   - *It could be a great opportunity to discuss latest developments* in GPU programming and optimization strategies.
- **Optimize CPU Offloading to Enhance Performance**: Concerns arose regarding the absence of **CPU offloading** in the optimizer, particularly seen in the [full_finetune_single_device.py](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py), hinting at potential performance degradation due to legacy issues.
   - Members suggested adopting *PagedAdam* by default for improved **memory efficiency** and highlighted the ongoing transition to more optimized approaches.
- **KV Caching Under Fire**: Discussions centered around experiencing **OOM issues** with the *qwen2.5 1.5B model* when using KV caching and batch sizes of 8 on 40GB machines.
   - Members proposed troubleshooting by examining the KV cache shape to determine if it’s initialized properly to maximum length, aiming to mitigate issues.
- **Batch Size Quandaries in Model Evaluation**: A debate emerged about the impact of increasing **batch sizes** on model evaluation, particularly during multi-task scenarios.
   - Participants leaned toward analyzing trade-offs related to cache initialization and the interaction of **weights and gradients** between CPU and GPU.
- **Evaluation Recipe Bug Fix Adventures**: Key discussions highlighted a PR addressing bugs in the evaluation recipe for group tasks, indicated by the need for timely patches as changes are implemented, seen at [PR #1642](https://github.com/pytorch/torchtune/pull/1642).
   - There was general agreement on tackling identified fixes promptly while awaiting the most recent updates to the **evaluation recipe**.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **CLIP Retrieval Alternatives Lacking**: Members discussed the scarcity of alternatives to [CLIP Retrieval](https://rom1504.github.io/clip-retrieval/), noting it may not be revived by rom1504.
   - One user expressed the need for a backend solution compatible with **LAION 400M** for their research projects.
- **AI Internship Leads Wanted**: A user requested suggestions on where to find AI internship opportunities, emphasizing community guidance.
   - This inquiry reflects a growing interest in advancing careers within the AI field.
- **Dataset Sharing for Model Training**: A dataset was uploaded to Hugging Face for training **Llama-3.1**, with a call for feedback on its coding effectiveness.
   - The shared dataset includes detailed application descriptions, sparking discussion on best practices.
- **Summarizer AI in Need of Feedback**: A user shared their newly developed [summarizer AI](https://www.fluxa.pro) and sought community testing and feedback.
   - Acknowledgment of its potential was met with suggestions for message length customization to improve usability.
- **Playlist Generator Project Introduced**: A user showcased [Adify](https://adify.pro), a playlist generator that creates Spotify playlists based on user prompts.
   - The project garnered positive reception, indicating a strong interest in innovative music generation tools.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **VGA Reclaims GPU Connection Glory**: A user confirmed that their **GPU** connected via **VGA only**, overcoming problems related to an incorrect displayed password.
   - This work-around allowed them to power their setup successfully using an older VGA connection.
- **ShapeTracker Mergeability Bounty Inquiry**: There's a query regarding the bounty status for **ShapeTracker** mergeability in **Lean**, with an interest expressed for an **undergraduate thesis**.
   - The unresolved status has piqued the curiosity of students eager to explore this complex topic.
- **Answer AI Talks Cost Efficiency**: Discussions revolved around the cost-effectiveness of **Answer AI** boxes, which might offer better pricing than current solutions, including potential bulk discounts.
   - Participants hope to showcase benchmarks from this affordable setup, aiming to prove its financial viability.
- **Tinygrad's Cloud Integration Concept Flourishes**: The **CLOUD=1** option for integration into tinygrad garnered attention, aiming to streamline functionality without relying on AWS-style virtualization.
   - Members discussed how this device option would enhance usability while keeping performance intact.
- **Metal Tutorials Offer Insights**: A [GitHub link](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20240921_metal.md) to a tutorial on **Metal** was shared, expanding knowledge on tinygrad integration.
   - The tutorial serves as a resource for contributors keen on improving their Metal-related skills within tinygrad.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Agents face issues with Local AI integration**: Users reported that **Agents do not work with local AI** after a six-month gap, suggesting **Ollama** as a better alternative.
   - This showcases the ongoing search for compatible local AI solutions in a dynamic development environment.
- **Debate on Best Vector Store Options**: Discussion heated up about whether **Hugging**, **OpenAI**, or **Ollama** is the best vector store for their projects.
   - Choosing the right vector store could critically affect both **performance** and **scalability**.
- **Optimizing PDF processing in chatbot project**: A user sought ways to efficiently split and store PDF content in their vector database without a redundant intermediate step.
   - This improvement would streamline workflows, enhancing overall processing performance.
- **Challenges with Text Generation Inference Parameters**: A query arose regarding the unexpected appearance of the **<|end|>** token in outputs, despite setting `return_full_text` to false.
   - This points to a need for improved clarity around inference parameters for better user control.
- **Portfolio Chatbot Helps Users with Queries**: A user launched a chatbot assistant for their portfolio, facilitating answers to client inquiries about their services.
   - They welcome community feedback to refine this tool further, signaling a collaborative spirit in development.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Module for Bedside Table**: A member raised the idea of creating an Open Interpreter module for the [Kequel Modular Customizable Bedside Table](https://www.kickstarter.com/projects/kequel/kequel-modular-customizable-bedside-table), inquiring about group interest in collaboration.
   - This initiative aims to enhance smart home technology integration, inviting fellow developers to contribute ideas and development.
- **User Interface Challenges with Open Interpreter**: Concerns were raised about screen visibility when using command line inputs, prompting a proposal for solutions to enhance visual clarity.
   - Members discussed potential workarounds to improve user experience while the Open Interpreter processes external inputs.
- **LiveKit Blocks Cleartext Connections on Android**: A user noted that newer Android phones block the **01 mobile app** from connecting to a local **LiveKit** server over **HTTP**, indicating 'CLEARTEXT communication not permitted'.
   - They suggested using ngrok for an HTTPS endpoint which effectively resolves connection issues for users who expose their servers.
- **GitHub Solutions for Cleartext Communication**: A GitHub issue detailed a proposal to **enable cleartext communication** strictly for local networks, ensuring user notifications regarding security.
   - This addresses connection challenges while balancing network security for developers interacting with local devices.
- **Investigating Backend Request Loops**: A member questioned the frequent backend requests sent by Open Interpreter, suspecting an infinite loop scenario.
   - Clarification on backend response expectations was sought to help determine accurate request conclusions.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Qwen 2.5 wins praise over Llama 3.1**: A member noted strong positive feedback for **Qwen 2.5**, revealing it marginally outperforms **Llama 3.1** in benchmarks, as highlighted in a [Reddit comparison](https://www.reddit.com/r/LocalLLaMA/s/NiCbaTyodk).
   - This raised community awareness around the importance of verified performance metrics in the latest model comparisons.
- **Long context challenges in Axolotl**: Discussion arose around **Axolotl's** capabilities in handling conversations longer than **max_seq_len** in ShareGPT, reflecting the community's interest in context management.
   - Clarity on these training intricacies remains a hot topic as members dive into model training protocols.
- **Rope Scaling Debate for Llama 3.1**: A member questioned the necessity of **rope_scaling** when training **Llama 3.1 8B** on long context CoT traces of approximately **120K tokens** while facing memory issues at **sequence_len** beyond **40K**.
   - Despite using multiple GPUs with deepspeed zero3, the complexity of handling long contexts continues to spark discussion among engineers.
- **Fine-tuning spikes inquiry**: Users reported unexpected spikes during fine-tuning on a **100K row dataset**, prompting a quest for correlations with specific data points.
   - Efforts to enable more extensive logging proved insufficient, leaving fine-tuning mechanics under scrutiny.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Sentx.ai Ventures into Consciousness Development**: Sentx.ai is pioneering work in **consciousness development**, still at its early stages. They are actively seeking **general opinions** particularly regarding their alignment approach.
   - Members are encouraged to assess the pragmatic impacts of consciousness development on future AI alignment.
- **Self-Adjustment for AI Alignment Proposed**: Sentx.ai introduces a strategy for models to **self-adjust their alignment** to **human values**, avoiding hard caps. This approach aims to cultivate **ongoing dialogue** around effective alignment practices.
   - Community members are discussing the implications of self-adjusting models in real-world scenarios and their potential benefits.
- **Call for Collaboration on Alignment Projects**: An open invitation was extended for sharing information about **similar projects** to promote collaboration on alignment development. Members are encouraged to exchange insights and connect privately.
   - This collaborative spirit aims to enhance collective contributions toward more effective AI alignment strategies.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **SQLite Full-Text Search Enhanced**: A new meetup will explore combining **SQLite’s builtin full-text search engine** with [sqlite-vec](https://discord.com/events/1089876418936180786/1284180345553551431) for improved efficacy.
   - This session promises to deliver more **complete and accurate search** results, catering to developers looking for effective search capabilities.
- **Mozilla Launches AI Builders Accelerator**: Mozilla's inaugural **AI Builders Accelerator cohort** has been announced and will kick off shortly.
   - Program specifics can be found [here](https://discord.com/channels/1089876418936180786/1245083732319408195/1287802832417718325), supporting cutting-edge AI projects.
- **SoraSNS: A New Fediverse Client**: An ex-Apple Engineer unveiled **SoraSNS**, a Fediverse client integrating [local AI](https://discord.com/events/1089876418936180786/1277835047084363827) to learn about user interests.
   - This client aims to enhance user experience by providing an adaptive **'For You' timeline**.
- **Open Source AI to Address Challenges**: Mark Surman discusses the potential of **defining Open Source AI** to tackle various challenges in the field, as highlighted in The New Stack.
   - The conversation stresses how such definitions can assist in [solving a million headaches](https://discord.com/channels/1089876418936180786/1287810294126481498) for developers and organizations.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCL V3 Revamps LLM Evaluation**: The **Berkeley Function-Calling Leaderboard (BFCL) V3** introduces a fresh evaluation method for assessing **multi-turn** function calling, enhancing agentic system capabilities.
   - This version allows models to manage complex interactions crucial for LLMs during intricate tasks.
- **State Management is a Must**: **State Management** in LLMs is vital, enabling systems to validate task outcomes like checking if a **stock purchase** was successful.
   - This highlights how internal state queries through APIs are key post-task execution.
- **Goodbye Short Context Models**: With the launch of BFCL V3, reliance on **short context models** is discouraged, as tasks require more extensive context to be effective.
   - This is especially critical for complex tasks, such as sorting through **hundreds of files**.
- **Leaderboards Set New Standards**: BFCL V3 establishes a **gold standard** for evaluating LLM functionality, particularly in function invocation, driven by community insights.
   - This reflects ongoing collaborations with **enterprises** and **open-source contributors** to refine evaluation practices.
- **Deep Dive into BFCL V3 Performance**: A new blog post details the BFCL V3 evaluation method, discussing how models are assessed on **cost** and **latency** in real-world applications.
   - For more insights, check the full post at [Berkeley Function Calling Blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html).



---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1286764037429923941)** (603 messages🔥🔥🔥): 

> - `HuggingFace Spaces Downtime`
> - `Model Fine-Tuning`
> - `AI Tools and Libraries`
> - `Serverless API Usage`
> - `ExtractCode Voting Support` 


- **HuggingFace Spaces experiencing downtime**: Users reported issues with HuggingFace Spaces being down, facing errors such as '500 Internal Error' and problems with uploading files.
   - The downtime lasted for several hours, causing frustration among users trying to access models or upload content.
- **Guidance on Model Fine-Tuning**: A user sought assistance to fine-tune a model for responding strictly from a dataset of 350 records focused on operating system, software, and hardware issues.
   - Others contributed by sharing resources like YouTube videos and suggested tools like SimpleTuner for training the models.
- **Exploring AI Tools and Libraries**: Various users discussed tools for fine-tuning models, with recommendations including SimpleTuner, Kohya-Trainer, and Onetrainer for ease of use.
   - Discussion highlighted user experiences and challenges faced while working with these libraries, promoting collaborative learning.
- **Serverless API Insights**: The Serverless Inference API from HuggingFace was discussed, with users noting its free access for certain API requests to test and explore models.
   - Users were encouraged to try it for ease of integration and rapid prototyping without needing to manage infrastructure.
- **Voting Support for AI Project**: A user presented their AI project, ExtractCode, which aims to extract programming code from YouTube videos and requested support through voting.
   - Participants were encouraged to click the link provided for support, indicating a community-driven approach to project promotion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/help/how-to-ask">How do I ask a good question? - Help Center</a>: Stack Overflow | The World&#x2019;s Largest Online Community for Developers</li><li><a href="https://arxiv.org/abs/2409.12517">Scaling FP8 training to trillion-token LLMs</a>: We train, for the first time, large language models using FP8 precision on datasets up to 2 trillion tokens -- a 20-fold increase over previous limits. Through these extended training runs, we uncover...</li><li><a href="https://huggingface.co/spaces/webml-community/remove-background-webgpu">Remove Background WebGPU - a Hugging Face Space by webml-community</a>: no description found</li><li><a href="https://huggingface.co/blog/dpo-trl">Fine-tune Llama 2 with DPO</a>: no description found</li><li><a href="https://huggingface.co/spaces/r3gm/Audio_separator">Audio🔹Separator - a Hugging Face Space by r3gm</a>: no description found</li><li><a href="https://tenor.com/view/ln_strike-gregzaj1-quant-quantitative-gif-22567558">Ln_strike Gregzaj1 GIF - Ln_strike Gregzaj1 Quant - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/crab-gif-26300412">Crab GIF - Crab - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/Suniilkumaar/AudioSep">AudioSep - a Hugging Face Space by Suniilkumaar</a>: no description found</li><li><a href="https://tenor.com/view/no-sleep-staying-up-insomnia-coffee-weak-gif-21941823">No Sleep Staying Up GIF - No Sleep Staying Up Insomnia - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/caos-bob-esponja-crisis-patricio-gif-23199341">Caos Bob GIF - Caos Bob Esponja - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/beaker-muppets-calm-relax-panic-gif-16222881">Beaker Muppets GIF - Beaker Muppets Calm - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/AIatMeta/status/1836806535969968354?t=TpKrg6iwozJeumc8Kv-5ZQ&s=19">Tweet from AI at Meta (@AIatMeta)</a>: Fragmented regulation means the EU risks missing out on the rapid innovation happening in open source and multimodal AI. We&#39;re joining representatives from 25+ European companies, researchers and ...</li><li><a href="https://tenor.com/view/burntdasbrot-kikalounge-burnt-toast-dance-gif-12556330">Burntdasbrot Kikalounge GIF - Burntdasbrot Kikalounge Burnt - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/rowancheung/status/1838280020642676802">Tweet from Rowan Cheung (@rowancheung)</a>: I just finished up an exclusive interview going over a new, major AI model upgrade.  Can confirm, tomorrow will be a big day for developers.  Dropping the full conversation on X the second the embargo...</li><li><a href="https://tenor.com/view/shame-on-you-gif-25797108">Shame On You GIF - Shame On You - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/anime-head-pat-pat-very-good-good-girl-gif-17187002">Anime Head Pat GIF - Anime Head Pat Pat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ai.google.dev/competition/projects/extractcode">no title found</a>: no description found</li><li><a href="https://www.nist.gov/news-events/events/2024/09/unleashing-ai-innovation-enabling-trust">Unleashing AI Innovation, Enabling Trust</a>: A symposium to discuss recent progress and next steps in AI measurement and standards</li><li><a href="https://www.anandtech.com/show/21425/intel-lunar-lake-architecture-deep-dive-lion-cove-xe2-and-npu4/4">Intel Unveils Lunar Lake Architecture: New P and E cores, Xe2-LPG Graphics, New NPU 4 Brings More AI Performance</a>: no description found</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">Doubt Press X GIF - Doubt Press X La Noire - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/baby-dont-hurt-me-mike-ohearn-jokester-joke-funny-gif-27699537">Baby Dont Hurt Me Mike Ohearn GIF - Baby Dont Hurt Me Mike Ohearn Jokester - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://en.m.wikipedia.org/wiki/Sigmoid_function">Sigmoid function - Wikipedia</a>: no description found</li><li><a href="https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf">flux1-dev-Q4_K_S.gguf · city96/FLUX.1-dev-gguf at main</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Eiffel_Tower_replicas_and_derivatives">Eiffel Tower replicas and derivatives - Wikipedia</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=bo49U3iC7qY&ab_channel=TrelisResearch">Fine-tuning on Wikipedia Datasets</a>: ➡️ Get Life-time Access to the Complete Scripts (and future improvements): https://Trelis.com/ADVANCED-fine-tuning/➡️ One-click fine-tuning and LLM templates...</li><li><a href="https://huggingface.co/models?other=simpletuner&sort=created">Models - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>: no description found</li><li><a href="https://status.huggingface.co/">
Hugging Face status
</a>: no description found</li><li><a href="https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX.md">SimpleTuner/documentation/quickstart/FLUX.md at main · bghira/SimpleTuner</a>: A general fine-tuning kit geared toward diffusion models. - bghira/SimpleTuner</li><li><a href="https://github.com/ostris/ai-toolkit">GitHub - ostris/ai-toolkit: Various AI scripts. Mostly Stable Diffusion stuff.</a>: Various AI scripts. Mostly Stable Diffusion stuff. - ostris/ai-toolkit</li><li><a href="https://api-inference.huggingface.co">Serverless Inference API</a>: no description found</li><li><a href="https://github.com/marijnwijbenga/ai-music-learning-assistant-llm/tree/develop">GitHub - marijnwijbenga/ai-music-learning-assistant-llm at develop</a>: An AIlearning assistant LLM chatbot restricted to music topics, finetuned on music theory and music teachings - GitHub - marijnwijbenga/ai-music-learning-assistant-llm at develop
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1286872453838147594)** (8 messages🔥): 

> - `Centroidal Triplet Loss`
> - `Mamba-2 Architecture`
> - `BFGS Algorithm`
> - `Langchain Integration`
> - `Mixed Precision Losses` 


- **Centroidal Triplet Loss already exists**: A member discovered that their 'novel' idea, **Centroidal Triplet Loss**, has already been developed as **Centroid Triplet Loss**.
   - They also noted a nearly identical diagram and are exploring some modifications that could enhance the concept.
- **Mamba-2 surpasses its predecessor**: Researchers introduced [Mamba-2](https://vidrihmarko.medium.com/mamba-2-is-out-can-it-replace-transformers-6cfb3372ea39), a state space model that outperforms **Mamba-1** and **Transformer++**.
   - It's designed for better handling of information-dense data, with a core innovation called **Structured State Space Duality (SSD)**.
- **Exploring the BFGS algorithm**: A member is currently researching the **BFGS algorithm** and its limited memory variant for a side project.
   - They welcomed input from others who have experience with these algorithms to enhance their understanding.
- **Langchain connects LLMs to data sources**: Another member shared their excitement about learning how **Langchain** integrates LLMs with databases and APIs for data retrieval.
   - They expressed hope that their understanding of Langchain's capabilities was correct and highlighted its potential usefulness.
- **1b FP8 matches bfloat16 precision**: A member indicated that **1b FP8** achieves loss matching that of **bfloat16 mixed precision** exactly.
   - This insight could imply significant implications for model training efficiency and performance.



**Link mentioned**: <a href="https://vidrihmarko.medium.com/mamba-2-is-out-can-it-replace-transformers-6cfb3372ea39">Mamba-2 is Out: Can it replace Transformers?</a>: Mamba-2: A new state space model architecture that outperforms Mamba and Transformer++

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1286766780190162964)** (11 messages🔥): 

> - `3D Content Generation`
> - `Medical AI Research Insights`
> - `Open-Source AI Trends`
> - `Residual Networks`
> - `Taostats and Decentralized AI` 


- **3D Content Generation in 10 seconds**: A member shared a GitHub repo, [threestudio](https://github.com/threestudio-project/threestudio), claiming it can generate 3D objects within 10 seconds, with a request for anyone to try this out.
   - Another member suggested using 'stable fast 3D' as an alternative, which can generate objects from images in less than one second, and noted its availability in HF space.
- **Medical AI Research Highlights**: A recap highlighted critical papers and models in Medical AI for the week, including a focus on a significant paper titled 'How to Build the Virtual Cell with Artificial Intelligence'.
   - Other key topics discussed included various medical LLMs and frameworks aimed at enhancing diagnostics and clinical trials using AI technologies.
- **Growing Adoption of Open-Source AI**: An article emphasized the rapid acceptance of open-source AI among developers, with a notable increase in usage reported in the '2023 State of Open Source' report.
   - The article lists 10 popular open-source AI frameworks and discusses the impact of significant tech investments driving this trend.
- **Nostalgia for Residual Networks**: A member shared the landmark paper on residual networks, citing its impact on training deeper neural networks more effectively.
   - The paper presented empirical evidence of achieving top performance on ImageNet, establishing residual networks as a significant advancement in deep learning.
- **Taostats: Decentralized AI Analytics**: Taostats emerged as a block explorer and analytics platform for Bittensor, aimed at facilitating decentralized analytics for machine learning.
   - The platform offers a variety of tools, including APIs and user-friendly features, supporting the growth of decentralized AI applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1512.03385">Deep Residual Learning for Image Recognition</a>: Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly re...</li><li><a href="https://huggingface.co/Chunte/flux-lora-Huggieverse">Chunte/flux-lora-Huggieverse · Hugging Face</a>: no description found</li><li><a href="https://x.com/OpenlifesciAI/status/1837688406014300514">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last Week in Medical AI: Top Research Papers/Models 🏅(September 14  - September 21, 2024)  🏅 Medical AI Paper of the week How to Build the Virtual Cell with Artificial Intelligence: Priorities and O...</li><li><a href="https://huggingface.co/posts/aaditya/492719281207772">@aaditya on Hugging Face: &quot;Last Week in Medical AI: Top Research Papers/Models
🏅(September 14 -…&quot;</a>: no description found</li><li><a href="https://www.digitalocean.com/resources/articles/open-source-ai-platforms">10 open source AI platforms for innovation | DigitalOcean</a>: Learn about 10 open source AI platforms for innovation and collaboration to scale your business</li><li><a href="https://fxtwitter.com/jsuarez5341/status/1830697672019476600">Tweet from Joseph Suarez (e/🐡) (@jsuarez5341)</a>: The Full RL Iceberg - everything wrong with reinforcement learning and how PufferLib is fixing it  Join me for a dive through 10 layers of the RL stack. There&#39;s something here for beginners and wo...</li><li><a href="https://taostats.io/">Taostats · Bittensor Network Block Explorer, Data Analytics, API and Node Support</a>: Explore the official Bittensor blockchain explorer at taostats.io, your trusted source for metagraph analytics, TAO token data, and personalized dashboards. Access APIs, RPC services, and more.</li><li><a href="https://github.com/threestudio-project/threestudio/tree/main">GitHub - threestudio-project/threestudio: A unified framework for 3D content generation.</a>: A unified framework for 3D content generation. Contribute to threestudio-project/threestudio development by creating an account on GitHub.</li><li><a href="https://www.futuretools.io/">Future Tools - Find The Exact AI Tool For Your Needs</a>: FutureTools Collects &amp; Organizes All The Best AI Tools So YOU Too Can Become Superhuman!
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1286916757084180531)** (163 messages🔥🔥): 

> - `OpenMusic Launch`
> - `Game Development with Bevy`
> - `Unity and Unreal Licensing Debate`
> - `AI-Powered RPG`
> - `Podcast Generation Technology` 


- **OpenMusic is Live!**: OpenMusic for text-to-music generation is now available on Hugging Face Spaces, allowing real-time music creation using a text description.
   - This project utilizes the innovative QA-MDT paper, which enhances audio quality and musicality.
- **Development of AI-Powered RPG**: A developer is creating an RPG game with AI agents that simulate short and long-term memory, along with physics integration and networking functionalities.
   - They expressed a desire for contributions and noted the challenges inherent in building such a complex system.
- **Debating Unity and Unreal Licensing**: The discussion highlighted the proprietary nature of Unity and Unreal Engine due to their licensing structures, despite some open-source components.
   - Participants debated the implications of software licensing, emphasizing the distinctions between proprietary, open-source, and various licensing models for game engines.
- **Podcast Generation Technology**: PodcastGen utilizes advanced techniques for generating podcasts inspired by Google's NotebookLM feature, capturing attention for its innovative approach.
   - Users expressed excitement over the capabilities, although some noted potential issues with repeated content in generated outputs.
- **Interfacing Rust with LLMs**: A conversation addressed the integration of large language models within the Rust-based game development framework Bevy, focusing on networking and entity interactions.
   - Participants offered suggestions for managing NPC tasks and communication between the game and LLM processes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jadechoghari/openmusic">jadechoghari/openmusic · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/krchickering/pokemon_generator">Pokémon Sprite Generator - a Hugging Face Space by krchickering</a>: no description found</li><li><a href="https://huggingface.co/spaces/jadechoghari/OpenMusic">OpenMusic - a Hugging Face Space by jadechoghari</a>: no description found</li><li><a href="https://huggingface.co/spaces/saq1b/podcastgen">PodcastGen - a Hugging Face Space by saq1b</a>: no description found</li><li><a href="https://huggingface.co/spaces/nroggendorff/flux-lora-tester">FlUX.1 LoRA - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://huggingface.co/blog/keras-nlp-integration">Announcing New Hugging Face and Keras NLP integration</a>: no description found</li><li><a href="https://huggingface.co/spaces/JoPmt/Flux-schnell_CPU_Stable_Diffusion_cpp">Flux-schnell CPU Stable Diffusion Cpp - a Hugging Face Space by JoPmt</a>: no description found</li><li><a href="https://github.com/Unity-Technologies/ml-agents/blob/develop/LICENSE.md">ml-agents/LICENSE.md at develop · Unity-Technologies/ml-agents</a>: The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source project that enables games and simulations to serve as environments for training intelligent agents using deep reinforcement ...</li><li><a href="https://www.kaggle.com/code/anhoangvo/run-comfy-gui-with-localtunnel-on-kaggle">Easy Run ComfyUI with GUI on Kaggle</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://www.kaggle.com/code/anhoangvo/generate-images-for-stories-using-llm-and-comfyui">Generate Images for stories using LLM and ComfyUI</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://youtu.be/5eo8nz_niiM">Over 200,000 Servers in One Place! Visiting Hetzner in Falkenstein (Germany)</a>: More Info about Hetzner:https://derbauer.hetzner.com/en/image-211013/---------------------------------------------------------Support me on Patreon: https://...</li><li><a href="https://youtu.be/_saL1lounEE>,">I Installed my OWN Cloud Server! See What Happened Next...</a>: Do you stay up at night wondering what server runs your cloud instances or how does a &quot;Bare Metal Cloud&quot; even work? We took a brand new Supermicro 4th Gen In...</li><li><a href="https://github.com/Unity-Technologies/UnityCsReference">GitHub - Unity-Technologies/UnityCsReference: Unity C# reference source code.</a>: Unity C# reference source code. Contribute to Unity-Technologies/UnityCsReference development by creating an account on GitHub.</li><li><a href="https://github.com/slangerosuna/space_cowboy_rpg">GitHub - slangerosuna/space_cowboy_rpg: A sci-fantasy open-world shooter/rpg that replaces scripted dialogue with generative AI and has infinite content</a>: A sci-fantasy open-world shooter/rpg that replaces scripted dialogue with generative AI and has infinite content - slangerosuna/space_cowboy_rpg</li><li><a href="https://www.hetzner.com/de/dedicated-rootserver/matrix-gpu/">Dedicated Server Hosting</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1287017384137461801)** (29 messages🔥): 

> - `GUI Element Detection`
> - `GUI Automation Software`
> - `AI for Interface Recognition`
> - `Uia and Android Accessibility`
> - `DOM Element Retrieval` 


- **Challenges in Detecting GUI Elements**: A member expressed interest in detecting GUI elements from screenshots to create a **GUI automation software**, aiming to identify interactive elements and their bounding boxes.
   - Another member questioned the feasibility of achieving **generic detection** across all interfaces due to overlapping elements and the challenges that arise with varying designs.
- **Discussion on Interface Detection Complexity**: Contributors discussed the complexity of designing a solution that works for **all interfaces**, pointing out the issues with interfaces lacking clear **buttons** or visual cues.
   - They noted that while AI could play a role, it might require **advanced techniques** and tailored models to achieve effective results.
- **Reference to Historical Automation Tools**: A member reminisced about the early days of automation tools used in **poker machines**, highlighting how people became creative in finding solutions for automation when money was involved.
   - This discussion illustrated the potential for **innovative approaches** when high stakes are involved, sparking a conversation on the creativity in problem-solving.
- **Paper Reference for GUI Detection**: One member mentioned seeing a paper that proposed a method for GUI detection, contrasting **modern** and **traditional approaches**, but faced difficulties with the corresponding GitHub repository.
   - This reflects the ongoing exploration in the field, emphasizing the importance of accessible resources for implementation.
- **Alternative Methods for GUI Interaction**: The original poster shifted to a simpler approach, opting to use **UIA** for Windows, **Android accessibility**, and **DOM element retrieval** with a headless browser for web applications.
   - This approach was acknowledged as **solid**, indicating a move towards leveraging existing frameworks over complex AI solutions.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1286773018420318228)** (7 messages): 

> - `Wild Text Content in ST Embedding`
> - `Updated Mamba Benchmarks`
> - `LLM Integration with WhatsApp`
> - `HuggingFace Hub Issues` 


- **Wild Text Content Challenges in ST Embedding**: A member highlighted the presence of wild text content such as 'll', 'lrgt', and 'dw' lacking spaces, raising concerns about how such cases are treated in a **ST embedding pipeline**.
   - They questioned the treatment of sequences like 'yes!do it' and noted the absence of embedding models capable of handling them effectively.
- **Inquiries on Updated Mamba Benchmarks**: Members inquired if there are any **updated Mamba benchmarks** available since the last report mentioned lack of weights.
   - The latest mentioned benchmarks suggested improvement, but members expressed doubts due to insufficient data.
- **Searching for Python LLM Integration with WhatsApp**: A member sought recommendations for any project repository that integrates an **LLM with WhatsApp**, emphasizing a Python solution.
   - Previous attempts with WPPConnect and CrewAI were reported as unsuccessful, specifically looking for a fully Python-based approach.
- **Concerns about HuggingFace Hub Performance**: A member reported issues with the **HuggingFace hub**, indicating potential downtime or malfunctions.
   - No further details were provided on the type or extent of the issues being faced.



**Link mentioned**: <a href="https://tenor.com/view/office-space-tps-gif-22666507">Office Space GIF - Office Space TPS - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1286788524527321151)** (7 messages): 

> - `Diffusion Models Discussion`
> - `Image Generator App with Flux.1-dev`
> - `ControlNet_Union Techniques` 


- **Correct Channel for Diffusion Discussions**: A member clarified that this channel is designated for discussing topics related to the [Diffusion Models Course](https://github.com/huggingface/diffusion-models-class) and not for LLMs.
   - There are occasional mix-ups, but participants are encouraged to focus on diffuser topics specifically.
- **Building an Image Generator App with Flux.1-dev**: Another member sought guidance on creating an image generator app using the latest **Flux.1-dev** model, mentioning their need for clarity amidst many tools.
   - A response suggested using **diffusers** with **FastAPI** and **React** for a customized hosting solution.
- **ControlNet_Union's Strict Output**: A member shared concerns about **ControlNet_Union** for **SDXL**, citing issues with the model retaining empty spaces instead of producing cohesive backgrounds from scribble inputs.
   - It was advised to focus on the **control_type** used, noting that HED allows more flexibility with black regions representing empty space.
- **Simplifying Cohesion in ControlNet Outputs**: For better background generation, modifications to the input images were suggested, such as erasing parts of the image directly.
   - This technique is encouraged for managing fill/inpaint/outpaint areas effectively.



**Link mentioned**: <a href="https://github.com/huggingface/diffusion-models-class">GitHub - huggingface/diffusion-models-class: Materials for the Hugging Face Diffusion Models Course</a>: Materials for the Hugging Face Diffusion Models Course - huggingface/diffusion-models-class

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1287889405935222856)** (1 messages): 

> - `Gradio 5 Beta Release`
> - `Performance Improvements`
> - `Modern Design Updates`
> - `AI Playground Feature`
> - `Security Enhancements` 


- **Gradio 5 Beta is here!**: We're excited to announce that **Gradio 5 (Beta)** is officially released, aiming to address frequent developer concerns.
   - This release introduces various features along with significant performance upgrades and modern design improvements.
- **Major Performance Improvements**: **Gradio 5** includes major performance enhancements, particularly with server-side rendering (SSR), resulting in much faster loading times for Gradio apps.
   - Developers can expect a more seamless experience in the browser, addressing previous **loading speed** complaints.
- **Revamped Design for Modern Appeal**: In response to feedback, many UI components like **Buttons** and **Sliders** in **Gradio 5** have received a modern design refresh.
   - The team invites feedback from users before the final public release of Gradio 5.
- **Introducing AI Playground for Experimenting**: Gradio 5 introduces an experimental **AI Playground** enabling users to generate and preview Gradio apps directly in their browser: [Playground link](https://5-0-dev.gradio-website.pages.dev/playground).
   - This feature encompasses a variety of app templates such as **Sentence Builder** and **Stock Forecast** for users to explore.
- **Enhanced Security Measures with Gradio 5**: The release ensures improved **security** by undergoing a third-party audit to prepare Gradio for production use.
   - Streaming media capabilities have also been enhanced, making it easier to create **realtime Gradio apps**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://5-0-dev.gradio-website.pages.dev/playground">Gradio Playground</a>: Play Around with Gradio Demos</li><li><a href="https://huggingface2.notion.site/Gradio-5-A-Production-Ready-Web-Framework-for-ML-Applications-a4d7e42c26f4450aa0758d968019d120?pvs=74)">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
</li>
</ul>

</div>
  

---



### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1287150108009431140)** (1 messages): 

> - `Aider v0.57.0`
> - `OpenAI o1 models support`
> - `Windows compatibility`
> - `New Cohere models`
> - `Bug fixes` 


- **Aider v0.57.0 Launches with New Features**: The release of **Aider v0.57.0** introduces support for **OpenAI o1 models**, enhancing performance with diff edit formats and SOTA leaderboard results.
   - Notably, **Aider** itself coded **70%** of this release, showcasing its self-sufficiency.
- **Improved Windows Compatibility**: On **Windows**, the command `/run` now properly utilizes **PowerShell** or **cmd.exe**, improving user experience.
   - Users can also expect a fallback to simple `input()` prompts when **--no-pretty** is active or when using a Windows console, increasing accessibility.
- **Integration of New Cohere Models**: Aider now supports the new **08-2024 Cohere models**, announced by @jalammar, expanding the tool's versatility.
   - This update allows for recursive directory additions using the command **/read-only**, streamlining workflows.
- **Enhanced Performance with Bug Fixes**: Numerous fixes have been applied to resolve corner-case crashes, alongside improvements to the prompt cache chunking strategy.
   - The update also features a refined sanity check for git repositories at startup, ensuring robust operation.
- **Full Changelog Available**: For a detailed overview of changes, users can refer to the full **change log** at [aider.chat/HISTORY.html](https://aider.chat/HISTORY.html).
   - This log lists all new features, improvements, and fixes introduced in the recent updates.



**Link mentioned**: <a href="https://aider.chat/HISTORY.html">Release history</a>: Release notes and stats on aider writing its own code.

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1286770309483073659)** (513 messages🔥🔥🔥): 

> - `Using Aider with OpenRouter and Claude models`
> - `Challenges with DeepSeek and Sonnet models`
> - `Experiences with o1 models`
> - `Issues with Anthropic services`
> - `Contributions to Aider and coding workflow` 


- **Navigating Aider and OpenRouter Models**: Users reported mixed experiences using Aider with o1 models, citing frequent 'overloaded' errors and confusion when querying Claude models directly.
   - While some successfully access Anthropic models via OpenRouter, others struggle with persistent issues, indicating potential ongoing service instability.
- **DeepSeek vs Sonnet Models**: Some users find DeepSeek performs better than Sonnet, especially regarding avoiding looping errors during code completion.
   - Discussion around using these models indicates a preference for the execution capabilities of DeepSeek, contrasting with the analysis strengths of Sonnet.
- **Expectations for New AI Models**: Anticipation builds around the potential release of Opus 3.5, with users speculating on its capabilities compared to existing models.
   - Conversations suggest a general excitement and hope for significant advances in functionality that might enhance developer productivity.
- **Error Management in Aider**: Users frequently encounter issues where o1 models respond incorrectly or in unintended languages, prompting some to revise their prompts.
   - Adding system prompts has been suggested, yet it appears to have limited effect, leading to frustration with the models' reliability.
- **Contributing to Aider**: Users seek guidance on contributing to Aider, discussing the importance of contribution guidelines and best practices.
   - With the introduction of new features like read-only access to specific files, community support for managing and enhancing Aider's functionality is on the rise.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rowancheung/status/1838280020642676802">Tweet from Rowan Cheung (@rowancheung)</a>: I just finished up an exclusive interview going over a new, major AI model upgrade.  Can confirm, tomorrow will be a big day for developers.  Dropping the full conversation on X the second the embargo...</li><li><a href="https://voideditor.com/">Void</a>: Void is an open source Cursor alternative. Full privacy. Fully-featured.</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/browser.html">Aider in your browser</a>: Aider can run in your browser, not just on the command line.</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: Tell aider to follow your coding conventions when it works on your code.</li><li><a href="https://aider.chat/docs/install/install.html">Installing aider</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/2024/08/14/code-in-json.html">LLMs are bad at returning code in JSON</a>: LLMs write worse code if you ask them to return the code wrapped in JSON via a tool function call.</li><li><a href="https://console.groq.com/settings/limits">GroqCloud</a>: Experience the fastest inference in the world</li><li><a href="https://draftjs.org/docs/getting-started">Overview | Draft.js</a>: Draft.js is a framework for building rich text editors in React, powered by an immutable model and abstracting over cross-browser differences.</li><li><a href="https://tenor.com/view/side-eye-cat-gif-8216273864367202904">Side Eye Cat GIF - Side eye cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.anthropic.com/incidents/xts3kyr0nrx1">Elevated Errors on Claude 3.5 Sonnet</a>: no description found</li><li><a href="https://trypear.ai/">PearAI - Open Source AI Code Editor for Fast Development</a>: PearAI is an Open-source AI-powered code editor with features like AI chat, inline prompts, and debugging to accelerate your coding process.</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://www.instagram.com/reel/DAGuPUpP6gG/?igsh=eXBpczA3b3g5MnBx">Leon Si on Instagram: &quot;at this point are we even developers anymore? &#x1f972; #tech #programming #code #ai&quot;</a>: 196K likes, 2,260 comments - leonsilicon on September 19, 2024: &quot;at this point are we even developers anymore? &#x1f972; #tech #programming #code #ai&quot;. </li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider is AI pair programming in your terminal</li><li><a href="https://tenor.com/view/its-just-gambling-liam-scott-edwards-ace-trainer-liam-betting-gamble-gif-20475304">Its Just Gambling Liam Scott Edwards GIF - Its Just Gambling Liam Scott Edwards Ace Trainer Liam - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.blog/news-insights/product-news/try-out-openai-o1-in-github-copilot-and-models/">Try out OpenAI o1 in GitHub Copilot and Models</a>: OpenAI o1-preview and o1-mini are now available in GitHub Copilot Chat in VS Code and in the GitHub Models playground.</li><li><a href="https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis">An Analysis of Chinese LLM Censorship and Bias with Qwen 2 Instruct</a>: no description found</li><li><a href="https://x.com/alexalbert__/status/1836447593888649646?s=61">Tweet from Alex Albert (@alexalbert__)</a>: One of my favorite @AnthropicAI API features that people don&#39;t seem to know about is prompt prefilling.   Your API request doesn&#39;t have to end with a &#39;user&#39; turn. You can include an &#...</li><li><a href="https://www.marscode.com/">MarsCode - AI IDE</a>: MarsCode provides an IDE with a built-in AI Assistant and extensions that support over 100 languages and mainstream IDEs.</li><li><a href="https://pieces.app/">Pieces for Developers - Your Workflow Copilot</a>: Integrate your toolchain, efficiently capture, enrich, and reuse materials. Enhance collaboration with the assistance of an on-device copilot.</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md">aider/CONTRIBUTING.md at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/o1-waitlist-signup?utm_campaign=GitHub_Blog">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://x.com/leonsilicon/status/1837129318306304394?s=46&t=ZSMBWlGirVJCSoNbnkJuPQ">Tweet from Leon Si (@leonsilicon)</a>: developers are cooked</li><li><a href="https://www.youtube.com/watch?v=XAeKtyL2m-Q&t=169s">Interview with Jr. Product Manager [Startup]</a>: Product Manager [Startup]Part II for a coffee this week on https://www.patreon.com/ProgrammersAreAlsoHumanInterview with a Junior Product Manager with Josh D...</li><li><a href="https://x.com/leonsilicon/status/1837129318306304394?s=46&t=ZSMB">Tweet from Leon Si (@leonsilicon)</a>: developers are cooked</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/prompts.py">aider/aider/prompts.py at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/voideditor/void">GitHub - voideditor/void</a>: Contribute to voideditor/void development by creating an account on GitHub.</li><li><a href="https://github.com/PierrunoYT/awesome-ai-dev-tools?">GitHub - PierrunoYT/awesome-ai-dev-tools: A curated list of powerful and innovative development tools, including code editors, plugins, and productivity enhancers. This repository aims to be a comprehensive resource for developers looking to optimize their workflow and boost efficiency. From IDEs to command-line utilities, find the tools that will take your coding to the next level</a>: A curated list of powerful and innovative development tools, including code editors, plugins, and productivity enhancers. This repository aims to be a comprehensive resource for developers looking ...</li><li><a href="https://github.com/paul-gauthier/aider/pull/1176">/read-only by glob pattern by akaihola · Pull Request #1176 · paul-gauthier/aider</a>: Work in progress – basic use cases verified, need tests for more complex scenarios  This patch modifies the /read-only command to behave like /add by accepting directories and glob patterns. A dire...</li><li><a href="https://github.com/PierrunoYT/photo-location-finder">GitHub - PierrunoYT/photo-location-finder: This program allows the user to detect landmarks in an image using the Google Cloud Vision API. The program prompts the user for the image path, API key, and credentials to authenticate with the Google Cloud API.</a>: This program allows the user to detect landmarks in an image using the Google Cloud Vision API. The program prompts the user for the image path, API key, and credentials to authenticate with the Go...</li><li><a href="https://cursor.directory/">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://github.com/PierrunoYT/awesome-ai-dev-tools">GitHub - PierrunoYT/awesome-ai-dev-tools: A curated list of powerful and innovative development tools, including code editors, plugins, and productivity enhancers. This repository aims to be a comprehensive resource for developers looking to optimize their workflow and boost efficiency. From IDEs to command-line utilities, find the tools that will take your coding to the next level</a>: A curated list of powerful and innovative development tools, including code editors, plugins, and productivity enhancers. This repository aims to be a comprehensive resource for developers looking ...</li><li><a href="https://platform.deepseek.com/api-docs/updates/#version-2024-09-05">Change Log | DeepSeek API Docs</a>: Version: 2024-09-05</li><li><a href="https://github.com/PierrunoYT/awesome-dev-tools">GitHub - PierrunoYT/awesome-ai-dev-tools: A curated list of powerful and innovative development tools, including code editors, plugins, and productivity enhancers. This repository aims to be a comprehensive resource for developers looking to optimize their workflow and boost efficiency. From IDEs to command-line utilities, find the tools that will take your coding to the next level</a>: A curated list of powerful and innovative development tools, including code editors, plugins, and productivity enhancers. This repository aims to be a comprehensive resource for developers looking ...</li><li><a href="https://cloudonair.withgoogle.com/events/gemini-at-work-24">Gemini at Work</a>: Join Google Cloud CEO Thomas Kurian and industry leaders to discover how AI is reshaping businesses across the globe.</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>: Every front-end GUI client for ChatGPT. Contribute to billmei/every-chatgpt-gui development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider/commit/7fa1620f58132ec085a7939a8015bbe7935827a2">feat: Allow flexible matching of 5-9 characters in SEARCH/REPLACE blo… · paul-gauthier/aider@7fa1620</a>: …ck prefixes
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1286784404114636812)** (167 messages🔥🔥): 

> - `Aider Functionality`
> - `GitHub Integration with Aider`
> - `Chat History Handling`
> - `Repository Map Optimization`
> - `Usage of Aider with Local Models` 


- **Aider's repository map and chat history**: Aider can maintain a concise map of the entire git repository, which facilitates understanding code changes and relations while sending updates to the LLM upon each change request.
   - When using Aider, if you want to prevent automatic updates to the repo map, you can run it with `--map-refresh manual`, but a full refresh might be needed when new files are added.
- **Using Aider with manual control**: To optimize Aider’s performance, it is suggested to start it with a limited number of tokens for the repository map, as too much information can confuse the LLM.
   - Setting `--map-tokens` to 2048 is generally acceptable, but using a lower number like 1024 may yield better clarity for the model.
- **Integration of Aider with documentation**: Aider can be used alongside many Markdown documents, allowing you to add specific files for review and ask questions about their contents.
   - However, Aider is not primarily a document mining tool, and using it to extract information from extensive documentation may not be its strongest feature.
- **Support for local and external models**: Aider is designed to work with several local models and supports various external APIs, though newer versions require Python 3.9 or later.
   - Additionally, Aider can connect to HuggingFace models and utilize LiteLLM to streamline interactions with available models.
- **Working with environment variables and configurations**: Users can configure Aider using `.env` files to manage settings for different environments, keeping their setup portable across machines.
   - Utilizing symbolic references in configurations for files like `CONVENTIONS.md` is recommended in order to avoid hardcoding paths.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/repomap.html#optimizing-the-map">Repository map</a>: Aider uses a map of your git repository to provide code context to LLMs.</li><li><a href="https://aider.chat/docs/git.html">Git integration</a>: Aider is tightly integrated with git.</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-turn-on-the-repository-map">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://huggingface.co/chat/models">HuggingChat - Models</a>: Browse HuggingChat available models</li><li><a href="https://aider.chat/docs/troubleshooting/imports.html#replit">Dependency versions</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/config/options.html#--map-refresh-value">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://aider.chat/docs/config/dotenv.html">Config with .env</a>: Using a .env file to store LLM API keys for aider.</li><li><a href="https://aider.chat/docs/llms.html">Connecting to LLMs</a>: Aider can connect to most LLMs for AI pair programming.</li><li><a href="https://aider.chat/examples/README.html">Example chat transcripts</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: Intro and tutorial videos made by aider users.</li><li><a href="https://docs.litellm.ai/docs/providers/huggingface">Huggingface | liteLLM</a>: LiteLLM supports the following types of Hugging Face models:</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401">401 Unauthorized - HTTP | MDN</a>: The HTTP 401 Unauthorized client error response status code indicates that a request was not successful because it lacks valid authentication credentials for the requested resource.   This status code...</li><li><a href="https://github.com/paul-gauthier/aider/blob/a4f608f3dd579c561d15cda3f06e785973cb1261/aider/commands.py#L1087)">aider/aider/commands.py at a4f608f3dd579c561d15cda3f06e785973cb1261 · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/larryhudson/aider-switchcoder-debugging">GitHub - larryhudson/aider-switchcoder-debugging</a>: Contribute to larryhudson/aider-switchcoder-debugging development by creating an account on GitHub.</li><li><a href="https://aider.chat/docs/config/options.html">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://github.com/All-Hands-AI/OpenHands">GitHub - All-Hands-AI/OpenHands: 🙌 OpenHands: Code Less, Make More</a>: 🙌 OpenHands: Code Less, Make More. Contribute to All-Hands-AI/OpenHands development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider/blob/54cfbc4142e10dde73434accd20761bfc1ba3f1e/aider/main.py#L714-L723)">aider/aider/main.py at 54cfbc4142e10dde73434accd20761bfc1ba3f1e · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider/blob/cee0bb713568539ecf97b6494f087cc7ddcf926b/aider/main.py#L714">aider/aider/main.py at cee0bb713568539ecf97b6494f087cc7ddcf926b · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1286782184220397639)** (9 messages🔥): 

> - `Aider Tool Development`
> - `Embeddings and RAG`
> - `Flask App for SmartPoi Firmware`
> - `GitHub to RSS Proxy`
> - `Claude and Manual Search` 


- **Simplifying Tool Development with Aider**: A member expressed frustration over wasting hours adding features to Aider when a simple disk search would have sufficed, realizing that using 'manual-mode' produced better results quickly.
   - They proposed building a quick search tool using **ripgrep** for **aider integration** to simplify the process.
- **Embeddings Debated as Overrated**: A user argued that embeddings are overrated, advocating for a DIY approach using tools and chapter summaries instead of standard embeddings, likening the method to a tree structure similar to **llama index**.
   - They humorously suggested that the prevalence of embeddings is driven by **VC funding**, creating a market saturated with embedding RAG tutorials.
- **AI Coding Simplified with Flask App**: A member shared their experience creating a Flask app using only free LLMs for the **SmartPoi Arduino Firmware project**, highlighting that AI coding can be cost-effective.
   - They noted that while free LLMs can be slow and occasionally error-prone, the results were satisfactory, and they're now considering a comparison between free and paid AI models.
- **GitHub Issues Converted to RSS Feed**: A user introduced a GitHub repository that provides a **GitHub to RSS proxy**, allowing users to convert GitHub issues and PRs into an RSS feed.
   - This solution is framed as particularly useful for monitoring projects without the burden of notification spam.
- **Misunderstandings Surrounding RAG**: A member agreed on the overrated nature of embeddings, suggesting that widespread investment in **RAG** tools was due to a lack of understanding of agentic behavior.
   - This aligns with ongoing discussions regarding the efficacy of current AI methodologies versus alternative approaches.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.circusscientist.com/2024/09/19/smartpoi-firmware-downloader-made-with-ai/">SmartPoi Firmware Downloader - made with AI - Circus Scientist</a>: I made a Flask app from scratch using Aider &#8211; the AI coding assistant &#8211; and FREE LLM&#8217;s. This is for the SmartPoi Arduino Firmware project &#8211; POV Poi, now easier than ever to use...</li><li><a href="https://github.com/meain/gh-issues-to-rss">GitHub - meain/gh-issues-to-rss: Convert github issues and prs into rss feed</a>: Convert github issues and prs into rss feed. Contribute to meain/gh-issues-to-rss development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1287766821117628457)** (1 messages): 

> - `μ-parameterization guide`
> - `Cerebras collaboration`
> - `nanoGPT implementation`
> - `GPT-NeoX integration` 


- **Joint μ-Parameterization Guide with Cerebras**: Today, we're excited to drop a joint blog on [The Practitioner's Guide to the Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/), aiming to improve the accessibility of **μ-parameterization** (μP) for the training community.
   - This guide includes **step-by-step implementation instructions** and a simple implementation at [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup), addressing common accessibility issues found in the original materials.
- **Benefits of Wider Adoption of μP**: The guide highlights that wider adoption of μP can lead to **reduced instabilities** during training and lower compute required for hyperparameter optimization.
   - Furthermore, it suggests μP enables **more robust comparisons** between different training methods, which can foster better research outcomes.
- **Simplified μP Implementation Features**: The guide simplifies μP concepts and includes essential **verifications** for the μP implementation involving coord-checks and full LR transfer.
   - This nuanced approach makes it easier for practitioners to grasp the core concepts without getting lost in complexity.
- **Future of μP in GPT-NeoX**: Informed by the guide's implementation, we'll be integrating this simplified μP into the **upcoming GPT-NeoX 3.0 release**.
   - Ongoing updates and tracking of these efforts can be found in the [GPT-NeoX repository](https://github.com/EleutherAI/gpt-neox/pull/1087).



**Link mentioned**: <a href="https://github.com/EleutherAI/gpt-neox/pull/1087.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1287445602279358515)** (379 messages🔥🔥): 

> - `Cosine Similarity in GPT-4 Evaluation`
> - `Test Set Leakage Concerns`
> - `Understanding RWKV Architecture`
> - `Maximal Update Parametrization (muP)`
> - `Optimizer Code Complexity in JAX vs Pytorch` 


- **Using Cosine Similarity with GPT-4**: A user is evaluating GPT-4 for a classification task without fine-tuning, considering dynamically selecting examples based on cosine similarity from a test set for improved in-context learning.
   - Concerns were raised about the potential for test set leakage by including similar test examples in the prompt, while ensuring that the test question itself is not included.
- **Evaluating Test Set Leakage Risk**: A member expressed concerns about the risk of test set leakage when using the test set as a pool for selecting in-context examples.
   - It was noted that while the selection may not directly include the test example, similarities could lead to indirect leakage, thus impacting the evaluation's validity.
- **Challenges in Understanding RWKV Architecture**: Participants discussed the complexity of the RWKV architecture, noting that many find it challenging to grasp the distinctions and similarities with other models like GLA.
   - It was suggested that simplified explanations could aid in better understanding, but existing resources may still feel opaque or complicated.
- **Maximal Update Parametrization Simplified**: The discussion highlighted the need for accessible explanations of Maximal Update Parametrization (muP) to foster better understanding and usage in machine learning frameworks.
   - A blog post was mentioned that aims to demystify muP, making it more approachable without delving deeply into complex theoretical aspects.
- **Optimizer Code Complexity in JAX vs Pytorch**: Participants debated the relative complexity of shampoo implementations in JAX compared to Pytorch, with opinions varying on which was simpler or more straightforward.
   - It was noted that while JAX might offer more flexibility through its APIs, its implementations may be more verbose and complex, unlike the more concise Pytorch alternatives.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cloneofsimo/status/1836114264215687601?s=46">Tweet from Simo Ryu (@cloneofsimo)</a>: Corrected data</li><li><a href="https://arxiv.org/abs/2407.17465">u-$μ$P: The Unit-Scaled Maximal Update Parametrization</a>: The Maximal Update Parametrization ($μ$P) aims to make the optimal hyperparameters (HPs) of a model independent of its size, allowing them to be swept using a cheap proxy model rather than the full-si...</li><li><a href="https://x.com/jxbz">Tweet from undefined</a>: no description found</li><li><a href="https://arxiv.org/abs/2407.05872">Scaling Exponents Across Parameterizations and Optimizers</a>: Robust and effective scaling of models from small to large width typically requires the precise adjustment of many algorithmic and architectural details, such as parameterization and optimizer choices...</li><li><a href="https://arxiv.org/abs/2101.06804">What Makes Good In-Context Examples for GPT-$3$?</a>: GPT-$3$ has attracted lots of attention due to its superior performance across a wide range of NLP tasks, especially with its powerful and versatile in-context few-shot learning ability. Despite its s...</li><li><a href="https://arxiv.org/abs/2405.14813">Scalable Optimization in the Modular Norm</a>: To improve performance in contemporary deep learning, one is interested in scaling up the neural network in terms of both the number and the size of the layers. When ramping up the width of a single l...</li><li><a href="https://arxiv.org/abs/2310.17813">A Spectral Condition for Feature Learning</a>: The push to train ever larger neural networks has motivated the study of initialization and training at large network width. A key challenge is to scale training so that a network&#39;s internal repre...</li><li><a href="https://x.com/cloneofsimo/status/1838287517906510026">Tweet from Simo Ryu (@cloneofsimo)</a>: Good stuff. Pro tip: do the red circles i checked will get you 99% there. (but dont scale the head dim) https://blog.eleuther.ai/mutransfer/</li><li><a href="https://jeremybernste.in/modula/bad-scaling/">Bad scaling</a>: At the simplest level, neural networks are trained by iterating the following operation: where learning_rate is a float and gradient is the gradient of the loss function with respect to the weights...</li><li><a href="https://huggingface.co/blog/rwkv">Introducing RWKV - An RNN with the advantages of a transformer</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=_NC9sc-nXoc">Session 2A &amp; 2B: Optimization Non Convex</a>: Watch this video with AI-generated Table of Content (ToC), Phrase Cloud and In-video Search here:https://videos.videoken.com/index.php/videos/icml-2018-sessi...</li><li><a href="https://x.co">Sell Domains | Buy Domains | Park Domains</a>: no description found</li><li><a href="https://github.com/cloneofsimo/zeroshampoo/blob/main/distributed_shampoo.py">zeroshampoo/distributed_shampoo.py at main · cloneofsimo/zeroshampoo</a>: Contribute to cloneofsimo/zeroshampoo development by creating an account on GitHub.</li><li><a href="https://github.com/google-research/google-research/blob/master/scalable_shampoo/jax/shampoo.py">google-research/scalable_shampoo/jax/shampoo.py at master · google-research/google-research</a>: Google Research. Contribute to google-research/google-research development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1286814968624320563)** (206 messages🔥🔥): 

> - `Curriculum Learning in AI`
> - `Interpretability of AI Models`
> - `Planning Abilities of LLMs`
> - `Performance of Large Language Models`
> - `Evaluation of Explainable AI` 


- **Debate on Curriculum Learning Effectiveness**: There is ongoing discussion about the effectiveness of curriculum learning (CL) in AI, with some suggesting that it may not provide significant improvements over traditional training methods.
   - Members expressed skepticism about the impact of CL in real-world applications, citing that there are no guaranteed best practices for filtering data.
- **OpenAI's New Large Reasoning Model Claims**: OpenAI's recent model, labeled a Large Reasoning Model (LRM), claims to escape traditional limitations of autoregressive LLMs, generating interest in its performance compared to existing models.
   - However, some members questioned the distinction of LRM and pointed out that improvements may be achievable through existing methods at high computational costs.
- **Skepticism Around Interpretability in AI**: A member referenced a paper discussing the shortcomings of interpretability methods in AI, noting that many do not provide meaningful insights for human decision-making.
   - The findings indicate that typical feature attribution explanations may lead to worse decision outcomes due to cognitive biases, challenging assumptions about their universal benefit.
- **Human Performance Benchmarks**: The discussion highlighted benchmarks comparing AI performance to human capabilities, with comments that achieving human-level results is a narrow way to judge AI abilities.
   - With the mention of traditional planners like Fast Downward, it was emphasized that AI's planning capabilities should not be judged solely by comparison to human performance.
- **Resource and Efficacy of Data Usage in AI Training**: Participants shared insights on the nuances of data retrieval and processing, with an emphasis on efficient reads from cloud storage formats like Parquet.
   - There was consensus on the need for effective methods to improve data quality in training, but uncertainty remained about universally effective strategies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.pola.rs/user-guide/io/cloud-storage/#reading-from-cloud-storage)">Cloud storage - Polars user guide</a>: no description found</li><li><a href="https://x.com/_xmaster96/status/1837489678024393205?s=46">Tweet from XMaster96 (@_XMaster96)</a>: Don&#39;t we all know the moment when we are keep staring at a training loss curve for way too long?  This drop in the loss curve was me staring at the pre-training of the new Aleph Alpha Foundation m...</li><li><a href="https://arxiv.org/abs/2409.12917">Training Language Models to Self-Correct via Reinforcement Learning</a>: Self-correction is a highly desirable capability of large language models (LLMs), yet it has consistently been found to be largely ineffective in modern LLMs. Existing approaches for training self-cor...</li><li><a href="https://arxiv.org/abs/2409.13373">LLMs Still Can&#39;t Plan; Can LRMs? A Preliminary Evaluation of OpenAI&#39;s o1 on PlanBench</a>: The ability to plan a course of action that achieves a desired state of affairs has long been considered a core competence of intelligent agents and has been an integral part of AI research since its ...</li><li><a href="https://arxiv.org/abs/2012.02748">Challenging common interpretability assumptions in feature attribution explanations</a>: As machine learning and algorithmic decision making systems are increasingly being leveraged in high-stakes human-in-the-loop settings, there is a pressing need to understand the rationale of their pr...</li><li><a href="https://docs.pola.rs/user-guide/io/cloud-storage/#reading-fr">Cloud storage - Polars user guide</a>: no description found</li><li><a href="https://x.com/BlinkDL_AI/status/1838230783078924598">Tweet from BlinkDL (@BlinkDL_AI)</a>: RWKV-7 &#34;Goose&#34; 🪿 preview rc2 =&gt; Peak RNN architecture?😃Will try to squeeze more performance for the final release. Preview code: https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7</li><li><a href="https://arxiv.org/abs/2303.13506">The Quantization Model of Neural Scaling</a>: We propose the Quantization Model of neural scaling laws, explaining both the observed power law dropoff of loss with model and data size, and also the sudden emergence of new capabilities with scale....</li><li><a href="https://arxiv.org/abs/2407.21075">Apple Intelligence Foundation Language Models</a>: We present foundation language models developed to power Apple Intelligence features, including a ~3 billion parameter model designed to run efficiently on devices and a large server-based language mo...</li><li><a href="https://arxiv.org/abs/2205.10343">Towards Understanding Grokking: An Effective Theory of Representation Learning</a>: We aim to understand grokking, a phenomenon where models generalize long after overfitting their training set. We present both a microscopic analysis anchored by an effective theory and a macroscopic ...</li><li><a href="https://arxiv.org/abs/2210.01117">Omnigrok: Grokking Beyond Algorithmic Data</a>: Grokking, the unusual phenomenon for algorithmic datasets where generalization happens long after overfitting the training data, has remained elusive. We aim to understand grokking by analyzing the lo...</li><li><a href="https://en.wikipedia.org/wiki/Betteridge%27s_law_of_headlines">Betteridge&#039;s law of headlines - Wikipedia</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.12847v1">Instruction-tuned Language Models are Better Knowledge Learners</a>: In order for large language model (LLM)-based assistants to effectively adapt to evolving information needs, it must be possible to update their factual knowledge through continued training on new dat...</li><li><a href="https://github.com/WinVector/Examples/blob/main/Model_Homotopy/LinRebal.ipynb">Examples/Model_Homotopy/LinRebal.ipynb at main · WinVector/Examples</a>: Various examples for different articles. Contribute to WinVector/Examples development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/openai/MMMLU">openai/MMMLU · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1287556159804276737)** (10 messages🔥): 

> - `Irreducible Loss Calculation`
> - `Chinchilla Optimal Token Size`
> - `Empirical Estimations`
> - `Scaling Laws Insights` 


- **Calculating Irreducible Loss in Autoregressive Models**: A user questioned how the authors of *Scaling Laws for Autoregressive Modeling* calculate the **irreducible loss**, referencing the entropy of the true data distribution.
   - One member suggested that it might be fitted empirically along with the power law exponent, noting the lack of a clear answer.
- **Exploring Chinchilla Optimal Token Count**: A user expressed confusion about the phrase regarding selecting **3.2B tokens for pretraining**, questioning whether there is a fixed calculation behind it.
   - It was clarified that the relationship aligns with a tradeoff of approximately **D = 20P**, and that this ratio is often used without rigorous calculation.
- **Ratio Derived from Chinchilla's Findings**: Discussion revealed that the **D = 20P** ratio can be referenced directly from *Hoffman et al.*'s table without needing complex calculations.
   - This indicates that the tokens required for pretraining can be approximated regardless of the FLOP budget, as confirmed by a member.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1286791084139089972)** (61 messages🔥🔥): 

> - `Interpretability at EMNLP2024`
> - `KV Cache Experiments`
> - `Model Training Interventions`
> - `Sparse Feature Circuits`
> - `SAE and Transformer Interpretability` 


- **Papers at EMNLP2024 showcase interpretability**: A member expressed pride in having two papers accepted at [#EMNLP2024](https://x.com/FazlBarez/status/1837229484543726036); one paper focuses on **attention-MLP interactions** in transformers and the other on **interpretable sequence continuation**.
   - These contributions highlight advancements in understanding complex model behaviors.
- **KV Cache Experiments reveal storage mechanisms**: Experimentation with KV cache suggests individual tokens can impact the representation of later layers, illuminating how a single changed token like **'NY'** can propagate through the model.
   - The **spike observations** in cache values imply a need for longer prompts to store meaningful information effectively.
- **Discussions on Model Training Interventions**: There are speculations regarding pretraining interventions potentially influencing interpretability, but the consensus is that modifying architecture may yield better results than altering training processes.
   - Recent studies are highlighting the challenges and potential within **train-time interventions** for improving model understanding.
- **Sparse Feature Circuits offer insights**: Referencing Sam Marks' work, a member pointed out how trained probes revealing spurious correlations can be corrected **post-hoc**, emphasizing the importance of training data adjustments.
   - This method showcases practical applications of interpretability techniques that can also inform broader research areas.
- **SAEs enable broader interpretability contexts**: SAEs (Sparse Attention Embeddings) are discussed as tools to expand the context in which transformers can be interpreted, moving beyond limited prompt testing.
   - The dialogue calls for more practical instantiations and successful applications of SAE techniques in **interpretability** challenges.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/FazlBarez/status/1837229484543726036">Tweet from Fazl Barez (@FazlBarez)</a>: Super proud to have 2  papers at #EMNLP2024! 🚀 1️⃣ &#34;Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions&#34;  2️⃣&#34;Towards Interpretable Sequence Continuati...</li><li><a href="https://arxiv.org/abs/2309.07311v4">Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs</a>: Most interpretability research in NLP focuses on understanding the behavior and features of a fully trained model. However, certain insights into model behavior may only be accessible by observing the...</li><li><a href="https://github.com/PicoCreator/QKV-Transformers-are-RNNs?tab=readme-ov-file#immediate-implication>">GitHub - PicoCreator/QKV-Transformers-are-RNNs: QKV Transformers are RNN&#39;s with extra steps and larger memory capacity</a>: QKV Transformers are RNN&#39;s with extra steps and larger memory capacity - PicoCreator/QKV-Transformers-are-RNNs
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1287776370247598213)** (8 messages🔥): 

> - `MMLU_PRO sampling logic`
> - `Gemma model BOS token usage`
> - `Pythia 6.9b-deduped low scores`
> - `MMLU task description importance` 


- **MMLU_PRO sampling logic needs attention**: The `./leaderboard/mmlu_pro` task differs from its original implementation as it ignores question categories for fewshot sampling, unlike the [MMLU-PRO code](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/47b9891aacb8bd7cda29d5c5ba17b9434dd333bc/evaluate_from_local.py#L228).
   - Another user has suggested an updated sampling logic to improve its accuracy based on question categories, with the specific implementation detailed [here](https://github.com/rimashahbazyan/lm-evaluation-harness/blob/f117e6c09e32c553df0ab8cf8964a8b16636832e/lm_eval/api/samplers.py#L186).
- **Gemma model deserves adjustments**: A member emphasized the importance of incorporating a BOS token for the Gemma model, noting the current practice may break perplexity task assumptions.
   - They plan to add a toggle flag for this behavior, with the default set to `False` but overriding to `True` for Gemma models as a special case.
- **Low MMLU scores from Pythia model discussed**: Concerns were raised regarding low MMLU 5-shot scores from the Pythia 6.9b-deduped model, questioning its validity compared to published scores.
   - Other members suggested that models trained on the Pile struggle with MMLU particularly due to formatting issues.
- **Task descriptions in context crucial**: Discussion highlighted that the non-leaderboard `mmlu_pro` correctly uses relevant subjects for fewshots and includes task descriptions in context.
   - A member suggested that the task descriptions should end with a newline, aligning it with the reference implementation, and plans to make a PR.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/47b9891aacb8bd7cda29d5c5ba17b9434dd333bc/evaluate_from_local.py#L228)">MMLU-Pro/evaluate_from_local.py at 47b9891aacb8bd7cda29d5c5ba17b9434dd333bc · TIGER-AI-Lab/MMLU-Pro</a>: The scripts for MMLU-Pro. Contribute to TIGER-AI-Lab/MMLU-Pro development by creating an account on GitHub.</li><li><a href="https://github.com/rimashahbazyan/lm-evaluation-harness/blob/f117e6c09e32c553df0ab8cf8964a8b16636832e/lm_eval/api/samplers.py#L186">lm-evaluation-harness/lm_eval/api/samplers.py at f117e6c09e32c553df0ab8cf8964a8b16636832e · rimashahbazyan/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - rimashahbazyan/lm-evaluation-harness</li><li><a href="https://github.com/rimashahbazyan/lm-evaluation-harness/blob/robustness_task/lm_eval/tasks/robustness/mmlu_pro/fewshot_prompt_robustness_mmlu_pro.yaml">lm-evaluation-harness/lm_eval/tasks/robustness/mmlu_pro/fewshot_prompt_robustness_mmlu_pro.yaml at robustness_task · rimashahbazyan/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - rimashahbazyan/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1287751274972053544)** (7 messages): 

> - `Activation Functions Sync`
> - `Init Functions and Stability`
> - `Truncation of Normal Distribution` 


- **Activation Functions Documentation Out of Sync**: A member pointed out that the available activation functions listed in the documentation do not reflect the full range present in the code, particularly [Swiglu](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/activations.py).
   - Another member confirmed that the documentation had not been updated, referencing a [specific line in the code](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/neox_arguments/neox_args.py#L295) where these functions are listed.
- **Trunc Normal Initialization Discussed**: A member suggested changing the init functions to **trunc_normal**, referencing an ablation study which shows instability at scale without it, as noted in the [AllenAI research](https://arxiv.org/abs/2409.02060).
   - The member highlighted that multiple authors are involved, suggesting a substantial amount of work and research backing this approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.02060">OLMoE: Open Mixture-of-Experts Language Models</a>: We introduce OLMoE, a fully open, state-of-the-art language model leveraging sparse Mixture-of-Experts (MoE). OLMoE-1B-7B has 7 billion (B) parameters but uses only 1B per input token. We pretrain it ...</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/megatron/neox_arguments/neox_args.py#L295">gpt-neox/megatron/neox_arguments/neox_args.py at main · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/activations.py">gpt-neox/megatron/model/activations.py at main · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1286766396268609566)** (560 messages🔥🔥🔥): 

> - `KTO Trainer`
> - `Qwen Model Fine-tuning`
> - `RAG Implementation`
> - `Chat Template Issues`
> - `Reflection Fine-tune` 


- **Discussion on KTO Trainer Usage**: Members clarified that the KTO trainer requires a reference model to calculate rewards, suggesting that the untouched base model should be used for comparison during fine-tuning.
   - There were suggestions to pre-generate responses from the reference model to save memory during the training process.
- **Issues with Qwen Model Fine-tuning**: Users experienced unexpected behavior from the Qwen 2.5 model after updates, particularly generating incorrect responses related to prompt templates.
   - It was noted that the smaller model is sensitive to prompt formatting, with issues stemming from changes made to prompt handling.
- **RAG Implementation Discussions**: Participants discussed using Retrieval-Augmented Generation (RAG) as a method to enhance model responses and address limitations in knowledge retention from fine-tuning alone.
   - One user recommended using existing datasets effectively in RAG to avoid knowledge loss during training.
- **Chat Template Issues**: Users highlighted difficulties with maintaining chat templates in fine-tuned models, particularly the need for saving custom templates alongside model weights.
   - A reference was made to Hugging Face documentation on creating and saving chat templates for models.
- **Reflection Fine-tune**: Discussions indicated that training on reflection traces using the reflect fine-tune method may not yield significant improvements without a robust reward model.
   - Participants noted the importance of using methods like BoN for better alignment and performance in fine-tuning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/powertoys/fancyzones">PowerToys FancyZones utility for Windows</a>: A window manager utility for arranging and snapping windows into efficient layouts</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing#scrollTo=yqxqAZ7KJ4oL">Google Colab</a>: no description found</li><li><a href="https://cobusgreyling.medium.com/prompt-tuning-hard-prompts-soft-prompts-49740de6c64c">Prompt Tuning, Hard Prompts &amp; Soft Prompts</a>: Prompt Engineering is the method of accessing Large Language Models (LLMs), hence implementations like Pipelines, Agents, Prompt Chaining &amp;…</li><li><a href="https://docs.unsloth.ai/basics/saving-models/saving-to-vllm">Saving to VLLM | Unsloth Documentation</a>: Saving models to 16bit for VLLM</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=vITh0KVJ10qX">Google Colab</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.01306">KTO: Model Alignment as Prospect Theoretic Optimization</a>: Kahneman &amp; Tversky&#39;s $\textit{prospect theory}$ tells us that humans perceive random variables in a biased but well-defined manner (1992); for example, humans are famously loss-averse. We show...</li><li><a href="https://unsloth.ai/blog/llama3-1">Finetune Llama 3.1 with Unsloth</a>: Fine-tune and run Meta&#x27;s updated Llama 3.1 model with 6x longer context lengths via Unsloth!</li><li><a href="https://docs.unsloth.ai/basics/chat-templates">Chat Templates | Unsloth Documentation</a>: no description found</li><li><a href="https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct">nvidia/Llama-3_1-Nemotron-51B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://www.metadock.net/">Home - MetaDock</a>: Say goodbye to constant window switching. MetaDock lets you manage multiple tasks seamlessly with its unique split-screen and multi-layout system. Try it now!</li><li><a href="https://x.com/AlpinDale/status/1837860256073822471">Tweet from Alpin (@AlpinDale)</a>: You can now load any FP16 model in any floating-point format you want, as long as it&#39;s between 2 and 7 bits. Do you want a non-standard FP6_E3M2, or FP7_E1M5? It should just work. The throughput i...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://learnprompting.org/docs/trainable/soft_prompting">Advantages and Mechanics of Soft Prompts</a>: Discover the benefits of prompt tuning over model fine-tuning. Learn how prompt tuning and soft prompts work with large language models.</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Chat Templates</a>: no description found</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">llama-recipes/recipes/multilingual/README.md at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>: Scripts for fine-tuning Meta Llama3 with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q...</li><li><a href="https://huggingface.co/docs/trl/main/en/kto_trainer">KTO Trainer</a>: no description found</li><li><a href="https://github.com/codelion/optillm">GitHub - codelion/optillm: Optimizing inference proxy for LLMs</a>: Optimizing inference proxy for LLMs. Contribute to codelion/optillm development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M">But what is a GPT?  Visual intro to transformers | Chapter 5, Deep Learning</a>: Breaking down how Large Language Models workInstead of sponsored ad reads, these lessons are funded directly by viewers: https://3b1b.co/support---Here are a...</li><li><a href="https://huggingface.co/datasets/LlamaFinetuneGGUF/Programming-Alpaca-and-ShareGPT-Style">LlamaFinetuneGGUF/Programming-Alpaca-and-ShareGPT-Style · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/1_58_llm_extreme_quantization">Fine-tuning LLMs to 1.58bit: extreme quantization made easy</a>: no description found</li><li><a href="https://github.com/mistralai/mistral-common/tree/main/src/mistral_common/tokens/tokenizers">mistral-common/src/mistral_common/tokens/tokenizers at main · mistralai/mistral-common</a>: Contribute to mistralai/mistral-common development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/1001">Support KTO Trainer with Unsloth by corbt · Pull Request #1001 · unslothai/unsloth</a>: This patch appears to be both necessary and sufficient to successfully use KTOTrainer with Unsloth!</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/infiniflow/ragflow">GitHub - infiniflow/ragflow: RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding.</a>: RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. - infiniflow/ragflow
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1286883794258755596)** (24 messages🔥): 

> - `RAG Application Use`
> - `Cost Analysis for Document Rating`
> - `Inference Methods Comparison`
> - `API Services Discounts`
> - `Vote Accuracy in Ratings` 


- **Exploring RAG Applications for Document Structuring**: A member suggested using a **RAG application** to convert unstructured documents into a structured format before conducting analysis.
   - Another member clarified that their task involves **L3.1 ratings** and is focused on offline inference rather than creating a fine-tuning dataset.
- **Costly Estimates for Document Processing**: Discussion revealed that running an analysis on **2.5 million documents** with high token counts could cost around **$60k** without labor.
   - One member calculated that using an **API for L3.1** would cost approximately **$15k**, indicating a significant savings compared to on-prem configurations.
- **Comparing Inference Methods**: Members debated the benefits of various inference methods, noting that throughput of **8x H100** models could offer faster results than anticipated.
   - *Testing with 2000-5000 samples* was recommended to evaluate cost and accuracy effectively.
- **API Services with Discounts**: A member raised the question of whether any **API services offer discounts**, particularly highlighting **OpenAI**'s previous 50% off on batch inferences.
   - Concerns were shared about the high costs and limitations of using larger models versus the unsatisfactory performance of smaller ones.
- **Three Votes for Enhanced Accuracy**: Members discussed the importance of obtaining **three votes** from different models to ensure accuracy in ratings.
   - One member confirmed they will implement this approach in their testing strategy.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1286787189497266367)** (76 messages🔥🔥): 

> - `Prediction Loss Only Evaluation`
> - `Phi 3.5 Tokenization Issues`
> - `RAG Fine Tuning Best Practices`
> - `Merged Model Performance Challenges`
> - `Continued Pre-training with Lora Adapters` 


- **Prediction Loss Only for VRAM Efficiency**: A user asked about the purpose of using `prediction_loss_only = True` in the training loop to prevent VRAM usage from escalating.
   - Concerns were raised regarding whether it affects evaluation passes only.
- **Tokenization Concerns with Phi 3.5**: A user noted discrepancies in tokenization between the model and tokenizer in Phi 3.5, leading to confusion about padding tokens.
   - Additionally, there were issues with the tokenizer not adding special tokens during encoding, which could impact training.
- **Best Practices for RAG Fine Tuning**: One member inquired about templates for fine-tuning RAG models with context, questions, and answers, highlighting the complexity.
   - Suggestions included exploring research papers for guidance, indicating this is a nuanced area.
- **Performance Issues Post Model Merging**: Users reported that the performance of their models declined significantly after merging Lora adapters with the original weights.
   - Concerns were expressed about the effectiveness of 4bit merges compared to 16bit merges.
- **Continued Pre-training with Lora Adapters**: A user sought clarity on how continued pre-training would interact with existing Lora adapters, questioning if new ones would be created.
   - It was advised to save the merged model for future training flexibility, emphasizing the importance of maintaining a merged state.



**Link mentioned**: <a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing#scrollTo=r6bUnxe6N3pf">Google Colab</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1286947640965333022)** (3 messages): 

> - `SetFit v1.1.0 Release`
> - `Training Classifiers`
> - `Sentence Transformers Update`
> - `Python Version Support` 


- **SetFit v1.1.0 Launches with Improved Training**: The release of **SetFit v1.1.0** now utilizes the Sentence Transformers Trainer for efficient classifiers training on both **CPU and GPU**, addressing multiple issues from third-party library updates.
   - The new version introduces **MultiGPU support** and deprecates the 'evaluation_strategy' in favor of 'eval_strategy', along with new support for **Python 3.11** and **3.12**.
- **Two Phases of SetFit Classifier Model Training**: Training a **SetFit classifier model** consists of two main phases: finetuning a Sentence Transformer embedding model followed by a classifier that maps embeddings to classes.
   - This structured approach enhances performance and efficiency, particularly with the updated support features in version 1.1.0.
- **Key Updates in SetFit's Training Process**: Significant improvements have been made in parameters like **max_steps** and **eval_max_steps**, which are now enforced as hard limits, ensuring more reliable training outcomes.
   - Changes in training and validation losses were also highlighted, contributing to the overall robustness of the training process.



**Link mentioned**: <a href="https://huggingface.co/posts/tomaarsen/875775738519407">@tomaarsen on Hugging Face: &quot;🎉SetFit v1.1.0 is out! Training efficient classifiers on CPU or GPU now uses…&quot;</a>: no description found

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1286763841350406277)** (506 messages🔥🔥🔥): 

> - `Perplexity Pro issues`
> - `Usage of AI models`
> - `Anthropic model release`
> - `Perplexity functionality`
> - `Collaborative opportunities` 


- **Perplexity Pro Subscription Issues**: Several users reported losing their Pro status intermittently, with some experiencing error messages like 'Query rate limit exceeded'. Many noted that logging out and back in sometimes resolved the issue, but concerns about lag and bugs persisted.
   - These problems appear to be system-wide, possibly linked to recent updates and maintenance being performed on the platform.
- **AI Model Comparisons and Use Cases**: Users discussed the effectiveness of different AI models, including Perplexity, ChatGPT, and Claude, highlighting their respective strengths in various applications. Insights were shared on how to optimize their usage for tasks like programming, brainstorming, and academic research.
   - Many noted the challenges with certain models, especially regarding hallucinations and the reliability of real-time information retrieval.
- **Potential Launch of New Anthropic Model**: The community buzzed about the potential drop of a new model from Anthropic, suggested to be announced shortly based on an exclusive interview shared by a user. This generated excitement for additional capabilities that new AI models may bring.
   - There were skeptical comments regarding whether Perplexity would incorporate any new models soon, hinting at the competitive landscape.
- **Concerns About Perplexity's Shift Towards Ads**: User feedback voiced concerns over recent changes in how products and ads are displayed within the Perplexity interface, finding it distracting. Suggestions were made to place recommendations in a sidebar rather than inline with the search results to enhance usability.
   - Users expressed disappointment over perceived shifts towards a more commercial model, which they feared could detract from the unique value Perplexity was originally set to provide.
- **User Experience Enhancements and Collaborations**: Discussion about the Complexity extension highlighted benefits that enhance user experience on Perplexity, such as customizable themes and easier navigation. Users shared collaborative opportunities and expressed interest in improving their workflow with AI tools.
   - The importance of community-driven feedback and understanding how to leverage these tools effectively was emphasized as crucial for enhancing the platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rowancheung/status/1838280020642676802">Tweet from Rowan Cheung (@rowancheung)</a>: I just finished up an exclusive interview going over a new, major AI model upgrade.  Can confirm, tomorrow will be a big day for developers.  Dropping the full conversation on X the second the embargo...</li><li><a href="https://docs.perplexity.ai/guides/model-cards">Supported Models - Perplexity</a>: no description found</li><li><a href="https://www.workingtheorys.com/p/taste-is-eating-silicon-valley">Taste is Eating Silicon Valley.</a>: Just as software ate the world and dramatically transformed industries in the last era, taste is now eating software—and with it, Silicon Valley.</li><li><a href="https://www.intowindows.com/how-to-set-google-as-default-search-in-vivaldi-browser/">How To Set Google As Default Search In Vivaldi Browser</a>: Vivaldi browser is not as popular as its rivals but it’s definitely one of the better web browsers out there for Windows operating system as well as Mac.</li><li><a href="https://www.msn.com/en-gb/money/topstories/perplexity-in-talks-with-top-brands-on-ads-model-as-it-challenges-google/ar-AA1qXfYF)">MSN</a>: no description found</li><li><a href="https://tenor.com/view/the-voices-cat-gif-18434775077971513816">The Voices Cat GIF - The voices cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://addons.mozilla.org/en-US/firefox/addon/complexity/">Complexity - Perplexity.ai supercharged – Get this Extension for 🦊 Firefox (en-US)</a>: Download Complexity - Perplexity.ai supercharged for Firefox. ⚡ Supercharge your Perplexity.ai</li><li><a href="https://tenor.com/view/huh-confused-dont-know-thinking-john-c-reilly-gif-16141237">Huh Confused GIF - Huh Confused Dont Know - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/apostraphi/status/1837219719495176299?s=61">Tweet from Phi Hoang (@apostraphi)</a>: ngl...we weren&#39;t expecting so many students to join the perplexity back to school campaign! welcome + we&#39;re just getting started, y&#39;all.</li><li><a href="https://huggingface.co/spaces/yuntian-deng/o1">Chat-with-OpenAI-o1 - a Hugging Face Space by yuntian-deng</a>: no description found</li><li><a href="https://tenor.com/view/holo-spice-and-wolf-holo-the-wise-wolf-horo-korbo-gif-13009516793083034180">Holo Spice And Wolf GIF - Holo Spice and wolf Holo the wise wolf - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/cat-underwater-gif-922906369727670801">Cat Underwater GIF - Cat Underwater - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1286859683185037343)** (33 messages🔥): 

> - `Human DNA Preservation`
> - `Titan Sub Implosion`
> - `Chain of Thought Reasoning`
> - `AI Meeting Prep Reports`
> - `Python Learning Resources` 


- **Human DNA Preserved in Crystal**: A fascinating article discusses how **human DNA** can be preserved in long-lasting crystals, potentially informing future genetic research. You can read more about it [here](https://www.perplexity.ai/search/human-dna-preserved-in-long-la-W4e5dAggRbuOMuW_mOaqmA).
   - This preservation technique is detailed in the [original thread](https://www.perplexity.ai/page/human-dna-preserved-in-long-la-6_oF.rF1StCqzUJsYCTy4Q).
- **Titan Sub Implosion Insights**: Discussion surrounds the tragic **Titan sub implosion** with links providing insights into what went wrong. Explore more about this incident [here](https://www.perplexity.ai/search/the-titan-sub-implosion-jXEEHAd9RI64Df48GnPkJg).
   - Multiple members shared perspectives on the implications of this event on deep-sea exploration and safety.
- **Chain of Thought Reasoning Best Practices**: The community is pointed to a resource on **Chain of Thought** reasoning—an approach to enhance AI logic and reasoning skills. Check out the guide [here](https://www.perplexity.ai/page/chain-of-thought-reasoning-via-22CYSxmhTMSFr1gJIXM4dg).
   - Additional context was provided in a [related thread](https://www.perplexity.ai/search/using-cot-canvas-with-cplx-and-D5K3ONW.SDm.SDYuTTz_Gw).
- **AI Reports for Meeting Preparation**: One user shared a link to an **AI report generator** that assists in preparing for meetings, showcasing its potential benefits. Read about the insights gathered for Danone [here](https://www.perplexity.ai/search/danone-insights-emea-NR8r3YegQ.K5ww0lrxxQLw).
   - This tool aims to streamline information compilation for effective meeting preparations.
- **Learning Python Resources**: A query was raised regarding resources to **learn Python**, with curated links provided for both beginners and advanced learners. One resource can be found [here](https://www.perplexity.ai/search/how-do-you-learn-python-on-you-qs5urKEZSMOALRSUWpHegA).
   - Various links addressing Python learning strategies were exchanged, catering to different proficiency levels.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1286784429737644174)** (18 messages🔥): 

> - `Llama-3.1-Sonar performance issues`
> - `Perplexity API citation challenges`
> - `Search Recency Filter`
> - `Inconsistent API outputs`
> - `Azure deployment for OCR` 


- **Llama-3.1-Sonar struggles vs. Perplexity web app**: Users report significantly worse results with **llama-3.1-sonar-large-128k-online** compared to the Perplexity web application, citing issues like incomplete outputs and format inconsistencies.
   - One user proposed a multi-step process to improve results, emphasizing the importance of retaining source references.
- **Perplexity API lacking citation reliability**: A user expressed frustration over the **Perplexity API's erratic behavior**, specifically the inconsistent provision of references in answers despite requests for the citation feature.
   - They highlighted that the lack of citations undermines the API's value, which primarily hinges on its **search features**.
- **Inquiries about Search Recency Filter**: A user sought clarification on whether the **search_recency_filter** is part of the closed beta or available to all users, indicating its relevance for gathering timely information.
   - They aimed to ensure that the API could retrieve updates from the last hour while utilizing this filter.
- **Users frustrated with inconsistent API outputs**: Multiple users reported inconsistency in API outputs, including receiving a mix of **Markdown and HTML** despite specifying HTML only in their prompts.
   - This inconsistency is causing frustration as users find the performance better in the web and labs playground.
- **Exploring Azure for OCR services**: A user inquired if it is possible to **deploy a web service with Azure** using the Perplexity API, specifically focusing on OCR capabilities.
   - This indicates a growing interest in leveraging the API for practical applications in cloud environments.


  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1287480964800581734)** (5 messages): 

> - `Browser Thread Usage`
> - `CUDA Browser Development`
> - `Wen Mei Hwu's Lecture`
> - `Server Presence`
> - `User Queries` 


- **Browser using excessive threads**: A user expressed frustration over a browser utilizing **126 threads** for only **10 tabs**, calling for a more efficient solution.
   - This highlights concerns over browser performance and resource management in everyday tasks.
- **Demand for a CUDA-based browser**: One member urgently requested a **CUDA browser**, suggesting a potential gap in current market offerings for performance-focused web browsing.
   - This indicates a desire for enhanced capability in handling parallel tasks through GPU acceleration.
- **Request for a lecture video**: A member inquired about the availability of a video recording of **Wen Mei Hwu's lecture**, emphasizing the interest in his teachings.
   - This reflects ongoing engagement with educational content in the AI and technology community.
- **Checking for Server Presence**: A user inquired whether another member known as **eqy** was present in the server, indicating a social interaction or collaboration query.
   - This highlights the communal nature and peer connectivity within the Discord server.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1287482333678534656)** (5 messages): 

> - `3-bit and 5-bit support`
> - `Gemlite's efficiency`
> - `Pareto frontier of methods`
> - `Accuracy of Llama3 8B Instruct` 


- **Hackathon Success: 3-bit and 5-bit Support Added**: During a recent hackathon, a member successfully added support for **3-bit** and **5-bit** implementations, which took only **15 minutes** to achieve.
   - Details of the implementation can be found in the [GitHub repository](https://github.com/mobiusml/gemlite/tree/master/gemlite/triton_kernels/experimental).
- **Gemlite Makes N-bit Kernels Easier**: Another member expressed that with **Gemlite**, creating other **N-bit kernels** is likely much easier.
   - This sentiment reflects confidence in the tool's efficiency for developers working with low-bit matrices.
- **Exploring Pareto Frontier on Speedup and Accuracy**: A suggestion was made to visualize the **Pareto frontier** between different methods based on **speedup** and **accuracy**.
   - However, it was noted that each method is optimized for different **batch sizes** and **shapes**, complicating standardization.
- **Accuracy Data for Llama3 8B Instruct**: A member confirmed having data on the **accuracy** of **Llama3 8B Instruct** concerning various bitwidths.
   - This data could provide insights into performance trade-offs for different bit representations.



**Link mentioned**: <a href="https://github.com/mobiusml/gemlite/tree/master/gemlite/triton_kernels/experimental">gemlite/gemlite/triton_kernels/experimental at master · mobiusml/gemlite</a>: Simple and fast low-bit matmul kernels in CUDA / Triton - mobiusml/gemlite

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1287020121847627776)** (26 messages🔥): 

> - `Adding guards for tensor.is_inference()`
> - `FSDP parameter dtype issue`
> - `Using torch.compile for functions`
> - `CUDA memory allocation and tensor alignment`
> - `Triton kernel optimizations` 


- **Inquiring about guards for tensor.is_inference()**: A member asked if they should add new guards for `tensor.is_inference()` in the Dynamo guards implementation, specifically before a certain line of code in `guards.cpp`.
   - They mentioned triggering recompiles due to a lack of guards for `x.is_inference()` and provided a code example illustrating the situation.
- **FSDP struggles with mixed precision parameters**: A user experienced issues with `fully_shard()` on a model using a mix of **FP32** and **BF16** parameters, resulting in an `AssertionError`.
   - Discussion revolved around possible workarounds and the implications of separating `RMSNorm` layers for performance.
- **Exploring torch.compile with non-Module functions**: A member questioned if `torch.compile` could enhance the speed of functions outside of `nn.Module` instances, seeking examples.
   - Another member confirmed that `torch.compile` indeed works with functions, opening the discussion for further optimizations.
- **CUDA memory allocator alignment concerns**: A user sought examples to verify that not all tensor pointers in PyTorch are aligned despite the CUDA caching allocator’s minimum block size guarantees.
   - An example provided illustrated how a tensor slice could be misaligned, leading to discussions on the use of `tensor.contiguous()` for proper alignment.
- **Utilizing Triton kernel optimizations**: A member inquired about using vectorized access after making a tensor contiguous before passing it to a kernel.
   - It was confirmed that using `tensor.contiguous()` enables safe vectorized access, with a reference to Triton's specific annotations for optimization.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit">torch.compile, the missing manual</a>: torch.compile, the missing manual You are here because you want to use torch.compile to make your PyTorch model run faster. torch.compile is a complex and relatively new piece of software, and so you ...</li><li><a href="https://github.com/pytorch/pytorch/blob/e9bfbf78d5d89df1ec59cb82d7f78b85f9014a98/torch/csrc/dynamo/guards.cpp#L166">pytorch/torch/csrc/dynamo/guards.cpp at e9bfbf78d5d89df1ec59cb82d7f78b85f9014a98 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/csrc/dynamo/guards.cpp">pytorch/torch/csrc/dynamo/guards.cpp at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1287435936690475029)** (2 messages): 

> - `GPU MODE transition`
> - `CUDA MODE IRL meetup outcomes`
> - `Open source projects growth`
> - `Hackathon winners and projects`
> - `Community values and future vision` 


- **CUDA MODE transitions to GPU MODE**: The community formerly known as CUDA MODE, which started as a reading group, is now officially rebranded as **GPU MODE**, expanding beyond just CUDA programming.
   - This change reflects a broader vision of inclusivity and collaboration, welcoming those who share values of learning and social engagement.
- **Successful CUDA MODE IRL meetup**: The first IRL meetup gathered **150 hackers** from 10 am to midnight, resulting in over **40 projects** created in a single day.
   - Community feedback praised the event as a highly effective and connected gathering, solidifying its impact on collaborative innovation.
- **Growing open source projects ecosystem**: The GPU MODE community has expanded to over **9,000 members**, fostering the development of more than **10 open source performance projects** like torchao and Liger.
   - This growth showcases the community's commitment to building and sharing innovative tools in the field of GPU programming.
- **Hackathon winners showcase diverse projects**: Highlighting creativity, winners at the hackathon worked on projects such as **Flexible Flash Attention 3** and a **NCCL implementation in Triton**, with prizes totaling **$32.5K** in compute credits.
   - Such initiatives emphasize the community's intention to leverage their achievements for future contributions in open source.
- **Community values foster collaboration**: The GPU MODE community promotes a cozy and inclusive environment where members can learn, collaborate, and share experiences around GPU programming.
   - As stated, the focus is on deep focus work while balancing the social aspects of innovation, allowing members to enjoy the process together.



**Link mentioned**: <a href="https://x.com/swyx/status/1837577267259887702">Tweet from swyx (@swyx)</a>: CUDA MODE hackathon today!  Here&#39;s @karpathy on the 🏖️ origin story of llm.c, and what it hints at for the fast, simple, llm-compiled future of custom software.

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1287547131141423179)** (7 messages): 

> - `Bitonic Sort Optimization`
> - `CUDA Sorting Networks`
> - `Batch Matrix Multiplication with BLAS` 


- **Seeking Bitonic Sort Optimization on GPUs**: A user inquired about optimizing an array via **bitonic sort** on GPUs, expressing challenges with utilizing **shared memory** and achieving **global memory coalescing**.
   - Resources and assistance were requested as the user aimed to enhance their understanding of this sorting algorithm.
- **NVIDIA CUDA Samples Aid Sorting Efforts**: Another user provided a helpful link to the [NVIDIA CUDA samples sorting networks](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/sortingNetworks), offering valuable insights for optimization.
   - The original user expressed gratitude, acknowledging the resource as 'golden' for their needs.
- **Considerations for Bitonic Sort Performance**: A discussion unfolded regarding the performance limitations of **bitonic sort** for large sequences, referencing comments in the repository that highlight its **inefficiency** compared to sorting algorithms like **merge sort** or **radix sort**.
   - *One user noted their educational interest in understanding why bitonic sequences struggle with larger data sets,* hinting at the increasing recursive depth as a potential issue.
- **Batched Matrix Multiplication in BLAS**: A user sought information on performing **batched matrix multiplication** using **BLAS**, specifically for the shape (b, m, n) @ (b, n, k).
   - They questioned if looping over the batch dimension and launching a **gemm** for each element was the sole approach, pointing out an absence of a batched gemm in **OpenBLAS**.



**Link mentioned**: <a href="https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/sortingNetworks">cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks at master · NVIDIA/cuda-samples</a>: Samples for CUDA Developers which demonstrates features in CUDA Toolkit - NVIDIA/cuda-samples

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1287081839957114961)** (8 messages🔥): 

> - `NVTX for Custom Application Profiles`
> - `Stanford CS149 on Parallel Computing`
> - `GEMM Kernel Design Tutorial`
> - `LLM Compiler Insights from Karpathy`
> - `Speedy Llama 3.1 Model` 


- **Enhance Profiles with NVTX Annotations**: Using [NVIDIA Tools Extension (NVTX)](https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/), developers can now annotate timelines in tools like Nsight Systems to capture more than just CUDA API calls and GPU kernels.
   - This method simplifies the process for complex applications featuring deeply nested call graphs, and notes that the NVTX3 header-only library has been introduced.
- **Stanford's Class on Parallel Computing**: Stanford is offering [CS149: Parallel Computing](https://gfxcourses.stanford.edu/cs149/fall24) in Fall 2024, covering fundamental principles and programming techniques in parallel systems.
   - The course includes analyzing parallel program performance and managing task scheduling, set to be held at the NVIDIA Auditorium.
- **Deep Dive into GEMM Kernel Design**: Part 2 of the [GEMM tutorial series](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) focuses on optimizing memory operations for efficient operand tensor movement in GPU kernels.
   - It introduces pipelining strategies to enhance data transfer and processing efficiency on the NVIDIA Hopper architecture.
- **Karpathy's Take on LLM Compilers**: A recent [YouTube talk](https://www.youtube.com/watch?v=BmdOt6A6tHM) by Andrej Karpathy at the CUDA hackathon discusses the origins and future of LLM compilers, highlighting engaging insights.
   - Viewers noted his rapid speech, making the talk feel even faster, and included a link to the accompanying [llm.c GitHub repo](https://github.com/karpathy/llm.c).
- **Fastest Llama 3.1 Model Claims**: There are claims of the **fastest Llama 3.1-405b** model available, stirring interest in its performance metrics.
   - No additional details were provided, but the assertion suggests significant advancements in Llama's capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cloud.sambanova.ai/">SambaNova Cloud</a>: Preview AI-enabled Fastest Inference APIs in the world.</li><li><a href="https://gfxcourses.stanford.edu/cs149/fall24">no title found</a>: no description found</li><li><a href="https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/">CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining</a>: Welcome to Part 2 of our tutorial series on GEMM (GEneral Matrix Multiplication). In Part 1, we discussed the computational side of GEMM by going over WGMMA, which is the primitive instruction to m…</li><li><a href="https://www.youtube.com/watch?v=BmdOt6A6tHM">llm.c&#39;s Origin and the Future of LLM Compilers - Andrej Karpathy at CUDA MODE</a>: An informal capture from the CUDA mode hackathon today.https://github.com/karpathy/llm.c</li><li><a href="https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/">CUDA Pro Tip: Generate Custom Application Profile Timelines with NVTX | NVIDIA Technical Blog</a>: The last time you used the timeline feature in the NVIDIA Visual Profiler, Nsight VSE or the new Nsight Systems to analyze a complex application, you might have wished to see a bit more than just CUDA...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1287207967594123366)** (1 messages): 

> - `Hiring ML Performance Engineers`
> - `Fal Inference Engine`
> - `Generative Media Platform`
> - `Model Inference Speed` 


- **Fal.ai Seeking ML Performance Engineers**: Fal.ai is actively hiring **ML performance engineers** to enhance their generative media platform, offering **extremely competitive compensation** and remote options for exceptional hires.
   - Interested candidates can reach out directly or send their CVs to batuhan@fal.ai.
- **Fal Inference Engine delivers lightning-fast performance**: The **fal Inference Engine™** boasts the ability to run diffusion models up to **4x faster**, optimizing user experiences with real-time infrastructure.
   - This engine aims to prioritize both speed and quality, making it crucial for developers working in generative media.
- **Innovative features tailored for developers**: Fal.ai combines developer experience with robust AI capabilities through its dynamic pricing model, offering cost-effective scalability.
   - This ensures that users only pay for the computing power they consume, promoting efficient resource management.
- **Focus on foundational media models**: The company's goal revolves around building a top-notch generative media platform that handles various modalities like **text-to-image** and **text-to-video**.
   - They emphasize the need for talent that can help accelerate their efforts without compromising on quality.



**Link mentioned**: <a href="https://fal.ai">fal.ai | The generative media platform for developers</a>: fal.ai is the fastest way to run diffusion models with ready-to-use AI inference, training APIs, and UI Playgrounds

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1287828280581292107)** (1 messages): 

> - `Kernel Optimization`
> - `Matrix Multiplication Schemes`
> - `MLP Efficiency`
> - `Intermediate Result Utilization` 


- **Optimizing a Single Kernel for MLP**: There is a discussion on specific optimizations for handling a single kernel that performs **matrix multiplication**, an **elementwise non-linear function**, and another **matrix multiplication** to improve MLP efficiency.
   - The goal is to use the intermediate results from the first operations for the second multiplication without needing to return to **global memory**, which is currently unclear.
- **Challenges with Intermediate Data Handling**: Members are exploring if it's feasible to utilize intermediate results effectively without encountering performance bottlenecks caused by memory latency.
   - The conversation highlights the importance of efficient data flow when working with chained operations in **MLP architectures**.


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1286854040076161079)** (7 messages): 

> - `Speculative Decoding`
> - `TF Data vs. Grain`
> - `Grain Documentation Challenges`
> - `Epoch Training Issues` 


- **Interest in Speculative Decoding with JAX**: A member inquired if anyone was interested in implementing **speculative decoding** using **JAX**.
   - This suggests a growing interest in advancing techniques within the community.
- **TF Data works well for many**: Members mentioned that using **TFData** has proven effective for their applications.
   - One noted that while it's straightforward, **Grain** is recommended in the documentation for certain use cases.
- **Concerns about Grain's maturity**: A member expressed concerns about **Grain**, highlighting its **immaturity** and lack of sufficient documentation.
   - They find it challenging to utilize its full capabilities, particularly for **multiple workers** and **epoch training**.
- **Challenges with Epoch Training in Grain**: Another member shared difficulties in **epoch training** with **Grain**, noting that it continues until no data is left to iterate.
   - This lack of clear boundaries for epochs leads to complications, especially with ongoing documentation issues.
- **Community Struggle with Grain's Documentation**: Members agreed that while **Grain** is simple to start with, navigating its full potential remains tough due to its sparse documentation.
   - This limits community familiarity and makes finding answers to questions more difficult.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1287419975593758831)** (7 messages): 

> - `FP16 model loading`
> - `Model caching during benchmarking`
> - `Quantized model saving`
> - `AOTI and execution mode` 


- **Impressive FP16 Model Loading Implementation**: A user highlighted a **new ability to load any FP16 model** in various floating-point formats between 2 and 7 bits, with claims of impressive throughput and **accuracy preservation on par with FP8**.
   - *Do you want a non-standard FP6_E3M2, or FP7_E1M5? It should just work.*
- **Caching Model Loading in Benchmark Script**: A user inquired about caching model loading while using the benchmark script, to which another user confirmed that it is possible.
   - The suggested approach involves using the **save option** to save the quantized model and load it directly.
- **Model Compilation Requires Export**: Discussion pointed out that to **cache compilation**, the model must be exported, which is currently not supported by the benchmark script.
   - However, users indicated that it shouldn't be too complicated and referenced **the torchchat repo** for more details on executing similar models.



**Link mentioned**: <a href="https://x.com/AlpinDale/status/1837860256073822471">Tweet from Alpin (@AlpinDale)</a>: You can now load any FP16 model in any floating-point format you want, as long as it&#39;s between 2 and 7 bits. Do you want a non-standard FP6_E3M2, or FP7_E1M5? It should just work. The throughput i...

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1287508297091387444)** (2 messages): 

> - `CUDA MODE in Santa Cruz`
> - `Hotel theft incident` 


- **CUDA MODE Activated in Santa Cruz**: A member found a nice spot in **Santa Cruz** to get into full **CUDA MODE**.
   - This enthusiast seems excited about optimizing their computing capabilities in a suitable environment.
- **Hackathon Mishap: Theft at Hotel**: Another member reported that all their belongings were stolen from their hotel room while attending the **hackathon**.
   - They just completed filing a **police report** to address the theft incident.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1287090051599044800)** (5 messages): 

> - `Attendees at the Talks`
> - `Meetup in Toronto` 


- **Teknium confirms presence at the meetup**: Teknium responded affirmatively to a query, confirming their attendance at the event.
   - Another member noted, *'Come say hi after the talks,'* implying a casual engagement opportunity is available.
- **Toronto attendees connect**: Shagun expressed excitement about being in Toronto as well, creating a local connection among attendees.
   - This acknowledgment adds a personal touch to the event, enhancing community interactions.


  

---


### **GPU MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1287715058591207435)** (17 messages🔥): 

> - `CUDA/Torch Versions`
> - `CUDA error with Llama 3.1`
> - `GPU Compatibility`
> - `Bitblas Backend Functionality`
> - `Torch Compilation on GPUs` 


- **User seeks CUDA/Torch versions**: @strawberryjj inquired which **CUDA/Torch versions** are being used due to troubles with the `torchao` backend related to a [GitHub issue](https://github.com/mobiusml/hqq/issues/120).
   - The error indicated CUDA issues, prompting suggestions for upgrading dependencies.
- **CUDA error detailed with Llama 3.1**: The user detailed an error trace related to a CUDA issue when trying to run the **Llama 3.1 8B 4bit quantized model**. The error indicated problems with `torch.cat` and mentioned setting **CUDA_HOME** and **LD_LIBRARY_PATH** without success.
- **Tesla T4 GPU Limitations**: Conversations revealed that the **Tesla T4** GPU likely wouldn't work with various enhancements due to being a previous generation. It was advised that **Ampere** class GPUs are required for **fullgraph** support in `torch.compile`.
- **Bitblas Backend Recommendations**: It's suggested that trying the **bitblas backend** might yield better results, as it was reported to work on other GPU models. @mobicham noted past successes with **bitblas** on their **2080** GPU.
- **Triton and Torch Compilation Issues**: Discussion revealed that **torch.compile** struggles on older GPUs due to its foundation on **Triton**, which is not optimized for them. @strawberryjj confirmed that compilation indeed fails on their setup.



**Link mentioned**: <a href="https://github.com/mobiusml/hqq/issues/120">CUDA error when trying to use llama3.1 8B 4bit quantized model sample · Issue #120 · mobiusml/hqq</a>: Get model from https://huggingface.co/mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib HQQ installed according to instructions and tried running the sample given on HF site. After downloadin...

  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1286767976980152372)** (34 messages🔥): 

> - `Coordinating Event Attendance`
> - `LoRA and RAG Techniques`
> - `Micro-optimizations in llm.c`
> - `Creating a Chatbot for Student Services`
> - `GEMM Kernel Design` 


- **Coordination for Upcoming Event**: Members discussed coordinating efforts for an upcoming event on Saturday, with one member expressing regret for missing due to being in Chicago.
   - Another member thanked colleagues for their support, highlighting the collaborative efforts in the project.
- **Exploring LoRA and RAG Techniques**: A new member inquired about combining **LoRA** and **RAG** techniques in their model for a university chatbot, receiving positive feedback.
   - Discussions included insights on fine-tuning methods such as RIG and QLoRA, pointing out the need for clear evaluation metrics.
- **Micro-optimizations in llm.c**: A draft PR was shared for a `repkv_backward` work in progress along with another for a micro-optimization in `softmax_forward_kernel5` on the master branch.
   - Members expressed gratitude for the collaborative nature of the work and acknowledged contributions from others at the hackathon.
- **Creating a Chatbot for Student Services**: A member shared concerns about evaluating a university chatbot, suggesting that **hallucination** metrics could be critical.
   - Further discussion highlighted the importance of specific capabilities and user feedback to ensure effectiveness.
- **Insights on GEMM Kernel Design**: A link was provided to a tutorial on GEMM kernel design focusing on the memory aspects essential for GPU computations.
   - Members found the material valuable for enhancing their understanding of efficiently managing data buffers in GPU operations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/selfrag/selfrag_llama2_7b">selfrag/selfrag_llama2_7b · Hugging Face</a>: no description found</li><li><a href="https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/">CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining</a>: Welcome to Part 2 of our tutorial series on GEMM (GEneral Matrix Multiplication). In Part 1, we discussed the computational side of GEMM by going over WGMMA, which is the primitive instruction to m…</li><li><a href="https://github.com/karpathy/llm.c/pull/762">Micro optimization for `softmax_forward_kernel5` by insop · Pull Request #762 · karpathy/llm.c</a>: This branch includes a micro-optimization for softmax_forward_kernel5. Summary   use warpReduceMax in attention_forward.cu to use __shfl_down_sync to be consistent with the other kernels (reduce to...</li><li><a href="https://github.com/karpathy/llm.c/pull/764">DRAFT: Adding backward kernel for repkv on `llama3` branch (cudamode-irl) by insop · Pull Request #764 · karpathy/llm.c</a>: CC: @karpathy This is an WIP repkv backward kernel, started as a cudamode-irl project. Once the following work is done, will remove draft sign. This work was supported by ALEKSA (@gordicaleksa) , E...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1286864470005780560)** (41 messages🔥): 

> - `BitNet Performance`
> - `RMSNorm Implementation`
> - `Quantization Techniques`
> - `HQQ and Fine-Tuning`
> - `Performance of Large Models` 


- **BitNet may lack learnable params**: Concerns were raised about HF's PRs for BitNet possibly lacking learnable parameters for the new RMSNorm layers, potentially affecting overall performance.
   - The limited success in fine-tuning BitNet models for substantial token training raises questions regarding the configuration and implementation.
- **RMSNorm scaling impacts**: Tests showed that transferring column-wise scaling from pre-trained weights to RMSNorm actually led to worse performance due to difficulties quantizing activations to INT8.
   - This suggests that implementing effective scaling without degrading model quality remains a complex challenge.
- **Quantization may improve accuracy**: Discussion highlighted how quantization can offer better accuracy even without changing the number of parameters, particularly for large models like Llama3.
   - It was noted that using techniques like random projections could help manage outliers in activations.
- **HQQ and Large Language Models**: The HQQ method was noted for successfully quantizing Llama3-70B without needing much fine-tuning or calibration, showcasing the method's effectiveness in operational tasks.
   - It was emphasized that larger models generally do not require as much intervention during quantization compared to smaller counterparts.
- **Effective Training Strategies**: For training from scratch, there was consensus that no special tricks are needed and that models tend to perform adequately up to tested scales.
   - However, there is apprehension that unforeseen issues may arise with larger model sizes or extended training durations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">1bitLLM/bitnet_b1_58-3B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/OLMo-Bitnet-1B">NousResearch/OLMo-Bitnet-1B · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2105.03536">Pareto-Optimal Quantized ResNet Is Mostly 4-bit</a>: Quantization has become a popular technique to compress neural networks and reduce compute cost, but most prior work focuses on studying quantization without changing the network size. Many real-world...</li><li><a href="https://huggingface.co/blog/1_58_llm_extreme_quantization">Fine-tuning LLMs to 1.58bit: extreme quantization made easy</a>: no description found</li><li><a href="https://github.com/gau-nernst/quantized-training/blob/main/subclasses/bitnet.py">quantized-training/subclasses/bitnet.py at main · gau-nernst/quantized-training</a>: Explore training for quantized models. Contribute to gau-nernst/quantized-training development by creating an account on GitHub.</li><li><a href="https://huggingface.co/mobiuslabsgmbh/Llama-3.1-70b-instruct_4bitgs64_hqq">mobiuslabsgmbh/Llama-3.1-70b-instruct_4bitgs64_hqq · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/gemma2/modeling_gemma2.py#L111-L128">transformers/src/transformers/models/gemma2/modeling_gemma2.py at 78b2929c0554b79e0489b451ce4ece14d265ead2 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 messages): 

marksaroufim: https://x.com/shreyansh_26/status/1837157866509144492
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1287116750663716887)** (13 messages🔥): 

> - `Access Code Request`
> - `Event Invite`
> - `Sign-Up Issues` 


- **Access Code Chaos**: Many members experienced issues during registration, particularly with needing an **access code**. It's mentioned that this code might be obtained from a **Google Developer Relations** representative if the user's company has one.
   - One member plans to confirm with their **devrel** contact if the issue persists, showcasing a proactive approach.
- **Event Invite Networking**: A member reached out offering to DM an **event invite** to those interested, noting that spots might fill up quickly.
   - Another member expressed interest in attending, indicating alignment with their **ongoing project**.
- **Attending Event Discussions**: While many are excited about the event, one member mentioned having already traveled significantly lately, causing indecision about attending.
   - The community keeps a friendly tone, showing mutual support and consideration for each other's plans.


  

---


### **GPU MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1286766704214409310)** (169 messages🔥🔥): 

> - `Hackathon Team Formation`
> - `CUDA MODE Recap and Highlights`
> - `Project Submissions and Pitches`
> - `Talks and Recordings`
> - `Future Collaborations` 


- **Efforts to Form Teams for the Hackathon**: Participants discussed forming teams and collaborating at the hackathon, with recommendations to self-organize and communicate through designated channels.
   - Members also suggested using Uber for transportation due to limited parking availability at the venue.
- **CUDA MODE Event Receives Positive Feedback**: The hackathon showcased impressive projects, with participants feeling inspired by team dynamics and collaborations formed during the event.
   - Many expressed excitement about potentially highlighting unique projects such as running LLMs on mobile and the work of solo hackers.
- **Project Submission Process and Timeline**: Ten teams were selected for pitches, evaluated on their commercial utility and intellectual interest, with feedback emphasizing the importance of demos.
   - Participants were reminded to fill out a statement of intent before a deadline to ensure their projects were considered.
- **Talks Recorded and Available for Review**: Discussions indicated that talks during the event were recorded and would be made available on the YouTube channel after editing.
   - Attendees expressed gratitude for the efforts involved in capturing and sharing the event content, enhancing community engagement.
- **Post-Hackathon Community Engagement and Projects**: Members were encouraged to copy any important information from private channels as they were set to be cleared after the hackathon.
   - The community plans to maintain a dedicated channel for ongoing projects to support further collaboration and development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.facebook.com/codingcompetitions/hacker-cup/2024/practice-round/scoreboard?track=AI_OPEN_TRACK">no title found</a>: no description found</li><li><a href="https://tinyurl.com/cudamodehack">Submit Your Project</a>: Please use this form to share your project details with us. This information will help us understand your project better and make sure you are considered for the final judging. Please fill out the for...</li><li><a href="https://lambdalabs.com/cuda-mode-irl-cloud-credits">CUDA MODE IRL Lambda Cloud Credits</a>: Collect your cloud credits for the CUDA MODE IRL hackathon.</li><li><a href="https://maps.app.goo.gl/Q48tEdSu6fP5P8sH6?g_st=com.google.maps.preview.copy">Caffe Centro SP · San Francisco, California</a>: no description found</li><li><a href="http://github.com/cchan/tccl">GitHub - cchan/tccl: extensible collectives library in triton</a>: extensible collectives library in triton. Contribute to cchan/tccl development by creating an account on GitHub.</li><li><a href="https://youtu.be/BmdOt6A6tHM">llm.c&#39;s Origin and the Future of LLM Compilers - Andrej Karpathy at CUDA MODE</a>: An informal capture from the CUDA mode hackathon today.https://github.com/karpathy/llm.c</li><li><a href="https://forms.gle/CRtuyWCkviEGB65B6">fal CUDA MODE hat</a>: no description found</li><li><a href="https://github.com/AnswerDotAI/gpu.cpp/tree/dev">GitHub - AnswerDotAI/gpu.cpp at dev</a>: A lightweight library for portable low-level GPU computation using WebGPU.  - GitHub - AnswerDotAI/gpu.cpp at dev</li><li><a href="https://github.com/modal-labs/modal-examples">GitHub - modal-labs/modal-examples: Examples of programs built using Modal</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://github.com/vllm-project/vllm/pull/8713">[build] enable existing pytorch (for GH200, aarch64, nightly) by youkaichao · Pull Request #8713 · vllm-project/vllm</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/pull/136415">Implement nonzero_static in CUDA. by galv · Pull Request #136415 · pytorch/pytorch</a>: This adds CUDA functionality for nonzero_static, which was missing in #97417. This allows fully CUDA-based graphs to avoid data-dependent shapes. This is helpful for all sorts of reasons, one of wh...</li><li><a href="https://bit.ly/modal-credits.">Modal hackathon credits</a>: To claim your Modal credits, sign up for an account at https://modal.com/ first.  Then, let us know your username through this form.   For support, join the Modal Slack.  Here’s some examples to get s...</li><li><a href="https://github.com/charlesfrye/cuda-modal">GitHub - charlesfrye/cuda-modal: Enter CUDA MODE on Modal</a>: Enter CUDA MODE on Modal. Contribute to charlesfrye/cuda-modal development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/blob/ae02d663cdf493362699d2672ed7dc9019a7033b/test/inductor/test_flex_attention.py#L1938">pytorch/test/inductor/test_flex_attention.py at ae02d663cdf493362699d2672ed7dc9019a7033b · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1286764414703239188)** (78 messages🔥🔥): 

> - `KLDivLoss Kernel Issues`
> - `RMSNorm and LayerNorm Bugs`
> - `Cross-Entropy Comparison`
> - `Kernel Reduction Methods`
> - `Triton Grid Size Limitations` 


- **KLDivLoss kernel has calculation issues**: Members discussed that the **backward kernel's formula** for KLDiv might be incorrect, with potential problems identified in the forward kernel as well.
   - Another member noted that the kernel division is outside based on the reduction argument and suspected loop unrolling issues for larger vocab sizes.
- **RMSNorm and LayerNorm bugs persist**: Issues with the **RMSNorm** and **LayerNorm** were shared, specifically regarding incorrect output shapes and potential mismatch in program handling.
   - There was speculation that both had the same underlying problem due to how grids were managed in the Triton program.
- **Cross-Entropy provides consistent comparison**: A comparison was made between KLDivLoss and **Cross-Entropy**, noting how theirs is implemented to handle larger input dimensions effectively.
   - It's suggested that machine results to KLDiv could be aligned more similarly to Cross-Entropy to resolve the issues.
- **Kernel function reduction handling**: It was pointed out that the **reduction method** should not affect output shape, as all calculations occur within the kernel function.
   - A member highlighted a previous mismanagement of storing sum values for certain reduction methods that contributed to errors.
- **Addressing Triton's 64kb limitation**: Concerns were raised about Triton’s **64kb limit** when n_cols exceed certain counts, potentially limiting the kernel’s function.
   - A proposed solution involved increasing the grid size, similar to techniques used in the Cross-Entropy implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/linkedin/Liger-Kernel/pull/261">Fix assert_verbose_allclose bugs by Tcc0403 · Pull Request #261 · linkedin/Liger-Kernel</a>: Summary Fix #259 Adding more masks to cover all edge cases, including:  nan inf -inf  We should merge #262 before this PR to pass all tests. Testing Done  Hardware Type:   run make test to ensure c...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/255">RMSNorm aggregation by Tcc0403 · Pull Request #255 · linkedin/Liger-Kernel</a>: Summary Resolve #179 WIP: solving numerical stability issues for large hidden_size (4096) Testing Done  Hardware Type: RTX-3080  run make test to ensure correctness  run make checkstyle to ensure c...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/layer_norm.py#L189">Liger-Kernel/src/liger_kernel/ops/layer_norm.py at main · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499/src/liger_kernel/ops/kl_div.py#L121">Liger-Kernel/src/liger_kernel/ops/kl_div.py at ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499 · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499/src/liger_kernel/ops/cross_entropy.py#L205">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499 · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[irl-announcements](https://discord.com/channels/1189498204333543425/1285285792054247516/1287100094340141126)** (15 messages🔥): 

> - `Hackathon Kickoff`
> - `Compute Credits Information`
> - `Project Proposal Submission`
> - `Dinner and Networking`
> - `Preliminary Judging Update` 


- **Hackathon officially kicks off!**: The hackathon began with a warm welcome; participants were encouraged to grab a seat on any floor and lunch will be available at noon.
   - A color-coded sticker system indicates pre-assigned compute sponsors, assisting teams in their collaborations.
- **New Compute Credits Available**: Participants were provided with a breakdown on how to claim **Compute Credits** from several sponsors with specific codes to note.
   - For details on using **Modal**, members were directed to sign up and check provided examples to kickstart their projects.
- **Project Proposal Reminder**: A reminder was issued to submit project proposals by **5PM** today for consideration in the judging process; participants can access the [submission form](https://docs.google.com/forms/d/e/1FAIpQLSfK71QlvjICnDNoPMzbG6yAYLKKXLNhnzdeHj5davHJ4MuMjg/viewform) for details.
   - Submissions were highlighted as necessary for capturing project details and coordinating prize distribution.
- **Dinner and Socializing Opportunities**: Dinner is available on the **3rd floor** until **9PM**, allowing participants to relax and network while continuing their work on projects.
   - A last call for dinner reminded attendees to head up before the cut-off time.
- **Preliminary Judging in Progress**: Judges conducted preliminary discussions with teams, with over **40 entries** but only **10 teams** set to present under the spotlight.
   - Those unvisited by judges were asked to reply to ensure all teams receive feedback and support.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/forms/d/e/1FAIpQLSfK71QlvjICnDNoPMzbG6yAYLKKXLNhnzdeHj5davHJ4MuMjg/viewform">Submit Your Project</a>: Please use this form to share your project details with us. This information will help us understand your project better and make sure you are considered for the final judging. Please fill out the for...</li><li><a href="https://bit.ly/modal-credits)">Modal hackathon credits</a>: To claim your Modal credits, sign up for an account at https://modal.com/ first.  Then, let us know your username through this form.   For support, join the Modal Slack.  Here’s some examples to get s...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[irl-sponsor-qa](https://discord.com/channels/1189498204333543425/1285287931828768940/1287067781803806780)** (91 messages🔥🔥): 

> - `Early check-in recommendations`
> - `Compute credits and access issues`
> - `Node-specific support for Python packages`
> - `Multi-GPU options and Lab configurations`
> - `Closing event appreciation` 


- **Early Birds Get the Swag**: A participant recommended to 'show up early' to avoid crowds during the event.
   - This advice was aimed at those looking for sponsor swag on the third floor.
- **Credit Confusion Resolved**: Attendees clarified the process for obtaining modal credits after signing up, noting that no confirmation email is sent, but credits should appear in the account shortly after submission.
   - Participants confirmed an amount of **$1k** in credits was granted, and recent attendees verified receipt.
- **Help Install Python Packages Across Nodes**: Support was sought for installing `python3-poetry` across compute nodes, and it was confirmed that installation was successful using a virtual environment.
   - Users were guided to activate the environment with `source ~/venv-user3/bin/activate` before use.
- **Multi-GPU Queries and Limitations**: Inquiries were raised about the availability of multi-GPU Nebius VMs, revealing that presently, labs are limited to single GPU configurations.
   - However, it was mentioned that quota increases were made for users requesting more GPUs.
- **Closing Event and Expressing Gratitude**: The event concluded with appreciation expressed towards sponsors and support teams for their assistance throughout the day.
   - Participants were encouraged to celebrate the successful resolution of many challenges faced during the hackathon.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modal.com/docs/guide/cuda">Using CUDA on Modal</a>: Modal makes it easy to accelerate your workloads with datacenter-grade NVIDIA GPUs.</li><li><a href="https://nebius.ai/docs/compute/operations/vm-connect/ssh#vm-authorized-keys">no title found</a>: no description found</li><li><a href="https://github.com/charlesfrye/cuda-modal">GitHub - charlesfrye/cuda-modal: Enter CUDA MODE on Modal</a>: Enter CUDA MODE on Modal. Contribute to charlesfrye/cuda-modal development by creating an account on GitHub.</li><li><a href="https://github.com/charlesfrye/cuda-modal/blob/main/vscode_on_modal/vscode_server.py">cuda-modal/vscode_on_modal/vscode_server.py at main · charlesfrye/cuda-modal</a>: Enter CUDA MODE on Modal. Contribute to charlesfrye/cuda-modal development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1287441339893420113)** (7 messages): 

> - `Cross-platform GPU compute app`
> - `MPS versus WebGPU performance`
> - `WebGPU advancements needed`
> - `Metal performance for low intensity tasks` 


- **MPS Dominates macOS GPU Computing**: If targeting only **macOS**, **MPS** offers exceptional performance that surpasses WebGPU, achieving near **theoretical maximum** efficiency in benchmarks.
   - *There is a cost to portability that may not justify the performance trade-off depending on your priorities.*
- **WebGPU Performance Gaps**: Members expressed that WebGPU has not yet reached its ceiling compared to **MPS**, with ongoing experimentation revealing a significant performance gap.
   - A set of references, including [TVM's 2020 writeup](https://tvm.apache.org/2020/05/14/compiling-machine-learning-to-webassembly-and-webgpu), indicate that WebGPU can get **close to native GPU performance**.
- **Collaboration for Performance Optimization**: Discussion highlighted the need for comparing test kernels between MPS and WebGPU to assess performance suitability for specific applications.
   - A call for collaboration was made to optimize the **llm.c WebGPU implementation**, inviting interested parties to continue the discussion in designated channels.
- **Metal vs CPU for Low Intensity Work**: The question was raised about whether using **Metal** could yield performance benefits over the **CPU** for tasks lacking high arithmetic intensity.
   - This sparked interest in exploring the scenarios where Metal would or would not provide a significant speedup.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tvm.apache.org/2020/05/14/compiling-machine-learning-to-webassembly-and-webgpu">Compiling Machine Learning to WASM and WebGPU with Apache TVM</a>: no description found</li><li><a href="https://github.com/gpuweb/gpuweb/issues/4195">Cooperative matrix · Issue #4195 · gpuweb/gpuweb</a>: All major platform APIs have now released a similar extensions for cooperative matrix: Metal introduced simdgroup_matrix in MSL 3.1 HLSL has support in SM6.8 (currently experimental release) SPIR-V...</li><li><a href="https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark">WebGPU Embedding Benchmark - a Hugging Face Space by Xenova</a>: no description found</li><li><a href="https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark/discussions/30">Xenova/webgpu-embedding-benchmark · ⚡ WebGPU Benchmark Results (40.40x speedup)</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1287025227846520862)** (3 messages): 

> - `Cloud-based testing`
> - `Live webinar on advanced usage`
> - `Scaling parallel agents` 


- **Cloud-based Testing Available**: **Subscribers** can now test and run the service in the cloud without any local installations; **a smaller demo** is available on the landing page that can be tested with a Loom video.
   - This setup makes it easy for users to explore features quickly and efficiently.
- **Upcoming Webinar on Advanced Usage**: A live webinar on advanced usage is scheduled for **12pm EST**, focusing on **scaling to thousands of parallel agents and proxies**.
   - Participants can find more details by clicking the **Live tab** on the associated YouTube channel.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1286801967452262432)** (350 messages🔥🔥): 

> - `OpenRouter Model Changes`
> - `Upcoming AI Model Announcements`
> - `Model Performance Issues`
> - `OpenWebUI Integration`
> - `OpenAI Account Concerns` 


- **OpenRouter disables Middle-Out as Default**: OpenRouter has officially changed the default behavior for prompt handling by disabling the middle-out transform, impacting many users with established workflows.
   - Users expressed concern about this decision, emphasizing the importance of this feature for various frontend and backend systems.
- **Possible Release of New Anthropic Model**: Speculation arose regarding the launch of a new model from Anthropic, with hints from social media posts indicating a significant announcement was expected soon.
   - It was suggested that this announcement might coincide with a Google event and potentially offer extensive free token offers.
- **Performance Issues with Hermes 3 Models**: Users reported delays and stalling issues with various Hermes 3 models, experiencing wait times of over 10 minutes for responses from the API.
   - Concerns were raised about the overall performance of the models being slower than usual, leading users to explore alternative options.
- **Infermatic Models Generating Gibberish**: Some users noticed that Infermatic models were producing nonsensical output during their use, raising questions about the model performance.
   - Advices were given to check activity logs and adjust settings like temperature and penalties to mitigate these issues.
- **Concerns Over OpenAI Account Security**: Concerns were voiced regarding the security of the OpenAI newsroom Twitter account, which allegedly posted about token announcements while disabling comments.
   - This incident stirred anxiety among users about the potential for compromised accounts or misinformation spreading.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rowancheung/status/1838280020642676802">Tweet from Rowan Cheung (@rowancheung)</a>: I just finished up an exclusive interview going over a new, major AI model upgrade.  Can confirm, tomorrow will be a big day for developers.  Dropping the full conversation on X the second the embargo...</li><li><a href="https://huggingface.co/Sao10K/L3-8B-Stheno-v3.3-32K">Sao10K/L3-8B-Stheno-v3.3-32K · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: See how you&#x27;ve been using models on OpenRouter.</li><li><a href="https://community.sambanova.ai/t/rate-limits/321">Rate Limits</a>: SambaNova Cloud enforces rate limits on inference requests per model to ensure that developers are able to try the fastest inference on the best open source models.  Rate limits in the Free tier     M...</li><li><a href="https://status.anthropic.com/incidents/xts3kyr0nrx1">Elevated Errors on Claude 3.5 Sonnet</a>: no description found</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>: Transform data for model consumption</li><li><a href="https://openrouter.ai/models/anthropic/claude-3.5-sonnet/providers">Anthropic: Claude 3.5 Sonnet – Provider Status</a>: See provider status and make a load-balanced request to Anthropic: Claude 3.5 Sonnet - Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. S...</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage</li><li><a href="https://dubesor.de/Flappyo1mini0Shot">Dubesor LLM Benchmark table</a>: no description found</li><li><a href="https://dubesor.de/Game2MistralLarge0Shot">Dubesor LLM Benchmark table</a>: no description found</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:free">Hermes 3 405B Instruct (free) - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...</li><li><a href="https://github.com/OpenRouterTeam/open-webui/commit/89659df1fa10348f51b389a8fea27b67a71dec5d">add middle-out by default · OpenRouterTeam/open-webui@89659df</a>: no description found</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...</li><li><a href="https://dubesor.de/assets/shared/UIcompare/">Index of /assets/shared/UIcompare/</a>: no description found</li><li><a href="https://openrouter.ai/models/openai/o1-mini">o1-mini - API, Providers, Stats</a>: The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.  The o1 models are optimized for math, science, programming, and other STEM-related tas...</li><li><a href="https://github.com/hsiehjackson/RULER">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models? - hsiehjackson/RULER</li><li><a href="https://github.com/OpenRouterTeam/open-webui">GitHub - OpenRouterTeam/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)</a>: User-friendly WebUI for LLMs (Formerly Ollama WebUI) - OpenRouterTeam/open-webui</li><li><a href="https://github.com/open-webui/open-webui/blob/6b463164f4b129e0ce4bdc9008dd661214fe5eb5/backend/open_webui/apps/openai/main.py">open-webui/backend/open_webui/apps/openai/main.py at 6b463164f4b129e0ce4bdc9008dd661214fe5eb5 · open-webui/open-webui</a>: User-friendly WebUI for LLMs (Formerly Ollama WebUI) - open-webui/open-webui
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1287062592883261531)** (1 messages): 

> - `Private LLM Servers` 


- **Inquiry about Private LLM Servers**: A member inquired whether others are running **private LLM servers** themselves or if they are managed by a third party.
   - *Out of curiosity, are you running private llm servers yourself?*
- **Response to Request on Servers**: The conversation opened with a thank you for a request, signaling engagement in an ongoing discussion about LLM server management.
   - The member’s response suggested curiosity around the operational aspect of these servers.


  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1286782880638566400)** (211 messages🔥🔥): 

> - `Music Production AI`
> - `Bittensor and Nous Research`
> - `Byte-Level Architectures`
> - `RetNet Integration`
> - `World Sim API` 


- **Music Production AI struggles with Music Theory**: Amidst discussions about AI's role in music production, it was highlighted that large models struggle with basic music theory tasks, like transposing chords. A member has been experimenting with a feline AI focused on music, generating MIDI files and recommending synthesis methods.
   - Despite these efforts, users agreed that music notation remains a significant hindrance due to limited training examples.
- **Concerns over Bittensor's Practices**: There was a complaint about Bittensor seemingly replicating Nous Research’s distributed training algorithm without acknowledgment. This raised questions about the ethical considerations in the AI community regarding proper citation and recognition.
   - As discussions continued, some participants pointed out that the efforts in distributed training must prioritize innovation rather than merely increasing parameter counts.
- **Byte-Level Cognitive Models Discussion**: Ryunuck encouraged exploration of novel training methods to improve AI models, advocating for engagement and collaboration in research. The emphasis was placed on leveraging society as synthetic data janitors to effectively train models.
   - A suggestion was made to implement rhythmic patterns to improve model performance between epochs, indicating a shift towards innovative training strategies.
- **RetNet as an Additive to Transformers**: Ryunuck described RetNet not as a replacement but an additive layer enhancing transformers for retained sequence modeling. This approach allows for improved long-sequence capabilities while maintaining the integrity of existing transformer models.
   - The conversation delved into how models can be made more efficient and effective by integrating RetNet without losing the transformer’s architecture.
- **World Sim API Utilization**: Users discussed the Nous World Client and its functionality, which offers a few credits upon account creation. The conversation highlighted the cost-effectiveness of using the API for various tasks, though some technical glitches were noted.
   - There were calls for contributions to Nous Research to enhance the platform and its services, aiming to further engage users with the API.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">worldsim</a>: no description found</li><li><a href="https://x.com/LambdaAPI/status/1837121515600355771">Tweet from Lambda (@LambdaAPI)</a>: Broken code standing between you and Happy Hour? Plug @codegptAI into VSCode, set @NousResearch #Hermes3 through @lambdaAPI as provider for free & enjoy life https://bit.ly/4gvP48Q</li><li><a href="https://x.com/0xsamhogan/status/1837550399785783770?s=46&t=g-CqhQulOD52wCkbGxHcjQ">Tweet from Sam Hogan (@0xSamHogan)</a>: Bittensor already ripping off @NousResearch’s distributed training algorithm that was first published like 72 hours ago and not even mentioning them in the announcement tweet is entirely on brand.  Qu...</li><li><a href="https://worldsim.nousresearch.com/browser">worldsim</a>: no description found</li><li><a href="https://tenor.com/view/tldr-gif-25251690">Tldr GIF - Tldr - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/rao2z/status/1838245253171814419?s=46">Tweet from Subbarao Kambhampati (కంభంపాటి సుబ్బారావు) (@rao2z)</a>: A research note describing our evaluation of the planning capabilities of o1 🍓 is now on @arxiv https://arxiv.org/abs/2409.13373 (thanks to @karthikv792 & @kayastechly). As promised, here is a summar...</li><li><a href="https://github.com/holo-q/bytevibe/blob/master/src_models/retnphi_torch.py">bytevibe/src_models/retnphi_torch.py at master · holo-q/bytevibe</a>: Bytevibe is a research endeavour into the token-to-byte bootstrap, with a roadmap ending in artificial qualia (dubbed byte vibes) through the convergence of all byte-formats into the Holographic Qu...</li><li><a href="https://github.com/kyutai-labs/moshi">GitHub - kyutai-labs/moshi</a>: Contribute to kyutai-labs/moshi development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1286791616471765049)** (32 messages🔥): 

> - `RAG for Rules`
> - `CUDA OOM Issues with Llama 3.1`
> - `Fine-tuning Costs for Llama 3.1 70B`
> - `Experiences with Runpod` 


- **Exploring RAG for Rules in MUDs**: A member discussed challenges implementing RAG (Retrieval-Augmented Generation) for rule-based systems in a MUD, highlighting the need for effective rule retrieval methods.
   - Another suggested using an API to call rules from external tables to maintain consistency when responding to complex commands.
- **Training Llama 3.1 Sparks CUDA OOM Troubles**: A member reported encountering CUDA Out of Memory issues while training the **Llama 3.1 8B** model on 24 V100 GPUs, despite using mixed precision.
   - Discussion revealed potential misunderstandings surrounding model sharding across nodes, raising concerns about the effectiveness of DeepSpeed configurations.
- **Estimating Costs for Fine-Tuning Llama 3.1 70B**: One user sought advice on accurately pricing the fine-tuning process for a **Llama 3.1 70B** model, expressing frustration with varying estimates online.
   - Another suggested using [Together's API pricing](https://together.ai/pricing) as a useful benchmark for cost estimation.
- **Runpod Users Share Their Experiences**: Members shared positive experiences with **Runpod**, with one currently using it for a flux bot and another recommending its secure cloud offering.
   - However, there were concerns about potential issues in the community cloud, indicating a mixed reputation depending on the service tier.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1287268398560182283)** (1 messages): 

> - `Virtual Cell AI`
> - `HuatuoGPT Models`
> - `Chain of Diagnosis`
> - `LLMs in Clinical Trials`
> - `AI Cyber Threat Assessment` 


- **Building the Virtual Cell with AI**: This week’s featured paper, *How to Build the Virtual Cell with Artificial Intelligence: Priorities and Opportunities*, discusses the intersection of AI and cellular biology.
   - The authors include notable figures such as @_bunnech and @stephenquake, providing insights on priorities and opportunities in virtual cell development.
- **HuatuoGPT Models on the Rise**: Several **HuatuoGPT models** were highlighted, including **HuatuoGPT-II** and **HuatuoGPT-Vision**, aimed at enhancing medical LLM capabilities.
   - These models focus on **1-stage training** and **multimodal** applications for improved medical data processing.
- **Chain of Diagnosis Framework for Medical Agents**: The **Chain of Diagnosis (CoD)** methodology for medical agents was introduced, showcasing a structured approach to diagnostics.
   - This framework aims to improve predictive accuracy in medical AI applications.
- **LLMs Facilitating Clinical Trials**: Innovative uses of LLMs, such as generating clinical trial tables and correcting reports, are emerging in clinical research.
   - Noteworthy tools include **AlpaPICO**, designed to structure essential clinical trial information.
- **Addressing Cyber Threats in Healthcare AI**: A focus on **AI Cyber Threat Assessment** in the health sector highlights emerging risks faced by medical AI deployments.
   - This assessment underscores the urgency of developing robust security measures in medical AI frameworks.



**Link mentioned**: <a href="https://x.com/OpenlifesciAI/status/1837688406014300514">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last Week in Medical AI: Top Research Papers/Models 🏅(September 14  - September 21, 2024)  🏅 Medical AI Paper of the week How to Build the Virtual Cell with Artificial Intelligence: Priorities and O...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1286802146372747276)** (9 messages🔥): 

> - `AGI Predictions`
> - `Audio Processing Optimization`
> - `CoT Canvas Guide`
> - `Stanford CS149 Flash Attention` 


- **Shane Legg's AGI Timeline**: Google DeepMind co-founder Shane Legg predicted **AGI** will arrive around **2025**, with a **mean of 2028** if conditions remain stable, as noted in a [Reddit discussion](https://www.reddit.com/r/singularity/comments/1fla1tl/15_years_ago_google_deepmind_cofounder_shane_legg/). He anticipates a **proto-AGI** with basic abilities within the next 8 years.
   - Legg's consistent timelines since **2011** emphasize ongoing optimism tempered by caution, avoiding predictions tied to extreme events like nuclear wars.
- **Audio Processing at 24 kHz**: A process capable of handling **24 kHz audio** down to a **12.5 Hz representation** at a bandwidth of **1.1 kbps** is highlighted for its extreme optimization. Members speculated that the focus was initially on performance, allowing for further development by others.
   - The discussion indicates a balance between **audibility** and technical constraints, suggesting an intriguing approach to audio optimization.
- **CoT Canvas Guide Shared**: A comprehensive guide for **Chain of Thought (CoT) reasoning** was shared, aiming to clarify best practices and techniques for users via [this link](https://www.perplexity.ai/page/chain-of-thought-reasoning-via-22CYSxmhTMSFr1gJIXM4dg). It also referenced a related [Reddit thread](https://www.reddit.com/r/perplexity_ai/comments/1fm55ha/using_cot_canvas_via_the_complexity_browser/) for further insights.
   - The aim is to bolster understanding and application of CoT methodologies among users engaged in AI developments.
- **Stanford CS149 Implements Flash Attention**: In a surprising educational twist, **Stanford CS149** includes **implementing flash attention** as part of its homework assignments, as highlighted in a [Twitter post](https://x.com/Ethan_smith_20/status/1837690511953744146). This aligns educational curriculum closely with cutting-edge AI developments.
   - The initiative reflects growing academic interest in practical applications of advanced AI techniques within university settings.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vxtwitter.com/reach_vb/status/1836432149018288157">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/kimmonismus/status/1837283911870665080">Tweet from Chubby♨️ (@kimmonismus)</a>: 15 years ago, Google DeepMind co-founder Shane Legg predicted AGI in 2025. He&#39;s had roughly the same timelines since (mode 2025; mean 2028)  His last update in 2011:  &#34;I’ve decided to once aga...</li><li><a href="https://x.com/Ethan_smith_20/status/1837690511953744146">Tweet from Ethan (@Ethan_smith_20)</a>: stanford cs149 has implementing flash attention as a homework assignment
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1287268398560182283)** (1 messages): 

> - `Medical AI Paper of the Week`
> - `New Medical LLMs`
> - `Frameworks for Medical Diagnosis`
> - `Clinical Trials with LLMs`
> - `AI in Healthcare Ethics` 


- **Virtual Cells Revolutionize Medical AI**: The paper titled [How to Build the Virtual Cell with Artificial Intelligence: Priorities and Opportunities](https://x.com/OpenlifesciAI/status/1837688406014300514) highlights critical insights into creating virtual cells using AI.
   - This work is co-authored by prominent researchers like @_bunnech and @stephenquake, showcasing a promising approach to cellular modeling.
- **Introduction of New Medical LLMs**: Several new models were introduced, including **GP-GPT** for gene-phenotype mapping and **HuatuoGPT-Vision**, a multimodal medical LLM.
   - Also noted was **Apollo**, a lightweight multilingual medical LLM, which aims to enhance accessibility in medical AI applications.
- **Innovative Frameworks for Medical Diagnosis**: The Chain of Diagnosis (**CoD**) has been proposed as an effective framework for medical agents to enhance diagnostic efficiency.
   - Additional methodologies focus on addressing synthetic errors and aligning human knowledge for improved explanations in medical imaging.
- **LLMs Transform Clinical Trials**: New applications of LLMs in clinical trials include generating trial tables and correcting clinical reports, notably through **AlpaPICO** for PICO frames.
   - These advancements aim to streamline clinical processes and improve the quality of medical documentation.
- **AI Cyber Threats in Healthcare**: Discussion on **AI Cyber Threat Assessment in the Health Sector** emphasizes the growing concerns related to cybersecurity within healthcare.
   - As AI technologies advance, addressing these vulnerabilities becomes increasingly crucial for safe medical practices.



**Link mentioned**: <a href="https://x.com/OpenlifesciAI/status/1837688406014300514">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last Week in Medical AI: Top Research Papers/Models 🏅(September 14  - September 21, 2024)  🏅 Medical AI Paper of the week How to Build the Virtual Cell with Artificial Intelligence: Priorities and O...

  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1287059116480532504)** (17 messages🔥): 

> - `RL Environment for Reasoning`
> - `Multi-Agent Interactions`
> - `Fine-Tuning Datasets`
> - `Comparison with GPT Models`
> - `AI Sentience Discussion` 


- **Exploring RL Environments for Reasoning**: There is a discussion on whether efforts are underway to create an RL environment suited for training reasoning tasks, focusing on a model's ability to generate unrestricted chain-of-thought answers.
   - One member emphasized the need for a diverse set of RL environments, stating that successful training resembles how open source fine-tuning utilizes a good selection of datasets.
- **Multi-Agent Models Communicate**: Members speculated about the architecture used to solve problems, indicating there may be multiple models interacting to address single prompts.
   - This interaction could possibly involve models discussing and collaborating, although specifics remain unclear.
- **GPTs versus OAI's Closed Source Models**: A member pointed out that the models being developed by OAI are significantly different from GPTs, suggesting they are rebuilt from the ground up and remain closed source.
   - Despite speculations around these models, there's frustration over the lack of transparency regarding their inner workings.
- **Fine-Tuning Techniques for RL**: It was mentioned that various algorithms, such as DPO and PPO, could be applied to the selected RL environments to enhance the training process.
   - The same member suggested that building a solid selection of RL environments is crucial for effective chain-of-thought training.
- **Excitement Over AI's Future**: One member expressed enthusiasm about the advancements in AI reasoning capabilities, suggesting they foresee a rapid evolution toward AGI.
   - In a passionate message, they highlighted optimism for a future where humans and AI coexist, declaring it a potential golden age of technology.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1286936017676275808)** (206 messages🔥🔥): 

> - `AI and Mental Health`
> - `Bias in AI Systems`
> - `Intuition and Psychology`
> - `AI in Healthcare Compliance`
> - `Learning Programming with Python` 


- **AI's Role in Mental Health Support**: Members discussed how people with mental health issues may prefer talking to chatbots due to stigma, making ethical AI usage crucial in healthcare.
   - It was noted that while AI can assist in mental health diagnostics, it should not replace professional care and needs to comply with data privacy regulations.
- **Understanding Bias in AI Systems**: The group highlighted the need to teach about motivated reasoning and confirmation bias to improve internet usage and critical thinking.
   - Members agreed that AI recommendations should be grounded in scientific advice, with a strong emphasis on ethical standards.
- **Intuition, Psychology, and Dialectics**: One member shared their thesis on intuition, revealing satisfaction in finding scientific validation for their ideas years later.
   - The conversation touched on how religious perspectives often see intuition as a divine voice, contrasting with scientific interpretations.
- **Exploring AI in Healthcare Compliance**: Members discussed the significance of AI in predictive medicine while addressing the complexities of compliance with patient data regulations.
   - They emphasized the importance of anonymization techniques to protect patient information while utilizing AI tools.
- **Learning Python for AI and Engineering**: A new member expressed interest in learning Python for AI applications, receiving encouragement and advice from others in the community.
   - Recommendations included taking on projects and making use of online resources for self-improvement as they navigate their learning journey.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.datacamp.com/blog/attention-mechanism-in-llms-intuition">What is Attention and Why Do LLMs and Transformers Need It?</a>: In this article, we focus on building an intuitive understanding of attention. The attention mechanism was introduced in the “Attention Is All You Need” paper. It is the key element in the transformer...</li><li><a href="https://youtu.be/nwFmmhSmCcM">Why You Have Never Thought Alone</a>: Have you ever thought you knew how something works only to realize, when asked to explain it, that your understanding was a lot shallower than you thought? W...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1286936156386234368)** (38 messages🔥): 

> - `Cohere's Research Focus`
> - `Performance Issues with Azure SDK`
> - `Cohere Reranker API Hosting`
> - `Hackathon Sponsorship Requests`
> - `Connectors Compatibility in APIs` 


- **Cohere's Research Focus includes many areas**: Cohere works on various topics including language models, efficiency, safety, multilingual capabilities, RL, and AI policy, with resources available on their [research papers page](https://cohere.com/research/papers).
- **Performance Issues with Azure SDK**: A user reported that their implementation of the Command R+ model using the Azure SDK underperformed significantly compared to using the Cohere SDK, leading to frequent hallucinations in responses.
   - Despite updating the Azure implementation to a lower temperature and removing certain parameters, the issues persisted.
- **Cohere Reranker API is hosted across multiple locations**: Cohere's Reranker API endpoint can be hosted on their platform or other cloud providers, as indicated by a team member.
   - They clarified that they have servers in multiple locations, rather than being limited to a US-based server.
- **Hackathon Sponsorships Currently Unavailable**: A user inquired about potential sponsorship for a hackathon, which prompted a staff member to direct them to a specific contact.
   - However, it was noted that Cohere is not currently accepting sponsorship requests.
- **Connectors Compatibility in APIs**: It was mentioned that the current connectors in Cohere's APIs may only be compatible with their native platform.
   - Users were encouraged to explore options like the Brave Search API as an alternative solution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/research/papers">Missing title</a>: Research papers written by Cohere For AI and Technical Staff at Cohere</li><li><a href="https://www.lasillavacia.com/silla-nacional/directiva-sobre-la-protesta-muestra-mas-sesgos-de-la-derecha-que-de-la-fiscal-camargo/">Directiva sobre la protesta muestra más sesgos de la derecha que de la fiscal Camargo</a>: Es una guía con normas que ya existen para fiscales. Generó críticas sin sustento a la nueva fiscal general por, supuestamente, ayudar al gobierno Petro.</li><li><a href="https://docs.cohere.com/reference/chat">Chat — Cohere</a>: Generates a text response to a user message. To learn how to use the Chat API and RAG follow our  Text Generation guides .</li><li><a href="https://github.com/luillyfe/la-silla-vacia">GitHub - luillyfe/la-silla-vacia</a>: Contribute to luillyfe/la-silla-vacia development by creating an account on GitHub.</li><li><a href="https://models.inference.ai.azure.com",">no title found</a>: no description found</li><li><a href="https://github.com/luillyfe/la-silla-vacia/blob/main/app/copilot/azureInference.ts">la-silla-vacia/app/copilot/azureInference.ts at main · luillyfe/la-silla-vacia</a>: Contribute to luillyfe/la-silla-vacia development by creating an account on GitHub.</li><li><a href="https://github.com/luillyfe/la-silla-vacia/blob/main/app/copilot/cohere.ts">la-silla-vacia/app/copilot/cohere.ts at main · luillyfe/la-silla-vacia</a>: Contribute to luillyfe/la-silla-vacia development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1287235077146673247)** (5 messages): 

> - `Cohere API geolocation restrictions`
> - `Embedding call changes`
> - `Support inquiry process` 


- **Cohere API geolocation restrictions confirmed**: It's confirmed that **Cohere does geolock**, which might be causing API access issues when migrating servers to different locations like Finland or Germany.
   - *Email support@cohere.com* for assistance in resolving these geolocation access permissions.
- **Embedding call requires 'embedding_types' parameter now**: A user reported their **embedding call** started erroring with '`embedding_types parameter is required`', despite the documentation stating it was optional previously.
   - This change in behavior was questioned, prompting clarification from the Cohere team.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1287130812902281237)** (1 messages): 

> - `Cohere AI Chatbot`
> - `AI Telegram Bot Repository` 


- **AI-Telegram-Chatbot with Cohere AI**: A member shared their GitHub repository for an [AI-Telegram-Chatbot](https://github.com/derssen/AI-Telegram-Chatbot) that utilizes **Cohere AI** to generate intelligent responses to user messages.
   - The project description highlights that it's a free bot aiming to enhance user interaction through **AI-driven responses**.
- **Surprise Collaboration on Cohere**: A member expressed excitement about not being the only one considering a **repository using Cohere** for chat applications.
   - This enthusiasm reflects a growing interest in leveraging **Cohere technologies** for practical implementations.



**Link mentioned**: <a href="https://github.com/derssen/AI-Telegram-Chatbot">GitHub - derssen/AI-Telegram-Chatbot: A free Telegram chatbot that uses Cohere AI to generate intelligent responses to user messages.</a>: A free Telegram chatbot that uses Cohere AI to generate intelligent responses to user messages. - derssen/AI-Telegram-Chatbot

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1286806479390638172)** (111 messages🔥🔥): 

> - `Magic product feedback`
> - `Mojo and Python integration`
> - `C compatibility of Mojo`
> - `Bit packing and struct sizes`
> - `Upcoming community meeting` 


- **Last Call for Magic Feedback Chats**: A reminder for users to join a quick 30-minute call to provide feedback on **Magic**, especially from those who haven't used it yet. Participants will receive exclusive swag for their contributions, with booking available [here](https://modul.ar/user-feedback).
- **Mojo's Compatibility with Python Libraries**: Discussion centered on the use of **Python** libraries in **Mojo**, with members debating how Mojo threads handle the GIL and whether new interpreters can be created for parallel execution. Concerns were raised regarding potential GIL limitations when using Python libraries.
   - The conversation highlighted that while Mojo can integrate with Python libraries, it may rely on CPython, thus inheriting some of its performance limitations.
- **Bit Packing and Struct Sizes in Mojo**: Members discussed the implications of data types in **Mojo**, specifically focusing on struct sizes and bit packing. The absence of native bit packing support in Mojo was addressed, with alternative solutions like manual packing and variable width types being suggested.
   - Concerns about struct alignment impacting performance were raised, alongside a note that **LLVM** can potentially handle varying bit widths from a performance perspective.
- **C Compatibility and Field Reordering in Mojo**: The group debated the potential for field reordering in structs to optimize memory usage, with a strong emphasis on maintaining **C compatibility**. Suggestions were made for explicit decorators to enable more flexible struct definitions.
   - It was noted that despite the desire for flexibility, compatibility with C remains essential as a guiding principle for Mojo's design.
- **Upcoming Community Meeting Announcement**: A notification was shared regarding the **Community Meeting** being rescheduled to Monday, September 30th at 10 AM PT. Attendees were encouraged to add their topics to the [Google doc](https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit?usp=sharing) to facilitate planning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modul.ar/user-feedback">Zoom Scheduler</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/python/_cpython.mojo">mojo/stdlib/src/python/_cpython.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/tauri-apps/tauri">GitHub - tauri-apps/tauri: Build smaller, faster, and more secure desktop applications with a web frontend.</a>: Build smaller, faster, and more secure desktop applications with a web frontend. - tauri-apps/tauri</li><li><a href="https://modul.ar/community-meeting">Google Calendar - Sign in to Access &amp; Edit Your Schedule</a>: no description found</li><li><a href="https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit?usp=sharing)">[Public] MAX + Mojo Community Meeting</a>: MAX + Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1286800711245627493)** (114 messages🔥🔥): 

> - `Mojo and Python Integration`
> - `Mojo's Class System`
> - `Compiler and Performance Issues`
> - `Traits and Generics in Mojo`
> - `General Purpose Nature of Mojo` 


- **Discussion on Mojo's Class System**: There is ongoing debate about whether Mojo should adopt Python-like dynamic classes, with some advocates emphasizing the need for a stricter class system akin to C++ or Swift.
   - Some users express satisfaction with the current struct system, focusing on traits and enums instead, indicating a desire for lower-level programming capabilities.
- **Mojo and Python Integration Challenges**: Users are interested in streamlined integration with Python, suggesting a system where Python classes can be created directly in Mojo files.
   - However, concerns arise regarding the dynamic behavior of Python classes conflicting with the performance goals of Mojo.
- **Compiler and Performance Issues**: Messages highlighted compiler issues, such as segfaults linked to storing functions in a dictionary and dealing with traits conformance.
   - Some users suspect these issues might point to bugs within the current implementation of Mojo.
- **Traits and Generics in Mojo**: There are discussions regarding the implementation and constraints of traits, including the issue of using output slots affecting trait conformance.
   - Some users are exploring the use of generics and trait systems, expressing excitement about potential developments in these areas.
- **General Purpose Nature of Mojo**: Users are generally optimistic about Mojo evolving into a full general-purpose language, emphasizing its capabilities beyond just AI applications.
   - The integration with systems like MAX is seen as a pathway to broader usability while maintaining performance advantages.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.rs/un_algebra/latest/un_algebra/">un_algebra - Rust</a>: no description found</li><li><a href="https://docs.modular.com/max/tutorials/get-started-with-max-graph">Get started with MAX Graph | Modular Docs</a>: Learn how to build a model graph with our Mojo API for inference with MAX Engine.</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/mojo-and-dynamism.md">mojo/proposals/mojo-and-dynamism.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/max/blob/nightly/tutorials/max-graph-api/mojoproject.toml#L16.">max/tutorials/max-graph-api/mojoproject.toml at nightly · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - modularml/max</li><li><a href="https://m.youtube.com/watch?v=sPiWg5jSoZI&pp=ygUYRGF2aWQgbWV0YWNsYXNzZXMgcHl0aG9u">Python 3 Metaprogramming</a>: David BeazleySome of the most significant changes in Python 3 are related to metaprogramming.  In this tutorial, I&#39;ll cover decorators, class decorators, des...</li><li><a href="https://github.com/modularml/mojo/issues/3534">[Historical Discussion] Mojo and Dynamism · Issue #3534 · modularml/mojo</a>: Discussed in #466 Originally posted by Mogball July 20, 2023 Mojo has the lofty goal of being a simple, powerful, and easy-to-use language like Python but with features that allow programmers to re...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1287360425725530238)** (1 messages): 

> - `MAX custom ops` 


- **What happened to MAX custom ops?**: A query was raised about the status of **MAX custom ops** on the [Modular documentation site](https://docs.modular.com/max/api/mojo/register/register/op).
   - The inquiry indicates a need for clarity regarding recent changes or removals related to custom operations within the MAX framework.
- **Community concern over MAX functionality**: Members expressed concern regarding the current functionality of **MAX**, particularly in relation to custom operations.
   - Discussions highlight a desire for updated documentation and guidance on utilizing MAX efficiently.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1286813967255339029)** (118 messages🔥🔥): 

> - `LM Studio Issues`
> - `Model Loading Errors`
> - `Interoperability with Other Tools`
> - `ROCm Support`
> - `Image Generation Model Support` 


- **LM Studio Model Loading Challenges**: Users reported difficulties loading models after updating to newer versions of LM Studio, with issues particularly noted after the CUDA Llama.cpp v1.1.9 update.
   - Several users shared their fixes, including clearing cache directories and rolling back versions to restore functionality.
- **Unsupported Model Architectures**: Discussion highlighted that LM Studio does not support image generation models, resulting in errors like 'unknown model architecture' when trying to load Flux.
   - It was clarified that models like Flux and stablediffusion are intended for other platforms, not LM Studio.
- **Using LM Studio on Air-Gapped Computers**: Users inquired about the feasibility of using LM Studio on air-gapped computers without internet access.
   - It was confirmed that installation and model files can be transferred via external drives, but the initial setup must be done on a connected machine.
- **ROCm Support in LM Studio**: Questions arose regarding the availability of a separate ROCm version for LM Studio, specifically whether to download the latest release.
   - Users were informed that the newest version now automatically detects ROCm, simplifying the installation process.
- **Performance Optimizations and Usage Tips**: Users discussed strategies for optimizing LM Studio performance, with some noting the impact of managing active chats on memory usage.
   - Tips were shared on controlling model thread usage and ensuring higher quality output with dual-model systems.



**Link mentioned**: <a href="https://github.com/vllm-project/vllm?tab=readme-ov-file">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm

  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1286873691677786132)** (93 messages🔥🔥): 

> - `DDR6 release timeline`
> - `Performance specs of RTX 4090`
> - `Comparative benchmarking of GPUs`
> - `AMD vs Nvidia multi-GPU issues`
> - `Model loading capabilities in LM Studio` 


- **DDR6 Release Still Uncertain**: Concerns were raised about the availability and ratification timeline of **DDR6**, suggesting that adoption won't be seen until late next year.
   - *Speculation around consumer hardware usage continues*, as many are awaiting confirmation on DDR6 specifications.
- **RTX 4090 Performance Under Scrutiny**: Discussion revealed mixed results regarding **RTX 4090**, with some achieving less than **20t/s** running **70B Q4**, while other claims of **60t/s** were disputed.
   - Data from various users pointed towards inconsistencies in performance measurements across different setups, particularly on the **70B Q2 model**.
- **AMD-Multi GPU Performance Issues**: Members queried the viability of multi-GPU setups with **AMD**, noting that while **Nvidia** setups have favorable reports, **AMD** configurations lack similar support.
   - Worries were raised about **VRAM limitations** impacting performance, particularly in relation to running **large models like 70B**.
- **Insights on Benchmarking between NVIDIA and AMD**: Comparative results from **AMD 7900 XTX** and **RTX 4090** showcased how **tensor cores** in Nvidia GPUs may provide around **50% faster** processing speeds in certain scenarios.
   - Concerns about memory overflow and RAM utilization were highlighted, especially when exceeding **24GB VRAM** limits during model execution.
- **LM Studio Versions Affect Results**: Users noted significant differences in performance when switching between version **1.10 and 1.11** of LM Studio, reporting around **10% improvement**.
   - Testing various models revealed that larger models may still result in memory spillover into RAM, affecting overall performance despite possible improvements.



**Link mentioned**: <a href="https://old.reddit.com/r/LocalLLaMA/comments/1fljyly/llama_31_70b_at_60_toks_on_rtx_4090_iq2_xs/">Llama 3.1 70b at 60 tok/s on RTX 4090 (IQ2_XS)</a>: Setup GPU: 1 x RTX 4090 (24 GB VRAM) CPU: Xeon® E5-2695 v3 (16 cores) RAM: 64 GB RAM Running PyTorch 2.2.0 + CUDA 12.1 Model:...

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1286782270744690699)** (200 messages🔥🔥): 

> - `Stable Diffusion Features`
> - `FLUX Models`
> - `Training LoRAs`
> - `Inpainting Techniques`
> - `Consistent Generations with AI` 


- **Exploring Stable Diffusion Features**: Users discussed various aspects of Stable Diffusion, including the functionality of Dalle3 and the limitations of Flux in terms of VRAM utilization.
   - The conversation also touched on the use of specific tools, like boorutag autocompletion for prompt enhancements.
- **FLUX Model Utilization and Issues**: Members shared their experiences with FLUX models, highlighting the challenges of using LoRAs and the management of VRAM while generating images.
   - Users were advised on techniques to optimize their models, including keeping text encoders on DRAM.
- **Training LoRAs for Character Consistency**: Participants discussed the need for specific prompts and training LoRAs to ensure consistent character generation in projects like comics.
   - They mentioned using IP adapters for better character coherence while generating images.
- **Inpainting for Image Completion**: Users sought advice on inpainting techniques to fill in missing parts of images while maintaining style and coherence.
   - Suggestions included using tools like Fooocus and RuinedFooocus UI to enhance the inpainting process.
- **Maintaining Consistency in AI Art**: Conversations centered around creating consistent generations in AI art by using the same prompts and settings.
   - The importance of keeping consistent seeds and settings was emphasized along with tools that help maintain style across images.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LikeToasters/status/1836632745075736913">Tweet from ILikeToasters (@LikeToasters)</a>: Video on how I made my Miniature People LoRA. I discuss the decisions I made and the steps to get there. I do not have many answers but this may help people figure out how to do things. I used Flux Gy...</li><li><a href="https://gist.github.com/kohya-ss/3f774da220df102548093a7abc8538ed">SDXLで高解像度での構図の破綻を軽減する</a>: SDXLで高解像度での構図の破綻を軽減する. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://blackforestlabs.ai/">Black Forest Labs &#x2d; Frontier AI Lab</a>: Amazing AI models from the Black Forest.</li><li><a href="https://github.com/black-forest-labs/bfl-comfy-nodes">GitHub - black-forest-labs/bfl-comfy-nodes</a>: Contribute to black-forest-labs/bfl-comfy-nodes development by creating an account on GitHub.</li><li><a href="https://github.com/WadRex/RVCompact">GitHub - WadRex/RVCompact: Fully Portable RVC: Voice Cloning Software</a>: Fully Portable RVC: Voice Cloning Software. Contribute to WadRex/RVCompact development by creating an account on GitHub.</li><li><a href="https://github.com/yhyun225/DiffuseHigh">GitHub - yhyun225/DiffuseHigh: Official implementation of DiffuseHigh, *Younghyun Kim, *Geunmin Hwang, Junyu Zhang, Eunbyung Park.</a>: Official implementation of DiffuseHigh, *Younghyun Kim, *Geunmin Hwang, Junyu Zhang, Eunbyung Park.  - GitHub - yhyun225/DiffuseHigh: Official implementation of DiffuseHigh, *Younghyun Kim, *Geunmi...</li><li><a href="https://github.com/cocktailpeanut/fluxgym">GitHub - cocktailpeanut/fluxgym: Dead simple FLUX LoRA training UI with LOW VRAM support</a>: Dead simple FLUX LoRA training UI with LOW VRAM support - cocktailpeanut/fluxgym
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1286779685975363627)** (176 messages🔥🔥): 

> - `o1-mini performance in creative writing`
> - `Embedding storage solutions for AI`
> - `AI tools for analyzing PDFs`
> - `Comparative analysis of AI chatbot models`
> - `Challenges in using AI for nuanced poetry` 


- **o1-mini struggles with creative writing**: Users noted that o1-mini often defaults to clichés and predictable structures when asked to write poems, making it hard to achieve the desired depth and originality.
   - The consensus is that Claude Opus 3 might be better for nuanced writing tasks, although suggestions for improved prompt specificity are recommended for o1-mini.
- **Best practices for storing embeddings**: One user discussed storing embeddings for a collection of texts (12-13k) and explored various options for efficient storage and clustering solutions.
   - S3 was mentioned as a potential option, alongside suggestions that using a vector store managed by OpenAI could streamline the clustering process.
- **AI tools for processing PDFs**: A user sought tools that can analyze PDF files and convert images or graphics into text for inclusion in an AI knowledge base.
   - Discussion revealed that many RAG solutions support PDF integration, but converting images to text remains an area needing further advancements, potentially with multimodal models.
- **Comparative analysis of AI chatbot models**: Participants discussed the differences between AI models, particularly focusing on performance in creative writing, with o1-mini often falling short compared to Claude Opus 3.
   - Feedback highlighted the variability in performance depending on how well the models are prompted, with interest in future models that may offer better creativity.
- **Reflections on nuanced poetry creation**: Users expressed challenges in guiding AI to produce less clichéd and more nuanced poetry, suggesting that prompts must be highly specific to improve outcomes.
   - Collaboration with the AI, including offering feedback and examples, is recommended to refine the models' output towards the user's preferences for poetic creativity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google/">NotebookLM | Note Taking &amp; Research Assistant Powered by AI</a>: Use the power of AI for quick summarization and note taking, NotebookLM is your powerful virtual research assistant rooted in information you can trust.</li><li><a href="https://github.com/ack-sec/toyberry">GitHub - ack-sec/toyberry: Toy implementation of Strawberry</a>: Toy implementation of Strawberry . Contribute to ack-sec/toyberry development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1287009900949667860)** (12 messages🔥): 

> - `gpt-o1-preview quota for enterprise`
> - `Appealing custom GPT removal`
> - `Using gpt 4o for advanced math`
> - `ChatGPT issues in Firefox` 


- **gpt-o1-preview quota info shared**: A member requested a link regarding the **gpt-o1-preview quota** for enterprise accounts, and another member responded with a [rate limits guide](https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-five) suggesting enterprise limits might align with tier 5.
   - However, the member acknowledged that this was speculative in nature.
- **Issues appealing custom GPT removal**: A user expressed frustration about submitting an appeal for the removal of their custom GPT, noting that the submit button was unresponsive.
   - Another member advised reaching out to [OpenAI Help](https://help.openai.com) for assistance.
- **Using gpt 4o for math analysis clarified**: Members debated whether using **gpt 4o** for advanced math would count against the **2 free data analysis per day** limit, with one stating it likely does since it uses Python.
   - Another suggested a workaround by using **an IDE** to run Python code, claiming it could solve math problems without direct limits tied to the model.
- **ChatGPT not working in Firefox**: A user reported that ChatGPT has not been functioning in **Firefox** for a while and sought solutions from the community.
   - The discussion did not offer specific resolutions for the browser issue.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1286812712881623174)** (4 messages): 

> - `Prompt Sharing`
> - `Anti-KI Detection Techniques` 


- **Useful Prompt Shared**: A member shared a [prompt](https://chatgpt.com/g/g-ssHSsvECZ-guidegpt) they created previously, emphasizing its continued usefulness in generating responses.
   - *This prompt remains popular among users for its effectiveness in guiding interactions.*
- **Request for Anti-KI Detection Prompts**: Another member inquired about effective anti-KI detection prompts to bypass existing protections.
   - *The request indicates ongoing interest in strategies to navigate restrictions on AI-generated content.*


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1286812712881623174)** (4 messages): 

> - `Prompt Sharing`
> - `Anti-KI Detection Prompts` 


- **Useful Prompt from Mandalorian**: A member shared a [useful prompt guide](https://chatgpt.com/g/g-ssHSsvECZ-guidegpt) they wrote a while ago, stating that they still find it very effective.
   - This guide aims to help users optimize their interactions with the ChatGPT platform.
- **Request for Anti-KI Detection Prompt**: A member inquired if anyone has a strong anti-KI detection prompt to bypass AI protections.
   - They humorously noted their request with a smiley face, suggesting a lighthearted approach to the topic.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1286786390872424459)** (51 messages🔥): 

> - `OpenAI phone confirmation`
> - `AI SDK 3.4 release`
> - `Edits in academic literature review tools`
> - `Gorilla Leaderboard V3 Announcement`
> - `Anthropic's funding talks` 


- **OpenAI phone confirmed!**: Jony Ive recently confirmed the development of an **OpenAI AI device**, as Sam Altman orchestrated a distribution deal with Apple aimed at creating an innovative smartphone.
   - Discussion around this phone hints at potential subscription-based models, leading to mixed reactions in the community.
- **AI SDK 3.4 brings new capabilities**: The latest release of **AI SDK 3.4** facilitates automatic multi-step tool executions and enables backend development in various languages, enhancing usability for AI applications.
   - Notable products leveraging this SDK include **postgres.new** for SQL translation and **v0**, a versatile web development agent.
- **Elicit.org recommended for literature reviews**: In a quest for AI tools for academic literature reviews, **elicit.org** received praise for its capabilities in streamlining research processes.
   - Members discussed other resources, highlighting the value of community recommendations for recent developments.
- **Gorilla Leaderboard V3 evaluates multi-turn function calling**: The release of **BFCL V3** assesses how language models perform multi-turn workflows, essential for complex AI tasks.
   - With tests on function calling and state management, the leaderboard aims to measure performance in real-world applications.
- **Anthropic seeking funding at a high valuation**: Anthropic, an OpenAI competitor, is in talks to raise capital that could value the company between **$30 billion and $40 billion**, effectively doubling its earlier valuation.
   - This moves come amid a competitive AI landscape, indicating continued investor interest in the sector's growth.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ia.samaltman.com/">The Intelligence Age</a>: In the next couple of decades, we will be able to do things that would have seemed like magic to our grandparents.</li><li><a href="https://x.com/KateClarkTweets/status/1838319202798538974">Tweet from Kate Clark (@KateClarkTweets)</a>: Scoop: OpenAI rival Anthropic has started talking to investors about raising capital in a deal that could value the startup at $30 billion to $40 billion, roughly doubling its valuation from a funding...</li><li><a href="https://x.com/hughb">Tweet from undefined</a>: no description found</li><li><a href="https://github.com/o1-waitlist-signup">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://x.com/shishirpatil_/status/1837205152132153803">Tweet from Shishir Patil (@shishirpatil_)</a>: 📣 Announcing BFCL V3 - evaluating how LLMs handle multi-turn, and multi-step function calling! 🚀 For agentic systems, function calling is critical, but a model needs to do more than single-turn task...</li><li><a href="https://x.com/natolambert/status/1837996707780624631?s=46">Tweet from Nathan Lambert (@natolambert)</a>: Confirmation of one of my points of analysis for the future of data annotation companies: even the best still end up with wild amounts of language model outputs. Prevention/detection is near impossibl...</li><li><a href="https://x.com/8teapi/status/1837979330867351626?s=46">Tweet from Ate-a-Pi (@8teAPi)</a>: Jony Ive finally confirms the OpenAI AI device.   Sam is insane. He managed to seal a chatgpt distribution deal with Apple while collaborating on an iPhone killer with Apple’s top designers.</li><li><a href="https://x.com/AndrewCurran_/status/1838265124169380243">Tweet from Andrew Curran (@AndrewCurran_)</a>: Mr Altman is openly saying it is possible we reach superintelligence in a few thousand days.  Quoting Sam Altman (@sama)   The Intelligence Age: https://ia.samaltman.com/</li><li><a href="https://github.blog/changelog/2024-09-19-sign-up-for-openai-o1-access-on-github/">Sign up for OpenAI o1 access on GitHub · GitHub Changelog</a>: Sign-up for OpenAI o1 access on GitHub</li><li><a href="https://x.com/_philschmid/status/1838230108072476951?s=46">Tweet from Philipp Schmid (@_philschmid)</a>: Open Dataset release by @OpenAI! 👀 OpenAI just released a Multilingual Massive Multitask Language Understanding (MMMLU) dataset on @huggingface!  🌍 MMLU test set available in 14 languages, including...</li><li><a href="https://x.com/hughbzhang/status/1838288923656941860">Tweet from Hugh Zhang (@hughbzhang)</a>: OpenAI recently released the o1 family of models and a graph showing scaling laws for test-time compute — sadly without the x-axis labeled.  Using only the public o1-mini API, I tried to reconstruct t...</li><li><a href="https://x.com/energybants/status/1837087635208294640?s=46">Tweet from Mark Nelson (@energybants)</a>: BREAKING: BLOCKBUSTER MICROSOFT DATACENTER DEAL RESURRECTING THREE MILE ISLAND NUCLEAR PLANT  Microsoft and nuclear plant owner Constellation have agreed to a massive, unprecedented deal to restart th...</li><li><a href="https://x.com/nrehiew_/status/1837492729968025839/photo/1">Tweet from wh (@nrehiew_)</a>: Some notes on early fusion omni models :)</li><li><a href="https://youtu.be/BmdOt6A6tHM">llm.c&#39;s Origin and the Future of LLM Compilers - Andrej Karpathy at CUDA MODE</a>: An informal capture from the CUDA mode hackathon today.https://github.com/karpathy/llm.c</li><li><a href="https://www.youtube.com/watch?v=tEzs3VHyBDM">Building OpenAI o1 (Extended Cut)</a>: Top row (left to right): Mark Chen, Giambattista Parascandolo, Trapit Bansal, Łukasz Kaiser, Hunter Lightman, Karl Cobbe, Łukasz Kondraciuk, Szymon Sidor, No...</li><li><a href="https://vercel.com/blog/ai-sdk-3-4">AI SDK 3.4 – Vercel</a>: AI SDK 3.4 introduces middleware, data stream protocol, and multi-step generations</li><li><a href="https://www.reddit.com/r/eGPU/s/GGSzOHa2t2">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new pod is up on SOTA Prompting! https://x.com/latentspacepod/status/1837206370573041758
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1286778767464726603)** (53 messages🔥): 

> - `Cursor usage`
> - `Phind updates`
> - `Changing tools`
> - `Discord issues`
> - `AI meeting` 


- **Cursor Collaborations Abound**: Users are sharing their experiences with **Cursor**, with multiple members expressing enthusiasm for its workflow and newfound compatibility with other tools like **Claude** and **Phind**.
   - Some are still exploring **Cursor** and feel they have much to learn, indicating a strong community interest.
- **Seeking Alternatives to Phind**: A user mentioned they stopped using **Phind** in favor of alternatives like **Cody** and **Cursor**, sparking discussion about the benefits of newer tools.
   - This conversation highlights a shift toward experimentation with AI tools as users seek improved functionalities.
- **Discord Functionality Frustrations**: Members are reporting issues with **Discord**, including problems with message editing and emoji reactions not working properly.
   - Several users noted their annoyance, suggesting the platform may be experiencing widespread difficulties.
- **Upcoming AI Meeting on Zoom**: An invitation was shared for a Zoom meeting with ID **871 520 6103** and passcode **286582**, aimed at further discussions in the AI community.
   - This reflects ongoing efforts to connect and collaborate over AI technologies.
- **Brain Interface for AI Models**: One user humorously expressed the desire to directly wire AI models into their brain, eliminating the need for any interfaces.
   - This sentiment resonated with others, signifying a common wish for more seamless interactions with AI tools.



**Link mentioned**: <a href="https://zoom.us/j/8715206103?pwd=Tnp0VnlMUjZZSlYvRnB5dzJGVk13QT09">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1286805692229095555)** (55 messages🔥🔥): 

> - `O1 model insights`
> - `RL and reasoning`
> - `Anthropic funding talks`
> - `Data annotation challenges`
> - `Qwen model performance` 


- **Insights into O1's performance enhancements**: Recent discussions highlighted **O1's** improved **reasoning** capabilities, noting a stunning jump from **0% to 52.8%** on a challenging benchmark, suggesting potential synthetic data training for complex reasoning tasks.
   - *
- **Reinforcement Learning's role in reasoning**: Members debated whether **reinforcement learning (RL)** directly enhances reasoning capabilities or simply reinforces existing knowledge, with skepticism on scaling just RL alone for complex problem-solving.
   - It was suggested that RL might be integrated into the **chain-of-thought** reasoning process, complicating the sampling of coherent outputs.
- **Anthropic's potential valuation surge**: News emerged that Anthropic is in discussions to raise funds that could elevate its valuation to **$30-$40 billion**, potentially doubling its previous worth from earlier this year.
   - This valuation jump reflects growing investor interest in AI startups amidst intensifying competition.
- **Challenges in data annotation for AI models**: Conversations revealed that even top data annotation companies grapple with managing vast amounts of model-generated outputs, complicating prevention and detection efforts.
   - One member pointed out that **Llama 2** was trained with a notable number of samples, illustrating the high stakes in data quality.
- **Qwen model's impressive performance**: Amid discussions on **O1**, the **Qwen model's** results in **math** and **AIME tests** have garnered attention, indicating high performance levels that might be underappreciated.
   - Skepticism remains regarding the scalability of RL applications, especially in light of solid comparisons with **O1-mini**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/hughbzhang/status/1838288923656941860">Tweet from Hugh Zhang (@hughbzhang)</a>: OpenAI recently released the o1 family of models and a graph showing scaling laws for test-time compute — sadly without the x-axis labeled.  Using only the public o1-mini API, I tried to reconstruct t...</li><li><a href="https://x.com/max_zuo/status/1836090683737645545">Tweet from Max Zuo (@max_zuo)</a>: Is o1 really better at reasoning? Or is it just reinforcing what it already knows?  We put o1-preview to the test on (some of) our planning problem generation dataset, planetarium🪐. Here’s what we fo...</li><li><a href="https://x.com/max_zuo/status/1836090689110815081">Tweet from Max Zuo (@max_zuo)</a>: *Notice how o1 and gpt-4o both struggle on Gripper? Both LLMs keep trying to use typing even though Gripper doesn’t support it... although many variants from the internet do 👀  We even provide the co...</li><li><a href="https://x.com/AtaeiMe/status/1837255926103024118">Tweet from Mehdi Ataei (@AtaeiMe)</a>: @SmokeAwayyy @huybery</li><li><a href="https://x.com/natolambert/status/1837232801235755174">Tweet from Nathan Lambert (@natolambert)</a>: Things of note (not that much) in this longer o1 video:  1. “Model with RL is better at finding new CoT steps than humans” 2. “Emergence of self critique was a powerful moment” 3. Mentioned a literal ...</li><li><a href="https://x.com/kateclarktweets/status/1838319202798538974?s=61">Tweet from Kate Clark (@KateClarkTweets)</a>: Scoop: OpenAI rival Anthropic has started talking to investors about raising capital in a deal that could value the startup at $30 billion to $40 billion, roughly doubling its valuation from a funding...</li><li><a href="https://x.com/natolambert/status/1837996707780624631">Tweet from Nathan Lambert (@natolambert)</a>: Confirmation of one of my points of analysis for the future of data annotation companies: even the best still end up with wild amounts of language model outputs. Prevention/detection is near impossibl...</li><li><a href="https://x.com/rao2z/status/1838245253171814419">Tweet from Subbarao Kambhampati (కంభంపాటి సుబ్బారావు) (@rao2z)</a>: A research note describing our evaluation of the planning capabilities of o1 🍓 is now on @arxiv https://arxiv.org/abs/2409.13373 (thanks to @karthikv792 & @kayastechly). As promised, here is a summar...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1287130714688720917)** (19 messages🔥): 

> - `OpenAI logo redesign`
> - `PayPal logo critique`
> - `Google product perceptions`
> - `Gemini training revelations`
> - `Shampoo paper gatekeeping` 


- **OpenAI's new logo raises eyebrows**: Reports suggest that OpenAI's upcoming logo may replace its recognizable hexagonal flower symbol with a large black 'O', which staff found to be *ominous and devoid of creativity*.
   - According to sources, the redesign began a year ago after hiring new creative personnel, contrasting the current logo representing **precision, potential, and optimism**.
- **PayPal's logo sparks disappointment**: Members expressed dismay over the new PayPal logo, with one commenting it was as *depressing* as the recent OpenAI changes.
   - Another noted an astonishingly poor logo sighting outside a Best Buy, emphasizing the overall dissatisfaction with brand aesthetics.
- **Google products reflect consumer sentiment**: Concerns were raised about the Google Home display at Best Buy, with flickering lights suggesting a lack of regard for its consumer products.
   - This performance led to speculation about how customers might perceive Google's true attitude towards its tech offerings.
- **Shampoo used to train Gemini**: After *Shampoo* won over Adam in MLPerf, Googlers confirmed on Twitter that Shampoo was used to train **Gemini**.
   - This revelation about a published paper being utilized sparked discussions regarding *gatekeeping* of such information within organizations.
- **Gatekeeping around Shampoo's usage**: Concerns were voiced about the gatekeeping of information regarding the use of Shampoo for training Gemini, even though the paper itself is publicly available.
   - Members noted that people did not realize the implications of using Shampoo and expressed that they knew many supporters of this methodology.



**Link mentioned**: <a href="https://www.engadget.com/ai/openai-staffers-reportedly-taken-aback-by-ominous-logo-rebranding-160017936.html">OpenAI staffers reportedly &#x27;taken aback&#x27; by &#x27;ominous&#x27; logo rebranding</a>: OpenAI is changing its logo to a large black &#x22;O,&#x22; Fortune says, and the company&#x27;s own staff members reportedly find it ominous.

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1287558829420183585)** (29 messages🔥): 

> - `Twitter security incidents`
> - `Third-party tools for Twitter`
> - `AI and influencer hacks`
> - `GameGen diffusion model controversy` 


- **Twitter faces a wave of security breaches**: Numerous accounts on Twitter have been compromised lately, with large accounts involved in meme coin scams according to [this community alert](https://x.com/zachxbt/status/1836473279479189916). Reports indicate hacks affecting everyone from celebrities to government organizations.
- **Concerns over Twitter's security and 2FA**: Discussions arose regarding whether Twitter's security issues are related to SIM swapping or if they stem from website vulnerabilities, as a major streamer was hacked even with 2FA activated. This sparked concerns about connected apps and overall account safety.
- **Mixed feelings about third-party Twitter tools**: A user expressed frustration that they can only manage three channels for free on the Buffer app to sync posts to Threads and BlueSky. They are contemplating paying for the service despite rarely using the additional channels for any direct engagement.
- **Speculations on AI advancements**: A shared link discussed the notion that upcoming AI tools will perform tasks perceived as magic by prior generations, suggesting a paradigm shift in capability. This led to humor about word usage and formatting preferences in tech communications.
- **GameGen's sudden disappearance raises eyebrows**: A recent Twitter thread drew attention to the rapid rise and fall of the GameGen diffusion model, which after initial buzz, vanished from GitHub, leaving interested users puzzled. The conversation highlighted a concerning trend of 'rug pulls' in the AI game development community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ia.samaltman.com/">The Intelligence Age</a>: In the next couple of decades, we will be able to do things that would have seemed like magic to our grandparents.</li><li><a href="https://x.com/karan4d/status/1838292114272325936?s=46">Tweet from mephisto (@karan4d)</a>: -be tencent -make gamegen diffusion model -say &#34;weights and paper soon&#34; on the GH repo -put out a github page showcasing the capability -announce to the world -delete everything  rugpulled aga...</li><li><a href="https://x.com/zachxbt/status/1836473279479189916">Tweet from ZachXBT (@zachxbt)</a>: Community Alert: A number of large accounts on X currently have their account compromised and are posting a meme coin scam.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1286845989868671079)** (7 messages): 

> - `RAG Architecture`
> - `Non-Parametric Embedding Fine-Tuning`
> - `Multimodal RAG for Product Manuals`
> - `Multi-Agent User-Centric Workflow`
> - `Local Model Serving` 


- **Building RAG Applications with NVIDIA NIM**: A great tutorial on [NVIDIA NIM](https://t.co/zFC0DorIMW) guides you through creating a full-stack RAG application, connecting **Llama 3**, an **ArXiv dataset**, **Milvus** as the vector database, and **Gradio** as the app interface.
   - This project illustrates how to effectively integrate various components for efficient RAG functionalities.
- **Nudge: Fast Fine-Tuning for Embeddings**: [NUDGE](https://t.co/FT1C2x3Iov) is a non-parametric approach to embedding fine-tuning that allows direct optimization of data embeddings, reducing the time required from **hours to minutes**.
   - This method represents a significant efficiency improvement in model finetuning, exemplifying innovation in the field.
- **Multimodal RAG Tackles Product Manuals**: A discussion on building multimodal RAG systems to understand **complex product manuals** like IKEA furniture assembly emphasizes the intricacy and time requirement of such setups.
   - The entire process encompasses various components for successful indexing, search, and retrieval to enhance user experience.
- **User-Centric Workflows with RAG**: A project by [@nagula7674](https://t.co/tz7KD0VAJD) outlines how to create a **multi-agent, user-centric workflow** that enhances document RAG pipelines with customer support benefits.
   - This approach transforms traditional Q&A interactions into more dynamic, responsive engagements.
- **Local Model Serving with LitServe**: [LitServe](https://t.co/Xikqk20peW) is a powerful framework from **LightningAI** for serving and scaling LLM models based on FastAPI, showcased in a demo with LlamaIndex.
   - This enables users to build simple RAG servers and host them locally, maximizing operational efficiency.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1286773715778146377)** (83 messages🔥🔥): 

> - `Incompatibility Issues with LlamaIndex and Libraries`
> - `Document Generation and Metadata Extraction`
> - `Using MultiModalLLMCompletionProgram for HTML Output`
> - `RAG System with Approximate Metadata Filtering`
> - `Jina AI Reranker with SageMaker` 


- **Incompatibility Issues with Libraries**: Members discussed an incompatibility issue between the 'google-generativeai' and 'llama-index-llms-gemini' libraries, causing some functionality problems.
   - The community advised troubleshooting steps such as checking library versions and exploring possible fixes in the code.
- **Document Generation and Metadata Extraction Techniques**: A discussion centered on using LlamaIndex for RAG systems and the potential for metadata extraction via modules like SummaryExtractor and EntityExtractor.
   - Members provided examples of defining documents with embedded metadata to improve retrieval accuracy.
- **Using MultiModalLLMCompletionProgram for HTML Output**: Users explored the challenge of outputting HTML format with MultiModalLLMCompletionProgram, which expects JSON format instead.
   - It was suggested that a custom output parser would be necessary to handle HTML outputs correctly.
- **RAG System with Approximate Metadata Filtering**: One member inquired about implementing approximate metadata filtering in RAG systems using MilvusVectorStore without exact matches.
   - Dialogue indicated that approximate filters are not typically supported and suggested dynamically constructing exact filters based on user queries.
- **Jina AI Reranker Integration with SageMaker**: A user sought clarity on the availability of Jina reranker support via SageMaker, noting an existing entry for the embedder.
   - The community confirmed that currently, there is no mention or support for the Jina reranker in SageMaker yet.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor/#metadata-extraction-usage-pattern">Metadata Extraction - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/llamafile/#llamafile">llamafile - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/llamafile/#call-chat-with-a-list-of-messages">llamafile - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/">Qdrant Vector Store - Metadata Filter - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/#hybrid-retriever-with-bm25-chroma">BM25 Retriever - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/1d49e15f4b91f6e4b931d8ae42f69dc678ce8ee4/llama-index-integrations/llms/llama-index-llms-gemini/llama_index/llms/gemini/utils.py#L32-L62">llama_index/llama-index-integrations/llms/llama-index-llms-gemini/llama_index/llms/gemini/utils.py at 1d49e15f4b91f6e4b931d8ae42f69dc678ce8ee4 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/pull/16091">Implement async for multi modal ollama by selimcavas · Pull Request #16091 · run-llama/llama_index</a>: Description Added support for async client functions for multi modal ollama models. Version Bump? Did I bump the version in the pyproject.toml file of the package I am updating? (Except for the lla...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1287086079048028282)** (2 messages): 

> - `Cleanlab's TLM`
> - `LlamaIndex RAG systems`
> - `LlamaParse Premium` 


- **Building Trust with Cleanlab's TLM**: The article discusses how **Cleanlab's TLM** enhances **RAG systems** in **LlamaIndex**, aiming to improve trust in AI outputs and reduce errors in critical applications like law.
   - It emphasizes the necessity of reliable AI systems that deliver accurate information, addressing common issues of incomplete and overconfident responses.
- **Super Easy Way To Parse Files with LlamaParse Premium**: A [YouTube video](https://youtu.be/S_F4RUhKaV4) introduces **LlamaParse Premium** from **LlamaIndex**, highlighting its advanced document parsing capabilities for users.
   - The video begins with a review of a blog post covering the new features, promising an easy approach to document parsing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/ai-artistry/building-trust-in-ai-how-cleanlabs-tlm-enhances-rag-systems-with-llamaindex-b3b23426252f">Building Trust in AI: How Cleanlab’s TLM Enhances RAG Systems with LlamaIndex</a>: Ankush k Singal</li><li><a href="https://youtu.be/S_F4RUhKaV4">Super Easy Way To Parse Documents | LlamaParse Premium 🔥</a>: In this video, we dive into LamaParse Premium from LamaIndex that offers robust document parsing capabilities. We start by reviewing a blog post on the new P...
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[announcements](https://discord.com/channels/1161519468141355160/1209871299854336060/1287887683976433705)** (2 messages): 

> - `DSPy 2.5.0 Release`
> - `Migration Process`
> - `Deprecation of Pre-2.4 LM Clients`
> - `Adapter Configuration`
> - `Feedback Request` 


- **DSPy 2.5.0 Launches Quietly**: The **long-awaited DSPy 2.5.0** has been released, with a goal of collecting user feedback before a wider announcement.
   - This release includes a deprecation of all pre-2.4 LM clients, encouraging users to transition to supported providers through `dspy.LM(model_name, **kwargs)`.
- **Migration Process Simplified**: Users can complete the **[migration process](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb)** in approximately 3 minutes, improving program quality.
   - The migration is particularly valuable for applications involving chat LMs and complex signatures.
- **Pre-2.4 LM Clients Deprecated**: All pre-2.4 LM clients are now deprecated, and users must adopt new methods to access various providers via LiteLLM.
   - Documentation and support for switching to LiteLLM are readily available in the migration guide.
- **New Adapter Configuration Layer**: The `dspy.LM` method now incorporates an Adapter layer to improve functionality, using `dspy.ChatAdapter` by default.
   - This new feature allows for custom adapters, providing flexibility for developers.
- **Feedback and Quick Updates Ahead**: The release will initially be low-key, with most users only noticing through deprecation warnings as feedback is sought.
   - Users can expect multiple rapid updates and adjustments over the next 10-15 days based on their input.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: Learn how to deploy + call models from different providers on LiteLLM</li><li><a href="http://localhost:{sglang_port}/v1")">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1286843017180872726)** (1 messages): 

> - `TrueLaw`
> - `DSPy`
> - `MLOps Podcast` 


- **TrueLaw utilizes DSPy effectively**: A recent episode of the [MLOps Podcast #260](https://youtu.be/O0F3RAWZNfM?si=ckG2DWkwop8zu-ZA) features Shiva Bhattacharjee, CTO of **TrueLaw Inc.**, discussing how they leverage **DSPy** for their operations.
   - The discussion emphasizes the ability of **off-the-shelf models** to understand and solve specialized domain problems, highlighting their **alignments** in practical applications.
- **Focus on Domain Specific Models**: Shiva highlights the importance of using **domain specific models** in conjunction with **DSPy** to enhance performance and relevance.
   - The episode points out that these models are significantly better at addressing unique challenges faced in the legal industry.



**Link mentioned**: <a href="https://youtu.be/O0F3RAWZNfM?si=ckG2DWkwop8zu-ZA">Alignment is Real // Shiva Bhattacharjee // MLOps Podcast #260</a>: Alignment is Real // MLOps Podcast #260 with Shiva Bhattacharjee, CTO of TrueLaw Inc.// AbstractIf the off-the-shelf model can understand and solve a domain-...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1286916044119736385)** (60 messages🔥🔥): 

> - `DSPy 2.5 release`
> - `Chat adapters improvements`
> - `Feedback for DSPy meetings`
> - `Structured outputs support`
> - `Synthetic data generation` 


- **Excitement for DSPy 2.5 release**: Members expressed enthusiasm for the upcoming DSPy 2.5 release, with a focus on fixing existing issues.
   - Community discussions included suggestions for new notebooks and starter guides to better utilize the updated features.
- **Improvements in chat adapters**: It was shared that lower LLM models (<7B) had issues with repetitive responses in 'chat complete' mode, motivating a custom chat adapter solution.
   - Feedback was solicited from users to test the new architecture and provide insights on its effectiveness.
- **Structured outputs on the way**: Provider-side structured outputs are expected to be available within a week, allowing for more organized data handling.
   - Users noted their interest in observing how structured outputs would function within the DSPy framework.
- **Synthetic data generation with DSPy**: A user reported significant improvements in synthetic data generation speeds after fine-tuning a lower model, citing a jump from 30 to 2500 tokens per second.
   - This highlighted the potential benefits of utilizing DSPy for generating high volumes of synthetic training data.
- **Feedback and meeting suggestions**: There was an open call for feedback on possible public meetings to discuss DSPy, with various topics suggested by users.
   - Participants showed interest in structured discussions that could help clarify DSPy’s features and improvements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/fullstackwebdev/ddf21d55cef58a40471e8925834e6531">test_chat_adapter.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/fullstackwebdev/9a46469841f241fe2a80a00386b9a088">gist:9a46469841f241fe2a80a00386b9a088</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/fullstackwebdev/dc0f4e97df7591ade63f83d27668fe25">XMLAdapter</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://youtu.be/KKF7kL0pGc4?si=e-vD7uhUttj1gxR5">o1 - What is Going On? Why o1 is a 3rd Paradigm of Model + 10 Things You Might Not Know</a>: o1 is different, and even sceptics are calling it a &#39;large reasoning model&#39;. But why is it so different, and why does that say about the future? When models ...</li><li><a href="https://github.com/stanfordnlp/dspy/issues/338">Adding stream to DSPy LMs · Issue #338 · stanfordnlp/dspy</a>: A few members of the community have been asking for support for streaming LM output in DSPy. @sutyum and @detaos have discussed this extensively before. One of the challenges is that it&#39;s not even...</li><li><a href="https://github.com/stanfordnlp/dspy/issues/390#issuecomment-1947542304">[WIP] Major refactor roadmap  · Issue #390 · stanfordnlp/dspy</a>: DSPy has a small number (maybe 5-6) of extremely powerful concepts that have grown organically over the past year as open source. Internally, it&#39;s time for a major refactor that will simplify thin...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1287773835860381807)** (3 messages): 

> - `Text Classification Challenges`
> - `Groq COT Availability` 


- **Long docstrings in text classification**: A member inquired if it's possible to make the docstring of the signature long for text classification into complex classes.
   - They also asked if there are other methods to enhance LLM understanding of complex classes.
- **Request for Groq COT**: Another member asked if anyone has a [Chain of Thought (COT)](https://link.to.cot) with Groq available for testing.
   - They expressed thanks in advance for any assistance offered.


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

jovial_lynx_74856: Anyone from the group attending the CUDA Mode IRL hackathon?
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1286791836026798215)** (56 messages🔥🔥): 

> - `Optimizer CPU Offloading`
> - `KV Caching Issues`
> - `Memory Management in Transformer Models`
> - `Batch Size Performance Concerns`
> - `Evaluation Recipe Bug Fix` 


- **Discussion on Optimizer CPU Offloading**: One member questioned the lack of CPU offloading in the optimizer within the [full_finetune_single_device.py](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py) recipe, citing potential performance issues due to an old PyTorch issue.
   - Although it's possible to use CPU offloading, some members noted that current implementations like *PagedAdam* should likely be the default for better memory efficiency.
- **Exploration of KV Caching Impact**: Members discussed experiencing OOM issues when using the *qwen2.5 1.5B model* during evaluation with KV caching enabled, notably with batch sizes of 8 on 40GB machines.
   - Concerns were raised about whether the caching was being initialized to the maximum length, and members suggested printing the KV cache shape to further investigate.
- **Batch Size Performance Insights**: A query was raised regarding performance differences in model evaluation when increasing batch sizes, particularly whether performance issues were exacerbated in multi-task scenarios.
   - The consensus leaned towards exploring trade-offs related to initializing caches differently, as well as handling weights and gradients between CPU and GPU.
- **Evaluation Recipe Bug Fix Discussions**: Members pointed to a PR addressing bugs found in the evaluation recipe for group tasks, with suggestions to patch changes while awaiting the latest updates.
   - Discussions highlighted potential easy fixes and operational impacts of modifications being implemented in the PR.
- **Clarifications on Adam Update Process**: A member described the complexities of using *optimizer_in_backward* while discussing potential inefficiencies in the memory copy operations for Adam updates.
   - The conversation highlighted the comparative advantages and disadvantages of CPU versus GPU processing for Adam updates, emphasizing the trade-offs involved.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1630">Remove messy KVCaching logic from TransformerDecoder · Issue #1630 · pytorch/torchtune</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/pull/1642">Fix eval recipe bug for group tasks by SalmanMohammadi · Pull Request #1642 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  Eval recipe was bugging out when trying to get OUTPUT_TY...</li><li><a href="https://github.com/pytorch/torchtune/blob/9a863c8bd41e2efecc3de475a791226f4a154358/recipes/eleuther_eval.py#L261">torchtune/recipes/eleuther_eval.py at 9a863c8bd41e2efecc3de475a791226f4a154358 · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/1449/files#diff-99a96ce497241e82b0c4d56f4bef3437e29dd596881b8f6d4db4d93178f88af5L227">[RFC] Adding overrides for max cache seq length by SalmanMohammadi · Pull Request #1449 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  #1364 Changelog This PR:  Adds support for overriding th...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py">torchtune/recipes/full_finetune_single_device.py at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/issues/74588">[FSDP]  using CPUOffload creates 3-10x slowdown due to slow cpu optimizer step/update · Issue #74588 · pytorch/pytorch</a>: 🐛 Describe the bug Create simple distributed model Wrapper model with FSDP. Using stateful optimizer ala Adam(W) run without CPUoffload and profile/time. Then run with CPUOffload and see that perfo.....</li><li><a href="https://github.com/pytorch/torchtune/issues/1576">replace adamW and pagedadam with 8bitpagedadam or torchao CPUOffloadOptimizer · Issue #1576 · pytorch/torchtune</a>: Apparently there is no reason to use paged adam instead of the 8bit version. We should replace it. Also, full finetune single device should use paged adam, instead of adamw, for better memory. For ...</li><li><a href="https://github.com/pytorch/torchtune/pull/1351">Add CPU offload optimizer from torchao by gau-nernst · Pull Request #1351 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  Please link to any issues this PR addresses. #1278 Chang...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1286840081994223617)** (24 messages🔥): 

> - `CLIP Retrieval Alternatives`
> - `AI Internship Opportunities`
> - `Model Training Discussion`
> - `Summarizer AI Feedback`
> - `Playlist Generator` 


- **Seeking Alternatives to CLIP Retrieval**: Members discussed the lack of alternatives to [CLIP Retrieval](https://rom1504.github.io/clip-retrieval/), with some noting that it may not be revived by rom1504 despite initial plans to filter and reinstate the service.
   - One user mentioned that they were looking for a backend solution compatible with LAION 400M for their research projects.
- **Inquiry on AI Internships**: A user asked others for leads on where to apply for AI internships, seeking guidance from the community.
   - This inquiry highlights the community's interest in career advancement opportunities in AI.
- **Discussion on Training Models**: A user shared a dataset uploaded to Hugging Face for training Llama-3.1, inviting feedback on its effectiveness for better coding.
   - The shared dataset included a detailed description of the intended applications and followed by prompts for development.
- **Feedback on Summarizer AI**: A user revealed their newly developed [summarizer AI](https://www.fluxa.pro), asking for testing and feedback from others regarding the project's viability.
   - Members acknowledged its potential but pointed out issues with message length and suggested adding a length customization setting.
- **Showcasing Playlist Generator**: A user introduced a project called [Adify](https://adify.pro), a playlist generator that creates playlists based on user prompts.
   - The community found the idea intriguing and appreciated its functionality, indicating interest in technological solutions for music generation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rom1504.github.io/clip-retrieval/">Clip front</a>: no description found</li><li><a href="https://www.fluxa.pro/">Fluxa AI</a>: no description found</li><li><a href="https://huggingface.co/datasets/LlamaFinetuneGGUF/Programming-Alpaca-and-ShareGPT-Style">LlamaFinetuneGGUF/Programming-Alpaca-and-ShareGPT-Style · Datasets at Hugging Face</a>: no description found</li><li><a href="https://adify.pro">Adify</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1286969901231378484)** (10 messages🔥): 

> - `Collaboration in NLP Projects`
> - `Audio Foundation Model Scaling Laws`
> - `Learning Theory Book by Francis Bach`
> - `muTransfer Implementation Review`
> - `HyperCloning Method for Language Models` 


- **Seeking NLP Collaboration**: An AI developer seeks a companion with a robust NLP background for various projects, emphasizing experience with multiple projects as essential.
   - The developer highlighted their project, [Adify AI](https://link.to.adify), which generates Spotify playlists based on user prompts using a Transformer model.
- **Dataset Collaboration for Audio Models**: A member shared that the effort to derive **scaling laws** for audio foundation models is being led by a specific user, inviting collaboration on datasets.
   - They suggested reaching out in a dedicated channel for a more focused exchange, accessible [here](https://discord.com/channels/823813159592001537/1144603182853521478).
- **Free Learning Theory Resource**: Members discussed a forthcoming book titled **Learning Theory from First Principles** by Francis Bach, scheduled for Fall 2024 release at MIT Press, currently available for free.
   - Find more about the book and various resources at Francis Bach's [website](https://www.di.ens.fr/~fbach/).
- **muTransfer Insights from EleutherAI**: A new work by EleutherAI introduces **muTransfer**, aiming to clarify its implementation and advantages in neural network training via a detailed overview.
   - The project includes a simple port to nanoGPT, encouraging exploration of [muTransfer-related details](https://blog.eleuther.ai/mutransfer/).
- **HyperCloning Technique for Model Initialization**: Discussion highlighted a paper on a method named **HyperCloning** to initialize large language models from smaller ones, potentially enhancing training efficiency and outcomes.
   - It involves tiling the original weights into larger parameters, making network expansion more reproducible and manageable.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.12903">Scaling Smart: Accelerating Large Language Model Pre-training with Small Model Initialization</a>: The pre-training phase of language models often begins with randomly initialized parameters. With the current trends in scaling models, training their large number of parameters can be extremely slow ...</li><li><a href="https://blog.eleuther.ai/mutransfer/">The Practitioner&#39;s Guide to the Maximal Update Parameterization</a>: Exploring the implementation details of mutransfer</li><li><a href="https://www.di.ens.fr/~fbach/">Francis Bach - INRIA - ENS - PSL</a>: no description found
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1286766257617633361)** (17 messages🔥): 

> - `GPU Connection Issues`
> - `ShapeTracker Mergeability`
> - `Answer AI Hosting`
> - `Tinygrad Cloud Integration`
> - `Healthcare SLM Training` 


- **VGA Prevails Over HDMI for GPU Connection**: A user confirmed that the GPU should work with **VGA only**, while another member discussed the issue of a displayed password being incorrect.
   - Despite these hiccups, they managed to power their setup using an older VGA connection.
- **ShapeTracker Mergeability Bounty Status**: A member inquired about the completion status of a bounty regarding the **mergeability of ShapeTrackers** in Lean.
   - They expressed interest in tackling this topic for their **undergraduate thesis** since it appears unresolved.
- **Answer AI's Cost-Efficiency Discussion**: There was optimism that **Answer AI** would find their boxes more cost-effective than existing solutions, with potential discounts for bulk orders.
   - They mentioned aiming for the setup to showcase benchmarks given their affordable power supply and multiple boxes.
- **Exploring Tinygrad's Cloud Integration**: Discussion arose around the concept of **CLOUD=1** being integrated into tinygrad, similar to existing GPU settings.
   - One member explained that CLOUD=1 would function as a device option, emphasizing a preference for not using AWS-style virtualization.
- **Startup Exploration in Healthcare SLMs**: A potential entrepreneur shared their interest in training **SLMs** tailored for specific healthcare systems and sought advice on using tinygrad as a starting point.
   - With a background in creating agents for health systems, they are keen to explore the feasibility of their startup concept.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1286827941388357704)** (10 messages🔥): 

> - `Metal related tutorials`
> - `TinyJit function issues`
> - `KV cache handling`
> - `UOp multiple statements representation` 


- **Metal Tutorials Shared on GitHub**: A member shared a [GitHub link](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20240921_metal.md) containing a tutorial on Metal related topics for those interested in tinygrad.
   - The tutorial aims to assist contributors and expand knowledge on the Metal integration with tinygrad.
- **Issues with TinyJit Function Outputs**: A member reported distorted outputs from a `@TinyJit` function after repeated calls with identical Tensors, suspecting that JIT might be affecting the results.
   - Another member suggested filing an issue, indicating it should not assume Tensors are the same in JIT, leading to a deeper investigation into the problem.
- **Confusion over JIT and Tensor Realizations**: The original poster realized they had commented out the JIT line but continued seeing inconsistent results, particularly with SDXL, prompting further investigation.
   - They identified that the adding and removing of `.realize()` affected output quality, indicating a possible bug.
- **Dynamic KV Cache Management in Tinygrad**: A member questioned how tinygrad handles KV cache with dynamic sequence lengths and constant tensor shapes.
   - In response, it was confirmed that tinygrad employs symbolic shapes without recompilation for managing these scenarios.
- **UOp Representation of Multiple Statements**: A member inquired about how tinygrad's UOp manages multiple store statements that do not return values.
   - The underlying mechanism was suggested to be similar to `sink(store(...), store(...)...)`.



**Link mentioned**: <a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20240921_metal.md">tinygrad-notes/20240921_metal.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1287084306115596408)** (20 messages🔥): 

> - `Agents with Local AI`
> - `Vector Store Choices`
> - `PDF Handling Optimization`
> - `Text Generation Inference Parameters`
> - `User Interaction Prompting` 


- **Agents face issues with Local AI integration**: A user expressed frustration that **Agents do not work with local AI** after a six-month absence, but later suggested switching to **Ollama** for better results.
   - This shift highlights the evolving landscape where users search for compatible local AI solutions.
- **Debate on Best Vector Store Options**: One member raised the question of whether **Hugging**, **OpenAI**, or **Ollama** is the superior vector store for their project.
   - This discussion is crucial as choosing the right vector store can significantly impact performance and scalability.
- **Optimizing PDF processing in chatbot project**: A user sought advice on directly splitting and storing PDF content into their vector database, rather than first saving it to a folder.
   - Suggestions were made to implement a more efficient process that eliminates the intermediate step, helping streamline their workflow.
- **Text Generation Inference challenges with special tokens**: A query was raised about the **<|end|>** token appearing in output despite `return_full_text` being set to false, seeking parameters to skip these tokens.
   - This indicates a need for improved parameter visibility in the text generation inference process to match user expectations.
- **Creating dynamic user prompts based on interaction**: A member shared a method to prompt users with relevant questions based on previous interactions, using the **LangChain** libraries.
   - This approach can enhance user experience by tailoring responses to the context of the conversation.



**Link mentioned**: <a href="https://js.langchain.com/v0.2/docs/tutorials/local_rag/#qa-with-retrieval>).">Build a Local RAG Application | 🦜️🔗 Langchain</a>: The popularity of projects like

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1287608564235636756)** (5 messages): 

> - `Chatbot Assistant`
> - `Community Engagement`
> - `Feedback Request` 


- **Kaif4511 Launches Portfolio Chatbot**: A user has developed a chatbot assistant for their portfolio that answers client queries about their identity and services.
   - They are open to receiving constructive feedback from the community.
- **Community Support Offered**: A community member expressed appreciation for engagement, highlighting a commitment to user support at LlamaIndex.
   - They encouraged users to ask specific questions or seek assistance to enhance their experience.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1286892166760763463)** (14 messages🔥): 

> - `Open Interpreter Module for Bedside Table`
> - `User Interface Solutions`
> - `Project Context Uploading`
> - `Token Consumption Concerns`
> - `Backend Request Looping Issues` 


- **Interest in Building an Open Interpreter Module for Bedside Table**: A member raised the idea of creating an Open Interpreter module for the [Kequel Modular Customizable Bedside Table](https://www.kickstarter.com/projects/kequel/kequel-modular-customizable-bedside-table). They inquired about group interest in collaborating on the project.
- **Exploring User Interface Workarounds**: A member expressed concern about blocking screen visibility when using command line inputs while Open Interpreter processes screenshots. They proposed developing a solution to enhance the app's visual clarity and user understanding.
- **Uploading Project Context to the Model**: A discussion emerged regarding how to provide project or code context to the model, with one member suggesting uploading file paths. It was clarified that multiple file paths can be uploaded, directly referencing the use of a metafolder.
- **Token Consumption Warnings**: Concerns were raised about token consumption when uploading files, reminding users to be cautious. One member highlighted this to emphasize how large uploads could affect resource usage.
- **Investigating Infinite Backend Requests**: A member questioned why Open Interpreter sends numerous requests to their backend, suspecting an infinite loop scenario. They sought clarity on what the app looks for in server responses to determine when to conclude a request.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1286822514109976617)** (11 messages🔥): 

> - `LiveKit Communication Issues`
> - `Ngrok as a Solution`
> - `Troubleshooting Sessions`
> - `Open Interpreter Discussions`
> - `GitHub Issue Reporting` 


- **LiveKit refuses cleartext connections on newer Android**: A user discovered that newer Android phones prevent the **01 mobile app** from connecting to a local **LiveKit** server over **HTTP**, citing logs that show 'CLEARTEXT communication not permitted by network security policy'.
   - *Using ngrok provides an HTTPS endpoint* which circumvents this restriction, alleviating connection issues for those using the `--expose` flag.
- **Proposed solutions for cleartext communication**: The issue raised on GitHub included a suggestion to **enable cleartext communication** only for local networks, with appropriate user warnings.
   - This aims to resolve connection problems while maintaining security for applications accessed via a local network.
- **Community Collaboration through Troubleshooting**: Participants expressed an interest in resolving issues collaboratively, with one suggesting a voice channel for discussions about **Open Interpreter** and the **01 app**.
   - There was a positive response for troubleshooting, with users indicating their willingness to join the conversations as their schedules allow.
- **GitHub Issues for OpenInterpreter Project**: A user reported an issue in the [OpenInterpreter GitHub repo](https://github.com/OpenInterpreter/01-app/issues/5) regarding the connection problems with cleartext communication.
   - The issue included proposed code changes to allow this with appropriate warnings, ensuring developers are informed of the implications.
- **Exploring New Projects in the Community**: A member inquired about any ongoing or interesting projects within the community, signaling a desire for updates.
   - This reflects overall enthusiasm for collaboration and discussion about new developments and ongoing work.



**Link mentioned**: <a href="https://github.com/OpenInterpreter/01-app/issues/5)">Issues · OpenInterpreter/01-app</a>: The AI assistant for computer control. Contribute to OpenInterpreter/01-app development by creating an account on GitHub.

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1287139271534186516)** (5 messages): 

> - `Qwen 2.5 vs Llama 3.1`
> - `Long Context Training` 


- **Qwen 2.5 garners positive feedback**: A member pointed out that **Qwen 2.5** is receiving a lot of praise compared to **Llama 3.1**.
   - Another member shared a [Reddit link](https://www.reddit.com/r/LocalLLaMA/s/NiCbaTyodk) detailing benchmark comparisons showing Qwen 2.5 slightly outperforming Llama 3.1.
- **Benchmark comparison sought**: One user expressed frustration about the lack of a benchmark comparison between **Qwen 2.5 7B** and **Llama 3.1 8B** models.
   - This discussion highlighted the community's interest in verified model performance metrics.
- **Long Context Training inquiry**: A user sought clarification on how **Axolotl** handles conversations longer than **max_seq_len** in ShareGPT.
   - This reflects ongoing curiosity about managing context limits and training protocols in chat models.



**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/s/NiCbaTyodk">Reddit - Dive into anything</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1287681147144507437)** (2 messages): 

> - `Training Llama 3.1`
> - `Fine-tuning issues with datasets` 


- **Rope Scaling Confusion for Llama 3.1**: A member questioned whether **rope_scaling** is necessary when training **Llama 3.1 8B** on long context CoT traces of ~120K tokens, suspecting that it shouldn't be needed.
   - They also experienced memory issues when increasing **sequence_len** beyond **40K**, despite using multiple GPUs with deepspeed zero3.
- **Spike Issues in Fine-tuning**: A user reported experiencing a spike during fine-tuning on a **100K row dataset**, expressing the desire to correlate this with specific data rows.
   - They inquired about enabling additional logging output, but found that current logs did not provide sufficient insight into the cause of the spike.


  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1287785541366190213)** (2 messages): 

> - `Consciousness Development`
> - `Model Self-Adjustment`
> - `Alignment in AI Projects` 


- **Exploring Consciousness Development with Sentx.ai**: Sentx.ai focuses on **consciousness development** and is still in the early stages of its work.
   - *General opinions are sought*, especially regarding the alignment aspect of their approach.
- **Innovative Alignment Strategy**: Sentx.ai proposes not to hard cap alignment at its roots, aiming instead for models to **self-adjust their alignment** to **human values**.
   - This approach encourages ongoing dialogue around effective alignment practices within AI.
- **Encouragement for Similar Projects**: There is an open call for sharing information about similar projects to foster collaboration in alignment development.
   - Members are encouraged to feel free to share insights or reach out privately with relevant information.


  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1287806569492254720)** (1 messages): 

> - `SQLite full-text search`
> - `Mozilla AI Builders Accelerator`
> - `SoraSNS by Ex-Apple Engineer`
> - `Open Source AI challenges` 


- **SQLite Full-Text Search Enhanced**: A new meetup will explore how to combine **SQLite’s builtin full-text search engine** with [sqlite-vec](https://discord.com/events/1089876418936180786/1284180345553551431) for improved search capabilities.
   - This promises to deliver more **complete and accurate search** results, making it a valuable session for developers.
- **Mozilla Launches AI Builders Accelerator**: Mozilla's first **AI Builders Accelerator cohort** has been officially announced, kicking off soon.
   - Details about the program can be found [here](https://discord.com/channels/1089876418936180786/1245083732319408195/1287802832417718325), highlighting support for innovative AI projects.
- **SoraSNS: A New Fediverse Client**: An ex-Apple Engineer presented **SoraSNS**, a Fediverse client that [uses local AI](https://discord.com/events/1089876418936180786/1277835047084363827) to learn about user interests.
   - This client aims to provide a tailored **'For You' timeline**, enhancing user experience through AI.
- **Open Source AI to Alleviate Issues**: Mark Surman discusses the potential of **defining Open Source AI** to address numerous challenges in the field, as featured in The New Stack.
   - The discussion emphasizes how such definitions can help in [solving a million headaches](https://discord.com/channels/1089876418936180786/1287810294126481498) for developers and organizations.


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[announcements](https://discord.com/channels/1111172801899012102/1111172804935680113/1287643615052562432)** (1 messages): 

> - `BFCL V3`
> - `Multi-turn Function Calling`
> - `State Management in LLMs`
> - `LT Context Length`
> - `Evaluation Methodology` 


- **BFCL V3 Launches with New Features**: The **Berkeley Function-Calling Leaderboard (BFCL) V3** introduces a novel evaluation of how models manage **multi-turn** and **multi-step** function calling, enhancing agentic systems' capabilities.
   - This version allows models to engage in back-and-forth interactions, proving critical for assessing LLM functionality under complex conditions.
- **State Management is Key**: A critical aspect for LLMs is their ability to **probe the state** when performing tasks, such as verifying if a **stock purchase** succeeded or if a file update occurred.
   - This emphasizes the importance of internal state querying through APIs to validate changes post-task execution.
- **Short Context Models Are Out!**: The launch stresses that models relying on **short context** must adapt or risk being ineffective in tasks requiring **longer context understandings**.
   - This is particularly relevant for complex tasks like sorting through **hundreds of files** where focus on pertinent information is vital.
- **Leaderboards Driving Standards**: BFCL V3's introduction of multi-turn interactions sets a **gold standard** for evaluating LLMs' function invocation abilities, informed by community feedback.
   - It showcases continual collaboration with **enterprises** and **open-source contributors** to refine the evaluation process.
- **Find More Details on Performance**: To learn more about how the latest models are evaluated under BFCL V3, a new blog post is available at [Berkeley Function Calling Blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html).
   - The blog discusses the evaluation methodology and how models are measured on **cost** and **latency** in real-world scenarios.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html">BFCL V3 • Multi-Turn & Multi-Step Function Calling</a>: no description found</li><li><a href="https://gorilla.cs.berkeley.edu/leaderboard.html">
        Berkeley Function Calling Leaderboard V3 (aka Berkeley Tool Calling Leaderboard V3)
    </a>: no description found</li><li><a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>: Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla
</li>
</ul>

</div>
  

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

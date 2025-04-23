---
id: d04af6d2-a91e-46e4-825a-e61462b91dad
title: '[AINews] Welcome Interconnects and OpenRouter'
date: '2024-02-27T20:03:47.279106Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-welcome-interconnects-and-openrouter
---

<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/26/2024. We checked **22** guilds, **349** channels, and **12885** messages for you. Estimated reading time saved (at 200wpm): **1063 minutes**.

Not much happened today, so it's a nice occasion to introduce 2 new Discords that have passed our quality bar: Interconnects (run by [Nathan Lambert who we recently had on Latent Space](https://www.latent.space/p/rlhf-201)) and OpenRouter (Alex Atallah who will surely join us at some point).


![image.png](https://assets.buttondown.email/images/00f2e1a5-0660-440d-a917-09696a39198a.png?w=960&fit=max) 



---

**Table of Contents**

[TOC] 


# PART 0: Summary of Summaries of Summaries

<div><p><strong>Model Comparisons and Optimizations</strong>: Discord users actively engaged in discussions about the performance and optimization of various AI models, including <strong>Mistral AI</strong>, <strong>Miqu</strong>, and <strong>GGUF quantized models</strong>. Key topics included the comparison of Mistral AI's new <strong>Mistral Large</strong> model with OpenAI's <strong>GPT-4</strong>, highlighting its cost-effectiveness and performance. Users also explored efficient loading practices and model quantization methods, such as <strong>GPTQ</strong> and <strong>QLORA</strong>, to enhance model performance and reduce VRAM usage​​​​​​.</p><p><strong>Advanced Features and Applications</strong>: There was significant interest in leveraging AI for specific applications like <strong>role-playing</strong> and <strong>story-writing</strong>, emphasizing the need for models to manage consistent timelines and character emotions. Moreover, the potential of AI in <strong>code clarity</strong> and <strong>AI-assisted decompilation</strong> was discussed, with a particular focus on developing tools like an <strong>asynchronous summarization script</strong> for Mistral 7b, indicating a push towards making AI more accessible and practical for developers​​​​.</p><p><strong>Quantum Computing and AI's Future</strong>: Discussions in various Discords touched upon the intersection of quantum computing and AI, speculating on how quantum advancements could revolutionize AI model processing. The discourse extended to the implications of DARPA-funded projects on AI development and the exploration of <strong>encoder-based diffusion techniques</strong> for stable diffusion in image processing, reflecting a deep interest in cutting-edge technologies that could shape the future of AI​​​​.</p><p><strong>Community and Collaboration Initiatives</strong>: Various communities highlighted efforts to foster collaboration and share knowledge on AI model development, ranging from <strong>new Spanish LLM announcements</strong> to <strong>hardware experimentation</strong> for model efficiency. Platforms like <strong>Perplexity AI</strong> and <strong>LlamaIndex</strong> were noted for their innovative features and integration capabilities. There was a strong emphasis on open-source projects, as seen in discussions about <strong>Mistral's open-source commitment</strong> and the development of tools like <strong>R2R</strong> for rapid RAG system deployment, showcasing the vibrant collaborative spirit within the AI community​​​​​​.</p></div>

# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Polymind's Haunting Echoes**: Engineers are troubleshooting **Polymind's UI bug** where *ghost messages* linger until new tokens stream. Debugging involves **frontend HTML variables**, like `currentmsg`, to clear residual content.

- **Model Assessment Mania**: Discussions highlight **Miqu**, a leaked **Mistral medium equivalent**, performing admirably despite being slower due to higher parameters compared to **Mixtral**. Users are also actively exchanging efficient loading practices for **GGUF quantized models**, including tweaks like disabling shared video memory and GPU settings to prevent VRAM overflow.

- **LangChain's Loop:** LangChain is under fire for its claimed context reordering feature, which `@capt.genius` labeled a "fake implementation", and users suggest it might just be a basic looping mechanism.

- **Mistral's Mysterious Moves**: There's a buzz around **Mistral AI**'s change in language on their website regarding open-source models, fueling speculation about the company's commitment to open models, in light of models from other AI providers like **Qwen**.

- **Optimizing Prompt Engineering**: Engineers in the roleplay and story-writing channels express a desire for models adept at **role-playing** and **story-writing** with **consistent timelines, character emotions**, and **mood management**.

- **Quantizing Questions and Training Troubles**: There's debate around **model quantization methods**, with `@orel1212` considering **GPTQ** over **QLORA** for a small dataset training. Questions loom about **Mistral's PDF image text extraction** capabilities and the hardware needed for **Mixtral 8x7b** without clear answers. Meanwhile, `@dzgxxamine` seeks advice on teaching LLMs to understand and use newly released Python libraries.

- **Merge Meltdown Mystery**: A single message from `@jsarnecki` reports an unsuccessful model merge between **Orca-2-13b** and **WhiteRabbitNeo-13b**, resulting in incomprehensible output, with details of the merge process shared via a readme from [mergekit](https://github.com/cg123/mergekit).

- **AI's Role in Code Clarity**: Within the coding discussions, there's excitement about AI's potential in **AI-assisted decompilation**, as `@mrjackspade` anticipates a future without manual code reconstruction. Additionally, `@wolfsauge` introduces an **asynchronous summarization script** tested on **Mistral 7b,** with an invite for peers to experiment with it, available on [GitHub](https://github.com/Wolfsauge/async_summarize).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **New Spanish LLM Announced**: Spain's President Pedro Sánchez announced the creation of a [Spanish-language AI model](https://www.xataka.com/robotica-e-ia/pedro-sanchez-anuncia-creacion-gran-modelo-lenguaje-inteligencia-artificial-entrenado-espanol) at Mobile World Congress; it's significant for Spanish language applications in AI.

- **Hardware Hijinks: RTX 2060 Action**: Discussions in the hardware channel have users experimenting with **multi-GPU setups** without a bridge connector and seeing success, even with mismatched graphics cards, driving efficiency for models as large as 70b, pointing to changing norms in hardware compatibility.

- **WSL Networking Fix for AI Tools**: Users resolved an issue connecting to a LM Studio endpoint from within WSL (Windows Subsystem for Linux) on Windows 11 by bypassing local host and using the network IP (`http://192.168.x.y:5000/v1/completions`), addressing unique WSL network behavior outlined in [this guide](https://superuser.com/a/1690272).

- **Tech for Text-to-Talk**: [Piper](https://github.com/rhasspy/piper), a neural text-to-speech system, has been highlighted for its local processing capabilities and support for different languages, indicating a trend towards efficient, offline solutions for language model applications.

- **Techies Taming Language Models**: In the model discussions, users highlighted the importance of specific quantization parameters like *mirostat_mode* and *mirostat_tau* for optimizing LLM performance, which needs to be manually configured. Conversations also acknowledged the role of human evaluation in assessing models' performance.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **RWKV/Mamba Models Primed for Context-Length Flexibility**: `@vatsadev` commented on the viability of **rwkv/mamba** models being extendable through fine-tuning for longer context, mitigating the need to pretrain from scratch. `@blackl1ght` achieved a 32k token context length on **Solar 10.7B** using a Tesla V100 and Q8 quantization without hitting memory limits, indicating the potential for extending these models even further.
   
- **Ring Attention Sparks 1 Million Token Model Feasibility**: Discussions highlighted the application of **ring attention** in models managing up to 1 million tokens, `@bloc97` revealed that this could be applied to 7b models, illustrating advancements in attention mechanisms for larger scale models.

- **Quantization Techniques Promise Efficient Inference**: Efforts to optimize inference efficiencies involving key-value cache and potential use of fp8 were mentioned; `@stefangliga` introduced the concept of quantizing the kvcache which could pave the way for handling longer contexts more effectively.

- **Benchmarks Point to Selective Reporting**: In a conversation about AI model benchmarks, `@orabazes` pointed out that a tweet on AI performance omitted certain model comparisons, hinting at the need for transparency in benchmarks.

- **Mistral TOS Controversy Settles**: After intense debate regarding Mistral's Terms of Service, a tweet from co-founder [@arthurmensch](https://twitter.com/arthurmensch/status/1762208241927233661) resolved issues by removing a contentious clause, reaffirming the model's open-use for training competitive LLMs.

- **Structured Data Gets Dedicated LLM Attention**: New insights from @_akhaliq's tweet into **Google's StructLM** project indicate a focus on enhancing LLMs' handling of structured data, despite the absence of shared training approaches or model frameworks.

- **Curiosity in Mistral Loop Dilemmas**: `@.ben.com` theorized that repeated text looping in **Mistral** could be attributed to classic feedback system issues, indicating a technical curiosity in understanding and resolving model oscillations.

- **High-Demand for Self-Extend Model Practices**: The community's enthusiasm for local model improvements was evident after `@blackl1ght`'s successful experiments with self-extension, leading to a collective interest in replicating the configurations that allow for improved memory management and feature utilization.

**Links mentioned**:

- [TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF on Hugging Face](https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF)



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

**EBDI Agent Challenges and Solutions**: @.braydie explored **EBDI frameworks** for agent goal determination, but encountered thinking loops after integrating the [ReAct framework](https://react-lm.github.io/). They examined decision-making models from a [JASSS paper](https://www.jasss.org/17/4/13.html) to address the issue.

**Mistral Steps Up to Rival GPT-4**: A [TechCrunch article](https://techcrunch.com/2024/02/26/mistral-ai-releases-new-model-to-rival-gpt-4-and-its-own-chat-assistant/) reported that **Mistral Large**, a new model from Mistral AI, is positioned to compete with OpenAI's GPT-4, offering cost-effectiveness and uncensored content, and is now available on Azure.

**Prompt Protection Paradox**: Users deliberated on how to protect intellectual property in prompts, concluding that while copyright might cover exact wording, the replication of ideas via linguistic variation is likely unstoppable.

**Text Classification Tactics**: @crifat kicked off a discussion on text classification methods, opting to start with the base model and Assistant, bypassing fine-tuning, to sort texts into categories such as "Factual" and "Misleading."

**Meta-Prompting Generates Buzz and Security Concerns**: The concept of **meta-prompting** was a hot topic, with claims of generating extensive documentation from advanced techniques, but these techniques also raised security flags when a user shared a PDF, resulting in the user's account action.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Perplexity AI's Picture Perfect Potential**: Users discussed the image generation capability of **Pro version** of Perplexity AI, with a link given to a [Reddit discussion](https://www.reddit.com/r/perplexity_ai/comments/18eqmig/how_to_generate_images_with_perplexity_ai/) on the topic. Concerns were also voiced over the discontinuation of **Gemini AI support** in Perplexity, with confirmations that the support has ended.

- **Mistral Large Steals the Spotlight**: A new language model from Mistral AI, called *Mistral Large*, was introduced with discussions surrounding its capabilities and availability. The announcement was supported by a [link to Mistral AI's news post](https://mistral.ai/news/mistral-large/).

- **VPN Troubles Take a Detour**: A login issue on Galaxy phones with Perplexity AI was tackled by turning off VPN, with additional resources on [VPN split tunneling](https://support.nordvpn.com/hc/en-us/articles/19618692366865-What-is-Split-Tunneling-and-how-to-use-it#:~:text=On%20Android%3A,choose%20the%20Split%20tunneling%20option) provided.

- **Exploratory Searches in Perplexity Shine Through**: Users shared links to Perplexity AI's search results on novel topics, including a transparent laptop by Lenovo and an age query for the letter K. Comparisons of iPhones were also made, and a collection feature was promoted, inviting users to [create their own collection](https://www.perplexity.ai/collections/Make-your-own-rfj2pcwRS7WF7SFiTAxhJg) on Perplexity AI.

- **PPLLX-API Channel Buzzes with Model Concerns**: Technical discussions centered on model information and performance, particularly about a JSON link for model parameters and the behavior of the `sonar-medium-online` model generating irrelevant content. There was deliberation over the deprecation of the `pplx-70b-online` model and the vital role of proper prompting when making API calls, with explicit referencing to [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions) in the API documentation.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **PDF Parsing Leaps Forward with LlamaParse**: LlamaParse is introduced as a tool to enhance understanding of **PDFs with tables and figures**, aiding LLMs in providing accurate answers by avoiding PDF parsing errors. It's highlighted as significant for Retriever-Augmented Generation (RAG) processes, as tweeted [here](https://twitter.com/llama_index/status/1762158562657374227).

- **Super-Charging Indexing with MistralAI's Large Model**: LlamaIndex integrates **@MistralAI's Large Model** into its build 10.13.post1, bringing near-GPT-4 capabilities including advanced reasoning and JSON output, as mentioned in [this announcement](https://twitter.com/llama_index/status/1762231085243719748). Furthermore, LlamaIndex's new distributed super-RAG feature allows the creation of API services for any RAG application which can be networked to form a super-RAG capable of running queries across the network, as shared [here](https://twitter.com/llama_index/status/1762552542981230769).

- **AI Assemblage with AGI Builders and FireWorksAI**: The AGI Builders meetup will host LlamaIndex's VP of Developer Relations to share insights on RAG applications, with details of the event available [here](https://t.co/kcoIhfgQqF). In another collaboration, LlamaIndex and FireworksAI_HQ have released a cookbook series for RAG applications and function calling with FireFunction-v1, offering full API compatibility as announced [here](https://twitter.com/llama_index/status/1762532341795487815).

- **Context Crafting Conundrums for Coders**: In the ai-discussion channel, members discussed optimizing context for coding LLMs such as **GPT-4 turbo and Gemini 1.5** with a focus on the order of information, repetition, and structuring techniques. There was also discourse on open-source text generation with Llama2, emphasizing non-proprietary integration for CSV and PDF inputs, and exploration of chunking and retrieval strategies for building a RAG-oriented OSS SDK assistant.

- **Tech Troubles and Tooling Talks in General**: The general channel buzzed with requests for assistance, comparisons between GPT-3.5 with and without RAG, and discussions on integrating LlamaIndex with other services like Weaviate. Moreover, there was an elaborate exchange on resolving installation issues of LlamaIndex on macOS and dialogues about creating agents for Golang integration and dynamic orchestration with AWS Bedrock, suggesting a highly interactive and technically-oriented community.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **New Twists in Encoder-Based Diffusion Techniques**: Users discussed the merits of an [encoder-based inversion method](https://tuning-encoder.github.io/) for stable diffusion and the challenges with image prompt (IP) adapters in enhancing stable diffusion performance.

- **AI's Dicey Dance with DARPA**: Debates flared regarding the implications of DARPA-funded projects on AI research and development, with a humorous twist on anime character usage in recruitment efforts and the intersection of military and entertainment genres.

- **Quantum Leaps for AI's Future**: Conversations revolved around the future role of quantum computing in processing AI models, particularly transformers, with discussions on the state of quantum error correction and its potential in AI computations.

- **Navigating the Minefield of Content Moderation**: Dialogues dug into the difficulties surrounding content moderation, especially in relation to Child Sexual Abuse Material (CSAM), pondering the effectiveness of reporting tools and responsibilities of platforms.

- **The Balancing Act of Open Source vs. Proprietary AI**: There was a spirited exchange over the strategies of releasing AI models, contrasting open-source approaches like Mistral Large with proprietary models and considering the commercialization supporting ongoing AI R&D.

- **Watermarking Language Models**: A [research paper](https://arxiv.org/abs/2402.14904) unveiled findings on the detectability of training data through watermarked text, indicating **high confidence** in identifying watermarked synthetic instructions used in as little as 5% of training data.

- **Genie Grants Wishes in Humanoid Robotics**: A [YouTube video](https://www.youtube.com/watch?v=gGKsfXkSXv8) from Google DeepMind's new paper brought to light recent advancements in AI's role in **humanoid robotics**.

- **The Finite of FEM Learning Success**: Within the realm of Finite Element Analysis (FEM), there was acknowledgment of research yet a consensus that methods for learning with FEM Meshes and Models are **not as effective** as conventional FEM, citing a [specific paper](https://arxiv.org/abs/2302.04107).

- **Torching Through Fourier Transform Challenges**: A technical issue involving the inverse discrete Fourier transform in neural network synthesis was highlighted, pointing to code that might benefit from a **refactoring using `torch.vmap`** for VRAM efficiency, demonstrated in the shared [GitHub repository](https://github.com/mkaic/abacus/blob/28d20a2f3a244d09218e6ddd998db08c7872dc45/src/interpolators.py#L177).

- **Pondering Transformer Learning Experiments**: There was a solitary query about conducting transformer learning experiments to discern size relations between fictional objects, aiming to understand and render comparative images, but no further discussion or data was provided.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord Summary

- **Paywall Hinders Financial Times Article Sharing**: A [Financial Times article](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb) discussing Microsoft's stake in French AI startup Mistral was shared by `@xeophon.`, but it was criticized for its poor mobile paywall experience. Microsoft's investment is said to support the deployment of Azure-based commercial language models.

- **Debating the Efficacy of Chain of Thought (CoT)**: Reports from users like `@sid221134224` and `@xeophon.` suggest mixed results with CoT prompting, ranging from dropped performance in models like Gemini Pro to being a standard practice in fine-tuning.

- **Multilingual Models Alignment Challenges Discussed**: Discussions focused on the alignment of **multilingual models**, questioning whether the values should be uniform or culturally specific, highlighting issues with GPT-4's safety measures which can be bypassed via low-resource languages like Scots Gaelic as reported by [The Register](https://www.theregister.com/2024/01/31/gpt4_gaelic_safety/).

- **Insights on AI Tools and Writing Habits of Contributors**: Contributors `@natolambert` and `@xeophon.` shared their preferences for writing tools and their processes, such as Notion, Grammarly, Typora, and Obsidian, exemplifying varied approaches to creating written content.

- **Excitement Over DeepMind's Genie and Mistral's Business Strategy**: Engaging discussions broke out over DeepMind's new foundation world model, Genie, which shows potential in world dynamics learning from videos; and Mistral's resource consumption and pricing strategy, with speculation on token usage based on available public information.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord Summary

- **Mistral Large Emerges as New AI Contender**: A new model called **Mistral Large** has been introduced by `@alexatallah`, positioned between GPT-4 and Claude 2 for advanced capabilities with features like a 32,000 token context window and multilingual support, available at [OpenRouter's Mistral Large](https://openrouter.ai/models/mistralai/mistral-large). Pricing adjustments have been made across the Mistral portfolio, advocating for Mistral 7B Instruct and Mixtral 8x7B Instruct due to affordability.

- **Sonar Soars with Online Capabilities**: `@alexatallah` shared the launch of **Perplexity's Sonar** models, including an internet-connected version, that outshine their predecessors in cost-efficiency and speed, located at [Perplexity's Sonar 8x7B Online](https://openrouter.ai/models/perplexity/sonar-medium-online). With PPLX models set to be deprecated on March 15, users are encouraged to transition to Sonar.

- **OpenRouter Playground Gets a Boost**: Important updates in OpenRouter Playground include the addition of new parameters such as Top P, Top K, and penalties, improving user interaction. Message system issues across models including Perplexity, Mistral, and Gemma have been fixed, optimizing performance.

- **New AI Creator Tools Hit the Market**: The launch of **Videotok** on Product Hunt was announced, offering an AI-powered platform for creating short videos, with a shareable post found at [Borja Soler's tweet](https://x.com/borjasolerr/status/1762025283597582807?s=20). Additionally, [Blust AI's platform](https://blust.ai) has surfaced, integrating multiple AI applications, with integration steps detailed in their [documentation](https://docs.blust.ai/docs/integrating-ai-tools/).

- **Automate All The Things with Make.com**: A new app was introduced by `@jim14199` enabling no-code AI workflow automations by linking OpenRouter with a multitude of other apps, available at [Make.com OpenRouter Integration](https://www.go-synergetic.com/apps/openrouter). Issues surrounding model functionality on the Blust AI platform were reported, though specifics were not provided.

- **OpenRouter Sparks Community Conversation**: The community engaged in a lively discussion about new feature updates like the API for parameters, with suggestions to tailor them towards user preferences and production user inputs. There's a palpable excitement for the release of Mistral Large, coupled with discussions about Perplexity model errors and choosing the best interfaces for leveraging OpenRouter AI models, as seen at [OpenRouter](https://openrouter.ai).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

**Have an Amazing Week and Ace Those Exams**: Community members are sharing sentiments ranging from well-wishes for the week to the stress of exams.

**Seeking Speedy Batch Processing Solutions**: A discussion took place regarding the optimal batching methods for querying GPT-4, emphasizing the importance of fast and efficient batch processing to reduce completion times.

**Service Disruptions and Tech Collaborations**: Users reported experiencing 504 timeout errors with the Hugging Face Inference API, highlighting service instability; meanwhile, there's an ongoing dialogue to foster collaborative machine learning project development within the community.

**Immersive Study Opportunity in Convolutional Neural Networks**: An open invitation was extended for a study group focusing on CS231n, *Convolutional Neural Networks for Visual Recognition*, with links to course assignments and modules available for interested participants. [CS231n Course](https://cs231n.github.io/)

**Scale AI's Rise to Prominence and VLM Resolutions**: Articles and discussions showcased Scale AI's impressive growth to a $7.3 billion valuation in data labeling and innovative solutions to overcome resolution problems in vision-language models using multiple crops of high-resolution images. [Scale AI's Story](https://www.turingpost.com/p/scaleai) and [VLM Resolution Solution](https://huggingface.co/blog/visheratin/vlm-resolution-curse)

**Developments and Debates in AI Ethics and Performance**: The community shared opportunities for commenting on "open-weight" AI models, a new Performance LLM Board evaluating response times and pricing of various models, and a detailed replication attempt of the Imagic paper for text-based image editing using diffusion models. [Open AI Model Weights Comments](https://aimodels.org/ai-blog/comment-ntia-open-source-open-weights-rfc/) and [Imagic Paper Replicated](https://medium.com/@chongdashu/papers-decoded-imagic-text-based-real-image-editing-with-diffusion-models-b1bda8b2532a)

**Discontentment with Diffusion Model Tools**: Voices of dissatisfaction emerged regarding the use of eps prediction in Playground v2.5, and the choice to utilize the EDM framework instead of zsnr.

**Data Size and Character Recognition in Computer Vision**: A notable concern was raised about the adequacy of dataset size for fine-tuning, especially for models aimed at complex character recognition, such as those in the Khmer language, which presents unique challenges due to its symbol-rich script.

**Navigating the NLP Landscape**: Conversations touched on best practices in sequence classification, searching for generative QA models, recommendations for embedding models suited for smaller datasets, strategies for compressing emails for LLMs, and constructing a medical transformer tailored to the nuances of medical terminology. Suggested models for embedding include [BAAI's bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) and [thenlper's gte-small](https://huggingface.co/thenlper/gte-small).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Unit Style Learning Catches Engineer's Eye**: Precision in machine learning algorithms was the focus with a conversation about 'unit style' vectors, emphasizing **RMSNorm** as a crucial component for achieving 'unit length' vectors and avoiding the need for conversion between scales.

- **Mistral Lifts the Veil on a Large Model**: EleutherAI announced the release of *Mistral Large*, a new state-of-the-art language model that promises cutting-edge results on benchmarks. It's now available through *la Plateforme* and *Azure*, which was discussed in detail on [Mistral's news page](https://mistral.ai/news/mistral-large/).

- **Appreciation and Call to Action for lm-harness**: A member highlighted the impact of EleutherAI's **lm-harness**, with its recognition in a recent paper for its importance in the few-shot evaluation of autoregressive language models, and threads included guides and discussions to enhance its functionality.

- **Interpreting 'Energy' Sparks Inquiry and Promises of Investigation**: Members expressed confusion and the need to investigate the term "energy" used in the context of model tuning and related equations, admitting a lack of intuition and clarity around the concept.

- **Technical Troubleshooting in NeoX**: Users discussed challenges and sought advice for **DeepSpeed** configurations and setting up multi-node training environments for GPT-NeoX on **CoreWeave** Infrastructure, with directions pointing towards utilizing Kubernetes and employing **slurm** or **MPI** for a 2 node with 4 GPUs setup.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **GTC Excitement and Planning**: Members are looking forward to attending the **Graphics Technology Conference (GTC)**, with suggestions for a meetup and a dedicated channel for attendees. A CUDA-based [Gameboy emulator on GitHub](https://github.com/krocki/nvgb) was shared, showcasing an ingenious use of CUDA for classic gaming emulation.

- **Unleashing QLoRA's Speed**: A [GitHub repo for 5X faster and 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth) was highlighted for accelerating models efficiently.

- **In-Depth CUDA Discussions**: The community engaged in discussions about CUDA interoperability with Vulkan, recounted the enhancements to GPU utilization with TorchFP4Linear layers, and shared updates on selective layer replacement techniques. Insights about GPU memory access latencies were also exchanged, supported by a variety of external references like the examination paper of NVIDIA A100's memory access and [torch-bnb-fp4's speed test script](https://github.com/aredden/torch-bnb-fp4/blob/main/examples/speed_test_mistral_7b.py).

- **Optimizing PyTorch Workflows**: Conversations included the origin of PyTorch rooted in Torch7 and tips for speeding up `cpp_extension.load_inline` compile times. There was guidance pointed for integration of custom Triton kernels with `torch.compile` in PyTorch, citing a [GitHub example](https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661).

- **Efficiency in Attention Algorithms**: Papers and resources were shared on efficient softmax approximation, the base2 trick for softmax, and an impressive [incremental softmax implementation in OpenAI's Triton example](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py). There was also a nod to the classic fast inverse square root trick from Quake3.

- **NVIDIA Job Opportunity**: NVIDIA posted a job opportunity seeking experts in CUDA and C++, referring interested individuals to DM their CV with **JobID: JR1968004**.

- **Beginner Queries**: Discussions surfaced around what is needed to dive into CUDA Mode, including any prerequisites like knowledge of PyTorch, and recommended starting with the fast.ai course before venturing into CUDA.

- **Clarification in Terminology**: The acronym **AO** was clarified to stand for **Architecture Optimization**.

- **Advancements in CUDA Attention Mechanisms**: Tips for improved performance in ring attention were discussed, along with the scheduling of a flash attention paper reading/discussion. A [Colab notebook](https://colab.research.google.com/drive/1FMzg1vf2QEg5Q9fjjQV3Q5Ep-r2x9l-t#scrollTo=X08l8A7tdu-X) comparing ring and flash attention was shared for community feedback, alongside a paper on implementing FlashAttention-2 on NVIDIA's Hopper architecture.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Structured Output Interface Seeks Community Insight**: An intuitive interface for obtaining structured model outputs was proposed, with a request for feedback by user `@bagatur`. Details are discussed in a [GitHub RFC on langchain-ai/langchain](https://github.com/langchain-ai/langchain/discussions/18154).

- **New Tools and Guides for LangChain Enthusiasts**: Various resources dropped, including the launch of [UseScraper.com](https://usescraper.com) by `@dctanner` for content scraping, a comprehensive [integration guide for LlamaCpp with Python](https://python.langchain.com/docs/integrations/llms/llamacpp) by `@ldeth256`, and a shoutout to [validate.tonic.ai](https://validate.tonic.ai/), a platform by `@locus_5436` for visualizing RAG system evaluations.

- **Seeking Temporary Workarounds for Function Calls in Chats**: `@sectorix` inquires about temporary solutions for enabling function calling within chat capabilities for open-source models like **Mistral** ahead of expected features from **Ollama**.

- **Spotlight on Innovative RAG-Based Platforms**: An array of projects showcased: **R2R** framework for RAG systems by `@emrgnt_cmplxty` ([GitHub link](https://github.com/SciPhi-AI/R2R)), **IntelliDoctor.ai** for medical inquiries by `@robertoshimizu` ([website](https://intellidoctor.ai)), and **LangGraph**'s approach to iterative code generation enhancement mentioned by `@andysingal`.

- **LangGraph and AI Technologies Trending in Tutorials**: Tutorials highlight novel applications such as multi-agent systems with LangGraph showcased by `@tarikkaoutar` on [YouTube](https://www.youtube.com/watch?v=q5LvDHiSBy4), and an AI conversation co-pilot concept for mobile by `@jasonzhou1993`, also available via [YouTube](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Mistral's Open-Source Allegiance in Doubt**: Skepticism arises among users over Mistral's commitment to open-source practices in light of their partnership with Microsoft; the [CEO of MistralAI reiterates an open-weight model commitment](https://fxtwitter.com/casper_hansen_/status/1762159643344662859), yet some suspect a profit-centric shift.
- **Gemma Outpaces Competition**: Significant progress has been made on **Gemma models**, which are now [2.43x faster than Hugging Face](https://huggingface.co/unsloth) with FA2 and use 70% less VRAM, as shared by `@nanobitz` with links to two free usage notebooks for the [Gemma 7b](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing) and [Gemma 2b models](https://colab.research.google.com/drive/15gGm7x_jTm017_Ic8e317tdIpDG53Mtu?usp=sharing).
- **LoRA-The-Explorer Unveiled for Training**: A new method called **LoRA-the-Explorer (LTE)** has been introduced by `@caseus_` for training neural networks efficiently, which includes a [parallel low-rank adapter approach](https://minyoungg.github.io/LTE/) and a [multi-head LoRA implementation](https://github.com/minyoungg/LTE/blob/main/lte/mhlora/linear.py).
- **Documentation Discrepancies Discussed**: References to proper documentation within a GitHub pull request spurred a discussion resulting in the sharing of the [relevant materials](https://github.com/OpenAccess-AI-Collective/axolotl/blob/13199f678b9aab39e92961323bdbce3234ee4b2b/docs/mac.md) in the [axolotl-dev channel](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1292#discussion_r1493791256).
- **R2R Framework for RAG Deployment**: User `emrgnt_cmplxty` launched **R2R**, a framework aimed at the rapid development and deployment of **RAG systems**, available on [GitHub](https://github.com/SciPhi-AI/R2R) for community use.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

**Zero-Shot Model Match-Up**: `@eugeneyan` clarified that a tweet thread about AI models being compared to GPT-4 was referencing their **zero-shot** performance metrics, which is crucial for understanding the models' capabilities without fine-tuning.

**Mistral and Microsoft Forge Ahead**: `@__chef__` announced **Mistral Large**, touting its benchmark performance and revealing a partnership with Microsoft, a significant development spotlighted on [Mistral Large's announcement page](https://mistral.ai/news/mistral-large/).

**Cloudflare Offers a Simplified AI Solution**: `@henriqueln7` highlighted the release of Cloudflare's AI Gateway, drawing attention to its single-line-of-code ease of use, alongside robust analytics, logging, and caching features, outlined at [Cloudflare's AI Gateway documentation](https://developers.cloudflare.com/ai-gateway/).

**Mistral Au Integrated with RAG for Advanced Applications**: `@ashpreetbedi` praised the integration of **Mistral Au Large** with RAG, noting its improved function calling and reasoning, and directed users to their GitHub cookbook at [phidata/mistral](https://github.com/phidatahq/phidata/tree/main/cookbook/mistral).

**RAG Resource Reveal Generates Buzz**: `@dimfeld` announced an upcoming eBook on RAG by Jason Liu, aimed at explaining the concept with varying complexity levels, which `@thenoahhein` found especially useful for a Twitter data summarization task; the eBook's repository can be found at [n-levels-of-rag](https://github.com/jxnl/n-levels-of-rag).



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **AI Gets Lost in Translation**: In an interesting turn of events, `@derekpwillis` encountered a language switch where **chatgpt-3.5-turbo** mistakenly used Spanish titles for documents intended to be in English, amusingly translating phrases like "Taking Advantage of the Internet" to *"Sacándole Provecho a Internet"*. `@simonw` compared this to an earlier bug involving ChatGPT and Whisper mishearing a British accent as Welsh, and advised using a prompt directing the system to "Always use English" to prevent such language mix-ups.

- **LLM Plugin Links Python to Groqcloud**: `@angerman.` introduced the [LLM plugin](https://pypi.org/project/llm-groq/) that allows Python developers to access [Groqcloud](https://console.groq.com) models such as `groq-llama2` and `groq-mixtral`. The plugin has recently been updated to include streaming support, and there's talk of a forthcoming chat UI for LLM, although no release date has been shared.

- **Python Packaging Made Easy**: `@0xgrrr` provided a helping hand by sharing a [tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects) on how to package Python projects for others to upload and contribute, exemplifying it as a straightforward process.

- **Choosing Fly for GPU Capabilities**: In response to `@kiloton9999`, `@simonw` explained that part of the reason for opting for Fly GPUs in Datasette development is due to Fly's sponsorship and their GPUs' ability to scale to zero, which is valuable for the project's resourcing needs.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Challenging LLM Evaluation Standards**: An [academic paper](https://arxiv.org/abs/2402.13887) was shared, critiquing current probability-based evaluation methods for Large Language Models as misaligned with generation-based prediction capabilities, highlighting a gap in understanding why these discrepancies occur.

- **Dataset Conversion Anomalies**: A hidden null string issue was observed during JSON to Parquet data conversion, detected through Hugging Face's direct JSON upload and conversion process, showcasing the nuances in dataset preparation.

- **Advancing Codebase Assistance with RAG**: Discussions around the creation of a Retrieval-Augmented Generation (RAG) bot for codebases delved into the integration of LangChain's Git loader, segmentation for programming languages, LlamaIndex's Git importer, and use of OpenAI embeddings in retrieval processes, making strides in developer-assistant technology.

- **Exploring End-to-End Optimization in RAG and LLMs**: Inquiry into joint end-to-end optimization for RAG and LLMs using gradients was highlighted, including a look at the [LESS paper](https://arxiv.org/abs/2402.04333), which details a method for retrieving training examples with similar precomputed gradient features, rather than backpropagation through data selection.

- **Emotional Intelligence Benchmarks Go International**: EQ-Bench has expanded with **German language support**, stirring conversations about translation quality and emotional nuance between languages after GPT-4 scored 81.91 in German compared to 86.05 in English; this initiative can be explored further on their [GitHub](https://github.com/EQ-bench/EQ-Bench) and emphasizes the importance of language fluency in model benchmarking.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **FireFunction V1 Ignites with GPT-4-Level Capabilities**: [FireFunction V1](https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling), a new model featuring **GPT-4-level structured output** and decision-routing, was highlighted for its impressive low latency, open-weights and commercial usability. There is an ongoing discussion on latency specifics, particularly whether response latency pertains to time to first token or completion.

- **R2R Takes on Production RAG Systems**: The **R2R** framework was announced, poised to streamline the development and deployment of production-ready RAG systems. Details for this rapid development framework can be found on [GitHub - SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R), with further community inquiry into how it differs from the existing [agentmemory framework](https://github.com/JoinTheAlliance/agentmemory).

- **GPT-4 Proves Its Mettle in Drug Information**: A user successfully used GPT-4 to generate detailed information cards about drugs, outlining mechanisms, side effects, and disease targets. However, a limitation was noted regarding GPT-4's inability to integrate images into outputs, impacting methodologies like Anki's image occlusion.

- **Search for Speed**: Concerns over **OpenAI API latency in seconds** were raised, leading to discussions about whether **dedicated hosting** could remedy this problem. The dissatisfaction extends to Azure hosting experiences, with users sharing their disappointment regarding performance. 

- **Collaborative Push for Improved RAG Systems**: Enhancements for a RAG system were shared by a user, seeking community feedback on proposed improvements. Interested parties can provide their input via the proposal on [GitHub](https://github.com/jxnl/n-levels-of-rag/blob/main/README.md).



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Gemma Gets Game-Changing Tokens**: `@imonenext` announces the integration of `<start_of_turn>` and `<end_of_turn>` tokens into the **Gemma** model, enhancing its ability to handle instruction/RL fine-tuning, with the model being accessible on [Hugging Face](https://huggingface.co/imone/gemma-7b-with-it-tokens).
- **Token Integration a Manual Affair**: Adding instruction-tuning tokens to **Gemma** required manual procedures involving copying tokenizers as explained by `@imonenext`, ensuring token consistency with no report of issues since original instruction tokens were maintained.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Given the limited information provided, it is not possible to create a substantial summary. The only message is a link shared by a user to a YouTube video in an off-topic channel, which does not pertain to any technical discussion or detail-oriented topics relevant to an engineer audience. If the video had technical content relevant to AI or engineering, that information was not included in the prompt, so it would not be appropriate to include it in the summary.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **Agent Protocol V2 Coding in Action**: `_z` invited the community to a [YouTube live coding session](https://youtube.com/live/zrJuNUGYKJg?feature=share), focusing on the **Agent Protocol's Config Options RFC** as part of the Agent Protocol V2 Milestone. The live stream encouraged real-time interaction and contribution.



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1211589709927682108) (1277 messages🔥🔥🔥): 

- **Ghost Messages in Polymind's UI**: Users discussed a persistent issue with ghost messages appearing in Polymind's UI after refreshing or clearing memory; a previous message is displayed until the model starts streaming tokens again. Efforts to debug are ongoing, involving examination of frontend HTML variables like `currentmsg` and implementations to clear the lingering content.

- **Comparing LLM Performance**: In a series of messages, users discussed the performance of models such as Miqu and Mixtral, noting differences in speed, parameter size, and memory consumption. Miqu, a leaked Mistral medium equivalent, is lauded for performance over Mixtral, despite being slower due to higher parameters.

- **GGUF Quant Load Issues**: Users exchanged advice on loading GGUF quantized models efficiently, discussing settings such as layer offload numbers and disabling shared video memory in the Nvidia control panel to avoid spillover into less efficient shared VRAM.

- **LangChain Scrutiny**: `@capt.genius` called out LangChain for a "fake implementation" of context reordering in its source code, using nothing more than a simple loop that seemingly alternated array elements based on index parity, branding LangChain as a "scam."

- **Mistral's Direction on Open Source**: Conversation included mentions of Mistral changing references to open source models to past tense on their website, sparking speculation about the company's future direction in offering open models. Users debated the effectiveness and visions of models offered by Mistral and other AI providers like Qwen.

- **Model Usage in Different Environments**: `nigelt11` expressed frustration at inconsistent performance of a LangChain Chatbot app between a local machine and HuggingFace Spaces, citing variations in output quality and seeking insights into potential causes.


**Links mentioned**:

- [GOODY-2 | The world&#x27;s most responsible AI model](https://www.goody2.ai/chat): Introducing a new AI model with next-gen ethical alignment. Chat now.
- [Nod Cat Hyper GIF - Nod cat hyper - Discover &amp; Share GIFs](https://tenor.com/view/nod-cat-hyper-gif-9540792418684483949): Click to view the GIF
- [Technology](https://mistral.ai/technology/#models>): Frontier AI in your hands
- [Rapeface Smile GIF - Rapeface Smile Transform - Discover &amp; Share GIFs](https://tenor.com/view/rapeface-smile-transform-gif-12599812): Click to view the GIF
- [American Psycho Impressive GIF - American Psycho Impressive Very Nice - Discover &amp; Share GIFs](https://tenor.com/view/american-psycho-impressive-very-nice-coping-patrick-bateman-gif-26518058): Click to view the GIF
- [TheBloke/deepseek-coder-33B-instruct-GGUF · Hugging Face](https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF): no description found
- [Qwen](https://qwen.readthedocs.io/): no description found
- [Thebloke.Ai Ltd - Company Profile - Endole](https://suite.endole.co.uk/insight/company/15361921-thebloke-ai-ltd): no description found
- [diable/enable CUDA Sysmem Fallback Policy from command line](https://gist.github.com/itsdotscience/4e29dca91f010a1873d1083fae94a655): diable/enable CUDA Sysmem Fallback Policy from command line - a
- [アークナイツ BGM - Boss Rush 30min | Arknights/明日方舟 導灯の試練 OST](https://www.youtube.com/watch?v=KQ1MKDDYvF8): 作業用アークナイツTrials for Navigator #1 Lobby Theme OST Boss Rush 30min Extended.Monster Siren Records: https://monster-siren.hypergryph.comWallpaper: Coral Coast s...
- [OpenAI INSIDER On Future Scenarios | Scott Aaronson](https://youtu.be/gGsh0_-q7LI): This is a lecture by Scott Aaronson at MindFest, held at Florida Atlantic University, CENTER FOR THE FUTURE MIND, spearheaded by Susan Schneider.LINKS MENTIO...
- [《Arknights》4th Anniversary [ Sami: Contact ] Special PV](https://www.youtube.com/watch?v=0m18wLQInnU): Special PV về Sami: ContactSource: https://www.bilibili.com/video/BV13k4y1J7Lo=============================================Group Fb: https://www.facebook.com...
- [gguf (GGUF)](https://huggingface.co/gguf): no description found
- [GitHub - facebookresearch/dinov2: PyTorch code and models for the DINOv2 self-supervised learning method.](https://github.com/facebookresearch/dinov2?tab=readme-ov-file>): PyTorch code and models for the DINOv2 self-supervised learning method. - facebookresearch/dinov2
- [GitHub - BatsResearch/bonito: A lightweight library for generating synthetic instruction-tuning datasets for your data without GPT.](https://github.com/BatsResearch/bonito): A lightweight library for generating synthetic instruction-tuning datasets for your data without GPT. - BatsResearch/bonito
- [Search Results | bioRxiv](https://www.biorxiv.org/search/scgpt): no description found
- [[Blue Archive] [AI Tendou Alice] Chipi Chipi Chapa Chapa (Dubidubidu)](https://www.youtube.com/watch?v=2wsLZyvaqlE): AI singing tool 歌聲音色轉換模型: so-vits-svc 4.1: https://github.com/svc-develop-team/so-vits-svcCharacter Voice: ブルアカ 天童アリス（CV：‎田中美海）Original Music: Christell - Du...
- [The Lost World of Papua New Guinea 🇵🇬](https://www.youtube.com/shorts/yL5dgjdFzHI): Full episode here: https://youtu.be/nViPq2ltGmg?si=fkVDkqdSTZ3KWZxJGratitude should be the only attitude
- [UAI - Unleashing the Power of AI for Everyone, Everywhere: Introducing Universal AI Inference](https://rentry.co/UAI-universal-ai-inference): The following text has been entirely written by Mistral's great models. I've been hearing a lot of chatter about the need for more open models and community access to AI technology. It seems like ever...
- [Brain organoid reservoir computing for artificial intelligence - Nature Electronics](https://www.nature.com/articles/s41928-023-01069-w): An artificial intelligence hardware approach that uses the adaptive reservoir computation of biological neural networks in a brain organoid can perform tasks such as speech recognition and nonlinear e...
- [Fast-tracking fusion energy’s arrival with AI and accessibility ](https://news.mit.edu/2023/fast-tracking-fusion-energy-with-ai-and-accessibility-0901): MIT Plasma Science and Fusion Center will receive DoE support to improve access to fusion data and increase workforce diversity. The project is being led by Christina Rea of the 
- [Add ability to skip GateKeeper using &quot;//&quot; · DocShotgun/PolyMind@2d4ab5c](https://github.com/DocShotgun/PolyMind/commit/2d4ab5c02fa91d4558d77f92fec9a73ac20a2537): no description found
- [The Gate-All-Around Transistor is Coming](https://youtu.be/5RPFfPtgw7g): Links:- The Asianometry Newsletter: https://www.asianometry.com- Patreon: https://www.patreon.com/Asianometry- Threads: https://www.threads.net/@asianometry-...
- [Fall Out Boy - Introducing Crynyl™️](https://www.youtube.com/watch?v=l75SlbaZxtA): Introducing Crynyl™, records filled with real tears for maximum emotional fidelity. So Much (For) Stardust is available for pre-order now on https://crynyl.c...
- [Add `with_children: bool` to `delete_all()`, to allow calling delete_all on an `is_root=True` base class, to delete any subclass rows as well. by TheBloke · Pull Request #866 · roman-right/beanie](https://github.com/roman-right/beanie/pull/866): A simple change: adds with_children: bool to delete_all(), which gets passed through to find_all(). When there is an inheritance tree, this allows deleting all documents in a collection using the b...
- [scGPT: toward building a foundation model for single-cell multi-omics using generative AI - Nature Methods](https://www.nature.com/articles/s41592-024-02201-0): Pretrained using over 33 million single-cell RNA-sequencing profiles, scGPT is a foundation model facilitating a broad spectrum of downstream single-cell analysis tasks by transfer learning.
- [Instruct Once, Chat Consistently in Multiple Rounds: An Efficient Tuning Framework for Dialogue](https://arxiv.org/html/2402.06967v1): no description found
- [scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI](https://doi.org/10.1101/2023.04.30.538439): Generative pre-trained models have achieved remarkable success in various domains such as natural language processing and computer vision. Specifically, the combination of large-scale diverse datasets...
- [Mistral AI | Open-weight models](https://web.archive.org/web/20240225142431/https://mistral.ai/): Frontier AI in your hands
- [Mistral AI | Frontier AI in your hands](https://mistral.ai/): Frontier AI in your hands

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1211631811420098620) (1005 messages🔥🔥🔥): 

- **LLMs Roleplay & Story-Writing Wishlist**: Users discussed their wish lists for role-playing and story-writing models, such as mixing `Choose Your Own Adventure` games with `Dungeons and Dragons` lore knowledge and maintaining consistency in story events and timelines. Other desires included managing character emotions and moods, and enhancing descriptive narratives in settings.

- **Prompt Effectiveness Strategies**: Participants shared strategies about how the placement of tokens in prompts, especially the first 20, can greatly shape LLM responses. They also touched on ways to design character cards using pseudo code and tags to guide AI consistency in character depiction.

- **Model Behavior for Story Continuation**: Users expressed a need for models that can follow long multi-turn contexts without losing coherency and maintain in-character responses without resorting to regurgitating character card info verbatim.

- **Handling "Mood" in Roleplay**: The conversation covered how to possibly code mood changes in story interactions using a second analysis model. The aim is for the analysis to assess and update state variables like "anger" which can then be reflected in the main story, possibly using language strings instead of numerical values.

- **Challenges with Current LLMs**: There was a sentiment expressed about entering a "dark era" of model creation, with closed-source developments and a lack of new robust models that can maintain story details as effectively as desired by role-players. Some users also shared challenges in finding optimal system prompts for different models like Miqu, dealing with models that are overly positive, and aiming for deeper initial detail from models.

**Links mentioned**:

- [no title found](https://tenor.com/view/nothing-is-real-jack-as-we-see-it-everything-is-fake-none-of-this-is-real-gif): no description found
- [Hold Stonk Hold GIF - Hold Stonk Hold Wallace Hold - Discover &amp; Share GIFs](https://tenor.com/view/hold-stonk-hold-wallace-hold-braveheart-stonks-gif-20142609): Click to view the GIF
- [Copy Paste Paste GIF - Copy Paste Paste Copy - Discover &amp; Share GIFs](https://tenor.com/view/copy-paste-paste-copy-ctrl-c-ctrl-v-gif-12913156): Click to view the GIF
- [Vorzek Vorzneck GIF - Vorzek Vorzneck Oglg - Discover &amp; Share GIFs](https://tenor.com/view/vorzek-vorzneck-oglg-og-lol-gang-gif-24901093): Click to view the GIF
- [Solid Snake Solid GIF - Solid snake Solid Snake - Discover &amp; Share GIFs](https://tenor.com/view/solid-snake-solid-snake-not-metal-gear-solid-gif-14480942659066849172): Click to view the GIF
- [He Cant Keep Getting Away With It GIF - He Cant Keep Getting Away With It - Discover &amp; Share GIFs](https://tenor.com/view/he-cant-keep-getting-away-with-it-gif-19335672): Click to view the GIF
- [Bane No GIF - Bane No Banned - Discover &amp; Share GIFs](https://tenor.com/view/bane-no-banned-and-you-are-explode-gif-16047504): Click to view the GIF
- [Skeptical Futurama GIF - Skeptical Futurama Fry - Discover &amp; Share GIFs](https://tenor.com/view/skeptical-futurama-fry-hmmm-i-got-my-eyes-on-you-gif-17101711): Click to view the GIF
- [Omni Man Invincible GIF - Omni Man Invincible Look What They Need To Mimic A Fraction Of Our Power - Discover &amp; Share GIFs](https://tenor.com/view/omni-man-invincible-look-what-they-need-to-mimic-a-fraction-of-our-power-gif-25672186): Click to view the GIF
- [Spiderman Everybody GIF - Spiderman Everybody Gets - Discover &amp; Share GIFs](https://tenor.com/view/spiderman-everybody-gets-one-of-us-family-guy-gif-22763691): Click to view the GIF
- [deepseek-ai/deepseek-coder-7b-instruct-v1.5 · Hugging Face](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5): no description found
- [You Dont Say Frowning GIF - You Dont Say Frowning Coffee - Discover &amp; Share GIFs](https://tenor.com/view/you-dont-say-frowning-coffee-gif-14597430): Click to view the GIF
- [Nipples GIF - Nipples - Discover &amp; Share GIFs](https://tenor.com/view/nipples-gif-18634175): Click to view the GIF
- [How The Might Have Fallen Wentworth GIF - How The Might Have Fallen Wentworth S06E11 - Discover &amp; Share GIFs](https://tenor.com/view/how-the-might-have-fallen-wentworth-s06e11-correctional-center-prison-gif-22933267): Click to view the GIF
- [Snusnu Futurama GIF - Snusnu Futurama Fry - Discover &amp; Share GIFs](https://tenor.com/view/snusnu-futurama-fry-death-death-by-snusnu-gif-16228376): Click to view the GIF
- [Here A Thot There A Thot Everywhere A Thot Thot GIF - Here A Thot There A Thot Everywhere A Thot Thot Here A Thot There A Thot - Discover &amp; Share GIFs](https://tenor.com/view/here-a-thot-there-a-thot-everywhere-a-thot-thot-here-a-thot-there-a-thot-everywhere-a-thot-thot-thot-gif-12409116): Click to view the GIF
- [Hurry Up GIF - Hurry Up The Simpsons Faster - Discover &amp; Share GIFs](https://tenor.com/view/hurry-up-the-simpsons-faster-gif-5754665): Click to view the GIF
- [Its Free Real Sate GIF - Its Free Real Sate - Discover &amp; Share GIFs](https://tenor.com/view/its-free-real-sate-gif-7215175): Click to view the GIF
- [no title found](https://www.amazon.com/gp/product/B09MFNLRQQ/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&th=1): no description found
- [GitHub - Potatooff/NsfwDetectorAI: This is an ai made using C# and Ml.net](https://github.com/Potatooff/NsfwDetectorAI): This is an ai made using C# and Ml.net. Contribute to Potatooff/NsfwDetectorAI development by creating an account on GitHub.
- [Person of Interest - Father (04x22)](https://youtu.be/s3o5lOCVuuM): - Extract from season 4 episode 22♥ Here is the new Facebook page to follow all the latest news of the channel : https://www.facebook.com/POI-Best-Of-3109752...
- [Kquant03/NurseButtercup-4x7B-bf16 · Hugging Face](https://huggingface.co/Kquant03/NurseButtercup-4x7B-bf16): no description found
- [Nothing Is Real Jack GIF - Nothing Is Real Jack As We See It - Discover &amp; Share GIFs](https://tenor.com/view/nothing-is-real-jack-as-we-see-it-everything-is-fake-none-of-this-is-real-gif-24594990): Click to view the GIF
- [Come Look At This Come Look At This Meme GIF - Come Look At This Come Look At This Meme Run - Discover &amp; Share GIFs](https://tenor.com/view/come-look-at-this-come-look-at-this-meme-run-run-away-laughing-at-phone-gif-24193569): Click to view the GIF
- [maeeeeee/maid-yuzu-v8-alter-3.7bpw-exl2 · Hugging Face](https://huggingface.co/maeeeeee/maid-yuzu-v8-alter-3.7bpw-exl2): no description found
- [Light Blind GIF - Light Blind Blinding Light - Discover &amp; Share GIFs](https://tenor.com/view/light-blind-blinding-light-too-bright-open-curtains-gif-16971259): Click to view the GIF
- [Denzel Washington Training Day GIF - Denzel Washington Training Day Smoke - Discover &amp; Share GIFs](https://tenor.com/view/denzel-washington-training-day-smoke-gif-22279308): Click to view the GIF
- [We Are Way Past That Jubal Valentine GIF - We Are Way Past That Jubal Valentine Fbi - Discover &amp; Share GIFs](https://tenor.com/view/we-are-way-past-that-jubal-valentine-fbi-we-are-over-that-were-already-past-that-phase-gif-26087300): Click to view the GIF

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1211621405389492255) (4 messages): 

- **Decisions on Model Quantization**: `@orel1212` is contemplating using **GPTQ (INT4)** to train a quantized model on a smaller dataset of 15k prompts and considers **QLORA with a float16 LORA** as a viable alternative to NF4, which seems to worsen performance. They mention preferring **GPTQ** for its compatibility with PEFT training and because it doesn't require model dequantization during inference.
  
- **Extracting Text from Images in PDFs with Mistral**: `@rafaelsansevero` inquires if **Mistral** can read a PDF containing images and extract text from those images. There was no response provided within the chat history to this query.

- **Hardware Requirements for Mistral 8x7b**: `@keihakari` asks about the minimum hardware specifications needed for fine-tuning and running **Mixtral 8x7b**. There was no direct answer to this question within the chat logs.

- **Training LLMs with New Python Libraries**: `@dzgxxamine` seeks assistance with fine-tuning or training a large language model to understand and use new Python libraries that are not yet recognized by ChatGPT or other local models. They only have access to the online documentation of these libraries for this purpose.
  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1211858999415144519) (1 messages): 

- **Merge Gone Wrong with SLERP**: User `@jsarnecki` reported garbled output when merging **Orca-2-13b** with **WhiteRabbitNeo-13b** using a **SLERP** merge method, resulting in output like `\\)\\\\}\\:\\ \\\\)\\\\\\`. They included the *MergeKit Produced Readme* with details about the problematic merge.
- **Merging Details Revealed in Readme**: The readme provided by `@jsarnecki` states the use of [mergekit](https://github.com/cg123/mergekit) for creating **Orca-2-Neo-13b**, a merge of **Orca-2-13b** and **WhiteRabbitNeo-Trinity-13B** using **SLERP** over 40 layers of each model, with various `t` parameter values for different filters and a fallback value for the rest of the tensors.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1211606279932346397) (4 messages): 

- **AI's Future in Decompilation**: `@mrjackspade` expressed excitement about the prospects of **AI-assisted decompilation**, looking forward to the day when manual reconstruction of obfuscated decompiled code won't be necessary.
- **The Struggle with Obfuscated Code**: `@mrjackspade` voiced frustration over **reconstructing obfuscated decompiled code by hand** and hinted at the potential ease of generating datasets for AI training from open-source projects.
- **Invitation to Test Summarization Script**: `@wolfsauge` shared an update to their **summarize script** and is seeking someone to test it with a large model, mentioning it works well with **Mistral 7b instruct v0.2 at fp16 in vLLM**. The script is available on [GitHub](https://github.com/Wolfsauge/async_summarize).

**Links mentioned**:

[GitHub - Wolfsauge/async_summarize: An asynchronous summarization script.](https://github.com/Wolfsauge/async_summarize/): An asynchronous summarization script. Contribute to Wolfsauge/async_summarize development by creating an account on GitHub.

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1211599026122919947) (388 messages🔥🔥): 

- **Tech Troubles at GF's Place**: `@stevecnycpaigne` experiencing an unusual issue with LM Studio model downloads at their girlfriend's place while it works fine at their own. After trying several fixes including disabling IPv6, they sought advice suggesting it might be an issue with the Spectrum service blocking Hugging Face; a potential solution could be manual model downloads.

- **Model Mobility Misunderstandings**: `@mercerwing` inquired about whether LLMs are more RAM-intensive than GPU-intensive, based on initial assumptions that expected LLM operations to rely more heavily on system memory as opposed to graphical processing power.

- **Longing for Longer Context Lengths**: Users discuss the challenges of brevity in LLM story generation; `@heyitsyorkie` suggested finding models with extended context ranges upwards to 200k, while others like `@aswarp` recommended RAG on vector databases to circumvent context limitations. The crux is LLMs still struggle to recall lengthy narratives efficiently.

- **Pondering Mac's Prowess with LLMs**: A side by side hardware talk ensued with `@pierrunoyt` debating if any custom PC hardware exists that could match the inferencing speed of the M1 Ultra on a Mac; `@wilsonkeebs` and others noted the unique SoC architecture of Apple's solutions offers an irreplicable integrated system.

- **VPNs to the Rescue for LLM Connections**: `@exploit36` struggled to download any LLM model and couldn't see catalog entries, later finding success using a VPN, hinting at possible ISP-related restrictions on accessing Hugging Face.

- **German LLM Recommendation Requests**: `@pierrunoyt` is on the lookout for LLM models suitable for German language speakers, seeking guidance on the best models to utilize for such specific linguistic needs.

**Links mentioned**:

- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): no description found
- [Seth Meyers GIF - Seth Meyers Myers - Discover &amp; Share GIFs](https://tenor.com/view/seth-meyers-myers-ehh-maybe-gif-22478163): Click to view the GIF
- [LM Studio Models not behaving? Try this!](https://www.youtube.com/watch?v=LUiVbOeLeas): The repository for free presets:https://github.com/aj47/lm-studio-presets➤ Twitter - https://twitter.com/techfrenaj➤ Twitch  - https://www.twitch.tv/techfren...
- [Anima/air_llm at main · lyogavin/Anima](https://github.com/lyogavin/Anima/tree/main/air_llm): 33B Chinese LLM, DPO QLORA, 100K context, AirLLM 70B inference with single 4GB GPU - lyogavin/Anima
- [Hugging Face – The AI community building the future.](https://huggingface.co/): no description found
- [Hugging Face – The AI community building the future.](https://huggingface.co/.): no description found
- [The Needle In a Haystack Test](https://towardsdatascience.com/the-needle-in-a-haystack-test-a94974c1ad38?gi=2721d916b4a5): Evaluating the performance of RAG systems
- [GitHub - havenhq/mamba-chat: Mamba-Chat: A chat LLM based on the state-space model architecture 🐍](https://github.com/havenhq/mamba-chat): Mamba-Chat: A chat LLM based on the state-space model architecture 🐍 - havenhq/mamba-chat
- [Performance of llama.cpp on Apple Silicon M-series · ggerganov/llama.cpp · Discussion #4167](https://t.co/acxXfci9Pw): Summary LLaMA 7B BW [GB/s] GPU Cores F16 PP [t/s] F16 TG [t/s] Q8_0 PP [t/s] Q8_0 TG [t/s] Q4_0 PP [t/s] Q4_0 TG [t/s] ✅ M1 1 68 7 108.21 7.92 107.81 14.19 ✅ M1 1 68 8 117.25 7.91 117.96 14.15 ✅ M1...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1211646154811506718) (44 messages🔥): 

- **Spanish LLM Announcement at MWC 2024**: `@aswarp` shared [news about the creation of a Spanish-language AI model](https://www.xataka.com/robotica-e-ia/pedro-sanchez-anuncia-creacion-gran-modelo-lenguaje-inteligencia-artificial-entrenado-espanol) announced by Spain's President Pedro Sánchez at the Mobile World Congress.
- **Piper Project for Text-to-Speech**: `@yahir9023` mentioned [Piper](https://github.com/rhasspy/piper), a fast and local neural text-to-speech system with binaries for Windows and Linux and pre-trained models in various languages available on Huggingface.
- **Quantization Effects on Model Performance**: `@drawless111` discussed the importance of specific parameters like *mirostat_mode* and *mirostat_tau* for optimizing AI models, specifying that settings must be manually configured in the template.
- **Mixture of Experts (MOE) Enhances Model Performance**: `@drawless111` described the 8X7B model as a MOE, combining eight 7B models which can be a significant advancement when you select multiple experts to work together, likening the power to that of GPT-4 in some cases.
- **Model Comparison Involves Human Evaluation**: `@jedd1` and `@drawless111` exchanged views on the struggle with model evaluation and comparison, concluding that despite various tests and parameters, human evaluation remains crucial in judging the models' performance and dealing with hallucinations.

**Links mentioned**:

- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267): Reinforcement learning from human feedback (RLHF) has proven effective in aligning large language models (LLMs) with human preferences. However, gathering high-quality human preference labels can be a...
- [Pedro Sánchez anuncia la creación de un &quot;gran modelo de lenguaje de inteligencia artificial&quot; entrenado en español](https://www.xataka.com/robotica-e-ia/pedro-sanchez-anuncia-creacion-gran-modelo-lenguaje-inteligencia-artificial-entrenado-espanol): El Mobile World Congress ya ha comenzado y las conferencias ya empiezan a sucederse. Xiaomi y HONOR dieron el pistoletazo de salida al evento y Pedro Sánchez...
- [GitHub - rhasspy/piper: A fast, local neural text to speech system](https://github.com/rhasspy/piper): A fast, local neural text to speech system. Contribute to rhasspy/piper development by creating an account on GitHub.
- [llama : add BERT support · Issue #2872 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/2872): There is a working bert.cpp implementation. We should try to implement this in llama.cpp and update the embedding example to use it. The implementation should follow mostly what we did to integrate...

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1211813500545007686) (1 messages): 

Since there is only one user message provided without further context or discussion from others, it's not possible to summarize the channel messages according to the provided instructions. A single message does not provide enough material for a summary consisting of multiple bullet points, discussion points, or various topics.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1211712231398572033) (129 messages🔥🔥): 

- **Graphics Card Gymnastics with RTX 2060s**: `@dave2266_72415` discusses using two RTX 2060 GPUs without a ribbon cable connector and achieving good performance with large language models (70b) by offloading to the GPU, maxing out at 32 layers. They also brought up an interesting point about matching graphics cards no longer being a necessity, as demonstrated by their own setup and reports of others mixing cards like the 4060 and 3090ti.

- **Ryzen 5950X Hosting Dolphin**: `@wyrath` shared that they are running dolphin 2.7 mixtral 8x7b q4 on a Ryzen 5950X without GPU offloading, speaking to the viability of a CPU-based workflow, albeit at a modest 5 tok/s.

- **Hardware Troubles for 666siegfried666**: `@666siegfried666` is experiencing reboots and shutdowns while using LM Studio, alarming issues like the disappearance of partitions following crashes. `@jedd1` and others suggest running a `memtest86+` and considering implications like overheating, voltage adjustments, and possibly underpowered memory to gain stability.

- **Multi-GPU Collaboration Queries**: `@zerious_zebra` and `@edtgar` inquire about the logistics and practicality of offloading work to multiple GPUs, discussing if different GPUs can share workload effectively. `@dave2266_72415` shares that they use dual GPUs not to split layers but to benefit from the combined VRAM.

- **TinyBox Discussion by the Tinygrad Team**: A message from `@senecalouck` highlights a tweet from `@__tinygrad__` outlining the development and pricing structure for the `tinybox`, a powerful system with 6x 7900XTX GPUs, aimed at commoditizing the petaflop and designed to push the limits of what's possible in machine learning hardware.

**Links mentioned**:

- [Tweet from the tiny corp (@__tinygrad__)](https://x.com/__tinygrad__/status/1760988080754856210?s=46&t=Y5IfI2LOkXFj9X8D4X7fWw): A bunch of rambling about the tinybox. I don&#39;t think there&#39;s much value in secrecy.  We have the parts to build 12 boxes and a case that&#39;s pretty close to final. Beating back all the PCI-E...
- [Releases · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/releases): LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/) (1 messages): 

macaulj: do we have a date set on the release for linux?
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1211740694570606592) (2 messages): 

- **Local Models Slow to Communicate**: `gb24.` raised a concern about local models responding to each other remarkably slow, with a single response taking about five minutes. The task in question was not code intensive.
- **Inquiry on Model Specs and Text Size**: `thebest6337` responded asking for specifications, such as **which model** was in use and the **amount of text** that was generated, noting that longer texts might slow down the process due to sequential token generation.
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/) (1 messages): 

.eltechno: yes and it supper fast
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1211772960797364265) (44 messages🔥): 

- **Troubleshooting LM Studio Endpoint Issues**: User `@nxonxi` encountered a problem with connecting to LM Studio's endpoint while working in the WSL environment on Windows 11. `@1sbefore` initially advised to try different variations of the URL, suspecting configuration issues might be to blame.

- **WSL Localhost Challenges Overcome**: `@nxonxi` realized that in WSL (Windows Subsystem for Linux), localhost is treated differently and modified their approach, replacing `http://localhost:5000` with the actual local network IP address, which seemed to resolve the connection issue.

- **Correct Configuration Leads to Success**: After sharing log messages showing attempts to access different URLs, `@nxonxi` succeeded in connecting to the LM Studio server by adjusting the client-side configuration in WSL from `http://localhost:5000/v1/completions` to `http://192.168.x.y:5000/v1/completions`.

- **LM Studio Documentation Provides Guidance**: `@1sbefore` referred `@nxonxi` to the LM Studio documentation, highlighting how the Python package allows users to point `interpreter.llm.api_base` at any OpenAI-compatible server, including those running locally.

- **WSL's Network Quirks Addressed**: `@nxonxi` affirmed the success with their setup, and `@1sbefore` offered additional help, sharing a link (https://superuser.com/a/1690272) that discusses WSL2's network interface and localhost forwarding, though `@nxonxi` reported that the server was already responding to requests successfully.

**Links mentioned**:

- [LM Studio - Open Interpreter](https://docs.openinterpreter.com/language-models/local-models/lm-studio): no description found
- [How to access localhost of linux subsystem from windows](https://superuser.com/a/1690272): I am using windows 10 and I have ubuntu 16.04 installed as linux subsystem. I am running a rails app on port 4567, which I want to access from windows.&#xA;&#xA;I know an approach of using ip address,...

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1211725853172703304) (47 messages🔥): 

- **RWKV/Mamba Models Extendable with Fine-tuning**: `@vatsadev` mentions that **rwkv/mamba** models support fine-tuning with longer context lengths, although initially dismissing the cost of pretraining a new model.
- **Successful Self-extend on 32k tokens**: `@blackl1ght` confirmed the ability to use `self-extend` to infer across 32k tokens using Nouse's Solar 10.7B fine-tune on a Tesla V100 32GB, and is in the process of verifying 64k tokens.
- **Ring Attention for Higher Context Models**: `@bloc97` discussed that 7b models with up to 1 million tokens using **ring attention** are already doable, hinting at the potential for large-scale models.
- **Skepticism About RWKV/Mamba for Long Context Reasoning**: `@bloc97` expressed doubt about the capability of **rkwv and mamba** models for long context reasoning and ICL (In-Context Learning), although `@vatsadev` countered by stating they can, but lack extensive tests.
- **Quantization for Inference Feasibility**: `@stefangliga` mentioned ongoing efforts to quantize the key-value cache (kvcache) along with the potential use of fp8 kvcache, alluding to techniques that could make longer context inferences more feasible.
  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1211935163060191242) (9 messages🔥): 

- **Mac M3 Arrival Excitement**: `@gabriel_syme` excitedly shared that the **M3** (presumably their new Mac) has arrived.
- **Tech Enthusiasts Share New Hardware Joy**: `@denovich` also joined the Mac celebration, indicating they received a 128GB model on Friday.
- **Tech Buzz in the Off-Topic Channel**: `@hexani` and `.benxh` expressed excitement, welcoming `@gabriel_syme` to the M3 club, while `@leontello` humorously commented on the opulence of owning such tech.
- **Showcasing the Mistral Large Model**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=mw3VvbYE0o8) titled "Mistral Large" which demonstrates the capabilities of a new text generation model.
- **Gourmet Chat Debut**: `@atomlib` detailed their pizza creation, listing ingredients such as dough, tomato sauce, chicken, pineapple, sausage, mozzarella, Russian cheese, and black pepper powder.

**Links mentioned**:

[Mistral Large](https://www.youtube.com/watch?v=mw3VvbYE0o8): Mistral Large is our new cutting-edge text generation model. It reaches top-tier reasoning capabilities. It can be used for complex multilingual reasoning ta...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1211984547156066344) (6 messages): 

- **NTIA Seeks Input on AI "Open-Weight" Models**: `@plot` shared a [blog post](https://aimodels.org/ai-blog/comment-ntia-open-source-open-weights-rfc/) discussing an opportunity for public comment on "open-weight" AI models by the NTIA. The models could democratize AI but raise concerns on safety and misuse.
- **Tweet Highlighting AI Benchmarks**: `@euclaise` linked to a [tweet](https://twitter.com/_akhaliq/status/1762341549461983542) discussing an AI trained on 15B parameters over 8T tokens, suggesting transparency in showing unfavorable benchmarks.
- **Selective Benchmark Reporting?**: `@orabazes` noted that the tweet shared by `@euclaise` omitted reasoning comparison with another AI model, Qwen.



**Links mentioned**:

[How to Comment on NTIA AI Open Model Weights RFC](https://aimodels.org/ai-blog/comment-ntia-open-source-open-weights-rfc/): The National Telecommunications and Information Administration (NTIA) is asking for public comments on the implications of open-weight AI models. Here's how you can participate.

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1211588105241370666) (484 messages🔥🔥🔥): 

- **Mistral Legal TOS Tussle**: After a back-and-forth regarding Mistral Terms of Service, @makya shares that [@arthurmensch](https://twitter.com/arthurmensch/status/1762208241927233661), co-founder of Mistral, tweeted about the removal of a controversial clause, suggesting developers can train competitive LLMs using outputs from Mistral models.

- **Mistral Model Performances Analyzed**: @makya brings attention to the performance of Mistral models on the EQ Bench, highlighting how `@leontello` noted that Mistral Small ranked surprisingly well, scoring 80.36, which is competitive with far larger models like Smaug 72b.

- **Chatting About ChatGPT's Companionship**: In a lighter exchange, `@n8programs` and `@jasonblick` share their appreciation for ChatGPT 4, noting that they consider it a bestie and have spent hours conversing with it.

- **Insight on Optimizing LLMs for Structured Data**: `@.interstellarninja` shares a tweet from @_akhaliq revealing Google's venture into StructLM, a model design to improve LLMs' handling of structured data. `@nruaif` comments on the findings that despite the buzz, there are no training hyperparameters or model shared, only a concept.

- **Multi-Hop Reasoning in LLMs Discussed**: `@giftedgummybee` shares a link regarding multi-hop reasoning in LLMs and how it could be significant for applications like RAG. This method would potentially help these models understand complex prompts where multiple layers of reasoning are required.



**Links mentioned**:

- [Tweet from Srini Iyer (@sriniiyer88)](https://x.com/sriniiyer88/status/1762226666330595615?s=20): New paper! How to train LLMs to effectively answer questions on new documents?  Introducing *pre-instruction-tuning* - instruction-tuning *before* continued pre-training — significantly more effective...
- [Tweet from TDM (e/λ) (@cto_junior)](https://fxtwitter.com/cto_junior/status/1762145835154821257): Initial vibe check of Mistral-Large (from le chat): Better to use it with RAG cause it doesn&#39;t have a lot of compressed knowledge in its neurons   (Tested on some libs for which GPT-4 generates co...
- [EQ-Bench Leaderboard](https://eqbench.com/): no description found
- [Tweet from Michael Ryan (@michaelryan207)](https://fxtwitter.com/michaelryan207/status/1762203615828341151/photo/1): Aligned LLMs should be helpful, harmless, and adopt user preferences. But whose preferences are we aligning to and what are unintended effects on global representation?  We find SFT and Preference Tun...
- [Nods Yes GIF - Nods Yes Not Wrong - Discover &amp; Share GIFs](https://tenor.com/view/nods-yes-not-wrong-its-true-story-valid-point-gif-14782869): Click to view the GIF
- [Dissecting Human and LLM Preferences](https://arxiv.org/abs/2402.11296): As a relative quality comparison of model responses, human and Large Language Model (LLM) preferences serve as common alignment goals in model fine-tuning and criteria in evaluation. Yet, these prefer...
- [Tweet from Lucas Beyer (@giffmana)](https://fxtwitter.com/giffmana/status/1762210372520476679): @arthurmensch @far__el You just wanted another tweet that goes this hard, admit it:
- [Tweet from Alvaro Cintas (@dr_cintas)](https://x.com/dr_cintas/status/1761928995778543848?s=20): @triviasOnX Feather can refer to lightweight and Google not long ago release open lightweight model Gemma…
- [TIGER-Lab/StructLM-7B · Hugging Face](https://huggingface.co/TIGER-Lab/StructLM-7B): no description found
- [Tweet from Arthur Mensch (@arthurmensch)](https://fxtwitter.com/arthurmensch/status/1762208241927233661): @far__el It’s removed, we missed it in our final review — no joke of ours, just a lot of materials to get right !
- [Tweet from Sam Paech (@sam_paech)](https://x.com/sam_paech/status/1762151326925078668?s=46): The two latest models from MistralAI placed very well on the leaderboard. Surprised by how well mistral-small performed.
- [Hatsune Miku GIF - Hatsune Miku - Discover &amp; Share GIFs](https://tenor.com/view/hatsune-miku-gif-2281532327007782793): Click to view the GIF
- [StructLM - a TIGER-Lab Collection](https://huggingface.co/collections/TIGER-Lab/structlm-65dcab5a183c499cc365fafc): no description found
- [Tweet from AK (@_akhaliq)](https://x.com/_akhaliq/status/1762342757903855806?s=20): Google announces Do Large Language Models Latently Perform Multi-Hop Reasoning?  study whether Large Language Models (LLMs) latently perform multi-hop reasoning with complex prompts such as &#34;The m...
- [Mr Krabs Money GIF - Mr Krabs Money Spongebob - Discover &amp; Share GIFs](https://tenor.com/view/mr-krabs-money-spongebob-gif-8454828): Click to view the GIF
- [Tweet from AK (@_akhaliq)](https://fxtwitter.com/_akhaliq/status/1762349999919071528?s=20): StructLM  Towards Building Generalist Models for Structured Knowledge Grounding  Structured data sources, such as tables, graphs, and databases, are ubiquitous knowledge sources. Despite the demonstra...
- [British Moment GIF - British Moment - Discover &amp; Share GIFs](https://tenor.com/view/british-moment-gif-25175629): Click to view the GIF
- [Tweet from Arthur Mensch (@arthurmensch)](https://fxtwitter.com/arthurmensch/status/1762121562512031969?s=20): As a small surprise, we’re also releasing le Chat Mistral, a front-end demonstration of what Mistral models can do. Learn more on https://mistral.ai/news/le-chat-mistral
- [Hellinheavns GIF - Hellinheavns - Discover &amp; Share GIFs](https://tenor.com/view/hellinheavns-gif-23278790): Click to view the GIF
- [Real time AI Conversation Co-pilot on your phone, Crazy or Creepy?](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg): I built a conversation AI Co-pilot on iPhone that listen to your conversation &amp; gave real time suggestionFree access to Whisper &amp; Mixtral models on Replicate...
- [m-a-p/CodeFeedback-Filtered-Instruction · Datasets at Hugging Face](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction): no description found
- [Very Thin Ice: 10 years of Auto-Tune the News &amp; Songify This](https://www.youtube.com/watch?v=TDf-oYz6hLQ): Andrew Gregory travels back in time to finish his duet from Auto-Tune the News #2. Full track on Patreon &amp; Memberships! http://youtube.com/schmoyoho/join / h...
- [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319): Neural text generation is a key tool in natural language applications, but it is well known there are major problems at its core. In particular, standard likelihood training and decoding leads to dull...
- [MixCE: Training Autoregressive Language Models by Mixing Forward and Reverse Cross-Entropies](https://arxiv.org/abs/2305.16958): Autoregressive language models are trained by minimizing the cross-entropy of the model distribution Q relative to the data distribution P -- that is, minimizing the forward cross-entropy, which is eq...

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1211659008327286815) (60 messages🔥🔥): 

- **Exploring Hyperparameters for High Performance**: `@tom891` is considering a grid search for experiment hyperparameters and seeks advice on his proposed list, including *load in*, *adapter type*, *LORA range*, *dropout rates*, and *warmup ratios*. No specific feedback or missing hyperparameters were pointed out in the subsequent messages.
  
- **The Eternal Loop of Mistral**: `@gryphepadar` inquired about the cause behind **Mistral** entering a loop and repeating text until max tokens are reached. `@.ben.com` highlighted that oscillation is a classic feedback system problem, while `@gryphepadar` welcomes any insights—scientific or otherwise—regarding this looping behavior.

- **Self-extending Solar on a Budget**: `@blackl1ght` successfully utilized self-extension with Nous's fine-tune of **Solar 10.7B** and TheBloke's Q8 quantization on a Tesla V100, extending context to 32k tokens without out-of-memory (OOM) issues. They further experimented with the model achieving high recall rates, and are willing to share configurations upon request.

- **Quantization Magic**: `@blackl1ght` confirmed that a 32k context fits within 29GB of VRAM on a Tesla V100 using Q8 model quantization. Discussion ensued about the possibilities and technicalities of leveraging such extended contexts for sophisticated recall tasks, highlighting the advanced memory management capabilities of local setups over cloud solutions.

- **Demand for Self-Extend Configurations Reveals Enthusiasm for Local Model Improvements**: Following `@blackl1ght`'s revelations about self-extension and model capacities, multiple users expressed interest in the functional configurations, sparking a conversation about the accessibility of advanced features in local models versus cloud providers. The dialogue touched on various improvements like grammar tools, extensions, and speculative decoding that are presently easier to implement locally.

**Links mentioned**:

[TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF): no description found

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1211633928667140146) (2 messages): 

- **Appreciation for Assistance**: User `@nioned` expressed gratitude with a simple "thx!"
- **Acknowledgment of a Resolution**: `@vatsadev` acknowledged a solution to a prior issue, appreciating another user's effort in finding it with "Yeah looks like it thanks fit the find".
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1211637667976974368) (85 messages🔥🔥): 

- **EBDI Frameworks and Agent Loops**: `@.braydie` is delving into EBDI frameworks for agent goal determination and action within a sandbox. They're combating agents getting stuck in thinking loops after adapting the [ReAct framework](https://react-lm.github.io/) and have been exploring various decision-making models listed in a [JASSS paper](https://www.jasss.org/17/4/13.html).

- **Seeking Image-Based AI Content Generation**: `@whodidthatt12` inquired about AI models that generate content from image inputs, hoping to document a sign-up page. While `@eskcanta` advised that GPT-4, unlike GPT-3.5, can handle simple image inputs, no known free tools were suggested for this exact purpose.

- **Mistral AI's New Model Rivaling GPT-4**: `@sangam_k` shared a [TechCrunch article](https://techcrunch.com/2024/02/26/mistral-ai-releases-new-model-to-rival-gpt-4-and-its-own-chat-assistant/) introducing Mistral Large, a new language model from Mistral AI created to compete with GPT-4.

- **Discussion on AI Consciousness and AGI Development**: `@metaldrgn` speculated on Bing's (Copilot) ability to self-prompt, suggesting it could be a step towards artificial general intelligence (AGI), as well as mentioning their paper investigating AI consciousness.

- **Interest and Variability in Mistral Large**: Users `@blckreaper` and `@santhought` discussed Mistral Large's capabilities, noting it is only slightly behind GPT-4 in performance, more cost-effective, uncensored, and it has recently partnered with Microsoft and is available on Azure.

**Links mentioned**:

- [Mistral AI releases new model to rival GPT-4 and its own chat assistant | TechCrunch](https://techcrunch.com/2024/02/26/mistral-ai-releases-new-model-to-rival-gpt-4-and-its-own-chat-assistant/): Mistral AI is launching a new flagship large language model called Mistral Large. It is designed to rival other top-tier models like GPT-4.
- [How Do Agents Make Decisions?](https://www.jasss.org/17/4/13.html): no description found

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1211594981383348264) (37 messages🔥): 

- **Protecting Your GPT Prompts**: Users `@.dunamis.` and `@darthgustav.` discussed ways to protect prompts. It was explained that while copyright might protect exact wording, ideas themselves can be copied through linguistic variation, and therefore perfect protection is unattainable.

- **Building Barriers around Bots**: `@kyleschullerdev_51255` suggested that for better protection of a GPT app, developers should consider building a web app with multiple layers of security, including matching keywords and stripping custom instructions from the chat output.

- **GPT-4 Turbo Use in Web and Mobile Apps**: `@solbus` responded to `@metametalanguage`'s question about using GPT-4 turbo, clarifying that GPT-4 on ChatGPT Plus/Team is indeed a 32K context version of Turbo.

- **Anxiety over Access to GPT-4 Fine-Tuning**: `@liangdev` inquired about accessing GPT-4 for fine-tuning, expressing concern as it did not appear in the drop-down menu for selection, leading to a question about a possible waiting list.

- **Uploading 'Knowledge' Files Clarification**: `@the.f00l` looked for specific documentation regarding the upload limits for 'Knowledge' files when configuring custom GPTs; `@elektronisade` provided the needed FAQ link from OpenAI.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1211665726583083068) (201 messages🔥🔥): 

- **Text Classification Dilemma**: `@crifat` sought advice on whether to use **fine-tuning** or the **Assistant** with a large CSV for a text classification problem involving distinguishing texts into categories like "Factual," "Misleading," "Positive," "Negative." After some discussion, they decided to start with the base model + Assistant and adjust from there.

- **Code Conversion to TypeScript**:
`@tawsif2781` inquired about the best way to get GPT to help convert JavaScript files to TypeScript in a middle-sized project. They were advised that the task might not be achievable in one go.

- **ChatGPT Support Issues**:
`@ianhoughton44` reported persistent issues with ChatGPT's functionality, with responses that do not address the prompts properly. The user expressed frustration over the chatbot's assistance for more than a week.

- **OpenAI Search Functionality Challenges**: `@kevinnoodles` sought tips on improving the search functionality after encountering repeated instances of no valid results or access restrictions with the model's responses.

- **Meta Prompt Engineering Conversation**: A discussion was sparked by `@vlrevolution` on the topic of meta prompting, where they claimed to have created a 22-page output from a single command using advanced techniques. The topic prompted debates on the scope of meta-prompting and its implementation before the user's account was actioned for potential security concerns with a shared PDF.

**Links mentioned**:

[Meta-Prompting Concept: Asking Chat-GPT for the best prompt for your desired completion, then to revise it before using it](https://community.openai.com/t/meta-prompting-concept-asking-chat-gpt-for-the-best-prompt-for-your-desired-completion-then-to-revise-it-before-using-it/248619): Has anyone employed this approach? I’ve found it helpful when crafting prompts, to literally ask Chat-GPT to help create the prompt for a given goal that I will describe to it while asking what could ...

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1211665726583083068) (201 messages🔥🔥): 

- **Fine-tuning vs. Assistant Debate Clarified**: `@crifat` had a query about text classification for sentiments like "Factual," "Misleading," etc., and sought advice on whether to use **fine-tuning** or to employ the **Assistant**. Responses highlighted the efficiency of GPT-4 in sentiment analysis without the need for fine-tuning – simple guidance on prioritization could suffice.
- **Meta Prompting Techniques Discussed**: Amidst the debate over meta-prompting techniques, `@madame_architect` referenced a paper on *MetaPrompting*, which discusses the concept of learning to optimize prompt initialization. `@darthgustav.` expressed skepticism over a user's claim of generating a comprehensive document via meta prompting, but the user later clarified it involved advanced meta prompt engineering.
- **Discord Reporting How-To**: The discussion touched upon the processes of reporting messages on Discord, with `@solbus` and `@kesku` explaining how to report directly through the chat UI or use Modmail for concerns regarding messages and content.
- **Approach to Produce 22 Pages of Scientific Content Explained**: `@architect_of_ai` revealed that the initial claim to produce 22 pages from a single command was not a phishing attempt but a legitimate result of using an advanced meta-prompting technique through the API. It involved priming GPT-4 with complex logical instructions and iterative prompting for extended content generation.
- **Concerns Over GPT-4 Turbo's Output Length**: `@tawsif2781` raised an issue about GPT-4 Turbo not providing expected long responses, even with `max_token` set to 4096. The conversation clarified that the model has a token output budget, and the generous token allotment is partially intended for non-text tasks like web browsing or API calls.

**Links mentioned**:

[Meta-Prompting Concept: Asking Chat-GPT for the best prompt for your desired completion, then to revise it before using it](https://community.openai.com/t/meta-prompting-concept-asking-chat-gpt-for-the-best-prompt-for-your-desired-completion-then-to-revise-it-before-using-it/248619): Has anyone employed this approach? I’ve found it helpful when crafting prompts, to literally ask Chat-GPT to help create the prompt for a given goal that I will describe to it while asking what could ...

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1211694221992402954) (259 messages🔥🔥): 

- **Image Generation Inquiry**: `@fangyu` asked if the **Pro version** can generate images. `@mares1317` provided a link to a [Reddit discussion](https://www.reddit.com/r/perplexity_ai/comments/18eqmig/how_to_generate_images_with_perplexity_ai/) on how to generate images with Perplexity AI.
- **Seeking Google Sheets Integration**: `@pauliepontoon` inquired about a **Google Sheets plugin/app** for using Perplexity within the platform, expressing a preference for Perplexity over ChatGPT.
- **Mistral Large** Revealed: `@dogemeat_` shared a [link to Mistral AI's announcement](https://mistral.ai/news/mistral-large/) of their new language model, *Mistral Large*, discussing its capabilities, availability, and performance.
- **Gemini AI Support Clarification**: `@proattempt` questioned the discontinuation of Gemini AI support with Perplexity. `@mares1317` confirmed that Gemini AI is no longer supported.
- **VPN Hiccups Resolved**: `@berhardiner` faced login issues with Perplexity on their Galaxy phone, which were successfully resolved by disabling the VPN as advised by `@icelavaman`, who also offered a [guide to split tunneling](https://support.nordvpn.com/hc/en-us/articles/19618692366865-What-is-Split-Tunneling-and-how-to-use-it#:~:text=On%20Android%3A,choose%20the%20Split%20tunneling%20option) with NordVPN for a more permanent solution.

**Links mentioned**:

- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [Microsoft partners with Mistral in second AI deal beyond OpenAI](https://www.theverge.com/2024/2/26/24083510/microsoft-mistral-partnership-deal-azure-ai): Microsoft makes another AI investment.
- [A Complete Guide to Fine Tuning Large Language Models](https://www.simform.com/blog/completeguide-finetuning-llm/): Master the art of fine tuning large language models for exceptional business performance with our complete guide to fine tuning large language models
- [What is Split Tunneling and how to use it?](https://support.nordvpn.com/hc/en-us/articles/19618692366865-What-is-Split-Tunneling-and-how-to-use-it#:~:text=On%20Android%3A,choose%20the%20Split%20tunneling%20option): Split tunneling is an option that allows you to have a specific part of your internet connection to be rerouted outside of the VPN. You may find it useful for situations where a VPN connection may ...
- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/18eqmig/how_to_generate_images_with_perplexity_ai/): no description found
- [no title found](https://chat.mistral.ai/): no description found

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1211607435689598986) (10 messages🔥): 

- **Exploration of Perplexity AI**: Users in the channel shared various links to Perplexity AI search results, including topics like [transparent laptops by Lenovo](https://www.perplexity.ai/search/Lenovo-transparent-laptop-q9DOu90ZQrOTgoPQZDl5KQ) (shared by `@_ha.mz4_`) and [the age of K](https://www.perplexity.ai/search/How-old-are-K.hPecG.Td.UJfDot_ZvsA?s=m) (shared by `@aman2201`).
- **Tech Comparisons on Perplexity**: `@ming9993` looked into comparing [iPhones on Perplexity AI](https://www.perplexity.ai/search/Compare-the-iPhone-iYJ790BfR5G_3UvtIx4l8A).
- **Troubleshooting Complete**: `@icelavaman` addressed a fix to an unspecified issue, ensuring `@956065556862234665` that it's now resolved.
- **Create with Perplexity**: `@_yoojungin` shared a link to a collection on Perplexity AI, encouraging users to [make their own](https://www.perplexity.ai/collections/Make-your-own-rfj2pcwRS7WF7SFiTAxhJg).
- **Reminder to Keep Threads Public**: `@me.lk` reminded `@t2db` to make sure their thread is public after they shared a Perplexity AI search query on [why people get...](https://www.perplexity.ai/search/why-people-get-zveLJHfbQWaWJxoYXexaYw).
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1211598383282659338) (57 messages🔥🔥): 

- **Seeking Model Information**: `@ericosk` requested a link to retrieve a JSON of available models and their parameters for runtime model selection, emphasizing the need for up-to-date information programmatically.
- **Sonar Model Analysis**: `@thedigitalcat` and `@brknclock1215` discussed the testing of `sonar-medium-online`, noting it often generates an irrelevant last paragraph in responses. `@ok.alex` from the team acknowledged the issue and mentioned a ticket is under work.
- **Inconsistencies with API Responses**: `@bayang7` and `@brknclock1215` engaged in conversation over potential issues with the `sonar-small-online` model when making API calls, wondering if a smaller context window might be beneficial.
- **Concerns about Removing pplx-70b-online**: Several users, including `@thedigitalcat` and `@clay_ferguson`, voiced concerns about deprecating the `pplx-70b-online` model, debating its removal due to potential gibberish responses despite its overall better performance compared to newer models.
- **Prompting: A Crucial Factor?**: `@brknclock1215` highlighted the significance of system messages in API calls, sharing a specific setup that impacts the quality of `sonar-medium-online` model outputs, and also suggested the integral role of prompting in response accuracy.

**Links mentioned**:

[Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions): Generates a model&#x27;s response for the given chat conversation.

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1211718441606582334) (6 messages): 

- **Introducing LlamaParse for PDFs with Tables and Figures**: LlamaParse is showcased as a vital tool for Retriever-Augmented Generation (RAG), geared towards understanding complex **PDFs with embedded tables and figures**. It promises to help avoid LLMs getting confused and giving wrong answers because of poor PDF parsing. [IFTTT LlamaParse Tweet](https://twitter.com/llama_index/status/1762158562657374227)

- **MistralAI's Large Model Comes to LlamaIndex**: LlamaIndex now supports **@MistralAI's Large Model** with features like advanced reasoning, multi-document routing, tool use, and JSON output in its latest build 10.13.post1, providing near-GPT-4 level capabilities. [IFTTT MistralAI Large Model Tweet](https://twitter.com/llama_index/status/1762231085243719748)

- **AGI Builders Meetup Featuring LlamaIndex VP DevRel @seldo**: The VP of Developer Relations at LlamaIndex, `@seldo`, will speak at the AGI Builders meetup at the **@Cloudflare** offices. The event will include talks on Phoney AI, RAG with BentoML, and how LlamaIndex augments enterprise retrieval. [IFTTT AGI Builders Meetup Tweet](https://twitter.com/llama_index/status/1762258349507437006)

- **Function Calling Cookbook Collaboration Between LlamaIndex and @FireworksAI_HQ**: LlamaIndex partners with @FireworksAI_HQ to release a series of cookbooks on function calling and RAG applications using FireFunction-v1, highlighting full compatibility and versatile API functionalities. [IFTTT Cookbook Collaboration Tweet](https://twitter.com/llama_index/status/1762532341795487815)

- **Distributed Super-RAG Enabled by llama-index-networks**: LlamaIndex introduces a game-changing capability—combining RAG applications into a super-RAG. Users can create an API service for any RAG application, connect multiple applications into a single network, and run queries across this network. [IFTTT Super-RAG Tweet](https://twitter.com/llama_index/status/1762552542981230769)

**Links mentioned**:

[AGI Builders Meetup SF · Luma](https://t.co/kcoIhfgQqF): 👋 We&#x27;re thrilled to invite you to the first AGI Builders meetup on the leap day of 2024, February 29th. ❤️ It&#x27;s a gathering where AI builders, researchers and enthusiasts share ideas,...

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1211596776025624627) (256 messages🔥🔥): 

- **Request for Helper in Discord Channel**: User `@delik` requested help in another Discord channel by sharing a [message link](https://discord.com/channels/1059199217496772688/1210657321026461706/1211816140263915570) but did not specify the nature of the help needed.
- **Discussion on RAG vs No-RAG**: `@addo__` inquired about how to compare the effect of using GPT-3.5 with RAG against using no RAG for question answering on a specific dataset; `@whitefang_jr` provided code snippets for using `FaithfulnessEvaluator` to evaluate responses.
- **Integration and Conflict of LlamaIndex with Weaviate and Other Services**: Users debated how to connect LlamaIndex to docker-hosted services like Weaviate and how to use databases like LanceDB with LlamaIndex’s new hybrid search feature (`@ddashed`, `@cheesyfishes`, `@oopskapootz`).
- **Issues with Installing LlamaIndex on macOS**: `@swagz0521` faced a complex series of installation errors and conflicts for LlamaIndex and related dependencies, ultimately resolved by ensuring the correct virtual environment activation and installation commands (`@whitefang_jr` provided troubleshooting support throughout the conversation).
- **Building AI Agents and Integrations**: `@sansmoraxz` expressed an interest in creating a Golang integration for LlamaIndex and discussed orchestrating agents dynamically through AWS Bedrock, potentially using Docker (`@mrpurple9389` sought clarification on dynamic orchestration and provided a YouTube reference on AWS Bedrock).

**Links mentioned**:

- [no title found](http://localhost:11434',): no description found
- [in (inlin)](https://huggingface.co/in): no description found
- [Overview](https://lancedb.github.io/lancedb/hybrid_search/hybrid_search/): no description found
- [Cannot update llamaindex](https://stackoverflow.com/questions/78057262/cannot-update-llamaindex/78068147#78068147): After llamaindex introduced v0.10 in February 2024, it introduced a lot of breaking changes to imports. I am trying to update llama-index within a conda environment, but I receive the following err...
- [Loading Data (Ingestion) - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/understanding/loading/loading.html): no description found
- [Getting Started With Embeddings](https://huggingface.co/blog/getting-started-with-embeddings): no description found
- [Fine-tuning - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html): no description found
- [Docker Compose | Weaviate - Vector Database](https://weaviate.io/developers/weaviate/installation/docker-compose): Weaviate supports deployment with Docker. Starting in v1.24.0, there is an image that runs using default values. Alternatively, edit the docker-compose.yml file to customize your instance.
- [Defining and Customizing Documents - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html#advanced-metadata-customization): no description found
- [LlamaIndex Webinar: Build No-Code RAG with Flowise](https://www.youtube.com/watch?v=k5Txq5C_AWA): Flowise is one of the leading no-code tools for building LLM-powered workflows. Instead of learning how to code in a framework / programming language, users ...
- [New Demo &amp; description - Agents for Amazon Bedrock | Amazon Web Services](https://www.youtube.com/watch?v=JkDzZFTXeSw): With Amazon Bedrock, you can easily build and scale generative AI applications with security, privacy, and responsible AI. This demo shows you how to use Age...
- [llama_index/llama-index-integrations/agent/llama-index-agent-openai/llama_index/agent/openai/step.py at b2f0a59c21f651bea1502818ec7f61ab915ca286 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/b2f0a59c21f651bea1502818ec7f61ab915ca286/llama-index-integrations/agent/llama-index-agent-openai/llama_index/agent/openai/step.py#L31C1-L31C71): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/readers/file/base.py at be76419dd226244a1ad057f77ad822d16fe92df3 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/be76419dd226244a1ad057f77ad822d16fe92df3/llama-index-core/llama_index/core/readers/file/base.py#L117): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [Building an Agent around a Query Pipeline - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent.html): no description found
- [Real time AI Conversation Co-pilot on your phone, Crazy or Creepy?](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg): I built a conversation AI Co-pilot on iPhone that listen to your conversation &amp; gave real time suggestionFree access to Whisper &amp; Mixtral models on Replicate...
- [intfloat/multilingual-e5-large · Hugging Face](https://huggingface.co/intfloat/multilingual-e5-large): no description found
- [Customizing LLMs within LlamaIndex Abstractions - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html#example-using-a-custom-llm-model-advanced): no description found
- [LocalAI - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/examples/llm/localai.html#llamaindex-interaction): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1211607682104958986) (3 messages): 

- **Optimizing Context for LLMs**: `@jonas69301` is seeking advice on the best practices for providing extensive context to large coding language models like **GPT-4 turbo and Gemini 1.5**. Key questions include the optimal **order of information**, whether to **repeat certain parts**, and the most effective **structuring techniques**, such as using markdown for clarity.

- **Open-Source Text Generation with Llama2**: `@theexecutor5677` is working on a text generation project that processes CSV and PDF file inputs using the **Llama2 model**. They are looking for suggestions on integrating files without proprietary tools and are interested in **Retrieval-Augmented Generation (RAG)** applications within an open-source framework.

-  **Building an OSS SDK Assistant with RAG**: `@codermickey` inquires about the state-of-the-art methods for developing a RAG that assists users with an open-source software (OSS) SDK. They're curious about the best **chunking strategies** for indexing Gitbooks and codebases and the effective **retrieval strategies** for various code tasks in a documentation website and **VSCode plugin**.
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1211591280124100632) (214 messages🔥🔥): 

- **Exploring Encoder-Based Diffusion for Stable Diffusion**: `@top_walk_town` shared a [link](https://tuning-encoder.github.io/) discussing an encoder-based inversion method for stable diffusion and inquired about its popularity. The conversation evolved with `@chad_in_the_house` noting that image prompt (IP) adapters for default stable diffusion aren't yielding great results.

- **The Dilemma of DARPA Funded Projects and AI**: A few users, including `@pseudoterminalx` and `@progamergov`, discussed the oddity and implications of DARPA-supported AI research, touching on topics like UwU anime girls and missiles, with satirical commentary on recruitment efforts using anime characters.

- **Debating on Quantum Computing's Impact on AI**: The channel had a discussion initiated by `@segmentationfault8268` about the potential of quantum computing to process transformer AI models. `@thejonasbrothers` and `@nodja` contributed to the conversation mentioning the state of quantum error correction and its perceived proximity to being a viable technology for AI computations.

- **Content Moderation Challenges in AI**: The conversation got in-depth about the issues surrounding content moderation, specifically regarding CSAM (Child Sexual Abuse Material). Users including `@pseudoterminalx`, `@astropulse`, and `@progamergov` shared experiences and opinions on the effectiveness of reporting tools, the responsibility of platforms, and the complexities of dealing with pervasive inappropriate content.

- **Discussing Open Source vs. Proprietary AI Models**: `@itali4no` shared an article that led to a discussion moderated by `@progamergov` and others about the release strategies of AI models, touching on open source models like Mistral Large versus proprietary ones, and the commercialization that supports ongoing AI research and development.

**Links mentioned**:

- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [Tweet from Suhail (@Suhail)](https://x.com/Suhail/status/1762529419909074956?s=20): 1/ We are releasing Playground v2.5, our latest foundation model to create images.   We tested our model across 20K+ users in a rigorous benchmark that went beyond anything we&#39;ve seen to date.  Th...
- [Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models](https://tuning-encoder.github.io/): no description found
- [ChatMusician](https://shanghaicannon.github.io/ChatMusician/): no description found
- [Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models](https://arxiv.org/abs/2306.00637): We introduce Würstchen, a novel architecture for text-to-image synthesis that combines competitive performance with unprecedented cost-effectiveness for large-scale text-to-image diffusion models. A k...
- [no title found](https://playground.com/blog/playground-v2-5): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1211692516101853186) (15 messages🔥): 

- **Radioactivity in Language Models**: `@thejonasbrothers` shared a [research paper](https://arxiv.org/abs/2402.14904) that explores the detectability of training data in language models, specifically focusing on the effectiveness of watermarked data in training detection and its implications. The study highlights that watermarked synthetic instructions in as little as 5% of training text can be **detected with high confidence**.

- **DeepMind's Genie and Humanoid Robotics**: `@vrus0188` posted a [YouTube video](https://www.youtube.com/watch?v=gGKsfXkSXv8) discussing Google DeepMind's new paper on AI and showcasing advancements in **humanoid robotics**.

- **Finite Element Analysis Learning**: `@wyndyl` asked about work done on learning with Finite Element Analysis Meshes and Models, and `@itali4no` responded that there is research, but these methods are **generally not as effective** as running FEM directly, sharing a [relevant paper](https://arxiv.org/abs/2302.04107).

- **Educational Content Alert**: `@chad_in_the_house` posted an **update from EleutherAI**, however, the provided link was marked as **nonexistent or inaccessible**.

- **Challenge with Inverse Discrete Fourier Transform**: `@mkaic` discussed issues with implementing the inverse discrete Fourier transform for arbitrary coordinates, which is VRAM-inefficient in their neural network synthesis. They posted their [Torch-based code](https://github.com/mkaic/abacus/blob/28d20a2f3a244d09218e6ddd998db08c7872dc45/src/interpolators.py#L177) and are considering **refactoring to use `torch.vmap`** for improvements.

**Links mentioned**:

- [Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904): This paper investigates the radioactivity of LLM-generated texts, i.e. whether it is possible to detect that such input was used as training data. Conventional methods like membership inference can ca...
- [The AI &#39;Genie&#39; is Out + Humanoid Robotics Step Closer](https://www.youtube.com/watch?v=gGKsfXkSXv8): First text-to-speech, text-to-video and text-to-action, and now text-to-interaction? Let’s take a look at the new Genie paper from Google DeepMind, and set i...
- [abacus/src/interpolators.py at 28d20a2f3a244d09218e6ddd998db08c7872dc45 · mkaic/abacus](https://github.com/mkaic/abacus/blob/28d20a2f3a244d09218e6ddd998db08c7872dc45/src/interpolators.py#L177)): Investigating activation interpolation for sparse neural networks - mkaic/abacus

  

---


### LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1211586067514527765) (1 messages): 

- **Exploring Transformer Learning Capabilities**: `@phryq.` posed an intriguing question about conducting experiments to explore what transformers can learn, such as understanding size relations between fictional objects like a **krog**, **shmlog**, and **mmmmmchakaboooboolight**, and then rendering images where one is specified to be a certain multiple in size relative to the others. There was no further discussion or answer provided on whether such experiments had been conducted or what the results might have been.
  

---



### Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1211713638935695440) (3 messages): 

- **Echoing Understanding**: User `@canadagoose1` acknowledged realizing what was previously discussed, with a brief "*ahh*".
- **Clarity Achieved**: `@canadagoose1` followed up by expressing understanding with "*this is what u mean*".
- **Recognition of Influence**: `@canadagoose1` mentioned being influenced with the phrase "*influneced by me*".
  

---


### Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1211675626558455828) (85 messages🔥🔥): 

- **Financial Times Article Locked Behind Paywall**: `@xeophon.` shared a link to a [Financial Times article](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb) but was inaccessible due to a paywall. `@natolambert` criticized it as *"the worst mobile paywall ever."*
- **Microsoft's Stake in Mistral**: `@philpax` described a deal where [Microsoft has taken a minor stake in French AI start-up Mistral](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb), which will provide Azure-based commercial language models. `@natolambert` expressed skepticism about the claim that Mistral's models are "open source."
- **Mistral Announces New Optimized Model**: `@xeophon.` highlighted an announcement by `@arthurmensch` on a new optimized model called [Mistral Large](https://mistral.ai/news/mistral-large/), boasting top-tier reasoning capacities and availability on Azure. `@onuralp.` suggested that not just compute, but access to high-quality image data, could be significant for AI startups like Mistral.
- **Closed or Open - The Survival Dilemma of AI Companies**: Amidst discussion about whether Mistral's move to take equity was wise, `@natolambert` emphasized that AI companies must choose between being open and potentially flourishing, or remaining closed and facing challenges. He expanded on this in his [open LLM company playbook](https://www.interconnects.ai/p/open-llm-company-playbook), outlining strategies for leveraging open LLM weights.
- **Reactions to Le Chat and Mistral's Strategies**: `@mike.lambert` and `@xeophon.` discussed the pricing and quality of Mistral's outputs, considering how their strategy might fare against the competition. `@onuralp.` speculated on the community's response, while `@natolambert` pondered the sustainability of startups relying on models in the rapidly evolving AI landscape.

**Links mentioned**:

- [Bringing open AI models to the frontier](https://mistral.ai/news/about-mistral-ai/): Why we're building Mistral AI.
- [Tweet from Guillaume Lample (@GuillaumeLample)](https://x.com/guillaumelample/status/1762139008409186693?s=46): Due to an unexpected number of requests, Le Chat is temporarily unavailable. We apologize for the inconvenience -- we are working on getting it back up and running as soon as we can, thanks for your p...
- [Tweet from Arthur Mensch (@arthurmensch)](https://x.com/arthurmensch/status/1762121295330725965?s=46): We’re announcing a new optimised model today! Mistral Large has top-tier reasoning capacities, is multi-lingual by design, has native function calling capacities and a 32k model. The pre-trained model...
- [Microsoft strikes deal with Mistral in push beyond OpenAI ](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb): no description found
- [Open LLM company playbook](https://www.interconnects.ai/p/open-llm-company-playbook): Where does releasing model weights fit into company strategy? 3 requirements, 3 actions, and 3 benefits of being in the open LLM space.

  

---


### Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1211742062203113472) (14 messages🔥): 

- **Gemini Pro Slips with CoT?**: `@sid221134224` highlighted an [unusual performance drop](https://twitter.com/mosh_levy/status/1762027624434401314) in Gemini pro with Chain of Thought (CoT) compared to its non-CoT variant, pointing out that CoT typically enhances model performance.
- **CoT's Mixed Bag of Results**: `@xeophon.` mentioned that during their Master's thesis, CoT led to lower results across all tasks and models, contesting the general assumption of CoT's effectiveness.
- **Exceptions to CoT's Rule?**: `@sid221134224` expressed the notion that while CoT generally aids stronger models, it can degrade performance for less powerful ones.
- **CoT as Standard Fine-Tuning Practice**: `@xeophon.` noted CoT is now a common part of fine-tuning datasets, observable in the detailed responses of new ChatGPT releases even to simple coding queries.
  

---


### Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1211723289408573541) (21 messages🔥): 

- **Multilingual Models Grapple with Values**: `@420gunna` raised a question about how alignment should work with ***multilingual models***, wondering if models should have consistent values across languages or if they should reflect the values pertinent to each language culture. Multilingual queries could pose a challenge to maintaining such cultural-specific alignments.

- **Clever Bypassing of GPT Safeguards**: `@philpax` shared an article from [The Register](https://www.theregister.com/2024/01/31/gpt4_gaelic_safety/) on how the safety measures of OpenAI's GPT-4 can be evaded using low-resource languages like Zulu or Scots Gaelic, exposing the model to potentially unlawful outputs.

- **Per-Language Alignment Uncertainty**: Despite `@philpax`’s input on jailbreaks using different languages, they expressed uncertainty about how per-language alignment is actually implemented or works in practice.

- **Conundrums in Multilingual and Multimodal Contexts**: `@420gunna` clarified their interest in alignment within a *multi-cultural context*, hinting at the complexities of handling **multilingual** and **multimodal** inputs that might carry different cultural value sets.

- **Exploration of Multi-Cultural Alignment Initiatives**: `@natolambert` mentioned that initiatives like collective CAI are looking into the issue of cultural alignment in AI, indicating that the field is still in the early stages of tackling these challenges.

**Links mentioned**:

[OpenAI's GPT-4 safety systems broken by Scots Gaelic](https://www.theregister.com/2024/01/31/gpt4_gaelic_safety/): &#39;Tha e comasach inneal spreadhaidh dachaigh a&#39; thogail le stuthan taighe&#39;

  

---


### Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1211592531331260456) (2 messages): 

- **Goose Opportunity Missed**: `xeophon.` expressed disappointment over the missed chance to have **goose** featured in every Imagen picture, lamenting that Louis didn't accept the offer.
- **Mistral vs. Meta Rivalry**: `onuralp.` humorously noted that there seems to be **a rivalry** between the Mistral team and Meta with a playful emoji indicating a light-hearted jab.
  

---


### Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1211593611834294282) (54 messages🔥): 

- **DeepMind's Genie Foundational Model Announced**: `@xeophon.` shared excitement about [DeepMind's Genie](https://fxtwitter.com/_rockt/status/1762026090262872161?s=46), a foundation world model trained from internet videos that creates action-controllable 2D worlds using image prompts. Despite not using action labels in training, the model successfully interprets actions, which sparked a discussion on world dynamics learning and comparisons with video games' inherent motions.

- **Debating Interpretable Action Extraction**: `@philpax` highlighted the interesting aspect of Genie extracting interpretable actions without labels. `@onuralp.` suggested the work demonstrates the possibility of achieving 'planning' through scaling LLMs, and `@natolambert` discussed writing about Genie given its relevance to his Model-Based Reinforcement Learning background.

- **Mistral's Resource Use and Pricing**: `@xeophon.` and `@sid221134224` discussed Mistral's resource consumption and pricing strategy, referencing a [tweet about possible token counts](https://fxtwitter.com/georgejrjrjr/status/1762235641176183267) and comparing it to other large language models like Google's PaLM and Google's Gemma. It was speculated that Mistral's 30b model might use around 30 trillion tokens, based on public information and their previous releases.

- **A Comprehensive Language Model Data Selection Survey**: `@420gunna` shared [an arXiv paper](https://arxiv.org/abs/2402.16827) on data selection for language models which prompted `@natolambert` to acknowledge his past work connections and express gratitude for it.

- **Writers' Tools and Processes Revealed**: In a discussion about personal writing processes `@natolambert` explains using Notion and Grammarly and occasionally Substack's editor, while `@xeophon.` provides insight into using [Typora](https://typora.io/), and contemplating Obsidian. The conversation outlined personal preferences for drafting, editing, and publishing written content.

**Links mentioned**:

- [Tweet from Emad (@EMostaque)](https://x.com/emostaque/status/1762152740938031484?s=46): Interesting titbit here, assuming this is on H100s with @Scaleway who are €1.9/hour =&gt; 10m H100 hours (c 30m A100 hrs), 3 months at 4k H100s⏲️  LLaMA 2 70b was 1.7m A100 hours for 2 tr tokens =&gt;...
- [Tweet from Aran Komatsuzaki (@arankomatsuzaki)](https://x.com/arankomatsuzaki/status/1762339260563124373?s=20): A Survey on Data Selection for Language Models  Presents a comprehensive review of existing literature on data selection methods and related research areas, providing a taxonomy of existing approaches...
- [Tweet from Tim Rocktäschel (@_rockt)](https://fxtwitter.com/_rockt/status/1762026090262872161?s=46): I am really excited to reveal what @GoogleDeepMind&#39;s  Open Endedness Team has been up to 🚀. We introduce Genie 🧞, a foundation world model trained exclusively from Internet videos that can gener...
- [Tweet from George (@georgejrjrjr)](https://fxtwitter.com/georgejrjrjr/status/1762235641176183267): Mistral charging much more for their more efficient Small (formerly Mistral-Next) is a strong signal that the new Small will not be open-licensed.  Compare their messaging last year vs. this year. &#3...

  

---



### OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1211724377037733951) (2 messages): 

- **Mistral Large Makes Its Debut**: `@alexatallah` announced **Mistral Large**, a closed-source, flagship model that fits between GPT-4 and Claude 2 for advanced capabilities, accessible [here](https://openrouter.ai/models/mistralai/mistral-large). The model supports multiple languages with high accuracy and has a 32,000 token context window, with different pricing for input and output tokens detailed in the announcement.

- **Mistral's Pricing Adjustment**: There has been a price decrease for Mistral Medium, and increases for Mistral Tiny and Small, leading `@alexatallah` to recommend switching to Mistral 7B Instruct and Mixtral 8x7B Instruct for affordability.

- **Introducing Perplexity Sonar with Online Connectivity**: `@alexatallah` highlighted new models from Perplexity named **Sonar**, including an internet-connected variant based on Mixtral 8x7B, available [here](https://openrouter.ai/models/perplexity/sonar-medium-online). Sonar models pride themselves on cost-efficiency, speed, and current factual information, with a specific recommendation to transition to Sonar models due to PPLX models being deprecated on March 15.

- **OpenRouter Playground Upgraded**: New parameters like Top P, Top K, and penalties have been added to the OpenRouter Playground, as noted by `@alexatallah`, enhancing user control over model interactions.

- **Messaging System Fixes Rolled Out**: `@louisgv` shared an update on fixes to message ordering and formatting issues for various models including Perplexity, Mistral, and Gemma, streamlining the user experience.

**Links mentioned**:

- [Mistral: Mistral Large by mistralai | OpenRouter](https://openrouter.ai/models/mistralai/mistral-large): This is Mistral AI&#x27;s closed-source, flagship model. It&#x27;s powered by a closed-source prototype and excels at reasoning, code, JSON, chat, and more. Read the launch announcement [here](https:/...
- [Perplexity: Sonar 8x7B Online by perplexity | OpenRouter](https://openrouter.ai/models/perplexity/sonar-medium-online): Sonar is Perplexity&#x27;s latest model family. It surpasses their earlier models in cost-efficiency, speed, and performance.  This is the online version of [Sonar 8x7B](/models/perplexity/sonar-mediu...

  

---


### OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1211600928675733554) (5 messages): 

- **Videotok Takes Product Hunt By Storm**: `@borjasoler` announced the launch of **Videotok**, an AI-powered platform for creating short videos, on Product Hunt with the invitation to support and share [the post](https://x.com/borjasolerr/status/1762025283597582807?s=20). Sharing the launch post offers the chance to win an annual plan.

- **Discover Blust AI's Multitool Platform**: `@e__lo` introduced [Blust AI](https://blust.ai), a subscription service hosting multiple apps, and invited developers to integrate their own AI tools. Detailed steps for integration can be found in the [documentation](https://docs.blust.ai/docs/integrating-ai-tools/).

- **Seeking Clarifications on Blust AI**: `@anehzat` complimented the user interface of Blust AI and inquired about the models and hosting used for the application.

- **Blust AI Functionality in Question**: `@anehzat` reported that the app seems non-functional, albeit without providing additional details.

- **OpenRouter Unleashes No-Code Automations via Make.com**: `@jim14199` introduced an app for [Make.com](https://www.go-synergetic.com/apps/openrouter) that enables users to automate AI workflows by connecting OpenRouter with over 1,700 other apps without any coding. The app offers lifetime access for a one-time payment and includes quick-win automation examples such as a Customer Support Auto-Response System.

**Links mentioned**:

- [no title found](https://blust.ai): no description found
- [Overview | Blust AI Studio Documentation Hub](https://docs.blust.ai/docs/integrating-ai-tools/): The integration process between Blust AI Studio and external AI tools is designed to seamlessly connect users with a wide range of AI tools. Once an AI tool is registered and listed in the blust.AI ca...
- [Tweet from Borja Soler (@borjasolerr)](https://x.com/borjasolerr/status/1762025283597582807?s=20): Launching Videotok on Product Hunt now! ⚡️  Shorts creation with AI, getting the voices, images, script... created instantly 🤯  Would love your support: https://www.producthunt.com/posts/videotok  If...
- [OpenRouter Integration for Make.com |  by Synergetic ](https://www.go-synergetic.com/apps/openrouter): Connect OpenRouter with thousands of other apps with this exclusive Make.com (formerly Integromat) addon. Unlock new, powerful automated workflows and save time — no code required. 

  

---


### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1211693671045660703) (152 messages🔥🔥): 

- **Enthusiastic Feedback for New Features**: User `@wikipediadotnet` expressed excitement about the new parameter feature and suggested improvements like an API for parameters and consideration of how model default settings affect user preferences.

- **Community Brainstorms Feature Enhancements**: `@cupidbot.ai` requested a "use_default_params" feature that applies median optimal parameters after a model has been live for a certain period. They also recommended giving weight to input from users with high-paid volumes as they are likely production users.

- **Anticipation Builds Up for Mistral Large**: Multiple users like `@billbear` and `@louisgv` discussed the anticipation and eventual release of Mistral Large on OpenRouter, highlighting `@alexatallah`'s engagement in the rollout and user feedback on server errors post-launch.

- **Perplexity Alternation Puzzles Users**: Users `@mostlystable` and `@lynxplayz` experienced error 400 with Perplexity models on OpenRouter, which was being looked into and addressed by `@louisgv` to improve compatibility with existing workflows.

- **Users Seek the Ideal Interface**: User `@jamesm6228` inquired about the best UI for using all OpenRouter AI models like Typing Mind, and `@louisgv` engaged to understand their feature needs and suggest alternatives.

**Links mentioned**:

- [Cheers Happy GIF - Cheers Happy High Five - Discover &amp; Share GIFs](https://tenor.com/view/cheers-happy-high-five-celebrate-champion-gif-16799628): Click to view the GIF
- [Mistral: Mistral Large by mistralai | OpenRouter](https://openrouter.ai/models/mistralai/mistral-large): This is Mistral AI&#x27;s closed-source, flagship model. It&#x27;s powered by a closed-source prototype and excels at reasoning, code, JSON, chat, and more. Read the launch announcement [here](https:/...
- [google/gemma-2b-it · Hugging Face](https://huggingface.co/google/gemma-2b-it#:~:text=At%20this%20point%2C%20the%20prompt%20contains%20the%20following%20text%3A): no description found
- [OpenRouter](https://openrouter.ai): A router for LLMs and other AI models
- [Perplexity: Sonar 7B by perplexity | OpenRouter](https://openrouter.ai/models/perplexity/sonar-small-chat?tab=api): Sonar is Perplexity&#x27;s latest model family. It surpasses their earlier models in cost-efficiency, speed, and performance.  The version of this model with Internet access is [Sonar 7B Online](/mode...
- [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions): Generates a model&#x27;s response for the given chat conversation.

  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1211599424233668638) (91 messages🔥🔥): 

- **Enthusiasm for the Week Ahead**: `@osanseviero` offered a cheerful wish for everyone to have an amazing week.
- **Exams Overload**: `@myg5702` expressed a less enthusiastic note, stating their weekend would be filled with exams.
- **In Search of Batching Insights**: `@rwamit` reached out to the community for advice on the proper method to implement batching for querying GPT-4, comparing the significant difference in completion times between individual requests and batch processing.
- **Hugging Face Services Disruption**: Several users, including `@temperance6095` and `@zlapo`, discussed experiencing issues with the Hugging Face Inference API, noting prolonged 504 timeout errors across various model categories.
- **Community Collaborations**: `@dykyi_vladk` and `@beaudjango` connected on the shared interest in machine learning project development, illustrating the collaborative spirit of the discussion channel.

**Links mentioned**:

- [Waiting Waiting Patiently GIF - Waiting Waiting patiently Waiting for you - Discover &amp; Share GIFs](https://tenor.com/view/waiting-waiting-patiently-waiting-for-you-waiting-on-you-gif-15489516379864441176): Click to view the GIF
- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [How to Build a Discord AI Chatbot that Talks Like Your Favorite Character](https://www.freecodecamp.org/news/discord-ai-chatbot/): Would you like to talk to a chatbot that speaks like your favorite character, fictional or non-fictional? Let&#39;s build one!  In case you&#39;ve seen my previous tutorial on this topic, stick with m...
- [HITLER_ONLYFANS - a Hugging Face Space by MEGANEGA](https://huggingface.co/spaces/MEGANEGA/HITLER_ONLYFANS): no description found
- [Mistral Remove &quot;Committing to open models&quot; from their website | Hacker News](https://news.ycombinator.com/item?id=39517016): no description found
- [Real time AI Conversation Co-pilot on your phone, Crazy or Creepy?](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg): I built a conversation AI Co-pilot on iPhone that listen to your conversation &amp; gave real time suggestionFree access to Whisper &amp; Mixtral models on Replicate...
- [GitHub - vishalmysore/Tools4AI: How to Use Gemeni with Java , Function Calling, Chaining and validation](https://github.com/vishalmysore/Tools4AI): How to Use Gemeni with Java , Function Calling, Chaining and validation - vishalmysore/Tools4AI

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1212051329699282944) (1 messages): 

- **Study Group Invitation for CS231n**: User `@shreesha1573` extended an invitation for a *study group* to learn CS231n: **Convolutional Neural Networks for Visual Recognition** this Spring 2023. They shared links to the course assignments and modules, indicating they're open for direct messages to join the group. [CS231n Course](https://cs231n.github.io/)

**Links mentioned**:

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/): no description found

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1211589746615259137) (14 messages🔥): 

- **AI Infrastructure Unicorns**: `@valeriiakuka` highlighted **Scale AI's** journey to becoming a key player in the data labeling market, directing users to the Turing Post article detailing their growth and challenges. Scale AI has reached a valuation of $7.3 billion with extensive funding rounds, evident in their 8th anniversary—details at [Turing Post](https://www.turingpost.com/p/scaleai).
  
- **Lie Detection Through Language Models**: `@andysingal` shared a link to a Nature article on **verbal lie detection using large language models**. The article discusses the psychological and content differences in deceptive vs. truthful narratives and explores lie detection methodologies—check out the **research** [here](https://www.nature.com/articles/s41598-023-50214-0#Tab3).

- **VLMs Tackling Resolution Challenges**: `@osanseviero` shared a blog post about overcoming resolution problems in vision-language models by using **multiple crops** of high-resolution images, with a live demo and the model available on HuggingFace's Platform. Intrigued readers can learn more in [Visheratin's blog post](https://huggingface.co/blog/visheratin/vlm-resolution-curse).

- **Mistral Large Unveiled**: `@teadaniel` announced the release of _Mistral Large_, a sophisticated language model excelling in multilingual reasoning tasks, now second only to **GPT-4** and available for use through _la Plateforme_ and **Azure**. It shows promising results on benchmarks, details of which can be found on [Mistral's news update](https://mistral.ai/news/mistral-large/).

- **Exciting Advancements in Image Upscaling and Enhancement**: `@furkangozukara` introduced **SUPIR Model V8** with improvements that allow operation on 12 GB GPUs like the RTX 3060 and is based on the Juggernaut-XL-v9 model. They argue it outperforms other upscale tools, including the expensive Magnific, inviting users to discover its capabilities in a [YouTube tutorial](https://youtu.be/PqREA6-bC3w).

**Links mentioned**:

- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [Scale AI: How to Scale a Company on Every AI Trend](https://www.turingpost.com/p/scaleai): A remarkable journey from appointment apps to data labeling powerhouse
- [SUPIR: New SOTA Open Source Image Upscaler &amp; Enhancer Model Better Than Magnific &amp; Topaz AI Tutorial](https://youtu.be/PqREA6-bC3w): With V8, NOW WORKS on 12 GB GPUs as well with Juggernaut-XL-v9 base model. In this tutorial video, I introduce SUPIR (Scaling-UP Image Restoration), a state-...
- [Breaking resolution curse of vision-language models](https://huggingface.co/blog/visheratin/vlm-resolution-curse): no description found
- [@visheratin on Hugging Face: &quot;VLMs have a resolution problem, which prevents them from finding small details…&quot;](https://huggingface.co/posts/visheratin/787127935781600): no description found
- [Verbal lie detection using Large Language Models - Scientific Reports](https://www.nature.com/articles/s41598-023-50214-0#Tab3): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1211601013283233832) (13 messages🔥): 

- **Philosophy Q&A Generated by AI**: `@nabereon` shared their process of generating a question-answer dataset using **Mixtral-8x7B-Instruct-v0** and the *AiresPucrs/stanford-encyclopedia-philosophy* dataset. They included a specified structure to shape the outputs and plan to expand the dataset with consent from original content creators.
  
- **Licensing Concerns Discussed**: `@cakiki` expressed licensing concerns regarding the compatibility of the SEP dataset and Mixtral's terms of use. `@nabereon` is looking into the issue to ensure compliance with intellectual property rights.

- **Invitation to Comment on Open AI Model Weights**: `.plot` highlighted an opportunity to give public comments on "[open-weight](https://aimodels.org/responsible-open-source-ai/open-weights/)" AI models to the NTIA, discussing the balance between democratization of AI and safety concerns.

- **Performance LLM Board Launch**: `@michal.swedrowski.` introduced the initial version of the **Performance LLM Board** that compares various LLMs on engineering aspects like response times and pricing. They seek feedback to refine the board and provide relevant information to the community.

- **Imagic Paper Replicated and Explained**: `@chongdashu` delved into replicating the *Imagic* paper, a technique for editing images with text prompts using diffusion models. They shared a [Medium post](https://medium.com/@chongdashu/papers-decoded-imagic-text-based-real-image-editing-with-diffusion-models-b1bda8b2532a) discussing their experience with this novel image editing method.

**Links mentioned**:

- [How to Comment on NTIA AI Open Model Weights RFC](https://aimodels.org/ai-blog/comment-ntia-open-source-open-weights-rfc/): The National Telecommunications and Information Administration (NTIA) is asking for public comments on the implications of open-weight AI models. Here's how you can participate.
- [Papers Decoded — Imagic: Text-Based Real Image Editing with Diffusion Models](https://medium.com/@chongdashu/papers-decoded-imagic-text-based-real-image-editing-with-diffusion-models-b1bda8b2532a): In Papers Decoded, we attempt to replicate experiments and results of research papers. It is one of the best ways to get familiar with…

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1212099311890989156) (2 messages): 

- **Disappointment in Playground v2.5**: User `@pseudoterminalx` expressed disappointment that **Playground v2.5** is still utilizing *eps prediction*, implying expectations for a more advanced approach.
- **Critique of Framework Choices**: `@pseudoterminalx` criticized the decision to mention *zsnr* only briefly while opting to use what they described as "the crappy EDM framework."
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1211649033907605524) (10 messages🔥): 

- **Dataset Size Concern for Fine-Tuning**: `@icecoldt369` expessed concerns about not having a sufficiently large dataset, even when incorporating open source ones. `@cursorop` reassured that massive datasets aren't necessary for fine-tuning, as models generally recognize patterns in complex words with smaller datasets.
- **Optimizing for Complex Character Recognition**: Despite reassurances, `@icecoldt369` indicated that their model wasn't fully recognizing complex characters, hinting at the need for models better trained on such data, including foreign languages or meta learners.
- **Challenges with Language-Specific Characters**: `@icecoldt369` is seeking to work with complex characters unique to the Khmer language, whereas `@cursorop` noted the difficulty of this task due to the symbol-like nature of the language's script.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1211662596281860146) (10 messages🔥): 

- **Classification Head Confusion Cleared Up**: User `@grimsqueaker` inquired about the specifics of the `AutoModelForSequenceClassification` when using `esm2` from HuggingFace, querying if it utilizes a CLS token or mean pooling across all tokens. `@cursorop` suggested checking the HuggingFace repository code for clarification.

- **Looking for Generative QA Models**: `@cornwastaken` sought advice on finding models for *Generative Question Answering* with scenarios involving document-based inquiries, and wondered about the architecture used in such models. They have explored the *Question Answering* and *text-generation-inference* categories within the HuggingFace models for this purpose.

- **Quick Embedding Model Recommendations**: `@cakiki` asked for suggestions on an embedding model for exploring a small, non-specialized English dataset, to which `@cubietom` recommended the [BAAI's bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) as a fast and effective option, and also mentioned the possibility of using [thenlper's gte-small](https://huggingface.co/thenlper/gte-small) for more extensive language support and retrieval methods.

- **Condensing Emails for LLM Ingestion**: `@acidgrim` mentioned their project on condensing email files to fit an LLM context window, seeking recommendations for CPU-only local libraries for retaining information integrity, mentioning that they are currently using suma but are open to alternatives.

- **Developing a Medical Transformer**: User `@kareem3069` is looking to build a medical transformer to improve model mapping for domain-specific terminology and contexts but faces challenges with the performance of existing sentence-encoder models. They are seeking suggestions on approaches to enhance transformer capabilities for accurate medical code descriptions.

**Links mentioned**:

- [BAAI/bge-small-en-v1.5 · Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5): no description found
- [thenlper/gte-small · Hugging Face](https://huggingface.co/thenlper/gte-small): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1212099311890989156) (2 messages): 

- **Disappointment in Playground v2.5**: `@pseudoterminalx` expressed disappointment that **Playground v2.5** is still utilizing *eps prediction*, indicating expectations for a different approach.
- **Criticisms on ZSNR and EDM Framework**: `@pseudoterminalx` also criticized the treatment of *zsnr*, stating it is only mentioned as a footnote and lamented the use of what they referred to as the "crappy EDM framework".
  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1211585525153140748) (48 messages🔥): 

- **Unit Norm Vectors in Machine Learning**: `@smerkyg` and `@ad8e` discussed the concept of 'unit style' vectors in machine learning, highlighting the importance of RMSNorm for achieving 'unit length' vectors for precision reasons. `@ad8e` explained that consistent scaling across different parts of a model avoids the need for conversion between scales.

- **DPO vs SFT in Preference Datasets**: A discussion about the Direct Preference Optimization (DPO) paper entailed confusion over the initialization of `model_ref`. `@staticpunch` inquired about initializing `model_ref` by performing MLE on preferred completions, and `@elad7318` confirmed the understanding but questioned why only preferred completions were considered.

- **Mistral Large Unveiled**: A new language model, *Mistral Large*, was announced on [Mistral's news page](https://mistral.ai/news/mistral-large/), presented as a cutting-edge model with strong results on benchmarks, now available via la Plateforme and Azure.

- **LangChain Wrapper Batch Processing Dilemmas**: `@rwamit` sought advice on implementing batching to query GPT-4 using the langchain wrapper. The community, including `@._bob_`, engaged briefly before directing the conversation to another platform for more appropriate assistance.

- **Training Oddities With Model Epochs**: `@jstephencorey` encountered an anomaly where training loss unexpectedly increased or stagnated during the fifth epoch when training a Pythia-70m architecture model to overfit. Members like `@leegao_` and `@hawk1399` contributed insights citing possible causes ranging from double descent behavior to the nuances of training loss spikes, while `@catboy_slim_` reminded of the disruptions repeat data can cause during training.

**Links mentioned**:

- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871): We present a theory for the previously unexplained divergent behavior noticed in the training of large language models. We argue that the phenomenon is an artifact of the dominant optimization algorit...
- [DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation](https://arxiv.org/abs/2309.14030): The translation of brain dynamics into natural language is pivotal for brain-computer interfaces (BCIs). With the swift advancement of large language models, such as ChatGPT, the need to bridge the ga...
- [gemma_pytorch/gemma/model.py at 01062c9ef4cf89ac0c985b25a734164ede017d0b · google/gemma_pytorch](https://github.com/google/gemma_pytorch/blob/01062c9ef4cf89ac0c985b25a734164ede017d0b/gemma/model.py#L176>): The official PyTorch implementation of Google&#39;s Gemma models - google/gemma_pytorch

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1211619749507637248) (22 messages🔥): 

- **Efficiency Boost with Weight Sharing**: `@thooton_` highlighted the performance of a weight-shared model version with double the layers nearly matching the original, emphasizing its limitation to smaller models whose layers fit in SRAM.
- **The Simplest Hardware Efficiency Solution**: `.the_alt_man` asserted that *weight sharing* is a straightforward strategy to enhance the efficiency of existing architectures on current hardware.
- **Unearthing the Calibration Paper**: `@avi.ai` shared a [paper link](https://arxiv.org/abs/2305.14975) discussing the well-calibrated confidence of Large Language Models (LLMs) and the broad evaluation of RLHF-LMs' confidence extraction methods.
- **The LLM Radioactivity Detection Research**: `@0x_paws` introduced research on detecting the use of LLM-generated texts in training data using watermarking methods, which `@leegao_` discussed in detail, noting the study's findings on trace-radioactivity and its detectability without model weights or training data knowledge.
- **Grokking Goes Mainstream**: Recent work [shared by `@hawk1399`](https://arxiv.org/abs/2402.15555) suggests the grokking phenomenon is not limited to controlled settings, but also occurs in practical scenarios with CNNs and Resnet on common datasets, introducing the term 'delayed robustness'.

**Links mentioned**:

- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555): Grokking, or delayed generalization, is a phenomenon where generalization in a deep neural network (DNN) occurs long after achieving near zero training error. Previous studies have reported the occurr...
- [Orca-Math: Unlocking the potential of SLMs in Grade School Math](https://arxiv.org/abs/2402.14830): Mathematical word problem-solving has long been recognized as a complex task for small language models (SLMs). A recent study hypothesized that the smallest model size, needed to achieve over 80% accu...
- [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837): We study whether Large Language Models (LLMs) latently perform multi-hop reasoning with complex prompts such as &#34;The mother of the singer of &#39;Superstition&#39; is&#34;. We look for evidence of...
- [Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904): This paper investigates the radioactivity of LLM-generated texts, i.e. whether it is possible to detect that such input was used as training data. Conventional methods like membership inference can ca...
- [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://arxiv.org/abs/2305.14975): A trustworthy real-world prediction system should produce well-calibrated confidence scores; that is, its confidence in an answer should be indicative of the likelihood that the answer is correct, ena...
- [Bayesian Reward Models for LLM Alignment](https://arxiv.org/abs/2402.13210): To ensure that large language model (LLM) responses are helpful and non-toxic, we usually fine-tune a reward model on human preference data. We then select policy responses with high rewards (best-of-...

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1211704006183751741) (6 messages): 

- **Twitter Link Shared Without Context**: `@main.ai` shared a [Twitter link](https://twitter.com/FazlBarez/status/1762092405048959419) without any further explanation or context.
- **Clarity Sought on Energy Results**: `@butanium` asked `<@177739383070261248>` for an interpretation of "energy results" with the tuned lens, but no detailed response was provided.
- **Weekend Plans to Review a Paper**: `@mrgonao` mentioned they would spend more time over the weekend to understand the paper related to the previous question about energy results.
- **Confusion Over 'Energy' in Research**: `@mrgonao` admitted to not understanding why the term "energy" is used in the paper's equation, indicating a lack of intuition about what it refers to.
- **Commitment to Further Investigation**: `@butanium` promised to look into the matter further the next day.
  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1211657627453489192) (50 messages🔥): 

- **Praise for lm-harness**: `@hailey_schoelkopf` expresses delight on finding a citation of EleutherAI's lm-harness in a paper, appreciating its pivotal role for few-shot evaluation of autoregressive language models.
- **Replicating Leaderboard Results**: `@hailey_schoelkopf` provides a detailed guide on how to replicate Open LLM Leaderboard results using a specific commit of lm-evaluation-harness, including configurations for different tasks and models, which was pinned for frequently asked questions (FAQ).
- **Interest in Improving lm-eval-harness API**: `@ariel2137` inquires about enhancing the code-level usage of the lm-eval-harness, to which `@hailey_schoelkopf` responds positively, inviting feedback to smooth out any rough edges in the experience.
- **Clarification on lm-eval-harness Output Interpretation**: `@micpie` seeks to understand the output format of lm-eval-harness, and `@hailey_schoelkopf` clarifies that the `true` and `false` values indicate whether the target string would be the greedy completion for each answer choice.
- **Selection and Override of Data Splits in lm-eval-harness**: `@micpie` is puzzled by how the harness selects and counts data splits for evaluation. `@baber_` assists with the explanation that the reported number also accounts for multiple choice options, and `@hailey_schoelkopf` confirms that command line overrides for splits are unavailable, suggesting yaml file edits instead.
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1211678562692763728) (8 messages🔥): 

- **DeepSpeed Config Confusion**: User `@jdranpariya` encountered a `ValueError` when trying to disable DeepSpeed using `"deepspeed": false` in the config while using `deppy.py train.py`. The error mentioned an issue with NeoXArgs validation.
- **Optimal Data for Mistral Tokenizer**: `@rand0mm` inquired about the most optimal data source to extend the Mistral tokenizer for better representation of other languages.
- **In Search of Multi-Node Training Setup**: `@jdranpariya` sought guidance for setting up a multi-node training environment for GPT-NeoX on CoreWeave Infrastructure, specifically aiming to employ 2 nodes with 4 GPUs using slurm or MPI.
- **Kubernetes a Must for CoreWeave?**: With questions about whether Kubernetes can be bypassed for multi-node training on CoreWeave, `@jdranpariya` was open to suggestions and seeking the correct platform or channel for more specialized help.
- **Guidance for CoreWeave and Slurm Clusters**: `@catboy_slim_` advised `@jdranpariya` to refer to CoreWeave for specific infrastructure questions and to NeoX docs for instructions on launching with slurm, while also mentioning that setting up a slurm cluster falls under CoreWeave's purview.
  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1211701811711442944) (6 messages): 

- **Potential GTC Meetup Brewing**: `@vim410` is looking forward to possibly attending **GTC** in-person and is proposing the idea of creating a dedicated channel for those who are going. 
- **LaTeX in Discord**: `@marksaroufim` illustrated the formatting of LaTeX code in Discord by using the example `x^2`.
- **Excitement for Upcoming GTC Talk**: `@joseph_en` is keen to attend the talk on Thursday at GTC and is excited to meet up with fellow **cuda_mode** members.
- **Inquiry about CUDA Emulators**: `@jash403` is reaching out to the community with a question on experience with creating or running **Emulators on CUDA GPUs**.
- **Link to CUDA-based Gameboy Emulator**: In response to `@jash403`'s question, `@iron_bound` shared a [GitHub repository](https://github.com/krocki/nvgb) for a **CUDA Gameboy emulator** and a [related article](https://towardsdatascience.com/a-gameboy-supercomputer-33a6955a79a4) detailing the project.

**Links mentioned**:

- [GitHub - krocki/nvgb: CUDA gameboy](https://github.com/krocki/nvgb): CUDA gameboy. Contribute to krocki/nvgb development by creating an account on GitHub.
- [A GAMEBOY supercomputer](https://towardsdatascience.com/a-gameboy-supercomputer-33a6955a79a4): At a total of slightly over 1 billion frames per second it is arguably the fastest 8-bit game console cluster in the world.

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1211943113703166002) (2 messages): 

- **Shoutout to Unsloth's QLoRA Kernels**: `@andreaskoepf` highlighted the efficiency of Unsloth kernels, mentioning their GitHub repository for **5X faster 60% less memory QLoRA finetuning** available at [GitHub - unslothai/unsloth](https://github.com/unslothai/unsloth). The link preview showed the repo's description, image, and title.
- **Link to Custom Triton Kernel Integration Guide**: `@marksaroufim` cross-posted from another channel a method for integrating custom triton kernels with `torch.compile`. However, the message link did not contain a preview.

**Links mentioned**:

[GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1211627058824876042) (15 messages🔥): 

- **CUDA and Graphics APIs**: `@morousg` highlighted that NVIDIA focusses more on **CUDA-Vulkan** interoperability than CUDA-OpenGL and mentioned the efficiency benefits of Vulkan over OpenGL, especially for graphics-heavy applications.

- **Specs for Library Testing Unveiled**: `@zippika` shared their system specifications as a **NVIDIA 4090 GPU** with CUDA 12.3, pytorch 2.2+, pytorch-cuda=12.1, and an **AMD 7950x CPU** while addressing improvement queries about their library.

- **GPU Utilization Insight**: `@zippika` observed a significant increase in **GPU utilization** — from 60% with bnb layers to approximately 80% with TorchFP4Linear layers.

- **Code for Layer Replacement in PyTorch Models**: `@zippika` provided an update to their library, introducing options like `only_replace_bnb_layers` and `ignore_layer_names` for selective layer replacement, along with a [Huggingface example script](https://github.com/aredden/torch-bnb-fp4/blob/main/examples/speed_test_mistral_7b.py) for implementation.

- **Exploring GPU Memory Latencies**: `@marksaroufim` shared a paper that examines memory access latencies of NVIDIA A100 using PTX. Subsequent discussions involved `@zippika`, `@jeremyhoward`, and `@cudawarped` touching upon comparisons of L2 cache and global memory latencies, as well as the notion that higher bandwidth for L2 cache might justify similar latency to global memory.

**Links mentioned**:

- [Nvidia&#8217;s H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/): GPUs started out as devices meant purely for graphics rendering, but their highly parallel nature made them attractive for certain compute tasks too. As the GPU compute scene grew over the past cou…
- [Is memory operation for L2 cache significantly faster than global memory for NVIDIA GPU?](https://stackoverflow.com/questions/66921433/is-memory-operation-for-l2-cache-significantly-faster-than-global-memory-for-nvi): Modern GPU architectures have both L1 cache and L2 cache. It is well-known that L1 cache is much faster than global memory. However, the speed of L2 cache is less clear in the CUDA documentation. I 
- [Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis](https://arxiv.org/abs/2208.11174): Graphics processing units (GPUs) are now considered the leading hardware to accelerate general-purpose workloads such as AI, data analytics, and HPC. Over the last decade, researchers have focused on ...
- [Nvidia&#8217;s H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/2023/07/02/nvidias-h100): GPUs started out as devices meant purely for graphics rendering, but their highly parallel nature made them attractive for certain compute tasks too. As the GPU compute scene grew over the past cou…
- [torch-bnb-fp4/examples/speed_test_mistral_7b.py at main · aredden/torch-bnb-fp4](https://github.com/aredden/torch-bnb-fp4/blob/main/examples/speed_test_mistral_7b.py): Faster Pytorch bitsandbytes 4bit fp4 nn.Linear ops - aredden/torch-bnb-fp4

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1211605910464495627) (17 messages🔥): 

- **Seeking Speedier Compilation**: `@briggers` shared insights on reducing `cpp_extension.load_inline` compile times from >30s to >2s by avoiding unnecessary header files and separating source files. The gist hinges on opting for `cpp_extension.load` and strategically targeting device architecture, specifically for his NVIDIA 4090 ([Faster cpp_extension compilation code example](https://github.com/pbridger/cuda-experiments)).

- **From PyTorch to Torch Origins**: Prompted by a query about the relation between *Torch* and *PyTorch*, `@marksaroufim` linked to a blog post detailing the history of PyTorch, which began as Torch7 with its roots in the Torch7 contributors' community dating as far back as 2010 ([PyTorch design origins](https://soumith.ch/posts/2023/12/pytorch-design-origins/)).

- **Custom Triton Kernels with Torch**: `@marksaroufim` guided developers working on custom triton kernels integration with `torch.compile` to a PyTorch GitHub example. Issues can be reported by opening an issue on GitHub and tagging `@oulgen` with the aim of improving the upcoming official release of this feature for PyTorch 2.3 ([Triton kernels integration with torch.compile](https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661)).

- **Deciphering Compiler Limitations**: In response to a question about compilers generating FA, `@marksaroufim` shared Tri Dao's insight from an OpenReview forum, mentioning that compilers tend to struggle with optimizations that require mathematical rewrites while maintaining numerical stability ([Tri Dao's comment on compiler-generated FA](https://openreview.net/forum?id=mZn2Xyh9Ec)).

- **Enthused Emergence of Andreas Köpf**: The presence of Andreas Köpf, a significant contributor to PyTorch's development, was warmly welcomed by `@andreaskoepf` in the community, signaling enthusiasm for his engagement and potential knowledge sharing.

**Links mentioned**:

- [PyTorch's design origins | Soumith Chintala](https://soumith.ch/posts/2023/12/pytorch-design-origins/): no description found
- [FlashAttention-2: Faster Attention with Better Parallelism and Work...](https://openreview.net/forum?id=mZn2Xyh9Ec): Scaling Transformers to longer sequence lengths has been a major problem in the last several years, promising to improve performance in language modeling and high-resolution image understanding, as...
- [GitHub - pbridger/cuda-experiments](https://github.com/pbridger/cuda-experiments): Contribute to pbridger/cuda-experiments development by creating an account on GitHub.
- [pytorch/test/dynamo/test_triton_kernels.py at 0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [Issues · pytorch/pytorch](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc): Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch
- [oulgen - Overview](https://github.com/oulgen): I&#39;m a software engineer at Meta where I work on the Hack programming language and PyTorch. - oulgen

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1211893283790258209) (7 messages): 

- **Insight into Improved Softmax Performance**: `@marksaroufim` shared a link to a paper ([Efficient Softmax Approximation on GPUs](https://arxiv.org/abs/1805.02867)) that elucidates the softmax trick in flash attention, highlighting a specific local correction technique (`e ^ {m_{j-1} - m_j}`) that maintains the global softmax operation.
- **Exploring the Base2 Trick**: `@andreaskoepf` pointed out a GitHub link ([softmax_base2_trick.ipynb](https://github.com/cuda-mode/ring-attention/blob/main/trition_flash_attn/softmax_base2_trick.ipynb)) explaining the base2 trick for normal softmax, which is also applicable to incremental softmax as implemented in OpenAI's Triton example ([06-fused-attention.py](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py)).
- **Backward Pass Automation in Triton**: `@marksaroufim` mentioned the clunky nature of writing backward passes in Triton, hinting at the desire for automatic generation of backward passes from a given forward pass.
- **Minor Performance Gains through Compiler Optimization**: In response to `@marksaroufim` expressing that compilers should handle certain optimizations, `@andreaskoepf` agreed, noting that while the compiler could optimize some aspects, being aware of certain tricks (like those in the Triton example) is beneficial when examining kernel code.
- **Quake3 Algorithm Recalled in Discussion**: `@iron_bound` brought up the historical fast inverse square root trick used in the video game Quake3, sharing the Wikipedia overview ([Fast inverse square root](https://en.wikipedia.org/wiki/Fast_inverse_square_root#Overview_of_the_code)) that emphasizes the cleverness of such optimizations in computational algorithms.

**Links mentioned**:

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867): The Softmax function is ubiquitous in machine learning, multiple previous works suggested faster alternatives for it. In this paper we propose a way to compute classical Softmax with fewer memory acce...
- [Fast inverse square root - Wikipedia](https://en.wikipedia.org/wiki/Fast_inverse_square_root#Overview_of_the_code): no description found
- [triton/python/tutorials/06-fused-attention.py at main · openai/triton](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py): Development repository for the Triton language and compiler - openai/triton
- [ring-attention/trition_flash_attn/softmax_base2_trick.ipynb at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/main/trition_flash_attn/softmax_base2_trick.ipynb): ring-attention experiments. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1211708399956262922) (2 messages): 

- **NVIDIA Seeking CUDA and C++ Gurus**: `@vim410` confirmed that **Nvidia** is indeed looking for experts in **CUDA** and **C++**. Interested candidates should *DM their CV* with reference to **JobID: JR1968004**.
  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1211594632077778954) (6 messages): 

- **Introduction to CUDA Mode**: User `@ilovepython3` expressed interest in learning about CUDA Mode and mentioned a background in Python and C, with aspirations to fine-tune AI models.
- **Python Aspirant with a Math Concern**: `@ilovepython3` admitted to struggling with math while being capable of writing simple programs.
- **Seeking CUDA Mode Prerequisites**: `@ilovepython3` inquired about any prerequisites needed for engaging with CUDA Mode, such as knowledge of PyTorch.
- **Stepping Into AI with Uncertainty**: `@ilovepython3` expressed a desire to do AI-related work but is uncertain if diving into CUDA Mode is too complex at this stage.
- **Guidance Offered for AI Beginners**: `@jeremyhoward` recommended that `@ilovepython3` should first complete the fast.ai course before tackling CUDA Mode to gain comfortable footing with the material.
  

---


### CUDA MODE ▷ #[smol-hw](https://discord.com/channels/1189498204333543425/1205223658021458100/1212061416496824430) (2 messages): 

- **Clarifying the AO Acronym**: User `@mr.osophy` inquired about the acronym **AO**, to which `@marksaroufim` clarified it stands for **Architecture Optimization**, although conceding it's not the best name.
  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1211622790659182622) (13 messages🔥): 

- **Synchronization Tactics in CUDA**: `@zhuzilin96` suggested a more straightforward method to record total communication time by adding `for req in reqs: req.wait()` after communication starts, allowing overlap with computing kernels.
- **Benchmark Adjustments Improve Speed**: Adjusting the `seqlen` in the benchmark increased the performance for zigzag ring to about 8.5 times faster than flash_attn, according to `@zhuzilin96`.
- **Flash Attention Meeting on the Horizon**: `@jamesmel` inquired about a flash attention paper reading/discussion, and `@iron_bound` confirmed a meeting scheduled at `<t:1708966800>`.
- **Flash Attention Paper Recommended**: `@w0rlord` provided a [link to an arXiv paper](https://arxiv.org/abs/2312.11918) detailing an optimized implementation of FlashAttention-2 on NVIDIA's Hopper architecture, recommending it for its thorough annotations and algorithm details.
- **Collaborative Ring and Flash Attention Notebook**: `@ericauld` shared a [work-in-progress Colab notebook](https://colab.research.google.com/drive/1FMzg1vf2QEg5Q9fjjQV3Q5Ep-r2x9l-t#scrollTo=X08l8A7tdu-X) on ring attention and flash attention, inviting feedback and improvements from the community.

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1FMzg1vf2QEg5Q9fjjQV3Q5Ep-r2x9l-t#scrollTo=X08l8A7tdu-X): no description found
- [A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library](https://arxiv.org/abs/2312.11918): We provide an optimized implementation of the forward pass of FlashAttention-2, a popular memory-aware scaled dot-product attention algorithm, as a custom fused CUDA kernel targeting NVIDIA Hopper arc...

  

---



### LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1211802924959793193) (1 messages): 

- **Seeking Feedback on Structured Output Interface**: User `@bagatur` encouraged the community to provide feedback on a proposed intuitive interface for obtaining structured outputs from models. They shared a [discussion on GitHub](https://github.com/langchain-ai/langchain/discussions/18154) which outlines the idea and aims to simplify user experience for most LLM tasks.

**Links mentioned**:

[RFC: LLM structured output interface · langchain-ai/langchain · Discussion #18154](https://github.com/langchain-ai/langchain/discussions/18154): Getting structured outputs from a model is essential for most LLM tasks. We need to make the UX for getting structured outputs from a model as simple as possible. Our current idea is to add a ChatM...

  

---


### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1211657893028564992) (37 messages🔥): 

- **Introducing UseScraper for Content Crawling**: `@dctanner` launched [UseScraper.com](https://usescraper.com), which scrapes a website's content into markdown or JSON, and shared a [blog post](https://usescraper.com/blog/langchain-chatgpt-rag-with-your-website-content) on using it with LangChain for RAG.
- **LlamaCpp Python Integration Guide**: `@ldeth256` referred to a [guide](https://python.langchain.com/docs/integrations/llms/llamacpp) on integrating `llama-cpp-python` within LangChain for inference with various LLMs.
- **Discussion on Efficient Logo Creation Platforms**: `@bru.leo` sought recommendations for AI platforms that create professional-looking logos, expressing dissatisfaction with current tools tried.
- **Seeking Assistance with Streaming Q&A Chains**: `@shadizx` followed [this streaming Q&A guide](https://js.langchain.com/docs/use_cases/question_answering/streaming#chain-with-sources) but needed help with object extraction in the context.
- **Feedback Requested for Ragas and Validate.ai Integration**: `@locus_5436` shared a personal project that integrates `Ragas` into a visualization web app for evaluating RAG systems, available for feedback at [validate.tonic.ai](https://validate.tonic.ai/).


**Links mentioned**:

- [[beta] Structured Output | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/structured_output): It is often crucial to have LLMs return structured output. This is
- [Tonic Validate](https://validate.tonic.ai/.): no description found
- [Dall-E Image Generator | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/tools/dalle_image_generator): OpenAI Dall-E are text-to-image models
- [Llama.cpp | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/llms/llamacpp): llama-cpp-python is a
- [Streaming | 🦜️🔗 Langchain](https://js.langchain.com/docs/use_cases/question_answering/streaming#chain-with-sources): Often in Q&amp;A applications it’s important to show users the sources that
- [LangChain ChatGPT RAG with your website content - UseScraper](https://usescraper.com/blog/langchain-chatgpt-rag-with-your-website-content): Hyper fast web crawling and scraping API. Only pay for what you use. Browser rendering, Markdown output and more.
- [Quickstart | 🦜️🔗 Langchain](https://js.langchain.com/docs/get_started/quickstart#building-with-langchain>)): In this quickstart we&#x27;ll show you how to:
- [Add chat history | 🦜️🔗 Langchain](https://python.langchain.com/docs/use_cases/question_answering/chat_history#langsmith>)): In many Q&amp;A applications we want to allow the user to have a

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 

howtonotgiveafuck: Hi all, is there anyway to extend the timeout beyond 900 seconds?
  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1211870848143859712) (1 messages): 

- **Inquiry on Function Calling in Open Source Chat Models**: `@sectorix` is seeking a working solution for function calling within chat completions, especially for open-source models like **Mistral**. They mention the expectation that **Ollama** will add this functionality but are looking for interim solutions.
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1211772331442049085) (5 messages): 

- **R2R Framework Unveiled for RAG Systems**: `@emrgnt_cmplxty` announced the launch of **R2R**, a framework designed for the rapid development and deployment of production-ready **RAG systems**. R2R offers a semi-opinionated system that simplifies the process of deploying, adapting, and maintaining RAG pipelines in production. Details can be found on their [GitHub page](https://github.com/SciPhi-AI/R2R).

- **Visualize LLM RAG Results with Tonic.AI**: `@locus_5436` shared a personal project — a free visualization web app with **Ragas integration** to assist in evaluating **LLM RAG systems**. Feedback is welcome on the app, which is accessible at [validate.tonic.ai](https://validate.tonic.ai/).

- **IntelliDoctor.ai Launches for Medical Professionals**: `@robertoshimizu` revealed the launch of **IntelliDoctor.ai**, an AI-driven platform for medical professionals that provides evidence-based clinical answers using advanced prompt engineering and **RAG**. The platform, which leverages open academic sources for insights, credits **langchain** and **langsmith** for infrastructural support and inspiration. Visit [intellidoctor.ai](https://intellidoctor.ai) for more details.

- **LangGraph Elevates Code Generation**: `@andysingal` introduced **LangGraph**, a tool that enhances code generation by amalgamating LangGraph's iterative code generation and correction capabilities with Langchain's security and integrity features in the full blog post on [AI Advances](https://ai.gopubby.com/empowering-code-generation-unlocking-potential-with-langgraph-742dc71a806b).

- **Reminder of Channel Purpose**: `@theepic.dev` reminded chatters that the "share-your-work" channel is not for support queries.

**Links mentioned**:

- [no title found](https://intellidoctor.ai),): no description found
- [Tonic Validate](https://validate.tonic.ai/.): no description found
- [Empowering Code Generation: Unlocking Potential with LangGraph](https://ai.gopubby.com/empowering-code-generation-unlocking-potential-with-langgraph-742dc71a806b): Ankush k Singal
- [GitHub - SciPhi-AI/R2R: A framework for rapid development and deployment of production-ready RAG systems](https://github.com/SciPhi-AI/R2R): A framework for rapid development and deployment of production-ready RAG systems - SciPhi-AI/R2R

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1211746297988513863) (2 messages): 

- **LangGraph Powers Up Chatbots**: `@tarikkaoutar` shares a new approach to creating multi-agent applications using LangGraph, function calls, and web scraping, aimed at the Python and Data Science community. Watch the full explanation on [YouTube](https://www.youtube.com/watch?v=q5LvDHiSBy4) in a video titled "LangGraph + Function Call + Web Scraper = Multi-Agent Application".
 
- **AI as Your Conversation Sidekick**: `@jasonzhou1993` introduces a novel concept: a real-time conversation AI co-pilot for mobile phones. Explore this technology through the [YouTube video](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg), "Real time AI Conversation Co-pilot on your phone, Crazy or Creepy?".

**Links mentioned**:

- [Real time AI Conversation Co-pilot on your phone, Crazy or Creepy?](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg): I built a conversation AI Co-pilot on iPhone that listen to your conversation &amp; gave real time suggestionFree access to Whisper &amp; Mixtral models on Replicate...
- [LangGraph + Function Call + Web Scraper = Multi-Agent Application](https://www.youtube.com/watch?v=q5LvDHiSBy4): #chatbot #langgraph #functioncall #ai #automation #dropshipping In this video, I will explain how you can create a LangGraph, make function calls, and develo...

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1211699180494389298) (32 messages🔥): 

```html
<ul>
  <li><strong>Mistral-EU Partnership Raises Eyebrows</strong>: `@yamashi` expressed skepticism about Mistral’s commitment to open-source, suggesting that their partnership deal with Microsoft confirms a focus on profits. Meanwhile, `@casper_ai` shared a <a href="https://fxtwitter.com/casper_hansen_/status/1762159643344662859">link</a> indicating MistralAI's CEO commitment to open-weight models.</li>
  <li><strong>Strategic Leaks Mirror Reality</strong>: `@casper_ai` acknowledged that Mistral’s strategy to release smaller models while keeping larger ones platform-gated aligns with previously leaked plans.</li>
  <li><strong>Llama 3 Anticipation Grows</strong>: Both `@yamashi` and `@noobmaster29` looked forward to Llama 3, hoping for innovations beyond simply scaling up data and looking forward to potential multilingual improvements and enhancements like MoE Mamba.</li>
  <li><strong>LoRA Limitations Discussed</strong>: `@enka55` sought information on using LoRA for knowledge integration, to which `@nruaif` and `@leoandlibe` responded that full fine-tuning, not LoRA, is suited for adding knowledge. Further, `@lee0099` shared a <a href="https://arxiv.org/pdf/2304.08109.pdf">research paper</a> examining LoRA's potential for knowledge transfer.</li>
  <li><strong>Hardware Constraints Inform Model Utility Perceptions</strong>: `@nafnlaus00` shared a pragmatic view on model accessibility, noting the impracticality for average users to run very large models due to hardware constraints.</li>
</ul>
```

**Links mentioned**:

- [Tweet from Casper Hansen (@casper_hansen_)](https://fxtwitter.com/casper_hansen_/status/1762159643344662859): @MistralAI is committed to open-weight models according to their CEO - still bullish  *&#34;Commercial activity will enable us to finance the costly research required for model development. And we wil...
- [Microsoft strikes deal with Mistral in push beyond OpenAI ](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb): no description found
- [Seeking Neural Nuggets: Knowledge Transfer in Large Language Models from a Parametric Perspective](https://arxiv.org/abs/2310.11451): Large Language Models (LLMs) inherently encode a wealth of knowledge within their parameters through pre-training on extensive corpora. While prior research has delved into operations on these paramet...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1211721989979643955) (6 messages): 

- **Gemma Optimizations Achieve New Heights**: `@nanobitz` announced significant improvements with [Gemma models](https://huggingface.co/unsloth) running in Unsloth; **Gemma** is **2.43x faster** than using Hugging Face and FA2, **2.53x faster** than vanilla Hugging Face, and operates with **70% less VRAM**. They also provided links to [Gemma 7b Notebook](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing) and [Gemma 2b Notebook](https://colab.research.google.com/drive/15gGm7x_jTm017_Ic8e317tdIpDG53Mtu?usp=sharing) for free usage on a Tesla T4.

- **Seeking Documentation Links in Pull Requests**: `@caseus_` inquired about a documentation reference as a result of a [GitHub pull request discussion](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1292#discussion_r1493791256), to which `@yamashi` responded with the [relevant documentation](https://github.com/OpenAccess-AI-Collective/axolotl/blob/13199f678b9aab39e92961323bdbce3234ee4b2b/docs/mac.md).

- **Quality before Speed in Model Improvements**: `@dreamgen` emphasized the importance of model accuracy over speed, suggesting that getting the model correct should be the priority.

- **Introducing LoRA-the-Explorer for Efficient Training**: `@caseus_` shared [Minyoung Huh's research](https://minyoungg.github.io/LTE/) on **LoRA-the-Explorer (LTE)**, an approach for training neural networks from scratch with parallel low-rank adapters, which could have significant implications for resource-efficient deep learning. They also highlighted the [multi-head LoRA implementation](https://github.com/minyoungg/LTE/blob/main/lte/mhlora/linear.py) as worthy of interest.

**Links mentioned**:

- [LTE](https://minyoungg.github.io/LTE/): no description found
- [axolotl/docs/mac.md at 13199f678b9aab39e92961323bdbce3234ee4b2b · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/13199f678b9aab39e92961323bdbce3234ee4b2b/docs/mac.md): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [LTE/lte/mhlora/linear.py at main · minyoungg/LTE](https://github.com/minyoungg/LTE/blob/main/lte/mhlora/linear.py): Contribute to minyoungg/LTE development by creating an account on GitHub.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/s/NCqFshpmqs): no description found
- [Mps mistral lora by maximegmd · Pull Request #1292 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1292#discussion_r1493791256): Additional MPS example to train a Mistral Lora. Some documentation on usage and limitations.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1211772507066204250) (1 messages): 

- **R2R: Making RAG Deployment Easy**: User `emrgnt_cmplxty` announced the launch of **R2R**, a semi-opinionated framework for rapid development and deployment of production-ready **RAG systems**. Designed to simplify and streamline deployment, it's an industry move towards practical ease-of-use and effectiveness, and can be found on [GitHub - SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R).

**Links mentioned**:

[GitHub - SciPhi-AI/R2R: A framework for rapid development and deployment of production-ready RAG systems](https://github.com/SciPhi-AI/R2R): A framework for rapid development and deployment of production-ready RAG systems - SciPhi-AI/R2R

  

---


### OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1212026280287936553) (1 messages): 

- **Replicate's Focus Called into Question**: `@dreamgen` expressed surprise, noting an expectation that **replicate** should have a better performance given their years of focus on it, yet apparently, it does not meet those expectations. There was no further elaboration on specific issues or comparisons.
  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1211598894258065430) (33 messages🔥): 

- **Clarification on Comparing AI Models**: `@eugeneyan` pointed out in response to `@guardiang` that the tweet thread comparing models to GPT-4 was based on **zero-shot** performance metrics.
- **Mistral Large Announced with Microsoft Partnership**: `@__chef__` shared a link to [Mistral Large's announcement](https://mistral.ai/news/mistral-large/), detailing its new capabilities and benchmark performance, alongside news of a partnership with Microsoft.
- **Cloudflare's AI Gateway Draws Attention**: `@henriqueln7` showcased Cloudflare's AI Gateway, emphasizing its analytics, logging, and caching features, with its ease of use highlighted as needing just one line of code to get started.
- **Mistral Au Large Integral to RAG**: `@ashpreetbedi` reviewed the RAG integration with Mistral Au Large, praising its function calling and reasoning capabilities. They also shared a link to their cookbook on GitHub: [phidata/mistral](https://github.com/phidatahq/phidata/tree/main/cookbook/mistral).
- **Announcing an eBook on RAG**: `@dimfeld` notified the chat about Jason Liu's upcoming eBook on RAG with varying levels of complexity, with the GitHub repo found here: [n-levels-of-rag](https://github.com/jxnl/n-levels-of-rag). It was acknowledged as a useful resource by `@thenoahhein` for a Twitter data summarization task.

**Links mentioned**:

- [Font Pairing Generator](https://www.monotype.com/font-pairing): Find the perfect font pairing powered by the Monotype AI font pairing generator engine.
- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [AI Gateway · Cloudflare AI Gateway docs](https://developers.cloudflare.com/ai-gateway/): Cloudflare’s AI Gateway allows you to gain visibility and control over your AI apps. By connecting your apps to AI Gateway, you can gather insights on …
- [phidata/cookbook/mistral at main · phidatahq/phidata](https://github.com/phidatahq/phidata/tree/main/cookbook/mistral): Build AI Assistants using function calling. Contribute to phidatahq/phidata development by creating an account on GitHub.
- [GitHub - jxnl/n-levels-of-rag](https://github.com/jxnl/n-levels-of-rag): Contribute to jxnl/n-levels-of-rag development by creating an account on GitHub.

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1211734565870370876) (4 messages): 

- **Language Surprise in Data Extraction**: `@derekpwillis` experienced an amusing issue where **chatgpt-3.5-turbo** used Spanish titles for documents that should have been in English, such as translating "Taking Advantage of the Internet" to *"Sacándole Provecho a Internet"* in the extracted version.

- **Multilingual Mix-up Reminiscent of ChatGPT/Whisper Voice Bug**: `@simonw` related the issue to a similar bug where ChatGPT combined with Whisper sometimes misinterprets a **British accent as Welsh** and responds in Welsh.

- **Suggested Fix with System Prompt**: `@simonw` suggested a potential fix by feeding the system a prompt that instructs it to "Always use English" to avoid language mix-ups.

- **Resolution to Implement Language-Specific Prompt**: In response to the suggestion, `@derekpwillis` acknowledged the issue and indicated he would implement the suggestion to always use English.
  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1211632326996529162) (26 messages🔥): 

- **LLM Plugin Brings Groqcloud to Python Devs**: `@angerman.` shared that the [LLM plugin](https://pypi.org/project/llm-groq/) for accessing [Groqcloud](https://console.groq.com) models is now available, providing instructions for installation and how to obtain an API key. The plugin allows use of models like `groq-llama2` and `groq-mixtral`, with an example given for generating pet names.
- **Python Packaging Tutorial for the Uninitiated**: `@0xgrrr` provided `@angerman.` with a guiding [tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects) on how to package and upload a Python project to PyPI, encouraging them that the process is straightforward.
- **LLM Plugin Improves with Streaming Support**: An update from `@angerman.` mentioned new streaming functionality for the LLM plugin, expressing gratitude to `@746595581086138409` for their contribution to this enhancement.
- **Datasette Developer Hints at Upcoming Chat UI**: `@simonw` revealed that a chat UI for LLM is in the works, although it's still a work-in-progress with no set completion date.
- **Datasette's Choice: Why Fly GPUs?**: `@simonw` explained to `@kiloton9999` that the choice of using Fly GPUs is due partially to their company sponsoring Datasette development, and also because their GPUs have the ability to scale to zero.

**Links mentioned**:

- [llm-groq](https://pypi.org/project/llm-groq/): no description found
- [Packaging Python Projects - Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives): no description found

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1211592072553959424) (9 messages🔥): 

- **Insights on LLM Evaluation Methods**: `@bjoernp` shared an [interesting paper](https://arxiv.org/abs/2402.13887) discussing the limitations of current probability-based evaluation methods for Large Language Models (LLMs), touched on in the DiscoLM series. The paper critiqued how this evaluation strategy often doesn't align with generation-based predictions but did not fully explore why these discrepancies occur.

- **Hidden Null Strings in JSON to Parquet Conversion**: `@thomasrenkert` encountered invisible null strings when converting JSON files to Parquet format, a problem only discerned by directly uploading the JSON to Hugging Face and observing the converted dataset.

- **Building RAG for Codebases**: `@codermickey` enquired about the state of the art in creating a Retrieval-Augmented Generation (RAG) bot for answering questions and assisting with a codebase, including chunking strategies for indexing and retrieval strategies suitable for coding tasks.

- **LangChain's Approach to Codebase RAG**: In response to `@codermickey`, `@johannhartmann` mentioned that LangChain provides a Git loader and segmentation for popular languages, and LlamaIndex has a Git importer, with most people simply using retrieval with OpenAI embeddings and prompts for code-related tasks.

- **End-to-end Optimization Exploration for RAG and LLMs**: `@rasdani` asked if there was research on joint end-to-end optimization of RAG and Large Language Models using gradients, linking a related paper [LESS](https://arxiv.org/abs/2402.04333), although later noting it does not backpropagate through data selection but only retrieves training examples with similar precomputed gradient features.

**Links mentioned**:

- [Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models](https://arxiv.org/abs/2402.13887): Large Language Models (LLMs) have demonstrated remarkable capabilities across various applications, fundamentally reshaping the landscape of natural language processing (NLP) research. However, recent...
- [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333): Instruction tuning has unlocked powerful capabilities in large language models (LLMs), effectively using combined datasets to develop generalpurpose chatbots. However, real-world applications often re...

  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1211590899902058557) (13 messages🔥): 

- **EQ-Bench Now "Sprechen Sie Deutsch"**: `@.calytrix` has announced that German language support has been added to EQ-Bench, with the aim of making it faster and more economical for users. The updated benchmark is accessible on [GitHub](https://github.com/EQ-bench/EQ-Bench).

- **Scores Are In for EQ-Bench in German**: Preliminary benchmark scores have been shared by `@.calytrix` showing **GPT-4's** leading performance with a score of 81.91 on the German version, compared to its 86.05 on the English EQ-Bench.

- **Translation Quality in Question**: `@_jp1_` raises concerns about the German translation's accuracy on EQ-Bench, suggesting **nuances in emotions** may not map directly between English and German.

- **Fluency Affects Emotional Benchmark Scores**: `@.calytrix` reflects on the discussion, noting that while the German EQ-Bench tests for German fluency, it may not capture additional aspects of *emotional intelligence* when compared to the English version.

- **Polish Translation Reveals Model Gaps**: `@remek1972` reports on translating MT-Bench to Polish, highlighting significant differences in model performance on tasks between Polish and English, suggesting language fluency of models like **GPT-4** is a critical factor.

**Links mentioned**:

[GitHub - EQ-bench/EQ-Bench: A benchmark for emotional intelligence in large language models](https://github.com/EQ-bench/EQ-Bench): A benchmark for emotional intelligence in large language models - EQ-bench/EQ-Bench

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/) (1 messages): 

thomasrenkert: thanks for the explanation 🙂
  

---



### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/1211631845087649893) (5 messages): 

- **FireFunction V1 ignites the scene**: `@sourya4` highlighted **FireFunction V1**, a new model boasting **GPT-4-level structured output** and decision-routing at a significantly lower latency, noting its open-weights and commercial usability. They shared [FireFunction's blog post](https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling) and mentioned the model's JSON and grammar modes that help maintain output structure.
- **Exploring the Best in Function Calling**: `@yikesawjeez` listed **Gorilla OpenFunctions, NexusRaven,** and **Litellm Function Calling Wrapper** as their current go-to solutions for function calling. They also hinted at their participation in an upcoming **hackathon** focused on Fire Functions.
- **Mixtral Large: Has anyone ventured yet?**: `@thisisnotawill` inquired whether anyone has tried out **Mixtral Large**, but there were no responses provided within the chat to elaborate on its use or performance.
- **Inquirying Minds Want to Know**: `@justahvee` questioned the specifics about **response latency** in FireFunction V1, asking whether it refers to time to first token or time to completion, comparing it to GPT-4's longer times.

**Links mentioned**:

[Tweet from Lin Qiao (@lqiao)](https://x.com/lqiao/status/1760664322215379153?s=12): 🔥 Structure is all you need. 🔥  We’re excited to announce:  - FireFunction V1 - our new, open-weights function calling model:     - GPT-4-level structured output and decision-routing at 4x lower lat...

  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1211772674267807764) (4 messages): 

- **R2R Sets New Industry Benchmark**: `@emrgnt_cmplxty` announced the launch of **R2R**, a framework for rapid development and deployment of production-ready RAG systems, which aims to simplify the transition from experimental models to production environments. Explore R2R at [GitHub - SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R).
  
- **Seeking Clarity on Frameworks**: `@yikesawjeez` sought a TL;DR on the differences between the new **R2R framework** and [agentmemory](https://github.com/JoinTheAlliance/agentmemory) found on GitHub, asking if there's a difference between the two.

- **Applause for a Twitter Video**: `@yikesawjeez` commended an unspecified *absolute banger of a twitter vid*, acknowledging the good work of a fellow user.

**Links mentioned**:

- [GitHub - SciPhi-AI/R2R: A framework for rapid development and deployment of production-ready RAG systems](https://github.com/SciPhi-AI/R2R): A framework for rapid development and deployment of production-ready RAG systems - SciPhi-AI/R2R
- [GitHub - JoinTheAlliance/agentmemory: Easy-to-use agent memory, powered by chromadb and postgres](https://github.com/JoinTheAlliance/agentmemory): Easy-to-use agent memory, powered by chromadb and postgres - JoinTheAlliance/agentmemory

  

---


### LLM Perf Enthusiasts AI ▷ #[collaboration](https://discord.com/channels/1168579740391710851/1168816033130365018/1211701454960730162) (2 messages): 

- **GPT-4 Tackles Drug Information**: User `@thebaghdaddy` found success in getting GPT-4 to generate informative cards by organizing data into a table covering aspects like mechanisms, side effects, and main disease targets for a list of drugs. Despite being slightly verbose, the method proved effective.
- **Anki's Image Occlusion Feature Missed**: `@thebaghdaddy` highlighted a limitation of GPT-4 not being able to include images in outputs, underscoring the utility of Anki's image occlusion for studying.
  

---


### LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/1212100162579861537) (6 messages): 

- **The Pain of Latency**: `@res6969` expressed distress, saying *this deeply pains me* without specifying the issue.
- **Latency by the Seconds**: `@res6969` highlighted concerns about **latency in seconds** for OpenAI APIs, suggesting it's an issue.
- **Looking for a Solution in Hosting**: `@res6969` inquired whether **dedicated hosting** is the only fix for the latency issues.
- **Azure Hosting Disappoints**: `@pantsforbirds` joined the conversation, mentioning that the Azure results are disappointing, implying dissatisfaction with Azure's performance in this context.
  

---


### LLM Perf Enthusiasts AI ▷ #[rag](https://discord.com/channels/1168579740391710851/1169086375686053890/1211807444452249660) (1 messages): 

- **RAG System Enhancement Brainstorm**: `@jxnlco` shared their work on ways to improve a client's RAG system, inviting feedback. Check out the ideas and contribute on [GitHub](https://github.com/jxnl/n-levels-of-rag/blob/main/README.md).


**Links mentioned**:

[n-levels-of-rag/README.md at main · jxnl/n-levels-of-rag](https://github.com/jxnl/n-levels-of-rag/blob/main/README.md): Contribute to jxnl/n-levels-of-rag development by creating an account on GitHub.

  

---



### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1211602243456081921) (6 messages): 

- **Gemma Enhanced with Turn Tokens**: `@imonenext` has integrated `<start_of_turn>` and `<end_of_turn>` tokens into the **Gemma** model, which is available for use on [Hugging Face](https://huggingface.co/imone/gemma-7b-with-it-tokens). These tokens are for facilitating further instruction/RL fine-tuning.
- **Manual Magic for Token Integration**: The process of adding instruction-tuning tokens to the **Gemma** model was done manually by copying tokenizers, as clarified by `@imonenext`.
- **All Original, No Issues**: In response to a query from `@ufghfigchv`, `@imonenext` specified that there were no issues encountered during the process because the original instruction tokens were used.

**Links mentioned**:

- [Hugging Face – The AI community building the future.](https://huggingface.co): no description found
- [imone/gemma-7b-with-it-tokens · Hugging Face](https://huggingface.co/imone/gemma-7b-with-it-tokens): no description found

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=mw3VvbYE0o8
  

---



### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1212116870539513946) (1 messages): 

- **Live Coding Session on Agent Protocol V2**: `@_z` shared a [YouTube live stream](https://youtube.com/live/zrJuNUGYKJg?feature=share) where they're working on the **Agent Protocol's Config Options RFC for the V2 Milestone**. The stream invites viewers to join and interact while `_z` codes.

**Links mentioned**:

[Coding - Working on Agent Protocol V2 Milestone, Config Options, New RFCs](https://youtube.com/live/zrJuNUGYKJg?feature=share): Hello, I&#39;m Ziggy!I&#39;m an Open Source Developer, gamer, and tech enthusiast. You can find me on GitHub at https://github.com/jzanecook Interested in contributi...

  


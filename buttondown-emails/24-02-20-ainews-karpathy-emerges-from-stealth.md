---
id: a38c933d-dc24-4e9a-809a-b1dd01a4842c
title: '[AINews] Karpathy emerges from stealth?'
date: '2024-02-21T01:54:38.604352Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-karpathy-emerges-from-stealth
---

<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/19/2024. We checked **20** guilds, **313** channels, and **3952** messages for you. Estimated reading time saved (at 200wpm): **346 minutes**.

As mentioned in yesterday's recap, Andrej shipped [his Tokenization tutorial](https://www.youtube.com/watch?v=zduSFxRajkE) with accompanying [github repo](https://github.com/karpathy/minbpe) ([tweet](https://twitter.com/karpathy/status/1759996549109776702)):

https://www.youtube.com/watch?v=zduSFxRajkE

It is sobering how this 2hr tutorial is necessary to fully understand tokenization up to the RegEx patterns used in GPT4's tokenizer, but, as Andrej notes, even then it is far from complete to get up to Llama 2 tokenization with SentencePiece, and yet tokenization was at the core of many LLM failure modes at least from GPT2-GPT4.

--

**Table of Contents**

[TOC] 


# PART 0: SuperSummary

- **Model Optimization and Efficiency**
  - **Quantization and Compatibility**: Discussions highlighted efforts in quantizing models like Mistral 7B and Zephyr-7B, focusing on reducing memory requirements for better compatibility with consumer hardware, notably for running on 8 GB VRAM CUDA GPUs. Intel's exploration into a new weight-only quantization algorithm for LLMs, despite lacking comprehensive documentation, sparked interest for its potential to enhance model efficiency without sacrificing performance
  - **Efficiency Improvements**: A significant focus was placed on improving computational efficiency and model robustness. Techniques include selective augmentation for classifiers, reducing computational costs by an average of 57.76%, and discussions on the efficient use of memory tokens versus traditional methods like kNN for Transformers
  - **Challenges in Model Implementation and Fine-Tuning
Technical Troubleshooting**: Communities shared challenges ranging from hardware compatibility issues (e.g., AVX2 support, multiple GPU configurations) to software-specific problems like VSCode not recognizing certain modules. There's a shared struggle in implementing and fine-tuning AI models, particularly noted in the difficulty of loading quantized versions of models for specific tasks like RAG
  - **Fine-Tuning and Model Merging**: The nuances of fine-tuning LLMs, including the use of LoRA configurations and the complexities of merging models fine-tuned on different datasets, were frequently discussed. These discussions highlight the technical depth and experimentation within the community to optimize model performance and output consistency
- **Advancements and Applications of LLMs**
  - **Innovative Uses of LLMs**: From enhancing RAG applications to exploring multi-model learning strategies, the AI communities are actively exploring ways to extend the capabilities and applications of large language models. The potential of LLMs to act as AI assistants or leverage meta-reasoning capabilities for improved reasoning structures represents the cutting edge of AI research and development
  - **Dataset and Model Accessibility**: The creation and sharing of encoded datasets for machine learning applications, as well as the open-source release of advanced models and codecs, indicate a strong community drive towards democratizing access to AI resources. This includes discussions on SDXL VAE encoded datasets and Audiogen AI's open-source audio codecs
- **Ethical Considerations and Community Engagement**
  - **Censorship and Ethical AI Use**: Conversations around implementing censorship in chat models and constructing model guardrails reflect ongoing concerns regarding ethical AI usage. The community explores various approaches, including discriminator models and prompt tuning, to ensure responsible model behavior
  - **Community Resources and Collaboration**: The AI communities actively collaborate on compiling resources, sharing knowledge, and troubleshooting. This includes the sharing of comprehensive documents to centralize AI resources, discussion on benchmarking models for efficiency and ethical considerations, and addressing technical challenges together.


---

# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Buzzing to a Halt with Linux Mint**: `@kalomaze` encountered frustrating faint buzzing sounds from speakers while using Linux Mint, exacerbated by system crashes affecting functionality like printscreen.

- **Colab's A100 somewhat Alpaca-norous**: `@karisna` faced memory issues while fine-tuning **Mixtral 8x7B** on a Colab A100, despite adjustments in settings and batch size reduction, with recommendations involving gguf management.

- **Intel's Mysterious Quantization Dance**: Intel's "auto-round" GitHub repo sparked conversations on a new weight-only quantization algorithm for LLMs, with community debates around potential benefits given incomplete documentation.

- **Bewildered Over Basic Code for Mistral 7B**: `@vivek2722` sought assistance for loading the quantized version of Mistral 7B for RAG, yet no immediate solutions surfaced in the messages available.

- **Censorship Conundrums**: Various approaches to implement censorship in chat models were discussed, including using a discriminator model, prompt tuning, and exploration of NVIDIA NeMo-Guardrails and reinforcement learning, as highlighted by `@jeremy.london` with a reference to the leaked GPT-4 prompt in a [YouTube video](https://youtu.be/70tZ43aa5J4?si=s5LBw-QLCiqXa_nm).

- **Template Turmoil in Model Merging**: `@givan_002` articulated concerns regarding which template to use after merging two different models, **NeverSleep/X-NoroChronos-13B** and **elinas/chronos-13b-v2** that were fine-tuned on **Vicuna** and **Alpaca** respectively, mindful of the inconsistency in inference outputs.

- **Quantum Leap in Dataset Handling**: Echoing community empathy, `@ikaridev` faced a dataset leak ordeal and addressed the nuances of balancing datasets, which could include sensitive content, in the context of roleplay and function-calling.

- **Guardrails on the Guard**: Drawing upon collective wisdom, strategies for constructing model guardrails were debated, suggesting the use of a more refined and nuanced discriminator model and the injection of a specific token sequence to steer towards desired model behavior.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **LLMs Augment Robustness and Efficiency**: A recent [preprint](https://arxiv.org/abs/2402.08225) highlighted by `@millander` points to the use of **LLMs (Large Language Models)** to improve classifier robustness by rewriting inputs to emulate in-distribution examples. Selective augmentation, through entropy-based evaluation, can also cut computational costs by an average of **57.76%**, offering a more efficient approach.

- **Creative Encoding and Model Portability Discussions**: SDXL VAE encoded datasets, such as **ArtBench and FFHQ**, are accessible on Hugging Face, and a [script](https://github.com/Birch-san/k-diffusion/blob/its-not-dit/imagenet_vae_loading.py) for the SDXL VAE encoding of ImageNet-1k was shared. For applying large-scale AI models in consumer hardware, a quantized **Mistral-7B** was recommended for its compatibility with an 8 GB VRAM CUDA GPU.

- **Memory Tokens and Model Structuring Insights**: Complications of **liquid neural networks** were debated, and distinctions between RMT-T memory tokens and kNN for Transformers were discussed. The impact of conversational context on **LLM** performance was questioned, suggesting decontextualized inputs might improve response quality. Language model training intricacies such as semantics and syntax are recognized as intertwined elements critical for prediction accuracy.

- **Synthetic Prompt Structuring for GPQA**: A new optional subfield in `fewshot_config` was proposed for GPQA to introduce structured fewshot prompts like those used in `minerva_math` or `gsm8k_cot`, as seen in a [GitHub example](https://github.com/idavidrein/gpqa/blob/main/prompts/chain_of_thought.txt).

- **Codec Advancements and GPT-NeoX's Development Path**: **Audiogen AI** announced open-source audio codecs with a discrete model providing a total bit rate of **13.2kbps**, and **GPT-NeoX** was noted for incorporating design elements from **Megatron** and focusing on correct development **priorities**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Pricing Perplexities and AI Alternatives**: The potential **cost of a new OpenAI release** and the **ChatGPT Plus's message cap** agitated users, leading to comparisons with **Google's premium model** and alternatives like **Gemini** for creative writing. Meanwhile, discussions surfaced about educators exploring AI tools like **Claude** and **Microsoft Co-Pilot** for classroom use, emphasizing the sector's eagerness to integrate AI without restrictive barriers like phone verification.
  
- **GPT-4 Browser Bafflement**: Issues with **GPT-4's responsiveness** in web browsers versus mobile for `@iamryuzaki` led to discussions on peer-to-peer AI, difficulties with custom knowledge bank retrieval, localization challenges for AI assistants, and optimizing **Custom GPTs** for voice interactions, particularly for job interview practice.
  
- **Email Categorization Hustles**: `@ben.30` strives to improve an **email categorization process** with a success rate of 75% and a threshold for the 'unknown' category, whereas **prompt debugging techniques** were shared to identify model inconsistencies and potential issues in performance.
  
- **Prompt Engineering Puzzles**: Intricacies in prompt constructions affecting AI outputs were highlighted, with `@eskcanta` advising on self-evaluating prompts using meta-prompts. `@drcapyahhbara` confronted challenges in using GPT for writing novels, with each sentence resembling an introduction, an issue open for guidance in prompt refinement.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **LM Studio's GGUF Model Limitation**: LM Studio has been clarified to support only GGUF models, and there's been no mention of BNF grammar compatibility.

- **Recommended Models for Academic Writing**: **Mistral 7B** was suggested for tasks like academic writing, with discussions touching upon hardware requirements for running such models effectively.

- **Integration and API Concerns with LM Studio**: Questions were raised about integration capabilities of LM Studio, specifically regarding API calls and its compatibility with Azure AI.

- **Hardware Compatibility and Model Installation**: Installation of models contingent upon AVX2 support was a hot topic, with solutions provided for `libcblast` errors and discussions on the potential use of multiple GPUs to handle more powerful models.

- **Model Capabilities and Quantization Methods**: A new llama version of **miniCPM** supporting `fp16/int4 ggufs` is available, while discussions included quantization efficiencies and comparison of the Qwen series in LM Studio.

- **GPU Selection Dilemma for AI**: The NVIDIA RTX **3090** was recommended over the **4070 Super** due to its larger VRAM capacity which is beneficial for running larger model sizes.

- **Attempted Mod for 3090 VRAM Expansion**: A mod to expand the RTX 3090's VRAM to 48GB was discussed, highlighting BIOS restrictions and referencing attempts shown on [Bilibili](https://www.bilibili.com/video/BV1wW4y1K7yE/) and [YouTube](https://www.youtube.com/watch?v=DbF02Y5yIaQ&t=677s).

- **RAM Upgrade Goes Unrecognized in LM Studio**: Even though LM Studio didn't recognize an upgrade from 16GB to 64GB RAM, the issue was identified as a display bug that doesn't affect model functionality.

- **Configuring LM Studio for Multiple GPUs**: There was a request for guidance on assigning LMStudio to a specific GPU when multiple models are available, with advice pointing to a useful instructional thread.

- **AMD GPUs and AI Workloads**: Users discussed the drawbacks of using AMD GPUs for AI workloads, emphasizing Nvidia's optimized hardware support as a preferable choice.

- **VSCode Ignoring 'crewai' Module**: There's a reported problem with Visual Studio Code not recognizing the 'crewai' module, although it appears in `piplist`, indicating a potential IDE-related issue.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **LLMs Can Be Stage Actors**: `@i_am_dom` elucidated that **LLMs** can be finetuned to act as AI assistants, emphasizing the flexibility in behavior shaping during the fine-tuning stage. `@jamshed1900`, `@mrdragonfox`, and `@drnicefellow` concurred **Mistral-next** eclipses its antecedents in reasoning capabilities.
  
- **Innovation in Multi-Model Learning**: `@mehdi_guel` revealed plans for an exploratory venture blending *in-context learning* with *chain-of-thought* strategies. Meanwhile, `@mrdragonfox` educated that **Mixtral's** MoE structure does not entertain the extraction of standalone experts, as the expertise is diffusely embedded within the model.

- **The Varied Mileage of VLLM**: `@ethux` noted inconsistent performance with **VLLM** in a sharded environment, in contrast to seamless operation with **TGI**. The deployment efficacy of **Mixtral** on `g5.48xlarge` instances remained unanswered.

- **The Finer Points of Fine-tuning**: `@timuryun` entered the fine-tuning fray with a question met by keen assistance, mostly from `@mrdragonfox`. Discussion threads touched on using Q&A approaches for model education and the nuances of **LoRA** configurations for fine-tuning a 7B **Mistral** model, advocating for a better grasp of parameters and tuning techniques.

- **Curating a Collective AI Knowledgebase**: User `@_red.j` shared an [AI master document](https://docs.google.com/document/d/1jeVpCc-uxYbxrN9oDmCZRlV3pnzQcpdyOiVD1xBJ9-M/edit?usp=sharing) to centralize resources for AI aficionados, following a conversation with ML experts on Twitter. The community was invited to expand the document with their top AI sources.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Upcoming LlamaIndex Webinar Lights Up the RAG Stage**: LlamaIndex has announced a [webinar for Thursday at 9am PT](https://lu.ma/czvaqzag), showcasing innovative uses of **Retrieval-Augmented Generation (RAG)** by the recent hackathon winners. The webinar reveals advanced knowledge synthesis and reasoning applications like **ADU Planner** and **Counselor Copilot**, providing a glimpse into RAG's potential beyond baseline chatbots.

- **Meta-Reasoning and RAG Reranking Touted in LLM Discussions**: A new paper titled *Self-Discover* posits the integration of **meta-reasoning capabilities in LLMs**, which `@peizNLP` highlighted could enhance traditional AI reasoning structures. Furthermore, Florian June's blog celebrated by LlamaIndex details improving RAG systems with reranking techniques, hinting at smarter data retrieval.

- **Tech Troubles and Tips Tackled in General Chat**: `@wrapdepollo` and `@whitefang_jr` assist users with broken links and updates to text nodes, referencing the [Document Management guide](https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management.html) for support. Meanwhile, `@david1542` and `@cheesyfishes` delve into clustering support in VectorDBs, recommending [usearch](https://github.com/unum-cloud/usearch) for including algorithms like K-Means and DBSCAN.

- **Agonizing Over Agent Behavior**: The agents' integration with tools sparks a conversation led by `@mst2205`, expressing challenges in getting agents to understand and combine results from different query engines. The discourse includes tips such as including the current date in prompts and references to [AgentBench GitHub](https://github.com/THUDM/AgentBench) for evaluating agents.

- **To Customize RAG or Not? A Discussion on Self-Hosting LLMs**: In a debate over creating a custom RAG system versus using RAG-as-a-service, `@skiboyec` and `@desk_and_chair` muse on the benefits of customization against service convenience. The discussion touches on self-hosting, scalability, API costs, and usage intent, reflecting on the specific needs and capabilities required by users.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **LayoutLMv3 Errors Persist**: `truedescription` ran into issues using **LayoutLMv3** from Hugging Face; despite setting truncation and padding to true, an error prevailed without a clear solution.

- **Creating Latin American RL Course Channel**: `sebaskja` showed interest in establishing a channel specifically for **Latin American Reinforcement Learning course members**, looking for guidance.

- **Sora Videos Spark Amusement**: A Twitter link shared by `chalm3rs.` displayed the latest **Sora videos from the OpenAI team**, causing a stir of interest and humor among users.

- **Diving into CI/CD for ML Ops**: `kingabzpro` introduced a [guide for CI/CD in Machine Learning](https://www.datacamp.com/tutorial/ci-cd-for-machine-learning), aimed at aiding the automated deployment and testing process.

- **Generative AI's Rise to Daily Relevance**: An article discussed how **generative AI** has woven itself into the fabric of everyday life in the UK and Australia, with the emergence of "prompt whisperers" shaping the utilization of the technology.

- **Advancing AI Intelligence Measurement**: An **older paper** proposed the necessity of a different feedback signal to gauge AI and human intelligence more accurately, contesting current benchmarking methods.

- **Quantized Zephyr-7B Tailored for Customer Support**: The **Zephyr-7B model** received fine-tuning attention — employing quantization and the [AutoGPTQ](https://huggingface.co/blog/gptq-integration) library — for a customer support chatbot application.

- **Exploring Banner Ads on HuggingFace Spaces**: `myg5702` initiated a discussion about the feasibility of incorporating banner ads on [HuggingFace spaces](https://huggingface.co/spaces), learning they may be acceptable for community-driven content.

- **Visualizing Multilingual Text with Aya Dataset**: `cakiki` highlighted the diversity of languages supported by **CohereForAI's Aya**, sharing a visualization of the [Aya dataset](https://x.com/christopher/status/1759644840885678210?s=20).

- **Annotated Mamba Project Completes**: The **Annotated Mamba**, a project by Sasha Rush, was made available at [Annotated Mamba](https://srush.github.io/annotated-mamba/hard.html), earning praise and discussion for its detailed explanation.

- **Melding Fashion and AI**: In the **#diffusion-discussions** channel, `mohdfaiez` sought assistance to create an AI tool for changing clothes on images, using an example from the [Pincel app](https://blog.pincel.app/photo-clothes-ai/).

- **QLoRA Finetuning Roadblocks**: `kingpoki` struggled with errors during the **QLoRA finetuning** process, seeking community advice on a `NotImplementedError` encountered with `AutoPeftModelForCausalLM.from_pretrained`.

- **Inquiries on Advanced NLP Models**: There were queries regarding code walkthroughs for **RA-DIT and REPLUG** instruction tuning, and challenges with **Whisper large v3** language transcription—which erroneously interpreted Spanish as Chinese—prompting requests for support.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Checkpoints for Keyboard Grace**: `@seungduk` queried the integration of **checkpointing** on a keyboard interrupt (ctrl+c) within the system, and `@nanobitz` confirmed that this feature may have been implemented previously but is uncertain of its current state. Further inspection of the code is initiated with a prompt to review at [OpenAccess AI Collective GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/train.py#L127-L138).

- **Yolo Learns the Hard Way**: `@le_mess` took a risk by running computations on an A100 for 10 days without setting **checkpoints**, prompting a blend of sympathy and humor from other members.`,@yamashi` tagged along with a "yolo" spirit, while `@noobmaster29` reacted with a facepalm emoji.

- **VRAM Diet for a 7B Giant**: `@noobmaster29` shared experiences related to the VRAM requirements for **quantizing a 7B model**, including a helpful script from [TheBlokeAI on GitHub](https://github.com/TheBlokeAI/AIScripts/blob/main/quant_autogptq.py).

- **Benchmarking Blues with BioMistral**: An accusation by `@yamashi` towards **BioMistral** for purportedly misreporting benchmarks sparked a critical exchange on the **accuracy** and **ethics** of model benchmark evaluations, referencing a relevant [tweet](https://x.com/vanstriendaniel/status/1759502442943942746).

- **RunPod Runaround**: Issues with **RunPod** setups were addressed with `@m4ttfl0` sharing a potential workaround for a directory clobbering problem, evidently a known issue [#813 on GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/issues/813). Additionally, `@noobmaster29` lamented over the lengthy and sometimes failing setup process, including an inquiry about an error code that might suggest a system memory shortfall.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Juggernaut Unleashed Without UI**: Members discussed how to operate the **Juggernaut XL** model sans UI, including a helpful checkpoint available at [Hugging Face](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9), accompanied by practical code for implementation.

- **Multipurpose Datasets and LoRA Implications**: Conversations touched on **SDXL VAE** preprocessed image/text datasets with a dataset example at [Hugging Face](https://huggingface.co/datasets/ptx0/photo-concept-bucket/viewer), and debated LoRA's effect on model realism versus trained aesthetic preferences following a [Reddit discussion](https://www.reddit.com/r/StableDiffusion/comments/1au9tfk/rethinking_lora_approaches_for_normal/).

- **Alpha-Prompt LORA Invites Scrutiny**: An **Alpha-Prompt LORA** model introduced by `@qwerty_qwer` and `TwoAbove` promises more detailed SD prompts, with testing welcomed at [Hugging Face](https://huggingface.co/blindsolitaire/Alpha-Prompt).

- **AI Resource Aggregation for the Hungry Minds**: A comprehensive document titled **The AI Info Diet ™️** was shared, featuring a curated list of AI tools, news, and resources, aimed at keeping engineers updated, and is open to community contributions ([Google Doc](https://docs.google.com/document/d/1jeVpCc-uxYbxrN9oDmCZRlV3pnzQcpdyOiVD1xBJ9-M/edit?usp=sharing)).

- **CAD Systems Await AI Revolution**: Inquests about AI-integration into CAD programs surfaced, pointing out the current complications due to a lack of datasets and standards for parametric 3D shapes, while a claim was made about Mistral's new 'Next' AI potentially surpassing capabilities seen in GPT-4 according to early testers ([Reddit source](https://www.reddit.com/r/singularity/comments/1auri0o/some_early_testers_of_mistrals_latest_open_source/)).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Groq Chips Away Competition**: Community members, including `@swyxio` and `@shivdinho`, engaged in discussions on [Groq's performance claims](https://x.com/amanrsanger/status/1759490599152185632?s=46&t=90xQ8sGy63D2OtiaoGJuww), questioning its cost-efficiency and speculating on its real-time application potential. The unique *no-DRAM* and horizontally scalable architecture, loaded with SRAM chips, is of particular interest for its potential in real-time LLM instances.

- **Retrieval Redirection**: `@fanahova` pointed out deficiencies in vector-based retrieval systems, referencing an article on [vector retrieval limitations](https://writer.com/blog/vector-based-retrieval-limitations-rag/) and noting the industry’s overlooking of advancements in graph-based models and methods like HNSW.

- **Chatbot RAG Rethought**: A conversation was sparked regarding the need for retrieval-augmented generation (RAG) in chatbots, where the use of LLMs for deducing user intent and implementing function calls was a central theme.

- **Benchmarking Brilliance with HELM**: The guild discussed the introduction of Prof. Percy Liang's [HELM benchmark](https://crfm.stanford.edu/2024/02/18/helm-instruct.html), a new framework for evaluating LLMs, recognized for its instructional approach and absolute ratings which promise a more nuanced assessment of models. 

- **Miscellaneous Mentions**: Various resources were shared, including an article on [MoonBit’s AI-native toolchain design](https://www.moonbitlang.com/blog/moonbit-ai), a talk by Google's Jeff Dean on [trends in ML](https://www.youtube.com/watch?v=oSCRZkSQ1CE), and the open-source AI wearable project [ADeus on GitHub](https://github.com/adamcohenhillel/ADeus), noting advancements and innovation in the AI space.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **CUDA Cores Unwrapped**: `@nshepperd` sought clarification on whether "cuda core" refers to the **fp32 and int32 arithmetic units**. `@_t_vi_` detailed that each unit executes a **warp's instruction**, with efficient register-based switching. This conversation can aid in better understanding of underlying **CUDA execution mechanics**.

- **PyTorch and GitHub Convergence**: `@p0.tato` pointed to `TensorListMetadata` and `multi_tensor_apply` contributions, while `@ardywibowo` shared a [blog post](https://pytorch.org/blog/accelerating-generative-ai-2/) on LLM acceleration using PyTorch. They also highlighted the existence of [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), and [MLC-LLM](https://github.com/mlc-ai/mlc-llm) for generative model optimization.

- **NVIDIA Secrets and Surprises**: `@apaz` discovered varying `graphics`, `memory`, `video` clock speeds via `nvidia-smi`. `@stefangliga` shared the interesting behavior of boost clocks on NVIDIA GPUs, depending on the temperature, hinting at performance tuning based on environmental conditions.

- **Ring Attention Deep Dive**: `@ericauld` critically assessed [`flash-attention`](https://github.com/Dao-AILab/flash-attention), questioning the backward implementation and sparking a broader discussion on cache mechanics and possible enhancements for ring attention led by `@iron_bound` and `@andreaskoepf`. An issue was opened at [`ring-attention`](https://github.com/cuda-mode/ring-attention/issues/4) for development of a naive version that also manages partial kv-blocks processing.

- **Flash Attention in JAX Spotlight**: `@nshepperd` embarked on integrating [flash attention bindings](https://github.com/nshepperd/flash_attn_jax) into JAX to explore SPMD pattern, and discussed the hurdles like removing Torch dependencies from Tri Dao's flash attention repo, illuminating JAX as an easier platform for this work.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Gemini Pro Houdini Act**: Users report that **Gemini Pro access** disappeared from Perplexity Pro, sparking speculations on potential updates that might introduce **Ultra or Pro 1.5 access**.
- **Timeline Tease for Updates**: A user's query about the update timeline was met with a cryptic reply and a Discord link by `@ok.alex`, suggesting more information might be available through the link.
- **Playground Limits Are Subscription-Free**: In a clarification, `@icelavaman` stated that text length limits in the Perplexity Playground are not tied to subscription levels, and pointed to the API documentation for details on context sizes.
- **Merch March Madness?**: A humorous exchange occurred when `@lord.wex` inquired about Perplexity merchandise, which led to the sharing of a speculative "merch by March" tweet by `@AravSrinivas`.
- **GPT-4 Turbo: Under Lock or Non-Existent?**: Debate ensues on whether Perplexity Pro is using a **GPT-4 turbo**, with confirmations that it's only the standard version, coupled with uncertainty about the availability of a turbo version.
- **Peeking Under Perplexity's Hood**: Articles on the mechanics and designer behind Perplexity AI shared by `@soabonen` and `@sjohri` respectively could provide deep dives for the interested: [How does Perplexity work?](https://www.perplexity.ai/search/How-does-the-8IWXZn7mRCKyB7iLobwTQg?s=c) and [Who designed Perplexity?](https://www.perplexity.ai/search/who-designed-perplexity-ga1T4hdNSxKZmke836IZHg).
- **Fine-Tuning Not an Option**: `@retonq`'s question about fine-tuning the **pplx-online model** was shot down with a clear **No** from `@icelavaman`.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Visualizing LangChain's Internal Workings**: Users discussed the need for visualization tools for LangChain's chains or calls, with some referencing the lack of current documentation on the feature. The [changelog](https://python.langchain.com/docs/changelog/langchain) and [LCEL documentation](https://python.langchain.com/docs/expression_language/) were shared to help users understand the updates and replacements of deprecated Chain Classes.

- **Addressing ChatVertexAI's Validation Errors**: There was a query regarding NEGLIGIBLE level Response Validation errors in ChatVertexAI, with no consensus reached on how to adjust the safety configurations or turn off the response validation.

- **Enhancing Chroma's Retrieval Efficiency**: An idea was floated about improving Chroma's retrieval performance by transforming questions into a list of keywords, intended to produce better results than the current method.

- **Demystifying LLM Parameters**: A brief explanation provided clarity on the parameters in large language models (LLMs), indicating they are weightings applied to tokens from a prompt to generate responses in the backend.

- **Learning Through LangChain Tutorials**: A comprehensive [LangChain tutorial playlist](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ) with over 60 videos was shared to educate users on LangChain use cases, offering a rich resource for developing applications with generative AI. The "LangGraph Retrieval Agent" [video](https://www.youtube.com/watch?v=DFT0tMBwh04) specifically details the use of Retrieval Agents within this context.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Multilingual Expert LLM Development Discussions**: Discussions focused on creating LLMs with expertise in multiple languages and domains, with suggestions like using **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** for pretraining LLMs in new languages and leveraging large-scale clusters utilizing frameworks such as **GPT-Neo-X**, **Megatron-LM**, and **[Axolotl](https://github.com/gretelai/axolotl)**.

- **Clarifying the Cost of Benchmarking**: Users discussed budget-friendly benchmarks, with **fasteval** mentioned as a quick, albeit not free, evaluation tool. It was noted that fasteval still costs over **5€ in OpenAI credits per model**.

- **Temporary Service Interruption Resolved**: The **[DiscoLM German 7b Demo](https://demo.discoresearch.org)** server experienced downtime as the GPUs were in use for evaluations, later confirmed to be back up and running.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Crowdsourced AI Resource Compilation**: User `@_red.j` shared a collaborative [*The AI Info Diet ™* Google Document](https://docs.google.com/document/d/1jeVpCc-uxYbxrN9oDmCZRlV3pnzQcpdyOiVD1xBJ9-M/edit?usp=sharing) with resources for keeping up with the latest in AI. The document welcomes contributions and includes the **Alignment Lab AI Discord server** as a resource.




---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1209047987502977104) (1195 messages🔥🔥🔥): 

- **Audio Buzzing Troubles on Linux Mint**: `@kalomaze` is experiencing random faint buzzing sounds from speakers on Linux Mint and lamenting the ongoing issues with the operating system, including a crash that resulted in the loss of printscreen functionality ([source](#@kalomaze)).
- **Model Training Challenges on Colab A100**: `@karisna` is attempting to fine-tune Mixtral 8x7B on a Colab A100 but runs into memory issues despite trying an array of settings and reducing the batch size significantly. Suggestions involve managing batch sizes and incorporating gguf for the process ([source](#@v2ray)).
- **Intel's Intriguing Quantization Method**: Intel's GitHub repo "auto-round" hints at a new weight-only quantization algorithm for LLMs that may offer better performance at lower precision, causing discussion about potential benefits and skepticism given the lack of full documentation ([source](#tibbnak)).
- **Chatbot Integration with Intel's New Quant**: `@tibbnak` noticed Intel uploaded some quants to Hugging Face, claiming to achieve good benchmark scores on quantization, suggesting it might be as effective as higher precision but at significantly reduced size ([source](#tibbnak)).
- **Concerns Over Model Merging Techniques**: `@givan_002` expressed concerns regarding model merging where different templates are used for base models (such as Vicuna and Alpaca), wondering which template should be used after merging to avoid inappropriate inference outputs ([source](#@givan_002)).

**Links mentioned**:

- [Groq](https://groq.com/): no description found
- [International Obfuscated Python Code Competition](https://pyobfusc.com/#winners>): Obfuscated Python competition
- [@macadeliccc on Hugging Face: &quot;Benefits of `imatrix` quantization in place of quip Quip-# is a quantization…&quot;](https://huggingface.co/posts/macadeliccc/247190826659941): no description found
- [Han Solo Star Wars GIF - Han Solo Star Wars Never Tell Me The Odds - Discover &amp; Share GIFs](https://tenor.com/view/han-solo-star-wars-never-tell-me-the-odds-dont-tell-me-gif-16636876): Click to view the GIF
- [no title found](https://www.amazon.de/Dremel-Werkzeugständer-15-Grad-Schritten-Teleskop-Werkzeugständer-Energieklasse/dp/B0012RQG94/>): no description found
- [Introducing GPTs](https://openai.com/blog/introducing-gpts): You can now create custom versions of ChatGPT that combine instructions, extra knowledge, and any combination of skills.
- [Terminator Terminator Robot GIF - Terminator Terminator Robot Looking - Discover &amp; Share GIFs](https://tenor.com/view/terminator-terminator-robot-looking-flex-cool-robot-gif-978532213316794273): Click to view the GIF
- [LLM Samplers Explained](https://gist.github.com/kalomaze/4473f3f975ff5e5fade06e632498f73e): LLM Samplers Explained. GitHub Gist: instantly share code, notes, and snippets.
- [Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators): We explore large-scale training of generative models on video data. Specifically, we train text-conditional diffusion models jointly on videos and images of variable durations, resolutions and aspect ...
- [Intel/neural-chat-7b-v3-3-int4-inc · Hugging Face](https://huggingface.co/Intel/neural-chat-v3-3-int4-inc): no description found
- [Spider Man Tom Holland GIF - Spider Man Tom Holland Yoink - Discover &amp; Share GIFs](https://tenor.com/view/spider-man-tom-holland-yoink-gif-11346283): Click to view the GIF
- [Samurai Ronin GIF - Samurai Ronin Katana - Discover &amp; Share GIFs](https://tenor.com/view/samurai-ronin-katana-gif-7951635): Click to view the GIF
- [Linux on a $0.15 CH32V003 RISC-V microcontroller #RISCV #Linux](https://blog.adafruit.com/2024/02/19/linux-on-a-0-15-ch32v003-risc-v-microcontroller-riscv-linux/): The linux-ch32v003 project enables the low cost CH32V003 microcontroller to run Linux. It achieves this by using an 8 megabyte SPI PSRAM chip and a RISC-V emulator (the very nice mini-rv32ima by cn…
- [Large Language Models and the Multiverse](https://docs.google.com/document/d/15i8nZSVJju73kHg7vkRbAw6LOknt9ORoqzdOrZu6UX4/edit?usp=sharing): no description found
- [Parov Stelar and Kovacs - Snake Charmer (Lyric Video)](https://www.youtube.com/watch?v=CxG5ckRTqy8): The new song Snake Charmer with KovacsOUT NOW!https://backl.ink/1956235&quot;I think he hypnotized me I feel I have to danceand every time he plays it he puts me ...
- [Reddit - Dive into anything](https://www.reddit.com/r/AskReddit/comments/1auu37t/how_do_you_feel_about_reddit_selling_user_data_t): no description found
- [exllamav2/tests at master · turboderp/exllamav2](https://github.com/turboderp/exllamav2/tree/master/tests): A fast inference library for running LLMs locally on modern consumer-class GPUs - turboderp/exllamav2
- [Reddit - Dive into anything](https://www.reddit.com/r/AskReddit/comments/1auu37t/how_do_you_feel_about_reddit_selling_user_data_to/): no description found
- [GitHub - I-S00N/I-S00N](https://github.com/I-S00N/I-S00N): Contribute to I-S00N/I-S00N development by creating an account on GitHub.
- [GitHub - intel/auto-round: SOTA Weight-only Quantization Algorithm for LLMs](https://github.com/intel/auto-round): SOTA Weight-only Quantization Algorithm for LLMs. Contribute to intel/auto-round development by creating an account on GitHub.
- [FasterTransformer/README.md at main · NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer/blob/main/README.md): Transformer related optimization, including BERT, GPT - NVIDIA/FasterTransformer
- [rtp-llm/README.md at main · alibaba/rtp-llm](https://github.com/alibaba/rtp-llm/blob/main/README.md): RTP-LLM: Alibaba&#39;s high-performance LLM inference engine for diverse applications. - alibaba/rtp-llm
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [GitHub - huggingface/chat-ui: Open source codebase powering the HuggingChat app](https://github.com/huggingface/chat-ui): Open source codebase powering the HuggingChat app. Contribute to huggingface/chat-ui development by creating an account on GitHub.
- [Gradio](https://huggingfaceh4-open-llm-leaderboard.hf.space/?__theme=light): no description found
- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [Alexa Skills Kit SDK for Python | Alexa Skills Kit](https://developer.amazon.com/en-US/docs/alexa/alexa-skills-kit-sdk-for-python/overview.html): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1209061609574768640) (90 messages🔥🔥): 

- **Seeking Clarity on Roleplay and Function-Calling**: User `@gman5723` inquired about a model adept at both roleplay and function calling. `@mrdragonfox` clarified that **function calling essentially involves JSON formatting** and requires manual execution.

- **Dataset Woes and Words of Support**: `@ikaridev` [shared a link to a dataset](https://huggingface.co/datasets/MinervaAI/Aesir-Preview) marked for sensitive content, then later mentioned **the unfortunate leak of their datasets** to `@c.gato`, evoking sympathy and support from the community.

- **Concern Over Deterministic Model Responses**: `@_dampf` observed that the `bagelmisterytour` model tends to repeat phrases, especially at the start, even when changing sampler settings. `@ycros` acknowledged this might be **related to sampler settings or long contexts influencing determinism**.

- **The Challenge of Using Deep Learning Models**: Discussion about using deep learning models included challenges like managing batch sizes, as `@kaltcit` noted a constraint to **batch size 1 with 10420 sequence length** and the problematic nature of a loss registering as 0.0.

- **Learning Rate Recommendations for DPO**: Amidst insights on **Deep Partial Optimization (DPO)**, `@c.gato` suggested much lower learning rates could be necessary, potentially in connection with **LoRA adapters** and double merges as a non-optimal but practiced method.

**Links mentioned**:

[MinervaAI/Aesir-Preview · Datasets at Hugging Face](https://huggingface.co/datasets/MinervaAI/Aesir-Preview): no description found

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1209176590701830164) (13 messages🔥): 

- **Tackling Censorship in Chat Models**:
`@octopus_` raised a question about how to implement censorship for a model, specifically avoiding discussing pricing without context. The strategy should be akin to how ChatGPT manages adult content censorship.

- **Strategies for Implementing Chat Model Guardrails**:
`@maldevide` suggested using a discriminator model to serve as a guardrail or fine-tuning the model according to the desired responses. Further, `@maldevide` proposed prompt tuning, which uses a specific token sequence to guide the model to the intended latent space.

- **Challenges with Discriminator Models**: 
`@octopus_` tried using a discriminator model for censorship but encountered many false positives. `@maldevide` advised increasing the examples in the n-shot to reduce false positives and adding a chain of thought workflow for better reasoning.

- **Insights on Model Censorship and Guardrails**:
`@jeremy.london` referenced a relevant paper and the NVIDIA NeMo-Guardrails project that discusses censorship and guardrails in models. This approach involves confirming whether generated content meets specific rules and then logging and flagging non-compliant outputs.

- **Guardrail Complexity and the Need for Reinforcement Learning**:
`@jeremy.london` noted that reinforcement learning is needed to refine guardrails for practical use, and shared a [YouTube video](https://youtu.be/70tZ43aa5J4?si=s5LBw-QLCiqXa_nm) on the leaked GPT-4 system prompt. Despite efforts, there's always a gap in model censorship which usually begins with the dataset.

**Links mentioned**:

[The LEAKED GPT-4 system prompt is Insane!](https://youtu.be/70tZ43aa5J4?si=s5LBw-QLCiqXa_nm): 🚨BUY or GIFT Beginners course of Generative AI (with 34% Discount) - https://bit.ly/3HQXsQd (Coupon: LETSGO) 🎉🔗 Links 🔗ChatGPT  History - https://chat.op...

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1209405523892633620) (1 messages): 

- **Confusion Over Model Merging and Inference Templates**: User `@givan_002` expressed confusion on which template should be used after merging two models, **NeverSleep/X-NoroChronos-13B** which comes from **Xwin-LM/Xwin-LM-13B-V0.2** and **elinas/chronos-13b-v2**. They are concerned about the potential for inappropriate token outputs since each base model was fine-tuned on different templates, **Vicuna** and **Alpaca** respectively.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1209112041298731028) (1 messages): 

- **Mistral 7B Loading Code Request**: `@vivek2722` asked for the basic code or any useful link to load the **quantised version of Mistral 7B** for Retrieval-Augmented Generation (RAG), mentioning that they were facing issues with the process. No solutions or links were provided in the available message history.
  

---



### Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1209303109407604747) (1 messages): 

- **LLM Augmentation Enhances Classifier Robustness**: `@millander` presented a new preprint showing how to improve **robustness of classifiers** by augmenting inputs using an **LLM (Large Language Model)**. Input rewriting by the LLM makes out-of-distribution inputs resemble in-distribution examples, often outperforming paraphrasing techniques. Read the full preprint on [Arxiv](https://arxiv.org/abs/2402.08225).

- **Reducing LLM Augmentation Costs with Selective Application**: The same preprint by `@millander` also details how **entropy-based selective augmentation** can reduce computational expense by focusing on uncertain model predictions, cutting the augmentation rate by an average of **57.76%**.

- **Discussion on Black Box Classifier Improvement Techniques**: `@millander` invites discussion on their work regarding **black-box classifiers** in `<#747850033994662000>` channel or through direct messaging for those interested in deeper engagement. For a summarized version, check out the [Twitter thread](https://twitter.com/KyleDevinOBrien/status/1758667079849480630?s=20).

**Links mentioned**:

[Tweet from Kyle O'Brien (@KyleDevinOBrien)](https://x.com/KyleDevinOBrien/status/1758667079849480630?s=20)): How can we make classifiers more robust when we can&#39;t modify the weights or assume its architecture  — effectively making it a black box? In our preprint, we demonstrate that we can improve robust...

  

---


### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1209106078810505256) (160 messages🔥🔥): 

- **SDXL VAE Encoded Datasets Availability**: User `@top_walk_town` inquired about image/text datasets preprocessed with the SDXL VAE. `@.mahouko` shared links to ArtBench and FFHQ on Hugging Face, and mentioned that ImageNet-1k was also processed but not publicly uploaded.

- **ImageNet-1k SDXL VAE Encoding Provisions**: `@.mahouko` offered a [converter script](https://github.com/Birch-san/k-diffusion/blob/its-not-dit/imagenet_vae_loading.py) to `@top_walk_town` for encoding ImageNet-1k using the SDXL VAE, claiming it's likely faster than an existing dask script used by `@top_walk_town`.

- **nanoT5 and Positional Embedding Challenges**: During a discussion on T5 training, `@.mahouko` noted that nanoT5's experimentation with ALiBi embeddings was less stable, citing their [GitHub](https://github.com/PiotrNawrot/nanoT5), and suggested that alternative weight initializations from Google's t5x might be nuanced, involving attention scale factor and choice of optimizer.

- **RLHF/RLAIF/Synthetic Data Hackathon Announcement**: User `@canadagoose1` mentioned an RLAIF hackathon on Saturday and referred to the same location as a past Eleuther meetup. The details of what occurs at such a hackathon were inquired about by user `@.the_alt_man`.

- **Large-Scale AI Model Portability for Consumer Hardware Concerns**: `@eyeamansh` sought benchmarks for AI model portability on typical consumer setups to develop open-source applications with models from Hugging Face. Users `@rallio.` and `@_3sphere` suggested looking into setups used by NSFW RP communities and koboldai, while `@philpax` recommended a quantized Mistral-7B for fitting on an 8 GB VRAM CUDA GPU.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/729741769192767510/730095596861521970/1205304062845779999): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [RSVP to Synth Labs Hackathon | Partiful](https://partiful.com/e/GtafWJYWu7DQli81LOHJ?): RLHF/RLAIF/Synthetic data hackathon
- [GitHub - PiotrNawrot/nanoT5: Fast &amp; Simple repository for pre-training and fine-tuning T5-style models](https://github.com/PiotrNawrot/nanoT5?t): Fast &amp; Simple repository for pre-training and fine-tuning T5-style models - PiotrNawrot/nanoT5
- [GitHub - PiotrNawrot/nanoT5: Fast &amp; Simple repository for pre-training and fine-tuning T5-style models](https://github.com/PiotrNawrot/nanoT5?tab=readme-ov-file#things-we-tried-and-didnt-work-out>): Fast &amp; Simple repository for pre-training and fine-tuning T5-style models - PiotrNawrot/nanoT5

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1209081321259802634) (173 messages🔥🔥): 

- **Liquid Networks Critiqued for Complexity**: `.the_alt_man` expressed skepticism regarding liquid neural networks, suggesting they complicate training without justifiable benefits. In a follow-up, they asserted that the introduction of pseudo-neuroscientific elements creates more challenges than enhancements in the models.
  
- **Differences in Memory Token Model Granularity Discussed**: `micpie` detailed the distinctions between RMT-T memory tokens and the kNN approach of memorizing Transformers, highlighting their unique granularities and retrieval methods.

- **Exploring Data Efficiency with Liquid Nets & CNN+Capsules**: `jckwind` spent significant time delving into liquid net structures and data-efficient modeling using a MNIST-based proof-of-concept combining CNNs, capsules, and liquid networks. They also found interest in a recent paper that combines capsules with multi-headed attention.

- **Insight into Model Training Dataset Influence**: `_lm` pondered the negative impact of conversational context on the performance of LLMs like ChatGPT-4, referencing a behavior where decontextualized questions sometimes yield better responses. `catboy_slim_` and `synquid` discussed related work, with `synquid` sharing a related paper on the critical role of causal reasoning in intelligence development.

- **Revisiting Coarse PoS and Semantics in Model Training**: In a discussion initiated by `jstephencorey` about language model training stages, `rybchuk` pointed out that the coarse part of speech might actually pertain to semantics, which is pivotal for next-token prediction. `miaumiks` and `rybchuk` exchanged views on how semantics, grammar, and syntax are all integral components of LLM training.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/729741769192767510/1209228205282689025/1209228205282689025): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Robust agents learn causal world models](https://arxiv.org/abs/2402.10877): It has long been hypothesised that causal reasoning plays a fundamental role in robust and general intelligence. However, it is not known if agents must learn causal models in order to generalise to n...
- [Linear Transformers with Learnable Kernel Functions are Better In-Context Models](https://arxiv.org/abs/2402.10644): Advancing the frontier of subquadratic architectures for Language Models (LMs) is crucial in the rapidly evolving field of natural language processing. Current innovations, including State Space Model...
- [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927): Attention-free language models that combine gating and convolutions are growing in popularity due to their efficiency and increasingly competitive performance. To better understand these architectures...
- [Quickstart - Neural Circuit Policies 0.0.1 documentation](https://ncps.readthedocs.io/en/latest/quickstart.html): no description found
- [AI hype has echoes of the telecoms boom and bust](https://www.ft.com/content/dc47c5f3-9bd4-4da0-a5cb-c795efd14c9c): no description found
- [DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows](https://arxiv.org/abs/2402.10379): Large language models (LLMs) have become a dominant and important tool for NLP researchers in a wide range of tasks. Today, many researchers use LLMs in synthetic data generation, task evaluation, fin...
- [Bytez: DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows](https://bytez.com/read/arxiv/2402.10379): DataDreamer is a tool that helps researchers work with large language models (LLMs), which are powerful AI models for understanding and generating human language. It simplifies tasks like creating syn...
- [Dataset generation - a stereoplegic Collection](https://huggingface.co/collections/stereoplegic/dataset-generation-65389dd75510eb595f8a3797): no description found

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1209208766659039292) (1 messages): 

- **Discussing Fewshot Config for GPQA**: `@hailey_schoelkopf` suggested that an **optional subfield** could be added to `fewshot_config` for GPQA, referencing a structured fewshot prompt like the one found on [GitHub](https://github.com/idavidrein/gpqa/blob/main/prompts/chain_of_thought.txt). They mentioned the possibility of incorporating a hardcoded prompt similar to those used in `minerva_math` or `gsm8k_cot`.

**Links mentioned**:

[gpqa/prompts/chain_of_thought.txt at main · idavidrein/gpqa](https://github.com/idavidrein/gpqa/blob/main/prompts/chain_of_thought.txt): Baselines and analysis for the Google-proof Q&amp;A (GPQA) dataset - idavidrein/gpqa

  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1209059753431666699) (5 messages): 

- **Open Source Audio Codecs by Audiogen**: `@elyxlz` shared a [tweet by Audiogen AI](https://twitter.com/audiogen_ai/status/1759497578331177195?s=19) announcing their open-source audio codecs. No further discussion provided on this tweet.
- **Audiogen Codec on GitHub**: `@elyxlz` provided a link to [Audiogen Codec's GitHub repository](https://github.com/AudiogenAI/agc), showing the source for Audiogen's audio codec efforts.
- **Audiogen Codec Bit Rate Details**: `@nostalgiahurts` highlighted the discrete model's codebook size of 2048, which results in a total bit rate of **13.2kbps** as detailed in the codec's [Hugging Face configuration](https://huggingface.co/Audiogen/agc-discrete/blob/main/config.json).
- **EMA Proves Beneficial for GAN Vocoder**: `@nostalgiahurts` remarked that using Exponential Moving Average (EMA) has proven beneficial, noting it's a newer application in the field of GAN vocoders though it's been seen before in BigGAN.
- **Low Hanging Fruit in GAN Vocoder Improvement**: Responding to the comment about EMA, `@elyxlz` acknowledged that employing EMA was quite a straightforward enhancement for their GAN vocoder.

**Links mentioned**:

[GitHub - AudiogenAI/agc: Audiogen Codec](https://github.com/AudiogenAI/agc): Audiogen Codec. Contribute to AudiogenAI/agc development by creating an account on GitHub.

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1209366212954423337) (2 messages): 

- **GPT-NeoX Gets Inspiration from Megatron**: User `@jdranpariya` acknowledged that elements of **GPT-NeoX** are derived from **Megatron**.
- **Affirmation of Prioritization**: In a succinct follow-up, `@jdranpariya` seemed to affirm that the development team has the right **priorities**.
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1209077913056710667) (103 messages🔥🔥): 

- **GPT Pricing Speculation and Employment Enthusiasm**: Users debated the potential cost of a new OpenAI release, with `@theultimateprompter` suggesting a pricing strategy based on multiplying frames by minutes and the DALL·E pricing model. `@solbus` chimed in, emphasizing the uncertainties around pricing and services but confirmed that **no official announcements** have been made yet.

- **The Cap Debate**: `@sevenero` expressed frustration with ChatGPT Plus's message cap, leading to a cancelled subscription, and compared it unfavorably to Google's premium model without a message cap, which sparked a discussion of supply and demand in relation to OpenAI's capacity.

- **Message Limit Frustrations and Google's AI Comparison**: User `@blckreaper` discussed using Google's Gemini for tasks due to ChatGPT's message limits and claimed Gemini's creative writing style bypasses **AI detection** successfully. The conversation continued with `@droggerhd` highlighting the superior accuracy of **GPT-4** and the anticipation for model 1.5 with a large context window.

- **Exploring AI Alternatives for Education**: `@smitha` inquired about how teachers could use ChatGPT without phone verification for students, sparking a discussion about using alternative AI tools like Claude in classroom settings and Microsoft Co-Pilot. 

- **Predictability in AI Joke Generation**: `@sugarsniper` observed a pattern in ChatGPT's responses when requesting "groaner jokes", leading to a further exploration with `@eskcanta` regarding how AI's structured training influences the diversity and creativity of its outputs.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1070955045639684096/1209358912797802506): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1209049375427395584) (115 messages🔥🔥): 

- **GPT-4 Browser Troubles for @iamryuzaki**: `@iamryuzaki` is facing issues with GPT-4 not responding on any browser despite having a subscription and trying multiple computers and browsers. The bot works on mobile but remains unresponsive in web browsers.

- **Discussing Peer-to-Peer AI Philosophy**: `@jay_low666` mused about the concept of an AI with peer-to-peer protocol, akin to Napster, utilizing the power of PCs worldwide. In response, `@darthgustav` humorously remarked on the potential inefficiency and insecurity, jesting about unwanted gaming hacks infiltrating AI.

- **The Ups and Downs of Custom Knowledge Banks**: `@jaredquek` reported problems with Custom GPT knowledge bank retrieval; despite proper instructions, the bot favors online searches or pulls entire texts. `@darthgustav` engaged actively, suggesting troubleshooting steps like disconnecting and reconnecting knowledge sources.

- **Global vs. Local Markets Chatbot Dilemma**: `@ricardop20` deliberated whether to target his AI assistant to the global market or focus on local needs in Portugal. `@darthgustav` advised on the versatility of localization which allows for catering to both markets effectively.

- **Voice Chat Optimizations for Interview Practice**: `@718moe` inquired about Custom GPTs optimized for voice chat, specifically for job interview practice. `@eskcanta` guided through an approach using basic ChatGPT to craft and refine instructions, suggesting an iterative process to develop a concise yet helpful bot.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1209257937600778270/1209257937600778270): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1209091190507569172) (44 messages🔥): 

- **Seeking Prompt Optimization for Email Categorization**: `@ben.30` is looking to improve the accuracy of an email categorization system using Power Automate and GPT-3.5; success rate currently is at 75%, with the system designed to attribute a confidence level to its assessments and to default to 'unknown' if below a certain threshold.
- **Complex Prompts and Model Response**: `@darthgustav` suggests that if previous API calls aren't linked, reviewing commonalities in failure cases could illuminate the root cause. They also mention the relative retrieval rate (51%) when GPT-3.5's context is about half full.
- **Evaluating and Refining AI Prompts**: `@eskcanta` offers advice to `@d1scobo1` on how to evaluate and rewrite complex AI prompts to reduce restrictive conditions and enhance performance; they share examples of how prompts affect AI responses.
- **Troubleshooting Knowledge Base From Uploaded Files**: `@eskcanta` raises awareness about a known bug affecting AI performance related to knowledge from uploaded files which could be impacting `@d1scobo1`'s assistant behavior.
- **Novel Writing Prompt Challenge**: `@drcapyahhbara` encounters difficulties with GPT creating narrative content where every sentence is treated as an introduction, resulting in unnatural transitions; they are seeking guidance in the prompt-engineering channel.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1209165553198043208): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1209091190507569172) (44 messages🔥): 

- **Optimizing Email Categorization with GPT**: `@ben.30` seeks to improve the success rate of an email categorization process that utilizes Power Automate with a GPT function. Despite a current 75% success rate and a 5% threshold for uncategorized emails, he is looking for further improvements and input on prompt structure.
- **In-Depth Prompt Structure Shared**: In a dialog with `@darthgustav`, `@ben.30` explains his detailed prompt structure which includes context, service descriptions, and keywords, and seeks an external review of his current prompt design for possible improvements.
- **Prompt Review and Debugging Advice**: `@eskcanta` offers an approach to reviewing and debugging prompts by submitting them to ChatGPT with a meta-prompt, to self-evaluate inconsistencies or potential issues which could be affecting model performance.
- **Streamlining Complex Instructions for Better AI Performance**: `@d1scobo1` embeds the AI on a website to answer client questions based on a provided file about software engineering career details. Upon feedback, they are optimizing the directives to allow more natural responses and address a potential bug shared by `@eskcanta`.
- **Dialogue on Enhancing Novel Writing Prompts**: `@drcapyahhbara` reports an issue regarding the AI's tendency to treat every sentence as an introduction to a novel, leading to unnatural transitions. `@eskcanta` offers support on prompt engineering in the `#api-discussions` channel.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1209165553198043208): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1209049897660317746) (141 messages🔥🔥): 

- **LM Studio Format Confusion**: User `@heyitsyorkie` explained that LM Studio only runs GGUF models, not other formats like `@suisse7` inquired about BNF grammars.
- **Model Recommendations and Usage Discussions**: `@egalitaristen` shared a [GitHub link](https://github.com/LargeWorldModel/LWM) and engaged in a discussion on what model might be best for tasks like academic writing, suggesting Mistral 7B. Conversation revolved around hardware capabilities for different context sizes.
- **Integration and API Queries**: Multiple users like `@akiratoya13`, `@kvrmd`, and `@i.apol0` inquired about various integration capabilities with LM Studio, such as sending system messages via API calls and connecting LM Studio to Azure AI.
- **Model Installation and Hardware Compatibility Issues**: Users like `@digit18` and `@krypt_lynx` discussed challenges and solutions around installing models and the necessity of AVX2 instruction support, with `@heyitsyorkie` providing a workaround for `libcblast` related errors.
- **Exploring Advanced Use Cases for LM**: `@krypt_lynx` and `@jedd1` talked about the potential for using multiple GPUs for LLMs, with the former user considering an unplanned system upgrade to accommodate more powerful models.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1111440136287297637/1208800020212621402): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html): no description found
- [no title found](https://news.ycombinator.com/item?id=24578591): no description found
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/beta-releas): Find, download, and experiment with local LLMs
- [teknium/OpenHermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B): no description found
- [GitHub - LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM): Contribute to LargeWorldModel/LWM development by creating an account on GitHub.
- [Introduction to Weight Quantization](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c): Reducing the size of Large Language Models with 8-bit quantization
- [NVIDIA Tesla P40 Specs](https://www.techpowerup.com/gpu-specs/tesla-p40.c2878): NVIDIA GP102, 1531 MHz, 3840 Cores, 240 TMUs, 96 ROPs, 24576 MB GDDR5, 1808 MHz, 384 bit

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1209092564645314570) (23 messages🔥): 

- **miniCPM Struggles and Anticipations**: `@dootmate` expressed frustrations as miniCPM still doesn't work, looking forward to a fix. The discussion pointed towards an [issue on GitHub](https://github.com/ggerganov/llama.cpp/issues/5276) mentioning lacks support in llama.cpp, with hopes that LMStudio's update to the latest build might resolve this.
- **miniCPM Now Supports llama.cpp**: `@dootmate` mentioned the release of a llama version of miniCPM, `openbmb/MiniCPM-2B-dpo-bf16-llama-format`, and further noted that it now has support for `fp16/int4 ggufs`.
- **Potency of Qwen Models in LMStudio**: In a comparative query, `@borisrusev` questioned the capabilities of q2_k versus q8 in the context of Qwen1.5-72b-chat, to which `@heyitsyorkie` humorously likened Q8 to the smart set and Q2 to the less capable counterpart.
- **Best LLM for Coding and Vision**: For coding LLMs, `@heyitsyorkie` recommended **Deepseek Coder 33b**, and `@r3vs_` inquired about llava-1.6-mistral-7b for vision, to which heyitsyorkie admitted a lack of experience with vision models.
- **Hardware Considerations for Running Large Models**: `@old_skooler` shared excitement about running Mixtral 8x7b Dolphin with expected new memory, and `@jedd1` provided practical information on model operation speeds, VRAM usage, and performance drops when exceeding VRAM capacity.

**Links mentioned**:

- [MiniCPM/README-en.md at main · OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM/blob/main/README-en.md): MiniCPM-2B: An end-side LLM outperforms Llama2-13B. - OpenBMB/MiniCPM
- [MiniCPM 2b model support? · Issue #5276 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5276): Feature Description Like Phi is supported, it would great to have this Mistral level 2b model ggufable. Motivation SOTA 2b model, a piece of art, read how they made it: https://shengdinghu.notion.s...
- [k-quants by ikawrakow · Pull Request #1684 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/1684): What This PR adds a series of 2-6 bit quantization methods, along with quantization mixes, as proposed in #1240 and #1256. Scalar, AVX2, ARM_NEON, and CUDA implementations are provided. Why This is...

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1209112330559037470) (63 messages🔥🔥): 

- **The Great GPU Debate**: User `@j.o.k.e.r.7` sought advice on whether to choose a **3090** or **4070 Super**, both available at the same price, sparking a discussion on performance and VRAM. `@heyitsyorkie` recommended the **3090 for its 24GB of VRAM** and superior performance in tasks like gaming, stable diffusion, and running up to 70b Q4 models, with `@nink1` suggesting looking for second-hand deals due to miners offloading cards.

- **Modding the 3090 for Extra VRAM**: `@.bambalejo` shared their interest in a VRAM upgrade mod for the RTX 3090 to reach 48GB, noting limitations due to VBIOS and sharing links to [Bilibili](https://www.bilibili.com/video/BV1wW4y1K7yE/) and a [Russian YouTube video](https://www.youtube.com/watch?v=DbF02Y5yIaQ&t=677s) detailing attempts at the modification.

- **Big RAM, No Recognition in LM Studio**: After `@ethanboyle` upgraded their RAM from 16GB to 64GB, LM Studio failed to recognize the new capacity. `@heyitsyorkie` explained this as a known bug but assured that the models would still work despite the inconsistency, advising that clearing specific cache locations might resolve the issue.

- **Choosing GPUs for LM Studio**: `@dyter07` inquired about how to assign LMStudio to use a specific GPU when multiple are installed, prompting `@jedd1` to reference a helpful thread that gives instructions on setting GPU preferences.

- **Steer Clear of AMD GPUs for AI?**: As `@seicross` explored how to utilize their AMD Rx 5500 xt for language models, `@exio4` commented on AMD's lack of optimized support for AI workloads, suggesting users might find better performance and value with Nvidia's hardware instead.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1136793122941190258/1208938884495450165): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Видеокарта из мусорки и можно ли 48 гигабайт поставить в ROG Strix RTX3090? Эксперимент за 50000руб.](https://www.youtube.com/watch?v=DbF02Y5yIaQ&t=677s): Можно ли в обычную видеокарту Asus ROG Strix RTX3090 установить 48Gb видеопамяти GDDR6X? Но для проведения эксперимента сначала достанем эту видеокарту из му...
- [不知天高地厚的up准备把3090 改成48g 能成吗_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1wW4y1K7yE/): no description found

  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1209319009556897882) (3 messages): 

- **VSCode unable to recognize 'crewai' module**: `@circulustreme` is experiencing an issue where **Visual Studio Code (VSC)** is not acknowledging the installed `crewai` module despite efforts to install, upgrade, and manage packages through various terminals and conda. The module appears in `piplist` but VSC doesn't seem to recognize it.
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1209059761639653396) (104 messages🔥🔥): 

- **Debunking LLM's Acting Skills**: `@i_am_dom` clarifies a common misconception about LLMs, explaining that their behavior is shaped during the fine-tuning stage to act as AI assistants. The fact that you can make an LLM "act any way you want" at this stage was emphasized to indicate the ability to shape its responses.
- **Mistral-next versus Llama Performance Inquiry**: Users `@jamshed1900`, `@mrdragonfox`, and `@drnicefellow` discuss the performance of Mistral-next. Although performance comparisons are limited, there's a consensus that Mistral-next shows better reasoning than its predecessors.
- **Finetuning Finesse Request**: `@timuryun` seeks assistance with finetuning, getting directed to someone experienced - `<@266127174426165249>`, by `@drnicefellow`.
- **Open Source or Not?**: Questions about whether models like Mistral-next are open source and available for download arose in conversation with `@timuryun`, `@drnicefellow`, and `@mrdragonfox`. It was clarified that Mistral-next is currently a prototype test on lmsys and not openly available.
- **Discussing AI Model Capabilities and Investments**: A discourse led by `@i_am_dom`, `@mrdragonfox`, and others suggests that the infrastructure, funding, and expertise behind Mistral are comparable to those at OpenAI, though decisions on training much larger models like a 100B from scratch are still under wraps.

**Links mentioned**:

[Chat with Open Large Language Models](https://chat.lmsys.org/): no description found

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1209087275175116830) (22 messages🔥): 

- **Experimenting with Multi-Model Approaches**: `@mehdi_guel` plans to experiment by combining *in-context learning* with *chain-of-thought* in a multi-model approach and will update the group on the results.

- **Expert Extraction Enigma**: `@redbrain` inquired about extracting individual experts from **Mixtral** for use as standalone dense **Mistral 7b models** for experimental purposes, recognizing the impracticality but expressing interest in the conceptual possibility.

- **Understanding MoE's Nature**: Multiple posts by `@mrdragonfox` clarified that a Mixture of Experts (MoE) model like **Mixtral** cannot have its experts extracted as standalone models because the expertise is not isolated; it's distributed across the model, and the routing happens at the token level.

- **Deconstructing MoE Could Be Futile**: In response to `@redbrain`'s continued interest, `@mrdragonfox` explained that attempting to deconstruct **Mixtral** would likely not produce coherent outputs, and emphasized that even if the process is feasible, the result wouldn't surpass a standard **7b Instruct 0.2 model**.

- **Performance Puzzle with GPU vs. CPU**: `@mikifireblue` observed a slow token generation rate when using a GPU (NVIDIA GTX 1660 TI) compared to using only CPU, while testing with the model "mistral-7b-instruct-v0.1.Q4_K_M.gguf" and llama-cpp, leading `@ginterhauser` to suggest trying an AWQ format as it is better suited for CPU usage.
  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1209119958928195644) (4 messages): 

- **Shard Struggles with VLLM**: `@ethux` mentioned that **VLLM** does not work well with **sharding** and expressed disappointment with performance.
- **Trouble Taming VLLM; TGI Triumphs**: In a separate message, `@ethux` confirmed having the same sharding issue with VLLM but reported no problems using **TGI**.
- **Inquiring About Mistral on a Giant**: User `@espadrine` questioned if **Mixtral** was successfully deployed on a `g5.48xlarge` instance, but no follow-up information was provided.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1209197658070192252) (22 messages🔥): 

- **Faint Presence of @timuryun Acknowledged**: The user `@timuryun` signified their presence and readiness to discuss fine-tuning.

- **Anticipation Builds as Question Looms**: `@mrdragonfox` is primed to assist, urging `@timuryun` to pose their fine-tuning query.

- **Elusive Fine-tuning Inquiry Emerges**: `@timuryun` queries `@266127174426165249`'s expertise in fine-tuning, though responses suggest more details are needed.

- **Dissatisfaction Acknowledged, Assistance Offered**: Despite `@timuryun` providing little detail, the community remains responsive, with `@mrdragonfox` encouraging a post of "stuff" for potential aid.

- **Contemplating the Depths of Fine-tuning Strategies**: `@sven_72358` opens a dialogue on the efficacy of using Q&A pairs for model education, referencing both a tryhellix ai article and personal attempts with a 7B model, which lead to an interjection from `@tom_lrd` discussing general skepticism around imparting knowledge through fine-tuning and a GitHub project.

- **LoRA Configurations Debated in the Finetuning Frontier**: `@iamcoming5084` reaches out for configuration advice on fine-tuning the 7B Mistral model, which spurs `@mrdragonfox` to differentiate between LoRA and full fine-tuning methodologies, emphasizing the need for careful parameter selection and methodology understanding.

**Links mentioned**:

[base_model: mistralai/Mistral-7B-v0.1model_type: MistralForCausalLMtokenizer - Pastebin.com](https://pastebin.com/a23QAq0X): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1209278986924851270) (2 messages): 

- **The AI Info Diet - A Master Doc for AI Enthusiasts**: User `@_red.j` shared a [master document](https://docs.google.com/document/d/1jeVpCc-uxYbxrN9oDmCZRlV3pnzQcpdyOiVD1xBJ9-M/edit?usp=sharing) aimed at helping people stay updated with the latest tools, news, and information in AI, created during a Twitter space conversation with ML experts.
- **Open Invitation to Contribute**: `_red.j` encouraged everyone to add their favorite AI news and information sources to the document, mentioning they're adding the server to it as well.

**Links mentioned**:

[The AI Info Diet ™️](https://docs.google.com/document/d/1jeVpCc-uxYbxrN9oDmCZRlV3pnzQcpdyOiVD1xBJ9-M/edit?usp=sharing): no description found

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1209268685793927262) (1 messages): 

- **LlamaIndex Webinar Announcement**: `@jerryjliu0` announced a **webinar for Thursday at 9am PT**, focusing on innovative community demos that won the recent LlamaIndex hackathon. The session will explore **RAG use cases** beyond basic chatbots, and viewers can [register here](https://lu.ma/czvaqzag).
- **Showcasing Hackathon Winners**: The LlamaIndex webinar will feature four projects that creatively use RAG for advanced knowledge synthesis and reasoning:
  - **ADU Planner**: Streamlining accessory dwelling unit planning, [view project](https://devpost.com/software/adu-planner).
  - **Counselor Copilot**: Aiding counselors with AI, [view project](https://devpost.com/software/counselor-copilot).
  - **neThing.xyz**: Enhancing learning and knowledge, [view project](https://devpost.com/software/nething-xyz).
  - **Home.AI**: Innovating home management, [view project](https://devpost.com/software/home-ai).

**Links mentioned**:

[LlamaIndex Webinar: RAG Beyond Basic Chatbots · Zoom · Luma](https://lu.ma/czvaqzag): RAG is one of the main use cases for LLMs, but many developers are using RAG to build basic Q&amp;A chatbots over simple, static datasets. What are use cases for RAG beyond basic chatbots? We&#x27;re....

  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1209180382642311289) (3 messages): 

- **Meta-Reasoning for LLMs in the Works**: `@peizNLP` introduced a new paper titled *Self-Discover*, which focuses on enhancing **LLMs with meta-reasoning capabilities** by having them autonomously select appropriate reasoning modules. This advancement could revolutionize the traditional fixed reasoning structures seen in AI. [See the tweet](https://twitter.com/llama_index/status/1759620529982755324).
  
- **Exploring RAG's Potential Beyond Q&A**: LlamaIndex announced a webinar discussing the diverse applications of **Retrieval-Augmented Generation (RAG)** beyond simple Q&A chatbots, hinting at innovative community use cases yet to be revealed. [Mark your calendars with this link](https://twitter.com/llama_index/status/1759708989682856343).

- **Enhancing RAG with Smart Reranking**: Florian June's blog post garnered praise from LlamaIndex for its accessible guide on implementing reranking techniques in **RAG systems**, including the usage of a BGE-based reranker and LLM-powered alternatives. [Deep dive into reranking](https://twitter.com/llama_index/status/1759749496161042876).
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1209064447021154375) (118 messages🔥🔥): 

- **Broken Notebook Link Alert**: `@wrapdepollo` noted that the example notebook linked on the [Document Management page](https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management.html) has an inaccessible link, but provided an alternative URL that works. They emphasized this in case the issue was unintentional or if others needed access to the notebook.
- **Discord User Requests Node Update Guidance**: `@yashshukla9279` sought advice on updating text in a node and ensuring metadata alignment within LlamaIndex. `@whitefang_jr` directed them to the [document management guide](https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management.html) for CRUD operations on nodes.
- **AzureOpenAI vs OpenAI Reliability Concerns**: User `@theoxd` shared their experience of AzureOpenAI being less reliable than the standard OpenAI interface, with tools ceasing to function every week or so. There's no follow-up response to this concern within the provided message history.
- **Querying VectorDB Clustering Support**: `@david1542` asked if anyone knew of a VectorDB that supports clustering algorithms like K-Means and DBSCAN. `@cheesyfishes` replied mentioning usearch's capabilities with a link to its [GitHub repository](https://github.com/unum-cloud/usearch).
- **Thorny Agent Interactions with Tools**: `@mst2205` described a difficulty in getting a ReActAgent to comprehend and combine the results of a date tool and an obsidian query engine for handling queries like "Which note did I write today?" `@cheesyfishes` suggested including the current date in the prompt and reflected on the general challenges of agent behavior with open-source LLMs.



**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1059199217496772688/1163395083475898410/1163727902169366559): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Google Colaboratory](https://colab.research.google.com/drive/1Ib6T6CqAEXrnbhTF0-auEHldfK2USknk?usp=sharing): no description found
- [Document Management - LlamaIndex 🦙 v0.10.8.post1](https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management.html): no description found
- [Customizing LLMs within LlamaIndex Abstractions - LlamaIndex 🦙 v0.10.8.post1](https://docs.llamaindex.ai/en/latest/module_guides/models/llms/usage_custom.html#example-using-a-custom-llm-model-advanced): no description found
- [GitHub - THUDM/AgentBench: A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR&#39;24)](https://github.com/THUDM/AgentBench): A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR&#39;24) - THUDM/AgentBench
- [[Bug]: OpenAIEmbeddings is broken in 0.10.6 · Issue #10977 · run-llama/llama_index](https://github.com/run-llama/llama_index/issues/10977): Bug Description Hey everyone :) I&#39;m trying to store &amp; embed some documents using OpenAI embeddings but the process seems to crash due to an illegal assignment to the embed_model object. This i...
- [Module Guides - LlamaIndex 🦙 v0.10.8.post1](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/modules.html#id1): no description found
- [Build Agents from Scratch (Building Advanced RAG, Part 3)](https://www.youtube.com/watch?v=T0bgevj0vto): In this third video of this series we teach you how to build LLM-powered agentic pipelines - specifically we teach you how to build a ReAct agent (Yao et al....
- [GitHub - unum-cloud/usearch: Fast Open-Source Search &amp; Clustering engine × for Vectors &amp; 🔜 Strings × in C++, C, Python, JavaScript, Rust, Java, Objective-C, Swift, C#, GoLang, and Wolfram 🔍](https://github.com/unum-cloud/usearch): Fast Open-Source Search &amp; Clustering engine × for Vectors &amp; 🔜 Strings × in C++, C, Python, JavaScript, Rust, Java, Objective-C, Swift, C#, GoLang, and Wolfram 🔍 - unum-cloud/usearch

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1209232534744072273) (3 messages): 

- **RAG Customization vs. Service**: `@skiboyec` queried about the advantages of building a custom RAG system over using a RAG-as-a-service provider, questioning if a custom system can offer better retrieval performance without concerns for self-hosting, scalability, or API costs.
- **Purpose Defines Building or Subscribing**: `@desk_and_chair` responded speculating that the decision might depend on the goal—whether to leverage RAG for personal use or to offer RAG as a service to others. They likened the situation to preferring a good burger without necessarily wanting to grill it themselves.
  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1209081735392796682) (63 messages🔥🔥): 

<ul>
  <li><strong>LayoutLMv3 Troubles</strong>: `truedescription` faced an error when using LayoutLMv3 from Hugging Face with the processor and model, even after setting truncation and padding to true. An explicit suggestion to resolve the issue wasn't provided in the discussion.</li>
  <li><strong>Craving for Creation</strong>: `sebaskja` expressed interest in creating a channel for Latin American RL course members and sought guidance on how to set it up.</li>
  <li><strong>Video Mayhem</strong>: `chalm3rs.` shared a link from Twitter showcasing the latest Sora videos by the OpenAI team, stirring interest and amusement amongst the users.</li>
  <li><strong>Quest for Knowledge on APIs</strong>: `dipto7613` sought assistance on making an API for illusion but faced challenges due to a profusion of endpoints and an expressed need for more information.</li>
  <li><strong>Conversational AI Performance Evaluation</strong>: `rwamit` asked for the best methods to evaluate a fine-tuned NER model, hinting at an interest in IOB tagging but the conversation ended without a clear resolution.</li>
</ul>

**Links mentioned**:

[Tweet from Borriss (@_Borriss_)](https://fxtwitter.com/_Borriss_/status/1759295962994835571): The Sora videos posted by the OpenAI team are getting wilder..  (Part 2)  7 new ones:  

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1209116496207413258) (4 messages): 

- **Generative AI Takes Over**: User `@erksu.` shared an article discussing how **generative AI** became integrated into the daily lives of the majority, with a specific focus on its use among teenagers in the UK and employees in Australia. The article highlights the shift from curiosity to practical applications such as studying, advice, and creating content, along with a mention of "[prompt whisperers](https://www.abc.net.au/news/science/2023-04-02/prompt-engineers-share-their-tips-on-using-chatgpt-generative-ai/102165132)."

- **Reflecting on the Nature of Intelligence**: `@sebaskja` linked to an **older paper** suggesting that benchmarking AI simply based on skill at specific tasks isn’t enough to measure true intelligence. The paper's abstract argues for the need for a better feedback signal to **evaluate AI and human intelligence** [Download PDF](https://arxiv.org/pdf/1911.01547.pdf).

- **Fine-Tuning Zephyr-7B**: `@not_lain` found an insightful blog detailing how **Zephyr-7B** was fine-tuned using quantization, PEFT, and SFTTrainer for a **customer support chatbot**. It also discusses the integration of the [AutoGPTQ](https://huggingface.co/blog/gptq-integration) library by Huggingface to enable low-precision operations on models.

- **Vibrant GIF Cheers Up the Chat**: `@moonmhmed` posted a humorous and lively GIF with the message "Why Should Your Mouth Have All The Fun", originally from **Saturday Night Live**. The GIF serves to inject a bit of fun into the conversation, featuring [Cecily Strong's swing dance](https://media1.tenor.com/m/mOfFINxBu5EAAAAC/why-should-your-mouth-have-all-the-fun-swing.gif).

**Links mentioned**:

- [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547): To make deliberate progress towards more intelligent and more human-like artificial systems, we need to be following an appropriate feedback signal: we need to be able to define and evaluate intellige...
- [Why Should Your Mouth Have All The Fun Swing GIF - Why Should Your Mouth Have All The Fun Swing Fun - Discover &amp; Share GIFs](https://tenor.com/2LAi.gif): Click to view the GIF
- [2023 was the year of generative AI. What can we expect in 2024?](https://theconversation.com/2023-was-the-year-of-generative-ai-what-can-we-expect-in-2024-219808): Generative AI has changed the ways we work, study and even pray. Here are some highlights of an astonishing year of change – and what we can expect next.
- [Finetuning using Zephyr 7B Quantized model on a custom task of customer support chatbot](https://medium.aiplanet.com/finetuning-using-zephyr-7b-quantized-model-on-a-custom-task-of-customer-support-chatbot-7f4fff56059d): 🤗 Huggingface in collaboration with bitsandbytes incorporated the AutoGPTQ library into Transformers. This integration enabled users to…

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1209090162521284688) (8 messages🔥): 

- **Banner Ads Inquiry on HuggingFace Spaces**: User `@myg5702` inquired about the possibility of having banner ads on [huggingface.co spaces](https://huggingface.co/spaces/FumesAI/Best-Image-Models-Demo), prompting a response from `@lunarflu`, who specified that ads might be allowed for community-incentivizing content like Patreon links but not random ads that could monetize HuggingFace's resources.
  
- **Launching a CI/CD Machine Learning Guide**: `@kingabzpro` announced a comprehensive [CI/CD for Machine Learning guide](https://www.datacamp.com/tutorial/ci-cd-for-machine-learning) that covers everything from GitHub repository setup to automating model testing and deployment with GitHub Actions, aimed at simplifying the journey into ML Ops.

- **Visualization of the Aya Dataset**: `@cakiki` shared a visualization of the [Aya dataset](https://x.com/christopher/status/1759644840885678210?s=20), showcasing the languages supported by CohereForAI's Aya.

- **Discussion on Server Specs for Fast Image Generation**: `@amirgame197` brought up the swift image generation performance of `@myg5702`'s server, to which the latter revealed the use of a powerful Nvidia A40 large instance on a cloud server.

**Links mentioned**:

- [Best Image Models Demo - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-Demo): no description found
- [Tweet from Christopher Akiki (@christopher)](https://x.com/christopher/status/1759644840885678210?s=20): The Languages of @CohereForAI&#39;s Aya.
- [A Beginner&#x27;s Guide to CI/CD for Machine Learning](https://www.datacamp.com/tutorial/ci-cd-for-machine-learning): Discover the most user-friendly MLOps guide online and master the process of automating model training, evaluation, versioning, and deployment with GitHub Actions. 
- [GitHub - kingabzpro/CICD-for-Machine-Learning: A beginner&#39;s project on automating the training, evaluation, versioning, and deployment of models using GitHub Actions.](https://github.com/kingabzpro/CICD-for-Machine-Learning): A beginner&#39;s project on automating the training, evaluation, versioning, and deployment of models using GitHub Actions. - kingabzpro/CICD-for-Machine-Learning

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1209149933513482300) (6 messages): 

- **Annotated Mamba Completed**: `@tea3200` shared a completed resource on the annotated mamba, a project by Sasha Rush, available at [Annotated Mamba](https://srush.github.io/annotated-mamba/hard.html).
- **Praise for Annotated Mamba**: `@lunarflu` expressed admiration for the annotated mamba, hinting at its potential to become a legendary blog post.
- **Clarification on Authorship**: `@tea3200` clarified that the annotated mamba was written by Sasha Rush, and the idea of having it posted on HuggingFace's platform was mentioned.
- **Vision Transformers on the Horizon**: `@tea3200` announced an intention to create a similar annotated resource, but for vision transformers.
- **Encouragement for New Project**: `@lunarflu` showed support for `@tea3200`'s upcoming project on vision transformers, encouraging them to proceed.

**Links mentioned**:

[Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1209087138897862686) (3 messages): 

- **Seeking Guidance on AI Clothing Tool**: User `@mohdfaiez` inquired about starting an AI tool to enable users to change clothes according to their needs.
- **Clarifying the Use Case**: In response to `@m.0861`'s query if the requirement was for a 3D model or image generation, `@mohdfaiez` shared a [blog post](https://blog.pincel.app/photo-clothes-ai/) about **Pincel**, an app that uses AI to change clothes on photos, indicating a similar image generation application is the goal.

**Links mentioned**:

[Change Clothes on Photo Using AI - Pincel](https://blog.pincel.app/photo-clothes-ai/): Change clothes on a photo effortlessly with Pincel AI, the best online app for fast and easy outfit changes using instant AI magic.

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1209201743662153879) (4 messages): 

- **QLoRA Finetuning Frustration**: User `@kingpoki` encountered an error while using `AutoPeftModelForCausalLM.from_pretrained` for **QLoRA finetuning**, which occurs during the merging process with the model. They posted their code snippet and a `NotImplementedError` stack trace, seeking insight into the issue.
  
- **Seeking Guidance on RA-DIT and REPLUG**: `@austintb.` inquired about any available code walkthroughs or demos on **RA-DIT** or **REPLUG** instruction tuning for **RALMs (Realm Adaptive Language Models)**.

- **Whisper Misinterpretation Mystery**: `@pantera4738` is struggling with **hugging face API** transcription using **whisper large v3**; the API outputs the transcription in Chinese instead of Spanish for the provided audio file. They shared their **Python** code to seek help with the language discrepancy in transcription.
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1209087138897862686) (3 messages): 

- **Quest for AI-powered Wardrobe Changes**: User `@mohdfaiez` inquired about building an AI tool that would enable users to change clothes on images. They are seeking guidance on where to start this endeavor.
- **Clarifying Concept Visualization**: `@m.0861` asked for clarification on whether `@mohdfaiez` aims to change clothes on a 3D model or through image generation.
- **Revelation of AI Fashion Tech**: `@mohdfaiez` shared an example with the [Pincel app](https://blog.pincel.app/photo-clothes-ai/), a photo editor that uses AI to change outfits in photos, inviting inspiration for their project. The app allows users to upload a photo, mark areas with a brush, and then swap clothes using AI, as highlighted in their infusion of creativity and technology.

**Links mentioned**:

[Change Clothes on Photo Using AI - Pincel](https://blog.pincel.app/photo-clothes-ai/): Change clothes on a photo effortlessly with Pincel AI, the best online app for fast and easy outfit changes using instant AI magic.

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1209048876632252437) (71 messages🔥🔥): 

- **Yolo Computational Risks**: `@le_mess` gambles with 10 days of A100 computing without checkpoints, prompting empathy and a humorous "yolo" acknowledgment from `@yamashi` and facepalm emoji from `@noobmaster29`.
- **Quantizing a Giant**: `@noobmaster29` discusses the VRAM requirements to quantize a 7B model, sharing experiences and resources such as a script from [TheBlokeAI's GitHub](https://github.com/TheBlokeAI/AIScripts/blob/main/quant_autogptq.py).
- **BioMistral Called Out**: `@yamashi` criticizes BioMistral for allegedly misreporting benchmarks, sparking a discussion about the accuracy and fairness of model benchmarking practices.
- **Training Time Teasers**: `@le_mess` jests about the extreme range of time it could take to fine-tune a 1.3B model, from "5 seconds to 109 years," and then provides a serious estimate of 6 days for `@qwerty_qwer`'s 2.4 million training pairs on a 4090 GPU.
- **Model Training Observations**: `@c.gato` contemplates how sample packing might affect training due to the higher effective learning rate for longer context samples and wonders if it should be a concern.

**Links mentioned**:

- [Tweet from Daniel van Strien (@vanstriendaniel)](https://x.com/vanstriendaniel/status/1759502442943942746): BioMistral is a new 7B foundation model for medical domains, based on Mistral and further trained PubMed Central. - top open-source medical Large Language Model (LLM) in its weight class - Apache Lice...
- [microsoft/phi-1_5 · Hugging Face](https://huggingface.co/microsoft/phi-1_5): no description found
- [vsungwaterloo](https://wandb.ai/vsungwaterloo/runpod-experiments/runs/icst9ntv/overview?workspace=user-vsungwaterloo): Weights & Biases, developer tools for machine learning
- [vsungwaterloo](https://wandb.ai/vsungwaterloo/runpod-experiments/runs/i): Weights & Biases, developer tools for machine learning
- [AIScripts/quant_autogptq.py at main · TheBlokeAI/AIScripts](https://github.com/TheBlokeAI/AIScripts/blob/main/quant_autogptq.py): Some simple scripts that I use day-to-day when working with LLMs and Huggingface Hub - TheBlokeAI/AIScripts

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1209075303683588097) (5 messages): 

- **Considering Checkpoint on Keyboard Interrupt**: `@seungduk` inquired about the possibility of the system saving a checkpoint when a keyboard interruption (ctrl+c) occurs, suggesting it could be configurable.
- **Past Feature of Checkpointing Revisited**: `@nanobitz` acknowledged that checkpointing during a keyboard interruption was previously implemented but expressed uncertainty about its proper functionality.
- **Check Underway for Implementation Integrity**: Following the discussion, `@seungduk` mentioned they would investigate the matter further.
- **Code Inspection by the Collective**: `@caseus_` provided a [GitHub link](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/train.py#L127-L138) to the specific code segment related to training and potentially checkpointing, inviting members to review the implementation.

**Links mentioned**:

[axolotl/src/axolotl/train.py at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/train.py#L127-L138): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1209313599693000704) (6 messages): 

- **Directory Clobbering Workaround Shared**: User `@m4ttfl0` provided a solution for the directory clobbering issue by suggesting the use of a custom template with a different persistent volume mountpoint, sharing the issue link for reference: [RunPod template not working with network volumes](https://github.com/OpenAccess-AI-Collective/axolotl/issues/813).

- **Inquiry About RunPod Setup Time**: User `@noobmaster29` asked how long it should take for RunPod to set up, noting that their setup seemed to be taking an unusually long time.

- **Frustration with Stuck Setups**: `@noobmaster29` expressed frustration over encountering several non-responsive ("dead") pods during the setup process.

- **Seeking Clarification for Error Code**: `@noobmaster29` asked if the error code `-9` indicated an out-of-system memory issue.

**Links mentioned**:

[RunPod template not working with network volumes, /workspace/axolotl empty · Issue #813 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/813): Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior Other users also encountered this: #467 According t...

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1209059348991578123) (67 messages🔥🔥): 

- **Juggernaut XL Checkpoint Discussed**: `@spirit_from_germany` inquired about generating images using the **Juggernaut XL** model without UI, leading to a conversation that included a checkpoint on [Hugging Face](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) and code snippets for implementation.
- **Image/Text Datasets with Preprocessed Embeddings**: `@top_walk_town` queried about datasets with images preprocessed by **SDXL VAE**, and `@pseudoterminalx` mentioned the existence of such datasets, though with certain quality limitations, providing a [Hugging Face link](https://huggingface.co/datasets/ptx0/photo-concept-bucket/viewer) for reference.
- **Reddit and LoRA Approaches Open for Debate**: `@segmentationfault8268` shared a [Reddit post](https://www.reddit.com/r/StableDiffusion/comments/1au9tfk/rethinking_lora_approaches_for_normal/) discussing LoRA approaches, which sparked a conversation around model realism and trained aesthetic preferences.
- **Alpha-Prompt LORA Shared for Testing**: `@qwerty_qwer` extended an invitation to test an **Alpha-Prompt LORA** model that was co-developed with `TwoAbove`, designed to generate detailed SD prompts from descriptions, and available on [Hugging Face](https://huggingface.co/blindsolitaire/Alpha-Prompt).
- **AI Tools, News, and Resources Compilation**: `@_red.j` introduced a master Google Doc titled **The AI Info Diet ™️**, compiled during a Twitter space with ML experts, meant to keep up with the latest in AI, and open for additional contributions ([link to document](https://docs.google.com/document/d/1jeVpCc-uxYbxrN9oDmCZRlV3pnzQcpdyOiVD1xBJ9-M/edit?usp=sharing)).

**Links mentioned**:

- [Image Gallery](https://tripleback.net/viewer/): no description found
- [RunDiffusion/Juggernaut-XL-v9 · Hugging Face](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9): no description found
- [Juggernaut XL - V9 + RunDiffusionPhoto 2 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/133005/juggernaut-xl): For business inquires, commercial licensing, custom models, and consultation contact me under juggernaut@rundiffusion.com Juggernaut is available o...
- [blindsolitaire/Alpha-Prompt · Hugging Face](https://huggingface.co/blindsolitaire/Alpha-Prompt): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/1au9tfk/rethinking_lora_approaches_for_normal/): no description found
- [The AI Info Diet ™️](https://docs.google.com/document/d/1jeVpCc-uxYbxrN9oDmCZRlV3pnzQcpdyOiVD1xBJ9-M/edit?usp=sharing): no description found
- [ptx0/photo-concept-bucket · Datasets at Hugging Face](https://huggingface.co/datasets/ptx0/photo-concept-bucket/viewer): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1209138056456830996) (5 messages): 

- **Goody2 Model Card Introduced**: User `helium__` shared a [link to the Goody2 model card](https://www.goody2.ai/goody2-modelcard.pdf), perhaps suggesting it as a resource for interested parties.
- **Search for AI-Enhanced CAD Programs**: User `glasscow` inquired about any CAD programs that incorporate AI for real-time model designing, aiming to simplify the creation of 2D or 3D models for use in the Unity engine.
- **Challenges in AI-Powered CAD Development**: `unjay.` highlighted difficulties in developing AI-powered CAD software, citing the lack of standardization in parametric 3D shapes and the absence of suitable datasets.
- **AI Precision vs Human Consistency**: User `atlasunified` made a point about precision in AI, contrasting it with humans' ability for consistent repetitive precision.
- **Mistral's 'Next' AI Might Outdo GPT-4**: `vrus0188` shared a [Reddit link](https://www.reddit.com/r/singularity/comments/1auri0o/some_early_testers_of_mistrals_latest_open_source/) stating that early testers believe Mistral's latest open-source 'Next' AI could surpass GPT-4.

**Links mentioned**:

[Reddit - Dive into anything](https://www.reddit.com/r/singularity/comments/1auri0o/some_early_testers_of_mistrals_latest_open_source/): no description found

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1209061513252573226) (70 messages🔥🔥): 

- **Groq's Competitive Speed and Cost**: `@swyxio` and others discussed the performance claimed by Groq, including speculation on how it achieves such speed and cost-efficiency. `@slono` mentioned applying for access hoping it transforms their work, while `@shivdinho` pondered the real-time applications it may enable.

- **Deep Dive into Groq's Technology**: `@coffeebean6887` spent time understanding Groq's approach, sharing their realization of its unique no-DRAM, horizontally scalable architecture, featuring a significant number of SRAM chips for real-time LLM instances. Economical viability, considering the high cost of Groq's infrastructure, is questioned by the community.

- **Limitations of Vector-Based Retrieval Highlighted**: In relation to an article on vector-based vs graph-based retrieval, `@fanahova` comments on the misleading marketing that disregards the modern advancements of methods like HNSW since their early beginnings.

- **Discussions on Chatbot RAG Implementation**: Users discussed determining the necessity of retrieval-augmented generation (RAG) for user messages in chatbots. Ideas included LLMs to deduce user intent, user controls, asynchronous multi-level responses, and the power of function calls by `@ashpreetbedi`.

- **New HELM Benchmark Announced by Percy Liang**: `@swyxio` shared the launch of Stanford Professor Percy Liang's new HELM benchmark, an instructional evaluation framework complete with absolute ratings for more comprehensive LLM assessment.

**Links mentioned**:

- [Tweet from Aman Sanger (@amanrsanger)](https://x.com/amanrsanger/status/1759490599152185632?s=46&t=90xQ8sGy63D2OtiaoGJuww): Groq looks very good  I’d suspect it’s possible to achieve this speed with bs=1, 4-bit weights, and speculative decoding on 4-8 H100s  But even on bs=4 H100 pricing, that would cost at least $2.5/1M t...
- [no title found](https://news.ycombinator.com/item?id=39428880.): no description found
- [no title found](https://news.ycombinator.com/item?id=39435930): no description found
- [Stanford CRFM](https://crfm.stanford.edu/2024/02/18/helm-instruct.html): no description found
- [MoonBit: Exploring the design of an AI-Native Language Toolchain | MoonBit](https://www.moonbitlang.com/blog/moonbit-ai): Exploring the design of an AI-Native Language Toolchain
- [Jeff Dean (Google): Exciting Trends in Machine Learning](https://www.youtube.com/watch?v=oSCRZkSQ1CE): Abstract: In this talk I’ll highlight several exciting trends in the field of AI and machine learning. Through a combination of improved algorithms and major...
- [untitled](https://asciinema.org/a/vANer5sMCVI4H6YE8tCeHJo43): Recorded by fanahova
- [The limitations of vector retrieval for enterprise RAG — and what to use instead](https://writer.com/blog/vector-based-retrieval-limitations-rag/): Vector retrieval has limitations in enterprise use cases, but graph-based RAG offers a superior approach for accurate knowledge retrieval.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1audftm/wow_this_is_crazy_400_toks/): no description found
- [GitHub - adamcohenhillel/ADeus: An open source AI wearable device that captures what you say and hear in the real world and then transcribes and stores it on your own server. You can then chat with Adeus using the app, and it will have all the right context about what you want to talk about - a truly personalized, personal AI.](https://github.com/adamcohenhillel/ADeus): An open source AI wearable device that captures what you say and hear in the real world and then transcribes and stores it on your own server. You can then chat with Adeus using the app, and it wil...

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1209115491801632828) (6 messages): 

- **Sketchy Feelings on Discord**: `@cropinky` expressed discomfort over an unspecified subject, saying, *“This feels illegal to do”*, a sentiment echoed by `@apaz` with, *“That does look really illegal”*.
- **Contentious Link Shared**: A questionable link was shared by `@euclaise`, which only shows a string followed by a long encrypted image data without further explanation.
- **Reassurance from joseph_en**: Addressing `@cropinky`'s concern, `@joseph_en` provided reassurance about needing to demonstrate **llama7B** and **13B** on a single system and explained it as a necessary workaround for a technical challenge.
- **Tips for Using NVIDIA Tools**: `@gogators.` advised that all NVIDIA tools are compatible with Python scripts for CUDA files. They recommend using breakpoints with `cuda-gdb` and cite the efficiency of debugging standalone `.cu` files over Python processes, with the assistance of automation features like **GPT-4** providing mock driver functions.

**Links mentioned**:

[Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html): no description found

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1209268494282129438) (3 messages): 

- **Sasha Rush's Latest Contribution**: User `@mortezism` shared a link to Sasha Rush's annotated **Mamba model**, which can be found at [Annotated Mamba](https://srush.github.io/annotated-mamba/hard.html). The message also contained an unprocessed image code.

- **Inquiry about Running FP8 Operations in Triton**: `@neuralink` asked if non-matrix multiplication operations like element-wise addition or square root could be run in FP8 using **Tensor Cores in Triton**.
  
- **Understanding Tensor Core Capabilities**: `@iron_bound` responded to `@neuralink`, explaining that **Tensor Cores** are specifically designed to perform matrix multiplication and accumulation. They did not provide further information on FP8 operations in Triton.

**Links mentioned**:

[Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html): no description found

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1209324504292134972) (1 messages): 

- **Pure, Native PyTorch Pushes the Limits**: `@ardywibowo` shared [a blog post](https://pytorch.org/blog/accelerating-generative-ai-2/) discussing the acceleration of generative AI models using pure, native PyTorch that focuses on LLM optimization. The post cites performance improvements over 8x for Segment Anything and spotlights open-source projects like [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), and [MLC-LLM](https://github.com/mlc-ai/mlc-llm).

- **Compiled Kernels vs. Traditional Libraries**: `@ardywibowo` is skeptical about the blog's claim that compiled kernels can outperform CuBLAS & FlashAttention2 and questions whether this is legitimate.
- **Seeking Benchmarks for Performance Comparison**: `@ardywibowo` inquires if there are comprehensive benchmarks available that compare PyTorch features with other solutions like FasterTransformer, TensorRT, etc.
- **When to Choose torch.compile Over CUDA?**: `@ardywibowo` seeks insights from the community on deciding when to opt for `torch.compile` as opposed to diving into CUDA mode for optimizations.

**Links mentioned**:

[Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/): This post is the second part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance...

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1209182535536353332) (6 messages): 

- **Parallel Algorithms Meet Hardware Efficiency**: `@ericauld` shared a [Twitter post](https://twitter.com/darkproger/status/1745041586394648975) highlighting how state space models like **Mamba** could benefit from hardware-aware parallel scans.

- **Deep Dive into Automatic Differentiation**: `@ericauld` expressed an interest in studying *automatic differentiation* more deeply and cited it in relation to **FlashAttention's** recomputation technique. They provided a link to a book on Amazon titled [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://www.amazon.com/Evaluating-Derivatives-Principles-Algorithmic-Differentiation/dp/0898716594), highlighting its potential usefulness.

- **Group Study on Gradient Checkpointing Proposed**: `@ericauld` mentioned the relevance of **gradient checkpointing**, referenced in a specific book, and showed interest in chapters 1-4. They invited others to read and discuss together.

- **A Call for Cost-Effective Learning**: `@msthil2` responded positively to the idea of group study and also lamented the high cost of academic books. They joked about being "noob tier" but willing to engage with the material.

- **Cost-Friendly Alternative Suggestion**: `@iron_bound` suggested exploring online libraries as a solution to access expensive academic content without the steep price tag.

**Links mentioned**:

[Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation: Griewank, Andreas, Walther, Andrea: 9780898716597: Amazon.com: Books](https://www.amazon.com/Evaluating-Derivatives-Principles-Algorithmic-Differentiation/dp/0898716594): no description found

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1209145848186019860) (10 messages🔥): 

- **Presentation Mix-Up Averted**: `@cs_os_05101` mistakenly attributed a presentation to `@euclaise` which led to a small confusion, but was quickly rectified by `@apaz` redirecting the credit to the correct individual, Jane (`<@354465570030092290>`).
- **GitHub Gems for PyTorch Enthusiasts**: `@p0.tato` clarified that their presentation was on OSS code from the PyTorch repository, specifically highlighting `TensorListMetadata` and `multi_tensor_apply` in [MultiTensorApply.cuh](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultiTensorApply.cuh#L42) and a related [PR that was reverted](https://github.com/pytorch/pytorch/pull/118604).
- **Discovering NVIDIA's Triple Clock Speeds**: `@apaz` learned about the three different clock speeds (`graphics`, `memory`, `video`) that can be queried using `nvidia-smi`, adding a new layer of insight into GPU performance monitoring.
- **NVIDIA's Boost Clock Secrets Revealed**: `@stefangliga` shared a fun fact that the advertised boost clock for NVIDIA GPUs is not absolute, and the actual clocks can vary depending on environmental conditions, with a rough estimate of gaining 50MHz for every 1°C reduction in temperature.

**Links mentioned**:

- [pytorch/aten/src/ATen/native/cuda/MultiTensorApply.cuh at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultiTensorApply.cuh#L42): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [Build software better, together](https://github.com/pytorch/pytorch/pull/118604.): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

  

---


### CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1209088319787638836) (4 messages): 

- **CUDA Core Clarification Request**: `@nshepperd` inquired whether the term "cuda core" specifically refers to the **fp32 and int32 arithmetic units**.
- **Understanding CUDA Core Processing**: `@nshepperd` speculated that there could be **interleaved processing** or pipelining when there are more threads than arithmetic units.
- **Insight on CUDA Execution Mechanics**: `@_t_vi_` explained that each of the four units within a CUDA core executes a **warp's or subwarp's instruction** at a given time, highlighting the efficient switching mechanism due to static registers within the **register file**.
- **Acknowledging the Explanation**: `@lucaslingle` expressed his understanding and gratitude for the clarification provided by `@_t_vi_`.
  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1209268484836429894) (10 messages🔥): 

- **Exploration into XLA and SPMD**: `@nshepperd` mentioned working on [flash attention bindings](https://github.com/nshepperd/flash_attn_jax) for JAX, gaining insights into how SPMD (Single Program, Multiple Data) works within XLA through this process.
- **Binding Projects Could Aid Ring Attention Development**: `@nshepperd` brought attention to the potential relevance of flash attention repo bindings for the Ring Attention project, suggesting a connection to the extensive Jax implementation mentioned by `@ericauld` in the Ring Attention channel.
- **JAX as a Preferred Environment for Flash Attention**: `@nshepperd` noted the existence of multiple flash attention projects coded in pure JAX, perhaps because it's perceived easier than using CUDA.
- **Torch Dependency in Tri Dao's Repo an Obstacle**: `@nshepperd` also addressed the challenge of removing Torch dependencies from Tri Dao's flash attention repo, implying it's a non-trivial task.
  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1209188522989330483) (28 messages🔥): 

- **Diving into `flash-attention` Mechanics**: `@ericauld` initiated a focused examination of the **[`flash-attention` GitHub repository](https://github.com/Dao-AILab/flash-attention)**, highlighting key areas of interest such as the `compute_attn_1rowblock` and `compute_dq_dk_dv_1colblock` methods. They underscored basic questions about the backward method's integration with PyTorch and the data structures involved.
- **Backtracking the Backward Pass**: `@ericauld` and `@mickgardner` exchanged insights on the complex backward implementation in the `flash-attention` repo, with `@mickgardner` acknowledging its daunting nature and `@ericauld` identifying the potential use of `flash::copy` for gradient communication.
- **TPU Architecture and Cache Mechanics Explored**: `@iron_bound` discussed aspects related to caching on Google's TPUs, sharing a [link to related code](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py) in the `jax` GitHub repository, while `@nshepperd` provided additional context on TPU architecture including references to scratch space and CMEM.
- **Ring Attention vs. Flash Attention Reimplementation**: `@ericauld` questioned the rationale behind rewriting the `flash-attention` logic in the `ring_attention.py` file, sparking a discussion about reusability and customization for ring attention, particularly from `@andreaskoepf`.
- **Contributor Coordination on Ring Attention**: `@andreaskoepf` proposed potential improvements to the `flash-attention` codebase to support ring attention via a PR and outlined plans to scrutinize the Jax implementation. An [issue](https://github.com/cuda-mode/ring-attention/issues/4) was created for developing a naive educational version that processes partial kv-blocks.

**Links mentioned**:

- [jax/jax/experimental/pallas/ops/tpu/flash_attention.py at main · google/jax](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py): Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - google/jax
- [Extend educational naive flash-attn impl to allow partial kv-block processing (create naive ring-attn) · Issue #4 · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/issues/4): Extend the naive flash-attn notebook to allow block-wise processing of only a fraction of the blocks at a time, i.e. pass in and out state required to continue updating the outputs for the current ...
- [GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention](https://github.com/Dao-AILab/flash-attention): Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.
- [A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library](https://arxiv.org/abs/2312.11918v1): We provide an optimized implementation of the forward pass of FlashAttention-2, a popular memory-aware scaled dot-product attention algorithm, as a custom fused CUDA kernel targeting NVIDIA Hopper arc...

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1209067938896814110) (34 messages🔥): 

- **Gemini Pro Access Vanishes**: Users `@harlon0389`, `.themantis`, and `@jaicraft` discuss the disappearance of **Gemini pro access** from Perplexity Pro, leading to speculations about updates potentially adding **Ultra or Pro 1.5 access**.
- **No Set Timeline for Updates**: `@sandeepmuthangi` inquires about a timeline for updates, to which `@ok.alex` responds by providing a vague reply and a Discord redirect link, potentially alluding to information related to the query.
- **Perplexity Chat Length Limits Clarification**: `@icelavaman` explains to `@roy_royce` that text length limits in the Perplexity Playground are independent of subscriptions. The API documentation outlines context sizes, indicating potential future increases; this info is independent of a Pro subscription.
- **Merchandise Inquiry Leads to Humor**: `@lord.wex` asks about **Perplexity merchandise** which prompts `@mares1317` to share a link to a speculative tweet by `@AravSrinivas` about "merch by March," causing amusement among users like `@jaicraft`.
- **Users Discuss GPT-4 Availability and Speed**: `@abiggenius` ponders whether Perplexity Pro uses **GPT-4 turbo**, but `@icelavaman` confirms it's the standard version. Further discussion by `@gooddawg10` suggests **availability is still uncertain**, while `@brknclock1215` shares an unrelated link discussing Perplexity's market strategy.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649619695390740): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1205101971980427274): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [More than an OpenAI Wrapper: Perplexity Pivots to Open Source](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/): Perplexity CEO Aravind Srinivas is a big Larry Page fan. However, he thinks he&#039;s found a way to compete not only with Google search, but with OpenAI&#039;s GPT too.
- [no title found](https://docs.perplexity.ai,): no description found
- [Supported Models](https://docs.perplexity.ai/docs/model-cards): no description found
- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/AravSrinivas/status/1743016479786950868?s=20): merch by march?

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1209100236291575849) (2 messages): 

- **Discovering Perplexity's Mechanics**: `@soabonen` shared a link exploring the inner workings of Perplexity: [How does Perplexity work?](https://www.perplexity.ai/search/How-does-the-8IWXZn7mRCKyB7iLobwTQg?s=c)
- **Unveiling the Designer of Perplexity**: `@sjohri` provided a link to find out who was behind the design of Perplexity: [Who designed Perplexity?](https://www.perplexity.ai/search/who-designed-perplexity-ga1T4hdNSxKZmke836IZHg)
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1209269722580000769) (2 messages): 

- **No Fine-tuning for pplx-online Model**: User `@retonq` inquired about the possibility of **fine-tuning a pplx-online model**. However, `@icelavaman` responded with a definitive **No**, accompanied by an emoji.
  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1209055692577574992) (18 messages🔥): 

- **Seeking Visualization of LangChain's Chains**: `@andreu.codina` asked about visualizing chains or calls in LangChain, noticing the absence of such information in the current documentation, referring perhaps to a feature observed a month ago.
  
- **LangChain Chain Classes Update**: `@rajvir3` provided a detailed list of deprecated Chain Classes linking to the [changelog](https://python.langchain.com/docs/changelog/langchain), and questioned how to implement `SimpleSequentialChain` and `Sequential Chain` now. `@theepic.dev` clarified that these are being replaced by LCEL, demonstrating with code examples and referring to the [LCEL documentation](https://python.langchain.com/docs/expression_language/).
  
- **Troubleshooting ChatVertexAI Configurations**: User `@molnarbalazs` sought assistance with an issue regarding NEGLIGIBLE level Response Validation errors when using ChatVertexAI, looking for a way to turn off this response validation or adjust safety configurations.
  
- **Chroma Retrieval Issues**: `@theepic.dev` experienced performance issues with invoking Chroma's retriever and theorized that transforming questions into a list of keywords might yield better results.
  
- **Understanding LLM Parameters**: `@nrs9044` inquired about the meaning of parameters in large language models (LLMs), with `@anthology_` explaining that parameters are weightings, and tokens from a prompt use these parameters in the backend to formulate responses.

**Links mentioned**:

- [langchain | 🦜️🔗 Langchain](https://python.langchain.com/docs/changelog/langchain): 0.1.0 (Jan 5, 2024)
- [Chains | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/chains): Chains refer to sequences of calls - whether to an LLM, a tool, or a
- [LangChain Expression Language (LCEL) | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/): LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together.
- [ChatOllama | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/chat/ollama#via-langchain): Ollama allows you to run open-source large
- [community: Add SparkLLM Text Embedding Model and SparkLLM introduction by liugddx · Pull Request #17573 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/pull/17573): Thank you for contributing to LangChain! Checklist:   PR title: Please title your PR &quot;package: description&quot;, where &quot;package&quot; is whichever of langchain, community, core, experimenta...

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1209367237551718430) (1 messages): 

- **LangChain Tutorials Galore**: User `@mehulgupta7991` shared a comprehensive [LangChain tutorial playlist](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ) with over 60 videos. These tutorials are aimed at educating users on various **use cases of LangChain**, a framework for developing applications with generative AI.

**Links mentioned**:

[Langchain](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=_roiySFJ_TtEtrPn): This playlist includes all tutorials around LangChain, a framework for building generative AI applications using LLMs

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1209153677386846238) (2 messages): 

- **Exploring LangGraph Retrieval Agent**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=DFT0tMBwh04) titled "LangGraph Retrieval Agent," explaining the use of **Retrieval Agents** in deciding when to retrieve from an index, and demonstrating how to implement one by providing specific instructions.

- **Diving into LangChain with a Tutorial Series**: `@mehulgupta7991` highlighted a comprehensive [playlist](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=_roiySFJ_TtEtrPn) with over 60 tutorials focused on different **LangChain use cases**. The tutorials serve as a resource for learning how to build generative AI applications using large language models (LLMs).

**Links mentioned**:

- [Langchain](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=_roiySFJ_TtEtrPn): This playlist includes all tutorials around LangChain, a framework for building generative AI applications using LLMs
- [LangGraph Retrieval Agent](https://www.youtube.com/watch?v=DFT0tMBwh04): Retrieval Agents are useful when we want to make decisions about whether to retrieve from an index.To implement a retrieval agent, we simple need to give an ...

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1209191784761135114) (5 messages): 

- **Mixing Language and Expertise in LLMs**: `@johannhartmann` expressed interest in an LLM with multiple experts for different languages and domains, pondering how to prompt accordingly to guide languages to the right models.
- **LLM Pretraining Resources & Language Model Discussions**: `@johannhartmann` mentioned [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a pretraining framework for various models like LLaMA and BLOOM, suggesting it could be used for pretraining LLMs in new languages and seeking experiences from those who pretrained top German models.
- **Expert Recommendations for LLM Training on Large-Scale Clusters**: `@bjoernp` recommended using GPT-Neo-X, Megatron-LM, or variants such as epfl and deepspeed for large-scale cluster pretraining (exceeding 128 GPUs), also mentioning [Axolotl](https://github.com/gretelai/axolotl), which supports sample packing and could approach the efficiency of more complex methods.
- **Acknowledgment of Large-Scale Pretraining Advice**: `@remek1972` confirmed they are training on a large-scale cluster and thanked `@bjoernp` for the helpful advice.
- **A Cry for Help?**: `@phantine` shared a distressing message stating they are "Trapped in mental hospital", with no further context provided.

**Links mentioned**:

[GitHub - hiyouga/LLaMA-Factory: Easy-to-use LLM fine-tuning framework (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM)](https://github.com/hiyouga/LLaMA-Factory): Easy-to-use LLM fine-tuning framework (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM) - hiyouga/LLaMA-Factory

  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1209097298756243466) (3 messages): 

- **Seeking Budget-Friendly Benchmarks**: User `@yobibyte` inquired about **free benchmarks** that might be similar to the **openllm leaderboard** for those with limited GPU resources.
- **Fasteval as an Alternative**: Despite the request for free options, `@johannhartmann` mentioned using **fasteval with mt-bench(-de)** for quick and less resource-intensive model evaluations.
- **Real Costs of Fasteval Clarified**: `@bjoernp` pointed out that even with the suggested solution, **fasteval** incurs costs of more than **5€ in OpenAI credits per model**, challenging the notion of it as a free benchmarking tool.
  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1209195806331248640) (4 messages): 

- **DiscoResearch Demo Downtime Explained**: `@maxmaier_` asked whether the demo server on https://demo.discoresearch.org was down just for them or for everyone. `@_jp1_` confirmed that it was down because the GPUs were used for evaluations and promised to bring it back up as soon as possible.
- **Server Back in Business**: Following up, `@_jp1_` notified that the demo server should be operational again, indicating the downtime was temporary and the issue resolved. `@maxmaier_` expressed their gratitude for the quick fix.

**Links mentioned**:

[DiscoLM German 7b Demo](https://demo.discoresearch.org): no description found

  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1209275143826972792) (1 messages): 

- **AI Enthusiasts Create Collaborative Master Doc**: User `@_red.j` shared a **Google Document** titled *The AI Info Diet ™* created during a Twitter space meeting with ML experts, intended as a resource for people to keep up with the **latest tools, news, & information in AI**. The document is available for anyone to contribute their favorite sources and [_red.j](https://docs.google.com/document/d/1jeVpCc-uxYbxrN9oDmCZRlV3pnzQcpdyOiVD1xBJ9-M/edit?usp=sharing) has also added the Alignment Lab AI Discord server to the list.

**Links mentioned**:

[The AI Info Diet ™️](https://docs.google.com/document/d/1jeVpCc-uxYbxrN9oDmCZRlV3pnzQcpdyOiVD1xBJ9-M/edit?usp=sharing): no description found

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=DFT0tMBwh04
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (1 messages): 

jeffreyw128: how do you access it? i can't for the life of me figure it out in the console lol
  

---



---




---
id: 851dc9c9-9602-4a29-ba5a-d534bbf64bd1
title: The Era of 1-bit LLMs
date: '2024-03-01T22:33:03.450029Z'
original_slug: ainews-the-era-of-1-bit-llms
description: >-
  The news discusses advancements in 1-bit large language models (LLMs),
  including the introduction of BitNet b1.58, which uses ternary weights and
  offers high performance with reduced energy and memory costs. It highlights
  research on quantization, multimodal models, AI security, and ethical
  considerations, along with community discussions on AI innovation, societal
  impact, and humor.
companies:
  - bitnet
  - gpt-4
  - openai
  - anthropic
  - google
  - huggingface
models:
  - bitnet-b1.58
  - gpt-4
  - gpt-3
  - gpt-3.5
  - gpt-3-turbo
topics:
  - quantization
  - large-language-models
  - multimodal
  - ai-security
  - ai-ethics
  - ai-research
  - ai-hardware
  - energy-efficient-ai
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 2/29/2024-3/1/2024. We checked [**356** Twitters](https://twitter.com/i/lists/1585430245762441216) and **22** Discords (**351** channels, and **6023** messages) for you. Estimated reading time saved (at 200wpm): **577 minutes**.

The most extreme form of Quantization is Binarization - chopping off all but 1 bit of the weights. TheBloke currently cuts it down to 4 bits but the loss in performance is dramatic. Usually.

[The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) paper has been catching quite some attention on [HN](https://news.ycombinator.com/item?id=39535800) and the Discords. The abstract is worth parsing carefully (with commentary from swyx):

- **Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs).** (the [BitNet paper](https://arxiv.org/abs/2310.11453) shows how to use a binary BitLinear function as a drop-in replacement of conventional matrix multiplication in order to train 1-bit weights from scratch with 38x energy cost reduction and competitive performance)
- In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which **every single parameter (or weight) of the LLM is ternary {-1, 0, 1}**. 
- It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance, while being **significantly more cost-effective in terms of latency, memory, throughput, and energy consumption**. 
- More profoundly, **the 1.58-bit LLM defines a new scaling law and recipe for training new generations of LLMs that are both high-performance and cost-effective**. Furthermore, it enables a new computation paradigm and opens the door for designing specific hardware optimized for 1-bit LLMs.

We would normally do a fuller parse of the paper but have to go do this dylan patel show. More in Latent Space's writeup this weekend.


---

**Table of Contents**

We are experimenting with removing Table of Contents as many people reported it wasn't as helpful as hoped. Let us know if you miss the TOCs, or they'll be gone permanently.


# PART X: AI Twitter Summary

### AI and Machine Learning Innovations

- **AGI and Big Tech Overlords:** Discussion on AGI's future societal impact is a prominent theme, with concerns over a small group gaining significant power. [levelsio discusses this potential future](https://twitter.com/levelsio/status/1763282668874010813).
- **Robotics and Multimodal Models:** Robotics is seeing advances with expanded multimodal models, emphasizing the future of humanoid robots and AI interaction in real-world scenarios. [gdb talks about expanding multimodal models for robotics](https://twitter.com/gdb/status/1763296690738720859), while [npew highlights collaboration with Figure\_robot team](https://twitter.com/npew/status/1763282241558573556).
- **Large Language Models (LLMs):** Novel methodologies like ResLoRA for fine-tuning LLMs indicate the continuous evolution of model efficiency and effectiveness. [\_akhaliq presents ResLoRAIdentity Residual Mapping in Low-Rank Adaption](https://twitter.com/_akhaliq/status/1763328999621529946).
- **Improving AI Security:** The discussion on model repository security underscores the importance of safe persistence methods and malware scanning to prevent malicious use of LLMs. [osanseviero shares efforts on model security at huggingface](https://twitter.com/osanseviero/status/1763331704146583806).
- **AI in Daily Use and Developer Tools:** There's a focus on AI's role in enhancing daily convenience and the development of tools that streamline AI research and application. [levelsio observes modern conveniences in developing countries](https://twitter.com/levelsio/status/1763298959169097755), [AravSrinivas talks about diverse podcast topics including AI](https://twitter.com/AravSrinivas/status/1763285261071650856).

### AI Research and Ethics

- **Ethical Considerations in AI:** The role of ethics in generative AI development is being examined, reflecting the community's cognizance of AI's societal implications. [mmitchell\_ai shares an Op-Ed on ethics in generative AI](https://twitter.com/mmitchell_ai/status/1763284604696629252).
- **Research Impact and Recognition:** The conversation around research impact underscores the need for humility and constructive engagement in the academic community. [deliprao speaks on researcher reactions](https://twitter.com/deliprao/status/1763298115195408757).

### Memes/Humor

- **Humor in the AI Community:** Jokes and memes within the AI engineering community provide levity and commentary on the quirks of the field. [nearcyan humorously discusses timing in business decisions](https://twitter.com/nearcyan/status/1763318561072713865), [ClementDelangue makes light of conversations with Stripe employees](https://twitter.com/ClementDelangue/status/1763350424050868547), and [deliprao vents frustration with chatbots](https://twitter.com/deliprao/status/1763341490246336764).


### Overall Summary

The discourse within the AI and technical engineering communities, as reflected through Twitter conversations, spans from profound concerns over the societal impact of AGI to detailed discussions on specific AI models and optimization techniques. The debate around the future economic landscape with AGI ([@levelsio](https://twitter.com/levelsio/status/1763282668874010813)) represents a significant concern about tech's expanding influence. Simultaneously, the dialogue on multimodal models and robotics ([@gdb](https://twitter.com/gdb/status/1763296690738720859)) reflects an enthusiasm for integrating AI with real-world applications.

There's a notable emphasis on enhancing efficiency and refining AI methodologies, with ResLoRA being discussed as an innovation in fine-tuning large language models ([@_akhaliq](https://twitter.com/_akhaliq/status/1763328999621529946)), while concerns over StackOverflow's future presence ([@fchollet](https://twitter.com/fchollet/status/1763306890161992143)) indicate the evolving landscape of developer resources in light of AI advancements. The curiosity towards model security and ethical AI showcases an industry prioritizing robust and responsible development ([@osanseviero](https://twitter.com/osanseviero/status/1763331704146583806)).

These discussions reflect the AI community's broad range of interests, from deep technical concerns to societal implications, indicating a diverse set of priorities and areas of interest among professionals and enthusiasts in the field.

---

# PART 0: Summary of Summaries of Summaries

<div><h2><strong>Anticipation for AI Goliaths and Model Development</strong>:</h2><ul><li>The <strong>Allen Institute for AI</strong> and <strong>OpenAI</strong> are at the forefront of AI advancements, with discussions around a <strong>65b AI model</strong> and <strong>GPT-6</strong>, hinting at future capabilities in AI technology. The community is eager about the potential of these models, comparing them to existing ones like <strong>Llama 3</strong> and speculating on their impact on AI research and applications​​​​.</li></ul><h2><strong>Legal and Ethical Debates</strong>:</h2><ul><li>Elon Musk's legal actions against OpenAI have sparked a debate regarding the organization's commitment to open AI technology. This controversy underscores the growing concerns over the ethics and governance of major AI entities, highlighting the complex relationship between innovation, ownership, and open-source principles​​.</li></ul><h2><strong>Technological Innovations and Challenges</strong>:</h2><ul><li>The <strong>Flipper Zero</strong> device and advancements in AI infrastructure, such as the <strong>Modular MAX Developer Edition</strong>, represent significant progress in hardware and tools for AI and hacking communities. These discussions reveal the continuous balancing act between innovation, regulation, and ethical hacking​​​​.</li></ul><h2><strong>Training and Quantization Techniques</strong>:</h2><ul><li>Deep technical discussions on training protocols, including the use of <strong>tinyllama</strong>, <strong>QLoRA</strong>, and <strong>quantization strategies</strong>, reflect the AI community's efforts to optimize AI model training and deployment. The exchange of scripts and articles for fine-tuning and deploying quantized models demonstrates a collaborative approach to overcoming technical challenges in AI model development​​​​.</li></ul><p>These themes indicate a vibrant ecosystem of developers, researchers, and enthusiasts engaged in pushing the boundaries of AI technology, grappling with its ethical implications, and exploring innovative applications and tools.</p></div>

---

# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Anticipation for AI Goliaths**: Significant buzz around the purportedly forthcoming **65b AI model** from the Allen Institute for AI stirred the community, evoking comparisons with OpenAI's LLMs and conjectures about **Llama 3**'s capabilities. The discussion included insights into the potential of such models and featured a [research link](https://arxiv.org/abs/2401.05566).

- **Musk's Legal Moves Stir Debate**: Elon Musk's [legal complaint](https://www.courthousenews.com/wp-content/uploads/2024/02/musk-v-altman-openai-complaint-sf.pdf) against OpenAI sparked controversy regarding the organization's commitment to open AI technology, with Musk's dissatisfaction illuminated through the filed court document.

- **Flipper Zero: A Hacker's Delight and Dilemma**: Conversations on Flipper Zero hardware focused on its uses in projects involving NFC and RFID, as well as challenges in debugging BLE problems. Users were also vocal about the device's price and perceived value, particularly post its ban and consequent price increase [Flipper Zero's product page](https://flipperzero.one/).

- **Training Scene Gets Quantized**: Technical exchanges delved into training protocols, including experiments with **tinyllama** using a **5e-6 learning rate**, effective use of **QLoRA**, and sequence strategies for training and quantizing models. A [Colab script](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) and a [Medium article](https://dsmonk.medium.com/training-and-deploying-of-quantized-llms-with-lora-and-gptq-part-2-2-ec7b54659c9e) were shared to aid in fine-tuning and deploying quantized models.

- **Roleplay Renders LLMs Livelier**: Intriguing dialogue about whether incorporating roleplay into LLMs could enhance their apparent intelligence, with observations suggesting that detailed characterization prompts could favor improved model predictions closely aligned with existing datasets. A practice highlighted by `@maldevide` for creating convincingly conversational characters.

- **Intricate Model Merging Methodologies**: The discussion in the model-merging channel touched on spherical linear interpolation (**slerp**) vs linear ties, diffusion and huggingface test methods, and an endorsement from `@alphaatlas1` advising on the use of **concatenation** over full weight merging while employing **PEFT**.

- **Cutting-edge Coding Collabs**: The announcement of the **Modular MAX Developer Edition** offers new possibilities for AI infrastructure, while the `semantic-chunking` package on [npm](https://www.npmjs.com/package/semantic-chunking) promises streamlined text processing for LLMs leveraging transformers.js and ONNX. Further discussions explored optimizing GPU utilization and potential performance enhancements using WebAssembly backends for ONNX.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **GPU Outshines CPU for Model Inference**: In the #deployment channel, it was emphasized that **GPU**, particularly with higher VRAM like that found in an **RTX 4090**, is crucial for running substantial models such as the full 7B Mistral. The discussions also touched on the limited performance of quantized models in larger contexts and the merits of specified-language models over multi-language ones.

- **Fine-Tuning Insight Remains Limited**: The #finetuning channel saw inquiries about the hours required to fine-tune a 7B model on H100, as well as speculation regarding the methods and datasets behind Mistral 7B instruct v0.2. However, detailed insights into the fine-tuning process of Mistral remain undisclosed.

- **Showcase Anticipation and Accessibility Queries**: Users in the #showcase and #random channels indicated a keen interest in upcoming project sneak peeks and how to access Google's 1M context AI. One user recommended a contact from *Deepmind* as a potential access point.

- **Uncertainties and Issues with Mistral API and AI Models**: In #la-plateforme, users clarified the absence of certain features in general, a model mismatch issue with the API, and validation errors suggesting a temporary inconsistency resolved with a fix in deployment.

- **Office Hours for Evaluation Strategies**: A single message from the #office-hour channel highlighted the upcoming discussion on **evaluation and benchmarking** scheduled for **March 5 at 5 pm CET**.

- **Suggesting Enhancements for Le Chat and CroissantLLM**: Participants in the #le-chat channel suggested various enhancements for **Le Chat**, while expressing dissatisfaction with **CroissantLLM**, hinting at potential improvements through finetuning.

- **Computational Resource Discussions Dominate**: Across multiple channels, conversations revolved around technical discussions related to computational resources such as VRAM requirements, the importance of GPUs over CPUs in inference, and hardware specifications like the efficacy of M2 and M3 Macs in computational tasks.

- **Prompts and Failures Offer Sparse Data**: The #failed-prompts and #prompts-gallery channels included messages alluding to failed prompts and model inaccuracies, yet lacked concrete data or examples that could be analyzed for meaningful AI development insights.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **GPT-3 Goes Invisible**: `@temparr` unexpectedly lost sight of their custom GPTs, but `@openheroes` quickly shed light on their location under the "mine" tab in the [OpenAI GPTs page](https://chat.openai.com/gpts/mine).
- **Real-world Beats Paper**: In the battle of **AI certifications vs. experience**, `@navs02`'s query was met with `@dezuzel` advocating for real-world AI savvy, and `.dooz` nodding towards Andrew Ng's courses and Andrej Karpathy's YouTube tutorials as the winning combo.
- **AI Sailors Navigate Spreadsheet Seas**: `@gatorsuf83` pondered using AI to chart boat data into spreadsheets, prompting `@.braydie` to hoist the sails with suggestions of CSV formats, and `@eskcanta` to reveal a treasure chest in the form of an [AI-generated Excel sheet](https://chat.openai.com/share/0581e083-d02a-4b9a-b02e-12b7997dc32f).
- **Upload Troubles on the Digital Seas**: A wave of complaints about file upload glitches was spotted on the horizon, with `@metaldrgn` and `@cqoker` among the navigators facing rough seas, igniting talks of usage caps and potential bugs in the system.
- **DALL-E 3's Prompting Puzzle**: Amidst diagramming discourse and character limit conundrums, `@madame_architect` hailed diagramming tools like Mermaid, while `@darthgustav` untangled a curly bracket parsing snag in DALL-E 3's JSON strings, and `@solbus` clarified foggy documentation on prompt character limits with a beacon from the [OpenAI Cookbook](https://cookbook.openai.com/articles/what_is_new_with_dalle_3).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Cosine Conundrum Cleared Up**: `@pseudoterminalx` clarified the confusion around **Cosine LR schedules**, emphasizing that many mistake the simpler version for the truly oscillating kind. They advocate for a more nuanced **Cosine schedule with decay**, akin to that which Pytorch categorizes as "cosine annealing with decay."

- **Ideogram Model - Mystery or Revolution?**: A newly released model, **Ideogram.ai**, designed by a former Google engineer piqued interest among members. Despite lacking substantial details, the community is abuzz with its potential, comparing it to other unreleased models, such as Google’s Imagen.

- **The Aesthetic AI Debate**: The guild discussed the balance between prompt adherence and aesthetic appeal in AI-generated images. `@devilismyfriend` pointed out that better aesthetics might sometimes require straying from precise prompt instructions.

- **Collaborative Captioning Initiative**: Techniques for captioning large image datasets were shared, including an offer from `@pseudoterminalx` to provide a volunteer cluster for the task. This underscores community efforts toward building high-quality captioned datasets.

- **Shared Wisdom on Model Training**: Guild members exchanged insights on model training and augmentation strategies, discussing topics from different resolutions to text incorporation using CLIP. There was talk of pooling text embeddings and adding them directly as register tokens during training.

**Key Resources Shared**:
  
- Paper discussing the use of registers in Vision Transformers, [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
- **kopyl**'s icon generation model available on [Hugging Face](https://huggingface.co/kopyl/ui-icons-256) and [Civitai](https://civitai.com/models/327499)
- RNN revival with [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427)



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **LM Studio Debates Preset Engagement and Forced Responses**: `@zorian_93363` feels the LM Studio presets are somewhat lacking and questions the efficacy of using system prompts to direct an AI's response. Meanwhile, debates over the simplicity and potential unwanted assumptions of a one-click model download feature within LM Studio arise, with `@fabguy` cautioning against any features that might limit user control.

- **Elevating Model Performance on Pi Devices with a Pinch of Google Coral**: The Google Coral accelerators are suggested as a means to enhance model execution on low-powered devices like the Raspberry Pi and Orange Pi 5, potentially bringing more firepower to compact form factors.

- **Hardware Headaches: Crashes, Coolers, and Configs**: Guild members tackle a slate of hardware issues from mysterious system crashes reported by `@666siegfried666` to searching for adequate cooling solutions and power supplies for high-spec systems featuring AMD Ryzen Threadripper Pro and multiple NVIDIA GPUs. Users also share aftermarket strategies to increase GPU utilization, like unlocking voltage control in MSI Afterburner.

- **The Constant Quest for Cognitive Comprehension**: Guild members exchange knowledge on local model recommendations for tasks like business document analysis and summarization, with a nod toward trending Huggingface models and specifics like Nous-Capybara-3B-V1.9 and MiniChat-2-3B. There's also a lighthearted comment on the diminishing returns of increased MoE counts in model performance.

- **From AI Gaming to Business Document Analysis**: A suggestion was made to transform AI interactions into a game or a TV show to entice AI to ask questions, and advice is sought on setting up a powerhouse PC optimized for business document analysis, though no specific model was recommended in the messages provided.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Perplexity Outshines ChatGPT**: In a comparison with ChatGPT, users shared insights on how Perplexity AI delivers more up-to-date information, akin to Google's offering. An article from IEEE spectrum was mentioned, elucidating Perplexity’s ambition to reinvent AI-powered search tools ([Perplexity-ai disrupts traditional chatbots](https://spectrum.ieee.org/perplexity-ai)).

- **AI Tools on Test Benches**: Community members evaluated the summarizing capabilities of Perplexity and Copilot, while also testing file upload and extraction features with a focus on output quality. Additionally, Copilot Pro's code interpreter feature was discussed, highlighting its availability to free users as well.

- **API Shenanigans and Teething Troubles**: Conversation in the API channel revealed challenges and confusion related to the use and documentation of Perplexity’s API, including model comparisons, performance issues, and the deprecation of `pplx-70b-online` slated for March 15. Users were directed to a [getting started guide](https://docs.perplexity.ai/docs/getting-started), as well as a [February 2024 API update changelog](https://docs.perplexity.ai/changelog/api-updates-february-2024).

- **Tacos and Tech Collide in AI Sharing**: In the sharing channel, playful and innovative uses of Perplexity AI were highlighted, including searching for the best taco recipe and generating podcast content. AI's prowess in creating portraits and audio content also featured, showcasing the platform's versatility and creative potential.

- **Contribution Call for Legacy Models**: Amidst updates and model changes, users beseeched not to phase out the favored `pplx-70b-online` model, debating its merits over newer ones such as `sonar-medium-online`. Shared experiences underscored the need for model stability and reliable performance.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Cheatsheet Launched for Budding AI Devs**: The **Foundation Model Development Cheatsheet** is now available, thanks to a collaborative endeavor led by `@hailey_schoelkopf`, with contributions from EleutherAI and various institutions. It provides a comprehensive guide for new open model developers, emphasizing parts of the process that are often overlooked, such as dataset documentation and licensing practices. The Cheatsheet comes in both a paper format (available [here](https://github.com/allenai/fm-cheatsheet/blob/main/app/resources/paper.pdf)) and as an interactive website (accessible [here](https://fmcheatsheet.org/)).

- **Understanding the Leaderboard Mystery**: Clarifications were made regarding the `mmlu_no_train` user presence on a leaderboard, which has been associated with automated downloads from lm-eval-harness rather than actual user engagement. Further discussion in the general channel pointed to resources such as a [blog post explaining multiple-choice normalization techniques](https://blog.eleuther.ai/multiple-choice-normalization/) and the potential to substitute model calls in the lm-evaluation-harness with custom code like TensorRT, as confirmed by `@slowturtle_p`.

- **Quantization's Role in Model Interpretability**: Speculations emerge about the interpretability of extremely quantized LLMs, possibly offering new insights because of simpler weights as discussed in a [recent paper](https://arxiv.org/pdf/2402.17764.pdf). Meanwhile, the difficulty of transformers to learn functions with high sensitivity to input alterations may inform biases towards low sensitivity functions and add to our burgeoning knowledge of these models' learning capabilities, as seen in [this paper](https://arxiv.org/abs/2402.09963).

- **Translations Impact LLM Performance**: `@marcobella` improved the Lambada dataset translations, which led to a significant 10%-15% increase in multilingual model accuracy, demonstrating the importance of high-quality translations on model performance. The revised translations are available on the [Hugging Face dataset page](https://huggingface.co/datasets/marcob/lambada_multilingual).

- **Deep Dive into GPT-NeoX and The Pile**: Infrastructure for **GPT-NeoX** needs manual setup and the validation set for **The Pile** was confirmed to be sampled uniformly at random, with deduplication performed before its creation. The details were clarified in response to questions about the sampling method and the creation of a canonical validation set, but specifics related to deduplication and the timing relative to **Pythia's** use of the dataset were not entirely clear.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **GPT-6 Gears Up for New Abilities**: A patent for **GPT-6** suggests potential advancements in agents and music generation. However, specific details of the patent were not shared in the discussion.

- **Fine-Tuning Tips for Gemma 7B**: A [video guide](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be) on how to fine-tune the **Gemma 7B** model using Unsloth, complete with a referenced Google Colab notebook, was shared among users for enhanced model performance.

- **1-bit Models Steal the Spotlight**: The [BitNet b1.58 paper](https://arxiv.org/abs/2402.17764) introduced a 1-bit Large Language Model with cost-effective performance, stirring conversations about hardware implementation due to its ternary and additive qualities.

- **Benchmarks Reveal Gemma Model Quirks**: Anomalies in Google's **Gemma** models' performance were highlighted, noting that the larger Gemma 8B model was underperforming when compared to the Gemma 2B on several benchmarks.

- **OpenAI Five Readies for Human Challenge**: [OpenAI's blog post](https://openai.com/research/openai-five-defeats-dota-2-world-champions) details the success of **OpenAI Five**, which has gone from beating Dota 2 bots to collaborating with human players, signaling an impending global exhibition to test its capabilities.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Groq's AI Hardware Innovations Captivate Audience**: At the [Web Summit in Qatar](https://youtu.be/IixoaS5ckBA), Groq CEO Jonathan Ross discussed the company's advancements in TPU and LPU technology, attracting attention for its potential impact on AI infrastructure.
- **Boundary-Pushing 1-Bit LLM Garners Mixed Reactions**: The [new "ternary parameters" paper](https://arxiv.org/abs/2402.17764) sparked debate on [Hacker News](https://news.ycombinator.com/item?id=39535800) with its claim of a 1-bit large language model matching the performance of full-precision LLMs, receiving skepticism about practicality and retraining needs.
- **Post-Mortem of Banana.dev's Serverless GPUs**: A [blog post](https://blog.erikdunteman.com/banana-pivot-unpeeled) detailing the rise and fall of Banana.dev's Serverless GPUs product stirred a conversation about the challenges facing AI startups, highlighting the complexity of the product-market fit in AI.
- **AI Product Management Discourse**: The guild traded notes on managing AI projects, where a Coursera specialization on AI Product Management from Duke University and a lecture from Fullstack Deep Learning on ML teams and project management were [recommended resources](https://www.coursera.org/specializations/ai-product-management-duke).
- **Representation Engineering Sessions Spark Inquisitiveness**: Discussions about **Representation Engineering** unveiled its foundational role in steerability and alignment, along with the planning of the LLM Asia Paper Club's schedule to accommodate members across time zones, and the positive reception to the idea of making the Representation Engineering Library work in a [Colab workbook](https://colab.research.google.com/).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Tuned Up Hybrid Search Unveiled**: LlamaIndex promotes an innovative approach using **Language Models (LLMs)** to fine-tune hybrid search efficacy by automatically adjusting the alpha parameter based on query types, shared via [Twitter](https://twitter.com/llama_index/status/1763252392639042024).

- **RAG Architecture Bridging Data Types**: The integration of structured data into the **Retrieval Augmented Generation (RAG)** framework is explored by LlamaIndex, with insights detailed in a [ClickHouseDB blog post](https://twitter.com/llama_index/status/1763282902358585445).

- **Webinar on Deploying Private RAG Local Systems**: LlamaIndex's CEO [@jerryjliu0](https://twitter.com/llama_index/status/1763304038442192978) announces a webinar showcasing the deployment of **local RAG systems** using LlamaIndex + Tonic Validate with Ollama, aiming to enhance data privacy.

- **Observability for LLM Apps through OpenLLMetry**: Future LlamaIndex webinar will feature techniques on implementing observability within LLM applications, emphasizing the need for detailed instrumentation, as per [this announcement](https://twitter.com/llama_index/status/1763364010676900080).

- **Anticipating the Future of Long-Context RAG Systems**: A [Twitter discussion](https://twitter.com/llama_index/status/1763620476847632744) by LlamaIndex speculates on the evolution of RAG systems in handling long-context models such as **Gemini 1.5 Pro**, hinting at adaptations in retrieval methodologies.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Axolotl Questionnaire Adjustments for User-Friendly Interface**: `@caseus_` updated their [**Axolotl End User Questionnaire**](https://docs.google.com/forms/d/e/1FAIpQLSeyJkTk7sCYWpCNfKNNpnlMQlT9XU2nt_TJCzP4GSZBT0vrRA/viewform) to require fewer mandatory fields after community feedback, aiming to gain insights on user interaction with axolotl.
- **TinyBox Unboxed for AI Prowess**: The [**TinyBox** system](https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production), equipped with six AMD Radeon RX 7900 XTX GPUs, was introduced by `@dreamgen`, highlighting its potential to deliver affordable PetaFLOPS-class performance for AI applications.
- **Innovation with Sophia and DropBP Algorithms**: [**Sophia optimizer**](https://arxiv.org/abs/2305.14342), a second-order optimizer, and [**DropBP**, a training time reduction approach](https://arxiv.org/abs/2402.17812), were shared for their efficiencies in model training, offering alternatives to traditional backpropagation and Adam optimization methods, respectively.
- **Starcoder2 Gains Community Footing**: Discussion and queries around **Starcoder2**'s integration and support were accompanied by [GitHub repository sharing](https://github.com/bigcode-project/starcoder2), underscoring interest in the emerging model's relevance and application.
- **Danish Mastery using Mistral**: `@le_mess` achieved comparable results to **ChatGPT 3.5** in Danish language tasks with a **7B Mistral model** through synthetic datasets, iterative model training, and [Scandeval.com](https://scandeval.com) benchmarks, emphasizing manual and automated curation processes for open-source commercial applications.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **WMMA Optimizes Tensor Ops on 7900xtx**: An [enablement of WMMA in MLIR/LLVM](https://github.com/joviliast/triton/commit/f2f21c1a17437d3cef42dc31c08c82491ce4b08b) has led to performance improvements for the **7900xtx**, with detailed metrics shared. `@iron_bound`'s success showcased the impact of precision formats on large matrix sizes.

- **Eradicating TypeError in Triton Debugging**: Setting the `TRITON_INTERPRET` environment variable resolves a `TypeError` when using the Triton debugger, as the keyword 'interpret' has been deprecated in **Triton 3.0.0 and 2.2.0**.

- **Ada Lovelace GPUs and FP8 Compute Limitations**: A conversation highlighted that although **FP8 intrinsics** are available, actual computations are limited on Ada Lovelace GPUs, with the lack of `wgmma.mma_async` being a notable shortfall. `@drisspg` referenced a [PyTorch discussion](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815) exploring these compute constraints.

- **Introducing BASED Architecture for Efficient Attention**: A new attention-based language model architecture named BASED, detailed in a [research paper](https://arxiv.org/pdf/2402.18668.pdf), was introduced – promising improved efficiency. Additionally, Hugging Face's Mistral implementation was noted to have questionable attention defaults, potentially problematic above 4k context, as evidenced by a [tweet](https://x.com/ph_singer/status/1763538607191527540?s=20) and the proposed [fix PR](https://github.com/huggingface/transformers/pull/29220).

- **Ring Attention Dynamics Cause Disarray**: Multiple issues plagued `@ericauld` and `@jamesmel` with ring attention, including incorrect gradients and pointer argument errors. A look into lucidrains' [repository history](https://github.com/lucidrains/ring-attention-pytorch/commits/main/) hinted at problematic custom kernel efforts, while GPU resource allocation conflicts were addressed by system reboots.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

**Serialization Hitch in LangChain**: `@thatdc` encountered an issue with **langserve** where only final outputs, not intermediate steps, were returned from their agent. An ongoing [GitHub issue #381](https://github.com/langchain-ai/langserve/issues/381) might have related info, but no definitive solution was provided.

**Curbing the CashGrab**: Multiple channels reported posts by `@skywalker09_` containing suspicious links promising a "$50 Gift", which may be a potential **scam**.

**Stocks Chatbot Using LangGraph**: User `@tarikkaoutar` demonstrated the integration of LangGraph with YahooFinance in a [YouTube video](https://www.youtube.com/watch?v=r2PvHdkaXWc&t=129s), creating a multi-agent stock analysis chatbot.

**Endoftext Streamlines Prompt Engineering**: `@cabreraalex` released **Endoftext**, an AI prompt editor offering suggestions and test cases, showcased in a [60-second demo](https://youtu.be/PGv5ymOhaNA) and available at [Endoftext's website](https://app.endoftext.app/).

**Data Integration via Airbyte and Langchain**: An article shared by `@andysingal` explains how Airbyte's combination with Langchain can improve data integration processes, further explored in a [Medium post](https://medium.com/ai-advances/airbyte-with-langchain-streamlining-data-integration-and-document-processing-8593db1fc3ad).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord Summary

- **Stripe Flags Prepaid, Not Virtual Cards**: `@fakeleiikun` encountered *error 402 or error 502* when using a prepaid card with Google Pay on OpenRouter; `@louisgv` mentioned that while Stripe Radar may flag cards like Discovery, virtual cards from supported banks work fine.
- **Helicone Meets OpenRouter**: `@wise_monkey_42910` sought assistance for integrating Helicone with OpenRouter; `@louisgv` provided an [integration example on GitHub](https://github.com/OpenRouterTeam/openrouter-examples/blob/main/examples/langchain/index.ts) and directed to the [Helicone documentation](https://docs.helicone.ai/getting-started/integration-method/openrouter).
- **Token Terminology Tidied Up**: In a discussion about streaming with function calling, `@alexatallah` explained that `native_tokens` represent tokens in the model's own tokenizer, and promised to update documentation to reflect that existing usage metrics are for native tokens.
- **Musk Myths Dispelled in OpenRouter Chat**: `@alexatallah` addressed speculation by `@telepathyx` about Elon Musk competing with OpenRouter, clarifying that Groq, rather than Grok, could be considered for future addition to OpenRouter, negating the idea of Musk's competition.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord Summary

- **Nate's Brush with AI Royalty**: In a random encounter, `@natolambert` met with AI luminary Yann LeCun, though he missed the opportunity to invite the renowned scientist to appear on the podcast.
- **Yann LeCun Shares Green Vision**: During the unexpected meeting, `@natolambert` engaged in a deep conversation with Yann LeCun about **green energy**, a topic of mutual interest.
- **Podcast Pep Talk**: After hearing about the encounter, `@philpax` cheerfully encouraged `@natolambert`, suggesting he'll ace his "charisma check" for future invitations.
- **Family Ties Tease**: The community bantered over a possible Lambert family connection, with proposals like adding a custom server emoji spurred by `@victory`, while `@mike.lambert` tried to play detective on familial links.
- **LeCun's Loneliness and RL Skepticism**: `@natolambert` shared insights from his talk with Yann LeCun — a personal sense of being alone in the push for open AI and his typical skepticism towards reinforcement learning, deemed as *normal yann stuff*.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Crafting High-Quality Negatives for DPR**: `@philipmay` recommended a strategy for improving Dense Passage Retrieval datasets by having a Language Model generate **intentionally incorrect answers**, which could yield more effective negatives for training purposes.

- **DiscoLM_German_7b Performance Quest**: `@mab3049` is hunting for the optimal settings for the **DiscoLM_German_7b model**, echoing challenges in replicating the demo’s performance outcomes.

- **Fine-Tuning Padding Dilemma**: A query was raised by `@silicagel_64242` about the suitable `pad_token` to use while fine-tuning models, with several tokens like `eos_token`, `unk_token`, and a specific `"[PAD]"` token being candidates, yet no consensus was reached.

- **In Search of German RAG Excellence**: `@bjoernwerner` is in the hunt for the most effective German embedding model for Retriever-Aggregator-Generator applications, exploring various **single and multi-vector embedding** options.

- **MT-Bench-X Sparks German Dataset Hunt**: The elusive MT-Bench-X dataset was under spotlight by `@crispstrobe`, who pointed to its Apache 2.0 license and potential for German language tasks according to a paper on [arxiv.org](https://arxiv.org/pdf/2402.13703v1.pdf); alternative suggestions like MT-Bench-DE and the manually-improved [MT-Bench-TrueGerman](https://huggingface.co/datasets/VAGOsolutions/MT-Bench-TrueGerman) were discussed as richer resources for genuine German language benchmarks.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Claude Ditches the Small Talk**: Users explored strategies to circumvent Claude's default conversational introductions by setting the initial characters it returns. [Anthropic's rewrite guidelines](https://docs.anthropic.com/claude/docs/ask-claude-for-rewrites) were referenced with one technique involving forcing Claude to start with a specific character like `<rewrite>` to bypass unnecessary lead-ins.

- **Local LLM Enthusiasm Meets Silence**: A request for recommendations of the best open source large language model (LLM) by `@gwthompson` that can be run locally and used with Datasette enrichment received no suggestions from the community.

- **The Silent Search for Clean C APIs**: `@florents_` inquired about the existence of LLMs with a clean C API to no avail, as no direct recommendations emerged from the conversation.

- **A Glimpse of LLM via llama.cpp**: `@agarcia_me` indicated the utility of llama.cpp for embedding support despite the need for a C++ compiler, mentioning an intention to release the code for a sqlite extension integrating LLM embeddings and highlighted the use of a C API.

- **Embedding Guidance with C Code Demonstration**: `@agarcia_me` shared a detailed C code snippet from [llama.cpp/examples](https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding/embedding.cpp) to show implementation of LLM embeddings, stressing that it was written in pure C, works with batch sizes of one, and pointed out that the `llama_batch` function encapsulates the core complexity.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Anthropic Said to Outdo Gemini 1.5**: Users discussed unverified claims that **Anthropic** outperforms **Gemini 1.5** in context length and accuracy. However, personal testing to confirm these rumors has not been conducted by the participants.
  
- **Quest for OpenAI Enhancements**: A member voiced a need for better **resources and information** related to OpenAI, specifically seeking advice on implementing OpenAI's **codeinterpreter** in a production setting.

- **The Enigma of System Prompts**: Discussants addressed the important but often opaque influence of **system prompts** on model outputs. It was highlighted that the effectiveness of prompts can be inconsistent due to model differences and frequent updates by research labs.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **UK AI Engineer Job Descriptions Sought**: `@peterg0093` is seeking examples of AI engineer job descriptions in the UK that conform to emerging standards. The Hex AI engineer careers page [Hex Careers](https://hex.tech/careers/ai-engineer/) was shared by `@swyxio` as a potential template showcasing company culture and role expectations.
  
- **AIEF Could Adopt AI-Infra's Structure**: `@swyxio` recommends that the AI Engineer Foundation might benefit from a structured approach similar to [AI-Infra](https://ai-infra.fun/) for organizing resources.

- **Event Organizer Recognized on LinkedIn**: Gratitude was expressed by `@huikang` for recognition on [LinkedIn](https://www.linkedin.com/posts/ai-eng-foundation_last-saturday-on-022424-sasha-organized-activity-7169152145336782850-_TsG) regarding their involvement in an event on 02/24/24.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Gemma 7B Gets a Finetuning Guide**: A [YouTube tutorial](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be) on finetuning **Gemma 7B** with Unsloth was shared, complete with an accessible [Colab notebook](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollTo...).
- **OpenCodeInterpreter Unveiled**: A [YouTube video](https://www.youtube.com/watch?v=cwJKopBBnWo&feature=youtu.be) introduces **OpenCodeInterpreter**, an open-source project for enhancing code generation with large language models.
- **In-Discord Talent Scouting**: @.papahh extended a job offer to a guild member, advising them to check their DM for more details.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **New Approach to Neural Network Interpretability**: `@camelfacts` introduced a paper that offers a **novel approach for interpreting neural network representations** by mapping out representational niches, leveraging concepts from economic and information theory. The paper, seeking feedback, has been shared on LessWrong with [this link](https://www.lesswrong.com/posts/7tSthxSgnNxbt4Hk6/what-s-in-the-box-towards-interpretability-by-distinguishing-1).



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1212680316624314378) (1182 messages🔥🔥🔥): 

- **Mixed Enthusiasm for Future AI Releases**: Users discussed the upcoming release of the alleged 65b AI model [by Allen Institute for AI](https://www.allenai.org/) and compared expectations with OpenAI's LLMs. Some speculated about how future models like Llama 3 may perform, with a [link to related materials](https://arxiv.org/abs/2401.05566).

- **Musk vs. OpenAI**: Conversations swirled around Elon Musk's legal actions against OpenAI, with a [filed complaint](https://www.courthousenews.com/wp-content/uploads/2024/02/musk-v-altman-openai-complaint-sf.pdf) suggesting Musk's discontent over OpenAI's operations, considering them against their commitment to open-sourcing AI technology.

- **Hacking & Hardware Discussions**: Netrve showed interest in the Flipper Zero for personal hardware projects and the multifunctional capabilities it offers, including NFC and RFID use.

- **Flipper Zero Product Chatter**: Users shared experiences with the Flipper Zero device, from debugging Bluetooth Low Energy (BLE) problems to nostalgic feelings invoked by its design. The [price of the Flipper Zero](https://flipperzero.one/) and its respective accessories were debated, alongside its apparent price increase post-ban.

- **Scripting for LLMs Chat**: Quantscope queried if anyone had experience coding scripts that leverage local LLMs, leading to discussions about personal projects and sharing resources like Hugging Face's library support.

**Links mentioned**:

- [no title found](https://www.amazon.ca/Rotring-500-Drafting-Pencil-0-5/dp/B001A1TN0G/ref=mp_s_a_1_5?crid=1KAN0GXZYPQ9F&dib=eyJ2IjoiMSJ9.ewJpEfa07fgtLJha4Np31X8Mo3wtqPvYsZmIMdwfI2McYeCH4TUyT_S5Nupclflsyu9iwCKyshKCCHTvWzpwWQHLtXweM6cOljB4YeRV8KTq3p1SWhDmByy8ts0N6-88ABZ5cxQp46WfiwRb2ikqKAiC8eHBogRUTpwS9a2fe2zBbGyOn3IenrxCUwbAT0XMB_kmh-IjxXTBqwqqNwJCPQ.ufZ3NQ-gq-88LR3hGsBagrP3kEc9xvQ1w-Uod9voRv0&dib_tag=se&keywords=rotring+500+0.5&qid=1709241957&sprefix=rotring+500%2Caps%2C135&sr=8-5)): no description found
- [EMO](https://humanaigc.github.io/emote-portrait-alive/): EMO: Emote Portrait Alive - Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions
- [Cerebras](https://www.cerebras.net/): Cerebras is the go-to platform for fast and effortless AI training. Learn more at www.cerebras.net.
- [Cat Cat Meme GIF - Cat Cat meme Funny cat - Discover &amp; Share GIFs](https://tenor.com/view/cat-cat-meme-funny-cat-cat-eating-cat-eating-chips-gif-10455465908695706650): Click to view the GIF
- [Sad GIF - Sad - Discover &amp; Share GIFs](https://tenor.com/view/sad-gif-7523306793289960933): Click to view the GIF
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566): Humans are capable of strategically deceptive behavior: behaving helpfully in most situations, but then behaving very differently in order to pursue alternative objectives when given the opportunity. ...
- [Futurama Bender GIF - Futurama Bender Dance - Discover &amp; Share GIFs](https://tenor.com/view/futurama-bender-dance-gif-4195777226086506084): Click to view the GIF
- [Thanos Perfectlybalanced GIF - Thanos Perfectlybalanced - Discover &amp; Share GIFs](https://tenor.com/view/thanos-perfectlybalanced-gif-18301221): Click to view the GIF
- [Netflix, hungry for more growth, signals more price hikes](https://arstechnica.com/gadgets/2024/01/netflix-hungry-for-more-growth-signals-more-price-hikes/): Basic ad-free plan being ripped from subscribers in Canada, UK first. 
- [How to enable Previous Versions to recover files on Windows 10 - Pureinfotech](https://pureinfotech.com/enable-previous-versions-recover-files-windows-10/): Windows 10 Previous Versions lets you restore files and folders using File Explorer, and in the guide, you&#039;ll how to configure it.
- [Product - Chip - Cerebras](https://www.cerebras.net/product-chip/): no description found
- [p1atdev/dart-v1-sft · Hugging Face](https://huggingface.co/p1atdev/dart-v1-sft): no description found
- [Tweet from Arthur Mensch (@arthurmensch)](https://x.com/arthurmensch/status/1762818733016322168?s=46&t=0D9nuNBS26GijH-DPnkpgw): Clarifying a couple of things since we’re reading creative interpretations of our latest announcements: - We’re still committed to leading open-weight models! We ask for a little patience, 1.5k H100s ...
- [adamo1139 (Adam)](https://huggingface.co/adamo1139): no description found
- [OpenAI&#39;s Statement SHOCK the Entire Industry! AI Riots vs &quot;Moore&#39;s Law for Everything&quot; by Sam Altman](https://youtu.be/JEFEJsTxPCc): Get on my daily AI newsletter 🔥https://natural20.beehiiv.com/subscribe[News, Research and Tutorials on AI]LINKS:Moore&#39;s Law for Everything:https://moores.sa...
- [Moore's Law for Everything](https://moores.samaltman.com/): We need to design a system that embraces this technological future and taxes the assets that will make up most of the value in that world–companies and land–in order to fairly distribute some of the c...
- [GitHub - facebookresearch/nougat: Implementation of Nougat Neural Optical Understanding for Academic Documents](https://github.com/facebookresearch/nougat): Implementation of Nougat Neural Optical Understanding for Academic Documents - facebookresearch/nougat
- [GitHub - vosen/ZLUDA: CUDA on AMD GPUs](https://github.com/vosen/ZLUDA): CUDA on AMD GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.
- [adamo1139/rawrr_v2 · Datasets at Hugging Face](https://huggingface.co/datasets/adamo1139/rawrr_v2): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1212689925678571551) (283 messages🔥🔥): 

- **Character Role Swap Delights**: `@lyrcaxis` brought up the idea of an in-character/out-of-character split to create more convincing roleplay setups, where the roleplayer communicates through the character rather than directly conveying thoughts.
- **Bots Silenced for Spam**: `@dampf` addressed `@mrdragonfox`'s inquiry about a tldr, informing that the caught spam messages were removed from the channel.
- **Roleplay Leading to Smarter LLMs?**: `@superking__` and others discussed whether making LLMs roleplay could help them appear smarter, with some experimentation suggesting roleplaying with instruct-driven prompts may yield better or more specific results.
- **Character Prompting Leads to Better Modeling**: `@maldevide` shared an extensive method of defining characters and using detailed prompts, which they believe positions the LLM to better predict subsequent dialogue by hinging closely to trained datasets.
- **Modest Midori Gets Modded**: `@c.gato` humorously discussed creating an LLM, and `@lisamacintosh` observed a curious transformation where an LLM named Midori started incorporating "vroom" into sentences after being depicted as a 2006 Honda Civic, showcasing the imaginative and sometimes unexpected outcomes within character modeling.

**Links mentioned**:

- [maldv/conversation-cixot · Datasets at Hugging Face](https://huggingface.co/datasets/maldv/conversation-cixot): no description found
- [Aeala (A&#39;eala)](https://huggingface.co/Aeala): no description found

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1212853189838249994) (31 messages🔥): 

- **Tinyllama Undergoes Learning Rate Experimentation**: `@maldevide` is conducting a test on **tinyllama** using a learning rate of `5e-6`, and plans to use **Supervised Fine-Tuning (SFT)** with all data to condition the model.

- **Training Tactics with QLoRA**: `@222gate` shared their method of using QLoRA and then merging the **LoRA** with the **4-bit quantized model**, specifically setting the adapter to "qlora" and the optimizer to "adamw_bnb_8bit".

- **Model Training and Quantization Puzzle**: `@orel1212` is curious about the correct sequence in training and quantizing models, prompting a discussion on whether to quantize before or after merging with base models. `@maldevide` mentions that **QLoRA's learned parameters** should be applied back to the base model before quantization.

- **Discovering Colab Resources**: `@orel1212` and `@222gate` share resources for training and fine-tuning quantized models with links to a [Colab script](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) and a [Medium article](https://dsmonk.medium.com/training-and-deploying-of-quantized-llms-with-lora-and-gptq-part-2-2-ec7b54659c9e).

- **Validation Set Dilemma in Model Pretraining**: `@cogbuji` inquires about the necessity of a validation set in unsupervised learning for pretraining models on raw text domain data, `@maldevide` clarifies that withholding a validation set prevents score bias but it's not mandatory, especially if the data is limited.

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1Xu0BrCB7IShwSWKVcfAfhehwjDrDMH5m#scrollTo=70zJf1hi0huQ): no description found
- [Finetune GPTQ model with peft and tlr](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996): Finetune GPTQ model with peft and tlr. GitHub Gist: instantly share code, notes, and snippets.
- [Fine-Tuning and Deploying of LLMs : PEFT and GPTQ! Part 2/2](https://dsmonk.medium.com/training-and-deploying-of-quantized-llms-with-lora-and-gptq-part-2-2-ec7b54659c9e): What if we start from a quantized model?

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1212826786367545485) (5 messages): 

- **Gratitude Expressed**: `@222gate` expressed thanks for shared information, though it’s unclear what specific content they were referring to.
- **Clarification on Slerp or Linear Ties**: `@222gate` inquired whether discussions were about **slerp (spherical linear interpolation)** or just linear ties.
- **Testing Methodologies in Discussion**: `@alphaatlas1` responded to @222gate's question clarifying that their diffusion test used **dare ties** and speculated that a huggingface test utilized **task arithmetic**.
- **Recommendation for PEFT Merging**: `@alphaatlas1` advised `@222gate` to try using **concatenation** ("**concat**") when doing PEFT merging, suggesting its effectiveness compared to full weight merging.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1212798213577248818) (6 messages): 

- **Modular MAX Platform's General Availability**: `@dirtytigerx` shared the announcement of [Modular's MAX Developer Edition](https://www.modular.com/blog/announcing-max-developer-edition-preview), which is founded on the vision to create a unified and efficient infrastructure for AI, making it usable and scalable for all developers. The platform promises to empower global developers and optimize AI hardware efficiency and total cost of ownership.

- **Semantic Chunking for JavaScript Launched**: `@jparkerweb` introduced a `semantic-chunking` package to efficiently break down large texts for LLM workflows without relying on heavier frameworks. The package is now available on [npm](https://www.npmjs.com/package/semantic-chunking), and it utilizes transformers.js and ONNX for operation.

- **GPU Optimization Discussion**: After `@jparkerweb` shared their `semantic-chunking` solution, `@dirtytigerx` mentioned a similar tool they developed using a node addon via Rust/Neon for better GPU utilization. They hinted that `@jparkerweb`'s package might also be enhanced with GPU support by setting the ONNX backend to `wasm`.

- **Exploring WebAssembly for Performance**: `@jparkerweb` expressed interest in exploring the WebAssembly (`wasm`) backend for ONNX as suggested by `@dirtytigerx` to potentially increase performance and efficiency in their `semantic-chunking` tool.

**Links mentioned**:

- [Modular: Announcing MAX Developer Edition Preview](https://www.modular.com/blog/announcing-max-developer-edition-preview): We are building a next-generation AI developer platform for the world. Check out our latest post: Announcing MAX Developer Edition Preview
- [semantic-chunking](https://www.npmjs.com/package/semantic-chunking): semantically create chunks from large text (useful for passing to LLM workflows). Latest version: 1.0.0, last published: a day ago. Start using semantic-chunking in your project by running `npm i sema...

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1212670228291522612) (724 messages🔥🔥🔥): 

- **VRAM Confusion Cleared Up**: `@mehdi1991_` had queries about the appropriate server for running models like Gemma 7B and VRAM requirements for a Python library, which users including `@ethux` helped clarify, explaining that VRAM refers to the GPU's memory, with a recommendation to use RTX 3090 for both large language models (LLMs) and libraries.

- **Potential Mistral AI App & Open Model Speculations**: `@pacificprime` asked about an upcoming Mistral app, while discussions by `@blacksummer99` and `@mrdragonfox` revolved around the speculative release timing of a new open-weight model by Mistral and its potential connection to Meta's launch of LLama3.

- **Discussions on Misuse of API and Protecting Against Fraud**: The worries expressed by `@foxalabs_32486` and `@chrunt` highlighted concerns around the misuse of APIs with stolen API keys, causing financial losses and potential security discussions that could influence company policies on API usage.

- **Microsoft Misinterpreted Alliance with Mistral**: There were concerns from `@kerunix` about the misconstrued nature of Microsoft and Mistral AI's relationship, magnified by media speculation. User `@lerela` corrected these misconceptions by pointing to official clarifications that downplayed the notion of an alliance.

- **Clone Model Discussions and Technical Specifications**: Various users including `@i_am_dom`, `@mrdragonfox`, and `@shaman6991` engaged in discussions regarding the intricacies of running clone models like Mixtral, intricacies of model switching, and the efficiency of retrieval-augmented generation (RAG) systems. Technical advice for concurrent query handling and suitable hardware was also shared by `@mrdragonfox`.

**Links mentioned**:

- [What Is Retrieval-Augmented Generation aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/): Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/#json-mode)): We provide client codes in both Python and Javascript.
- [NVIDIA Chat With RTX](https://www.nvidia.com/fr-fr/ai-on-rtx/chat-with-rtx-generative-ai/): Personnalisez et déployez votre chatbot d&#39;IA.
- [vLLM | Mistral AI Large Language Models](https://docs.mistral.ai/self-deployment/vllm/): vLLM can be deployed using a docker image we provide, or directly from the python package.
- [Klopp Retro GIF - Klopp Retro Dancing - Discover &amp; Share GIFs](https://tenor.com/view/klopp-retro-dancing-liverpool-champions-gif-19224858): Click to view the GIF
- [Microsoft made a $16M investment in Mistral AI | TechCrunch](https://techcrunch.com/2024/02/27/microsoft-made-a-16-): Microsoft is investing €15 million in Mistral AI, a Paris-based AI startup working on foundational models.
- [Microsoft made a $16M investment in Mistral AI | TechCrunch](https://techcrunch.com/2024/02/27/microsoft-made-a-16-million-investment-in-mistral-ai/amp/): Microsoft is investing €15 million in Mistral AI, a Paris-based AI startup working on foundational models.
- [openchat/openchat-3.5-0106 · Hugging Face](https://huggingface.co/openchat/openchat-3.5-0106): no description found
- [Legal terms and conditions](https://mistral.ai/terms/#terms-of-use): Terms and conditions for using Mistral products and services.
- [Pixels - Actualités, vidéos et infos en direct ](https://www.lemonde.fr/pixels/article/2024/03/01/on-a-teste-le-chat-l-et): Toute l’actualité sur le sujet Pixels. Consultez l’ensemble des articles, reportages, directs, photos et vidéos de la rubrique Pixels publiés par Le Monde.
- [Pixels - Actualités, vidéos et infos en direct ](https://www.lemonde.fr/pixels/article/2024/03/01/on-a-teste-le-chat-l-etonnant-chatgpt-a-la-francaise-de-mistral-ai_6219436_4408996.html),): Toute l’actualité sur le sujet Pixels. Consultez l’ensemble des articles, reportages, directs, photos et vidéos de la rubrique Pixels publiés par Le Monde.

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1212686973882335263) (14 messages🔥): 

- **Query Complexity for Mistral-7B**: `@sanipanwala` asked if Mistral-7B-v0.1 could handle complex SQL queries, providing an example involving `SELECT` statements with `INNER JOIN` and `OUTER APPLY`. `@tom_lrd` confirmed that Mistral models can attempt any query and provided an example structure to test performance.
- **Specific SQL Query Crafting**: In a follow-up, `@sanipanwala` inquired about customizing SQL queries to select specific fields from tables, and `@tom_lrd` demonstrated how to frame the request to Mistral by providing an intricate SQL query example.
- **Math Dataset Embedding Inquiry**: `@aky6691` queried the group about experiences with embedding math datasets, but without specifying the type of math or the intended use. `@tom_lrd` asked for clarification on what exactly was meant by "embedding of maths dataset."
- **Mixed Results with Image Prompting**: `@h0rizons` shared their experience that Mistral's large model doesn't perform as well as GPT-4 when creating prompts for AI image generators, noting an improvement when instructing it explicitly to not repeat art prompts.
- **Mistral Pricing Structure Discussion**: `@jb_5579` inquired about the API rates for Mistral Large and Next, which led `@mrdragonfox` to share a comprehensive link to Mistral pricing. This confirmed the costs for various models, including Mistral Large at $8 per 1M tokens for input and $24 per 1M tokens for output, as seen on [Mistral's pricing page](https://docs.mistral.ai/platform/pricing/).

**Links mentioned**:

[Pricing and rate limits | Mistral AI Large Language Models](https://docs.mistral.ai/platform/pricing/): Pay-as-you-go

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1212670570878214174) (153 messages🔥🔥): 

- **GPU Over CPU for Inference**: `@ethux` mentioned that for model inference, **GPU** plays a crucial role while **CPU** is important only to some extent. They also discussed the significance of GPU memory, indicating that to run the full 7B Mistral model, a **RTX 4090** with its VRAM is required.
  
- **Quantized Model Shortcomings**: In the conversations between `@frigjord`, `@_._pandora_._`, and `@ethux`, they expressed dissatisfaction with the performance of quantized models, especially with larger contexts and coding tasks, confirming that quantized models are inferior to the full versions.
  
- **Debating the Merits of Specified-Language Models**: `@frigjord` and `@mrdragonfox` debated whether creating models focused on specific programming languages like JS might outperform multi-language models. `@mrdragonfox` suggested that variety in training languages can lead to better generalization.

- **Affordability and Efficiency in Hardware**: Users `@frigjord` and `@sublimatorniq` discussed the costs and benefits associated with using high-spec hardware like the M2 and M3 Macs. `@frigjord` pointed out the speed advantage on an M2, while `@sublimatorniq` shared his experience with a 96GB setup.

- **Data Cleaning Prevails as a Significant Challenge**: In a lengthy discussion, `@mrdragonfox` underscored the extensive effort required for data cleaning, which constitutes a vast majority of the work in data preparation for model training, and shared his personal hardware setup used to tackle such tasks.

**Links mentioned**:

- [starling-lm](https://ollama.com/library/starling-lm): Starling is a large language model trained by reinforcement learning from AI feedback focused on improving chatbot helpfulness.
- [Jurassic Park GIF - Jurassic Park World - Discover &amp; Share GIFs](https://tenor.com/view/jurassic-park-world-velociraptor-clever-gif-25116052): Click to view the GIF
- [Tags · mixtral](https://ollama.com/library/mixtral/tags): A high-quality Mixture of Experts (MoE) model with open weights by Mistral AI.

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1212733431931346974) (18 messages🔥): 

- **Inquiring Fine-Tuning Hours**: `@atip` asked about the hours required to fully fine-tune a 7B model on H100, with `@mrdragonfox` responding that it varies based on dataset size.
- **Seeking Fine-Tuning Secrets of Mistral 7B**: `@pteromaple` sought details on the fine-tuning methods and datasets for Mistral 7B instruct v0.2, but `@mrdragonfox` confirmed that this information is not disclosed and speculated on the quality measurements and data preparation involved.
- **Unlocking Fine-Tuning Mysteries**: `@pteromaple` and `@mrdragonfox` discussed the technicalities of the significant improvement from Mistral 7B instruct v0.1 to v0.2 and compared its performance with Google's Gemma 7B IT.
- **Fine-Tuning via API**: `@claidler` inquired about the possibility of fine-tuning closed models through an API, and `@ethux` pointed out a clue in the API response indicating a future goal for model fine-tuning.
- **Top of the Class Finetuned Model Query**: `@kunpengguo` asked whether `mixtral-8x7b-instruct-v0.1` is the best Mistral finetuned model, with `@mrdragonfox` affirming its status.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1212823146282098788) (2 messages): 

- **Curiosity for Upcoming Projects**: `@akshay_1` expressed interest in seeing a preview of a project that `@patagonia50` plans to work on, asking for a **sneak peek** when possible.
  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1212767292195078204) (2 messages): 

- **Accessing Google's 1M Context AI**: User `@j673912` inquired about how to access Google's 1M context AI. `@dawn.dusk` recommended having contact with someone from *Deepmind*.
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1212685615989002250) (21 messages🔥): 

- **Confusion on Chat Availability**: `@paul.martrenchar_pro` clarifies that a feature **is not available** in general and it's solely present on **Le Chat** at the moment.

- **Mistral API Model Mismatch Query**: `@ls_rageux_` expresses confusion regarding the API returning **open-mixtral-8x7b** when **mistral-small** is requested, appearing to reveal a discrepancy in model handling.

- **Mistral System Role Support Clarification**: `@lerela` confirms that prefixing prompts with the **system/assistant** or **assistant** roles before the user is not supported in **Mistral Large**.

- **System Role in Mistral Workarounds**: `@zerosignal_x` inquires about system/assistant pairings in models like medium, while `@not__cool` and `@skisquaw` discuss alternative methods such as using the **system role** within the user prompt in **Mistral Large**.

- **Clarifications on Mistral's Tool Calls and Responses**: `@januify` seeks clarification on the absence of **`tool_calls` in the response body** when making requests to **Mistral Large**. `@lerela` explains that using a tool is at the model's discretion even with `tool_choice: "any"` and requests a more detailed example to investigate.

- **ValidationError and Python Client Mismatch in Mistral's API**: `@proffessorblue` reports a ValidationError related to **ChatCompletionResponse**, potentially indicating a temporary inconsistency between the Mistral API and the Python client. `@lerela` acknowledges a **brief deployment inconsistency**, which has been fixed, prompting `@proffessorblue` to further note the need for an updated API specification document.
  

---


### Mistral ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/1212716795262402570) (1 messages): 

- **Mark Your Calendars for Evaluation Strategies**: `@sophiamyang` announced that the next office hour on **March 5 at 5 pm CET** will focus on **evaluation and benchmarking**. The team is eager to discuss various methods of evaluation and benchmarking with participants.
  

---


### Mistral ▷ #[le-chat](https://discord.com/channels/1144547040454508606/1211692704363323453/1212691647226454027) (212 messages🔥🔥): 

- **Seeking Mistral's Touch in CroissantLLM**: `@tom_lrd` expressed disappointment with **CroissantLLM**, feeling it lacks **Mistral's** capabilities. They suggest finetuning with a French-English Hermes dataset could be beneficial but remain uncertain about the potential improvement.

- **Unsloth's Speedy & Efficient Finetuning**: `@foxalabs_32486` shared the [Unsloth repository](https://github.com/unslothai/unsloth), highlighting its claims of 5x faster finetuning with 60% less memory, while `_._pandora_._`, `@sublimatorniq`, and `@foxalabs_32486` discuss whether the performance improvements are as significant as advertised.

- **Le-Chat Enhancement Suggestions Roll In**: Multiple suggestions to improve **Le Chat** included **real-time token counts**, **design tweaks**, **image input**, and **"NEW CHAT" button adjustments**. `@sophiamyang` invited feedback, leading to a robust discussion with contributions from users like `@_._pandora_._`, `@foxalabs_32486`, and `@sublimatorniq`.

- **Using Mistral's Output for Fine-tuning Datasets**: `@dedded___` inquired about the feasibility of using **Mistral Large** for creating a dataset, with `@mrdragonfox` clarifying that while smaller datasets are an option, competing with large models would be a gargantuan task.

- **Clarity on Mistral AI Model Updates**: In a back-and-forth about **Mistral models**, `@lifeverygoode` sought to confirm whether the model `78x7` would remain open source, with `@ethux` affirming its open-source status and mentioning the release of new models rather than updates to existing ones.

**Links mentioned**:

- [Endpoints and benchmarks | Mistral AI Large Language Models](https://docs.mistral.ai/platform/endpoints/#benchmarks-results): We provide five different API endpoints to serve our generative models with different price/performance tradeoffs and one embedding endpoint for our embedding model.
- [Why 2024 Will Be Not Like 2024](https://medium.com/@unravelingentertainment/why-2024-will-be-not-like-2024-8799121ee791): In the ever-evolving landscape of technology and education, a revolutionary force is poised to reshape the way we learn, think, and…
- [Unsloth update: Mistral support + more](https://unsloth.ai/blog/mistral-benchmark#Benchmark%20tables): We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...
- [GitHub - jondurbin/airoboros: Customizable implementation of the self-instruct paper.](https://github.com/jondurbin/airoboros): Customizable implementation of the self-instruct paper. - jondurbin/airoboros
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.

  

---


### Mistral ▷ #[failed-prompts](https://discord.com/channels/1144547040454508606/1212715819159654451/1212715907898671166) (10 messages🔥): 

- **Unclear Definition of Failure**: `@notan_ai` commented ambiguously, hinting at a potential failed prompt but did not provide specific information on the failure scenario.
- **The Halfway Math Mystery**: `@blueaquilae` mentioned a math-related failure with a humorous note, "math, halfway there (pun intended) on large chat," but did not provide details of the prompt or the failure.
- **Prompt Failure Confirmation Eludes Us**: `@blacksummer99` referred to *Mistral next on le chat* as failing on a particular prompt, yet no details of the prompt, expected output, or model output were given.
- **Date Discrepancy Detected**: `@aiwaldoh` raised a concern about an inconsistency regarding the founding year of an unnamed entity, querying if "Fondée en 2016?!" while suggesting it might be related to a particular website.
- **Webpage and Founding Year**: `@aiwaldoh` added that although a webpage was mentioned, which cites 2023, the issue seems unresolved without further context.
- **Dedication to Discovery**: `@_._pandora_._` recognized the dedication of `@aiwaldoh` in searching for the origin of the discrepancy, commending their effort, to which `@aiwaldoh` replied affirmatively.
  

---


### Mistral ▷ #[prompts-gallery](https://discord.com/channels/1144547040454508606/1212717054302625843/1212717273610063902) (5 messages): 

- **Prompt Sharing Space Announced**: `@sophiamyang` has initiated the **prompts-gallery** channel inviting members to share their best prompts using a specific format outlining the model, prompt, and output.

- **Unclear Message Posted**: `@akshay_1` simply posted "DSPy" which lacks context and does not follow the channel's prompt sharing format.

- **Curiosity About SudoLang**: `@notan_ai` expressed interest in "SudoLang" but seemed confused about the purpose of the channel.

- **Unformatted Prompt Contribution Attempts**: `@blacksummer99` twice attempted to submit a prompt titled "Mistral next le chat," but did not provide the required details such as model, prompt, and output.
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1212743318388088852) (44 messages🔥): 

- **GPTs Gone Missing**: `@temparr` reported that all of their custom GPTs have disappeared. `@openheroes` promptly guided them to find their GPTs under the "mine" tab on the [OpenAI GPTs page](https://chat.openai.com/gpts/mine).

- **AI Certification vs. Real-world Experience**: Young developer `@navs02` inquired about AI certifications and `@dezuzel` responded by emphasizing the importance of real-world AI examples over certifications, while `.dooz` highlighted free courses by **Andrew Ng** and YouTube tutorials by **Andrej Karpathy** for hands-on learning and CV enhancement.

- **Reporting a Bounty-Hunting Bug**: User `@l0k1_b24` inquired about reporting an exploit and earning a bounty. `@solbus` referred them to [OpenAI's security information](https://chat.openai.com/.well-known/security.txt) and `@aminelg` reminded them to read the full description of the bug bounty program before reporting.

- **Lexideck Professional Pulls the CSS Strings**: `@beanz_and_rice` complimented `@darthgustav.` on their dexideck & website, to which they credited Lexideck Professional for the creation and clarified that related GitHub accounts might not actually represent them.

- **Discussing the Reality of AI**: In a philosophical turn, `@drinkoblog.weebly.com` questioned the reality of artificial creations, which sparked a discussion about the definition of "real" and "artificial", with inputs from `@aminelg`, and a reference to synthetic bacteria by `@eskcanta` linking back to a [The Guardian article](https://www.theguardian.com/science/2019/may/15/cambridge-scientists-create-worlds-first-living-organism-with-fully-redesigned-dna). 

- **AI and Spreadsheet Collaboration**: `@gatorsuf83` enquired about using AI to organize boat data into a spreadsheet, with `@.braydie` suggesting CSV or markdown tables as an approach and providing strategies to guide GPT efficiently.`@eskcanta` shared a successful test with a ready-to-go Excel download using AI: [AI-generated Excel sheet](https://chat.openai.com/share/0581e083-d02a-4b9a-b02e-12b7997dc32f).

**Links mentioned**:

[World’s first living organism with fully redesigned DNA created](https://www.theguardian.com/science/2019/may/15/cambridge-scientists-create-worlds-first-living-organism-with-fully-redesigned-dna): Researchers create altered synthetic genome, in move with potential medical benefits

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1212706826852892743) (78 messages🔥🔥): 

- **Confusion Over Knowledge File Handling and Web Browsing in GPT**: Users, including `@darthgustav.` and `@yami1010`, debated whether GPT can "read" large knowledge files and whether using python for searches disables web browsing. `@yami1010` shared [screenshots](https://cdn.discordapp.com/attachments/) suggesting misleading behavior regarding web search capabilities, prompting discussions on AI transparency.

- **OpenAI Discord Faces Upload Issues**: Multiple users, with `@metaldrgn` and `@cqoker` sharing specific experiences, reported issues with uploading files, particularly image and data files, leading to error messages and intermittent upload success. This raised concerns about potential usage caps and led to suggestions that there's a broader bug affecting the upload functionality.

- **Misunderstandings Concerning File Upload Limits**: There was confusion around file upload caps, highlighted by `@cqoker` and `@darthgustav.`, with a focus on whether usage caps pertain to total uploads or if they specifically relate to GPT knowledge file uploads. This caused a back-and-forth discussion attempting to clarify the restrictions and their applicability.

- **Annual Usage Caps Discussed but Not Clarified**: `@cqoker` and `@darthgustav.` engaged in a discussion regarding potential 10GB usage caps, but were unable to determine if this referred to lifetime, daily, or another timeframe, leading to further uncertainty about the policy. 

- **Concerns Over Transparency and Model Updates**: Conversations among users, including `@darthgustav.` and `@cqoker`, reflected concerns about the lack of clear documentation and understanding of the current state due to constant model updates, and how this affects user experience with the GPT's abilities and limitations related to file uploads and other functionalities.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1212701829348200538) (108 messages🔥🔥): 

- **Seeking Assistance for a Mystery List**: `@remi1054` inquired about a list they were curious to find; `@madame_architect` offered to upload the most recent version to the aiempower GitHub after her morning coffee routine.
- **Diagramming as Code Discussion**: `@madame_architect` shared her passion for diagramming as code, mentioning that tools like Mermaid, Mathplotlib, and PlantUML revolutionize her workflow in creating diagrams.
- **DALL-E 3 Parser Wrinkle Identified**: `@darthgustav` humorously recounted resolving a junior developer error in DALL-E 3's parser, which was failing to interpret standard curly brackets in JSON strings.
- **Fine-Tuning Chatbot Response Strategies**: In a detailed discussion, users, including `@beanz_and_rice`, `@darthgustav`, and `.braydie`, discussed strategies for enhancing the creativity and effectiveness of prompts for ChatGPT and DALL-E, with strategies like invoking the "Creative Process" or using "chain-of-thought" reasoning.
- **Documentation Discrepancies for DALL-E Cause Frustration**: `@solbus` referenced the correct character limit for prompts in DALL-E-3 after `@darthgustav` grappled with outdated documentation, which led to a moment of frustration regarding inconsistent reference materials.

**Links mentioned**:

[What&#x27;s new with DALL·E-3? | OpenAI Cookbook](https://cookbook.openai.com/articles/what_is_new_with_dalle_3): no description found

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1212701829348200538) (108 messages🔥🔥): 

- **Awaiting GitHub Repository Update**: `@remi1054` was inquiring about a list's location and `@madame_architect` committed to uploading the latest update to the AIempower GitHub repository after morning coffee.
- **Diagramming with ChatGPT Assistance**: `@madame_architect` shared the use of Mermaid, mathplotlib, and PlantUML for diagramming as code, while **Darthgustav.** noted a testing session that unearthed a parsing error with the DALL-E 3 JSON strings related to escape characters.
- **Navigating Large Documents with AI**: `@enkai3526` sought advice on how to prompt ChatGPT to effectively search for relevant text in large documents, which sparked discussion on using code interpreters and keyword-based searching methods.
- **Prompting ChatGPT for Sequential Question Answering**: `@etheraland1way` asked for help on creating a prompt that could process a long text followed by questions, with `@eskcanta` providing a simple solution by placing each question right after the text for structured responses.
- **DALL-E 3 Prompt Clarifications**: There was a discussion about DALL-E 3's prompt character limit led by **Darthgustav.**, with `@beanz_and_rice` initially challenging the misunderstood limit, which was later clarified by `@solbus` using the official OpenAI documentation.

**Links mentioned**:

[What&#x27;s new with DALL·E-3? | OpenAI Cookbook](https://cookbook.openai.com/articles/what_is_new_with_dalle_3): no description found

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1212737859447160842) (295 messages🔥🔥): 

- **Cosine LR Schedule Controversy**: `@pseudoterminalx` expressed frustration that when most people talk about a "Cosine LR schedule", they're referring to what he considers a simplified version that doesn't truly oscillate and lacks control over intervals between peaks. He distinguished his own Cosine schedule method which does oscillate and allows for that control, a method Pytorch labels as "cosine annealing with decay."
  
- **Ideogram Model Sparks Curiosity**: The chat showed interest in Ideogram.ai, a new model released by an ex-Google engineer made known by `@pseudoterminalx`. The model promises a departure from existing architectures, but details are scant, leading to speculation about its efficacy and the quality of similar unreleased models such as Google's Imagen.

- **Caption Quality in AI Discussed**: Users like `@pseudoterminalx` and `@thejonasbrothers` debated the differences between AI models in following prompts and creating aesthetically pleasing images. The discussion included an observation by `@devilismyfriend` suggesting that maintaining aesthetics often means not adhering strictly to prompts.

- **Collaborative Efforts for Captioning Large Image Sets**: `@pseudoterminalx` and `@thejonasbrothers` engaged in a conversation about techniques for creating datasets with high-quality captions, with `@pseudoterminalx` offering access to a volunteer cluster for captioning large image sets.

- **Ruminations on Model Training and Augmentation**: Various members, including `@thejonasbrothers`, `@pseudoterminalx`, and `@chad_in_the_house`, exchanged tips and techniques on model training strategies, the use of augmentations, training with different resolutions, and incorporating text from tools like CLIP into training. There was a mention of pooling text embeddings and adding them as register tokens, as well as discussions on how to best utilize the limited CLIP tokens during model training.

**Links mentioned**:

- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588): Transformers have recently emerged as a powerful tool for learning visual representations. In this paper, we identify and characterize artifacts in feature maps of both supervised and self-supervised ...
- [panopstor/nvflickritw-cogvlm-captions · Datasets at Hugging Face](https://huggingface.co/datasets/panopstor/nvflickritw-cogvlm-captions): no description found
- [ptx0/photo-concept-bucket · Datasets at Hugging Face](https://huggingface.co/datasets/ptx0/photo-concept-bucket): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1212736552933789766) (23 messages🔥): 

- **Icon Generation Model Unveiled**: User `@kopyl` revealed a new **State-of-the-Art (SOTA) model** for generating icons, invested $2000 in training it, and is offering it openly after monetization attempts failed. The icon model, along with usage instructions and a call for collaboration, can be found at [Diffusers by Kopyl](https://huggingface.co/kopyl/ui-icons-256).

- **RNN Spotlight**: `@twoabove` stirred nostalgia and attention with a link to a paper reviving **Recurrent Neural Networks (RNNs)**, discussing a new linear recurrence architecture [available on arXiv](https://arxiv.org/abs/2402.19427).

- **On the Merit of Simplicity in Model Inputs**: In a conversation about image formats for model training data, `@nodja` stated the advantage of using simple BMP format **to avoid the complexity of decoding**, which would waste computational resources.

- **Contrastive Learning in Model Distillation**: `@jh0482` sought information on papers exploring **distillation learning in language models**, questioning the use of contrastive learning when the target is a continuous space.

- **RNN Resurgence Banter**: With the mention of **RNNs**, `@thejonasbrothers` quipped about their devotion to the architecture, humorously personifying anticipation for their "recurrent messiah."

**Links mentioned**:

- [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427): Recurrent neural networks (RNNs) have fast inference and scale efficiently on long sequences, but they are difficult to train and hard to scale. We propose Hawk, an RNN with gated linear recurrences, ...
- [I Can See It Timothee Chalamet GIF - I Can See It Timothee Chalamet Paul Atreides - Discover &amp; Share GIFs](https://tenor.com/view/i-can-see-it-timothee-chalamet-paul-atreides-dune-i-can-visualize-it-gif-18400807): Click to view the GIF
- [kopyl/ui-icons-256 · Hugging Face](https://huggingface.co/kopyl/ui-icons-256): no description found
- [UI icons - v1.0 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/327499): SOTA model for generating icons. Motivation: I spent $2000 of my own money to train this model. I was unable to monetize it, so I&#x27;m sharing it with...

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1212670791691542538) (151 messages🔥🔥): 

- **LM Studio Presets and Forced Responses**: `@zorian_93363` finds the LM Studio presets somewhat empty and ponders over the use of a system prompt to force the Assistant response to begin with a certain string.
- **Running Models on Low-Powered Devices**: `@zorian_93363` responds to `@n3kosenpai`, a full-stack blockchain developer, suggesting that Google Coral accelerators are compatible with systems like Raspberry Pi and could potentiate models on devices like the Orange Pi 5.
- **Ultra-Fast AI Chat Bots Spark Interest**: `@pierrunoyt` shares a link to Groq's ultra-fast AI chatbot (broken link not included), while `@nullt3r` finds it pricey at 20k EUR from [Mouser](https://eu.mouser.com/ProductDetail/BittWare/RS-GQ-GC1-0109?qs=ST9lo4GX8V2eGrFMeVQmFw%3D%3D), and notes it has only 230MB of "RAM".
- **Model Execution Issues in LM Studio**: `@barnley` reports network errors in LM Studio when trying to download models from Huggingface or use search options, and `@heyitsyorkie` suggests checking for potential internet access issues such as work restrictions or blocks by ISPs or countries.
- **Replacing OpenAI API with LM Studio in Applications**: `@veryvanya` seeks guidance replacing the OpenAI key with an LM Studio server in an application's config and `@heyitsyorkie` provides an example on how to set the `base_url` for the OpenAI client to point at a local LM Studio server, advising to set `api_key` to `"not-needed"`.

**Links mentioned**:

- [no title found](http://192:168:0:100:1234/v1",): no description found
- [GroqChat](https://groq.com/): no description found
- [Continue](https://continue.dev/): no description found
- [MaziyarPanahi/dolphin-2.6-mistral-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF · Hugging Face](https://huggingface.co/MaziyarPanahi/dolphin-2.6-mistral-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF): no description found
- [mradermacher/miquliz-120b-v2.0-i1-GGUF · Hugging Face](https://huggingface.co/mradermacher/miquliz-120b-v2.0-i1-GGUF): no description found
- [vGPU Nvidia Tesla M10 32 GB RAM GDDR5 PCIe 3.0 x16 64 virtuelle Benutzer  | eBay](https://www.ebay.de/itm/126344098433): no description found
- [NVIDIA Tesla K80 Dual GPU 24GB PCI-E Computing Accelerator - 699-22080-0200-511](https://www.gekko-computer.de/NVIDIA-Tesla-K80-Dual-GPU-24GB-PCI-E-Computing-Accelerator-699-22080-0200-511.html): Es handelt sich um Gebrauchtware, welche von unserem Technikerteam getestet wurde. Sie ist technisch und optisch in einem einwandfreien Zustand.
- [nVidia Tesla M60 Dual GPU 16GB PCI-E Computing Accelerator - 900-2G402-0010-000](https://www.gekko-computer.de/nVidia-Tesla-M60-Dual-GPU-16GB-PCI-E-Computing-Accelerator-900-2G402-0010-000.html): Es handelt sich um Gebrauchtware, welche von unserem Technikerteam getestet wurde. Sie ist technisch und optisch in einem einwandfreien Zustand.
- [Add support for StarCoder2 by pacman100 · Pull Request #5795 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5795): What does this PR do?  Adds support for StarCoder 2 models that were released recently.

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1212749005117984789) (13 messages🔥): 

- **Engaging AI with Questions**: User `@pwrreset` outlined two strategies to have an AI ask questions: guiding users through prompts or turning interactions into a game or a TV show format.
- **Powerhouse PC Seeks AI for Business Docs**: User `@redcloud9999` asked for the best setup to analyze and write business documents with their high-spec machine (14900k, 192GB RAM, 1x 4090, Windows 11). No specific model recommended in the messages provided.
- **Local Model Recommendations**: User `@heyitsyorkie` suggested to `@redcloud9999` to download and test LLMs, recommending searching for GGUF quants by "TheBloke". `@coachdennis.` also advised looking at trending models on Huggingface for the latest and suitable choices.
- **Pinpointing Summarization Solutions**: User `@tay2win` sought recommendations for datasets and models adept at summarization for a short-term memory system, initially using phi-2 but finding it unsatisfactory. User `@drawless111` recommended various models to try, including Nous-Capybara-3B-V1.9 and MiniChat-2-3B, and advised lowering the temperature setting for smaller models to improve results.
- **Mixing Up Modeling Expertise**: User `@goldensun3ds` inquired why increasing the Mixture of Experts count often does not enhance model performance, though expecting it should. The question went largely unaddressed except for a light-hearted metaphorical suggestion from `@tay2win`: too many cooks in the kitchen.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1212866535366983720) (127 messages🔥🔥): 

- **Troubleshooting Tricky Hardware Crashes**: `@666siegfried666` reports frequent system crashes without leaving any error logs or dumps. They discuss various potential causes such as issues with RAM, Wi-Fi card (`AX200 - The network adapter has returned an invalid value to the driver`), and even PSU cabling. Despite extensive hardware testing, including Memtest86+ and PSU voltage measurements, the exact cause remains elusive. User `@wolfspyre` suggests booting into Linux as a diagnostic step to determine if it's a hardware or driver issue.

- **Boosting GPU Utilization with MSI Afterburner**: `@goldensun3ds` experiences a significant improvement in GPU utilization for a Dolphin 2 6 Mixtral 7B Q3KM model after unlocking voltage control in MSI Afterburner, achieving about 15 tokens per second with dual RTX 4060 Ti GPUs.

- **Potential High-End System Build for LLMs**: User `@razdnk` proposes a system build for language model work, which includes an ASUS Pro WS WRX90E-SAGE SE motherboard, AMD Ryzen Threadripper Pro 7965WX, and multiple NVIDIA 3090 GPUs. They seek advice for CPU coolers, cases, and power supplies that support such a high-end setup.

- **Open Rack Builds for Heat Management**: `@nink1` and `@heyitsyorkie` discuss challenges with air-cooling for multi-GPU setups and recommend using 1x risers and open rack builds or server racks to manage heat. `@heyitsyorkie` provides a [link](https://www.emilwallner.com/p/ml-rig) to Emil Wallner's ML rig as a worthwhile reference for assembling high-performance ML hardware.

- **Graphic Card Combinations and Overclocking Discussions**: Users share experiences and inquiries about different graphics card configurations. `@wilsonkeebs` asks about running NVIDIA 4090 with 3090, and `@ben.com` and `@heyitsyorkie` advise against watercooling ML rigs due to the complexity and maintenance it introduces.

**Links mentioned**:

- [James Bond 007 GIF - James Bond 007 Voodoo - Discover &amp; Share GIFs](https://tenor.com/view/james-bond-007-voodoo-must-be-the-gif-13955810): Click to view the GIF
- [Warhammer40k Angron GIF - Warhammer40k Angron Primarch - Discover &amp; Share GIFs](https://tenor.com/view/warhammer40k-angron-primarch-angry-ultra-angry-gif-17080819): Click to view the GIF
- [Helluva Boss Helluva GIF - Helluva Boss Helluva Loona Helluva Boss - Discover &amp; Share GIFs](https://tenor.com/view/helluva-boss-helluva-loona-helluva-boss-blitzo-eat-this-gif-25976859): Click to view the GIF
- [(4k) RTX 3090*4! It is a Luxury in Dreams](https://m.youtube.com/watch?v=fdtAOPyZ9z8): This computer first wanted to install air -cooled heat dissipation.Later, because the original graphics card was too thick and could not be installed, it was...
- [How I built a €25K Machine Learning Rig](https://www.emilwallner.com/p/ml-rig): How to plan, buy, build, and store your 2-10 GPU machine learning servers and PCs
- [NVIDIA Tesla K80 Dual GPU 24GB PCI-E Computing Accelerator - 699-22080-0200-511](https://www.gekko-computer.de/NVIDIA-Tesla-K80-Dual-GPU-24GB-PCI-E-Computing-Accelerator-699-22080-0200-511.html): Es handelt sich um Gebrauchtware, welche von unserem Technikerteam getestet wurde. Sie ist technisch und optisch in einem einwandfreien Zustand.
- [Amazon.com: StarTech.com PCI Express X1 to X16 Low Profile Slot Extension Adapter - PCIe x1 to x16 Adapter (PEX1TO162) : Electronics](https://www.amazon.com/gp/aw/d/B0039XPS5W/): no description found
- [Pro WS WRX90E-SAGE SE｜Motherboards｜ASUS Global](https://www.asus.com/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/): ASUS Workstation motherboards are designed for professionals in AI training, deep learning, animation, or 3D rendering. Featuring expandable graphics, storage, impressive connectivity and reliability,...

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1212808339847651398) (7 messages): 

- **Vision Model Download Clarification**: `@hypocritipus` inquired about the possibility of downloading Llava-supported models, including the vision adapter, within LM Studio in a future update. They shared a [link to the available models](https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1) and the [demo video](https://x.com/LMStudioAI/status/1734640355318944190?s=20) from the release notes.
  
- **Double Click to Download Models**: `@jedd1` responded to `@hypocritipus` explaining that currently, the process to download the vision adapter and the primary model requires two separate actions and there is no one-click solution within LM Studio.

- **Doubts Against One-Click Model Downloads**: `@fabguy` commented on the complexity of providing multiple options for repositories and expressed concerns that a one-click download feature could cause LM Studio to make unwanted assumptions, possibly obscuring choices from the users.

**Links mentioned**:

- [Vision Models (GGUF) - a lmstudio-ai Collection](https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1): no description found
- [Tweet from LM Studio (@LMStudioAI)](https://x.com/LMStudioAI/status/1734640355318944190?s=20): Counting penguins can be challenging 🧐🐧  New in LM Studio 0.2.9:   🎉 Local & Offline Vision Models!  In this demo: the small and impressive Obsidian Vision 3B by @NousResearch.

  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/) (1 messages): 

1sbefore: Yeah I agree that's not so common not to have conf in a .py only used for that
  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1212707066167296030) (166 messages🔥🔥): 

- **Perplexity AI vs. ChatGPT**: In response to a question by `@marshmodem`, `@bartleby0` explained that Perplexity is more like Google in providing up-to-date information as opposed to ChatGPT, sharing an article link for deeper insight.
- **Perplexity and Copilot Summaries Compared**: `@jaicraft` tested summarizing capabilities with both Perplexity and Copilot, concluding that both services provide satisfactory results, though Copilot may require more prompting for longer summaries.
- **File Upload Testing on Pro Versions**: `@jaicraft` and `@dailyfocus_daily` discussed file upload and summary extraction tests on Perplexity Pro and Copilot Pro, exploring the quality and efficiency of the outputs.
- **Code Interpreter Integration**: `@dailyfocus_daily` and `@jaicraft` conversed about Copilot Pro's code interpreter feature, which is also available to free users.
- **AI Search Engine Exposure**: `@.nohler` shared an article from [IEEE spectrum](https://spectrum.ieee.org/perplexity-ai) regarding Perplexity AI's approach compared to traditional chatbots like ChatGPT, highlighting Perplexity’s intent to create AI-powered search tools.

**Links mentioned**:

- [Summarize this PDF](https://copilot.microsoft.com/sl/jD0bsv95Qd2): Here's an answer I got using Microsoft Copilot, the world's first AI-powered answer engine. Select to see the full answer or try it yourself. 
- [Summarize this PDF](https://copilot.microsoft.com/sl/iGdav3ZxB4S): Here's an answer I got using Microsoft Copilot, the world's first AI-powered answer engine. Select to see the full answer or try it yourself. 
- [Perplexity.ai Turns Tables on Google, Upends SEO Credos](https://spectrum.ieee.org/perplexity-ai): AI search leader mixes Meta-built smarts with scrappy startup fervor
- [Retrieval Augmented Generation Research: 2017-2024](https://scalingknowledge.substack.com/p/rag): RAG literature review including: REPLUG, Fusion-in-Decoder, KNN-LM, RETRO, FLARE, HyDe, SILO, WebGPT, Toolformer, Self-RAG, GRIT &amp; more
- [Perplexity - AI Companion](https://chrome.google.com/webstore/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo): Ask anything while you browse
- [Perplexity - AI Search](https://chrome.google.com/webstore/detail/perplexity-ai-search/bnaffjbjpgiagpondjlnneblepbdchol): Upgrade your default search engine

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1212675530273464330) (9 messages🔥): 

- **Taco Tuesday Made Techy**: `@bonhart5` shared a [Perplexity AI query](https://www.perplexity.ai/search/Best-taco-recipe-rA8ta05ATtabuvA91gKLVg) for the **best taco recipe**.
- **Podcast Innovation with AI**: `@_paradroid` posted a [48 Hours of AI podcast prompt and result](https://www.perplexity.ai/search/You-will-act-hEljiMC4SqWMacvlhk4Njw), showing how AI can generate podcast content.
- **AI Portrait Creation**: `@dailyfocus_daily` linked to an [EMO Emote Portrait search](https://www.perplexity.ai/search/EMO-Emote-Portrait-oCaUsYWbS2CFU_Qr8ig5SQ), generated using AI.
- **Driving with AI Audio Content**: `@_paradroid` discussed the quality of AI-generated audio, mentioning how the combination of Perplexity and ElevenLabs content is **as enjoyable as a podcast** during drives. The related audio content can be found in the [Community-Projects](https://www.perplexity.ai/search/Fully-review-Richard-2GzIaJMnTw6Lo5WZFaAhDQ).
- **AI Explores Current Events**: The topic of **wildfires** was explored with AI on Perplexity, as indicated by `@_paradroid`'s [shared link](https://www.perplexity.ai/search/TOPIC-The-wildfires-MAUe_J7rRYaEdeIxdSbA.Q).
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1212673291576287242) (51 messages🔥): 

- **Confusion Over Subscriptions and Using API Only**: User `@monish0612` queried about how to subscribe only to the API instead of the Pro subscription. `@mares1317` provided a comprehensive [guide on getting started with the Perplexity API](https://docs.perplexity.ai/docs/getting-started), which includes registering a credit card and generating an API key.

- **Model Comparisons and Availability Concerns**: `@icelavaman` informed that `pplx-70b-online` will be deprecated on March 15, prompting a debate over `sonar-medium-online` versus `pplx-70b-online` models. Users like `@thedigitalcat` and `@lazysucker` favored `pplx-70b-online` for its quality, leading to discussions on model performance and requests for not phasing it out prematurely.

- **Issues with `sonar-medium-online` Performance**: Multiple users, including `@tob1724` and `@brknclock1215`, reported strange behaviors with the `sonar-medium-online` model, such as incomplete responses and lack of temporal awareness. Users shared different experiences and tried various system prompts to mitigate the issues.

- **Requests and Clarifications on API Documentation**: Users like `@jeffworthington` faced issues with OpenAPI definitions and `@tom_primozic` sought alternatives to backward engineering the website's WebSocket protocol. `@jeffworthington` and `@yury.zem.` also encountered challenges with API key authentication and free credits availability.

- **Latest Model Updates and User Feedback**: `@mares1317` shared a [change log link](https://docs.perplexity.ai/changelog/api-updates-february-2024) detailing the phasing out of certain models and introducing new models like `sonar-small-chat`. Feedback continued with users discussing the merits and reliability of the `pplx-70b-online` and `sonar` models in terms of accuracy and up-to-dateness.

**Links mentioned**:

- [Getting Started with pplx-api](https://docs.perplexity.ai/docs/getting-started): You can access pplx-api using HTTPS requests. Authenticating involves the following steps:Start by visiting the Perplexity API Settings page. Register your credit card to get started. This step will n...
- [Tweet from Phi Hoang (@apostraphi)](https://x.com/apostraphi/status/1762870577444847964?s=20): Ship it, because it&#39;s good enough, then make it better.
- [API Updates February 2024](https://docs.perplexity.ai/changelog/api-updates-february-2024): Announcing Our Newest ModelWe are excited to announce the launch of our latest Perplexity models: sonar-small-chat and sonar-medium-chat, along with their search-enhanced versions, sonar-small-online ...
- [More than an OpenAI Wrapper: Perplexity Pivots to Open Source](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/): Perplexity CEO Aravind Srinivas is a big Larry Page fan. However, he thinks he&#039;s found a way to compete not only with Google search, but with OpenAI&#039;s GPT too.

  

---



### Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1212781444720627783) (1 messages): 

- **Launch of the Foundation Model Development Cheatsheet**: `@hailey_schoelkopf` announced the **Foundation Model Development Cheatsheet**, a guide created for new open model developers, with contributions from individuals at EleutherAI and various institutions. The Cheatsheet aims to assist developers through the entire open model development pipeline, with attention to underdiscussed areas like dataset documentation and licensing practices.
- **Cheatsheet as a Response to the Growth of Open Models**: The resource was created following the significant increase in new models with open weights, highlighted by the release of the Pythia model suite and other projects like LLM360's Amber and AI2's OLMo. The initiative is designed to foster more entry points into the field of open model development.
- **Access the Foundation Model Development Cheatsheet**: The comprehensive resource is available to read in [paper format](https://github.com/allenai/fm-cheatsheet/blob/main/app/resources/paper.pdf) and can be explored as an [interactive website](https://fmcheatsheet.org/). Additional insights can be gained from the accompanying [blog post](https://blog.eleuther.ai/fm-dev-cheatsheet/) and [Twitter thread](https://twitter.com/AiEleuther/status/1763219826602901518).
  

---


### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1212679303720992768) (70 messages🔥🔥): 

- **Mamba Sequence Classification Inquiry**: `@_michaelsh` asked if there is a pretrained mamba model for sequence classification, and `@frazermc` clarified that although it likely doesn't exist, it is possible to train a classification head on top of a pretrained checkpoint.
- **Automated Downloads Confuse Rankings**: `@ad8e` and `@ilovescience` clarified that the user `mmlu_no_train` on a leaderboard appears to be associated with automated downloads from lm-eval-harness, not actual user engagement.
- **Harness Evaluation Methodology Query**: `@slowturtle_p` asked about the calculation of normalized accuracy scores in the lm-evaluation-harness, with `@stellaathena` pointing them to a [detailed blog post on multiple-choice normalization](https://blog.eleuther.ai/multiple-choice-normalization/).
- **Harness Custom Code Substitution**: `@maya_liv` inquired about substituting model calls with custom code in the lm-evaluation-harness, which `@slowturtle_p` confirmed is viable based on their personal experience with TensorRT.
- **LLM Pretraining Loss Spike Bingo**: `@staticpunch` faced an abnormal loss spike during LLM pretraining and got comprehensive feedback from `@lucaslingle`, `@cubic27`, and others suggesting it could be due to factors like an unsuitably high learning rate or issues with data loader resumption, with various optimization strategies like changing random seeds or using different optimizers like Lion.

**Links mentioned**:

- [Multiple Choice Normalization in LM Evaluation](https://blog.eleuther.ai/multiple-choice-normalization/): There are multiple ways of evaluating multiple choice tasks on autoregressive LMs like GPT-3/Neo/J. This post lays out the current prevalent normalization methods.
- [A Large Batch Optimizer Reality Check: Traditional, Generic...](https://openreview.net/forum?id=E9e18Ms5TeV): Recently the LARS and LAMB optimizers have been proposed for training neural networks faster using large batch sizes. LARS and LAMB add layer-wise normalization to the update rules of Heavy-ball...
- [Oogway Master Oogway GIF - Oogway Master Oogway Kung Fu Panda - Discover &amp; Share GIFs](https://tenor.com/view/oogway-master-oogway-kung-fu-panda-gif-26485559): Click to view the GIF
- [How does Groq LPU work? (w/ Head of Silicon Igor Arsovski!)](https://www.youtube.com/watch?v=WQDMKTEgQnY): Become a Patreon: https://www.patreon.com/theaiepiphany👨‍👩‍👧‍👦 Join our Discord community: https://discord.gg/peBrCpheKEI invited head of silicon at Groq...
- [Issues · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/978)): A framework for few-shot evaluation of language models. - Issues · EleutherAI/lm-evaluation-harness

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1212694151859019796) (77 messages🔥🔥): 

- **Model Collapse Mystery**: `@kramakek` reported a 70B LLM collapsing during continuous pretraining without a corresponding spike in training loss, showing benchmark metric drops and text artifacts. Discussion suggested potential causes, like catastrophic forgetting, while `@kramakek` clarified a 1.5e-5 learning rate, 10x less than its pretraining rate.

- **Foundation Model Debate Intensifies**: `@aaron_wtr` questioned the appropriateness of the term "foundation model" in biology, setting off a discussion about the term's ambiguity and legal implications. `@valiant` speculated that "foundation model" might become a legal term, while `@xylthixlm` clarified that "dual-use" in executive orders means potential military applications.

- **ResLoRA Enhances LoRA**: `@jckwind` brought attention to ResLoRA, an enhanced low-rank adaptation framework improving the training and inference efficiency of large language models. Some community members questioned the need for ResLoRA, with `@power_cookie` uncertain about the backward path length issue, while `@xylthixlm` justified gradient flow improvements near skip connections.

- **The Next Wave of Efficient Models**: `@trre` introduced BitNet b1.58, a new 1-bit ternary weight LLM, which claims to match full-precision LLMs in performance but with higher cost-effectiveness. `@honolouloute` shared a paper related to emerging models like Hyena and Mamba, while `@random_string_of_character` discussed papers on activation sparsity and the release of final checkpoints by PowerInfer researchers.

- **Ruminations on Digital World Simulation**: `@fairy8767` relayed the concept of bGPT, a model designed for next byte prediction to simulate digital operations, claiming to match specialized models across text, audio, and images. In a lecture quote shared by `@jckwind`, Geoffrey Hinton reflected on neural activity timescales in the brain, stimulating ideas about implementing variable learning rates in models for short-term memory, but `@thooton_` noted the lack of recurrence in transformers not supporting such agent-based architecture.

**Links mentioned**:

- [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427): Recurrent neural networks (RNNs) have fast inference and scale efficiently on long sequences, but they are difficult to train and hard to scale. We propose Hawk, an RNN with gated linear recurrences, ...
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750): Efficiently serving large language models (LLMs) requires batching many requests together to reduce the cost per request. Yet, the key-value (KV) cache, which stores attention keys and values to avoid...
- [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516): Activation sparsity refers to the existence of considerable weakly-contributed elements among activation outputs. As a prevalent property of the models using the ReLU activation function, it has been ...
- [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668): Recent work has shown that attention-based language models excel at recall, the ability to ground generations in tokens previously seen in context. However, the efficiency of attention-based models is...
- [ReLU$^2$ Wins: Discovering Efficient Activation Functions for Sparse LLMs](https://arxiv.org/abs/2402.03804): Sparse computation offers a compelling solution for the inference of Large Language Models (LLMs) in low-resource scenarios by dynamically skipping the computation of inactive neurons. While tradition...
- [RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval](http://arxiv.org/abs/2402.18510): This paper investigates the gap in representation powers of Recurrent Neural Networks (RNNs) and Transformers in the context of solving algorithmic problems. We focus on understanding whether RNNs, kn...
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [Trajectory Consistency Distillation](http://arxiv.org/abs/2402.19159): Latent Consistency Model (LCM) extends the Consistency Model to the latent space and leverages the guided consistency distillation technique to achieve impressive performance in accelerating text-to-i...
- [LeoLM: Igniting German-Language LLM Research | LAION](https://laion.ai/blog/leo-lm/.): &lt;p&gt;We proudly introduce LeoLM (&lt;strong&gt;L&lt;/strong&gt;inguistically &lt;strong&gt;E&lt;/strong&gt;nhanced &lt;strong&gt;O&lt;/strong&gt;pen &lt;strong&gt;L&lt;/strong&gt;anguage &lt;stron...
- [Beyond Language Models: Byte Models are Digital World Simulators](https://arxiv.org/abs/2402.19155): Traditional deep learning often overlooks bytes, the basic units of the digital world, where all forms of information and operations are encoded and manipulated in binary format. Inspired by the succe...
- [Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion](http://arxiv.org/abs/2310.02279): Consistency Models (CM) (Song et al., 2023) accelerate score-based diffusion model sampling at the cost of sample quality but lack a natural way to trade-off quality for speed. To address this limitat...
- [Paper page - ResLoRA: Identity Residual Mapping in Low-Rank Adaption](https://huggingface.co/papers/2402.18039): no description found
- [Language Modeling by Estimating the Ratios of the Data Distribution | Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/#learning-concrete-scores-with-score-entropy): no description found
- [“What&#39;s wrong with LLMs and what we should be building instead” - Tom Dietterich - #VSCF2023](https://youtu.be/cEyHsMzbZBs?si=8iY9GdeDK6XSLxHN): Thomas G. Dietterich is emeritus professor of computer science at Oregon State University. He is one of the pioneers of the field of machine learning. He ser...
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834): Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models ...
- [GitHub - louaaron/Score-Entropy-Discrete-Diffusion: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (https://arxiv.org/abs/2310.16834)](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion): Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (https://arxiv.org/abs/2310.16834) - louaaron/Score-Entropy-Discrete-Diffusion
- [Language Modeling by Estimating the Ratios of the Data Distribution | Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/): no description found
- [Tweet from Aaron Lou (@aaron_lou)](https://fixupx.com/aaron_lou/status/1763242384958386306): Announcing Score Entropy Discrete Diffusion (SEDD) w/ @chenlin_meng @StefanoErmon.  SEDD challenges the autoregressive language paradigm, beating GPT-2 on perplexity and quality!  Arxiv: https://arxiv...

  

---


### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1212728547677052958) (3 messages): 

- **Brief Exchange on GIF Creation**: `@kyo_takano` mentioned using `imageio` presumably related to making a GIF animation. `@.the_alt_man` expressed surprise, asking if it was *made* with `imageio`.
- **Curiosity About Image Processing**: `@karatsubabutslower` chimed in with a "++ curious" indicating interest in the `imageio` discussion.
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1212869551084015616) (21 messages🔥): 

- **Quantization's Potential for Model Interpretability**: `@jstephencorey` inquired about interpretability in 1/1.58 bit language models, referencing a [recent paper](https://arxiv.org/pdf/2402.17764.pdf), suggesting quantized models might be more interpretable. `@woog` indicated that, due to the recency of the paper and lack of released code, there has likely been no work on its interpretability as of yet.

- **Replication Needed for Interpretability Study**: In response to whether very quantized LLMs might be more interpretable, `@woog` suggested that a replication of the model in question would be a necessary first step before studying its interpretability.

- **Transformers' Learning Sensitivity Explored**: `@dashiell_s` shared a [paper](https://arxiv.org/abs/2402.09963) discussing transformers' difficulties in learning functions sensitive to input, leading to a bias towards low sensitivity which may explain certain learnability limitations.

- **Positive Reception to Sensitivities-based Function Learning Theory**: `@norabelrose`, `@quintinpope`, and `@karatsubabutslower` expressed enthusiasm about the paper highlighted by `@dashiell_s`, recognizing its potential contributions to understanding transformers' learning abilities.

- **Linking Sensitivity with Theoretical Computer Science**: `@stellaathena` elaborated on the significance of the paper's insights, connecting sensitivity to theoretical computer science complexity measures and suggesting that low-degree functions correspond to low sensitivity.

**Links mentioned**:

[Why are Sensitive Functions Hard for Transformers?](https://arxiv.org/abs/2402.09963): Empirical studies have identified a range of learnability biases and limitations of transformers, such as a persistent difficulty in learning to compute simple formal languages such as PARITY, and a b...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1212693457211105310) (15 messages🔥): 

- **Progress Update on Task Modifications**: `@asuglia` acknowledged `@981242445696221224`'s ping and informed that they identified the major areas for modification in an ongoing task, but programming changes have been delayed due to other priorities.

- **Lambada Dataset Enhanced Translations**: `@hailey_schoelkopf` announced that `@946388490579484732` improved the translations of the Lambada dataset, surpassing the machine translation quality. They intended to add this to an evaluation harness, referencing the [Hugging Face dataset page](https://huggingface.co/datasets/marcob/lambada_multilingual).

- **Quality Issues with Multilingual Translations**: `@marcobella` pointed out issues with machine translations of the Lambada dataset in various languages, including incorrect punctuation and spacing. They also noted the addition of Dutch and Portuguese (Brazilian) languages to the new translations.

- **Manual Validation Reveals Better Performance**: After manually checking the translations, `@marcobella` found that the quality of translations impacted model performance significantly, with a 10%-15% increase in accuracy for multilingual models after improving the translations.

- **Attempted GPT-4 Translation Abandoned**: `@marcobella` intended to use GPT-4 for translating documents but had to abandon the approach due to a subset of documents that triggered the terms of use violation, leading to manual translations for those cases.

- **Inquiry into Multiple Answer Benchmarks**: `@pbevan1` sought examples of tasks with multiple answers for a single prompt for their implementation of EQ-bench. `@hailey_schoelkopf` suggested the truthfulqa_mc2 as a potential reference.

**Links mentioned**:

- [marcob/lambada_multilingual · Datasets at Hugging Face](https://huggingface.co/datasets/marcob/lambada_multilingual): no description found
- [GitHub - EQ-bench/EQ-Bench: A benchmark for emotional intelligence in large language models](https://github.com/EQ-bench/EQ-Bench/tree/main_v2_1): A benchmark for emotional intelligence in large language models - EQ-bench/EQ-Bench

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1212783352684478514) (6 messages): 

- **Clarification on GPT-NeoX Infrastructure**: `@triggerhappygandhi` clarified that containers for **GPT-NeoX** need to be set up beforehand, as NeoX does not make any infrastructure assumptions, except providing a Slurm script for multinode running.

- **Inquiry about the Pile Validation Set**: `@pietrolesci` asked for details on how the validation set was sampled from **The Pile** dataset, curious about whether it was stratified by source or uniformly sampled.

- **Uniform Sampling Confirmed for Pile Validation**: Responding to `@pietrolesci`, `@hailey_schoelkopf` shared a quote from **The Pile** paper confirming that both validation and testing data were sampled uniformly at random, although the exact timing regarding up/downsampling relative to the creation of the validation set remained unclear.

- **Details on Deduplication and Validation Set**: `@hailey_schoelkopf` informed `@pietrolesci` that deduplication as described in **The Pile** paper occurred before creating its validation set, and also noted the absence of a canonical val set for the deduped dataset used in **Pythia**.
  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1212725947778994177) (10 messages🔥): 

- **GPT-6 Speculation Intensifies**: `@0xevil` mentioned a patent out for GPT-6, suggesting that it could be related to agents and music generation, although details weren't provided.

- **A Guide to Fine-Tuning Gemma 7B**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be) that explains how to fine-tune the Gemma 7B model using Unsloth, also linking to a relevant Google Colab.

- **Vocaloid Ingenuity**: `@everyoneisgross` successfully created a text-to-speech system using a pocket MIKU Vocaloid synth, able to convert English sentences to Vocaloid phonetics and SysEx commands.

- **Elon Musk vs. Open AI**: `@mautonomy` posted that Elon Musk reportedly filed a lawsuit against Open AI and Sam Altman, accusing them of breaching a founding agreement to remain a non-profit.

- **Bittensor Registration Troubles**: `_terps` is looking for assistance with Bittensor registration scripts, having difficulty in acquiring a low registration fee.

**Links mentioned**:

- [Tweet from X News Daily (@xDaily)](https://fxtwitter.com/xDaily/status/1763464048908382253): BREAKING: Elon Musk has filed a lawsuit against Open AI and Sam Altman for breach of contract.  The lawsuit accuses Altman et al with having betrayed an agreement from Open AI&#39;s founding to remain...
- [Finetune Gemma 7B with Unsloth](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be): We will take a look at how to finetune Gemma model using unslothhttps://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollT...
- [OpenCodeInterpreter](https://www.youtube.com/watch?v=cwJKopBBnWo&feature=youtu.be): OpenCodeInterpreter is a suite of open-source code generation systems aimed at bridging the gap between large language models and sophisticated proprietary s...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1212746283467546684) (11 messages🔥): 

- **1-bit Large Language Models on the Horizon**: User `@deki04` expressed surprise upon discovering a new era of 1-bit Large Language Models with the [BitNet b1.58](https://arxiv.org/abs/2402.17764) paper, which claims cost-effective performance matching that of full-precision models. `@max_paperclips` emphasized its potential for hardware implementation due to its ternary and additive nature.
- **Scaling law debate intrigues Nous researchers**: `@sherlockzoozoo` mentioned the `Multiplicative scaling law` from the same [BitNet b1.58 paper](https://arxiv.org/abs/2402.17764), contrasting it with additive scaling which doesn't scale well with model size.
- **Benchmarks for Large Language Models Pique Curiosity**: `@tarruda` shared a [new benchmark for LLMs](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html) with real-world tests. Additional benchmarks were conducted on Nous Research models and can be viewed in a [YouTube video comparison](https://www.youtube.com/watch?v=IH2htfsciO4).
- **Orca-Math Tackles Mathematical Word Problems**: A paper on [Orca-Math](https://arxiv.org/abs/2402.14830) was linked, whereby smaller language models achieve over 80% accuracy on the GSM8K benchmark, suggesting new strategies for problem-solving effectiveness.
- **WeightWatcher Detects Overfitting in LLMs**: `@charlesmartin14` shared his blog post on the WeightWatcher project that helps detect overfitting in fine-tuned LLMs using the concept of Double Descent, along with a link to the tool at [weightwatcher.ai](https://weightwatcher.ai).

**Links mentioned**:

- [Orca-Math: Unlocking the potential of SLMs in Grade School Math](https://arxiv.org/abs/2402.14830): Mathematical word problem-solving has long been recognized as a complex task for small language models (SLMs). A recent study hypothesized that the smallest model size, needed to achieve over 80% accu...
- [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427): Recurrent neural networks (RNNs) have fast inference and scale efficiently on long sequences, but they are difficult to train and hard to scale. We propose Hawk, an RNN with gated linear recurrences, ...
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [
      My benchmark for large language models
    ](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html): no description found
- [Describing Double Descent with WeightWatcher](https://calculatedcontent.com/2024/03/01/describing-double-descent-with-weightwatcher/): Double Descent (DD) is something that has surprised statisticians, computer scientists, and deep learning practitioners&#8211;but it was known in the physics literature in the 80s: And while DD can…
- [Mistral Large vs GPT4 - Practical Benchmarking!](https://www.youtube.com/watch?v=IH2htfsciO4): ➡️ One-click Fine-tuning &amp; Inference Templates: https://github.com/TrelisResearch/one-click-llms/➡️ Trelis Function-calling Models (incl. OpenChat 3.5): http...
- [GitHub - microsoft/azure-openai-dev-skills-orchestrator: Building a set of semantic kernel skills to act as a virtual developer team](https://t.co/1VYs4RU3x8): Building a set of semantic kernel skills to act as a virtual developer team - microsoft/azure-openai-dev-skills-orchestrator

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1212707717358026832) (89 messages🔥🔥): 

- **AI Showdown on the Dota2 Battlefield**: `@afterhoursbilly` shared an [OpenAI blog post](https://openai.com/research/openai-five-defeats-dota-2-world-champions) highlighting how **OpenAI Five** transitioned from defeating bots to cooperating with humans in Dota 2. OpenAI Five is poised for an internet-wide exhibition to uncover strengths and exploitability.
- **Textbook Control with GitHub Actions**: `@thilotee` announced the opening of a [pull request](https://github.com/nomic-ai/gpt4all/pull/2054/files) on GPT4All for the model Notre-Hermes-2-Mistral-7B-DPO, seeking recommendations for system prompts and addressing changes in the codebase's end-of-sentence tokenization.
- **Apple Teases Foray into Generative AI**: `@teknium` speculated Apple's 2023 announcement by Tim Cook on breaking new ground in GenAI might just result in a 3B model running on mobile devices, prompting discussions on potential improvements to Siri.
- **Troubles in Google's Gemma-land?**: Various users, including `@teknium`, noted anomalies with the performance of Google's Gemma models, where larger models like Gemma 8B were underperforming compared to smaller counterparts like Gemma 2B on various benchmarks.
- **RAG-narok Evolved**: `@gabriel_syme` may have cracked the final step in evolving the RAG model, working towards generating interesting questions to navigate through a knowledge base without supervision.

**Links mentioned**:

- [Tweet from Maxime Labonne (@maximelabonne)](https://fxtwitter.com/maximelabonne/status/1763262504883380462?s): It looks like Gemma-7b actually underperforms Gemma-2b on AGIEval, GPT4All, and Bigbench.  I&#39;ve never seen that before, this model is really strange. Any ideas?  🤗 Gemmalpaca-7B: https://huggingf...
- [Tweet from Maxime Labonne (@maximelabonne)](https://fxtwitter.com/maximelabonne/status/1763262504883380462?s=20): It looks like Gemma-7b actually underperforms Gemma-2b on AGIEval, GPT4All, and Bigbench.  I&#39;ve never seen that before, this model is really strange. Any ideas?  🤗 Gemmalpaca-7B: https://huggingf...
- [Tweet from Daniel Han (@danielhanchen)](https://x.com/danielhanchen/status/1763613620909580505?s=12&t=qi3AsKtzHSMXVDOdzD4icQ): Maybe I found another Gemma bug? Can anyone from the #Gemma team confirm if Gemma uses approx gelu or exact gelu? Keras=approx Gemma_pytorch=exact HF=exact When comparing Keras to HF, torch.dist gets ...
- [OpenAI Five defeats Dota 2 world champions](https://openai.com/research/openai-five-defeats-dota-2-world-champions): OpenAI Five is the first AI to beat the world champions in an esports game, having won two back-to-back games versus the world champion Dota 2 team, OG, at Finals this weekend. Both OpenAI Five and De...
- [pansophic/new_model_test · Hugging Face](https://huggingface.co/pansophic/new_model_test): no description found
- [Tweet from Philipp Schmid (@_philschmid)](https://fxtwitter.com/_philschmid/status/1763607891343225217?s=20): Zehpyr 7B Gemma released!🔷🔶 We are excited to announce a Zephyr Gemma, the best-fine-tuned version of @Google Gemma 7B. Outperforming Google Gemma Instruct on 5 out 6 benchmarks, including MT Bench,...
- [Tweet from TechCrunch (@TechCrunch)](https://x.com/techcrunch/status/1762942326391906352?s=46): Tim Cook says Apple will ‘break new ground’ in GenAI this year https://tcrn.ch/3Ig8TAX
- [GitHub - datadreamer-dev/DataDreamer: DataDreamer: Prompt. Generate Synthetic Data. Train &amp; Align Models.    🤖💤](https://github.com/datadreamer-dev/DataDreamer): DataDreamer: Prompt. Generate Synthetic Data. Train &amp; Align Models.    🤖💤 - datadreamer-dev/DataDreamer
- [Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus · Datasets at Hugging Face](https://huggingface.co/datasets/Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus): no description found
- [Models: Remove system prompt of Nous-Hermes-2-Mistral-7b-DPO by ThiloteE · Pull Request #2054 · nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all/pull/2054/fil): Describe your changes  Adds &quot;accepts various system prompts&quot; Removes system prompt fix whitespace  Checklist before requesting a review   I have performed a self-review of my code.  If it is...
- [Models: Remove system prompt of Nous-Hermes-2-Mistral-7b-DPO by ThiloteE · Pull Request #2054 · nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all/pull/2054/files)): Describe your changes  Adds &quot;accepts various system prompts&quot; Removes system prompt fix whitespace  Checklist before requesting a review   I have performed a self-review of my code.  If it is...

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1212801067247603802) (41 messages🔥): 

- **Moving Chaos**: `@slono` returned from a two-week absence filled with moving and life challenges, and `@swyxio` expressed interest in the update.
  
- **AI in the Hardware Lane**: `@guardiang` shared a [YouTube video](https://youtu.be/IixoaS5ckBA) of Jonathan Ross at Web Summit Qatar, discussing Groq's advancements in TPU and LPU technology.
  
- **1-Bit LLM Breakthrough**: `@nembal` highlighted the new "ternary parameters" paper and its claim to match full-precision LLMs with a 1-bit variant, sparking a [Hacker News discussion](https://news.ycombinator.com/item?id=39535800) and a dose of healthy skepticism from `@fanahova` about the need for retraining.

- **Banana.dev Bites the Dust**: `@swyxio` shared a [post-mortem blog](https://blog.erikdunteman.com/banana-pivot-unpeeled) discussing the rise and fall of Banana.dev's Serverless GPUs product and `@stealthgnome` found part of the tale particularly melancholic.

- **AI Product Management Resources**: `@swizec` sought resources on product managing AI projects, with `@420gunna` recommending a Coursera specialization on AI Product Management from Duke University, and `@mrjose9` proposing a Josh Tobin lecture from the Fullstack Deep Learning course.

**Links mentioned**:

- [AI Infrastructure Landscape](https://ai-infra.fun/): no description found
- [no title found](https://news.ycombinator.com/item?id=39535800): no description found
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [Banana Pivot: Unpeeled](https://blog.erikdunteman.com/banana-pivot-unpeeled): Erik Dunteman
- [Tweet from clem 🤗 (@ClementDelangue)](https://x.com/clementdelangue/status/1763328911365353933?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Top HF users by number of model downloads, dataset download, Spaces likes & profile followers. Congrats to everyone in there & to all who will get there soon!  @mvaloatto created the space: https://hu...
- [The Full Stack - Lecture 8: ML Teams and Project Management](https://fullstackdeeplearning.com/course/2022/lecture-8-teams-and-pm/): Building ML-powered products and the teams who create them
- [Jonathan Ross at Web Summit Qatar](https://youtu.be/IixoaS5ckBA?si=): Groq CEO &amp; Founder, Jonathan Ross, on Center Stage at #WebSummitQatar2024, discussing how to make AI Real.X (fka Twitter): @WebSummitQatarInstagram: @WebSumm...
- [Jonathan Ross at Web Summit Qatar](https://youtu.be/IixoaS5ckBA?si=iTQFG-k_SQd6OP8H): Groq CEO &amp; Founder, Jonathan Ross, on Center Stage at #WebSummitQatar2024, discussing how to make AI Real.X (fka Twitter): @WebSummitQatarInstagram: @WebSumm...
- [[AINews] Dia de las Secuelas (StarCoder, The Stack, Dune, SemiAnalysis)](https://buttondown.email/ainews/archive/ainews-dia-de-las-secuelas-starcoder-the-stack/): AI News for 2/28/2024. We checked 356 Twitter feeds and 22 Discords (351 channels, and 9043 messages) for you. Estimated reading time saved (at 200wpm): 860...
- [AI Product Management](https://www.coursera.org/specializations/ai-product-management-duke): Offered by Duke University. Manage the Design &amp; Development of ML Products. Understand how machine learning works and when and how it can be ... Enroll for free.

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1212700149315276810) (8 messages🔥): 

- **Representation Engineering 101 Stage Announcement**: `@ivanleomk` announced that `@aimuggle` would be presenting Representation Engineering 101 in the channel **soon**. 
- **Swyxio Expresses Interest in a Recording**: `@swyxio` regretted missing the session and showed interest in a **recorded version**.
- **Ivanleomk Suggests a Second Round**: `@ivanleomk` proposed the idea of `@aimuggle` doing **round 2** of the Representation Engineering 101 session.
- **Aimuggle Entertains the Idea of a Follow-up**: `@aimuggle` responded playfully to the suggestion and mentioned the possibility of a second session **maybe in a couple weeks**.
- **Making RepEng Library More Accessible**: `@aimuggle` indicated a plan to get the representation engineering library working in a **Colab workbook** on the free tier to make it more accessible.
  

---


### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1212699746758688802) (52 messages🔥): 

- **Seeking Schedule Sweet Spot**: `@aimuggle` and `@youngphlo` discussed timings for the **LLM Asia Paper Club**, considering the diverse time zones of members. A starting time around **8 or 9 pm Singapore Time** or possibly during lunch hours was proposed, with intentions to begin sessions at 6:05 pm to accommodate late joiners.
  
- **Representation Engineering 101**: `@ivanleomk` introduced the topic of **Representation Engineering**, highlighting the importance of understanding and manipulating neural network intermediate representations for applications in steerability and alignment. The session was geared towards beginners and encouraged open participation.

- **The Quest for Clarity**: Users engaged in a detailed discussion about **representation engineering** concepts, with `@fx2y`, `@bryanblackbee`, and `@danial.alh` seeking clarification on topics like the difference between representation and embedding, the computation of vector differences, and methods of evaluation for control vectors.

- **Maneuvering Models on the Fly**: The conversation revolved around the practical application of control vectors, with `@fx2y` and `@jytan` curious about potential stacking of multiple control vectors and inferences being made on-the-fly without additional fine-tuning, which was confirmed as a typical approach.

- **Linear Representation Hypothesis Explored**: `@healthymonkey` questioned the nature of the **linear representation** in the context of the discussed topic, leading to explanations about how shifts in representation space can reflect the meaning of concepts like "good" and "not good" in oppositional directions.

**Links mentioned**:

- [Nextra: the next docs builder](https://llm-paper-club-asia-notes.vercel.app/): Nextra: the next docs builder
- [Representation Engineering Mistral-7B an Acid Trip](https://vgel.me/posts/representation-engineering/#How_do_we_make_one?_Is_it_hard?'): no description found
- [Representation Engineering 101](https://tana.pub/OG9hf2MA4tNS/representation-engineering-101): no description found

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1212811979274846230) (6 messages): 

- **LLMs fine-tune Hybrid Search**: LlamaIndex introduced a method using **LLMs** to optimize hybrid search, automatically setting an alpha parameter based on the query category. They shared a [Twitter post](https://twitter.com/llama_index/status/1763252392639042024) detailing the approach.
  
- **Addressing the need for structured data in RAG**: The LlamaIndex team featured a [blog post by @ClickHouseDB](https://twitter.com/llama_index/status/1763282902358585445) discussing how **RAG architecture** could be modified to handle both unstructured and structured data within the same vector database.

- **Webinar on Local RAG Deployment**: An upcoming webinar featuring @ollama and @tonicfakedata will showcase how to build **local RAG systems** for enhanced privacy, as announced by CEO [@jerryjliu0](https://twitter.com/llama_index/status/1763304038442192978). It will demonstrate deploying LlamaIndex + Tonic Validate with Ollama to maintain data privacy.

- **Improving LLM Apps with Observability**: @jerryjliu0 and @nir_ga from @traceloopdev will demonstrate how to add observability to query pipelines using **OpenLLMetry** in a LlamaIndex webinar. Their [tweet](https://twitter.com/llama_index/status/1763364010676900080) highlights the importance of tracing and instrumenting complex queries.

- **Envisioning Long-Context RAG**: LlamaIndex speculated on the future of RAG in the context of long-context LLMs like **Gemini 1.5 Pro** and discussed possible changes to retrieval techniques in a [Twitter post](https://twitter.com/llama_index/status/1763620476847632744).

**Links mentioned**:

- [Preserve privacy using local RAG with Tonic.ai + LlamaIndex | Webinars | Tonic.ai](https://t.co/ke1XgF5Qb9): Learn how to develop local Retrieval Augmented Generation (RAG) systems and receive a hands-on demonstration showcasing how LlamaIndex + Tonic Validate can be deployed locally using Ollama for complet...
- [Embeddings &amp; NLP](https://t.co/NFXtvm7K4z): mixedbread.ai offers simple text embedding generation, designed to enhance the developing experience in your AI projects.

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1212677950827270175) (84 messages🔥🔥): 

- **Page-Wise Content Extraction Challenge**: `@ansumansatapathy` inquired about extracting page-wise content using Llama Parse. `@whitefang_jr` explained that LlamaParse currently does not include page numbers and suggested sending page-wise PDF files as a workaround.

- **LlamaParse Getting Page Numbers Soon?**: `@cheesyfishes` mentioned that markdown output from LlamaParse will likely use `\n---\n` as a page divider and hinted at upcoming advanced output formats.

- **Groq LLM Access Clarified**: `@sridhar_10158` encountered issues accessing the Groq object within llama_index.llms, which was resolved after reinstalling packages. `@cheesyfishes` responded to queries about Groq's availability and suggested using recent versions of llama_index.

- **Combining Postprocessors in Query Engines**: `@mysterious_avocado_98353` received confirmation from `@cheesyfishes` that multiple Node Postprocessor Modules, like *MetadataReplacementPostProcessor* and *FixedRecencyPostprocessor*, can be chained in a query engine and are applied in the order they are listed.

- **Clarifying Subquery QA Implementation**: `@andreipopg` sought assistance on accessing source nodes for each subquestion to extract metadata and text for QA pairs when implementing subquery QA, but there was no apparent resolution provided in the messages.

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1sQhOI7TN6CUfHp90Uvs8YmHmznpLp0qn?usp=sharing): no description found
- [Ollama - Llama 2 7B - LlamaIndex 🦙 v0.10.14](https://docs.llamaindex.ai/en/stable/examples/llm/ollama.html): no description found
- [no title found](https://www.secinsights.ai/): no description found
- [llama_index/llama-index-packs/llama-index-packs-fuzzy-citation at main · run-llama/llama_index](https://github.com/run-llama/llama_index/tree/main/llama-index-packs/llama-index-packs-fuzzy-citation): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [GitHub - run-llama/sec-insights: A real world full-stack application using LlamaIndex](https://github.com/run-llama/sec-insights): A real world full-stack application using LlamaIndex - run-llama/sec-insights

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1212739586166628382) (20 messages🔥): 

- **Seeking Axolotl User Insights**: `@caseus_` requests users to fill out a [**questionnaire**](https://docs.google.com/forms/d/e/1FAIpQLSeyJkTk7sCYWpCNfKNNpnlMQlT9XU2nt_TJCzP4GSZBT0vrRA/viewform), aiming to understand how end-users interact with axolotl. They later updated the form to make fewer fields required in response to user feedback.
- **Terminology Tweaks Needed**: `@nanobitz` corrects the language used in `@caseus_`'s questionnaire, suggesting "use" axolotl rather than "buy" it.
- **TinyBox Packs a Punch for AI**: `@dreamgen` shares a link about [**TinyBox**](https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production), a high-performance AI system utilizing six AMD Radeon RX 7900 XTX GPUs, aiming to democratize PetaFLOPS-class AI performance.
- **Next Mistral AI Office Hour Announced**: `@casper_ai` posts a Discord invite linking to information about the upcoming office hour for **Mistral AI**, but with no further details provided.
- **Training Loss Troubles**: `@nruaif` reports training losses and gradient norms, indicating potential issues with missing gradient values represented by `nan`.

**Links mentioned**:

- [Join the Mistral AI Discord Server!](https://discord.gg/mistralai?event=1204405056825327677): Check out the Mistral AI community on Discord - hang out with 13789 other members and enjoy free voice and text chat.
- [TinyBox packs a punch with six of AMD's fastest gaming GPUs repurposed for AI &mdash; new box uses Radeon 7900 XTX and retails for $15K, now in production](https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production): Startup wants to offer high AI performance using Radeon RX 7900 XTX.
- [Axolotl End User Questionnaire](https://docs.google.com/forms/d/e/1FAIpQLSeyJkTk7sCYWpCNfKNNpnlMQlT9XU2nt_TJCzP4GSZBT0vrRA/viewform): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1212789413080203304) (38 messages🔥): 

- **Sophia Optimizer Sparks Interest**: `@casper_ai` shared a link to an [arXiv paper](https://arxiv.org/abs/2305.14342) on Sophia, a second-order optimizer claimed to be twice as fast as Adam algorithms, which could significantly reduce the time and cost of training models. They also provided a link to an [implementation of Sophia in Jax (not Torch)](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/optim/sophia.py).

- **Dropping Backwards, Not Standards**: `@suikamelon` introduced DropBP, a novel approach described in an [arXiv paper](https://arxiv.org/abs/2402.17812), that drops layers only during backward propagation to maintain forward propagation's accuracy, and the approach is backed by code that reportedly achieved training time reductions.

- **StarCoder2 Supported**: `@faldore` queried about support for StarCoder2 and subsequently shared a [GitHub repository](https://github.com/bigcode-project/starcoder2) and mentioned an associated pull request for adding StarCoder2 to the project.

- **Unsloth Training on a Single GPU**: `@faldore` expressed interest in training models similarly to how "unsloth training 70b on a single H100" was achieved, as per a [Twitter post](https://twitter.com/danielhanchen/status/1752707608488923614). `@caseus_` responded, mentioning the limitation of unsloth OSS only supporting lora on a single GPU unless integrated with Axolotl, while `@giftedgummybee` noted that most Axolotl hobbyists operate under that same limitation.

- **Issues with TRL's KTO Trainer**: `@giftedgummybee` raised concerns about the KTO trainer in TRL, warning of its poor performance with lora configurations, lack of support for bnb 4 bit, and inefficient computations leading to slow execution. These observations were supported by detailed error logs indicating segmentation faults and other compatibility warnings.

**Links mentioned**:

- [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342): Given the massive cost of language model pre-training, a non-trivial improvement of the optimization algorithm would lead to a material reduction on the time and cost of training. Adam and its variant...
- [DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation](https://arxiv.org/abs/2402.17812): Training deep neural networks typically involves substantial computational costs during both forward and backward propagation. The conventional layer dropping techniques drop certain layers during tra...
- [levanter/src/levanter/optim/sophia.py at main · stanford-crfm/levanter](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/optim/sophia.py): Legible, Scalable, Reproducible Foundation Models with Named Tensors and Jax - stanford-crfm/levanter
- [GitHub - bigcode-project/starcoder2: Home of StarCoder2!](https://github.com/bigcode-project/starcoder2?tab=readme-ov-file#training): Home of StarCoder2! Contribute to bigcode-project/starcoder2 development by creating an account on GitHub.
- [GitHub - OpenLLMAI/OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework (Support 70B+ full tuning &amp; LoRA &amp; Mixtral &amp; KTO)](https://github.com/OpenLLMAI/OpenRLHF): An Easy-to-use, Scalable and High-performance RLHF Framework (Support 70B+ full tuning &amp; LoRA &amp; Mixtral &amp; KTO) - OpenLLMAI/OpenRLHF
- [Add Prodigy, SophiaG optimizers by Kimiko-AI · Pull Request #1350 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1350): Description https://arxiv.org/pdf/2306.06101.pdf https://arxiv.org/abs/2305.14342  Motivation and Context   How has this been tested? No    Screenshots (if appropriate) Types of changes  Social Han...
- [NobodyExistsOnTheInternet/KTO-PRM-small · Datasets at Hugging Face](https://huggingface.co/datasets/NobodyExistsOnTheInternet/KTO-PRM-small): no description found
- [add starcoder2 by ehartford · Pull Request #1349 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1349/files): add starcoder2 Description add starcoder2 Motivation and Context add starcoder2 How has this been tested? i ran a build with it, and it worked Screenshots (if appropriate) Types of changes add star...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1212752354408140890) (4 messages): 

- **Community Cloud Quality Variance**: `@dreamgen` mentioned that on **Community Cloud** the quality of services is varied, indicating a lack of consistency.
- **RunPod Secure Cloud Usage Query**: `@dreamgen` also questioned if anyone is utilizing **RunPod's secure cloud**, suggesting it may not be worth the investment.
- **Starcoder2 Compatibility Check**: `@faldore` inquired about the compatibility of a certain entity or function with **starcoder2**, but did not specify what they were attempting to work with.
- **DPO Training Guide Request**: `@wizmak` is seeking examples or articles on how to train a model using **DPO on axolotl**, indicating a need for instructional resources.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1212754289269809224) (17 messages🔥): 

- **Danish Domination with Mistral**: `@le_mess` shared a success, stating their **7B Mistral model** matches **ChatGPT 3.5's performance** in Danish tasks, using a **synthetic data approach**.
- **Iterative Model Training Yields Results**: Through training over **30 iterative models**, `@le_mess` achieved improved model responses over time, all without using GPT-4, and only for **open source commercial use**.
- **The Human Touch in Automated Curation**: Initially, `@le_mess` manually curated 1000 responses, then used trained models for further automated curation to refine the outputs and retrain the models.
- **Validation Secrets Revealed!**: When `@nanobitz` asked about the evaluation datasets, `@le_mess` clarified they actually referred to a **validation dataset** and mentioned they use the benchmark from [Scandeval.com](https://scandeval.com).
- **Benchmarking Basics**: `@le_mess` confirmed not creating their own benchmarking tools, directing a user to the external resource they utilize, hinting at the complexities behind making one’s own evaluation datasets.
  

---



### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1212693857360285706) (17 messages🔥): 

- **Tensor Performance Tuning Success**: `@iron_bound` revealed they solved performance issues with tensors on the **7900xtx** by enabling WMMA in MLIR/LLVM, sharing detailed performance metrics for different precision formats on large matrix sizes. A commit link explaining the changes was provided: [Tensor core fix on RDNA3](https://github.com/joviliast/triton/commit/f2f21c1a17437d3cef42dc31c08c82491ce4b08b).
  
- **Troubleshooting Triton Debugger**: `@kierandidi` encountered a `TypeError` when attempting to use the Triton debugger with an unexpected keyword argument 'interpret' in both **Triton 3.0.0 and 2.2.0**. `@andreaskoepf` suggested setting the `TRITON_INTERPRET` environment variable, which `@marksaroufim` confirmed as the correct approach due to the deprecation of the previous method.

- **Seeking Segfault Solutions**: `@andreaskoepf` shared their experiences with segfault issues in Triton, which were resolved by reordering code lines and modifying the use of `num_warps`, alongside links to both the problematic [code](https://gist.github.com/andreaskoepf/833aac25c6e049e37ddadb5d0ad1ef48) and the [revised version](https://gist.github.com/andreaskoepf/4916d2a010f175b25aaa0655c8e5c9b4).

- **Full Day of Triton Development Joy**: `@andreaskoepf` expressed enthusiasm for spending a full day developing with Triton, while `@drisspg` inquired if the development was on Triton code or within the Triton environment itself, seeking to understand the context of the development work.

**Links mentioned**:

- [[AMD][Navi31] Convert WMMA dot op to LLVM · joviliast/triton@f2f21c1](https://github.com/joviliast/triton/commit/f2f21c1a17437d3cef42dc31c08c82491ce4b08b#diff-c3f95d90ba556d38204257db3be8b6ae4f66f08d247ea8684ffec76432f6e05c): Add WMMA conversion logic for dot operation.  Signed-off-by: joviliast &lt;iveselov.nn@gmail.com&gt;
- [flash_attn_bias.py](https://gist.github.com/andreaskoepf/833aac25c6e049e37ddadb5d0ad1ef48): GitHub Gist: instantly share code, notes, and snippets.
- [fash_attn_triton_working.py](https://gist.github.com/andreaskoepf/4916d2a010f175b25aaa0655c8e5c9b4): GitHub Gist: instantly share code, notes, and snippets.

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1212784318867312650) (12 messages🔥): 

- **FP8 Intrinsics Availability Confirmed**: `@zippika` pointed out that **fp8 (8-bit floating-point)** intrinsics are still available in [CUDA's documentation](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__MISC.html#group__CUDA__MATH__FP8__MISC), requiring inclusion of the header file `cuda_fp8.h` in programs.

- **FP8 Primarily for Data Storage**: `@zippika` emphasized that fp8 is mainly a 'data' format and is not commonly used for actual computations.

- **cudaMallocManaged vs. malloc Discussion**: `@vim410` discussed the differences between `malloc` and `cudaMallocManaged`, referencing a blog post about heterogeneous memory management (HMM), indicating the latter is better than `malloc` but not as fast as `cudaMalloc`.

- **Limited FP8 Compute Operations on Ada Lovelace GPUs**: `@drisspg` shared insights into fp8 computations, referencing a [PyTorch discussion](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815), which mentions limited support for fp8 compute operations on the Ada Lovelace GPUs, specifically the lack of the `wgmma.mma_async` instruction.

- **Unified Memory in CUDA and PyTorch**: `@marksaroufim` shared a [Github link](https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L181) while discussing unified memory and `cudaMallocManaged`, noting that if unified memory allows writing faster code compared to CPU offloading, then it might be seen as a better default option for resource-constrained GPU setups.

**Links mentioned**:

- [CUDA Math API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__MISC.html#group__CUDA__MATH__FP8__MISC): no description found
- [bitsandbytes/bitsandbytes/functional.py at main · TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L181): Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1212979036054229032) (6 messages): 

- **BASED Attention Paper Shared**: `@marksaroufim` shared a link to a research paper exploring the efficiency of attention-based language models and introduced BASED, a new architecture. The paper is available at [arXiv](https://arxiv.org/pdf/2402.18668.pdf).
- **Attempting Sliding Window Attention**: `@drisspg` is working on adding [sliding window attention biases to PyTorch](https://github.com/pytorch/pytorch/pull/120143), which could improve memory consumption issues during inference.
- **Discussion on Abstract and Mask Implementation**: `@andreaskoepf` provided the abstract link to the same [paper](https://arxiv.org/abs/2402.18668), and `@marksaroufim` queried if different implementations could be achieved by changing the mask in scaled dot-product attention (sdpa).
- **Laughter at BASED Attention**: With a tongue-in-cheek comment, `@marksaroufim` responded to the naming of BASED attention in the discussed paper with "based attention lmao."
- **Concerns Over Default Attention in HF Transformers**: `@marksaroufim` expressed surprise at finding that Hugging Face's Mistral implementation defaults to sdpa without using sliding_window attention, leading to potential issues above 4k context. The message references a tweet with concerns and links to the [relevant code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L1004) and a proposed [fix PR](https://github.com/huggingface/transformers/pull/29220).

**Links mentioned**:

- [Tweet from Philipp Singer (@ph_singer)](https://x.com/ph_singer/status/1763538607191527540?s=20): Apparently current HF transformers Mistral implementation has default attention set to sdpa, but does not use sliding_window. I was seeing weird behavior above 4k context for a while. So if I am not m...
- [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668): Recent work has shown that attention-based language models excel at recall, the ability to ground generations in tokens previously seen in context. However, the efficiency of attention-based models is...
- [Add sliding window attention bias  by drisspg · Pull Request #120143 · pytorch/pytorch](https://github.com/pytorch/pytorch/pull/120143): Summary WIP still See #119653 for more context

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1212776944077377556) (34 messages🔥): 

- **Gradient Confusion**: `@ericauld` questioned an issue related to the backward pass, later clarified by `@andreaskoepf` that it seemingly pertains to incorrect gradients.
- **Errors plague execution**: `@ericauld` encountered multiple issues when attempting to run a test script, including typos and missing imports, leading to abandoning the effort.
- **Troubling Triton message**: `@jamesmel` pointed out that setting `cuda = True` leads to issues, highlighting an error with triton and pointer arguments.
- **Commit History Clues at Broken Code**: `@iron_bound` suggested that the commit history of lucidrains' repository could indicate issues with the custom kernel attempted therein, linked [here](https://github.com/lucidrains/ring-attention-pytorch/commits/main/).
- **Ringing in the GPU Errors**: `@andreaskoepf` noted strange behavior with GPU resource allocation and missing modules, prompting a reboot of the system for resolution.

**Links mentioned**:

- [ring-attention-pytorch/ring_attention_pytorch/ring_flash_attention_cuda.py at df48d4d338f5b970086aec2df75e4be34080de1b · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/blob/df48d4d338f5b970086aec2df75e4be34080de1b/ring_attention_pytorch/ring_flash_attention_cuda.py#L61): Explorations into Ring Attention, from Liu et al. at Berkeley AI - lucidrains/ring-attention-pytorch
- [Commits · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/commits/main/): Explorations into Ring Attention, from Liu et al. at Berkeley AI - Commits · lucidrains/ring-attention-pytorch
- [GitHub - zhuzilin/ring-flash-attention: Ring attention implementation with flash attention](https://github.com/zhuzilin/ring-flash-attention/): Ring attention implementation with flash attention - zhuzilin/ring-flash-attention
- [A ring attention with flash attention kernel implementation · Issue #4 · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/issues/4#issuecomment-1969440025): Hi! Thank you for your work on implementing the ring attention in pytorch! I&#39;ve just tried to implement a ring_flash_attn_qkvpacked_func (corresponding to flash_attn_qkvpacked_func in flash attent...
- [ring-attention-pytorch/ring_attention_pytorch/ring_flash_attention_cuda.py at df48d4d338f5b970086aec2df75e4be34080de1b · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/blob/df48d4d338f5b970086aec2df75e4be34080de1b/ring_attention_pytorch/ring_flash_attention_cuda.py#L349): Explorations into Ring Attention, from Liu et al. at Berkeley AI - lucidrains/ring-attention-pytorch

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1212679781355888700) (30 messages🔥): 

- **Seeking Environment Advice for ChatVertex AI**: User `@irfansyah5572` inquired about the best environment to use when working with ChatVertex AI using Langchain, but no further details or responses were provided.
- **JSON Schemas and LLMs Integration**: `@kamakshi08` shared a [link](https://python.langchain.com/docs/modules/model_io/output_parsers/types/json) explaining how to use JSON schemas with large language models (LLMs) for generating well-formed JSON outputs. They posted a follow-up question on how to use this parser with `llava` downloaded via `ollama`, mainly concerning multi-modal models.
- **Troubleshooting in DataBricks Workflow Jobs**: User `@hanumantgarad_25732` discussed an issue where `SQLDatabase.from_databricks` works in Databricks notebooks but fails in Databricks workflow jobs with an `AttributeError`. The user hypothesized that the error was due to the absence of the `DatabricksReplContext` object outside of notebook environments.
- **Exploring Retries for Custom LangChain Tools**: User `@abinandan` sought a method for retrying a custom LangChain tool upon a `ToolException`, and was supported by `kapa.ai` suggesting user-shared workarounds involving outputting known values or raising exceptions with identifiable text for retry conditions.
- **Questions on LangChain Security and Capabilities**: `@.suzerain` asked if LangChain employs extra safeguards in their lcel but no direct answer was provided by `kapa.ai`. User `@akis_21513` linked to an existing LangChain [GitHub issue](https://github.com/langchain-ai/langchain/issues/18292) reflecting a similar problem they encountered but no solution was suggested in the chat.
- **Automating Shopify Customer Support with AI**: `@erikk4` described a process for using AI tools to automate customer support for Shopify-related queries and asked for tool recommendations besides LangChain. There was no follow-up discussion or further guidance provided.
- **Questions and Issues with Weaviate and LangChain**: Users `@dazzling_puppy_08816` and `@chayan_systango` expressed issues with using LangChain, specifically when trying to get it working in VSCode and initializing Weaviate for existing indexes respectively, but no solutions were presented in the messages.
- **Implementing GPTCache and Handling Batching in LangChain**: `@tawsif2781` discussed the complexities of using batch in conjunction with invoking a chain and finding a way to mix both methods using LCEL, while `@david_zoe` sought assistance with implementing GPTCache and faced an Onnx runtime error but no guidance was provided.

**Links mentioned**:

- [no title found](https://js.langchain.com>)): no description found
- [JSON parser | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/model_io/output_parsers/types/json): This output parser allows users to specify an arbitrary JSON schema and
- [How to do RAG on a already created weaviate persisted data · langchain-ai/langchain · Discussion #15332](https://github.com/langchain-ai/langchain/discussions/15332): Hi All, I am exploring weaviate with langchain. I have loaded a bunch of pdf’s and do the standard splitting and create a weaviate class as class_obj = { &quot;class&quot;: &quot;WMOInfo&quot;, &quot;...
- [Custom Agent Class fails with object has no attribute &#39;is_single_input&#39; · Issue #18292 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/18292): Checked other resources I added a very descriptive title to this issue. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and di...
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/10714>).): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1212755246955237459) (12 messages🔥): 

- **Langserve Troubleshooting by thatdc**: `@thatdc` is facing an issue where **langserve** is not returning the intermediate steps from their agent, only the final output. They believe the problem lies within the RemoteRunnable object's `_invoke` method and the `_decode_response` method, specifically at `output = serializer.loadd(obj["output"])`.

- **Workarounds Suggested by veryboldbagel**: `@veryboldbagel` suggested to use `Any` in the `output_type` to possibly solve the issue. They also pointed to an unresolved [GitHub issue #381](https://github.com/langchain-ai/langserve/issues/381) related to serialization with intermediate steps and further recommended adding an extra part to the chain for handling serialization as a workaround.

- **API Request Investigation**: `@thatdc` shared a **curl** command for testing the API which demonstrates their call to the agent, and subsequently posted the JSON response they received, showing only the final output.

- **Details on Agent Executor Configuration**: `@thatdc` posted the configuration for their **AgentExecutor**, highlighting `return_intermediate_steps=True` and `streaming=True` in hope of receiving intermediate steps in the output.

- **Spammed Gift Link**: `@skywalker09_` posted an unsolicited link to a purported $50 steam gift which seems unrelated to the discussion and may be considered spam.

**Links mentioned**:

[Serialization issues with intermediate_steps for AgentExecutor · Issue #381 · langchain-ai/langserve](https://github.com/langchain-ai/langserve/issues/381): I experimented with a use case in which I initialize an AgentExecutor with an agent chain that is a RemoteRunnable. i.e., the client side looks like this: from langchain.agents import AgentExecutor...

  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1212720596207673394) (2 messages): 

- **Inquiry on Generating Templates**: `@tigermusk` asked about creating a template similar to the one found at [Smith Langchain's React Chat JSON template](https://smith.langchain.com/hub/hwchase17/react-chat-json). There was no follow-up information provided on how to accomplish this in Python code.
- **Spam Alert**: `@skywalker09_` posted a message that appears to be spam, offering a "$50 Gift" with a link to [steamcommunity.com/gift/50](https://u.to/eA9sIA).

**Links mentioned**:

[LangSmith](https://smith.langchain.com/hub/hwchase17/react-chat-json): no description found

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1212986681922289674) (5 messages): 

- **Prompt Editing Made Easy with Endoftext**: `@cabreraalex` introduced **Endoftext**, an AI-powered prompt editor that generates suggestions and test cases for better AI prompts. Watch their 60-second demo on [YouTube](https://youtu.be/PGv5ymOhaNA) and try the beta version at [Endoftext](https://app.endoftext.app/).

- **Airbyte Meets Langchain**: `@andysingal` shared an article on how the integration of Airbyte with Langchain can streamline data integration and document processing. Learn more about their synergetic use in this [Medium post](https://medium.com/ai-advances/airbyte-with-langchain-streamlining-data-integration-and-document-processing-8593db1fc3ad).

- **SimplyAnalyze AI Launches Developer Preview for Conversation Analytics**: `@petervandijck_68934` announced the launch of **SimplyAnalyze AI**, a platform similar to Google Analytics but tailored for analyzing conversations. Early adopters can sign up for a free one-year account at [SimplyAnalyze.AI](https://simplyanalyze.ai).

**Links mentioned**:

- [Airbyte with Langchain: Streamlining Data Integration and Document Processing](https://medium.com/ai-advances/airbyte-with-langchain-streamlining-data-integration-and-document-processing-8593db1fc3ad): Ankush k Singal
- [endoftext Demo -  An AI-powered Prompt Editor](https://youtu.be/PGv5ymOhaNA): endoftext helps you write better prompts with suggested edits, prompt rewriting, and test case generation. Check it out at https://endoftext.app
- [endoftext | AI-powered prompt editor](https://app.endoftext.app/): Take the guesswork out of prompt engineering with prompt suggestions, smart re-writing, and synthetic data generation. endoftext is an AI-powered prompt writing assistant that helps you quickly improv...

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1212826031908589578) (3 messages): 

- **LangGraph Combined with YahooFinance**: User `@tarikkaoutar` shared a [YouTube video](https://www.youtube.com/watch?v=r2PvHdkaXWc&t=129s) demonstrating how LangGraph can be used to create a multi-agent stock analysis chatbot by integrating Function Call and YahooFinance. The video was highlighted for those looking to understand the application of LangGraph in different scenarios.
- **Suspicious Steam Link Alert**: User `@skywalker09_` posted a link claiming to be a $50 gift via Steam ([steamcommunity.com/gift/50](https://u.to/eA9sIA)). However, this should be approached with caution as it may potentially be a phishing link or scam.
- **GPTCache Implementation Inquiry**: User `@david_zoe` asked the community for assistance regarding the implementation of GPTCache from Langchain, facing an "Onnx runtime error". They expressed interest in exploring embedding options from OpenAI or HuggingFace's SafeTransformers and are seeking guidance to resolve caching issues.

**Links mentioned**:

[LangGraph + Function Call+ YahooFinance =  Multi-Agent Application](https://www.youtube.com/watch?v=r2PvHdkaXWc&t=129s): #chatbot #animation #trading  #ai #machinelearning #datascience In this video, you will make an AI stock analysis chatbot with LangGraph, Function call and C...

  

---



### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1212735927546421288) (45 messages🔥): 

- **Prepaid Card Predicament**: User `@fakeleiikun` inquired about prepaid card support on OpenRouter and mentioned issues such as *error 402 or error 502* when using Google Pay, despite the card functioning on other sites. `@louisgv` advised that prepaid cards like Discovery may be flagged by Stripe Radar, but virtual cards from supported banks are generally accepted.
  
- **Asking for Assistance with Helicone Integration**: User `@wise_monkey_42910` sought help integrating Helicone with OpenRouter using Langchain ChatOpenAI. `@louisgv` provided a helpful link to an [example on GitHub](https://github.com/OpenRouterTeam/openrouter-examples/blob/main/examples/langchain/index.ts) and the [Helicone documentation](https://docs.helicone.ai/getting-started/integration-method/openrouter) for proper integration.

- **Token Troubles Clarified**: `@cupidbot.ai` asked about streaming with function calling and the distinction between `native_tokens_prompt` and `tokens_prompt`. `@alexatallah` clarified that `native_tokens` refers to tokens in the model's own tokenizer, and existing usage metrics are indeed native, with plans to update the documentation accordingly.

- **Elon Musk and OpenRouter**: The conversation took a turn when `@telepathyx` suggested that Elon Musk might be entering a space that competes with OpenRouter. Though `@louisgv` was surprised at first, `@alexatallah` corrected that Groq, not Grok, could be a potential future addition to OpenRouter once rate limitations are addressed, debunking the idea of Musk's direct competition.

**Links mentioned**:

- [OpenRouter - Helicone](https://docs.helicone.ai/getting-started/integration-method/openrouter): no description found
- [openrouter-examples/examples/langchain/index.ts at main · OpenRouterTeam/openrouter-examples](https://github.com/OpenRouterTeam/openrouter-examples/blob/main/examples/langchain/index.ts): Examples of integrating the OpenRouter API. Contribute to OpenRouterTeam/openrouter-examples development by creating an account on GitHub.

  

---



### Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1212891180728188949) (16 messages🔥): 

- **Nate's Chance Encounter with Yann**: `@natolambert` shared an exciting personal update that he met Yann LeCun, although he didn't muster up the courage to invite him onto the podcast.
- **Charisma Boost Needed**: In response to `@natolambert`'s hesitation, `@philpax` playfully suggested that he'll succeed in his "charisma check" next time.
- **Green Energy Topics with Yann**: The conversation with Yann LeCun was engaging, as `@natolambert` and Yann discussed **green energy** extensively.
- **The Lambert Connection**: Chat about a Lambert family connection humorously materialized, with `@victory` hinting at a custom server emoji, and `@mike.lambert` trying to identify which Lambert `@victory` is related to, speculating it might be Nate.
- **Yann's Insider Outlook**: `@natolambert` highlighted that Yann LeCun seemed pretty chill and open, but expressed feelings of solitude regarding the fight for openness in AI, also showing skepticism towards reinforcement learning (RL), which he summarized as *normal yann stuff*.
  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1212763024054816789) (4 messages): 

- **Generating False Answers for DPO Dataset**: `@philipmay` suggested a method for creating negatives in a Dataset for Dense Passage Retrieval (DPR) by **asking the LLM to generate intentional wrong answers** based on a given context and question.
- **DiscoLM_German_7b Settings Search**: `@mab3049` is seeking insight into the settings used for the **DiscoLM_German_7b demo model** as their own attempts have not matched the demo's results.
- **Padding Token Confusion in Fine Tuning**: `@silicagel_64242` inquired about which token should be used as the `pad_token` during Fine Tuning. They've encountered conflicting advice, referencing `eos_token`, `unk_token`, and an explicit `"[PAD]"` token.
- **Seeking the Best German Embedding Model for RAG**: `@bjoernwerner` requested opinions on the most effective embedding model for German text in domain-specific Retriever-Aggregator-Generator (RAG) applications, listing several potential **single and multi-vector embeddings** for consideration.
  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1212877396936753214) (5 messages): 

- **Searching for Elusive MT-Bench-X**: `@crispstrobe` is looking for the **MT-Bench-X** dataset and mentioned its Apache 2.0 license as per a [paper on arxiv.org](https://arxiv.org/pdf/2402.13703v1.pdf). The specific interest is to find a model that performs well in German language tasks.
- **Alternative German Dataset Discovered**: `@bjoernp` hasn't seen MT-Bench-X but suggests the **MT-Bench-DE** on Hugging Face, which might be helpful for those seeking German language benchmarks.
- **Advocating for True German Benchmarks**: `@crispstrobe` recommends the manually improved [MT-Bench-TrueGerman](https://huggingface.co/datasets/VAGOsolutions/MT-Bench-TrueGerman) dataset, underscoring the scarcity of authentic German benchmarks and the pitfalls of using GPT-4 translations for this purpose.


**Links mentioned**:

- [VAGOsolutions/MT-Bench-TrueGerman · Datasets at Hugging Face](https://huggingface.co/datasets/VAGOsolutions/MT-Bench-TrueGerman): no description found
- [LeoLM/MT-Bench-DE · Datasets at Hugging Face](https://huggingface.co/datasets/LeoLM/MT-Bench-DE): no description found

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1212858233925013587) (3 messages): 

- **EQ-Bench Adds German Prompts**: User `@crispstrobe` shared a [GitHub pull request](https://github.com/EQ-bench/EQ-Bench/pull/12) announcing that EQ-Bench now supports a set of German prompts for quick evaluations. The results show that models like `gpt-4-1106-preview` and `gpt-3.5-turbo-0125` achieved high scores, and the German translations were done using ChatGPT-4-turbo.

- **Possible Mistakes in Ollama Model Templates**: Additionally, `@crispstrobe` referenced an [issue on GitHub](https://github.com/ollama/ollama/issues/1977) discussing potential mistakes in the template definitions of models downloadable from ollama.ai, which could be impacting model performance. 

- **Ongoing Discussions on Discord**: `@_jp1_` pointed to an [extensive discussion](https://discord.com/channels/1178995845727785010/1183158791605330051/1211590899902058557) concerning the topic, although no specific details of the discussion were provided.

**Links mentioned**:

- [GitHub: Let’s build from here](https://github.com/): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...
- [Build software better, together](https://github.com/EQ-bench/EQ-Bench/pull/12):): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [Mistakes in template definitions on models available to download from https://ollama.ai · Issue #1977 · ollama/ollama](https://github.com/ollama/ollama/issues/1977): Hi, Some of the mistakes in the TEMPLATE definitions for the models you can download from https://ollama.ai are hurting the models to varying degrees. I only found this by accident when experimenti...

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1212690854653861938) (3 messages): 

- **Claude's Intro Quirk**: `@justinpinkney` shared a technique to avoid Claude's tendency to start responses with phrases like "Sure here's a..." by setting the initial characters returned by the model, as detailed in [Anthropic's rewrite guidelines](https://docs.anthropic.com/claude/docs/ask-claude-for-rewrites). This can force Claude to start with a specific character such as `<rewrite>` to bypass unhelpful introductions.
- **Nudging Claude in the Right Direction**: `@derekpwillis` concurred with the difficulty in bypassing Claude's intro comments and has experimented with forcing it to start with `{`, although Claude often insists on explaining its actions.

**Links mentioned**:

[Ask Claude for rewrites](https://docs.anthropic.com/claude/docs/ask-claude-for-rewrites): If Claude gives a response that is close to, but not quite what you&#x27;re looking for, you can ask Claude to rewrite it. In Slack this can be as simple as telling Claude to &quot;Try again&quot; aft...

  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1213152538988253265) (8 messages🔥): 

- **Looking for the Best Open Source LLM**: `@gwthompson` asked for recommendations on the best open source model that can be run locally with LLM and used with Datasette enrichment, but no recommendations were provided in the messages.
- **Seeking Clean C APIs for LLM**: `@florents_` inquired about LLMs with a clean C API for text embedding but did not receive direct recommendations for his query.
- **Introducing Llama.cpp with C API**: `@agarcia_me` mentioned the availability of embedding support in Llama.cpp, which needs a C++ compiler but provides a C API. They also noted their intention to share the code for a sqlite extension for embeddings soon.
- **Clarification on C API Usage**: In response to `@florents_`, `@agarcia_me` provided a clarification that `embedding.cpp` uses only a few functions from `common.h`, suggesting ripping out necessary functions and relying directly on the C APIs.
- **Sharing Code Snippet for LLM Embeddings in C**: `@agarcia_me` shared a detailed C code snippet to demonstrate how LLM embeddings could be implemented, mentioning it works for batch sizes of one and is in pure C, and later clarified that `llama_batch` is the most complex part of the process.

**Links mentioned**:

[llama.cpp/examples/embedding/embedding.cpp at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding/embedding.cpp): LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.

  

---



### LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1213215210278752348) (3 messages): 

- **Anthropic pulls ahead of Gemini 1.5**: User `@res6969` mentioned rumors that **Anthropic** is outclassing **Gemini 1.5** in context length capabilities when tested behind closed doors.
- **Anthropic also leads in accuracy**: Alongside context length, `@res6969` also heard that Anthropic showed **significantly better accuracy** compared to Gemini 1.5.
- **Lack of personal testing**: Despite the buzz, `@res6969` has noted they **haven't been able to test** these capabilities personally.
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1212764382644146186) (2 messages): 

- **In Search of OpenAI Resources**: `@res6969` expressed a need to find better **resources** and **information sources** on OpenAI.
- **Request for Production-Grade Tips**: `@res6969` is looking for **resources** or guidance on implementing OpenAI's **codeinterpreter** in a production environment.
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1212999028678074388) (2 messages): 

- **The Secret Sauce in System Prompts**: `@robertchung` brought up the topic of **system prompts** and their impact on model outputs, noting the significant yet somewhat mysterious role they play, but also mentioned the lack of available resources on the subject.
- **Model Behavior Affected by Variability and Updates**: `@jeffreyw128` suggested that the effectiveness of system prompts might vary depending on the **specific model** and **ongoing updates** conducted by labs, indicating an element of unpredictability in their performance.
  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1212713293693718598) (4 messages): 

- **Seeking AI Engineer Job Description Examples**: User `@peterg0093` is looking for good job description examples for AI engineers in the UK and is keen on adopting emerging standard language for the recruitment process.
- **Sharing a Descriptive Job Example**: `@swyxio` provided a link to Hex's AI engineer careers page offering insights into the company's culture, mission, and the role expectations, potentially serving as a useful template for job descriptions: [Hex Careers](https://hex.tech/careers/ai-engineer/).
- **Suggestion for AI Engineer Foundation Model**: `@swyxio` suggested that the "AI Engineer Foundation" (AIEF) could use a structured setup similar to the one found at [AI-Infra](https://ai-infra.fun/) to organize resources.

**Links mentioned**:

- [AI Infrastructure Landscape](https://ai-infra.fun/): no description found
- [AI Engineer - Careers | Hex ](https://hex.tech/careers/ai-engineer/): Work at the cutting edge of production AI applications.

  

---


### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1212946349813661769) (1 messages): 

- **Recognition for Event Organization**: User `@huikang` expressed gratitude for being mentioned in connection with an event on [LinkedIn](https://www.linkedin.com/posts/ai-eng-foundation_last-saturday-on-022424-sasha-organized-activity-7169152145336782850-_TsG), highlighting involvement in a recent event organized last Saturday, 02/24/24.
  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1212804141626294282) (2 messages): 

- **Finetuning Gemma 7B Explored**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be) titled "Finetune Gemma 7B with Unsloth," providing a walkthrough on finetuning the Gemma model, alongside a [Colab notebook](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollTo...).
- **Introduction to OpenCodeInterpreter**: `@pradeep1148` also posted a [YouTube video](https://www.youtube.com/watch?v=cwJKopBBnWo&feature=youtu.be) about "OpenCodeInterpreter," revealing an open-source initiative for code generation systems designed to work with large language models.

**Links mentioned**:

- [OpenCodeInterpreter](https://www.youtube.com/watch?v=cwJKopBBnWo&feature=youtu.be): OpenCodeInterpreter is a suite of open-source code generation systems aimed at bridging the gap between large language models and sophisticated proprietary s...
- [Finetune Gemma 7B with Unsloth](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be): We will take a look at how to finetune Gemma model using unslothhttps://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollT...

  

---


### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1192042724480794685/1212786676825329695) (1 messages): 

- **Recruitment Offer in the Chat**: User `.papahh` reached out directly to `@1117586410774470818` with a job offer, asking them to check their DM for details. No further context or information was provided in the message.
  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1212862697079574591) (1 messages): 

- **Neural Network Niche Mapping**: `@camelfacts` has introduced a paper that presents a new approach to interpreting **neural network representations** by creating a map of representational niches. The paper combines economic and information theory and has been shared for feedback on LessWrong at [What’s in the box? Towards interpretability by distinguishing...](https://www.lesswrong.com/posts/7tSthxSgnNxbt4Hk6/what-s-in-the-box-towards-interpretability-by-distinguishing-1).

**Links mentioned**:

[What’s in the box?! – Towards interpretability by distinguishing niches of value within neural networks. — LessWrong](https://www.lesswrong.com/posts/7tSthxSgnNxbt4Hk6/what-s-in-the-box-towards-interpretability-by-distinguishing-1): Abstract Mathematical models can describe neural network architectures and training environments, however the learned representations that emerge hav…

  


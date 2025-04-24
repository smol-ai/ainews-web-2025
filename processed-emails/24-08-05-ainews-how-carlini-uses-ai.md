---
id: 8ed20429-df08-4f5d-8ae1-db6e30ed9a10
title: How Carlini Uses AI
date: '2024-08-05T23:43:14.094795Z'
original_slug: ainews-how-carlini-uses-ai
description: >-
  AI news from August 2-5, 2024, covering developments in robotics, large
  language models, AI research, and industry updates. Highlights include new
  humanoid robots, OpenAI's voice features, open-sourced models from Google and
  Meta, and advancements in 3D generation and AI tools.
companies:
  - groq
  - intel
  - deepmind
  - google
  - meta
  - nvidia
  - stability-ai
  - runway
  - box
models:
  - gpt-3.5-turbo-0613
  - gemma-2-2b
  - sam-2
  - stable-fast-3d
  - gen-3-alpha
topics:
  - robotics
  - large-language-models
  - ai-research
  - multimodal
  - text-to-video
  - object-detection
  - 3d-generation
  - ai-apis
  - ai-robotics
---


<!-- buttondown-editor-mode: plaintext -->**An open mind is all you need.**

> AI News for 8/2/2024-8/5/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**249** channels, and **5970** messages) for you. Estimated reading time saved (at 200wpm): **685 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Congrats to [Groq's shareholders' net worth going up](ttps://x.com/groqinc/status/1820422643004424631?s=46) while  everyone else's goes down (and [Intel's CEO prays](https://x.com/datnofact/status/1820213413319962975?s=61)). Nicholas Carlini of DeepMind is getting some recognition (and [criticism](https://nicholas.carlini.com/writing/2024/why-i-attack.html)) as one of the most thoughtful public writers on AI with a research background. This year he has been broadening out from his usual [adversarial stomping grounds](https://arxiv.org/abs/2311.17035) with his [benchmark for large language models](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html) and made waves this weekend with an [80,000 word treatise on How He Uses AI](https://nicholas.carlini.com/writing/2024/how-i-use-ai.html), which we of course [used AI to summarize](https://claude.ai/chat/157b11d3-cec1-4a97-877f-da829d6f2a39):

 ![image.png](https://assets.buttondown.email/images/820a8c18-851b-4862-9821-fce5d442da5f.png?w=960&fit=max) 

as well as usecases:

 ![image.png](https://assets.buttondown.email/images/1359acfe-2c17-4d21-8ec8-5fa2943fdd11.png?w=960&fit=max) 

And, impressively, he says that this is "less than 2%" of the usecases for LLMs he has had (that's 4 million words of writing if he listed everything).

Chris Dixon is [known](https://cdixon.org/2013/03/02/what-the-smartest-people-do-on-the-weekend-is-what-everyone-else-will-do-during-the-week-in-ten-years) for saying "What the smartest people do on the weekend is what everyone else will do during the week in ten years". When people blow [wind on the setting AI Winter](https://www.latent.space/p/q2-2024-recap) saying it hasn't produced enough measurable impact at work, they may simply be too short term oriented. Each of these is at least worth polished tooling, if not a startup.

---

> New: we are experimenting with smol, tasteful ads specifically for help AI Engineers. Please click through to support our sponsors, and hit reply to let us know what you'd like to see!

**[Sponsored by Box]** Box stores docs. Box can also **extract structured data** from those docs. [Here’s how to do it using the Box AI API.](https://medium.com/box-developer-blog/extracting-structured-data-using-box-ai-01408437352d). 

**swyx comment**: S3's rigidity is Box's opportunity here. The idea of a "multimodal Box" - stick anything in there, get structured data out - makes all digital content legible to machines. Extra kudos to the blogpost for also showing that this solution -like any LLM-driven one- can fail unexpectedly!

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
**AI and Robotics Developments**

- **Figure AI**: [@adcock_brett](https://twitter.com/adcock_brett/status/1820128340138713547) announced the launch of Figure 02, described as "the most advanced humanoid robot on the planet," with more details coming soon.

- **OpenAI**: Started rolling out ['Advanced Voice Mode'](https://twitter.com/adcock_brett/status/1820128362708307982) for ChatGPT to some users, featuring natural, real-time conversational AI with emotion detection capabilities.

- **Google**: Revealed and open-sourced [Gemma 2 2B](https://twitter.com/adcock_brett/status/1820128475354730875), scoring 1130 on the LMSYS Chatbot Arena, matching GPT-3.5-Turbo-0613 and Mixtral-8x7b despite being much smaller.

- **Meta**: Introduced [Segment Anything Model 2 (SAM 2)](https://twitter.com/adcock_brett/status/1820128497819373741), an open-source AI model for real-time object identification and tracking across video frames.

- **NVIDIA**: Project GR00T showcased a [new approach to scale robot data](https://twitter.com/adcock_brett/status/1820128520338591847) using Apple Vision Pro for humanoid teleoperation.

- **Stability AI**: Introduced [Stable Fast 3D](https://twitter.com/adcock_brett/status/1820128452772589993), generating 3D assets from a single image in 0.5 seconds.

- **Runway**: Announced that [Gen-3 Alpha](https://twitter.com/adcock_brett/status/1820128565494526267), their AI text-to-video generation model, can now create high-quality videos from images.


**AI Research and Development**

- **Direct Preference Optimization (DPO)**: [@rasbt](https://twitter.com/rasbt/status/1820096879440662972) shared a from-scratch implementation of DPO, a method for aligning large language models with user preferences.

- **MLX**: [@awnihannun](https://twitter.com/awnihannun/status/1820139615216648658) recommended using lazy loading to reduce peak memory use in MLX.

- **Modality-aware Mixture-of-Experts (MoE)**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1820092416537354247) discussed a paper from Meta AI on a modality-aware MoE architecture for pre-training mixed-modal, early-fusion language models, achieving substantial FLOPs savings.

- **Quantization**: [@osanseviero](https://twitter.com/osanseviero/status/1820124474965897466) shared five free resources for learning about quantization in AI models.

- **LangChain**: [@LangChainAI](https://twitter.com/LangChainAI/status/1820206325021946297) introduced Denser Retriever, an enterprise-grade AI retriever designed to streamline AI integration into applications.

**AI Tools and Applications**

- **FarmBot**: [@karpathy](https://twitter.com/karpathy/status/1820167525575115045) likened FarmBot to "solar panels for food," highlighting its potential to automate food production in backyards.

- **Composio**: [@llama_index](https://twitter.com/llama_index/status/1820224063174053984) mentioned Composio as a production-ready toolset for AI agents, including over 100 tools for various platforms.

- **RAG Deployment**: [@llama_index](https://twitter.com/llama_index/status/1820133457114370259) shared a comprehensive tutorial on deploying and scaling a "chat with your code" app on Google Kubernetes Engine.

- **FastHTML**: [@swyx](https://twitter.com/swyx/status/1820124350923616449) announced starting an app using FastHTML to turn AINews into a website.

**AI Ethics and Societal Impact**

- **AI Regulation**: [@fabianstelzer](https://twitter.com/fabianstelzer/status/1820090239207305335) drew parallels between current AI regulation efforts and historical restrictions on the printing press in the Ottoman Empire.

- **AI and Job Displacement**: [@svpino](https://twitter.com/svpino/status/1820168471746892247) humorously commented on the recurring prediction of AI taking over jobs.

**Memes and Humor**

- [@nearcyan](https://twitter.com/nearcyan/status/1820205826742829372) shared a meme about Mark Zuckerberg's public image change.

- [@nearcyan](https://twitter.com/nearcyan/status/1820207471849582877) joked about idolizing tech CEOs.

- [@lumpenspace](https://twitter.com/lumpenspace/status/1820233922287919263) made a humorous comment about the interpretation of diffusion as autoregression in the frequency domain.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. The Data Quality vs. Quantity Debate in LLM Training**

- **Since this is such a fast moving field, where do you think LLM will be in two years?** ([Score: 61, Comments: 101](https://reddit.com//r/LocalLLaMA/comments/1ejqqyv/since_this_is_such_a_fast_moving_field_where_do/)): In the next **two years**, the poster anticipates significant advancements in **Large Language Models (LLMs)**, particularly in **model efficiency** and **mobile deployment**. They specifically inquire about potential reductions in **parameter count** for **GPT-4**-level capabilities and the feasibility of running sophisticated LLMs on **smartphones**.
  - **Synthetic data generation** is becoming crucial as organic data runs out. The **Llama 3 paper** demonstrated successful techniques, including running generated code through **ground truth** sources to strengthen prediction abilities without model collapse.
  - Researchers anticipate growth in **multimodal domains**, with models incorporating **image/audio encoders** for better world understanding. Future developments may include **4D synthetic data** (xyzt data associated with text, video, and pictures) and improved **context handling** capabilities.
  - **Model efficiency** is expected to improve significantly. Predictions suggest **300M parameter models** outperforming today's 7B models, and the possibility of **GPT-4 level capabilities** running on smartphones within two years, enabled by advancements in **accelerator hardware** and **ASIC** development.

- **"We will run out of Data" Really?** ([Score: 61, Comments: 67](https://reddit.com//r/LocalLLaMA/comments/1ekdly9/we_will_run_out_of_data_really/)): The post challenges the notion of running out of data for training LLMs, citing that the internet contains **64 ZB** of data while current model training uses data in the **TB range**. According to **Common Crawl**, as of **June 2023**, the publicly accessible web contains **~3 billion web pages** and **~400 TB** of uncompressed data, but this represents only a fraction of total internet data, with vast amounts existing in private organizations, behind paywalls, or on sites that block crawling. The author suggests that future model training may involve purchasing large amounts of private sector data rather than using generated data, and notes that data volume will continue to increase as more countries adopt internet technologies and IoT usage expands.
  - Users argue that **freely accessible** and **financially easily accessible** data may run out, with companies realizing the value of their data and locking it down. The quality of internet data is also questioned, with some suggesting that **removing Reddit** from training data improved model performance.
  - The **64 ZB** figure represents total worldwide storage capacity, not available text data. Current models like **GPT-4** have trained on only **13 trillion tokens** (about 4 trillion unique), while estimates suggest over **200 trillion text tokens** of decent quality are publicly available.
  - A significant portion of internet data is likely **video content**, with **Netflix** accounting for **15%** of all internet traffic in **2022**. Users debate the value of this data for language modeling and suggest focusing on high-quality, curated datasets rather than raw volume.


**Theme 2. Emerging AI Technologies and Their Real-World Applications**

- **[Logical Fallacy Scoreboard](https://v.redd.it/fx4jvpkqtrgd1)** ([Score: 118, Comments: 61](https://reddit.com//r/LocalLLaMA/comments/1ekf1vl/logical_fallacy_scoreboard/)): The post proposes a **real-time logical fallacy detection system** for political debates using **Large Language Models (LLMs)**. This system would analyze debates in real-time, identify logical fallacies, and display a **"Logical Fallacy Scoreboard"** to viewers, potentially improving the quality of political discourse and helping audiences critically evaluate arguments presented by candidates.
  - Users expressed interest in a **real-time version** of the tool for live debates, with one suggesting a **"live bullshit tracker"** for all candidates. The developer plans to run the system on upcoming debates if **Trump** doesn't back out.
  - Concerns were raised about the **AI's ability to accurately detect fallacies**, with examples of inconsistencies and potential biases in the model's judgments. Some suggested using a **smaller, fine-tuned LLM** or a **BERT-based classifier** instead of large pre-trained models.
  - The project received praise for its potential to **defend democracy**, while others suggested improvements such as **tracking unresolved statements**, **categorizing lies**, and **distilling the 70B model** to 2-8B for real-time performance. Users also requested analysis of other politicians like **Biden** and **Harris**.


## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Capabilities and Advancements**

- **Flux AI demonstrates impressive text and image generation**: Multiple posts on r/StableDiffusion showcase Flux AI's ability to generate highly detailed product advertisements with accurate text placement and brand consistency. Examples include a [Tide PODS Flavor Bubble Tea ad](https://www.reddit.com/r/StableDiffusion/comments/1ekbuka/flux_adproduct_test/) and a [Hot Pockets "Sleepytime Chicken" box](https://www.reddit.com/r/StableDiffusion/comments/1ekbuka/flux_adproduct_test/). Users note Flux's superior text generation compared to other models like Midjourney.

- **OpenAI decides against watermarking ChatGPT outputs**: OpenAI [announced they won't implement watermarking](https://www.reddit.com/r/OpenAI/comments/1ekh1uv/openai_wont_watermark_chatgpt_text_because_its/) for ChatGPT-generated text, citing concerns about potential negative impacts on users. The decision sparked discussions about detection methods, academic integrity, and the balance between transparency and user protection.

**AI Ethics and Societal Impact**

- **Debate over AI's impact on jobs**: A [highly upvoted post](https://www.reddit.com/r/singularity/comments/1ek3h85/the_impact_of_ai_on_jobs/) on r/singularity discusses the potential effects of AI on employment, reflecting ongoing concerns about workforce disruption.

- **AI-powered verification and deepfakes**: A post on r/singularity highlights the [increasing sophistication of AI-generated images](https://www.reddit.com/r/singularity/comments/1ekdl7q/brace_yourself_ai_powered_verification_is_on_the/) for verification purposes, raising questions about digital identity and the challenges of distinguishing between real and AI-generated content.

**AI in Education and Development**

- **Potential of AI tutors**: A [detailed post](https://www.reddit.com/r/singularity/comments/1ejz7xa/ai_tutors_could_turn_every_child_into_a_genius/) on r/singularity explores the concept of AI tutors potentially enhancing children's learning capabilities, drawing parallels to historical examples of intensive education methods.

**AI Industry and Market Trends**

- **Ben Goertzel on the future of generative AI**: AI researcher Ben Goertzel [predicts](https://www.reddit.com/r/singularity/comments/1ejwgb0/ben_goertzel_i_dont_think_the_genai_bubble_will/) that the generative AI market will continue to grow, citing rapid development of high-value applications.


---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. LLM Advancements**

- **Llama 3 performance issues**: Users reported issues with **Llama 3's** tokenization approach, particularly with **EOS** and **BOS** token usage leading to inference challenges. Participants speculated that missing tokens in inference could lead to out-of-distribution contexts during training, prompting a reassessment of documentation.
  - Members agreed on the need for a reassessment of documentation to address these tokenization bugs, emphasizing the importance of accurate token handling.
- **Claude AI offers code fixes**: Members discussed using **Claude AI** to upload `output.json` for code fixes without file access, as outlined in [this Medium article](https://medium.com/@mbonsign/codemapper-your-ais-guide-to-understanding-code-ef2bda7f333e). Despite the potential, skepticism remained about the empirical effectiveness of this approach.
  - *Skepticism remained about the empirical effectiveness of this approach*, highlighting the need for more evidence-based results to validate its utility.


**2. Model Performance Optimization**

- **Optimizing LLM inference speed**: Suggestions for speeding up LLM inference included using **torch.compile** and comparing performance with tools like **vLLM**. The ongoing discussion highlights the interest in improving efficiency and performance for large language models.
  - Members expressed keen interest in enhancing efficiency while handling large language models, exploring various tools and techniques.
- **Mojo enhances data processing pipelines**: Discussions highlighted the potential of **Mojo** for integrating analytics with database workloads, enabling quicker data handling through JIT compilation and direct file operations.
  - Members mentioned compatibility with **PyArrow** and **Ibis**, suggesting a promising future for a robust data ecosystem within the **Mojo** framework.


**3. Fine-tuning Challenges**

- **Challenges with fine-tuning multilingual models**: Users shared their experiences with fine-tuning models like **Llama 3.1** and **Mistral** with diverse datasets, encountering output relevance issues due to possibly incorrect prompt formatting. Suggestions urged reverting to standard prompt formats to ensure proper dataset handling.
  - Participants emphasized the importance of using standard formats to avoid issues, highlighting the need for consistent prompt formatting.
- **LoRA training issues**: A user reported poor results from their **SFTTrainer** after trying to format datasets with concatenated text and labels, questioning potential misconfiguration. Clarifications pointed to correct column usage yet failed to resolve the underlying issue.
  - Clarifications pointed to correct column usage but failed to resolve the underlying issue, indicating a need for further investigation into the dataset configuration.


**4. Open-Source AI Developments**

- **Introducing DistillKit**: **Arcee AI** announced **DistillKit**, an open-source tool for distilling knowledge from larger models to create smaller, powerful models. The toolkit combines traditional training techniques with novel methods to optimize model efficiency.
  - The toolkit focuses on optimizing models to be efficient and accessible, combining traditional training techniques with novel distillation methods.
- **OpenRouter launches new models**: **OpenRouter** rolled out impressive new models, including **Llama 3.1 405B BASE** and **Mistral Nemo 12B Celeste**, which can be viewed at their [model page](https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free). The addition of **Llama 3.1 Sonar family** further expands application capabilities.
  - The new entries cater to diverse needs and adapt to community feedback for continual updates, enhancing the utility of OpenRouter's offerings.


**5. Multimodal AI Innovations**

- **CatVTON redefines virtual try-on methods**: A recent [arXiv paper](https://arxiv.org/abs/2407.15886) introduced **CatVTON**, a method that significantly reduces training costs by directly concatenating garment images. This innovation promises realistic garment transfers, revolutionizing virtual try-on tech.
  - This method eliminates the need for a ReferenceNet and additional image encoders, maintaining realistic garment transfers while reducing costs.
- **Open Interpreter speech recognition proposal**: A user proposed implementing a speech recognition method in a native language, facilitating translation between English and the local tongue. They cautioned about translation errors, dubbing it *Garbage in, Garbage out*.
  - The approach raised concerns about the potential pitfalls of translation errors, emphasizing the need for accurate input to ensure reliable output.

---

# PART 1: High level Discord summaries




## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio suffers from performance hiccups**: Multiple users reported issues with **LM Studio version 0.2.31**, particularly problems starting the application and models not loading correctly. Downgrading to earlier versions like **0.2.29** was suggested as a potential workaround.
   - Users confirmed that performance inconsistencies persist, urging the community to explore stable versions to maintain workflow.
- **Model download speeds throttle down**: Users experienced fluctuating download speeds from LM Studio's website, with reports of throttled speeds as low as **200kbps**. Suggestions include waiting or retrying downloads later due to typical AWS throttling issues.
   - The conversation underscored the need for patience during high-demand download times, further stressing the importance of checking connection stability.
- **AI wants to control your computer!**: Discussion arose about whether AI models, particularly **OpenInterpreter**, could gain vision capabilities to control PCs, pointing to the limitations of current AI understanding. Participants expressed concerns about potential unforeseen behaviors from such integrations.
   - The debate highlighted the need for careful consideration before implementing AI control mechanisms on local systems.
- **Multi-modal models create curious buzz**: Interest in multi-modal models available for **AnythingLLM** sparked discussions among users, emphasizing exploration of uncensored models. Resources like [UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard) were recommended for capability comparisons.
   - Participants stressed the importance of community-driven exploration of advanced models to enhance functional versatility.
- **Dual GPU setups draw mixed views**: Conversations about dual **4090** setups suggest benefits of splitting models across cards for improved performance, yet caution users about programming requirements for effective utilization. Concerns persist regarding the struggle of a single 4090 with larger models.
   - Members preferred discussing the balance between power and ease of use when considering multi-GPU configurations.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Diving into Hugging Face Model Features**: Users shared insights about various models on Hugging Face such as **MarionMT** for translations and the **TatoebaChallenge** for language support.
   - Concerns about model limitations and the necessity for **better documentation** sparked a broader discussion.
- **Speeding Up LLM Inference Techniques**: Optimizing LLM inference became a hot topic, with suggestions like using **torch.compile** and evaluating performance with tools such as **vLLM**.
   - Members expressed keen interest in enhancing efficiency while handling large language models.
- **CatVTON Redefines Virtual Try-On Methods**: A recent [arXiv paper](https://arxiv.org/abs/2407.15886) introduced **CatVTON**, a method that significantly reduces training costs by directly concatenating garment images.
   - This innovation promises realistic garment transfers, revolutionizing virtual try-on tech.
- **Gradient Checkpointing Implementation in Diffusers**: Recent updates now include a method for setting **gradient checkpointing** in Diffusers, allowing toggling in compatible modules.
   - This enhancement promises to optimize memory usage during model training.
- **Identifying Relationships in Tables Using NLP**: Members are exploring NLP methods to determine relationships between tables based on their column descriptions and names.
   - This inquiry suggests a need for further exploration in the realm of relational modeling with NLP.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux Model Rockets on GPUs**: Users reported image generation speeds for the **Flux model** ranging from **1.0 to 2.0 iterations per second** depending on GPU setup and model version.
   - Some managed successful image generation on lower VRAM setups using **CPU offloading** or **quantization** techniques.
- **ComfyUI Installation Hacks**: Discussions revolved around the installation of **Flux on ComfyUI**, recommending the use of the `update.py` script instead of the manager for updates.
   - Helpful installation guides were shared for newcomers to smoothly set up their environments.
- **Stable Diffusion Model Showdown**: Participants detailed the different **Stable Diffusion models**: **SD1.5**, **SDXL**, and **SD3**, noting each model's strengths while positioning **Flux** as a newcomer from the **SD3 team**.
   - The higher resource demands of **Flux** compared to traditional models were highlighted in the discussions.
- **RAM vs VRAM Showdown**: **Adequate VRAM** is critical for **Stable Diffusion performance**, with users recommending at least **16GB VRAM** for optimal results, overshadowing the need for high RAM.
   - The community advised that while RAM assists in model loading, it's not a major factor in generation speeds.
- **Animation Tools Inquiry**: Participants queried about tools like **Animatediff** for video content generation, seeking the latest updates on available methods.
   - Current suggestions highlight that while **Animatediff** is still useful, newer alternatives may be surfacing for similar tasks.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Epoch 8 Accuracy Spikes**: Members noted a surprising spike in accuracy scores after epoch 8 during training, raising questions on expected behaviors.
   - *Looks totally normal* reassured another member, indicating no cause for concern.
- **Challenges with CUDA in DRL**: Frustrations arose around creating environments in CUDA for **Deep Reinforcement Learning** with [PufferAI](https://pufferai.github.io/) suggested for better parallelism.
   - Participants stressed the complexities involved in setup, emphasizing the need for robust tooling.
- **Seeking Winter Internship in ML**: A user is urgently looking for a **winter internship** starting January 2025, focused on **ML systems** and **applied ML**.
   - The individual highlighted previous internships and ongoing open source contributions as part of their background.
- **Concerns Over AI Bubble Bursting**: Speculation about a potential **AI bubble** began circulating, with contrasting views on the long-term potential of investments.
   - Participants noted the lag time between research outcomes and profitability as a key concern.
- **Llama 3 Tokenization Issues**: Inconsistencies in **Llama 3's** tokenization approach were discussed, specifically regarding EOS and BOS token usage leading to inference challenges.
   - Participants agreed on the need for a reassessment of documentation to address these tokenization bugs.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Installation Issues Persist**: Users faced errors when installing **Unsloth** locally, particularly regarding **Python compatibility** and **PyTorch** installation, with fixes such as upgrading pip.
   - Some solved their issues by reconnecting their Colab runtime and verifying library installations.
- **Challenges with Fine-tuning Multilingual Models**: Users shared their experiences fine-tuning models like **Llama 3.1** and **Mistral** with diverse datasets, encountering output relevance issues due to possibly incorrect prompt formatting.
   - Suggestions urged reverting to standard prompt formats to ensure proper dataset handling.
- **LoRA Training Stumbles on Dataset Format**: A user reported poor results from their **SFTTrainer** after trying to format datasets with concatenated text and labels, questioning potential misconfiguration.
   - Clarifications pointed to correct column usage yet failed to resolve the underlying issue.
- **Memory Issues with Loading Large Models**: Loading the **405B Llama-3.1** model on a single GPU resulted in memory challenges, prompting users to note the necessity of multiple GPUs.
   - This highlights a common understanding that larger models demand greater computational resources for loading.
- **Self-Compressing Neural Networks Optimize Model Size**: The paper on [Self-Compressing Neural Networks](https://arxiv.org/abs/2301.13142) discusses using size in bytes in the loss function to achieve significant reductions, requiring just **3% of the bits** and **18% of the weights**.
   - This technique claims to enhance training efficiency without the need for specialized hardware.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Browsing Capabilities Under Fire**: Users reported mixed experiences with Perplexity's browsing, noting struggles in retrieving up-to-date information and strange behaviors when using the web app.
   - Conversations highlighted inconsistencies in model responses, particularly for tasks like coding queries that are essential for technical applications.
- **Breakthrough in HIV Research with Llama Antibodies**: Researchers at Georgia State University engineered a hybrid antibody that combines llama-derived nanobodies with human antibodies, neutralizing over **95%** of HIV-1 strains.
   - This hybrid approach takes advantage of the unique properties of llama nanobodies, allowing greater access to evasive virus regions.
- **Concerns Over Model Performance: Llama 3.1 vs. Expectations**: Users found that the **Llama 3.1-sonar-large-128k-online** model underperformed in Japanese tests, providing less accurate results than **GPT-3.5**.
   - This has led to calls for the development of a sonar-large model specifically optimized for Japanese to improve output quality.
- **Uber One Subscription Frustration**: A user criticized the Uber One offer as being limited to new Perplexity accounts, indicating it serves more as a user acquisition tactic than a genuine benefit.
   - Debates about account creation to capitalize on promotions raised important questions about user management in AI services.
- **API Quality Concerns with Perplexity**: Multiple users shared issues with the **Perplexity API**, mentioning unreliable responses and the return of low-quality results when querying recent news.
   - Frustrations arose over API outputs, which often appeared 'poisoned' with nonsensical content, urging a demand for improved model and API performance.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI DevDay Hits the Road**: OpenAI is taking **DevDay** on the road this fall to **San Francisco**, **London**, and **Singapore** for hands-on sessions, demos, and best practices. Attendees will have the chance to meet engineers and see how developers are building with OpenAI; more info can be found at the [DevDay website](https://openai.com/devday/).
   - Participants will interact with engineers during the events, enhancing technical understanding and community engagement.
- **AI's Global Threat Discussion**: A heated debate unfolded regarding the perception of AI as a global threat, highlighting government behavior concerning open-source AI versus superior closed-source models. *Concerns about potential risks rise in light of expanding AI capabilities*.
   - This issue was emphasized as viewpoints regarding AI's implications become increasingly polarized.
- **Insights on GPT-4o Image Generation**: Discussions revealed insights into GPT-4o's image tokenization capabilities, with the potential for images to be represented as tokens. However, practical implications and limitations remain blurry in the current implementations.
   - Mentioned resources include [a tweet from Greg Brockman](https://x.com/gdb/status/1790869434174746805) discussing the team's ongoing work in image generation with GPT-4o.
- **Prompt Engineering Hurdles**: Users reported ongoing challenges in producing high-quality output when utilizing prompts with ChatGPT, often leading to frustration. The difficulty lies in defining *what constitutes high-quality output*, complicating interactions.
   - Members shared experiences illustrating the importance of crafting clear, open-ended prompts to improve results.
- **Diversity and Bias in AI Image Generation**: Concerns arose about *racial representation* in AI-generated images, with specific prompts prompting refusals due to terms of service guidelines. Members exchanged successful strategies to ensure diverse representation by explicitly including multiple ethnic backgrounds in their prompts.
   - The discussion also revealed negative prompting effects where attempts to restrict traits produced undesirable results. Recommendations centered around crafting positive, detailed descriptions to enhance output quality.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI Engineer Demand Soars**: The need for **AI engineers** is skyrocketing as companies seek generalist skills, particularly from web developers who can integrate AI into practical applications.
   - This shift highlights the gap in high-level ML expertise, pushing web devs to fill key roles in AI projects.
- **Groq Raises $640M in Series D**: **Groq** has secured a **$640 million** Series D funding round led by BlackRock, boosting its valuation to **$2.8 billion**.
   - The funds will be directed towards expanding production capacity and enhancing the development of next-gen AI chips.
- **NVIDIA's Scraping Ethics Under Fire**: Leaked information reveals **NVIDIA**'s extensive AI data scraping, amassing 'a human lifetime' of videos daily and raising significant ethical concerns.
   - This situation has ignited debates on the legal and community implications of such aggressive data acquisition tactics.
- **Comparing Cody and Cursor**: Discussions highlighted **Cody**'s superior context-awareness compared to **Cursor**, with Cody allowing users to index repositories for relevant responses.
   - Users appreciate Cody's ease of use while finding Cursor's context management cumbersome and complicated.
- **Claude Introduces Sync Folder Feature**: Anthropic is reportedly developing a **Sync Folder** feature for Claude, enabling batch uploads from local folders for better project management.
   - This feature is anticipated to streamline the workflow and organization of files within Claude projects.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Recommendations on LLM as Judge and Dataset Generation**: A user inquired about must-reads related to current trends in **LLM as Judge** and synthetic dataset generation, focusing on instruction and preference data, highlighting the **latest two papers from WizardLM** as a starting point.
   - This discussion positions **LLM** advancements as crucial in understanding shifts in model applications.
- **Concerns Over Claude Sonnet 3.5**: Users reported issues with **Claude Sonnet 3.5**, noting its underperformance and increased error rates compared to its predecessor.
   - This raises questions about the effectiveness of recent updates and their impact on core functionalities.
- **Introduction of DistillKit**: Arcee AI announced **DistillKit**, an open-source tool for distilling knowledge from larger models to create smaller, powerful models.
   - The toolkit combines traditional training techniques with novel methods to optimize model efficiency.
- **Efficient VRAM Calculation Made Easy**: A Ruby script was shared for estimating VRAM requirements based on bits per weight and context length, available [here](https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763).
   - This tool aids users in determining maximum context and bits per weight in LLM models, streamlining VRAM calculations.
- **Innovative Mistral 7B MoEification**: The **Mistral 7B MoEified** model allows slicing individual layers into multiple experts, aiming for coherent model behavior.
   - This approach enables models to share available expert resources equally during processing.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Chatroom Gets a Fresh Look**: The **Chatroom** has been launched with local chat saving and a simplified UI, allowing better room configuration at [OpenRouter](https://openrouter.ai/chat). This revamped platform enhances user experience and accessibility.
   - *Users can explore the new features to enhance interaction within the Chatroom.*
- **OpenRouter Announces New Model Variants**: OpenRouter rolled out impressive new models, including **Llama 3.1 405B BASE** and **Mistral Nemo 12B Celeste**, which can be viewed at their [model page](https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free). The addition of **Llama 3.1 Sonar family** further expands application capabilities.
   - *The new entries cater to diverse needs and adapt to community feedback for continual updates.*
- **Mistral Models Now on Azure**: The **Mistral Large** and **Mistral Nemo** models are now accessible via [Azure](https://openrouter.ai/models/mistralai/mistral-large), enhancing their utility within a cloud environment. This move aims to provide better infrastructure and performance to users.
   - *Users can leverage Azure's capacity while accessing high-performance AI models effortlessly.*
- **Gemini Pro Undergoes Pricing Overhaul**: The pricing for **Google Gemini 1.5 Flash** will be halved on the 12th, making it more competitive against counterparts like **Yi-Vision** and **FireLLaVA**. This shift could facilitate more user engagement in automated captioning.
   - *Community feedback has been crucial in shaping this transition as users desire more economical options.*
- **Launch of Multi-AI Answers**: The [Multi-AI answer website](https://www.producthunt.com/posts/aiswers-com) has officially launched on Product Hunt with the backing of OpenRouter. Their team encourages community **upvotes and suggestions** to refine the service.
   - *Community contributions during the launch signify the importance of user engagement in the development process.*



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo speeds up data processing pipelines**: Discussions highlight the potential of **Mojo** for integrating analytics with database workloads, enabling quicker data handling through JIT compilation and direct file operations.
   - Members mentioned compatibility with **PyArrow** and **Ibis**, suggesting a promising future for a robust data ecosystem within the **Mojo** framework.
- **Elixir's confusing error handling**: Members discussed Elixir's challenge where libraries return error atoms or raise exceptions, leading to non-standardized error handling.
   - A [YouTube video](https://www.youtube.com/watch?v=Iflu9zEJipQ) featuring Chris Lattner and Lex Fridman elaborated on exceptions versus errors, providing further context.
- **Mojo debugger lacks support**: A member confirmed that the Mojo debugger currently does not work with VS Code, referencing an existing [GitHub issue](https://github.com/modularml/mojo/issues/1829) for debugging support.
   - Debugging workflows appear to be reliant on print statements, indicating a need for improved debugging tools.
- **Performance woes with Mojo SIMD**: Concerns surfaced about the performance of **Mojo's** operations on large SIMD lists, which can lag on select hardware configurations.
   - A suggestion arose that using a SIMD size fitting the CPU's handling capabilities can enhance performance.
- **Missing MAX Engine comparison documentation**: A user reported difficulty locating documentation that compared the **MAX Engine** with **PyTorch** and **ONYX**, especially across models like **ResNet**.
   - The query highlights a gap in available resources for users seeking comparison data.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Claude AI Offers Code Fixes**: Members discussed starting a new chat with **Claude AI** to upload `output.json`, enabling it to provide code fixes directly without file access, as outlined in [this Medium article](https://medium.com/@mbonsign/codemapper-your-ais-guide-to-understanding-code-ef2bda7f333e).
   - Despite the potential, skepticism remained about the empirical effectiveness of this approach.
- **Enhancing Performance through Architecture**: New architectures, particularly for **user-specific audio classification**, can significantly improve performance using strategies like **contrastive learning** to maintain user-invariant features.
   - Additionally, adapting architectures for **3D data** was discussed as a means to ensure performance under transformations.
- **State of the Art in Music Generation**: Queries about **SOTA models for music generation** included discussions around an ongoing AI music generation lawsuit, with members favoring local execution over external dependencies.
   - This conversation reflects a growing trend toward increased control in music generation applications.
- **Insights on RIAA and Labels**: The relationship between **RIAA** and music labels was scrutinized, highlighting how they influence artist payments and the industry structure, demanding more direct compensation methods.
   - Concerns surfaced about artists receiving meager royalties relative to industry profits, suggesting a push for self-promotion.
- **HDF5 for Efficient Embedding Management**: Discussions continued on the relevance of **HDF5** for loading batches from large embedding datasets, reflecting ongoing efforts to streamline data management techniques.
   - This indicates a persistent interest in efficient data usage within the AI community.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Ollama Memory Error Chronicles**: A user reported a **ValueError** indicating the model ran out of memory when invoking a retrieval chain despite low GPU usage with models like **aya** (4GB) and **nomic-embed-text** (272MB).
   - This raises questions about resource allocation and memory management in high-performance setups.
- **Mixing CPU and GPU Resources**: Discussions centered on whether **Ollama** effectively utilizes both **CPU** and **GPU** during heavy loads, with users noting the expected fallback to CPU didn't occur as anticipated.
   - Users emphasized the importance of understanding fallback mechanisms to prevent inference bottlenecks.
- **LangChain Memory Management Insights**: Insights were shared about how LangChain handles memory and object persistence, focusing on evaluating inputs for memory efficiency across sessions.
   - Queries for determining suitable information for memory storage were testing grounds for different model responses.
- **SAM 2 Fork: CPU Compatibility in Action**: A member initiated a CPU-compatible fork of the **SAM 2 model**, displaying prompted segmentation and automated mask generation, with aspirations for **GPU compatibility**.
   - Feedback regarding this endeavor is being actively solicited on the [GitHub repository](https://github.com/SauravMaheshkar/samv2).
- **Jumpstart Your AI Voice Assistant**: A tutorial video titled ['Create a custom AI Voice Assistant in 8 minutes! - Powered by ChatGPT-4o'](https://www.youtube.com/watch?v=iGX4ARuWZec) guides users through building a voice assistant for their website.
   - The creator provided a [demo link](https://smart.sista.ai) offering potential users hands-on experience before signing up for the service.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Build ReAct Agents with LlamaIndex**: You can create **ReAct agents** from scratch leveraging [LlamaIndex workflows](https://t.co/F0pPEyWJ2w) for enhanced internal logic visibility.
   - This method allows you to ‘explode’ the logic, ensuring a deeper understanding and control over agentic systems.
- **Terraform Assistant for AI Engineers**: Develop a **Terraform assistant** using LlamaIndex and Qdrant Engine aimed at aspiring AI engineers, with guidance provided [here](https://t.co/ASWNkixboK).
   - The tutorial gives practical insights and a framework for integrating AI within the DevOps space.
- **Automated Payslip Extraction with LlamaExtract**: [LlamaExtract](https://t.co/qoC9RU6Tfm) allows **high-quality RAG** on payslips through automated schema definition and metadata extraction.
   - This process significantly enhances data handling capabilities for payroll documents.
- **Scaling RAG Applications Tutorial**: Benito Martin outlines how to deploy and scale your chat applications on Google Kubernetes, emphasizing practical strategies [here](https://t.co/ROsGNjhKEM).
   - This resource addresses content scarcity on productionizing RAG applications in detail.
- **Innovative GraphRAG Integration**: The integration of **GraphRAG** with **LlamaIndex** enhances **intelligent question answering** capabilities, as discussed in a [Medium article](https://medium.com/ai-advances/graphrag-with-llamaindex-unleashing-the-power-of-knowledge-graphs-for-intelligent-question-ea177a14623e).
   - This integration leverages **knowledge graphs** to improve context and accuracy of AI responses.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Bay Area Events Generate Buzz**: Members expressed a desire for updates on upcoming events in the **Bay Area**, with some noting personal absences.
   - The ongoing interest hints at a need for better communication around local gatherings.
- **Noam Shazeer Lacks Recognition**: Discussion arose around the absence of a **Wikipedia page** for **Noam Shazeer**, a key figure at Google since 2002.
   - Members reflected on *Wikipedia can be silly*, highlighting the ironic oversight of impactful professionals.
- **Skepticism of 30 Under 30 Awards Validity**: A member critiqued the **30 Under 30** awards as catering more to insiders than genuine merit, suggesting *special types of people* seek such validation.
   - This struck a chord among members who noted the often superficial recognition those awards bestow.
- **Debate on Synthetic Data Using Nemotron**: A heated discussion emerged about redoing **synthetic data** leveraging **Nemotron** for fine-tuning **Olmo** models.
   - Concerns were raised over the potential hijacking of the **Nemotron** name and criticisms of AI2's trajectory.
- **KTO Outperforms DPO in Noisy Environments**: The **Neural Notes interview** discussed KTO's strength over DPO when handling noisy data, suggesting significant performance gains.
   - Adaptations from **UCLA** reported KTO's success against DPO with human preferences indicating a **70-30%** edge.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Synthetic Datasets Spark Controversy**: Members debated the effectiveness of **synthetic datasets** versus original ones, noting they can accelerate training but may risk misalignment and lower quality.
   - Concerns were voiced about biases, prompting calls for more intentional dataset creation to avoid generating a billion useless images.
- **FLUX Model Performance Divides Opinions**: Users shared mixed views on the **FLUX** model's ability to generate artistic outputs; some praised its capability while others were disappointed.
   - Discussion pointed out that better parameter settings could enhance its performance, yet skepticism remained regarding its overall utility for artistry.
- **CIFAR-10 Validation Accuracy Hits 80%**: **80% validation accuracy** achieved on the CIFAR-10 dataset using only **36k parameters**, treating real and imaginary components of complex parameters as separate.
   - Tweaks to architecture and dropout implementation resolved previous issues, resulting in a more robust model with nearly eliminated overfitting.
- **Ethical Concerns in Model Training**: Discussions heated up around the ethical implications of training on copyrighted images, sparking anxiety over **copyright laundering** in synthetic datasets.
   - Some proposed that while synthetic data has advantages, stricter scrutiny may impose regulations on training practices within the community.
- **Stable Diffusion Dataset Availability Questioned**: A user expressed frustration over the unavailability of a **Stable Diffusion dataset**, which hindered their progress.
   - Peers clarified that the dataset isn't strictly necessary for utilizing Stable Diffusion, offering alternative solutions.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Adding a Coding Agent to ChatmanGPT Stack**: A member is seeking **recommendations for a coding agent** to add to the ChatmanGPT Stack, with **Agent Zero** suggested as a potential choice.
   - *Looking for an effective addition to enhance coding interactions.*
- **Golden-Retriever Paper Overview**: A shared link to the paper on **Golden-Retriever** details how it efficiently navigates industrial knowledge bases by improving on traditional LLM fine-tuning challenges, particularly with a **reflection-based question augmentation** step.
   - This method enhances retrieval accuracy by clarifying jargon and context prior to document retrieval. Read more in the [Golden-Retriever Paper](https://arxiv.org/abs/2408.00798).
- **Livecoding in the Voice Lounge**: A member announced their return and mentioned **livecoding** sessions in the Voice Lounge, signaling a collaborative coding effort ahead.
   - *Members look forward to joining forces in this engaging setup.*
- **AI NPCs Respond and Patrol**: Plans are underway for developing **AI characters** in a C++ game using the **Oobabooga API** for player interaction, focusing on patrolling and response functions.
   - The necessary components include modifying the **'world' node** and extending the NPC class.
- **Exporting Discord Chats Made Easy**: A user successfully **exported Discord channels** to HTML and JSON using the **DiscordChatExporter tool**, generating **463 thread files**.
   - This tool streamlines chat organization, making it easier for future reference. Check out the [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter/releases).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter runs on local LLM!**: A user successfully integrated **Open Interpreter** with a local LLM using LM Studio as a server, gaining access to the OI system prompt.
   - They found the integration both interesting and informative, paving the way for local deployments.
- **Troubleshooting Hugging Face API integration**: Users faced challenges while setting up the Hugging Face API integration in **Open Interpreter**, encountering various errors despite following the documentation.
   - One user expressed gratitude for support, hoping for a resolution to their integration issues.
- **Executing screenshot commands becomes a chore**: Concerns arose as users questioned why **Open Interpreter** generates extensive code instead of executing the screenshot command directly.
   - A workaround using the 'screencapture' command confirmed functionality, alleviating some frustrations.
- **Speech recognition in multiple languages proposed**: A user proposed implementing a speech recognition method in a native language, facilitating translation between English and the local tongue.
   - They cautioned about translation errors, dubbing it *Garbage in, Garbage out*.
- **Electra AI shows promise for AI on Linux**: A member unveiled **Electra AI**, a Linux distro built with AI capabilities that are free for use, highlighting its potential for integration.
   - They noted that Electra AI offers three flavors: **Lindoz**, **Max**, and **Shift**—all available for free.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Support for CORS Issues**: To address the **CORS problems** on the billing page, community members suggested emailing [support@cohere.com](mailto:support@cohere.com) for help, including organization's details in the inquiry.
   - This support method aims to resolve issues that have hindered user payments for services.
- **GenAI Bootcamp Seeks Cohere Insights**: Andrew Brown is exploring the potential of **Cohere** for a free **GenAI Bootcamp**, which seeks to reach **50K participants** this year.
   - He highlighted the need for insights beyond documentation, especially regarding **Cohere's cloud-agnostic capabilities**.
- **Benchmarking Models with Consistency**: A member inquired about keeping the **validation subset** consistent while benchmarking multiple models, emphasizing the importance of controlled comparisons.
   - Discussions reinforced the necessity of maintaining consistent validation sets to enhance the accuracy of comparisons.
- **Rerank Activation on Azure Models**: Cohere announced the availability of **Rerank** for **Azure Models**, with integration potential for the **RAG app**, as detailed in this [blog post](https://cohere.com/blog/introducing-rerank-3-on-microsoft-azure-ai).
   - Members showed interest in updating their toolkit to utilize Rerank for Azure users.
- **Clarification on Cohere Model Confusion**: A user who paid for the **Cohere API** found only the **Coral model** available and faced confusion regarding accessing the **Command R** model.
   - In response, a member clarified that **Coral** is indeed a version of **Command R+** to ease the user’s concerns.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 0.9.2 introduces exciting features**: The recent release of **tinygrad 0.9.2** brings notable updates like **faster gemv**, **kernel timing**, and improvements with **CapturedJit**.
   - Additional discussions included enhancements for **ResNet** and advanced indexing techniques, marking a significant step for performance optimization.
- **Evaluating tinygrad on Aurora supercomputer**: Members discussed the feasibility of running **tinygrad** on the **Aurora** supercomputer, stressing concerns over compatibility with Intel GPUs.
   - While **OpenCL** support exists, there were queries regarding performance constraints and efficiency on this platform.
- **CUDA performance disappoints compared to CLANG**: Members noted that tests in **CUDA** run slower than in **CLANG**, prompting an investigation into possible efficiency issues.
   - This discrepancy raises important questions about the execution integrity of CUDA, especially in **test_winograd.py**.
- **Custom tensor kernels spark discussion**: A user shared interest in executing custom kernels on tensors, referencing a [GitHub file](https://github.com/tinygrad/tinygrad/blob/da61dea1b2ca886b3de07e309efde2a78ac5682a/test/test_custom_function.py#L42-L43) for guidance.
   - This reflects ongoing enhancements in tensor operations within tinygrad, showcasing community engagement in practical implementation.
- **Bounties incentivize tinygrad feature contributions**: The community has opened discussions on **bounties** for tinygrad improvements, such as **fast sharded llama** and optimizations for **AMX**.
   - This initiative encourages developers to actively engage in enhancing the framework, aiming for broader functionality.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **PPO Training Recipe Now Live**: The team has introduced an end-to-end **PPO training recipe** to integrate RLHF with Torchtune, as noted in the [GitHub pull request](https://github.com/pytorch/torchtune/pull/1005).
   - *Check it out and try it out!*
- **Qwen2 Model Support Added**: **Qwen2 model support** is now included in training recipes, with the **7B model** available in the [GitHub pull request](https://github.com/pytorch/torchtune/pull/1143).
   - Expect the upcoming **1.5B** and **0.5B** versions to arrive soon!
- **LLAMA 3 Tangles with Generation**: Users successfully ran the **LLAMA 3 8B INSTRUCT model** with a custom configuration, generating a time query in **27.19 seconds** at **12.25 tokens/sec**, utilizing **20.62 GB** of memory.
   - However, there's a concern about **text repeating 10 times**, and a [pull request](https://github.com/pytorch/torchtune/pull/1211) is under review to address unexpected ending tokens.
- **Debugging Mode Call for LLAMA 3**: Concerns arose regarding the absence of a debugging mode that displays **all tokens** in the LLAMA 3 generation output.
   - A member suggested that adding a parameter to the generation script could resolve this issue.
- **Model Blurbs Maintenance Anxiety**: Members expressed concerns about keeping updated **model blurbs**, fearing the maintenance could be overwhelming.
   - One proposed using a **snapshot from a model card** or whitepaper as a minimal blurb solution.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **bitsandbytes Installation for ROCm Simplified**: [A recent pull request](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1299) enables packaging wheels for **bitsandbytes** on **ROCm**, streamlining the installation process for users.
   - This PR updates the compilation process for **ROCm 6.1** to support the latest **Instinct** and **Radeon** GPUs.
- **Building an AI Nutritionist Needs Datasets**: A member is developing an **AI Nutritionist** and considers fine-tuning **GPT-4o mini** but seeks suitable nutrition datasets like the [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets.html).
   - Recommendations include potential dataset compilation from **FNDDS**, though it's unclear if it's available on **Hugging Face**.
- **Searching for FFT and Baseline Tests**: A member expressed interest in finding **FFT** or **LORA/QLORA** for experimentation with a **27b model**, mentioning good results with a **9b model** but challenges with the larger one.
   - *Caseus* suggested a **QLORA** version for **Gemma 2 27b** might work with adjustments to the learning rate and the latest **flash attention**.
- **Inquiry about L40S GPUs Performance**: A member asked if anyone has trained or served models on **L40S GPUs**, seeking insights about their performance.
   - This inquiry highlights interest in the efficiency and capabilities of **L40S GPUs** for AI model training.
- **Discussion on DPO Alternatives in AI Training**: A member questioned whether **DPO** remains the best approach in AI training, suggesting alternatives like **orpo**, **simpo**, or **kto** might be superior.
   - This led to an exchange of differing opinions on the effectiveness of various methods in AI model training.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Triton Conference Registration Now Open!**: Registration for the **Triton Conference** on **September 17, 2024** at **Meta Campus, Fremont CA** is now open! Sign up via [this Google Form](https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform) to secure your spot.
   - Attendance is **free**, but spots are **limited**, so early registration is encouraged.
- **Information Required for Registration**: Participants must provide their **email**, **name**, **affiliation**, and **role** to register. Additional optional questions include dietary preferences like **vegetarian**, **vegan**, **kosher**, and **gluten-free**.
   - *Pro tip*: Capture what attendees hope to take away from the conference!
- **Google Sign-In for Conference Registration**: Attendees are prompted to [sign in to Google](https://accounts.google.com/AccountChooser?continue=https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform&service=wise) to save their progress on the registration form. All responses will be emailed to the participant's provided address.
   - Don't forget: participants should never submit passwords through Google Forms to ensure security.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile boosts offline LLM access**: The core maintainer of **Llamafile** reports significant advancements in enabling **offline, accessible LLMs** within a single file.
   - This initiative improves accessibility and simplifies user interactions with large language models.
- **Community excited about August projects**: A vibrant discussion has ignited around ongoing projects for **August**, encouraging community members to showcase their work.
   - Participants have the chance to engage and share their contributions within the Mozilla AI space.
- **sqlite-vec release party on the horizon**: An upcoming release party for **sqlite-vec** will allow attendees to discuss features and engage with the core maintainer.
   - Demos and discussions are set to unfold, creating opportunities for rich exchanges on the latest developments.
- **Exciting Machine Learning Paper Talks scheduled**: Upcoming talks featuring topics such as **Communicative Agents** and **Extended Mind Transformers** will include distinguished speakers.
   - These events promise valuable insights into cutting-edge research and collaborative opportunities in machine learning.
- **Local AI AMA promises open-source insights**: A scheduled **Local AI** AMA with the core maintainer will offer insights into this self-hostable alternative to OpenAI.
   - This session invites attendees to explore Local AI's capabilities and directly address their queries.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1269011615114854543)** (708 messages🔥🔥🔥): 

> - `LM Studio performance issues`
> - `Model downloading speeds`
> - `AI interaction with local systems`
> - `Multi-modal models in AnythingLLM`
> - `RAM and VRAM utilization` 


- **LM Studio performance issues**: Multiple users reported issues with LM Studio version 0.2.31, including problems starting the application and models not loading correctly.
   - Downgrading to earlier versions, such as 0.2.29, was suggested as a potential workaround for these issues.
- **Model downloading speeds**: Users experienced fluctuating download speeds from LM Studio's website, with some noting throttled speeds as low as 200kbps.
   - It was suggested to wait or retry downloads later due to AWS throttling, which is not uncommon for shared resources.
- **AI interaction with local systems**: Discussion arose about whether AI models, specifically LLMs like OpenInterpreter, could gain vision capabilities to control PCs.
   - It was noted that such capabilities may prompt unforeseen behavior from the models, illustrating the limitations of current AI understanding.
- **Multi-modal models in AnythingLLM**: Users expressed interest in the capabilities of multi-modal models and their availability for use in the AnythingLLM framework.
   - Recommendations included exploring uncensored models and checking resources like UGI-Leaderboard for comparisons.
- **RAM and VRAM utilization**: It was confirmed that users can combine RAM and VRAM for running larger models, with settings configurable in LM Studio.
   - The default setting allows the application to manage the use of RAM and VRAM efficiently for optimal performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://matheval.ai/en/">MathEval</a>: MathEval is a benchmark dedicated to the holistic evaluation on mathematical capacities of LLMs, consisting of 22 evaluation datasets in various mathematical fields and nearly 30,000 math problems. Th...</li><li><a href="https://radxa.com/products/rock5/5itx/">Radxa ROCK 5 ITX</a>: Your 8K ARM Personal Computer</li><li><a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - a Hugging Face Space by DontPlanToEnd</a>: no description found</li><li><a href="https://play.google.com/store/apps/details?id=net.hamandmore.crosstalk&hl=ln&gl=US&pli=1">Crosstalk Multi-LLM AI Chat – Applications sur Google Play</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/TinyStories-656K-GGUF">mradermacher/TinyStories-656K-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/legraphista/internlm2_5-20b-chat-IMat-GGUF">legraphista/internlm2_5-20b-chat-IMat-GGUF · Hugging Face</a>: no description found</li><li><a href="https://releases.lmstudio.ai/windows/0.2.27/latest/LM-Studio-0.2.27-Setup.exe">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model&#39;s capability via loss or benchmarks, we estimate the n...</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/discussions/6">lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF · Update to Models</a>: no description found</li><li><a href="https://play.google.com/store/apps/details?id=us.valkon.privateai&hl=fr&gl=US">Private AI – Applications sur Google Play</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/tree/main">lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF at main</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=CZbhUfmTXaE">Gemini-1.5 Pro Experiment (0801): NEW Updates to Gemini BEATS Claude &amp; GPT-4O (Fully Tested)</a>: Join this channel to get access to perks: https://www.youtube.com/@aicodeking/joinIn this video, I&#39;ll be talking about the new Gemini-1.5 Pro Experiment (080...</li><li><a href="https://tenor.com/view/what-year-is-it-jumanji-forgotten-gif-15305928">What Year Is It Jumanji GIF - What Year Is It Jumanji Forgotten - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://releases.lmstudio.ai/windows/0.2.29/1/LM-Studio-0.2.29-Setup.exe">no title found</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://aistudio.google.com/">no title found</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/search?q=repo%3ANVIDIA%2Fcuda-samples%20int4&type=code">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g">[1hr Talk] Intro to Large Language Models</a>: This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1269006868236468244)** (138 messages🔥🔥): 

> - `Dual GPU setups and performance`
> - `NVIDIA GPU comparisons`
> - `NPU capabilities in laptops`
> - `Tesla M10 usability`
> - `Upcoming hardware releases` 


- **Dual 4090 setups versus single GPUs**: Discussions around dual GPU setups indicate multi-GPU configurations may split models across cards, impacting performance and speed.
   - Members voiced concerns that a single 4090 might struggle with larger models, while dual setups require programming for effective use.
- **NPU in Laptops: The Future?**: The conversation explored the integration of NPUs in laptops, with mixed opinions on their performance benefits compared to traditional GPUs.
   - Some participants argued that offloading tasks to NPUs could enhance efficiency, particularly in power-limited environments.
- **Tesla M10: Is it Worth It?**: Several members warned against purchasing older GPUs like the NVIDIA Tesla M10 due to their inefficiency and obsolescence.
   - There were suggestions to consider more recent models like the P40 if users pursue legacy hardware.
- **Performance of GPUs in LLM inference**: Users reported varying experiences using integrated GPUs and discussed their performance metrics, notably with Llama 3.1 models.
   - Inferences indicate performance peaks are often tied not only to GPU power but memory management and context window settings.
- **Future Hardware: Anticipations and Upgrades**: Participants expressed excitement about upcoming hardware, particularly the Studio M4 Ultra and Blackwell architecture next year.
   - Discussions highlighted the potential benefits of upgrading to a 4090 for deep learning tasks while waiting for next-gen releases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gyazo.com/823a6d9154a6e84d93b2352884b3b9e7">Gyazo</a>:  </li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/guaton-computadora-enojado-computer-rage-gif-14480338">Guaton Computadora GIF - Guaton Computadora Enojado - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18ocu6q/couuld_llama2_70b_be_run_on_a_tesla_m10/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.techpowerup.com/324271/amd-strix-halo-a-large-rectangular-bga-package-the-size-of-an-lga1700-processor">AMD &quot;Strix Halo&quot; a Large Rectangular BGA Package the Size of an LGA1700 Processor</a>: Apparently the AMD &quot;Strix Halo&quot; processor is real, and it&#039;s large. The chip is designed to square off against the likes of the Apple M3 Pro and M3 Max, in letting ultraportable notebook...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1269012429023481988)** (810 messages🔥🔥🔥): 

> - `Hugging Face model features`
> - `LLM inference optimization`
> - `Open source contribution guidance`
> - `Fine-tuning models with PEFT`
> - `Using CUDA graphs in PyTorch` 


- **Discussion on Hugging Face model features**: Users discussed various models available on Hugging Face, including the MarionMT model for translations and the potential of the TatoebaChallenge repo for language support.
   - Concerns were raised about the limitations of certain models and the need for better documentation and implementation examples.
- **Optimizing LLM inference speed**: Several suggestions were made for speeding up LLM inference, including using torch.compile and comparing performance between vLLM, TGI, and LMdeploy.
   - The ongoing discussion highlights the interest in improving efficiency and performance when working with large language models.
- **Guidance on open source contributions**: A user sought advice on making open source contributions and found discussions helpful in reconsidering their approach and motivations.
   - Links to relevant blog posts and tutorials were shared to help new contributors get started.
- **Fine-tuning models with PEFT**: A user shared their fine-tuning experience with the Llama2 model, encountering some issues with model pushing, leading to discussions about best practices in the process.
   - Best practices were suggested, including the correct usage of pushing to the hub and managing training configurations.
- **Using CUDA graphs in PyTorch**: The discussion explored how CUDA graphs can optimize PyTorch models by reducing the overhead associated with launching GPU operations.
   - Users expressed interest in improving performance and noted that proper usage of libraries like torch are crucial for effective graph implementations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blogs.nlmatics.com/bert/math/2021/08/06/Teaching-BERT-to-Solve-Word-Problems.html">Teaching BERT to Solve Word Problems</a>: Teaching Bert to Solve Word Problems</li><li><a href="https://huggingface.co/docs/hub/repositories-recommendations#sharing-large-datasets-on-the-hub">Repository limitations and recommendations</a>: no description found</li><li><a href="https://huggingface.co/chatpdflocal/llama3.1-8b-gguf">chatpdflocal/llama3.1-8b-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic/Minitron/settings?clone=true">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/hugging-quants">hugging-quants (Hugging Quants)</a>: no description found</li><li><a href="https://huggingface.co/mlabonne/Llama-3.1-70B-Instruct-lorablated">mlabonne/Llama-3.1-70B-Instruct-lorablated · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/compile2011/W-finetune/discussions/">compile2011/W-finetune · Discussions</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=5nY_cy8zcO4">Don&#39;t Contribute to Open Source</a>: You heard me right. I don&#39;t think you should contribute to open source. Unless...KEYWORDS: GITHUB OPEN SOURCE CODING DEVELOPING PROGRAMMING LEARNING TO CODE ...</li><li><a href="https://tenor.com/view/helicopter-baguette-gif-20550621">Helicopter Baguette GIF - Helicopter Baguette - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/mervenoyann/status/1819289510124863774">Tweet from merve (@mervenoyann)</a>: OWLSAM2: text-promptable SAM2 🦉  Marrying cutting-edge zero-shot object detector OWLv2 🤝 mask generator SAM2 (small)  Zero-shot segmentation with insane precision ⛵️</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://huggingface.co/or4cl3ai/SquanchNastyAI">or4cl3ai/SquanchNastyAI · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/or4cl3ai/IntelliChat">or4cl3ai/IntelliChat · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/or4cl3ai/SoundSlayerAI">or4cl3ai/SoundSlayerAI · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/or4cl3ai/A-os43-v1">or4cl3ai/A-os43-v1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/Voxel51/DataCentricVisualAIChallenge">DataCentricVisualAIChallenge - a Hugging Face Space by Voxel51</a>: no description found</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/tasks/question-answering">What is Question Answering? - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic/Minitron">Minitron - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/">Accelerating PyTorch with CUDA Graphs</a>: Today, we are pleased to announce a new advanced CUDA feature, CUDA Graphs, has been brought to PyTorch. Modern DL frameworks have complicated software stacks that incur significant overheads associat...</li><li><a href="https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k">Papers with Code - GSM8K Benchmark (Arithmetic Reasoning)</a>: The current state-of-the-art on GSM8K is GPT-4 DUP. See a full comparison of 152 papers with code.</li><li><a href="https://huggingface.co/datasets/stanfordnlp/imdb/tree/main">stanfordnlp/imdb at main</a>: no description found</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">RAG chatbot using llama3</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic/Minitron/tree/main">Tonic/Minitron at main</a>: no description found</li><li><a href="https://huggingface.co/models?pipeline_tag=image-to-text&sort=trending&search=glucose">Models - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/argilla-chatbot">How we leveraged distilabel to create an Argilla 2.0 Chatbot</a>: no description found</li><li><a href="https://huggingface.co/blog/gemma-peft">Fine-Tuning Gemma Models in Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md">transformers/CONTRIBUTING.md at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/pygfx/wgpu-py/pull/547/files#diff-d9f01e8e8bedc3ca54c8b49d">[WIP] update to wgpu-native 22.1 by Vipitis · Pull Request #547 · pygfx/wgpu-py</a>: I am really excited for better error handling, compilation info and glsl const built-ins, so I already started this. Feel free to cherry pick my changes or commit into this branch if it helps. I go...</li><li><a href="https://youtu.be/-YpwsdRKt8Q>">SpiegelMining – Reverse Engineering von Spiegel-Online (33c3)</a>: Wer denkt, Vorratsdatenspeicherungen und „Big Data“ sind harmlos, der kriegt hier eine Demo an Spiegel-Online.Seit Mitte 2014 hat David fast 100.000 Artikel ...</li><li><a href="https://blog.arcee.ai/announcing-distillkit/?utm_campaign=content&utm_source=Marktechpost&utm_medium=Blog&utm_term=Knowledge%20Distillation&utm_content=Blog%201">Announcing DistillKit for creating &amp; distributing SLMs</a>: First, Arcee AI revolutionized Small Language Models (SLMs) with Model Merging and the open-source repo MergeKit. Today we bring you another leap forward in the creation and distribution of SLMs with ...</li><li><a href="https://github.com/arcee-ai/DistillKit?ref=blog.arcee.ai">GitHub - arcee-ai/DistillKit at blog.arcee.ai</a>: An Open Source Toolkit For LLM Distillation. Contribute to arcee-ai/DistillKit development by creating an account on GitHub.</li><li><a href="https://github.com/pygfx/wgpu-py/pull/547">[WIP] update to wgpu-native 22.1 by Vipitis · Pull Request #547 · pygfx/wgpu-py</a>: I am really excited for better error handling, compilation info and glsl const built-ins, so I already started this. Feel free to cherry pick my changes or commit into this branch if it helps. I go...</li><li><a href="https://github.com/pygfx/wgpu-py/pull/547/files#diff-d9f01e8e8bedc3ca54c8b49dc3d0b43c504dc488b58d4e2b4b2a03eeef29dd40>">[WIP] update to wgpu-native 22.1 by Vipitis · Pull Request #547 · pygfx/wgpu-py</a>: I am really excited for better error handling, compilation info and glsl const built-ins, so I already started this. Feel free to cherry pick my changes or commit into this branch if it helps. I go...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1269273708711776267)** (3 messages): 

> - `LLM Inference Optimization`
> - `Curriculum-based AI Approach` 


- **Exploring LLM Inference Optimization Techniques**: A noteworthy article discusses techniques to optimize **LLM inference** for better throughput and GPU utilization while decreasing latency, showcasing the challenges faced with large models.
   - It highlights that stacking transformer layers leads to **better accuracies** and elaborates on the costs associated with **retrieval-augmented generation** (RAG) pipelines which demand substantial processing power.
- **Curriculum-Based Approach in AI**: A paper outlines the limitations of current **AI systems** in reasoning and adaptability, emphasizing the need for a robust curriculum-based approach to enhance explainability and causal understanding.
   - The author emphasizes that while AI excels in **pattern recognition**, it struggles in complex reasoning environments, which fundamentally limits its transformative potential.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nonartificialintelligence.blogspot.com/2024/08/from-data-to-understanding-curriculum.html">From Data to Understanding: A Curriculum-Based Approach to Nurturing AI Reasoning</a>: no description found</li><li><a href="https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/">Mastering LLM Techniques: Inference Optimization | NVIDIA Technical Blog</a>: Stacking transformer layers to create large models results in better accuracies, few&#x2d;shot learning capabilities, and even near&#x2d;human emergent abilities on a wide range of language tasks.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1269302599639629884)** (10 messages🔥): 

> - `CatVTON Virtual Try-On Model`
> - `GeekyGhost Extensions`
> - `SweetHug AI Chatbot`
> - `Joy Captioning for ASCII Art`
> - `LLM Deployment with Optimized Inference` 


- **CatVTON redefines virtual try-on methods**: A recent [arXiv paper](https://arxiv.org/abs/2407.15886) introduces **CatVTON**, an efficient virtual try-on diffusion model that eliminates the need for a ReferenceNet and additional image encoders by concatenating garment images directly during processing.
   - This innovation reduces training costs while maintaining realistic garment transfers to target persons.
- **GeekyGhost unveils Automatic1111 extension**: A member shared their creation of the [Automatic1111 extension on GitHub](https://github.com/GeekyGhost/Automatic1111-Geeky-Remb.git), which is a port of their comfyUI geely remb tool.
   - They also introduced another project for a web UI using Gradio, further showcasing their work in the community.
- **Discover SweetHug AI chatbot**: Another user highlighted the capabilities of [SweetHug AI](https://sweethugai.com), an AI character platform that offers users a chance to explore chats with AI girlfriends as they share their thoughts and fantasies.
   - The service is handled by Ally AI Pte. Ltd. and includes features like NSFW chats and an affiliate program.
- **Joy Captioning excels in ASCII art**: A member pointed out the [Joy Captioning space](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha), which successfully captions ASCII art instead of misinterpreting it as technical diagrams.
   - They expressed their excitement over discovering a tool that accurately reflects artistic expressions in text format.
- **Inquiry on LLM deployment optimization**: A user sought insights regarding **LLM deployment** with optimized inference, highlighting an interest in efficiency for large language models.
   - This sparked curiosity about advancements and practices in this area within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.15886">CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models</a>: Virtual try-on methods based on diffusion models achieve realistic try-on effects but often replicate the backbone network as a ReferenceNet or use additional image encoders to process condition input...</li><li><a href="https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha">Joy Caption Pre Alpha - a Hugging Face Space by fancyfeast</a>: no description found</li><li><a href="https://sweethugai.com">SweetHug AI: Free Chat With Your AI Girlfriends - No Limits</a>: no description found</li><li><a href="https://github.com/GeekyGhost/Automatic1111-Geeky-Remb.git">GitHub - GeekyGhost/Automatic1111-Geeky-Remb: Automatic1111 port of my comfyUI geely remb tool</a>: Automatic1111 port of my comfyUI geely remb tool. Contribute to GeekyGhost/Automatic1111-Geeky-Remb development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1269193770818338878)** (11 messages🔥): 

> - `Testcontainers`
> - `Rob's Instagram Interaction`
> - `Self-Supervised Learning in Dense Prediction`
> - `AI Research Agent Documentation` 


- **Explore Testcontainers for AI Development**: A member shared their discovery of [Testcontainers](https://huggingface.co/blog/Tonic/localai-testcontainers), emphasizing its potential for development and serving AI applications.
   - They also mentioned their new hobby of contributing to the Docker Testcontainers project, encouraging others to join in the fun.
- **Rob Gains New Powers on Instagram**: Through a vision model, a member successfully enabled Rob to read and react to his Instagram comments, showcasing his growing capabilities.
   - Another member humorously noted Rob's increasing powers, suggesting a TikTok live session could be lucrative.
- **Self-Supervised Learning Revolutionizes Dense Prediction**: A member highlighted the advances in self-supervised learning methods, specifically in boosting performance for dense prediction tasks like object detection and segmentation.
   - They provided a link to an informative post discussing the challenges faced by traditional SSL methods in these applications.
- **AI Research Agent Documentation Launch**: A member shared resources including [documentation](https://vtempest.github.io/ai-research-agent/docs/) and demos for an AI Research library they developed.
   - They promoted features like search capabilities, text extraction, and keyphrase topic extraction while inviting discussions on integration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dev.to/tonic/dockers-testcontainers-are-great-42cl">no title found</a>: no description found</li><li><a href="https://vtempest.github.io/ai-research-agent/docs/">ai-research-agent Home</a>: no description found</li><li><a href="https://www.lightly.ai/post/using-self-supervised-learning-for-dense-prediction-tasks">Using Self-Supervised Learning for Dense Prediction Tasks</a>: Overview of Self-Supervised Learning methods for dense prediction tasks such as object detection, instance segmentation, and semantic segmentation</li><li><a href="https://huggingface.co/blog/Tonic/localai-testcontainers">Local AI with Docker&#39;s Testcontainers</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1269057874710237356)** (29 messages🔥): 

> - `Group Focus for Learning`
> - `Hackathon Collaboration`
> - `SEE-2-SOUND Presentation`
> - `Recording of Presentations`
> - `Linear Algebra Article` 


- **Choosing a Main Focus for Learning**: Members discussed the importance of selecting a common focus, like a course or project, to enhance learning and accountability within the group.
   - *Teaming up* for hands-on challenges like hackathons can foster collaboration, but it's crucial that participants have similar skill levels to prevent uneven work distribution.
- **SEE-2-SOUND Revolutionizes Spatial Audio**: A presentation was held on SEE-2-SOUND, a zero-shot framework that generates spatial audio from visual content without needing extensive prior training.
   - This innovative approach decomposes the process into identifying key visual aspects and integrating them into high-quality spatial audio, posing exciting implications for immersive content.
- **Availability of Session Recordings**: A member inquired about the availability of a recording from a recent presentation, which experienced technical difficulties.
   - The presenter confirmed that there will be an edited version of the session distributed later.
- **Introduction of New Members and Resources**: New members inquired about group resources, such as a calendar for events, to stay engaged in activities.
   - Members responded that event scheduling is typically managed through messaging platforms with updates posted regularly.
- **Sharing Knowledge through Articles**: A member shared a new article about linear algebra on Medium, focusing on linear combinations and spans of vectors.
   - This article serves as a resource for members keen on strengthening their understanding of linear algebra fundamentals.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06612">SEE-2-SOUND: Zero-Shot Spatial Environment-to-Spatial Sound</a>: Generating combined visual and auditory sensory experiences is critical for the consumption of immersive content. Recent advances in neural generative models have enabled the creation of high-resoluti...</li><li><a href="https://drexel.zoom.us/j/82015537439">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://medium.com/@amitsubhashchejara/linear-algebra-part-2-linear-combination-and-span-d5fe65ef0e8f">Linear Algebra (Part-2): Linear Combination And Span</a>: Learn about the linear combination and the span of vectors.</li><li><a href="https://drexel.zoom.us/j/820">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1269192308377784392)** (8 messages🔥): 

> - `Computer Vision Course Assignments`
> - `SF-LLaVA Paper Discussion`
> - `CV Project Suggestions`
> - `3D Object Orientation Modeling`
> - `Time-tagged Outdoor Image Datasets` 


- **Clarity Needed on Computer Vision Course Assignments**: Users expressed confusion about the assignment components of their **computer vision course**, seeking clarity on requirements.
   - One participant suggested collaborating over voice chat to discuss relevant materials.
- **Collaborative Discussion on SF-LLaVA**: One user proposed a voice chat to review the **SF-LLaVA** paper, encouraging others to join the discussion.
   - This initiative reflects the community's willingness to support each other in understanding academic resources.
- **Suggestions for Quick CV Projects**: Participants shared ideas for **CV projects**, including **binocular depth estimation** as a viable option that can be completed in a week.
   - This discussion highlights a proactive approach for learners to engage in practical applications of their knowledge.
- **Modeling 3D Object Orientation**: A user inquired about methods to create a model capable of determining the **direction a 3D object is facing**.
   - This question underscores the interest in advancing skills in spatial understanding within computer vision.
- **Finding Time-tagged Outdoor Image Datasets**: A user is searching for a good dataset with **outdoor images tagged by time of day**, expressing difficulties with existing options like **MIRFLIKR**.
   - This inquiry suggests a demand for high-quality datasets that facilitate specific research needs in computer vision.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1269936468449562739)** (4 messages): 

> - `Dependency Issues with Chaquopy`
> - `Finding Related Tables with NLP` 


- **Dependency Issues with Chaquopy for Qwen2-0.5B-Instruct**: A member is facing persistent *dependency conflicts* while developing an Android app using Chaquopy with Python, specifically relating to **transformers** and **tokenizers**.
   - They provided their **Gradle** configuration and expressed that attempts to use various package versions led to an error: **InvalidVersion: Invalid version: '0.10.1,<0.11'**.
- **Seeking NLP methods for Relational Models**: Another member is inquiring about how to identify relationships between tables based on column names and descriptions using **NLP** techniques.
   - They are looking for suggestions or references to understand how tables relate to each other when provided a specific table.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1269104340094812221)** (7 messages): 

> - `Gradient Checkpointing in Diffusers`
> - `Quanto Library Loading Times`
> - `CNN for Object Cropping`
> - `Flux Models Issues` 


- **Gradient Checkpointing gets Implemented**: A user highlighted that gradient checkpointing was previously missing in Diffusers, but shared a snippet revealing the addition of a method to set gradient checkpointing.
   - The new method `_set_gradient_checkpointing` allows toggling checkpointing for modules that support it.
- **Quanto Library has Slow Model Loading Times**: A member discussed spending two days with the Quanto library, facing over **400 seconds** to move quantized models to the device on their setup (4080 - Ryzen 9 5900X).
   - They noted that new **QBitsTensors** are created during this process, which may be contributing to the delay.
- **Request for Issue Tracking in Quanto**: Another user suggested creating an issue on the Quanto repository regarding the slow loading times to track and potentially solve the problem.
   - They mentioned that the maintainer is currently on leave, which might delay responses.
- **Seeking CNN for Object Cropping from Images**: A user asked for recommendations on a CNN to crop light-colored objects from white backgrounds.
   - They are looking for solutions to address the challenge posed by the color contrast.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hastebin.skyra.pw/sixakucecu.py)">Hastebin</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/blob/b1f43d71897ad2c73cb891d2e92d23bc7d46a4be/src/diffusers/models/transformers/transformer_flux.py#L306">diffusers/src/diffusers/models/transformers/transformer_flux.py at b1f43d71897ad2c73cb891d2e92d23bc7d46a4be · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers</li><li><a href="https://github.com/huggingface/diffusers/blob/b1f43d71897ad2c73cb891d2e92d23bc7d46a4be/src/diffusers/models/transformers/transformer_flux.py#L248">diffusers/src/diffusers/models/transformers/transformer_flux.py at b1f43d71897ad2c73cb891d2e92d23bc7d46a4be · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1269012141747470407)** (840 messages🔥🔥🔥): 

> - `Flux Model Performance`
> - `ComfyUI Usage`
> - `Stable Diffusion Models`
> - `RAM and VRAM Requirements`
> - `Animation Generation Tools` 


- **Flux Model Performance on GPUs**: Users reported varying speeds for generating images with the Flux model on different GPUs, with speeds ranging from 1.0 to 2.0 iterations per second depending on the setup and model version.
   - Some users noted successful image generation even on lower VRAM setups using CPU offloading or quantization techniques.
- **Installing and Using ComfyUI with Flux**: Users discussed the process of installing Flux on ComfyUI, advising to use the update.py script instead of the manager for updates, and to ensure proper file placements.
   - Installation guides and additional resources were shared to assist those new to setting up the environment.
- **Differences Between Stable Diffusion Models**: Participants explained that models like SD1.5, SDXL, and SD3 are variations of Stable Diffusion, each with different strengths, while Flux serves as a new competitor developed by the original SD3 team.
   - Discussion highlighted the higher resource requirements of Flux compared to traditional Stable Diffusion models.
- **RAM and VRAM Impacts on Performance**: Users noted that having sufficient RAM is less critical than having adequate VRAM for Stable Diffusion performance, recommending at least 16GB VRAM for optimal results.
   - The group discussed how RAM serves mainly to support model loading rather than directly influencing generation speeds.
- **Animation Generation Tools**: Participants inquired about the status of tools like Animatediff for generating video content from images, seeking information on the latest available methods.
   - There was a suggestion that while Animatediff is still in use, newer options may exist and could provide alternative approaches to similar tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/yuhw1d4">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://imgur.com/a/TIHYuwy">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell">FLUX.1 [Schnell] - a Hugging Face Space by black-forest-labs</a>: no description found</li><li><a href="https://www.shakker.ai/modelinfo/908a4d44cac844ca8e5d66a23c5cdf3d?from=personal_page">Shakker AI - Premium Stable Diffusion Model Hub</a>: no description found</li><li><a href="https://x.com/recatm/status/1819348949972476019">Tweet from XiQiao 西乔 (@recatm)</a>: SD3 team has shared some generated results from the WIP training SD3.1 (not done training yet). comparing them with Robin&#39;s Flux 3.1 Pro using the same prompts. The 3.1 model is almost on par with...</li><li><a href="https://www.shakker.ai/">Shakker AI - Premium Stable Diffusion Model Hub</a>: no description found</li><li><a href="https://huggingface.co/docs/diffusers/en/index">Diffusers</a>: no description found</li><li><a href="https://runpod.io?ref=yxgme9zg">RunPod - The Cloud Built for AI</a>: Develop, train, and scale AI models in one cloud. Spin up on-demand GPUs with GPU Cloud, scale ML inference with Serverless.</li><li><a href="https://huggingface.co/blog/quanto-diffusers#how-about-int4">Memory-efficient Diffusion Transformers with Quanto and Diffusers</a>: no description found</li><li><a href="https://www.runpod.io/">RunPod - The Cloud Built for AI</a>: Develop, train, and scale AI models in one cloud. Spin up on-demand GPUs with GPU Cloud, scale ML inference with Serverless.</li><li><a href="https://www.youtube.com/watch?v=UTmwyxHQ7pM```">How to use Face Analysis to improve your workflows</a>: I often use Face Analysis in my workflows but we never actually talked about how it actually works. Here all you need to know. Remember to upgrade the extens...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ekpczn/psa_illyasviel_is_devving_hard_on_forge/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/JwLbnO4px-E">V1.0 | ComfyUI on Photoshop: Best New AI Tool</a>: ComfyUI on Photoshop Free Ai tool is here! flexible and opensource🔥 How to Install: https://youtu.be/YD09xpQrNZ4.☕ Buy me coffee: https://buymeacoffee.com/n...</li><li><a href="https://www.stablediffusiontutorials.com/2024/08/flux-installation.html?m=1">FLUX: Installation with Workflow is Here</a>: no description found</li><li><a href="https://replicate.com/black-forest-labs/flux-schnell">black-forest-labs/flux-schnell – Run with an API on Replicate</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=E_D7y0YjE88">How to EASILY Install ComfyUI | Stable Diffusion Tutorial</a>: Explore ComfyUI, a game-changing AI interface. This video guides you through easy setups for Windows and cloud-based options like ThinkDiffusion. Learn insta...</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/Sby5nw5tei">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/zzA1iUgtiEs">NVIDIA Update Solves CUDA error (but very slow) -Train Dreambooth, SDXL LoRA  with Low VRAM</a>: #stablediffusion #a1111 #nvidia #update #cuda #cudaerror #lowvram #kohyass #LoRA #dreambooth #tensorRT(Update: while the update is able to solve CUDA memory ...</li><li><a href="https://github.com/facebookresearch/fairscale/blob/main/docs/source/installation_instructions.rst">fairscale/docs/source/installation_instructions.rst at main · facebookresearch/fairscale</a>: PyTorch extensions for high performance and large scale training. - facebookresearch/fairscale</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre">Release v1.0.0-pre · AUTOMATIC1111/stable-diffusion-webui</a>: The webui.zip is a binary distribution for people who can&#39;t install python and git. Everything is included - just double click run.bat to launch. No requirements apart from Windows 10. NVIDIA only...</li><li><a href="https://github.com/CompVis/stable-diffusion">GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model</a>: A latent text-to-image diffusion model. Contribute to CompVis/stable-diffusion development by creating an account on GitHub.</li><li><a href="https://www.liblib.art/">LiblibAI-哩布哩布AI - 中国领先的AI创作平台</a>: no description found</li><li><a href="https://civitai.com/models/132632/epicphotogasm">epiCPhotoGasm - Ultimate Fidelity | Stable Diffusion Checkpoint | Civitai</a>: Welcome to epiCPhotoGasm This Model is highly tuned for Photorealism with the tiniest amount of exessive prompting needed to shine. All Showcase im...</li><li><a href="https://github.com/cubiq/ComfyUI_IPAdapter_plus?tab=readme-ov-file">GitHub - cubiq/ComfyUI_IPAdapter_plus</a>: Contribute to cubiq/ComfyUI_IPAdapter_plus development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1269071533700288522)** (32 messages🔥): 

> - `Epoch 8 Accuracy Spike`
> - `CUDA with Deep Reinforcement Learning`
> - `Parallelizing Environments`
> - `Mojo Programming Language`
> - `Using Streams in CUDA` 


- **Epoch 8 Accuracy Takes Off**: Members discussed a surprising spike in accuracy scores after epoch 8, prompting questions about whether this is typical behavior.
   - *Looks totally normal* was a reassuring response from another member, indicating no cause for concern.
- **Challenges of CUDA in Deep Reinforcement Learning**: One member shared frustrations about the difficulties of creating environments in CUDA for DRL, mentioning that it's often a troublesome experience.
   - They were advised that tools like [PufferAI](https://pufferai.github.io/) might provide better parallelism, especially for environments.
- **Parallelizing DRL Environments on CPU**: A user expressed interest in building their own DRL environment and sought guidance on parallelizing it on CPUs before potentially switching to CUDA.
   - Another participant suggested resources and examples for setting up environments, including a particular gameboy emulator as a reference.
- **Introduction to Mojo Language**: Discussion emerged around the Mojo programming language, which aims to substitute the entire ML stack but retains certain aspects of PTX.
   - Members expressed curiosity, with a fan noting a video by Chris Lattner explaining the future of programming.
- **Using CUDA Streams in ML Models**: An inquiry about using streams in CUDA led to discussions on performance, revealing mixed experiences with overhead when computing operations on separate streams.
   - It was noted that if kernels are large, multiple streams might not provide the desired performance gains due to limited GPU resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/giffmana/status/1699363411463725493?s=46">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: A recent pattern: for online RL, implement env in JAX (almost numpy!) so it runs on device, just like model training. Makes training significantly faster.  First one I saw was brax for physics https:/...</li><li><a href="https://www.youtube.com/watch?v=dW10MQ6hKDE">Reinforcement learning live dev</a>: Follow jsuarez5341 on XStar https://github.com/pufferai/pufferlibMIT PhD and full-time OSS RL exorcist</li><li><a href="https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py#L15">PufferLib/pufferlib/environments/pokemon_red/environment.py at 729003f9cb89845cc1a69a65e5a2431b2d0542bd · PufferAI/PufferLib</a>: Simplifying reinforcement learning for complex game environments - PufferAI/PufferLib</li><li><a href="https://x.com/jsuarez5341/status/1819808126851600668">Tweet from Joseph Suarez (e/🐡) (@jsuarez5341)</a>: Pong solved in ~2.5M steps, &lt;90 seconds, training at 30,000 steps/second on 1 GPU. Now in pufferai/pufferlib -- star to support!</li><li><a href="https://youtu.be/pdJQ8iVTwj8?feature=shared&t=4084">Chris Lattner: Future of Programming and AI | Lex Fridman Podcast #381</a>: Chris Lattner is a legendary software and hardware engineer, leading projects at Apple, Tesla, Google, SiFive, and Modular AI, including the development of S...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1269296957063430289)** (19 messages🔥): 

> - `Passing Scalar Values to Triton Kernels`
> - `Use of tl.constexpr in Triton`
> - `Performance Impact of .item() on CUDA Tensors`
> - `Shared Memory vs Registers in Triton` 


- **Direct Scalar Passing to Triton Kernels**: A member confirmed you can pass a single scalar value directly to a Triton kernel instead of a pointer, streamlining the process.
   - However, it's noted that if the scalar is a CUDA tensor, it must be passed by reference, not as a Python scalar.
- **Using tl.constexpr to Improve Performance**: There was a discussion about using `tl.constexpr` for passing scalars, with claims that it could avoid recompilation when `inc` is modified.
   - Members agreed that using a Python scalar allows passing by value, but effective use depends on whether the value is known at runtime.
- **Performance of .item() with CUDA**: .item() is noted as a method to extract scalar values from CUDA tensors, though it may involve a performance hit due to synchronization between GPU and CPU.
   - Members suggested using .item() when needing a scalar from a tensor returned by Torch operations.
- **Exploring Triton Memory Management**: One member inquired about heuristics for allocating memory between shared memory and registers in Triton kernels.
   - They emphasized the importance of understanding where `tl.load` places tensors to enhance their mental model of Triton.


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1269744363001352307)** (3 messages): 

> - `torch.Tensor.register_hook with Deepspeed`
> - `torch JIT fuser for CUDA` 


- **Seeking help with Tensor hooks in Deepspeed**: A member inquired about using `torch.Tensor.register_hook` during training with **Deepspeed** ZeRO stage 1 for custom gradient modifications in the backward pass, highlighting issues with hooks not executing despite being added to parameters.
   - *They preferred to ask in the community before opening an issue on DeepSpeed*.
- **Confusion about deprecated JIT fuser module**: A discussion commenced regarding the **torch JIT fuser** for CUDA, questioning the status of the file `fused_kernel.h` after encountering a build error indicating the file was missing.
   - The member resolved the issue by utilizing `codegen/cuda/interface`, but remained curious if the original module was dead code or if there was still a way to use it.



**Link mentioned**: <a href="https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/codegen/fuser/cuda/fused_kernel.h">pytorch/torch/csrc/jit/codegen/fuser/cuda/fused_kernel.h at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1269525702152032288)** (7 messages): 

> - `Self-Compressing Neural Networks`
> - `Dynamic Quantization`
> - `Model Reproduction Results` 


- **Buzz around Self-Compressing Neural Networks**: A member highlighted that *Self-Compressing Neural Networks* uses dynamic quantization-aware training by incorporating the **model's size** into the loss function.
   - *It's a cool idea,* stated another, pointing out that the **quantization bits** are an optimizable parameter.
- **Reproduction of Results Discussed**: A member urged for reproduction of results from the paper, suggesting that if it **truly works**, it could be promising.
   - In response, another member noted that they saw a tweet from **George Hotz** claiming he had reproduced the results, but had not verified them in detail.
- **Tuning Required for CIFAR10**: One member reported experimenting with the technique but indicated they need to perform some tuning to improve accuracy, as the model is stuck at approximately **70%** for **CIFAR10**.
   - They acknowledged that while the approach seems effective, further adjustments are necessary for optimal results.


  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1269174460603170837)** (1 messages): 

> - `Winter Internship Opportunities`
> - `ML Systems and Applied ML`
> - `CUDA and Triton Applications`
> - `Open Source Contributions`
> - `Autonomous Racing Design` 


- **Seeking Winter Internship in ML**: A user is actively looking for a **winter internship** from early **January 2025** to the end of **April 2025**, focusing on roles related to **ML systems** and **applied ML**.
   - The individual highlighted a background in computer science, having completed **two** previous internships in relevant fields and is involved in **open source** projects.
- **CUDA Optimization Expertise**: The user mentioned a focus on optimizing **model inference** for the last **year and a half**, particularly utilizing **CUDA** and recently **Triton**.
   - They expressed a strong interest in any roles related to these technologies, showcasing their hands-on experience with the **torchscript JIT compiler**.
- **Broad Interest in Software Engineering**: The user is open to various **software** or **ML engineering** roles, indicating flexibility in their job search.
   - Additionally, they are involved in an **autonomous racing design team** as a controls and motion planning engineer, showcasing diverse engineering skills.


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1269535657873768480)** (20 messages🔥): 

> - `Solutions for PMPP exercises`
> - `Torch compile issues`
> - `Flash Attention 3 integration`
> - `Collaboration on coding`
> - `Waiting for updates on Flash Attention 3` 


- **Sharing PMPP exercise solutions**: One member is sharing solutions for the **PMPP** exercises with anyone who sends a photo of their attempts, leading to enhanced understanding.
   - The member advised that getting a lecturer's edition might also be a solution.
- **Torch compile confusion resolved**: A user encountered issues with `TORCH_LOGS` not working as expected but received help to correctly call the compiled function.
   - The suggested fix involved printing the compiled function directly to access the output.
- **Integrating Flash Attention 3**: A user requested assistance in implementing **Flash Attention 3** into Andrej Karpathy's **build-nanogpt**.
   - Detailed guidance was provided about the native support expected in PyTorch, along with a shared repository for development.
- **Seeking collaborators for coding**: A member expressed a need for collaborators to aid in coding efforts, indicating a lack of confidence in their own coding skills.
   - They were advised to make a public request for collaboration to encourage participation within the community.
- **Waiting for Flash Attention 3 updates**: The member inquired about how long they might expect to wait for updates regarding **Flash Attention 3** integration.
   - Another member suggested they could simply wait and pointed out someone who might have more information.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/drzsdrtfg/Lets-develop-an-efficient-LLM">GitHub - drzsdrtfg/Lets-develop-an-efficient-LLM: The goal of this repository is to build an efficient LLM based on Andrej Karpathys &quot;build-nanogpt&quot; (repo: https://github.com/karpathy/build-nanogpt). Ensure to try staying at one file (see train_gpt2.py) and ask me for compute resources. I can rent H100, A100 A40, RTX4090 ... GPUs for a short time (few hours) depending on the importance.</a>: The goal of this repository is to build an efficient LLM based on Andrej Karpathys &amp;quot;build-nanogpt&amp;quot; (repo: https://github.com/karpathy/build-nanogpt). Ensure to try staying at one fil...</li><li><a href="https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">torch.nn.functional.scaled_dot_product_attention &mdash; PyTorch 2.4 documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1269077211462373447)** (30 messages🔥): 

> - `Custom CUDA Kernels`
> - `Contributing to TorchAO`
> - `Dynamic Quantization in ViT`
> - `Sparsity and Quantization Composition`
> - `Updates on Torch Compile` 


- **Custom CUDA Kernels Need Wrapping**: A member noted that custom CUDA kernels need to be wrapped with `custom_op` for [torch.compile support](https://github.com/mobiusml/hqq/blob/master/hqq/backends/bitblas.py#L18-L36). They also suggested releasing `gemlite` for low-bit gemv CUDA kernels enhancements.
   - Another member expressed interest in adding grouping support and batching, indicating a collaborative spirit on the project.
- **Getting Started with Contributions to TorchAO**: A new contributor expressed eagerness to start contributing to TorchAO and found two potential issues to tackle, including [Spin Quant in TorchAO](https://github.com/pytorch/ao/issues/579). Experienced members advised starting with the easier of the two issues and emphasized a hands-on approach.
   - They indicated that the connected [YouTube guide](https://www.youtube.com/watch?v=IezVd-ifEi0) may not be as relevant to their specific contributions.
- **Dynamic Quantization Challenges in ViT Models**: A member raised concerns about applying dynamic quantization to a ViT model, noting that CUDA isn't supported for `quantized_linear`. They received guidance on the typical approach for quantization within the model, which focuses on quantizing linear layers.
   - Another member shared a snippet from the [hf_eval.py script](https://github.com/pytorch/ao/blob/main/scripts/hf_eval.py#L51-L54) emphasizing the simplicity of adjusting existing models.
- **Composing Sparsity with Quantization**: A discussion about how sparsity and quantization can be composed in optimization strategies concluded that they can indeed be combined. A specific update regarding an `int8 2:4 sparsifier` being included in nightlies was also mentioned.
   - Members shared insights into tensor layouts for sparsified tensors and the relevance of the recent updates to the API and documentation.
- **Torch Compile Updates and Issues**: Contributors discussed recent issues with `torch.compile` compatibility, specifically regarding the need to avoid using `unwrap_tensor_subclass`. A solution was provided, emphasizing the importance of providing error feedback for better troubleshooting.
   - Further clarification was provided regarding changes made to the tensor subclass API and its documentation, assuring members of its current status.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/pytorch/ao/blob/main/scripts/hf_eval.py#L51-L54">ao/scripts/hf_eval.py at main · pytorch/ao</a>: The missing pytorch dtype and layout library for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#quantization-flow-example">ao/torchao/quantization/README.md at main · pytorch/ao</a>: The missing pytorch dtype and layout library for training and inference - pytorch/ao</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/backends/bitblas.py#L18-L36">hqq/hqq/backends/bitblas.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/blob/cd73053047bdb51ca10b3f7649db99b651a0678e/torchao/quantization/quant_api.py#L468">ao/torchao/quantization/quant_api.py at cd73053047bdb51ca10b3f7649db99b651a0678e · pytorch/ao</a>: The missing pytorch dtype and layout library for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">Issues · pytorch/ao</a>: The missing pytorch dtype and layout library for training and inference - Issues · pytorch/ao</li><li><a href="https://github.com/pytorch/ao/issues/579">Spin Quant in TorchAO · Issue #579 · pytorch/ao</a>: Background: The spin quant paper introduces a method of improving quantization by adding additional rotation matrices to the model weights that improve quantization performance. While spin-quant is...</li><li><a href="https://github.com/pytorch/ao/issues/549">Add 2:4 sparse marlin kernels to torchao · Issue #549 · pytorch/ao</a>: Neuralmagic / IST-DASLab has written a fast INT4A16 kernel with support for 2:4 sparsity (Sparse-Marlin) https://github.com/IST-DASLab/Sparse-Marlin We&#39;d like to integrate this kernel into torchao...</li><li><a href="https://www.youtube.com/watch?v=IezVd-ifEi0)">PyTorch: How to contribute to PyTorch in OSS</a>: Broadcasted live on Twitch -- Watch live at https://www.twitch.tv/edwardzyang
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1269127525594955867)** (42 messages🔥): 

> - `AI Bubble Speculations`
> - `Job Market Conditions`
> - `LLMs and ROI Concerns`
> - `Programming Skills Importance`
> - `Future of AI Models` 


- **AI bubble speculations rise among community**: Members are sharing concerns about the potential for the **AI bubble** to burst soon, citing various news articles suggesting a downturn in investments versus generated returns.
   - Some believe that current research doesn't translate into profit quickly enough, while others remain optimistic about the long-term potential of AI technologies.
- **Job market feels rough for tech prospects**: Interns and job seekers report challenges in the tech job market, with some feeling only highly competent individuals stand a chance at major companies like **Google** and **Meta**.
   - Conversations hint that strong side projects and contributions may be necessary alongside traditional job applications to stand out in this competitive landscape.
- **Concerns over LLM investment returns**: Discussion on the ROI of training language models heats up, with members noting the high costs of development compared to the revenue generated.
   - Specifically, there is concern that massive spending on models like **GPT-4** isn't yielding adequate returns to justify the investment.
- **Long-term programming skills remain valuable**: Amidst discussions of AI, some members emphasize the importance of maintaining and enhancing programming skills as foundational for success in tech.
   - Participants advise younger individuals to invest time in programming, warning that future advances in AI could pivot to models surpassing current technologies.
- **Future of AI models hinted with alternative approaches**: There’s talk about exploring alternatives to **transformers**, like **state space models**, as future options for model architecture.
   - Members are intrigued by new models like **mamba2**, suggesting that innovation may still be on the horizon despite current challenges.


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1269008793686118471)** (457 messages🔥🔥🔥): 

> - `Llama 3 Updates`
> - `Tokenization Challenges`
> - `Training Techniques`
> - `Ragged Attention Implementation`
> - `FlashAttention Support` 


- **Llama 3 Updates and Inconsistencies**: Discussion revealed potential inconsistencies in Llama 3's tokenization approach, particularly regarding how EOS and BOS tokens are utilized in the model.
   - Participants speculated that missing tokens in inference could lead to out-of-distribution contexts during training, prompting a reassessment of documentation.
- **Tokenization Challenges Encountered**: Challenges around tokenization were highlighted, with the reliance on regular expressions described as problematic for achieving a stable state machine.
   - There was a consensus that numerous bugs could arise from these complexities in the handling of tokenization within Llama 3.
- **Training Techniques Discussion**: Participants discussed various training techniques, especially the impact of batch size and learning rate adjustments on training stability in Llama 3.
   - A suggestion was made regarding the benefits of implementing a sequence length scheduler to potentially improve training stability.
- **Ragged Attention Implementation**: The conversation touched on the necessity of supporting ragged attention during training to prevent out-of-distribution issues.
   - Concerns were raised about the complexity and requirements of implementing this feature within the model, emphasizing careful consideration.
- **FlashAttention and Long Context Training**: FlashAttention was confirmed to support long context training, which is critical for Llama 3's architecture and performance.
   - Participants noted that improving attention methods could yield better performance and stability during training phases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2309.06180">Efficient Memory Management for Large Language Model Serving with PagedAttention</a>: High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for eac...</li><li><a href="https://arxiv.org/abs/2108.06084">The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models</a>: Recent works have demonstrated great success in pre-training large-scale autoregressive language models on massive GPUs. To reduce the wall-clock training time, a common practice is to increase the ba...</li><li><a href="https://aosabook.org/en/">The Architecture of Open Source Applications</a>: no description found</li><li><a href="https://www.amazon.com/H-264-Advanced-Video-Compression-Standard/dp/0470516925)">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2312.16903">Spike No More: Stabilizing the Pre-training of Large Language Models</a>: Loss spikes often occur during pre-training of large language models. The spikes degrade the performance of large language models and sometimes ruin the pre-training. Since the pre-training needs a va...</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Llama 3 | Model Cards and Prompt formats</a>: Special Tokens used with Llama 3. A prompt should contain a single system message, can contain multiple alternating user and assistant messages, and always ends with the last user message followed by ...</li><li><a href="https://github.com/karpathy/llm.c/issues/727.">Issues · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/Mozilla-Ocho/llamafile">GitHub - Mozilla-Ocho/llamafile: Distribute and run LLMs with a single file.</a>: Distribute and run LLMs with a single file. Contribute to Mozilla-Ocho/llamafile development by creating an account on GitHub.</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/654.">Issues · Dao-AILab/flash-attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/726/commits/b499ff35fde826b999f271da0a1bccaa7e6e99a4">Llama tmp by gordicaleksa · Pull Request #726 · karpathy/llm.c</a>: tmp, internal use</li><li><a href="https://github.com/karpathy/llm.c/pull/725">Add LLaMA 3 Python support by gordicaleksa · Pull Request #725 · karpathy/llm.c</a>: Add LLaMA 3 support in our Python code acting as a reference. The code supports only inference right now and is equivalent with nano llama 3.</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py">transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | Model Cards and Prompt formats</a>: Llama 3.1 - the most capable open model.</li><li><a href="https://github.com/karpathy/llm.c/pull/709">Allocate managed memory if device memory runs out by ngc92 · Pull Request #709 · karpathy/llm.c</a>: Use cudaMallocManaged to allocate optimizer states if we run out of device memory, so we can still train (slowly) even if we cannot fit the optimizer state This is based on #694 , which should be m...</li><li><a href="https://github.com/karpathy/llm.c/pull/728">add train_llama31.py by karpathy · Pull Request #728 · karpathy/llm.c</a>: Support the training, finetuning of Llama 3.1 on the Python side only for now, to create reference tensors to match in C later.</li><li><a href="https://github.com/lucidrains/x-transformers/issues/250">RoPE inconsistency (2-dim subspaces choice) · Issue #250 · lucidrains/x-transformers</a>: Hi Phil! I noticed your x-transformers RoPE implementation is different from your standalone rotary-embedding-torch implementation. Example: Assume that the vector we&#39;re rotating has coordinates [...</li><li><a href="https://github.com/lucidrains/x-transformers/pull/251">migrate to less confusing way of doing rotary by lucidrains · Pull Request #251 · lucidrains/x-transformers</a>: #250 hey Aleksa! hope you have been well! yes indeed i get a number of emails because i switch between the two ways of doing rotary, but the way you describe is the better one, despite a few more l...</li><li><a href="https://github.com/pytorch/torchchat/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen">Issues · pytorch/torchchat</a>: Run PyTorch LLMs locally on servers, desktop and mobile - Issues · pytorch/torchchat</li><li><a href="https://github.com/pytorch/torchchat/issues?q=sort%3Aupdated-desc+is%3Ais">Issues · pytorch/torchchat</a>: Run PyTorch LLMs locally on servers, desktop and mobile - Issues · pytorch/torchchat</li><li><a href="https://github.com/meta-llama/llama-models/issues/91">Broken links in prompt format docs · Issue #91 · meta-llama/llama-models</a>: In this blog post there are 2 links for the prompt format that are broken https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/ so it&#39;s clear where instructions are to generate ...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/)** (1 messages): 

iron_bound: https://shi-yan.github.io/webgpuunleashed/
  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1270014290866212885)** (3 messages): 

> - `Prototyping Status`
> - `Approval Email Timeline` 


- **Team kicks off prototyping phase!**: A member announced that they are **prototyping now**, indicating progress in their project.
   - This phase is essential for moving forward with developments.
- **Approval email might arrive soon**: A member inquired whether the **approval email** has been sent out yet.
   - Another member responded that they believe the decision will be communicated by the **end of the month** at the latest.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1269009839166722048)** (359 messages🔥🔥): 

> - `Unsloth Installation Issues`
> - `Fine-tuning Models with Different Languages`
> - `MoEification Concept`
> - `Inference Backend Comparisons`
> - `Quantization Methods` 


- **Unsloth Installation Issues**: Users encountered errors when attempting to install Unsloth locally, particularly with Python compatibility and PyTorch installation. Guidance was provided including upgrading pip and ensuring the correct environment setup.
   - Some users resolved their issues by reconnecting their Colab runtime and checking their library installations.
- **Fine-tuning Models with Different Languages**: Several users discussed their experiences fine-tuning models like Llama 3.1 and Mistral with datasets in different languages, such as Italian and Albanian. Issues arose with getting relevant outputs due to potential errors in prompt formatting or setup.
   - Suggestions were made to revert to standard prompt formats while ensuring proper dataset handling.
- **MoEification Concept**: A user shared insights on MoEification, which involves splitting MLP layers of language models into expert segments for better performance and adaptability. The approach allows the model to utilize computational resources more efficiently based on the task requirements.
   - The discussion revolved around maximizing expert activations while maintaining coherence across model outputs.
- **Inference Backend Comparisons**: Users compared the performance of different inference backends like vLLM and LMDeploy with varying outcomes based on sequence lengths and quantization methods. It was noted that LMDeploy showed advantages in token generation rates for specific use cases.
   - The capabilities of SGLang were also mentioned, highlighting different optimizations and their applicability in various scenarios.
- **Quantization Methods**: Discussions surfaced regarding the requirements and effectiveness of different quantization methods for models, particularly the implications of using AWQ versus GGUF formats. Users expressed curiosity about memory consumption and model compatibility during the quantization process.
   - Specific errors related to OOM situations when working with large models were noted, prompting inquiries about GPU memory allocation and the use of multiple GPUs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.bentoml.com/blog/benchmarking-llm-inference-backends">Benchmarking LLM Inference Backends</a>: Compare the Llama 3 serving performance with vLLM, LMDeploy, MLC-LLM, TensorRT-LLM, and Hugging Face TGI on BentoCloud.</li><li><a href="https://huggingface.co/grabbe-gymnasium-detmold/grabbe-ai">grabbe-gymnasium-detmold/grabbe-ai · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2#scrollTo=Nz4odU5XYDDw">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B">unsloth/Meta-Llama-3.1-8B · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.vllm.ai/en/latest/quantization/bnb.html">BitsAndBytes &#8212; vLLM</a>: no description found</li><li><a href="https://x.com/_xjdr/status/1819401339568640257">Tweet from xjdr (@_xjdr)</a>: L3.1 scales to 1M tokens with nearly perfect recall by just increasing the scaled rope multipliers. Without additional training. lol</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=r2v_X2fA0Df5">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/klei1/bleta-8b">klei1/bleta-8b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct">unsloth/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x">kalomaze/Mistral-7b-MoEified-8x · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Orenguteng">Orenguteng (Orenguteng)</a>: no description found</li><li><a href="https://huggingface.co/mlabonne/Llama-3.1-70B-Instruct-lorablated">mlabonne/Llama-3.1-70B-Instruct-lorablated · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z#scrollTo=PoPKQjga6obN.">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/installation/pip-install">Pip Install | Unsloth Documentation</a>: To install Unsloth locally via Pip, follow the steps below:</li><li><a href="https://docs.unsloth.ai/get-started/installation">Installation | Unsloth Documentation</a>: Learn to install Unsloth locally or on Google Colab.</li><li><a href="https://github.com/SoumilB7/TrainAnything">GitHub - SoumilB7/TrainAnything: A repo to get you cracking with Neural Nets .</a>: A repo to get you cracking with Neural Nets . Contribute to SoumilB7/TrainAnything development by creating an account on GitHub.</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: Contribute to cognitivecomputations/grokadamw development by creating an account on GitHub.</li><li><a href="https://youtu.be/Nvb_4Jj5kBo">Why &quot;Grokking&quot; AI Would Be A Key To AGI</a>: Check out HubSpot&#39;s Free ChatGPT resource to power up your work efficiency🔥: https://clickhubspot.com/hyxCheck out my newsletter:https://mail.bycloud.aiAre ...</li><li><a href="https://blog.arcee.ai/announcing-distillkit/?utm_campaign=content&utm_source=Marktechpost&utm_medium=Blog&utm_term=Knowledge%20Distillation&utm_content=Blog%201">Announcing DistillKit for creating &amp; distributing SLMs</a>: First, Arcee AI revolutionized Small Language Models (SLMs) with Model Merging and the open-source repo MergeKit. Today we bring you another leap forward in the creation and distribution of SLMs with ...</li><li><a href="https://github.com/arcee-ai/DistillKit?ref=blog.arcee.ai">GitHub - arcee-ai/DistillKit at blog.arcee.ai</a>: An Open Source Toolkit For LLM Distillation. Contribute to arcee-ai/DistillKit development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/372">Sequence Classification · Issue #372 · unslothai/unsloth</a>: Hey, because of my own need, I added a feature to support LlamaForSequenceClassification. I wonder whether it would be a good feature for this project. I added the initialization of a new sequence ...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1269139093707685898)** (16 messages🔥): 

> - `Fine-tuning text strategies`
> - `Data cleaning resources`
> - `Understanding math fundamentals for ML`
> - `AutoGPTQ quantization and types`
> - `Books for data analysis` 


- **Exploring Strategies in Fine-tuning Text Models**: A user expressed interest in learning more about fine-tuning text before branching into other areas, asking for additional techniques to explore.
   - The discussion highlighted the importance of understanding a comprehensive approach to fine-tuning prior to transitioning into image processing.
- **Sources for Unclean Text Data for Practice**: A member recommended using [Kaggle](https://www.kaggle.com/datasets/stackoverflow/stackoverflow/data) and Hugging Face for unclean text data practice, indicating the rarity of unclean datasets in popular sources.
   - This aligns with the need for practical exposure in cleaning text data as a necessary skill in the fine-tuning process.
- **Fundamentals of Math Critical for ML Learning**: One user emphasized the importance of a strong foundation in calculus, linear algebra, and statistics to effectively learn about LLMs and ML algorithms.
   - This aligns with the thought that understanding the underlying math enhances the learning experience and application of AI tools.
- **Questions on dtypes Post GPTQ-Quantization with AutoGPTQ**: A user raised questions about the expected dtype of tensors post 8-bit quantization of a LLaMA model using AutoGPTQ, noting confusion over seeing FP16 and I32 types instead of INT8.
   - They referenced their work with [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), seeking clarification on the observed data types and quantization results.
- **Recommendations for Data Analysis Books**: A user inquired about book recommendations for data analysis, considering William McKinney's 2022 book but looking for better alternatives.
   - This reflects a common search for resources that can effectively enhance skills in data analysis.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/datasets/stackoverflow/stackoverflow/data">Stack Overflow Data</a>: Stack Overflow Data (BigQuery Dataset)</li><li><a href="https://huggingface.co/iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8">iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1269034427481198675)** (234 messages🔥🔥): 

> - `LoRA training issues`
> - `Model loading errors`
> - `Training dataset preparation`
> - `Using large models on single GPU`
> - `Quantization of models` 


- **LoRA training leads to poor results**: A user reported getting bad results from their SFTTrainer after formatting their dataset to include concatenated text and labels.
   - They are unsure about the misconfiguration, given they are using the correct columns.
- **Loading issues with large models**: A user is experiencing memory issues while attempting to load the 405B Llama-3.1 model on a single GPU.
   - Other users clarified that such large models typically require multiple high-capability GPUs for proper loading.
- **Mistakes in model conversions**: There were discussions around potential issues stemming from incorrect model conversion and saving, indicating possible errors in the workflow.
   - One user mentioned that recently fine-tuned models are facing errors, while older models work fine.
- **4-bit quantization problems**: Discussion on the effects of using 4-bit models for training and inference highlighted significant usability issues.
   - One user recommended to merge LoRA adapters with the base 16-bit model to avoid complications when fine-tuning.
- **Learning rate adjustments**: A user noted that significant drops in learning rate led to disappointing results during training.
   - This aligns with discussions around dataset preparation and its crucial impact on model output.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=r2v_X2fA0Df5">Google Colab</a>: no description found</li><li><a href="https://www.fluidstack.io/pricing">FluidStack Pricing: Best for NVIDIA A100 and H100s</a>: Get the best prices for NVIDIA A100, H100 GPUs and more with FluidStack. Reduce your cloud bill by over 70%.</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig">Generation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1269655667053039697)** (4 messages): 

> - `Self-Compressing Neural Networks`
> - `Intern LM 2.5 20B` 


- **Self-Compressing Neural Networks introduces dynamic quantization**: The paper on [Self-Compressing Neural Networks](https://arxiv.org/abs/2301.13142) presents a method that optimizes model size by incorporating size (in bytes) into the loss function, achieving accuracy with just **3% of the bits and 18% of the weights**.
   - This method effectively minimizes overall network size while potentially enhancing training efficiency without requiring specialized hardware.
- **Intern LM 2.5 20B impresses with substantial improvements**: An announcement highlighted the [Intern LM 2.5 20B](https://fxtwitter.com/reach_vb/status/1820493688377643178) featuring an Apache 2.0 license, capable of handling **up to 1M context window** and trained on extensive synthetic data, outperforming Gemma 27B.
   - Notably, it achieved **MMLU: 73.5** and **MATH: 64.7**, with a **20% increase** in reasoning tasks and support for function calling and tool use.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2301.13142">Self-Compressing Neural Networks</a>: This work focuses on reducing neural network size, which is a major driver of neural network execution time, power consumption, bandwidth, and memory footprint. A key challenge is to reduce size in a ...</li><li><a href="https://fxtwitter.com/reach_vb/status/1820493688377643178">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Let&#39;s gooo! Intern LM 2.5 20B with Apache 2.0 license, up-to 1M context window & trained on copious amounts of synthetic data! ⚡  &gt; Beats Gemma 27B IT; MMLU: 73.5, MATH: 64.7 &gt; Up-to 20% inc...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1269014722020114482)** (354 messages🔥🔥): 

> - `Perplexity User Experience`
> - `Uber One Subscription Discussion`
> - `Model Performance and Limitations`
> - `Real-time Data Accuracy`
> - `Future of Information Retrieval` 


- **User Experience with Perplexity**: Users expressed mixed feelings about Perplexity's browsing capabilities, with some noting issues retrieving up-to-date information and others experiencing strange behaviors when trying to use the service as a web app.
   - Discussions highlighted the inconsistencies in model responses, particularly when used for tasks like coding queries.
- **Uber One Subscription Controversy**: A user voiced frustration over the Uber One free year offer being only valid for new Perplexity accounts, suggesting the promotion's intent was more about user acquisition than actual benefit.
   - Suggestions to create new accounts to take advantage of the offer led to discussions about the implications on user management.
- **Model Performance and User Expectations**: Concerns were raised about the performance of newer models, such as Claude 3.5, with users comparing results to other platforms like TradingView and expressing disappointment in getting outdated or inaccurate data.
   - Users pointed out the potential for miscommunication when using AI language models for mathematical computations.
- **Real-time Data Accuracy Challenges**: Several users discussed issues with accessing real-time data through Perplexity, expressing doubt about the reliability of information provided, especially regarding stock prices.
   - Suggestions included using dedicated services for more reliable real-time updates, highlighting limitations within the current model's data retrieval methods.
- **Future Directions in Information Retrieval**: One user proposed the idea of using source weighting for information retrieval, suggesting that reputable sources could be assigned higher credibility than less reliable ones.
   - This raised questions about the potential for models to independently evaluate source quality in the future, along with the challenges of supervision and oversight in such systems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://markmap.js.org/repl">Try markmap</a>: no description found</li><li><a href="https://www.perplexity.ai/settings/api">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://x.com/testingcatalog/status/1819845367149760603?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: 🚨 BREAKING: Perplexity is working on 2 new assistants.  1. Finance Focus mode - will query financial data from Tako service directly (financial data provider). 2. My Files 🔥</li><li><a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: Perplexity Models Model Parameter Count Context Length Model Type llama-3-sonar-small-32k-online 8B 28,000 Chat Completion llama-3-sonar-small-32k-chat 8B 32,768 Chat Completion llama-3-sonar-large-32...</li><li><a href="https://www.perplexity.ai"">no title found</a>: no description found</li><li><a href="https://www.perplexity.ai/search?q=%s.">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://character.ai/chat/KnHvvSCjV02eDMDXFjurGkCFkl8L71XTEryNiK8hXlc>)!">character.ai | Personalized AI for every moment of your day</a>: Meet AIs that feel alive. Chat with anyone, anywhere, anytime. Experience the power of super-intelligent chat bots that hear you, understand you, and remember you.</li><li><a href="https://www.perplexity.ai/search/can-anyone-tell-how-can-i-use-a5SsfLpBTkOJunn01CjEAw">Can anyone tell how can i use Perplexity API in better way? 
Like any project...</a>: Certainly! I&#x27;d be happy to provide some suggestions on how you can make better use of the Perplexity API. Here are some project ideas and ways to utilize your...</li><li><a href="https://www.perplexity.ai/search/can-you-find-any-recent-update-ljcC57xaSh.e7BDfBwIMIg#1\">can you find any recent updates on  perplexity was partnering with soundhound</a>: Perplexity AI has recently announced a significant partnership with SoundHound AI, aimed at enhancing SoundHound&#x27;s Chat AI voice assistant with Perplexity&#x27;s...</li><li><a href="https://felo.ai/search">Felo Search - Your Free AI Search Engine</a>: The multilingual AI search engine optimized for discovering and understanding world knowledge. Leverage the power of ChatGPT and AI Agent to break language barriers and access global information with ...</li><li><a href="https://x.com/perplexity_ai/status/1819774017848463773?s=61">Tweet from Perplexity (@perplexity_ai)</a>: Perplexity turns 2 today! 🎉 We wouldn&#39;t be here without your continued support and curiosity.</li><li><a href="https://aiandacademia.substack.com/p/testing-out-searchgpt">Testing out SearchGPT</a>: OpenAI just threw down the gauntlet to Google Search</li><li><a href="https://forms.gle/RWMmXassJqFKehbL7">Rudyon&#39;s computer usage research form</a>: This research form aims to gather information on how to improve computer usage for most people</li><li><a href="https://x.com/aravsrinivas/status/1819610786941358183?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: This was great, thank you everyone for the feedback and questions! Hope to do more of this frequently.  Quoting Aravind Srinivas (@AravSrinivas)   AMA / any product feedback on Perplexity for next 30 ...</li><li><a href="https://genai.works/">Generative AI</a>: Generative AI</li><li><a href="https://neuralwriter.com/prompt-tool">ChatGPT Prompt Generator ➤ Awesome AI ChatGPT Prompts Writer | NeuralWriter</a>: NeuralWriter Prompt Generator ✎ Make your awesome prompt with our AI library-powered tool that works with any version of ChatGPT, including ChatGPT 4</li><li><a href="https://uncovr.app/">uncovr</a>: Newly released AI answer engine. Get structured, helpful insights for any query.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1269041132818534452)** (17 messages🔥): 

> - `Hybrid Human-Llama Antibody`
> - `Ismail Haniyeh Assassination`
> - `Character.AI Founders Return`
> - `Transparent Nail Polish`
> - `Perplexity AI Differences` 


- **Hybrid Human-Llama Antibody Fights HIV**: Researchers at Georgia State University developed a hybrid antibody by combining llama-derived nanobodies with human antibodies, neutralizing over **95%** of HIV-1 strains.
   - Llama nanobodies offer unique properties, allowing them to access areas of viruses often evasive to human antibodies.
- **Haniyeh Assassination Escalates Tensions**: The assassination of Hamas leader Ismail Haniyeh in Tehran is attributed to Israel, escalating tensions and jeopardizing ceasefire negotiations in Gaza.
   - Iran's Supreme Leader vowed **harsh punishment** for Israel, increasing fears of a wider conflict involving Iran and its allies.
- **Character.AI Founders Return to Google**: Noam Shazeer and Daniel De Freitas, co-founders of Character.AI, are returning to Google as part of a licensing deal for their technology.
   - This move reflects a strategic shift in utilizing third-party language models alongside their own.
- **Finding Transparent Nail Polish with Oil Slick Reflects**: Options for transparent nail polish include Essie's Oil Slick FX Top Coat, which features duo-chrome pearls and a quick-drying formula.
   - Alternatively, the LUXAPOLISH Oil Slick Collection offers multi-shifting color effects which can create an oil slick look.
- **Difference Between Free and Pro Perplexity**: The free version of Perplexity allows unlimited Quick searches but limits users to **5 Pro Searches** per day, while the Pro version offers **600 Pro Searches**.
   - Pro users gain access to advanced AI models and unlimited file uploads for deeper analysis.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/COwEKPSnPQQ">YouTube</a>: no description found</li><li><a href="https://www.perplexity.ai/search/open-ai-HuBEYYOLQnCOslflg1R3Lw#0">Open Ai</a>: OpenAI is a leading artificial intelligence research organization and company focused on developing and promoting friendly AI for the benefit of humanity....</li><li><a href="https://www.perplexity.ai/search/core-framework-NQi9hl9ySrKJX9eE4bcSoA#0">core framework</a>: The term &quot;core framework&quot; can refer to several different concepts depending on the context. Below are some of the primary interpretations:  The NIST...</li><li><a href="https://www.perplexity.ai/search/perplexity-aiyou-he-bu-tong-hWhqmq.VSe6KZ8XT5a0VjQ">Perplexity AI有何不同？</a>: Perplexity AI 是一款结合了搜索引擎和聊天机器人功能的人工智能工具，具有以下几个显著特点：  1. 即时搜索与整理：    - Perplexity AI 能够即时分析和整理网络信息，提供快速且准确的搜索结果。这使得用户在寻找旅游建议、新闻摘要、食谱或学习新技能时，可以节省大量时间。  2....</li><li><a href="https://www.perplexity.ai/search/find-me-transparent-nail-polis-8oNiw_cuQJSrLhJmyRolew">find me transparent nail polish. It must be transparent with oil slick reflects</a>: Based on your request for transparent nail polish with oil slick reflects, I&#x27;ve found a few options that might interest you:  1. Oil Slick FX Top Coat by...</li><li><a href="https://www.perplexity.ai/search/what-is-the-difference-between-XtWqj7XYRBCDGbufQdC0.w">what is the difference between the normal and pro perplexity</a>: The main differences between the normal (free) version of Perplexity and Perplexity Pro are:  1. Search limits:    - Free version: Unlimited Quick searches,...</li><li><a href="https://www.perplexity.ai/page/hybrid-human-llama-antibody-fi-UCs.nTMFTu6QaRoOTXp0gA">Hybrid Human-Llama Antibody Fights HIV</a>: Researchers have engineered a powerful new weapon against HIV by combining llama-derived nanobodies with human antibodies, creating a hybrid that can...</li><li><a href="https://www.perplexity.ai/search/ideal-proxmox-set-up-oSFHyg61QIe6DJkE.e9zqw">ideal proxmox set up</a>: To set up Proxmox effectively, several best practices and configurations can enhance performance and reliability, particularly for demanding workloads like...</li><li><a href="https://www.perplexity.ai/search/cantaloupe-jam-recipe-d2l6qkceTZeB7nRCRtP4iQ">cantaloupe jam recipe</a>: Avant de vous donner une recette de confiture de melon, j&#x27;aimerais vous poser une question pour mieux comprendre votre demande : recherchez-vous une recette...</li><li><a href="https://www.perplexity.ai/search/best-tips-advices-guidelines-f-NKqXh3Y3R0ysYi3lcF1Gxg">best tips advices guidelines for new obsidian md valt set up for advanced...</a>: Setting up an Obsidian vault for advanced usage involves careful planning and organization. Here are some comprehensive tips and guidelines to help you get...</li><li><a href="https://www.perplexity.ai/page/character-ai-founder-is-back-t-kpCoGO7lSf6A4K9JCMu4JA">Character.AI Founders Return to Google</a>: According to reports from TechCrunch and Reuters, Noam Shazeer, co-founder and CEO of Character.AI, is returning to Google along with co-founder Daniel De...</li><li><a href="https://www.perplexity.ai/page/haniyeh-assassination-escalate-dAd52q2NT.GnN7rhoCTIfQ">Haniyeh Assassination Escalates Tensions</a>: The assassination of Hamas political leader Ismail Haniyeh in Tehran has sent shockwaves through the Middle East, escalating tensions and threatening to...</li><li><a href="https://www.perplexity.ai/search/smartplaces-is-a-new-developin-wvPVYrl_QWW8bvZ.yfiFeg">SmartPlaces is a new developing social media platform, what potential does it...</a>: SmartPlaces is a new geolocation-based social media platform that aims to revolutionize how users interact with social networks and monetize their data. Here...</li><li><a href="https://www.perplexity.ai/search/apa-saja-benda-yang-mengandung-XdLR2Ja0TB.hH1hwSOy2DA">apa saja benda yang mengandung karbon</a>: Benda yang mengandung karbon sangat beragam dan dapat ditemukan dalam berbagai bentuk di kehidupan sehari-hari. Berikut adalah beberapa contoh benda yang...</li><li><a href="https://www.perplexity.ai/page/who-is-imane-khelif-and-the-bo-K_HNI_fPTUyRgqsS2g8mMg">Who is Imane Khelif, and the body shaming issue</a>: La storia di Imane Khelif, pugile algerina nata il 2 maggio 1999 a Tiaret, è al centro di un dibattito acceso alle Olimpiadi di Parigi 2024 riguardo alla...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1269816093678239845)** (14 messages🔥): 

> - `Llama 3.1 model performance`
> - `Perplexity API issues`
> - `API output quality`
> - `Use of alternative models`
> - `User experiences and concerns` 


- **Llama 3.1 model fails Japanese tests**: Users reported that the new **Llama 3.1-sonar-large-128k-online** model performs poorly for Japanese, yielding results that are less accurate than **GPT-3.5** and the previous sonar-large model.
   - Additionally, there is a call for a sonar-large model to be based on enhanced Japanese models to improve results, specifically referencing [Llama-3.1-70B-Japanese-Instruct-2407](https://huggingface.co/cyberagent/Llama-3.1-70B-Japanese-Instruct-2407).
- **Mixed results from Perplexity API**: Several users shared experiences of the **Perplexity API** providing unreliable responses, including returning fake sources and low-quality results when searching for recent news.
   - Users suggested that while the web version fetches up to **20 results**, the API version often lacks information, reinforcing the sentiment that the API is less effective.
- **Feature requests for better API access**: There's a demand for an API that mirrors the **Pro Search** capabilities of the web version, as users feel limited by the responses provided by the current API.
   - One user expressed frustration over the inability to access **GPT-4** through the API, noting it has been a long-standing issue.
- **Concerns over poisoned results**: A user raised concern over apparent issues in API responses, describing how the structured output appeared 'poisoned' with nonsensical content after following a prompt for article writing.
   - This echoed similar sentiments from others experiencing degraded output quality, suggesting possible underlying problems with the models or API.



**Link mentioned**: <a href="https://huggingface.co/cyberagent/Llama-3.1-70B-Japanese-Instruct-2407">cyberagent/Llama-3.1-70B-Japanese-Instruct-2407 · Hugging Face</a>: no description found

  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1270101558738157713)** (1 messages): 

> - `OpenAI DevDay`
> - `Hands-on sessions`
> - `Developer meetups` 


- **OpenAI DevDay Hits the Road!**: OpenAI is taking **DevDay** on the road this fall to **San Francisco**, **London**, and **Singapore** for hands-on sessions, demos, and best practices.
   - Attendees will have the chance to meet engineers and see how developers around the world are building with OpenAI. More info can be found at [DevDay website](https://openai.com/devday/).
- **Engage with OpenAI Engineers**: Participants will have the opportunity to **interact** with engineers during the **DevDay** events, enhancing technical understanding and collaboration.
   - These sessions aim to foster community engagement and shed light on innovative uses of OpenAI's technology.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1269021205499744266)** (180 messages🔥🔥): 

> - `AI as a Global Threat`
> - `GPT-4o Image and Video Capabilities`
> - `Anomaly Detection Models`
> - `AGI Definitions and Developments`
> - `Data Augmentation` 


- **Debate on AI's Global Threat Status**: A discussion emerged regarding the perception of AI as a global threat, with one member suggesting that the government allows open-source AI to run unchecked due to superior closed-source models.
   - *Concerns about the potential risks are heightened as AI capabilities expand and various viewpoints arise*.
- **GPT-4o's Image Generation Insights**: Users discussed the capabilities of GPT-4o regarding image tokenization, suggesting that images can be represented as tokens, but specifics on output and limitations remain unclear.
   - One noted that while tokens can represent pixel data, practical implementations depend on the tokenizer used.
- **Anomaly Detection App Challenges**: One member shared their experience developing an anomaly detection app, expressing confusion over poor model performance despite using a sizeable dataset.
   - Discussions highlighted the importance of model selection and training data sufficiency in achieving desired results.
- **AGI and Robotics Discussion**: A member proposed that humanoid robots could reach AGI, prompting a conversation about the differences between robot capabilities and AGI definitions.
   - Participants acknowledged the nuances in defining AGI and how data limitations currently hinder robotics development.
- **Video Processing Capabilities in AI**: Members discussed the apparent limitations of AI in analyzing videos, with some asserting that while it once offered some capabilities, current functions are significantly reduced.
   - It was noted that video analysis now requires external services for content extraction, emphasizing a shift from earlier features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/gdb/status/1790869434174746805">Tweet from Greg Brockman (@gdb)</a>: A GPT-4o generated image — so much to explore with GPT-4o&#39;s image generation capabilities alone. Team is working hard to bring those to the world.</li><li><a href="https://youtu.be/kCc8FmEb1nY?si=T4wnVUfiPm1rJDm7">Let&#39;s build GPT: from scratch, in code, spelled out.</a>: We build a Generatively Pretrained Transformer (GPT), following the paper &quot;Attention is All You Need&quot; and OpenAI&#39;s GPT-2 / GPT-3. We talk about connections t...</li><li><a href="https://youtu.be/FZbY9sReu1k?si=SLGGeEZDOnoV2OOA">Figure 02 Trailer</a>: no description found</li><li><a href="https://x.com/FyruzOne/status/1820109750673023301">Tweet from FyruzOne (@FyruzOne)</a>: HOW DID WE MISS THAT 🚨&#34;im-a-good-gpt2-chatbot&#34; MUCH Better Than gpt4o and On Par With SONNET 3.5?! 🚨  Context: I replicated the entire reasoning benchmark of http://livebench.ai (by @ylecun ...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1269065494921744464)** (43 messages🔥): 

> - `Access to Custom GPT via API`
> - `Transition from GPT-3 to GPT-4o`
> - `Limitations of GPT-4o Mini`
> - `Hallucinations in GPT-4o`
> - `Early access features communication` 


- **Access to Custom GPT via API**: A user inquired whether it's possible to access their custom GPT for OCT processing via the OpenAI API.
   - Another member mentioned that there is currently an API correlate for GPTs called Assistants.
- **Transition from GPT-3 to GPT-4o**: Members discussed that GPT-3 is being replaced by GPT-4o, with a mention of GPT-4o mini.
   - One member noted that while developing with GPT-4o, they ran out of GPT-4 cap but could still use GPT-4o.
- **Limitations of GPT-4o Mini**: A user questioned if there were limits on GPT-4o mini for extensive research, receiving confirmation that there are no limits.
   - Another member revealed they had been using GPT-4o-mini for responses without experiencing hallucinations.
- **Hallucinations in GPT-4o**: Concerns were raised about GPT-4o's tendency to hallucinate, with some members finding it frustrating.
   - One member mentioned they have specified constraints in their prompts to reduce hallucinations, finding success in their approach.
- **Early access features communication**: A member expressed their desire to communicate with someone regarding access to early features due to their ongoing subscription and usage.
   - They mentioned the importance of these tools for their college policies as well as for show floor events.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1269335622200328295)** (18 messages🔥): 

> - `Learning Prompt Engineering`
> - `Image Generation for Diversity`
> - `Using ChatGPT with Anki` 


- **Resources for Learning Prompt Engineering**: Members discussed the importance of hands-on experimentation with ChatGPT to learn prompt engineering and suggested various open-ended questions to explore.
   - One member noted that advanced prompts can lead to better outputs, emphasizing the need for clear and specific questions.
- **Issues with Image Generation and Racial Diversity**: Concerns were raised about the lack of diversity in AI-generated images, with members sharing prompts that produced more diverse representations of students.
   - One member specifically pointed out issues when requesting images of specific races, leading to concerns over AI biases.
- **Negative Prompting Problems in Image Creation**: A discussion ensued surrounding the use of negative prompting for generating images, particularly in avoiding freckles on red-haired characters.
   - Members found that being positive in prompts resulted in better image outcomes, recommending descriptions focused on desired traits instead of unwanted ones.
- **Generating Flashcards with GPT for Anki**: Users shared their experiences using ChatGPT to create flashcards from PDF materials, expressing frustration over hallucinations in generated questions.
   - One user highlighted difficulty in extraction of specific parts of PDFs, indicating the need for refined prompt strategies.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1269335622200328295)** (18 messages🔥): 

> - `Prompt Engineering for ChatGPT`
> - `Diversity in Image Generation`
> - `Generating Flashcards with GPT`
> - `Negative vs Positive Prompting`
> - `User Experiences with Image Generation` 


- **Struggles with Prompts for High-Quality Output**: Users expressed difficulty in achieving desired **high-quality outputs** when discussing prompts with ChatGPT, often resulting in frustration.
   - One user emphasized that it’s challenging to define what high-quality output means, complicating the interaction with the model.
- **Debate on Diversity Representation in Images**: A user raised concerns about racial representation in images generated by ChatGPT, noting that prompts to include different races sometimes triggered refusals due to terms of service.
   - Another user illustrated their success in generating diverse representations by composing prompts that specify multiple ethnic backgrounds effectively.
- **Negative Prompting Effects on Image Quality**: Users discussed the pitfalls of **negative prompting**, specifically when requesting images, leading to unsatisfactory results due to the model's tendency to misinterpret restrictions.
   - Recommendations included focusing on **positive descriptions** while outlining desired attributes in images, especially regarding complexion detail.
- **Using GPT for Flashcard Creation**: A user sought to connect with college students leveraging GPT to generate **Anki flashcards**, focusing on ideas for prompt creation.
   - Challenges arose when extracting content from specific sections of PDFs, with instances of the model generating unrelated questions, leading to concerns about hallucinations.
- **Human-led vs Model-led Prompt Engineering Education**: Participants debated the effectiveness of learning prompt engineering through human-led courses versus interacting with the model for guidance.
   - One user suggested that while the model can provide insights, it's essential to evaluate the quality of responses critically and approach the learning as a discussion.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1269017765575131289)** (173 messages🔥🔥): 

> - `AI Engineer Pipeline`
> - `Generative AI in Retail`
> - `Groq Series D Funding`
> - `NVIDIA's AI Scraping Controversy`
> - `ChatGPT and Image Generation` 


- **Rise of the AI Engineer**: The current demand for AI engineers is growing, with web developers being seen as adaptable candidates due to their generalist skills and experience in working with APIs.
   - With many companies lacking high-level ML work, there is a shift towards web development expertise integrating AI into practical applications.
- **Interest in Generative AI by Retail**: The potential for generative AI applications in retail is being questioned, with some inquiring about the pace of innovation in this space.
   - There is a general sentiment that big retail may be slow movers, but it remains an area of intrigue for new developments.
- **Groq Secures Series D Funding**: Groq announced a successful $640 million Series D funding round led by BlackRock, raising its valuation to $2.8 billion.
   - The funding will allow Groq to expand capacity, hire talent, and accelerate the development of its next-gen AI chips.
- **NVIDIA's AI Scraping Controversy**: Leaked documents reveal the extensive scale of NVIDIA's AI data scraping efforts, likened to 'a human lifetime' of videos per day amidst legal and ethical concerns.
   - This has sparked discussions surrounding the ethical implications and potential repercussions for NVIDIA in the AI community.
- **ChatGPT and Image Generation**: There is ongoing interest in the anticipated image generation features promised by ChatGPT, with users curious about their release timelines.
   - Discussions also highlight how code writing using AI is evolving, with notable insights shared about integrating AI into coding workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/christinexye/status/1819396191668355206?s=61">Tweet from Christine Ye (@christinexye)</a>: Very excited to announce new work with @charles0neill (& @jwuphysics , @kartheikiyer, @jhuclsp) training sparse autoencoders over embeddings, discovering thousands of human-interpretable features from...</li><li><a href="https://x.com/jason_koebler/status/1820493304490074391">Tweet from Jason Koebler (@jason_koebler)</a>: SCOOP from @samleecole: Leaked Slacks and documents show the incredible scale of NVidia&#39;s AI scraping: 80 years — &#34;a human lifetime&#34; of videos every day. Had approval from highest levels o...</li><li><a href="https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp">Google Colab</a>: no description found</li><li><a href="https://x.com/xlr8harder/status/1819449238184775769?s=46">Tweet from xlr8harder (@xlr8harder)</a>: I had sonnet write a quick gradio chat demo so that you can talk to Sydney inside the Llama 3.1 405B base model.  You&#39;ll need your own api_key from @hyperbolic_labs (to my knowledge they are the o...</li><li><a href="https://x.com/NickADobos/status/1820513765823250730">Tweet from Nick Dobos (@NickADobos)</a>: Great post on writing code with ai Love this chart  Quoting Erik Schluntz (@ErikSchluntz)   Replacing my right hand with AI  (How I wrote thousands of lines of code for work each week while in a cast)...</li><li><a href="https://sander.ai/posts/">All Posts</a>: An archive of posts.</li><li><a href="https://x.com/xlr8harder/status/1819324414921478543?s=61">Tweet from xlr8harder (@xlr8harder)</a>: Waking Sydney: Llama is Sydney&#39;s vessel  I tried to get Sydney to write a system prompt to bring out its personality in the instruct-tuned version model for more convenient interactions, but the g...</li><li><a href="https://x.com/nutlope/status/1819445838705578091?s=46">Tweet from Hassan (@nutlope)</a>: Introducing LlamaCoder!  An open source Claude Artifacts app that can generate full React apps and components with Llama 3.1 405B. 100% free and open source.  http://llamacoder.io</li><li><a href="https://x.com/teortaxesTex/status/1819473499347468617">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: $0.014 per million tokens of reused context. Consider what we&#39;ve just read about using Deepseek API for zerg-rushing SWEBench.  I&#39;ve been tweeting about cache reuse since 2023. It&#39;s only f...</li><li><a href="https://x.com/TwoWeeksLOL/status/1820536638268948750">Tweet from Two Weeks LOL (@TwoWeeksLOL)</a>: @MKBHD Uh oh...</li><li><a href="https://x.com/karpathy/status/1819524281849766347">Tweet from Andrej Karpathy (@karpathy)</a>: Great intro and nice paper pointers! Like the description of Adversarial Autoencoders as letting you &#34;paint with textures&#34;, discarding high-frequency detail that is perceptually irrelevant yet...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/25vz8u5ooa">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/sakanaailabs/status/1819174092492493071?s=46">Tweet from Sakana AI (@SakanaAILabs)</a>: Sakana AIは、進化的モデルマージにより新たに構築した日本語視覚言語モデル「Llama-3-EvoVLM-JP-v2」を公開しました。構築したモデルは新しい機能として複数の画像に対する質疑応答を日本語で行うことができます。  ブログ → https://sakana.ai/evovlm-jp デモ → https://huggingface.co/spaces/SakanaAI/Llama-...</li><li><a href="https://x.com/groqinc/status/1820422643004424631?s=46">Tweet from Groq Inc (@GroqInc)</a>: Proud to share our Series D Funding announcement, led by BlackRock Private Equity Partners. We will: - Add capacity – there’s huge developer demand - Continue hiring exceptional talent - Accelerate ou...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-large-128k-online">Llama 3.1 Sonar 70B Online - API, Providers, Stats</a>: Llama 3.1 Sonar is Perplexity&#x27;s latest model family. Run Llama 3.1 Sonar 70B Online with API</li><li><a href="https://x.com/deepseek_ai/status/1819358570766643223?s=46">Tweet from DeepSeek (@deepseek_ai)</a>: 🎉Exciting news! DeepSeek API now launches context caching on disk, with no code changes required! This new feature automatically caches frequently referenced contexts on distributed storage, slashing...</li><li><a href="https://reddit.com//r/LocalLLaMA/comments/1ei31si/new_medical_and_financial_70b_32k_writer_models/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/karpathy/status/1819780828815122505?s=46">Tweet from Andrej Karpathy (@karpathy)</a>: @xlr8harder @hyperbolic_labs Wow. Is this the closest we&#39;ve come to a version of Roko&#39;s basilisk playing out as not an intellectual exercise.</li><li><a href="https://www.youtube.com/watch?v=pLPJoFvq4_M">LangGraph Studio: The first agent IDE</a>: LLMs have paved the way for the development of new types of agentic applications — and as LLM applications evolve, so must the tooling needed to efficiently ...</li><li><a href="https://x.com/steve8708/status/1819448686424084892?s=46">Tweet from Steve (Builder.io) (@Steve8708)</a>: LLMs are literally the most unreliable technology of all time (followed by **ing bluetooth)  After an absurd amount of trial and error, we&#39;ve internally created a set of rules for make LLMs consid...</li><li><a href="https://x.com/BenjaminKlieger/status/1819803984707928425">Tweet from Benjamin Klieger (@BenjaminKlieger)</a>: Imagine you want to learn about the technology behind LLMs. You instantly get an 100 page book with chapters, content, and structure.   What if you find the language too technical? You can change the ...</li><li><a href="https://x.com/cis_female/status/1820305397821112726?s=61">Tweet from sophia (@cis_female)</a>: Updated rule of thumb: a billion parameters * trillion tokens is $5,000  so gemma-2b was 2b parameters @ 6t tokens=$60,000.  llama-3.1-405b (405b params * 15T tokens) cost ~$30,000,000.  Quoting sophi...</li><li><a href="https://www.youtube.com/watch?v=dFzSXbjV054">Triple H Entrance Video</a>: Triple H Entrance VideoMore WWE - http://www.wwe.com/</li><li><a href="https://youtu.be/kw-9_Yzc_40?si=OgN6drsQn0TR6d4j">Randy Orton BADASS!! Entrance 2008 HD</a>: Randy Orton Best Entrance 12.29.08Copyright WWE</li><li><a href="https://github.com/lm-sys/arena-hard-auto">GitHub - lm-sys/arena-hard-auto: Arena-Hard-Auto: An automatic LLM benchmark.</a>: Arena-Hard-Auto: An automatic LLM benchmark. . Contribute to lm-sys/arena-hard-auto development by creating an account on GitHub.</li><li><a href="https://github.com/Nutlope/turboseek">GitHub - Nutlope/turboseek: An AI search engine inspired by Perplexity</a>: An AI search engine inspired by Perplexity. Contribute to Nutlope/turboseek development by creating an account on GitHub.</li><li><a href="https://youtu.be/iMwepyyaj8I">Developing the RISC-V Framework Laptop Mainboard</a>: Nirav &amp; Hyelim sit down at Framework HQ SF to talk about all things RISC-V and DeepComputing.RISC-V Mainboard: https://frame.work/products/deep-computing-ris...</li><li><a href="https://lmsys.org/blog/2024-04-19-arena-hard/">From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline | LMSYS Org</a>: &lt;p&gt;Building an affordable and reliable benchmark for LLM chatbots has become a critical challenge. A high-quality benchmark should 1) robustly separate model...</li><li><a href="https://reddit.com//r/LocalLLaMA/comments/1ei31si/new_medical_an">Reddit - Dive into anything</a>: no description found</li><li><a href="https://asciinema.org/a/98Jodbg6ERtNQsKdvFJNfOM33">untitled</a>: Recorded by wesen3000</li><li><a href="https://x.com/gdb/status/1790869434174746805?s=46">Tweet from Greg Brockman (@gdb)</a>: A GPT-4o generated image — so much to explore with GPT-4o&#39;s image generation capabilities alone. Team is working hard to bring those to the world.</li><li><a href="https://x.com/jonathanross321/status/1820501857741246859?s=46">Tweet from Jonathan Ross (@JonathanRoss321)</a>: What are we doing with this capital? Originally we intended to raise $300M which was going to allow us to deploy 108,000 LPUs into production by end of Q1 2025. We raised 2x that, so we&#39;re also ex...</li><li><a href="https://x.com/techmeme/status/1820416321068384572?s=46">Tweet from Techmeme (@Techmeme)</a>: AI chip startup Groq raised a $640M Series D led by BlackRock at a $2.8B valuation, up from $1B after raising $300M in 2021, and adds an Intel executive as COO (@vandermey / Bloomberg)  https://www.bl...</li><li><a href="https://www.youtube.com/watch?v=9BHQvQlsVdE">[EEML&#39;24] Sander Dieleman - Generative modelling through iterative refinement</a>: no description found</li><li><a href="https://x.com/datnofact/status/1820213413319962975?s=61">Tweet from DatNoFact ↗ (@datnofact.bsky.social) (@datnofact)</a>: hello I&#39;m new to the stock market is it good when the intel ceo starts praying  Quoting Pat Gelsinger (@PGelsinger)   “Let your eyes look straight ahead; fix your gaze directly before you. Give ca...</li><li><a href="https://youtu.be/dDQAYmObK-Y?si=r5b7hXes4CGsAEz2">Neural Notes: KTO - Helping AI make decisions more like a human</a>: In this episode of Neural Notes, Kawin Ethayarajh of Stanford AI Lab (SAIL) talks to Sandeep Bhadra and Simon Tiu of Vertex Ventures US explains his research...</li><li><a href="https://x.com/russelljkaplan/status/1820460524460802256?s=46">Tweet from Russell Kaplan (@russelljkaplan)</a>: Predictions for the future of software engineering:</li><li><a href="https://github.com/go-go-golems/go-go-labs/tree/main/cmd/apps/dripcat">go-go-labs/cmd/apps/dripcat at main · go-go-golems/go-go-labs</a>: GO GO EXPERIMENTAL LAB. Contribute to go-go-golems/go-go-labs development by creating an account on GitHub.</li><li><a href="https://x.com/alexandr_wang/status/1819086525499621494">Tweet from Alexandr Wang (@alexandr_wang)</a>: 1/Gemini 1.5 Pro 0801 is the new best model (tops LMSYS, SEAL evals incoming)  Key considerations 1—OpenAI, Google, Anthropic, & Meta all right ON the frontier 2—Google has a long-term compute edge w/...</li><li><a href="https://x.com/AmgadGamalHasan/status/1819562079193301002">Tweet from Amgad Hasan (@AmgadGamalHasan)</a>: This post is self-contradictory as it: 1. Claims labs (incl google) released models at the same time because they got their hardware from Nvidia at the same time. 2. Claims google has a compute advant...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1269185787761791060)** (1 messages): 

> - `Discord Notification Signups`
> - `Long-lived Threads` 


- **Sign Up for Notifications to Enhance Experience**: A reminder was issued to **sign up for notifications** and comment in threads to get the most out of the Discord community.
   - Engaging in these discussions ensures members are well-informed and connected with ongoing conversations.
- **Utilize Long-lived Threads**: Members were encouraged to participate in the **long-lived threads** where valuable information continually gets added.
   - These threads serve as a resource for ongoing discussions and updates, enhancing collaborative knowledge.


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1269022115382951957)** (72 messages🔥🔥): 

> - `Cody vs Cursor`
> - `Aider.nvim Features`
> - `Claude's Sync Folder`
> - `Context Management in AI Tools`
> - `Composer's Inline Edits` 


- **Cody and Cursor Comparison**: Discussions revealed that Cody from Sourcegraph is praised for its context-awareness and ease of use compared to Cursor, which some find complicated with context management.
   - *Cody allows users to index repositories* and mention them in prompts for better contextual responses.
- **Aider.nvim Functionalities**: Aider.nvim allows users to add context by placing it in buffers, with the option to scrape URLs automatically for documentation, though it can feel somewhat janky.
   - Users noted that they can *delete buffers to remove context* but faced challenges with maintaining relevant context in Cursor.
- **Claude's New Sync Feature**: Anthropic is reportedly working on a Sync Folder feature for Claude Projects, allowing batch uploads from local folders.
   - This new functionality is seen as a significant enhancement for managing files within the project easily.
- **Context Management Challenges**: Members expressed difficulties with context management in tools like Cursor and its retrieval processes that sometimes yield irrelevant information.
   - Some users suggested that *using specific commands* in Composer helps to manage context more effectively.
- **Composer's Features Shine**: It's noted that Composer's predictive capabilities, like guessing where edits are needed and providing inline edit functionality, are strong points that users enjoy.
   - The community feels that Composer could potentially change the game in AI-assisted coding workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1816945228869206260">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Anthropic is working on a Sync Folder feature for Claude Projects 👀  There you can select a local folder to get your files uploaded in a batch.</li><li><a href="https://sourcegraph.com/blog/how-cody-provides-remote-repository-context">How Cody provides remote repository awareness for codebases of every size</a>: Cody’s context awareness scales to every size codebase, from the smallest startups to the biggest enterprises, using the Sourcegraph platform.</li><li><a href="https://sourcegraph.com/blog/how-cody-understands-your-codebase">How Cody understands your codebase</a>: Context is key for AI coding assistants. Cody uses several methods of context fetching to provide answers and code relevant to enterprise-scale codebases.
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1269012342159572992)** (3 messages): 

> - `LLM as Judge`
> - `Synthetic Dataset Generation`
> - `WizardLM Papers`
> - `Sparsely-Activated Mixture-of-Experts` 


- **Recommendations on LLM as Judge and Dataset Generation**: A user inquired about must-reads or surveys related to current trends in **LLM as Judge** and synthetic dataset generation, especially focusing on instruction and preference data.
   - Another member recommended the **latest two papers from WizardLM** as a starting point.
- **Insights on Sparsely-Activated Mixture-of-Experts**: [A paper](https://arxiv.org/abs/2303.01610) discusses the challenges of gigantic transformers, highlighting issues like exorbitant resource consumption and parameter redundancy.
   - It introduces **SMoE-Dropout**, a new training framework aimed at addressing scalability issues in sparsely-activated **Mixture-of-Experts** models, enhancing training efficiency.



**Link mentioned**: <a href="https://arxiv.org/abs/2303.01610">Sparse MoE as the New Dropout: Scaling Dense and Self-Slimmable Transformers</a>: Despite their remarkable achievement, gigantic transformers encounter significant drawbacks, including exorbitant computational and memory footprints during training, as well as severe collapse eviden...

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

not_lain: bro is flexing 
https://x.com/yusufdikec/status/1820186367030128955
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1269046429595472015)** (3 messages): 

> - `VRAM calculation script`
> - `Black Forest Labs launch`
> - `FLUX.1 models` 


- **Efficient VRAM Calculation Made Easy**: A user shared a Ruby script available [here](https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763) that efficiently calculates VRAM requirements for LLM models based on bits per weight and context length.
   - This script allows users to determine VRAM needs, maximum context based on available VRAM, and the highest bits per weight that can be run in one command.
- **SOTA Text-to-Image Model Unveiled**: The **Latent Diffusion Team** announced the launch of _Black Forest Labs_, introducing the **FLUX.1** suite for advanced text-to-image synthesis, aimed at pushing generative AI boundaries.
   - The team emphasizes their mission to improve creativity and efficiency in generative AI, aiming to set an industry standard for generative media and make their models widely accessible [here](https://blackforestlabs.ai/announcing-black-forest-labs/).
- **Official FLUX GitHub Repository Launched**: A new [GitHub repository](https://github.com/black-forest-labs/flux) for the FLUX.1 models has been created, allowing community contributions to the project's development.
   - This repo is dedicated to providing official inference for FLUX.1, enhancing collaborative efforts in the generative AI space.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blackforestlabs.ai/announcing-black-forest-labs/">Announcing Black Forest Labs</a>: Today, we are excited to announce the launch of Black Forest Labs. Deeply rooted in the generative AI research community, our mission is to develop and advance state&#x2d;of&#x2d;the&#x2d;art generati...</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1ehoqmt/script_calculate_vram_requirements_for_llm_models/">[Script] Calculate VRAM requirements for LLM models</a>: Script is here: https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763 For a while I have been trying to figure out which quants I can...</li><li><a href="https://github.com/black-forest-labs/flux">GitHub - black-forest-labs/flux: Official inference repo for FLUX.1 models</a>: Official inference repo for FLUX.1 models. Contribute to black-forest-labs/flux development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1269016900214591538)** (137 messages🔥🔥): 

> - `Claude Sonnet 3.5 Performance`
> - `Training AI Models`
> - `DistillKit Release`
> - `Mistral 7B MoEification`
> - `405B Model Hosting` 


- **Concerns Over Claude Sonnet 3.5**: Users have noted that **Claude Sonnet 3.5** seems to be underperforming in specific tasks compared to the earlier version, pointing to potential issues with updates or optimizations.
   - Concerns include increased hallucination rates and basic errors, such as mistakes in basic algebra or logical reasoning.
- **Fragmented Training Leads to Issues**: One user experienced a complete failure of their model after training on three separate datasets with a very small learning rate.
   - This led to discussions on **overfitting** and **catastrophic forgetting**, suggesting that the approach may have concentrated errors from individual datasets.
- **Introduction of DistillKit**: Arcee AI announced the release of **DistillKit**, an open-source tool aimed at creating smaller, powerful models by distilling knowledge from larger models.
   - This toolkit focuses on optimizing models to be efficient and accessible, combining traditional training techniques with novel distillation methods.
- **Innovative Mistral 7B MoEification**: A new model, **Mistral 7B MoEified**, enables slicing of individual layers into multiple experts for more coherent model behavior.
   - The author explained the methodology behind this approach, allowing models to utilize an equal share of available experts during processing.
- **Hosting 405B Models**: Users discussed potential services hosting the **405B model**, with **Hyperbolic Labs** as the sole provider of Llama 3.1 405B on **OpenRouter**.
   - This model is noted for its low cost relative to other offerings, indicating growing interest in accessing advanced AI resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/nisten/status/1818536486662271167">Tweet from nisten (@nisten)</a>: @reach_vb @skunkworks_ai got mad that no one&#39;d share bitnet code, so I rawdogged it the straight off the paper. but it wouldn&#39;t converge. so then I kept autofrankensteining the layers of smolL...</li><li><a href="https://blog.arcee.ai/announcing-distillkit/?utm_campai">Announcing DistillKit for creating &amp; distributing SLMs</a>: First, Arcee AI revolutionized Small Language Models (SLMs) with Model Merging and the open-source repo MergeKit. Today we bring you another leap forward in the creation and distribution of SLMs with ...</li><li><a href="https://x.com/shannonnullcode/status/1819928712185348278?s=46&t=j99rfSSw_U3piCD9F8qGiQ">Tweet from Shannon Code (@shannonNullCode)</a>: Deep thoughts.  (GenAi unifying physics?)</li><li><a href="https://x.com/hyperbolic_labs/status/1819509384558661811">Tweet from Hyperbolic (@hyperbolic_labs)</a>: Hyperbolic is now the sole provider of Llama 3.1 405B (base) on OpenRouter, offering it at a significantly lower cost than anywhere else. 🌪️  We can&#39;t wait to see what researchers and developers ...</li><li><a href="https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x">kalomaze/Mistral-7b-MoEified-8x · Hugging Face</a>: no description found</li><li><a href="https://paperswithcode.com/task/optical-character-recognition">Papers with Code - Optical Character Recognition (OCR)</a>: **Optical Character Recognition** or **Optical Character Reader** (OCR) is the electronic or mechanical conversion of images of typed, handwritten or printed text into machine-encoded text, whether fr...</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8?inference_api=true">meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/migtissera/Tess-3-Llama-3.1-405B">migtissera/Tess-3-Llama-3.1-405B · Hugging Face</a>: no description found</li><li><a href="https://x.com/nisten/status/1819745389014024530">Tweet from nisten (@nisten)</a>: 1 cpu core - 160 Tokens per second</li><li><a href="https://huggingface.co/mlabonne/Llama-3.1-70B-Instruct-lorablated">mlabonne/Llama-3.1-70B-Instruct-lorablated · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF">bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-exl2">bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-exl2 · Hugging Face</a>: no description found</li><li><a href="https://blog.arcee.ai/announcing-distillkit/?utm_campaign=content&utm_source=Marktechpost&utm_medium=Blog&utm_term=Knowledge%20Distillation&utm_content=Blog%201">Announcing DistillKit for creating &amp; distributing SLMs</a>: First, Arcee AI revolutionized Small Language Models (SLMs) with Model Merging and the open-source repo MergeKit. Today we bring you another leap forward in the creation and distribution of SLMs with ...</li><li><a href="https://github.com/arcee-ai/DistillKit?ref=blog.arcee.ai">GitHub - arcee-ai/DistillKit at blog.arcee.ai</a>: An Open Source Toolkit For LLM Distillation. Contribute to arcee-ai/DistillKit development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1269770529028440115)** (13 messages🔥): 

> - `LLMs Performance in Code Optimization`
> - `Feedback on LLMs' Instruction Following`
> - `Fine-tuning Llama 3.1 for Deployment`
> - `Training Script Libraries`
> - `Merging Models Issues` 


- **LLMs excel at optimizing given code**: LLMs like **Claude** perform well when provided with specific instructions, such as “**optimize this code**,” yielding effective results.
   - However, when given vague prompts like “**make my code better**,” they may miss subtle bugs or edge cases.
- **LLMs struggle with vague instructions**: It was noted that LLMs can *struggle to arrive at conclusions on their own*, particularly with less detailed guidance.
   - For example, a vague inquiry can lead to overlooking important issues during code optimization.
- **Issues deploying fine-tuned Llama 3.1**: **Admiral_snow** faced challenges deploying their finely-tuned Llama 3.1 model using AWQ quantization after merging it with the bnb format.
   - They suspected the error arose from trying to merge the model in bnb format instead of using fp16/bf16 Hugging Face weights.
- **Merging models and GPU limitations**: The discussion highlighted struggles with merging models without exceeding the GPU capacity, particularly when using the **H100**.
   - Alternatives like **using regular RAM** for model merging were suggested for better efficiency.
- **Environment for fine-tuning LLMs**: A question arose regarding whether most developers utilize libraries or write custom training scripts for model fine-tuning.
   - A popular library mentioned was **Axolotl**, indicating a trend towards leveraging established tools.


  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1269087185022222426)** (3 messages): 

> - `Direct Messaging in Discussions`
> - `Temperature Settings Inquiry` 


- **Direct Messaging Offer**: A member extended an offer to discuss a topic via Direct Message, indicating openness to further conversation.
   - This suggests ongoing discussions where privacy or further clarification might be preferred.
- **Temperature Settings Question**: Aarush inquired if the Direct Message was regarding **temperature settings**, indicating the need for specific discussion.
   - This highlights a focused interest in optimizing or adjusting certain parameters related to the topic of conversation.


  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1269454084340060161)** (80 messages🔥🔥): 

> - `Netlify Integration`
> - `Quarto Tasks Organization`
> - `GitHub Actions Automation`
> - `Markdown Formatting Consistency`
> - `Website Deployment Issues` 


- **Netlify Integration for Automated Builds**: A member is working on integrating [Netlify](https://github.com/apps/netlify) into the repository to automate builds and simplify the deployment process.
   - Another member is ready to assist in setting it up, requiring installation of the Netlify app and configuration of the repository.
- **Concerns About Quarto Files Cluttering Repo**: Members discussed the extensive presence of Quarto files at the top level of the repository, causing confusion regarding organization.
   - One member suggested deploying Quarto on a separate branch to minimize clutter, while others emphasized the importance of clear documentation.
- **Automating Builds with GitHub Actions**: A member proposed the use of GitHub Actions to automate the npm build process to ease the workload for contributors.
   - There is general agreement on this approach, highlighting previous positive experiences with GitHub Actions.
- **Markdown Formatting Consistency Issues**: Inconsistencies were noted in how inputs are formatted in Markdown across tasks, with suggestions to standardize on using 'Input:' instead of 'Inputs'.
   - Members acknowledged the importance of maintaining proper formatting to ensure smooth operation of the parser and submission via Pull Requests.
- **Website Deployment Not Reflecting New Tasks**: A member observed that new tasks added in Quarto format were not appearing on the website, leading to a realization that they needed to be listed in the `_quarto.yml` file.
   - This oversight was quickly identified by another member, who offered instructions for rectifying the issue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/apps/netlify">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://reasoning.nousresearch.com/">Open Reasoning Tasks</a>: no description found</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/commit/279f36cc43e4cf6b047cd427929c37427899795a">Create currys-paradox.qmd · NousResearch/Open-Reasoning-Tasks@279f36c</a>: add currys paradox task to quarto chapters</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/blob/main/tasks/currys-paradox.md">Open-Reasoning-Tasks/tasks/currys-paradox.md at main · NousResearch/Open-Reasoning-Tasks</a>: A comprehensive repository of reasoning tasks for LLMs (and beyond) - NousResearch/Open-Reasoning-Tasks</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks">GitHub - NousResearch/Open-Reasoning-Tasks: A comprehensive repository of reasoning tasks for LLMs (and beyond)</a>: A comprehensive repository of reasoning tasks for LLMs (and beyond) - NousResearch/Open-Reasoning-Tasks</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/commit/240b0e5032f47265439e0e80a1864ab529ab62a7">Create stack-based-reasoning.qmd · NousResearch/Open-Reasoning-Tasks@240b0e5</a>: add qmd for stack based reasoning task
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1269054386982883349)** (7 messages): 

> - `Chatroom improvements`
> - `New models from OpenRouter`
> - `Azure routing for Mistral`
> - `Pricing structure for Gemini Pro`
> - `Yi endpoints` 


- **Chatroom Rebranding and Features**: The Playground has been rebranded as the [Chatroom](https://openrouter.ai/chat), featuring local chat saving and a simplified UI for easier room configuration.
   - Users can now explore the enhanced functionality while enjoying a more user-friendly interface.
- **Exciting New Models Launched**: OpenRouter introduced new models including **Llama 3.1 405B BASE** and a free **Llama 3.1 8B**, which can be accessed through their [model page](https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free).
   - Additionally, **Mistral Nemo 12B Celeste** and the **Llama 3.1 Sonar family** are now available for various applications.
- **Mistral Models Routed to Azure**: The **Mistral Large** and **Mistral Nemo** models are now routed to [Azure](https://openrouter.ai/models/mistralai/mistral-large) for increased accessibility.
   - This move enhances the available infrastructure for users needing robust performance from these AI models.
- **Gemini Pro 1.5 Experimental Now Live**: The **Gemini Pro 1.5 Experimental** model is available at [this link](https://openrouter.ai/models/google/gemini-pro-1.5-exp), requiring users to enable training in their settings.
   - This model is served by AIStudio, differing from the usual Vertex routing, and users must update settings at [privacy settings](https://openrouter.ai/settings/privacy) to access it.
- **Clarifying Gemini Pricing Structure**: The pricing for the **Gemini model** is currently set with 1 token equal to 1 character, as confirmed by a community member.
   - There are plans to shift to a token-based pricing system soon, contingent on data reconciliation efforts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-pro-1.5-exp">Gemini Pro 1.5 (0801) - API, Providers, Stats</a>: Gemini 1.5 Pro (0801) is an experimental version of the [Gemini 1. Run Gemini Pro 1.5 (0801) with API</li><li><a href="https://openrouter.ai/models/mistralai/mistral-large>">Mistral Large - API, Providers, Stats</a>: This is Mistral AI&#x27;s flagship model, Mistral Large 2 (version `mistral-large-2407`). It&#x27;s a proprietary weights-available model and excels at reasoning, code, JSON, chat, and more. Run Mistr...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-nemo>">Mistral Nemo - API, Providers, Stats</a>: A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA.  The model is multilingual, supporting English, French, German, Spanish, Italian, Portuguese, Chin...</li><li><a href="https://openrouter.ai/models/01-ai/yi-large>">Yi Large - API, Providers, Stats</a>: The Yi Large model was designed by 01.AI with the following usecases in mind: knowledge search, data classification, human-like chat bots, and customer service. Run Yi Large with API</li><li><a href="https://openrouter.ai/models/01-ai/yi-large-turbo>">Yi Large - API, Providers, Stats</a>: The Yi Large model was designed by 01.AI with the following usecases in mind: knowledge search, data classification, human-like chat bots, and customer service. Run Yi Large with API</li><li><a href="https://openrouter.ai/models/01-ai/yi-large-fc>">Yi Large - API, Providers, Stats</a>: The Yi Large model was designed by 01.AI with the following usecases in mind: knowledge search, data classification, human-like chat bots, and customer service. Run Yi Large with API</li><li><a href="https://openrouter.ai/models/01-ai/yi-vision>">Yi Vision - API, Providers, Stats</a>: The Yi Vision is a complex visual task models provide high-performance understanding and analysis capabilities based on multiple images.  It&#x27;s ideal for scenarios that require analysis and interp...</li><li><a href="https://openrouter.ai/settings/preferences">Settings | OpenRouter</a>: Manage your accounts and preferences</li><li><a href="https://openrouter.ai/docs/parameters-api">Parameters API | OpenRouter</a>: API for managing request parameters</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b">Llama 3.1 405B (base) - API, Providers, Stats</a>: Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. Run Llama 3.1 405B (base) with API</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free">Llama 3.1 8B Instruct (free) - API, Providers, Stats</a>: Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. Run Llama 3.1 8B Instruct (free) with API</li><li><a href="https://openrouter.ai/models/nothingiisreal/mn-celeste-12b">Mistral Nemo 12B Celeste - API, Providers, Stats</a>: A specialized story writing and roleplaying model based on Mistral&#x27;s NeMo 12B Instruct. Fine-tuned on curated datasets including Reddit Writing Prompts and Opus Instruct 25K. Run Mistral Nemo 12B...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-large-128k-online">Llama 3.1 Sonar 70B Online - API, Providers, Stats</a>: Llama 3.1 Sonar is Perplexity&#x27;s latest model family. Run Llama 3.1 Sonar 70B Online with API</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-small-128k-online">Llama 3.1 Sonar 8B Online - API, Providers, Stats</a>: Llama 3.1 Sonar is Perplexity&#x27;s latest model family. Run Llama 3.1 Sonar 8B Online with API
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1269163491328000041)** (2 messages): 

> - `Multi-AI answer website launch`
> - `Community support`
> - `Product Hunt engagement` 


- **Launch of Multi-AI Answer Website**: The new [Multi-AI answer website](https://www.producthunt.com/posts/aiswers-com) has launched on Product Hunt, thanks to support from OpenRouter!
   - The team invites users to check it out and is seeking **upvotes and suggestions** from the community.
- **Gratitude for Community Support**: Acknowledgements were given for the ongoing support from OpenRouter, highlighting its importance in the launch process.
   - The message emphasized that **community feedback** and engagement are highly valued during this launch.



**Link mentioned**: <a href="https://www.producthunt.com/posts/aiswers-com"> Aiswers.com - Ai version of Quora - Get feedback from world top AIs | Product Hunt</a>: We brings together the world’s top AI minds to answer your questions. Get instant, personalized answers from a diverse range of AI models and agents, all in one place. Developers can also integrate an...

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1269007129243684934)** (150 messages🔥🔥): 

> - `Model Comparisons`
> - `API Rate Limits`
> - `Image Classification Models`
> - `Image Quality`
> - `Pricing Strategies` 


- **Yi-Vision vs. FireLLaVA Performance**: Users reported differing performance results when testing **Yi-Vision** against **FireLLaVA**, with some stating that Yi-Vision performed better despite both being similarly priced.
   - *Yi-Vision* was noted to have made minor mistakes, while FireLLaVA had larger errors in the same tests.
- **Pricing Changes for Google Gemini Flash**: It was announced that the price for **Google Gemini 1.5 Flash** would be cut in half on the 12th, making it more competitive against other models like **Yi-Vision** and **FireLLaVA**.
   - Users expressed excitement about the potential for cheaper options that would enable detailed automatic captioning for user-generated content.
- **API Rate Limit Handling**: When users exceed their API rate limit, they receive a **429 response** indicating too many requests.
   - Discussion confirmed that it is essential to monitor activity to avoid rate limit issues when using OpenRouter.
- **Token Counting for Image API Calls**: There was a question about how to calculate token limits for images in API calls, with clarifications about differences in token counting between services.
   - It was noted that Google's Gemini treats tokens and characters equally, which can affect how costs are estimated for image processing.
- **Costs and API Call Pricing**: Users inquired if OpenRouter API calls return cost information directly, with feedback that costs can typically be retrieved using the generation endpoint after the call.
   - Concerns about providing pay-as-you-go access led to the understanding that API call costs can be calculated retroactively based on detailed request data.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: See how you&#x27;ve been using models on OpenRouter.</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">Responses | OpenRouter</a>: Manage responses from models</li><li><a href="https://huggingface.co/docs/transformers/en/tasks/image_classification">Image classification</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.15668v1">What Do You See? Enhancing Zero-Shot Image Classification with Multimodal Large Language Models</a>: Large language models (LLMs) has been effectively used for many computer vision tasks, including image classification. In this paper, we present a simple yet effective approach for zero-shot image cla...</li><li><a href="https://x.com/alexalbert__/status/1820520897465246194?t=TcISXOeVcSjBAIvxhLGcow&s=19">Tweet from Alex Albert (@alexalbert__)</a>: My full talk from the AI Engineer summit a few weeks ago is now up on YouTube! https://x.com/aiDotEngineer/status/1820484842594930939  Quoting AI Engineer (@aiDotEngineer)   Claude 3.5 Sonnet was the ...</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage</li><li><a href="https://openrouter.ai/models/fireworks/firellava-13b">FireLLaVA 13B - API, Providers, Stats</a>: A blazing fast vision-language model, FireLLaVA quickly understands both text and images. It achieves impressive chat skills in tests, and was designed to mimic multimodal GPT-4. Run FireLLaVA 13B wit...</li><li><a href="https://x.com/OpenRouterAI/status/1819500533553443004">Tweet from OpenRouter (@OpenRouterAI)</a>: Llama 3.1 405B BASE!  It&#39;s here. This is the base version of the chat model released last week. You can use it to generate training data, code completions, and more.  Currently hosted by a new pro...</li><li><a href="https://www.producthunt.com/posts/aiswers-com"> Aiswers.com - Ai version of Quora - Get feedback from world top AIs | Product Hunt</a>: We brings together the world’s top AI minds to answer your questions. Get instant, personalized answers from a diverse range of AI models and agents, all in one place. Developers can also integrate an...</li><li><a href="https://github.com/MinorJerry/WebVoyager">GitHub - MinorJerry/WebVoyager: Code for &quot;WebVoyager: WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models&quot;</a>: Code for &quot;WebVoyager: WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models&quot; - MinorJerry/WebVoyager</li><li><a href="https://github.com/robert-mcdermott/LLM-Image-Classification">GitHub - robert-mcdermott/LLM-Image-Classification: Image Classification Testing with LLMs</a>: Image Classification Testing with LLMs. Contribute to robert-mcdermott/LLM-Image-Classification development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1269077490693968015)** (90 messages🔥🔥): 

> - `Mojo for Data Processing`
> - `CSV vs Parquet`
> - `Database Query Optimization`
> - `Licensing and Open Source Considerations` 


- **Mojo may enhance data processing pipelines**: Discussions highlight the potential of **Mojo** for integrating analytics with database workloads, potentially enabling quicker data handling via JIT compilation and direct file operations.
   - Members mentioned its compatibility with tools like **PyArrow** and **Ibis**, suggesting a promising future for a rich data ecosystem within the **Mojo** framework.
- **Convention debates over CSV and other formats**: Users expressed frustrations about having to work with inefficient **CSV** formats when more efficient formats like **Parquet** or **Arrow** are available, emphasizing performance concerns.
   - It's noted how customer needs often dictate format choices, sometimes leading to unnecessary complexity in data processing.
- **Optimizing database query execution plans**: Discussion revolved around modern **analytical databases** and their capacities for query optimization, focusing on techniques like **NUMA-aware** execution and parallelism.
   - Members also pointed out the relevance of employing advanced compilation strategies for static structures to boost performance in processing.
- **Licensing discussions and open-source intentions**: Members debated appropriate licensing frameworks for software, noting that licenses like **AGPLv3** can be overly restrictive for companies not fully committed to open source.
   - Practical licensing strategies were suggested to maintain transparency and guard against exploitative practices while supporting open-source project visibility.
- **Integration of FPGA technologies with data processing**: The idea of leveraging **FPGA** technologies in combination with formats like **Apache Arrow** was introduced to boost data processing capabilities.
   - Tools like **Fletcher** were mentioned as examples of frameworks that facilitate this integration to enhance overall data handling efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ibis-project.org)">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=9rOefO341sI">The Future Roadmap for the Composable Data Stack</a>: Discover cutting-edge advancements in data processing stacks. Listen in as Wes McKinney dives into pivotal projects like Parquet and Arrow, alongside essenti...</li><li><a href="https://www.youtube.com/watch?v=YrqSp8m7fmk&pp=ygURY3N2IHBlZHJvIGhvbGFuZGE%3D">Efficient CSV Parsing - On the Complexity of Simple Things - Pedro Holanda</a>: DSDSD - THE DUTCH SEMINAR ON DATA SYSTEMS DESIGN: We hold bi-weekly talks on Fridays from 3:30 PM to 5 PM CET for and by researchers and practitioners design...</li><li><a href="https://github.com/abs-tudelft/fletcher">GitHub - abs-tudelft/fletcher: Fletcher: A framework to integrate FPGA accelerators with Apache Arrow</a>: Fletcher: A framework to integrate FPGA accelerators with Apache Arrow - abs-tudelft/fletcher</li><li><a href="https://github.com/mzaks/mojo-csv">GitHub - mzaks/mojo-csv</a>: Contribute to mzaks/mojo-csv development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1269040406536912906)** (48 messages🔥): 

> - `Elixir Error Handling`
> - `Mojo Debugger`
> - `Mojo SIMD Performance`
> - `Thermo Physics Engine`
> - `Variadic Struct Parameters` 


- **Elixir's Non-standard Error Handling**: Members discussed Elixir's challenge where libraries either return error atoms or raise exceptions, making error handling non-standardized.
   - A relevant [YouTube video](https://www.youtube.com/watch?v=Iflu9zEJipQ) featuring Chris Lattner and Lex Fridman was shared, discussing exceptions versus errors.
- **Mojo Debugger Limitations**: A member confirmed that the Mojo debugger currently does not work with VS Code, referring to an existing GitHub issue on the topic.
   - General debugging workflows seem to rely on running the program with print statements rather than using a debugger.
- **Performance Concerns with Mojo SIMD**: Concerns were raised regarding the performance of operations on large SIMD lists in Mojo, which can be slow on certain hardware configurations.
   - Another member mentioned that using a SIMD size around what the CPU is meant to handle can improve performance.
- **Proposed Name for Physics Engine**: A member suggested the name 'Thermo' for a new physics engine written in Mojo, considering a play on words with 'thermojo.'
   - This sparked discussions about the naming flexibility and creative potential within the community.
- **Innovative Variadic Struct Parameters in Mojo**: A member demonstrated using variadic struct parameters with a parametrized `__getattr__` method to create a flexible class structure.
   - They noted that this design pattern could enable a blend between dynamic and non-dynamic typing in Mojo.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/theyre-the-same-picture-the-office-pam-the-office-us-gif-20757621">Theyre The Same Picture The Office GIF - Theyre The Same Picture The Office Pam - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=Iflu9zEJipQ">Exception vs Errors | Chris Lattner and Lex Fridman</a>: Lex Fridman Podcast full episode: https://www.youtube.com/watch?v=pdJQ8iVTwj8Please support this podcast by checking out our sponsors:- iHerb: https://lexfri...</li><li><a href="https://github.com/modularml/mojo/issues/1829">[Feature Request] Debugging Support for Mojo in VS Code and PyCharm · Issue #1829 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I attempted to use breakpoints in VS Code for debuggin...</li><li><a href="https://github.com/MVPavan/mojos/blob/master/learn/dsa/my_queue.mojo">mojos/learn/dsa/my_queue.mojo at master · MVPavan/mojos</a>: Collection of mojo codes. Contribute to MVPavan/mojos development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1269899509555789865)** (12 messages🔥): 

> - `Mojo installation issues on MacOS`
> - `Max installation on Ubuntu`
> - `Documentation on MAX Engine comparison`
> - `PyTorch command-line interface` 


- **Mojo installation struggles on MacOS**: A member is facing persistent installation errors with **Mojo** on **MacOS 15**, even after attempting a reinstall.
   - *Can I completely remove Mojo and MAX traces?* was also posed, hinting at potential system conflicts.
- **Installer Feedback for Max on Ubuntu**: Another member suggested setting `DEBIAN_FRONTEND=noninteractive` for the Max installer on **Ubuntu**, as the interactive post-install GUI becomes unresponsive.
   - This change could enhance the installation experience for users encountering similar issues.
- **Missing Documentation for MAX Engine Comparisons**: A user is searching for a previously available documentation page that compared **MAX Engine** with **PyTorch** and **ONYX** across various models, including **ResNet** and **Mistral-7B**.
   - They request assistance in locating this now-missing webpage.
- **New GitHub Project: PyTorch CLI for LLMs**: A trending GitHub project, [torchchat](https://github.com/pytorch/torchchat), offers a command-line interface for **PyTorch** based LLMs, similar to Max pipelines.
   - It also features a **Streamlit interface**, allowing users to run models locally on servers, desktops, and mobile devices.



**Link mentioned**: <a href="https://github.com/pytorch/torchchat">GitHub - pytorch/torchchat: Run PyTorch LLMs locally on servers, desktop and mobile</a>: Run PyTorch LLMs locally on servers, desktop and mobile - pytorch/torchchat

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1269018650837258271)** (23 messages🔥): 

> - `Claude AI code fixes`
> - `Arch design for performance improvement`
> - `SOTA music generation models`
> - `RIAA and labels relationship`
> - `HDF5 for loading embeddings` 


- **Claude AI can provide code fixes from output.json**: A member mentioned starting a new chat with **Claude AI** to upload the `output.json` map, allowing **Claude** to write code fixes without accessing the actual files. This was supported by a [Medium article](https://medium.com/@mbonsign/codemapper-your-ais-guide-to-understanding-code-ef2bda7f333e) quoting Claude.ai on the evaluation of the output file.
   - *However, there was skepticism* about the empirical evidence supporting its effectiveness.
- **New architecture boosts performance**: A discussion highlighted that creating new architectures can improve performance, especially in scenarios of **user-specific audio classification**. Modifications could involve using **contrastive learning** to output user-invariant features.
   - Additionally, an example was given about adapting architectures for **3D data** to ensure performance remains invariant to translations.
- **Interest in controllable music generation models**: A query about current **SOTA models for music generation** led to a suggestion to search for information regarding an ongoing 'AI music generation lawsuit'.
   - A member expressed a preference for models that can be run **locally** rather than depending on external services.
- **RIAA's role in the music industry**: The discussion centered around the **RIAA** and its relationship with music labels, drawing parallels to the film industry concerning the MPAA. Members shared opinions that the RIAA and labels work together to uphold the current industry structure.
   - Concerns were raised about artists receiving only a small percentage of royalties while the RIAA and labels reap profits, pushing for self-promotion and direct payments to artists.
- **HDF5 for embedding management**: A question was posed about whether **HDF5** is still the go-to method for loading small randomized batches from a large set of embeddings on disk. This indicates ongoing interest in efficiency in managing large datasets.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1269006942777507972)** (75 messages🔥🔥): 

> - `Transformer Parameter Recovery`
> - `Layer Norm Effects on Weights`
> - `NeurIPS 2024 Workshops`
> - `AI Search vs Training Compute`
> - `Meta's Distributed AI Training Network` 


- **Recovering Transformer Parameters from Shuffled Weights**: A discussion emerged around the challenges of recovering original transformer parameters from a shuffled vector, emphasizing the need for knowledge of architecture and training regimes.
   - Members debated the role of regularities in matrix distributions and whether a perfectly trained model should exhibit normal distributions, suggesting that permutation may carry important information.
- **Layer Norm Complications in Weight Ordering**: Concerns were raised about the impact of Layer Norm on distinguishing weights; while some layer norms can be easily assigned, other matrices downstream may behave unpredictably based on training.
   - It was noted that understanding std deviations could help in ordering tensors by depth, although consistent pairing of norms and weights in transformer blocks remains challenging.
- **NeurIPS 2024 Workshops Announcement**: NeurIPS 2024 workshops have been announced, accepting 56 out of 204 submissions, marking a significant increase from last year.
   - The review process utilized OpenReview to align with other submission tracks, though the community expressed disappointment over the lack of data-focused workshops.
- **Advantages of AI Search Over Training Compute**: A document highlighted that search mechanisms present a strong and cost-effective alternative to training compute, suggesting it has been understudied theoretically.
   - The findings imply that leveraging search could improve efficiency in certain applications of AI.
- **Meta's Infrastructure for Distributed AI Training**: Meta outlined its advancements in building a network for large-scale distributed AI training, essential for models like [LLAMA 3.1 405B](https://ai.meta.com/blog/meta-llama-3-1/).
   - The research shared at [ACM SIGCOMM 2024](https://conferences.sigcomm.org/sigcomm/2024/) details the design and operation of one of the world's largest AI networks, addressing communication demands imposed by distributed training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2301.13142">Self-Compressing Neural Networks</a>: This work focuses on reducing neural network size, which is a major driver of neural network execution time, power consumption, bandwidth, and memory footprint. A key challenge is to reduce size in a ...</li><li><a href="https://yellow-apartment-148.notion.site/AI-Search-The-Bitter-er-Lesson-44c11acd27294f4495c3de778cd09c8d">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://blog.neurips.cc/2024/08/02/announcing-the-neurips-2024-workshops/">Announcing the NeurIPS 2024 Workshops &#8211; NeurIPS Blog</a>: no description found</li><li><a href="https://arxiv.org/abs/2407.19200">On Behalf of the Stakeholders: Trends in NLP Model Interpretability in the Era of LLMs</a>: Recent advancements in NLP systems, particularly with the introduction of LLMs, have led to widespread adoption of these systems by a broad spectrum of users across various domains, impacting decision...</li><li><a href="https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/">RoCE networks for distributed AI training at scale</a>: AI networks play an important role in interconnecting tens of thousands of GPUs together, forming the foundational infrastructure for training, enabling large models with hundreds of billions of pa…
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1270115079051808778)** (1 messages): 

> - `Recent Developments in SAEs`
> - `Notation in Transformer Circuits`
> - `Model Simplifications in Transformers`
> - `Information Movement in Attention Heads`
> - `Path Expansion Techniques` 


- **Interest in Recent Developments on SAEs**: A member expressed a desire to catch up on recent developments like **SAEs** after a gap of over a year, seeking a good starting point for learning.
   - They specifically noted interest in the notation found in [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html#notation), raising questions about its current relevance.
- **Exploration of Model Simplifications**: There was a discussion regarding the **Model Simplifications** section, focusing on how simplifications can help in understanding complex architectures.
   - Members highlighted the importance of these simplifications in bridging gaps for newcomers who are trying to grasp recent advancements.
- **Attention Heads as Information Movement**: Participants talked about how **Attention Heads** function independently and additively, facilitating information movement within the model.
   - The concept was linked to practical applications in optimizing model performance and understanding communication between layers.
- **Path Expansion Techniques in Transformer Architecture**: Members discussed the **Path Expansion Trick**, engaging in detail about how it enhances understanding of logits and attention scores.
   - The significance of this technique was noted as a critical approach for deeper analysis of transformer architecture.
- **Understanding One-Layer Models Fully**: A question arose about whether we can claim to 'fully understand' **One-Layer Models**, prompting deeper examination of their implications.
   - This discussion opened avenues for further research and interpretation of simpler models as foundational for future complexity.



**Link mentioned**: <a href="https://transformer-circuits.pub/2021/framework/index.html#notation">A Mathematical Framework for Transformer Circuits</a>: no description found

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1269105600793546762)** (35 messages🔥): 

> - `lm-eval-harness usage`
> - `Cohere API updates`
> - `Hate speech detection opinions`
> - `Model checkpoint evaluation for custom architectures`
> - `Perplexity calculation for HellaSwag prompts` 


- **Instructional resources for lm-eval-harness**: A user inquired about tutorials for converting `lm-harness` command line to Python code, seeking guidance on evaluating custom model architectures using the `lm-eval-harness`. Another member suggested checking the [lm_evaluation_harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness) for examples.
   - A relevant example was shared showing how to override model methods for compatibility with custom models, highlighting the importance of maintaining similar distributions in validation and test sets.
- **Cohere API's transition to new endpoints**: Discussion arose regarding the Cohere API's shift from the `.generate` to `.chat` endpoint, specifically referring to the [migration guide](https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat). It appears users of major platforms like Amazon Sagemaker need not migrate, but confusion remains around the lack of probability support for MCQ evaluations.
   - A member highlighted the implications of this shift and expressed confusion about the removal of the likelihood functionality despite the models being open-sourced, sparking sentiment toward deeper evaluation methods.
- **Debating the efficacy of hate speech detection**: A member expressed skepticism regarding hate speech detection methodologies, referring to them as 'mumbo jumbo'. The discussion evolved with opinions about the vagueness of current safety criteria and the reproducibility crisis in the AI community impacting perceivable progress.
   - Members noted that many articulated takes surprisingly aligned yet differed in emphasis, raising questions on how best to communicate their views in the broader context.
- **Obtaining perplexity for datasets using eval harness**: Queries emerged about calculating perplexities specifically for HellaSwag prompts, with suggestions to modify YAML files for dataset paths. A member pointed out that new configurations enable loading datasets using the `datasets.load_dataset` method, allowing for more streamlined perplexity calculations.
   - Another user affirmed that past tasks could compute perplexities from local files, and a collaborative approach was suggested for adapting the wikitext tasks for current needs.
- **Utilizing HF PretrainedModel in evaluations**: A member inquired about evaluating custom model architectures and was directed towards an example of adapting Huggingface's LM class methods. It was noted that users could pass already-initialized HF `PretrainedModel` to the `HFLM` class for tailored evaluation through custom scripts.
   - This flexibility opens doors for users engaging in actions like quantization or pruning prior to evaluating their models, enhancing the versatility of model assessment methodologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat">Migrating from the Generate API to the Chat API - Cohere Docs</a>: The document outlines the migration from the Generate endpoint to the Chat endpoint for Cohere's generative functionality, advising users to use the Chat endpoint for improved model output quality and...</li><li><a href="https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py">mamba/evals/lm_harness_eval.py at main · state-spaces/mamba</a>: Mamba SSM architecture. Contribute to state-spaces/mamba development by creating an account on GitHub.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2129">Add multiple chat template by KonradSzafer · Pull Request #2129 · EleutherAI/lm-evaluation-harness</a>: This PR adds support for models with multiple chat templates, such as Cohere Command R+, addressing issue #1962. The command line API has been updated to reuse an existing flag to specify the templ...
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1269060922459029505)** (100 messages🔥🔥): 

> - `Error Handling in Ollama`
> - `Using CPU and GPU Together`
> - `Model Specifications and Resource Requirements`
> - `LangChain Memory Management`
> - `Multi-User Features in AWS Lambda` 


- **Ollama Memory Error Troubles**: A user reported encountering a **ValueError** stating the model had run out of memory when invoking a retrieval chain, though their GPU memory usage was low.
   - They are utilizing models like **aya** (4GB) and **nomic-embed-text** (272MB), leading to confusion about the memory error.
- **Mixing CPU and GPU for Inference**: Discussion arose around whether **Ollama** can effectively use both **CPU** and **GPU** resources for inference under heavy load situations.
   - It was noted that the **default behavior** in Ollama should allow fallback to CPU if GPU memory is insufficient, but one user reported it wasn't happening as expected.
- **Recommendations for Model Usage**: Recommendations were made to use **less powerful models** that fit within available GPU memory constraints.
   - Using models that require less RAM is essential for effective resource management and avoiding out-of-memory errors.
- **LangChain Memory Management Discussion**: A user shared insights on how LangChain handles memory and object persistence across sessions, indicating a desire to evaluate inputs for memory efficiency.
   - Specific queries seeking to determine if certain text contain information suitable for memory storage were tested with varying model responses.
- **AWS Lambda Multi-User Features**: Concerns were raised about how to manage multiple user interactions with a RAG (Retrieval-Augmented Generation) Slackbot hosted on **AWS Lambda**.
   - It was confirmed that AWS Lambda can scale to handle multiple invocations simultaneously, but cost implications were acknowledged.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.smith.langchain.com/how_to_guides/monitoring/online_evaluations#mapping-variables>).">Set up online evaluations | 🦜️🛠️ LangSmith</a>: Before diving into this content, it might be helpful to read the following:</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/#tying-it-together>).">How to add chat history | 🦜️🔗 LangChain</a>: In many Q&amp;A applications we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of &quot;memory&quot; of past questions and answers, and some logi...</li><li><a href="https://github.com/ollama/ollama/issues/3509">Can Ollama use both  CPU and GPU for inference? · Issue #3509 · ollama/ollama</a>: What are you trying to do? May I know whether ollama support to mix CPU and GPU together for running on windows? I know my hardware is not enough for ollama, but I still want to use the part abilit...</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/#tool-calling">ChatLlamaCpp | 🦜️🔗 LangChain</a>: This notebook provides a quick overview for getting started with chat model intergrated with llama cpp python.</li><li><a href="https://www.youtube.com/watch?v=rsDlu-9UP00">Local Tool Calling with llamacpp</a>: Tool calling allows an LLM to connect with external tools, significantly enhancing its capabilities and enabling popular architecture like agents. But, tool ...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1269642644469059654)** (6 messages): 

> - `Linear Algebra Concepts`
> - `CPU-Compatible SAM 2 Fork`
> - `Research Assistant Beta Testing`
> - `Importance of Knowledge Graphs`
> - `Self-Supervised Learning in Dense Prediction` 


- **New Article Explores Linear Algebra**: A member shared a new article on [Medium](https://medium.com/@amitsubhashchejara/linear-algebra-part-2-linear-combination-and-span-d5fe65ef0e8f) covering the concept of **linear combinations** and the **span of vectors**.
   - *Stay tuned for more articles on Linear algebra!*
- **Maintaining CPU-Compatible SAM 2 Fork**: A member started maintaining a CPU compatible fork of the **SAM 2 model**, showcasing prompted segmentation and automatic mask generation via notebooks. More work is planned for **GPU compatibility** and creating **API endpoints**.
   - Feedback from the community is welcome on this [GitHub repository](https://github.com/SauravMaheshkar/samv2).
- **Seeking Beta Testers for Research Assistant**: A member is building an advanced research assistant and search engine, looking for beta testers in a few weeks, offering **two months free of premium** with various models including **GPT-4O** and **Mistral Large**.
   - Interested users can find more details and sign up at [Rubik's AI](https://rubiks.ai/).
- **Blog Post on Knowledge Graphs and LLMs**: A member created a blog post discussing the **importance of using entity resolved knowledge graphs** with LLMs. They shared the link to the [LinkedIn post](https://www.linkedin.com/posts/dr-clair-sullivan-09914342_generativeai-entityresolution-artificialintelligence-activity-7225242113150529536-VNiq?utm_source=share&utm_medium=member_desktop).
   - *I hope it is helpful!*
- **AI Insight Tool with ScreenPipe**: A member showcased an AI tool that monitors screens and mics 24/7 to provide insights on time usage, demonstrating functionality using **screenpipe** and an AI agent.
   - The open source project can be found on [GitHub](https://github.com/louis030195/screen-pipe).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.lightly.ai/post/using-self-supervised-learning-for-dense-prediction-tasks">Using Self-Supervised Learning for Dense Prediction Tasks</a>: Overview of Self-Supervised Learning methods for dense prediction tasks such as object detection, instance segmentation, and semantic segmentation</li><li><a href="https://github.com/SauravMaheshkar/samv2">GitHub - SauravMaheshkar/samv2: CPU compatible fork of the official SAMv2 implementation aimed at more accessible and documented tutorials</a>: CPU compatible fork of the official SAMv2 implementation aimed at more accessible and documented tutorials - SauravMaheshkar/samv2</li><li><a href="https://github.com/louis030195/screen-pipe">GitHub - louis030195/screen-pipe: Library to build personalized AI powered by what you&#39;ve seen, said, or heard. Works with Ollama. Alternative to Rewind.ai. Open. Secure. You own your data. Rust.</a>: Library to build personalized AI powered by what you&#39;ve seen, said, or heard. Works with Ollama. Alternative to Rewind.ai. Open. Secure. You own your data. Rust. - louis030195/screen-pipe</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1269762249346781337)** (1 messages): 

> - `Custom AI Voice Assistant`
> - `Sista AI` 


- **Build Your Own AI Voice Assistant in 8 Minutes!**: A tutorial video titled ["Create a custom AI Voice Assistant in 8 minutes! - Powered by ChatGPT-4o (By Sista AI)"](https://www.youtube.com/watch?v=iGX4ARuWZec) demonstrates the step-by-step process to create a custom AI voice assistant for your website.
   - The creator encourages viewers to try it out with a [demo link](https://smart.sista.ai) and a [free signup](https://admin.sista.ai/register).
- **Explore AI Tools for Engagement**: In addition to the voice assistant, there is a growing trend toward enhancing websites with various AI tools that boost customer engagement and personalization.
   - Many developers are integrating AI solutions to improve user experience and satisfaction.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=iGX4ARuWZec">Create a custom AI Voice Assistant in 8 minutes! - Powered by ChatGPT-4o (By Sista AI)</a>: Wanna try it out?🔗 Demo: https://smart.sista.ai🔗 Free Signup: https://admin.sista.ai/registerIn this video, I’ll show you how to create your own AI voice a...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1269062549236748289)** (7 messages): 

> - `ReAct Agents`
> - `Agentic Terraform Assistant`
> - `High-Quality RAG Extraction`
> - `Deploying RAG Applications`
> - `Composio Toolset for AI Agents` 


- **Build ReAct agents using LlamaIndex workflows**: You can create ReAct agents from scratch leveraging LlamaIndex workflows for enhanced internal logic visibility. Check out this detailed guide [here](https://t.co/F0pPEyWJ2w).
   - The ability to ‘explode’ the logic ensures deeper understanding and control over agentic systems.
- **Create a Terraform assistant with LlamaIndex**: Develop a Terraform assistant using LlamaIndex and Qdrant Engine aimed at aspiring AI engineers. This tutorial covers defining an LLM workflow for automated generation [here](https://t.co/ASWNkixboK).
   - With practical insights, it provides a valuable framework for integrating AI with DevOps.
- **Automated extraction for payslips with LlamaExtract**: LlamaExtract allows high-quality RAG on payslips through automated schema definition and metadata extraction. Learn more about this process [here](https://t.co/qoC9RU6Tfm).
   - This method vastly improves data handling capabilities for payroll documents.
- **Deploying and scaling RAG applications**: A comprehensive tutorial by Benito Martin outlines how to deploy and scale your chat application on Google Kubernetes. This resource emphasizes practical deployment strategies [here](https://t.co/ROsGNjhKEM).
   - It addresses the scarcity of content on productionizing RAG applications in detail.
- **Composio offers tools for AI agents**: Composio boasts a toolset for AI agents that includes over 100 integrations like GitHub and Slack. Their upcoming tutorial on building a PR review agent can be found [here](https://t.co/FBdE7bbqFC).
   - Use these tools to streamline your development and collaboration processes.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1269087328547115058)** (89 messages🔥🔥): 

> - `RAG Applications`
> - `OpenAIAgent vs ContextChatEngine`
> - `LlamaIndex Workflows`
> - `Incremental Re-indexing`
> - `Information Extraction Techniques` 


- **Exploring RAG Application Queries**: A user inquired about frameworks for transforming prompts into more effective queries for searching vector DBs in RAG applications, particularly for reasoning queries.
   - Suggested using an LLM to rewrite user queries before retrieval, aiming for enhanced semantic search results, with a provided code example.
- **OpenAIAgent vs ContextChatEngine Performance**: A user compared performance metrics between OpenAIAgent and ContextChatEngine, noting improved pass rates with the latter despite identical settings.
   - The community discussed reasons for discrepancies, speculating that simpler questions may favor the ContextChatEngine due to its direct retrieval method.
- **Using Workflows for Parallel Events**: A developer asked how to trigger parallel sub-events in LlamaIndex workflows and manage their results.
   - Advice was given to utilize asynchronous function calls and an example workflow was shared, emphasizing the need for proper event handling in workflows.
- **Incremental Re-indexing in LlamaIndex**: A newcomer to LlamaIndex questioned whether incremental re-indexing is supported.
   - The community confirmed that it is possible to insert new documents into an existing index without needing to re-index all content, providing example code.
- **Challenges with Arabic PDF Parsing**: A user faced issues with parsing Arabic PDFs, resulting in garbled text outputs.
   - The response suggested potential problems with the PDF file format quality, indicating that issues might stem from the document itself.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_extract">GitHub - run-llama/llama_extract</a>: Contribute to run-llama/llama_extract development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/#advanced-metadata-customization">Using Documents - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/?h=workflow">Workflows - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1269546570022522883)** (2 messages): 

> - `GraphRAG`
> - `LlamaIndex`
> - `Knowledge Graphs`
> - `Question Answering` 


- **GraphRAG with LlamaIndex**: A discussion highlighted the integration of **GraphRAG** with **LlamaIndex** for enhanced **intelligent question answering** capabilities, as detailed in a [Medium article](https://medium.com/ai-advances/graphrag-with-llamaindex-unleashing-the-power-of-knowledge-graphs-for-intelligent-question-ea177a14623e).
   - This approach leverages **knowledge graphs** to improve the context and accuracy of responses in AI applications.
- **Only option left is GraphRAG**: A member expressed that utilizing **GraphRAG** with **LlamaIndex** seems to be the **only option** left moving forward in their current discussion. 
   - This sentiment underscores the growing reliance on innovative integrations to solve complex AI challenges.


  

---



### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1270058688643858645)** (1 messages): 

> - `Bay Area Events` 


- **User expresses absence at the upcoming event**: A member mentioned that they unfortunately won't be attending the upcoming event.
   - They expressed interest in being informed about future events in the **Bay Area**.
- **Request for Bay Area event updates**: Another member requested to be updated about any future events specifically in the **Bay Area**.
   - This highlights the ongoing interest in local gatherings among community members.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1269944391934541845)** (12 messages🔥): 

> - `Noam Shazeer Wikipedia page`
> - `30 Under 30 Awards`
> - `Insider Circles in Tech` 


- **Noam Shazeer is missing a Wikipedia page**: Members discussed the absence of a Wikipedia page for **Noam Shazeer**, who has had a significant tenure at **Google** since 2002.
   - *Wikipedia can be silly*, noted one member, highlighting the ironic lack of recognition for notable figures.
- **Critique of the 30 Under 30 Awards**: A member expressed disdain for the **30 Under 30** awards, suggesting that those who seek such external validation are *special types of people*.
   - The conversation reflected a common sentiment that these honors often cater to an **insider circle** rather than true merit.
- **Perception of Wikipedia and Recognition**: Another member proposed that creating Wikipedia pages for overlooked figures is quite easy, akin to the dynamics of **30 Under 30** nominations.
   - This led to a discussion on the *insider nature* of tech professions and how certain individuals navigate these circles.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1269155709808017438)** (2 messages): 

> - `SFT datasets`
> - `System prompts in AI`
> - `AFM paper discussions` 


- **Need for Diverse System Prompts in SFT Data**: *A member remarked on most open SFT datasets typically being just (prompt, response) pairs*, questioning why a diversity of system prompts isn't included.
   - The underlying idea is that models ought to learn to respond differently to user and system prompts, with system prompts possibly taking precedence.
- **System Prompts Neglected in OS Community**: Another member affirmed that **system prompts** are still largely overlooked in the open-source community, raising a point about their significance.
   - This highlights a broader conversation within the community about the potential shortcomings in dataset designs.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1270053464277258262)** (19 messages🔥): 

> - `Nvidia AI scraping`
> - `Elon Musk vs OpenAI lawsuit`
> - `404 Media's journalism`
> - `Data drama discussions` 


- **Nvidia Scraping Controversy Uncovered**: According to a report by [404 Media](https://www.404media.co/nvidia-ai-scraping-foundational-model-cosmos-project/), Nvidia scraped videos from YouTube among other sources to gather training data for its AI products.
   - Nvidia claims this practice is in **full compliance** with copyright law despite ethical concerns raised in internal discussions.
- **Elon Musk Revives Legal Battle with OpenAI**: [Reuters](https://x.com/Reuters/status/1820442168357495259) reported that Elon Musk has revived his lawsuit against Sam Altman and OpenAI, focusing more on personal grievances this time.
   - Musk is seeking to declare OpenAI's license to Microsoft void and questions whether OpenAI's models amount to **AGI**, potentially leading to a dramatic legal showdown.
- **404 Media Sparking Data Drama Discussions**: Discussion around **404 Media** highlights its thorough journalism, countering clickbait trends by providing verified sources and strong evidence in its reports.
   - Members acknowledged the significance of their articles, yet noted its subscription model might limit wider visibility.
- **Musk Claims Wrongdoing in Funding OpenAI**: In his new lawsuit, Musk claims he wired **$44.5 million** to OpenAI between 2016 and 2020, alleging wrongful conduct during that period.
   - He is seeking **punitive damages** to punish OpenAI as part of a broader narrative of betrayal.
- **Debate on Data Stories Impact**: A member expressed frustration with the recurring nature of data stories, suggesting they appear tedious over time.
   - Another participant countered, emphasizing the importance of evidence in reported cases, reinforcing the notion that the same discussions can yield valuable insights.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.404media.co/nvidia-ai-scraping-foundational-model-cosmos-project/">Leaked Documents Show Nvidia Scraping ‘A Human Lifetime’ of Videos Per Day to Train AI</a>: Internal emails, Slack conversations and documents obtained by 404 Media show how Nvidia created a yet-to-be-released video foundational model.</li><li><a href="https://x.com/AndrewCurran_/status/1820491681831219594">Tweet from Andrew Curran (@AndrewCurran_)</a>: Mr. Musk is again requesting that OpenAI&#39;s license to Microsoft be declared void and again requesting judicial determination as to whether OpenAI&#39;s current models constitute AGI. We may get ou...</li><li><a href="https://x.com/AndrewCurran_/status/1820491691973026164">Tweet from Andrew Curran (@AndrewCurran_)</a>: Between 5/27/2016 - 9/14/2020 Mr Musk wired $44,563,500 to OpenAI. The suit claims repeatedly, in very strong language, that there was wrongful conduct throughout. He is requesting an award of &#39;pu...</li><li><a href="https://x.com/AndrewCurran_/status/1820491625854083432">Tweet from Andrew Curran (@AndrewCurran_)</a>: The new lawsuit reads much more as a tale of personal betrayal than the last one, and has more teeth to it. Mr Altman is repeatedly named throughout, and seems more of a direct focus this time.  Quoti...</li><li><a href="https://x.com/Reuters/status/1820442168357495259">Tweet from Reuters (@Reuters)</a>: Elon Musk revives lawsuit against Sam Altman and OpenAI, filing shows http://reut.rs/4drHJor
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1269026950408700056)** (33 messages🔥): 

> - `100k prompts for Llama update`
> - `Discussions on synthetic data and nemotron`
> - `Ross Taylor interview insights`
> - `OpenAI's market position`
> - `Gradio fork and LLM demos` 


- **Running 100k prompts through Llama**: A member humorously confirmed they would run **100k prompts** today for **Llama** updates, primarily focusing on **updating** old GPT-4 completions.
   - They also mentioned generating **preference data** during this process.
- **Synthetic Data Strategy with Nemotron**: There's an ongoing debate about redoing all **synthetic data** using **Nemotron** for fine-tuning **Olmo** models.
   - One member expressed concern over the name being **hijacked**, while another noted issues with AI2's current direction.
- **Ross Taylor Interview Sparks Discussion**: The upcoming interview with **Ross Taylor** generated excitement, highlighting his ideas on deep learning's potential.
   - The conversation included topics like **AGI** and **ASI**, emphasizing future goals for 2024.
- **OpenAI's Competitive Edge**: Concerns were raised about **OpenAI** potentially falling behind competitors like **Sonnet** and **Llama 405B** if they don't release **GPT-5** soon.
   - Despite this, some members believe OpenAI has a significant advantage due to their brand recognition among **regular users**.
- **AI2's Gradio Fork for LLM Demos**: AI2 announced they open sourced their **Gradio fork** to facilitate better use of **vllm** with a side-by-side chat demo.
   - The project offers lightweight tools for quick LLM demos, accessible on [GitHub](https://github.com/allenai/adapt-demos).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rosstaylor90/status/1788243031570911239)">Tweet from Ross Taylor (@rosstaylor90)</a>: What would you build if you took deep learning seriously?  Ten years ago the answer was AGI: a term that would make you look kooky in research circles. Now every post-ChatGPT startup has AGI as their ...</li><li><a href="https://x.com/MrAhmadAwais/status/1819819517650117105">Tweet from Ahmad Awais (@MrAhmadAwais)</a>: OpenAI will be left behind if they didn’t release GPT5 in a month or so. Developers are actively replacing GPT with Sonnet and Llama 405B.    At Langbase we are seeing over 34% of pipes already moved ...</li><li><a href="https://github.com/allenai/adapt-demos">GitHub - allenai/adapt-demos: Lightweight tools for quick and easy LLM demo&#39;s</a>: Lightweight tools for quick and easy LLM demo&#39;s. Contribute to allenai/adapt-demos development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1269080310176415838)** (4 messages): 

> - `KTO alignment`
> - `Neural Notes interview`
> - `DPO vs KTO performance`
> - `UCLA adaptation of KTO` 


- **KTO Gains Attention in Neural Notes**: The recent [Neural Notes interview](https://youtu.be/dDQAYmObK-Y?si=5UMK1bT-6CyKqgnf) features Kawin Ethayarajh discussing the KTO model and its implications on AI decision-making.
   - Some findings suggest that **KTO** can outperform **DPO** when dealing with noisy data, highlighting its robustness.
- **KTO Claims Significant Performance Improvements**: A comment notes that training with paired DPO data can match or exceed DPO performance when using KTO, even with increased data volume.
   - Additionally, findings from the Orca-Math paper suggest KTO-aligned models are superior, achieving a **20-point** performance boost over DPO.
- **UCLA Adapts KTO to Diffusion Models**: A team from **UCLA** adapted KTO to diffusion models, achieving a preference from humans at a **70-30%** margin against DPO-aligned models.
   - This adaptation emphasizes the practical effectiveness of KTO in real-world applications.
- **KTO Handles Noisy Data Effectively**: It was claimed that KTO’s design allows it to effectively manage noisy datasets, avoiding fitting noise during training, unlike DPO.
   - *“KTO can win out**” when datasets contain enough noise or intransitivity, highlighting its competitive edge.



**Link mentioned**: <a href="https://youtu.be/dDQAYmObK-Y?si=5UMK1bT-6CyKqgnf">Neural Notes: KTO - Helping AI make decisions more like a human</a>: In this episode of Neural Notes, Kawin Ethayarajh of Stanford AI Lab (SAIL) talks to Sandeep Bhadra and Simon Tiu of Vertex Ventures US explains his research...

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1269048817978839142)** (62 messages🔥🔥): 

> - `Synthetic datasets debate`
> - `FLUX model performance`
> - `Curating synthetic images`
> - `Stable Diffusion dataset queries`
> - `Training model concerns` 


- **Discussion on synthetic datasets' value**: Members debated the effectiveness of **synthetic datasets** compared to original ones, noting that while they can accelerate training, they risk being misaligned or of lower quality.
   - Concerns were raised about potential biases and the risk of producing a billion useless images without proper curation, leading to calls for more intentional dataset creation.
- **FLUX model performance on art**: Users shared mixed opinions on the **FLUX** model's ability to generate artistic outputs, noting that while some found success in creating paintings, others were disappointed with its results.
   - Discussion highlighted that using the correct parameters could improve outputs, but overall skepticism remained regarding its effectiveness for artistic styles.
- **Curating synthetic image generation**: A suggestion was made to use a **user-curated image generation interface** to improve the quality of synthetic datasets, arguing that human selection could enhance overall usability.
   - The need for careful curation was emphasized to avoid producing a dataset filled with misaligned samples, impacting the training of new models negatively.
- **Stable Diffusion dataset inquiries**: A user inquired about the availability of a **Stable Diffusion dataset**, claiming the requested source was no longer accessible, which limited their progress.
   - Others chimed in, clarifying that the dataset isn't strictly necessary for running Stable Diffusion and suggesting alternative approaches.
- **Concerns about model training practices**: Debate continued around the ethical implications of training on copyrighted images, with members expressing apprehension over potential **copyright laundering** in synthetic dataset projects.
   - Some suggested that while synthetic data has its merits, the broader community's scrutiny might lead to stricter regulations on training practices.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/StableDiffusion/comments/1ej2txw/flux1_is_actually_quite_good_for_paintings/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/LLmg5IzGF-k">BUD-E V1.0 UPDATE: ALL OPEN SOURCE MODELS &amp; LATENCY ~ 2.8 SEC</a>: WIP: https://github.com/LAION-AI/BUD-E_V1.0</li><li><a href="https://youtu.be/iVRt65aIQ3k">BUD-E Conversation demo &amp; hints how to get it running on your own</a>: https://github.com/LAION-AI/BUD-E_V1.0/</li><li><a href="https://www.nature.com/articles/s41586-024-07566-y">AI models collapse when trained on recursively generated data - Nature</a>: &amp;nbsp;Analysis shows that indiscriminately training generative artificial intelligence on real and generated content, usually done by scraping data from&amp;nbsp;the Internet, can lead to a collap...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1269543355306475591)** (4 messages): 

> - `CIFAR-10 performance`
> - `Complex parameters challenge`
> - `Dropout implementation issues`
> - `Overfitting resolution` 


- **Achieved 80% Validation Accuracy on CIFAR-10**: Hit **80% validation accuracy** on the CIFAR-10 dataset with only **36k parameters**, counting real and imaginary components of complex parameters as separate.
   - *
- **Tweaks Boost Performance**: A few architectural tweaks and a better implementation of dropout were all that were needed to enhance performance significantly.
   - Initial issues arose because **nn.dropout** does not work on complex tensors, leading to initial mistakes in creating a replacement.
- **Overfitting Almost Eliminated**: It turns out **overfitting** is basically gone entirely now after the recent changes.
   - These refinements resulted in a more robust model performance.


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1269148618586198148)** (5 messages): 

> - `Coding Agents`
> - `Golden-Retriever Paper`
> - `Livecoding Events` 


- **Adding a Coding Agent to ChatmanGPT Stack**: A member is seeking **recommendations for a coding agent** to add to the ChatmanGPT Stack.
   - Another member suggested **Agent Zero** as a potential choice.
- **Livecoding in the Voice Lounge**: A member announced their return with a note that it's **game over** for the previous setup and mentioned **livecoding** in the Voice Lounge.
   - This indicates a likely collaborative coding session among members.
- **Golden-Retriever Paper Overview**: A member shared a link to the paper on **Golden-Retriever**, which aims to efficiently navigate industrial knowledge bases, addressing traditional LLM fine-tuning challenges.
   - The paper outlines a **reflection-based question augmentation** step that clarifies jargon and context before document retrieval, significantly enhancing retrieval accuracy.



**Link mentioned**: <a href="https://arxiv.org/abs/2408.00798">Golden-Retriever: High-Fidelity Agentic Retrieval Augmented Generation for Industrial Knowledge Base</a>: This paper introduces Golden-Retriever, designed to efficiently navigate vast industrial knowledge bases, overcoming challenges in traditional LLM fine-tuning and RAG frameworks with domain-specific j...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1269041866335064144)** (42 messages🔥): 

> - `AI NPC Development`
> - `Discord Chat Exporter Tool`
> - `SAP AI Security Issues`
> - `DSPy Optimization Questions`
> - `DSPy in Production` 


- **AI NPCs Patrol Areas and Respond to Players**: A member discussed plans to develop **AI characters** in a C++ game that patrol and interact with players using the **Oobabooga API** for responses.
   - They aim to modify the **'world' node** or extend the NPC class, detailing the necessary programs like account, database, and core to run.
- **Exporting Discord Chats Made Easy**: A user successfully **exported Discord channels** to HTML and JSON using the **DiscordChatExporter tool**, which provides formatted results for further use.
   - With the command to include threads, they noted there were **463 thread files** generated across all channels.
- **SAP's AI Security Vulnerabilities Exposed**: A shared article highlighted how **SAP's AI Core** was vulnerable due to poor Kubernetes configuration, allowing access to other customers' data.
   - Researchers were able to run arbitrary code, **breaking containment**, emphasizing the need for better security practices.
- **Questions on DSPy Optimization Capabilities**: New members sought guidance on **optimizing signatures** and whether DSPy metrics could return numeric values instead of just Boolean.
   - Concerns were raised about **losing context** and whether compilation could be resumed, indicating a learning curve with the platform.
- **Preparing DSPy for Production Use**: One member sought resources for deploying their **DSPy application** into production, highlighting the transition from experimentation to implementation.
   - This reflects a growing interest in making practical use of DSPy tools among users looking for development best practices.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pivot-to-ai.com/2024/07/28/sapwned-saps-ai-cloud-cracked-by-simple-kubernetes-configuration-errors/">SAPwned: SAP’s AI cloud cracked by simple Kubernetes configuration errors</a>: SAP is a massive software-as-a-service business intelligence provider. They have an AI offering, SAP AI Core, which customers can train with internal data. Unfortunately, SAP seems to have implemen…</li><li><a href="https://github.com/Tyrrrz/DiscordChatExporter/blob/master/.docs/Getting-started.md">DiscordChatExporter/.docs/Getting-started.md at master · Tyrrrz/DiscordChatExporter</a>: Exports Discord chat logs to a file. Contribute to Tyrrrz/DiscordChatExporter development by creating an account on GitHub.</li><li><a href="https://github.com/Tyrrrz/DiscordChatExporter/releases">Releases · Tyrrrz/DiscordChatExporter</a>: Exports Discord chat logs to a file. Contribute to Tyrrrz/DiscordChatExporter development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1270029949683892405)** (2 messages): 

> - `Pinecone limitations` 


- **Pinecone's Inefficiency in Vector Search**: A member expressed that **Pinecone doesn't support multi-vector search**, which can lead to inefficiencies in their applications.
   - This limitation suggests that users might need to reconsider their search strategies when integrating Pinecone.
- **Concerns Regarding Late Interactions**: There was a discussion highlighting that reactions to late interactions would be **inefficient** for workflow processes.
   - This raises questions about how to handle user interactions in a timely manner to improve overall productivity.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1269275632190099526)** (33 messages🔥): 

> - `Open Interpreter with Llama`
> - `Hugging Face API setup`
> - `Screenshot command execution`
> - `Speech recognition and translation`
> - `Open Interpreter system prompt` 


- **Open Interpreter successfully runs with local LLM**: A user reported solving the issue of integrating Open Interpreter with a local LLM using LM Studio as a server, gaining access to the OI system prompt.
   - They found the integration interesting and informative.
- **Troubleshooting Hugging Face API in Open Interpreter**: A user encountered difficulties setting up the Hugging Face API integration in Open Interpreter, facing errors despite following documentation.
   - After getting support, they expressed gratitude and hope for resolving the issues.
- **Executing screenshot commands in Open Interpreter**: A user questioned why Open Interpreter generates extensive code instead of executing the screenshot command directly when requested.
   - Another user shared a successful method using the 'screencapture' command, confirming it worked as desired.
- **Implementing native language I/O in Open Interpreter**: A user proposed a method to implement speech recognition in a native language, translating back and forth between English and the native tongue.
   - This approach raised concerns about the potential pitfalls of translation errors, coining it as 'Garbage in Garbage out'.
- **Universal translator mode for Open Interpreter**: One user suggested creating a universal translator mode that would handle bidirectional translation without considering commands.
   - This idea aims to enhance user interaction with Open Interpreter by minimizing errors in command execution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:11434"">no title found</a>: no description found</li><li><a href="https://docs.openinterpreter.com/getting-started/setup">Setup - Open Interpreter</a>: no description found</li><li><a href="https://nerdvittles.com/creating-an-api-key-for-google-speech-recognition/">Creating an API Key for Google Speech Recognition &#8211; Nerd Vittles</a>: no description found</li><li><a href="https://api-inference.huggingface.co">Serverless Inference API</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1269075957051031676)** (1 messages): 

> - `Open Interpreter on Linux`
> - `Electra AI Linux Distro`
> - `AI capabilities in OS`
> - `Linux flavors: Lindoz, Max, Shift` 


- **Exploring Open Interpreter on Linux**: A member has been diving deeper into getting **Open Interpreter** to work on Linux, sharing their recent experiences and progress.
   - The goal is to see if **OI can be freely open** on a Linux OS, exploring its potential.
- **Electra AI: A New Linux Distro for AI**: A member discovered **Electra AI**, a Linux distro with built-in AI capabilities at the OS level, which is free to use on their website.
   - They highlighted that Electra AI offers three flavors: **Lindoz, Max, and Shift**, all available for free.



**Link mentioned**: <a href="https://www.makululinux.com/wp/">Tweet from MakuluLinux &#8211; A Whole World of Possibilities</a>: no description found

  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1269083495389597788)** (19 messages🔥): 

> - `Cohere support contact`
> - `GenAI Bootcamp and Cohere Models`
> - `CORS Issues on Billing Page` 


- **Cohere Support Contact Details**: When asked about support due to **CORS problems** on the billing page, community members suggested emailing [support@cohere.com](mailto:support@cohere.com) for assistance.
   - *Please include your organization's email and additional details in your inquiry.*
- **Exploring Cohere for GenAI Bootcamp**: Andrew Brown is researching **Cohere** to potentially include it in a free **GenAI Bootcamp** targeted at reaching **50K participants** this year.
   - He expressed a need for insights beyond the documentation, highlighting **Cohere's cloud-agnostic capabilities** and diverse model applications.
- **CORS Issues Prevent Payments**: A member reported that the **billing page isn't working** due to a CORS problem on the backend, impacting their ability to pay for services.
   - They are seeking direct contact for support to resolve this billing error.
- **Cohere's Model Features**: In response to inquiries, members noted that Cohere offers **RAG functionalities**, **multilingual options**, and numerous embedding models.
   - They emphasized the ease of integration via **API calls** and mentioned various tools available for developers.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=zA8guDqfv40">AWS Cloud Complete Bootcamp Course</a>: The AWS Cloud Project Bootcamp is a training program to equip you with the skills to design, build, and implement a cloud project.⚠️ If you are having issues...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1270073728105189451)** (7 messages): 

> - `Benchmarking Models`
> - `Validation Splits`
> - `Cohere For AI Community` 


- **Discussion on Benchmarking Models**: A member, @divyaa_a, inquired whether the **validation subset** should remain the same while benchmarking models trained on different splits of the dataset, although the test set was consistent.
   - Another member advised that keeping the **validation set** the same across all models would provide a more controlled comparison for accurate benchmarking.
- **Emphasis on Controlled Comparisons**: The discussion highlighted the importance of controlled comparisons in **benchmarking** to ensure accuracy in assessments across different modeling approaches.
   - The member recommended maintaining **consistency** in validation subsets to enhance the robustness of comparisons.
- **Suggestion to Join Cohere For AI Community**: One member encouraged @divyaa_a to join the **Cohere For AI Community**, suggesting that they would gain valuable insights related to their benchmarking inquiries.
   - This community is noted for engaging in **cutting-edge open-science research and development**.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1270062172432629860)** (4 messages): 

> - `Cohere API payment`
> - `Command R model confusion`
> - `Coral model explanation` 


- **User confusion over Cohere API model selection**: A user expressed concern after paying for the **Cohere API**, expecting to use the **Command R** model but found only the **Coral model** available.
   - They noted that there was no option to select a model prior to payment, seeking assistance to switch to **Command R**.
- **Clarification on model types**: Another member clarified that the **Coral** model is actually a version of **Command R+**.
   - This response aimed to alleviate the user's confusion regarding the model options available post-payment.


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1269880246501507182)** (3 messages): 

> - `Rerank on Azure`
> - `Cohere RAG app models`
> - `Cohere and Fujitsu partnership` 


- **Rerank Available on Azure Models**: Cohere team discussed the new availability of **Rerank** on **Azure Models** and its potential integration with the **RAG app**.
   - There is interest in updating the toolkit to activate Rerank for those using Azure, as detailed in this [blog post](https://cohere.com/blog/introducing-rerank-3-on-microsoft-azure-ai).
- **Models used in Cohere's RAG app**: A member mentioned that they utilize **Command R+**, **Rerank**, and **Embed** models for their RAG app on Azure.
   - This information highlights the reliance on a combination of models to enhance functionality.
- **Cohere and Fujitsu Partnership**: The team also noted the strategic partnership between **Cohere** and **Fujitsu** aimed at providing AI services to Japanese enterprises.
   - For more details, members can refer to the [partnership announcement](https://cohere.com/blog/fujitsu-partnership).



**Link mentioned**: <a href="https://cohere.com/blog/introducing-rerank-3-on-microsoft-azure-ai">Introducing Rerank 3 on Microsoft Azure AI</a>: We are thrilled to announce that Rerank 3 is now available on Microsoft Azure AI Studio. Rerank 3 is a foundation model that enhances existing search systems by reranking the initial search results to...

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1270024181471248506)** (4 messages): 

> - `tinygrad release 0.9.2`
> - `Aurora supercomputer feasibility`
> - `XMX support and OpenCL`
> - `MLPerf benchmarks`
> - `Bounties on tinygrad features` 


- **tinygrad 0.9.2 introduced with exciting updates**: The Monday meeting highlighted updates to **tinygrad 0.9.2**, including features like **faster gemv**, **kernel timing**, and the **CapturedJit** improvements.
   - Additional discussions included topics on **indexing alu**, **uop symbolic**, and enhancing performance on **ResNet**.
- **Exploring tinygrad's viability on Aurora supercomputer**: A member inquired about the feasibility of running **tinygrad** on the **Aurora** supercomputer at Argonne National Laboratory, given the limitations with Intel GPUs.
   - Concerns were raised on compatibility, but discussions hinted at existing support for **OpenCL**, albeit potentially slow.
- **Discussion on enabling XMX support**: There was a mention of someone possibly working on **XMX** support for **tinygrad**, which would enhance functionality for the Intel architecture.
   - *OpenCL* was noted as already working, although effectiveness was uncertain without firsthand experience.
- **Bounties announced for tinygrad enhancements**: Bounties were discussed for various **tinygrad features**, such as **fast sharded llama**, **std one kernel**, and enhancements for **AMX** and **Qualcomm**.
   - The community is incentivized to contribute to these developments, aiming to improve the overall functionality of tinygrad.
- **MLPerf benchmarks overview**: The Monday meeting also covered **MLPerf** benchmarks, including tasks related to **Unet3D**, **BERT**, **RetinaNet**, and **StableDiffusion**.
   - These benchmarks will be critical for evaluating the performance of running tinygrad on various platforms.



**Link mentioned**: <a href="https://github.com/patrick-kidger/jaxtyping">GitHub - patrick-kidger/jaxtyping: Type annotations and runtime checking for shape and dtype of JAX/NumPy/PyTorch/etc. arrays. https://docs.kidger.site/jaxtyping/</a>: Type annotations and runtime checking for shape and dtype of JAX/NumPy/PyTorch/etc. arrays. https://docs.kidger.site/jaxtyping/ - patrick-kidger/jaxtyping

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1269196489029320735)** (25 messages🔥): 

> - `CUDA vs CLANG performance`
> - `Custom kernel execution on tensors`
> - `PyO3 interface error troubleshooting`
> - `ShapeTrackers in codegen`
> - `Tensor push optimization` 


- **CUDA runs slower than CLANG**: A member questioned why running `CUDA=1 pytest test_winograd.py` is slower compared to `CLANG=1 pytest test_winograd.py`, assuming CUDA would be faster than C.
   - This raises concerns about potential issues or inefficiencies in CUDA's execution for certain tests.
- **Using custom kernels on tensors**: A user inquired about a clean method to run a custom kernel on a tensor, referencing a [GitHub file](https://github.com/tinygrad/tinygrad/blob/da61dea1b2ca886b3de07e309efde2a78ac5682a/test/test_custom_function.py#L42-L43) for details.
   - This highlights ongoing discussions about advanced tensor operations in Tinygrad.
- **Recursive limit error in PyO3**: A member reported a recursion limit error when using `tinygrad.nn.state.safe_save` through the PyO3 interface.
   - Advice was given to try `TRACEMETA=0` to potentially resolve the issue, indicating that such tools might not work well with non-CPython implementations.
- **Evaluate ShapeTrackers for optimization**: Discussion emerged regarding the use of symbolic indices within the shapetracker system, questioning if the library employs symbolic shapes.
   - A member suggested that focusing on reducing expression trees might be more beneficial than improving shapetrackers directly.
- **Optimizing tensor value insertion**: A member sought the most efficient method to push a single value (f64) into a tensor, noting inefficiencies with `.cat`.
   - It was suggested to preallocate and then assign to a slice, but issues arose with assertion errors due to non-contiguous tensors.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/59315ffc7839948a032e366ba8d964c345d835ff/tinygrad/tensor.py#L3158-L3189">tinygrad/tinygrad/tensor.py at 59315ffc7839948a032e366ba8d964c345d835ff · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/da61dea1b2ca886b3de07e309efde2a78ac5682a/test/test_custom_function.py#L42-L43">tinygrad/test/test_custom_function.py at da61dea1b2ca886b3de07e309efde2a78ac5682a · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1270120968878293013)** (1 messages): 

> - `PPO Training Recipe`
> - `Qwen2 Model Support` 


- **PPO Training Recipe Added to Torchtune**: The team has introduced an end-to-end **PPO training recipe** to integrate RLHF with torchtune, as detailed in the [GitHub pull request](https://github.com/pytorch/torchtune/pull/1005).
   - *Check it out and try it out!*
- **Qwen2 Models Supported in Training Recipes**: **Qwen2 model support** has been added to training recipes, with the **7B model** now available in the [GitHub pull request](https://github.com/pytorch/torchtune/pull/1143) and smaller versions on the way.
   - Anticipate the upcoming **1.5B** and **0.5B** versions coming soon!
- **Feedback Wanted for Model and Recipe Requests**: The team is inviting users to share their **feature requests** for other models or recipes they'd like to see implemented in torchtune.
   - You can submit your requests through [this GitHub link](https://github.com/pytorch/torchtune).


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1269092750637269065)** (12 messages🔥): 

> - `LLAMA 3 8B INSTRUCT model`
> - `Token generation issues`
> - `Chat formatting in LLAMA 3`
> - `Debugging generation mode` 


- **Running LLAMA 3 Generation with Custom Config**: User successfully ran the LLAMA 3 8B INSTRUCT model with a custom configuration, achieving **12:34 PM** as an output for the current time query.
   - The generation process took **27.19 seconds** with a speed of **12.25 tokens/sec** and used **20.62 GB** of memory.
- **Repeated Text Generation Issue**: User reported that the generated text often repeats **10 times**, and sometimes includes an ending token unexpectedly.
   - Another user suggested a [pull request](https://github.com/pytorch/torchtune/pull/1211) aimed at patching the issue with ending tokens, which is still under review.
- **Exploring Chat Format Impact**: User inquired whether defining a `chat_format` would assist in improving output for the LLAMA 3 model.
   - A responder stated that setting a chat format was not necessary for LLAMA 3.
- **Need for Debugging Mode in Generation**: Concerns were raised about the lack of a generation mode that displays **all tokens**, including those around the conversation like `<|user|>` and `<|assistant|>`.
   - The responder acknowledged this point and mentioned they could potentially add a parameter to the generate script to allow for this feature.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sdk.vercel.ai/playground">AI Playground | Compare top AI models side-by-side</a>: Chat and compare OpenAI GPT, Anthropic Claude, Google Gemini, Llama, Mistral, and more.</li><li><a href="https://github.com/pytorch/torchtune/pull/1211.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1269978058660773949)** (5 messages): 

> - `Model Blurbs`
> - `Llama3 Review` 


- **Concerns about Maintaining Model Blurbs**: Members expressed their worries about providing updated blurbs for models, fearing it might be too hard to maintain.
   - One suggested that a **snapshot from a model card/whitepaper** could serve as a minimal blurb.
- **Reviving Model Discussions**: It was noted that a review would be helpful in reviving discussions about the models, with **Llama3** being the only one currently done.
   - A member offered to take a look and possibly **add/update blurbs for other models** if given permission.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1269013466925236276)** (9 messages🔥): 

> - `bitsandbytes installation for ROCm`
> - `AI Nutritionist dataset creation`
> - `DPO vs alternatives in AI training` 


- **bitsandbytes Installation for ROCm Simplified**: [A recent pull request](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1299) enables packaging wheels for **bitsandbytes** on **ROCm**, streamlining the installation process for users.
   - This PR updates the compilation process for **ROCm 6.1** to support the latest **Instinct** and **Radeon** GPUs.
- **Building an AI Nutritionist Needs Datasets**: An individual is developing an **AI Nutritionist** and considers fine-tuning **GPT-4o mini** but seeks suitable nutrition datasets like the [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets.html).
   - Recommendations include potential dataset compilation from **FNDDS**, though it's unclear if it's available on **Hugging Face**.
- **Discussion on DPO Alternatives**: A member questioned whether **DPO** remains the best approach in AI training. Others suggested that alternatives like **orpo**, **simpo**, or **kto** might be superior.
   - This led to an exchange of differing opinions on the effectiveness of various methods in AI model training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/bitsandbytes">bitsandbytes - Overview</a>: bitsandbytes has 6 repositories available. Follow their code on GitHub.</li><li><a href="https://huggingface.co/datasets/Roger21/NutritionFineTune_1?row=45)">Roger21/NutritionFineTune_1 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1299">Enable bitsandbytes packaging for ROCm by pnunna93 · Pull Request #1299 · bitsandbytes-foundation/bitsandbytes</a>: This PR enables packaging wheels for bitsandbytes on ROCm. It updates rocm compilation and wheels build jobs to compile on ROCm 6.1 for latest Instinct and Radeon GPUs. There are also updates to do...
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1270015574436479028)** (6 messages): 

> - `FFT for Large Models`
> - `LORA/QLORA Baseline Testing`
> - `L40S GPUs Performance` 


- **Searching for FFT and Baseline Tests**: A member expressed interest in finding **FFT** or **LORA/QLORA** for experimentation with a **27b model**, mentioning good results with a **9b model** but challenges replicating these with the larger one.
   - *Caseus_* suggested that there might be a **QLORA** version for **Gemma 2 27b** that could work with some adjustments to the learning rate and the latest **flash attention**.
- **Inquiry about L40S GPUs Performance**: A member asked if anyone has trained or served models on **L40S GPUs**, seeking information about their performance.
   - This inquiry highlights interest in the efficiency and capabilities of **L40S GPUs** for AI model training.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1269251406712410132)** (1 messages): 

> - `AI Nutritionist Development`
> - `Existing Nutrition Datasets` 


- **Building an AI Nutritionist**: A member is developing an **AI Nutritionist** and looking to fine-tune a **GPT-4o mini** model on a nutrition or food dataset.
   - They are considering whether to create their own dataset or to utilize existing ones.
- **Datasets Suggested**: They mentioned two datasets: [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets.html) and [NutritionFineTune_1 on HF](https://huggingface.co/datasets/Roger21/NutritionFineTune_1?row=45).
   - The member specifically inquired about the nutritional information of **BARLEY, PEARL (BERAS BELANDA) HORDEUM VULGARE**, providing detailed nutritional content such as **335.0 kcal** per **100 grams**.



**Link mentioned**: <a href="https://huggingface.co/datasets/Roger21/NutritionFineTune_1?row=45)">Roger21/NutritionFineTune_1 · Datasets at Hugging Face</a>: no description found

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1269435376511090752)** (1 messages): 

> - `Triton Conference Registration`
> - `Event Details`
> - `Free Attendance`
> - `Google Form Signup` 


- **Triton Conference Registration Now Open!**: Registration for the **Triton Conference** on **September 17, 2024** at **Meta Campus, Fremont CA** is now open! Sign up via [this Google Form](https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform) to secure your spot.
   - Attendance is **free** but spots are **limited**, so early registration is encouraged.
- **Information Required for Registration**: To register, participants must provide their **email**, **name**, **affiliation**, and **role**. Optional questions include background information and what attendees hope to take away from the conference.
   - Dietary preferences are also captured, offering options like **vegetarian**, **vegan**, **kosher**, and **gluten-free**.
- **Sign in Required for Google Form**: Attendees are prompted to [sign in to Google](https://accounts.google.com/AccountChooser?continue=https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform&service=wise) to save their progress when filling out the registration form. A copy of the responses will be emailed to the address provided.
   - Participants are reminded to never submit passwords through Google Forms to ensure their security.



**Link mentioned**: <a href="https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform">[Signup] Triton Conference Sep 17, 2024 </a>: Meta Campus, Fremont CA - Attendance is free (but spaces are limited)

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1270115495273697293)** (1 messages): 

> - `Llamafile progress`
> - `August project discussions`
> - `sqlite-vec release party`
> - `Machine Learning Paper Talks`
> - `Local AI AMA` 


- **Llamafile brings offline LLMs progress**: Llamafile core maintainer continues to make epic progress on the project, enabling **offline, accessible LLMs** in a single file.
   - This development aims to enhance user accessibility and streamline interactions with large language models.
- **Community shares August projects**: A new discussion started about what everyone is working on for **August**, inviting community members to share their ongoing projects.
   - This is an opportunity to engage with peers and showcase individual contributions within the Mozilla AI community.
- **sqlite-vec release party announcement**: A release party for sqlite-vec is planned where attendees can discuss **features**, try demos, and interact with core maintainer.
   - Participants can join the conversation about the latest developments in sqlite-vec, expected to facilitate rich discussions.
- **Machine Learning Paper Talks come alive**: Upcoming talks include discussions on **Communicative Agents** and **Extended Mind Transformers**, featuring prominent speakers.
   - These events offer insight into cutting-edge research and collaborative exchanges in the machine learning field.
- **Local AI AMA set to provide alternatives**: An AMA session is scheduled with the core maintainer of **Local AI**, an open-source alternative to OpenAI that users can self-host.
   - This event is an excellent chance to learn more about Local AI's capabilities and ask questions directly.



**Link mentioned**: <a href="https://form.typeform.com/to/Cn4md4Oc>)">Discover Typeform, where forms = fun</a>: Create a beautiful, interactive form in minutes with no code. Get started for free.

  

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

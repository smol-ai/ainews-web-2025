---
id: de6fed18-27ad-4a29-829f-3d6a4ec67722
title: >-
  Anthropic's "LLM Genome Project": learning & clamping 34m features on Claude
  Sonnet
date: '2024-05-21T22:47:46.990001Z'
original_slug: ainews-anthropic-cracks-the-llm-genome-project
description: >-
  **Anthropic** released their third paper in the MechInterp series, **Scaling
  Monosemanticity**, scaling interpretability analysis to **34 million
  features** on **Claude 3 Sonnet**. This work introduces the concept of
  **dictionary learning** to isolate recurring neuron activation patterns,
  enabling more interpretable internal states by combining features rather than
  neurons. The paper reveals abstract features related to code, errors,
  sycophancy, crime, self-representation, and deception, demonstrating
  intentional modifiability by clamping feature values. The research marks a
  significant advance in **model interpretability** and **neural network
  analysis** at frontier scale.
companies:
  - anthropic
  - scale-ai
  - suno-ai
  - microsoft
models:
  - claude-3-sonnet
  - claude-3
topics:
  - model-interpretability
  - dictionary-learning
  - neural-networks
  - feature-activation
  - intentional-modifiability
  - scaling
  - mechanistic-interpretability
people:
  - emmanuel-ameisen
  - alex-albert
---


<!-- buttondown-editor-mode: plaintext -->**Dictionary Learning is All You Need.**

> AI News for 5/20/2024-5/21/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**376** channels, and **6363** messages) for you. 
Estimated reading time saved (at 200wpm): **738 minutes**. The Table of Contents and Discord Summaries have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

A relatively news heavy day, with monster funding rounds from **Scale AI** and **Suno AI**, and ongoing reactions to **Microsoft Build** announcements (like [Microsoft Recall](https://x.com/dsiroker/status/1792956339515273537)), but we try to keep things technical here. 

Probably the biggest news is Anthropic's [Scaling Monosemanticity](https://www.anthropic.com/research/mapping-mind-language-model), the third in their modern MechInterp trilogy following from [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html#strategic-ways-out) (2022) and [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (2023). The first paper focused on "Principal Component Analysis" on very small ReLU networks (up to 8 features on 5 neurons), the second applied sparse autoencoders on a real transformer (4096 features on 512 neurons), and this paper now scales up to **1m/4m/34m features on Claude 3 Sonnet**. This unlocks all sorts of intepretability magic on a real, frontier-level model:

 ![image.png](https://assets.buttondown.email/images/74a296cf-65a2-45c6-9c6a-46ad01c4fdb4.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/d96d1cac-e45b-40b0-8011-83b9223c0096.png?w=960&fit=max) 

> Definitely check out [the feature UMAPs](https://transformer-circuits.pub/2024/scaling-monosemanticity/umap.html?targetId=1m_1013764)

Instead of the relatively highfaluting "superposition" concept, the analogy is now "**dictionary learning**", which Anthropic explains as: 

> borrowed from classical machine learning, which **isolates patterns of neuron activations that recur across many different contexts**. In turn, any internal state of the model can be represented in terms of a few active features instead of many active neurons. Just as every English word in a dictionary is made by combining letters, and every sentence is made by combining words, every feature in an AI model is made by combining neurons, and every internal state is made by combining features. (further reading in the [notes](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#related-work-dictionary))

Anthropic's 34 million features encode some very interesting "abstract features", like code features and even [errors](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#assessing-sophisticated-code-error):

 ![image.png](https://assets.buttondown.email/images/8dd74aaf-5d74-4869-af68-55ca90142411.png?w=960&fit=max) 

[sycophancy](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#safety-relevant-sycophancy), [crime/harm](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#safety-relevant-criminal), [self representation](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#safety-relevant-self), and [deception and power seeking](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#safety-relevant-deception):

 ![image.png](https://assets.buttondown.email/images/ca16bd0c-da17-45d1-b6bd-d010bf3f9c8b.png?w=960&fit=max) 

The signature proof of complete interpretability research is intentional modifiability, which Anthropic shows off by clamping features from -2x to 10x its maximum values:

{% if medium == 'web' %}
 ![image.png](https://assets.buttondown.email/images/2b5bdf89-5b41-4350-96df-09b1825efbec.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/874d1492-5ac9-435f-be00-5afb8dea588e.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/92206ea3-5e0d-48d4-9ccc-ef90aedfaf7f.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/b675d446-aa5c-45e3-9528-c00efa8adade.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/646a3f7c-63e0-4e99-8c16-0479d3d73a7f.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/d619163e-0536-4d75-b82e-a145030cdf91.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/0f8d5ee9-d72e-42d3-bd25-bab68efe196d.png?w=960&fit=max) 

{% else %}

> You're reading this on email. We're moving more content to the web version to create more space and save your inbox. **Check out the excerpted diagrams on the [web version]({{ email_url }}) if you wish.**

{% endif %}

Don't miss the breakdowns from [Emmanuel Ameisen](https://x.com/mlpowered/status/1792948212728524917), [Alex Albert](https://x.com/alexalbert__/status/1792936647665107108?s=46&t=90xQ8sGy63D2OtiaoGJuww), [Linus Lee](https://x.com/thesephist/status/1793031719244734923) and [HN](https://news.ycombinator.com/item?id=40429326).

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 


{% endif %}



---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Microsoft Launches Copilot+ PCs for AI Era**

- **Copilot+ PCs introduced as the biggest update to Windows in 40 years**: [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1792624197572653351) noted Copilot+ PCs are the fastest, most powerful AI-ready PCs anywhere, re-inventing PCs for the AI era with the whole stack re-crafted around Copilot.
- **Real-time AI co-creation and camera control demoed on Copilot+ PCs**: [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1792632555797180553) showed Copilot controlling Minecraft gameplay, while [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1792626471552336101) demoed real-time AI co-creation on the PCs. 
- **Copilot+ PCs feature photographic memory and fastest performance**: [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1792637482988614027) highlighted Copilot's photographic memory of everything done on the PC. He also called them the [fastest, most powerful and intelligent Windows PCs ever](https://twitter.com/yusuf_i_mehdi/status/1792620591930826879).

**Scale AI Raises $1B at $13.8B Valuation**

- **Scale AI raises $1B at $13.8B valuation in round led by Accel**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1792905417065914858) announced the funding, stating Scale AI has never been better positioned to accelerate frontier data and pave the road to AGI.
- **Scale AI powers nearly every leading AI model by providing data**: As one of the three fundamental AI pillars alongside compute and algorithms, [@alexandr_wang](https://twitter.com/alexandr_wang/status/1792905420744581597) explained Scale supplies data to power nearly every leading AI model.
- **Funding to accelerate frontier data and pave road to AGI**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1792905424251060575) said the funding will help Scale AI move to the next phase of accelerating frontier data abundance to pave the road to AGI.

**Suno Raises $125M to Build AI-Powered Music Creation Tools** 

- **Suno raises $125M to enable anyone to make music with AI**: [@suno_ai_](https://twitter.com/suno_ai_/status/1792922276683297162) will use the funding to accelerate product development and grow their team to amplify human creativity with technology, building a future where anyone can make music.
- **Suno hiring to build the best tools for their musician community**: Suno believes their community deserves the [best tools](https://twitter.com/suno_ai_/status/1792922276683297162), which requires top talent with technological expertise and genuine love for music. They invite people to join in shaping the future of music.

**Open-Source Implementation of Meta's Automatic Test Generation Tool Released**

- **Cover-Agent released as first open-source implementation of Meta's automatic test generation paper**: [@svpino](https://twitter.com/svpino/status/1792897013920538944) shared Cover-Agent, an open-source tool implementing Meta's February paper on automatically increasing test coverage over existing code bases.
- **Cover-Agent generates unique, working tests that improve coverage, outperforming ChatGPT**: [@svpino](https://twitter.com/svpino/status/1792897013920538944) highlighted that while automatic unit test generation is not new, doing it well is difficult. Cover-Agent only generates unique tests that run and increase coverage, while ChatGPT produces duplicate, non-working, meaningless tests.

**Anthropic Releases Research on Interpreting Leading Large Language Model**

- **Anthropic provides first detailed look inside leading large language model in new research**: In a [new research paper and blog post](https://twitter.com/AnthropicAI/status/1792935506587656625) titled "Scaling Monosemanticity", Anthropic offered an unprecedented detailed look inside a leading large language model.  
- **Millions of interpretable features extracted from Anthropic's Claude 3 Sonnet model**: Using an unsupervised learning technique, [@AnthropicAI](https://twitter.com/AnthropicAI/status/1792935511582986466) extracted interpretable "features" from the activations of Claude 3 Sonnet, corresponding to abstract concepts the model learned.
- **Some extracted features relevant to safety, providing insight into potential model failures**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1792935524220481777) found safety-relevant features corresponding to concerning capabilities or behaviors like unsafe code, bias, dishonesty, etc. Studying these features provides insight into the model's potential failure modes.

**Memes and Humor**

- **Scarlett Johansson's voice cloned without permission by OpenAI draws Little Mermaid comparisons**: [@bindureddy](https://twitter.com/bindureddy/status/1792683787647848880) and [@realSharonZhou](https://twitter.com/realSharonZhou/status/1792688472861573192) reacted to news that OpenAI cloned Scarlett Johansson's voice for their AI assistant without permission, drawing comparisons to The Little Mermaid plot.
- **Heated coffee cup collection sadly unused due to electronic mug**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1792901873696956867) mused if battery density is good enough for a heated stir stick to bring electronic temperature control to any cup, as his wife's Ember mug leaves her other cups unused. 
- **Linux permissions meme reacting to Microsoft Copilot's photographic memory**: [@svpino](https://twitter.com/svpino/status/1792957041612337331) shared a meme about Linux file permissions in response to Microsoft's Copilot having a photographic memory.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**OpenAI Controversies and Legal Issues**

- **Scarlett Johansson considering legal action against OpenAI**: In /r/OpenAI, it was discussed that [**Scarlett Johansson has issued a statement condemning OpenAI for using an AI voice similar to hers in the GPT-4o demo after she declined their request**](https://www.reddit.com/r/OpenAI/comments/1cwucf9/psa_yes_scarlett_johansson_has_a_legitimate_case/). OpenAI claims it belongs to a different actress, but Johansson is exploring legal options. Further discussion in /r/OpenAI suggests that [OpenAI CEO Sam Altman's tweet referencing "Her" before the launch and reaching out to Johansson again may strengthen her case](https://www.reddit.com/r/OpenAI/comments/1cw9gxj/open_ai_respondes_to_sky_sounding_like_scarlett/) that they intentionally copied her likeness.
- **OpenAI removes "Sky" voice option**: In response to the controversy, OpenAI has [removed the "Sky" voice that sounded similar to Scarlett Johansson](https://www.reddit.com/r/OpenAI/comments/1cwenul/sky_assistant_voice_is_paused_for_sounding_like/), claiming the actress was hired before reaching out to Johansson. Debate in /r/OpenAI on whether [celebrities should have ownership over similar sounding voices](https://www.reddit.com/r/OpenAI/comments/1cwwhnt/openai_appears_to_be_expanding_their_legal_team/).

**GPT-4o and Copilot Demos and Capabilities**

- **Microsoft demos GPT-4o powered Copilot in Windows 11**: A [video shared on Twitter](https://x.com/msftcopilot/status/1792626848641274342?s=46) shows Microsoft demonstrating GPT-4o based Copilot features integrated into Windows 11, including [**real-time voice assistance while gaming and life guidance**](https://www.reddit.com/r/OpenAI/comments/1cwman0/is_this_why_openai_didnt_release_their_desktop/). Some in /r/OpenAI speculate this deep OS integration is why OpenAI hasn't released their own desktop app.
- **GPT-4o voice/vision features coming to Plus users**: Images shared in /r/OpenAI from the GPT-4o demo state that the [new voice and vision capabilities will roll out to Plus users in the coming months](https://www.reddit.com/gallery/1cwqhcb), rather than weeks as initially indicated. ([Image source](https://i.redd.it/qh9kczvw1n1d1.png))
- **Impressive OCR capabilities**: A post in /r/singularity shares an [example of GPT-4o's OCR successfully reading and correcting partially obscured text in an image](https://www.reddit.com/r/singularity/comments/1cwil8s/just_had_an_interesting_experience_with_4o_doing/), demonstrating advanced computer vision.
- **Potential increase in hallucinations**: Some users in /r/OpenAI report [GPT-4o seeming more prone to hallucinations compared to the base GPT-4 model](https://www.reddit.com/r/OpenAI/comments/1cwi1dl/does_4o_seem_more_prone_to_hallucinating_than_4/), possibly due to the additional modalities.

**AI Progress and the Path to AGI**

- **GPT-4 shows human-level theory of mind**: A [new Nature paper](https://www.nature.com/articles/s41562-024-01882-z) finds that GPT-4 demonstrates human-level theory of mind, detecting irony and hints better than humans. Its main limitations seem to come from the guardrails on expressing opinions.
- **Concerns about reasoning advancement**: A post in /r/singularity expresses [concern that despite GPT-4's capabilities, reasoning and intelligence haven't significantly improved in the year since its release](https://www.reddit.com/r/singularity/comments/1cwe0yc/is_anyone_else_concerned_thats_its_been_over_a/), slowing the path to AGI.

**Humor and Memes**

- A [meme image jokingly suggests Joaquin Phoenix is considering suing OpenAI](https://www.reddit.com/r/singularity/comments/1cwwhnt/breaking_joaquin_phoenix_now_considering_suing/) for hiring a man with a similar mustache, mocking the Scarlett Johansson controversy. 
- An [image macro meme pokes fun at /r/singularity's reaction to the GPT-4o hype](https://i.imgur.com/63WoZO2.png).
- An example of AI generated absurdist humor is shared, depicting [Abraham Lincoln meeting Hello Kitty in 1864 to discuss national security](https://i.redd.it/e0bvhii75m1d1.jpeg).

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **Optimizing Models to Push Boundaries**:

  - **Transformer Integrations and Model Contributions Generate Buzz**: Engineers are integrating [ImageBind](https://arxiv.org/abs/2305.05665) with the `transformers` library, while another engineer's [PR got merged](https://github.com/huggingface/transformers/pull/29004), fixing an issue with finetuned AI models. Moreover, the **[llama-cpp-agent](https://huggingface.co/spaces/pabloce/llama-cpp-agent)** suggests advancements in computational efficiency by leveraging ZeroGPU.
   - **[LLM Efficiency Gains with Modular](https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering)**: Modular's new nightly release, bolstered by improved SIMD optimization and async programming techniques, promises large performance gains with methods like k-means clustering in Mojo.

   - Members highlighted the importance of tools like [Torch's mul_()](https://x.com/mlpowered/status/1792948212728524917) and the practical uses of [vLLM and memory optimization techniques](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) to enhance model performance on limited VRAM systems.

2. **ScarJo Strikes Back at AI Voice Cloning**:

   - **[Scarlett Johansson's OpenAI lawsuit](https://www.npr.org/2024/05/20/1252495087/openai-pulls-ai-voice-that-was-compared-to-scarlett-johansson-in-the-movie-her)**: Johansson sues OpenAI for voice replication controversy, forcing the company to remove the model and potentially reshaping legal landscapes around AI-generated voice cloning.

   - Discussions highlighted the ethical and legal debates over [voice likeness and consent](https://platformer.news/open-ai-scarlett-johansson-her-voice-sam-altman) amid industry comparisons to unauthorized content removals featuring musicians like Drake.

3. **New AI Models Set Benchmarks Aflame**:

   - **Phi-3 Models and ZeroGPU Excite AI Builders**: Microsoft launched **Phi-3 small (7B)** and **Phi-3 medium (14B)** models with 128k context windows that excel in [MMLU and AGI Eval tasks](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct), revealed on HuggingFace. Complementing this, HuggingFace's new **[ZeroGPU initiative](https://huggingface.co/zero-gpu-explorers) offers $10M in free GPU access**, aiming to boost AI demo creation for independent and academic sectors.

   - **Discovering Documentary Abilities of PaliGemma**: [Merve highlighted](https://x.com/giffmana/status/1791541209883717973?s=46) the document understanding prowess of **PaliGemma** through a series of links to Hugging Face and related tweets. Inquiries about Mozilla's DeepSpeech and various resources from [LangChain](https://python.langchain.com/v0.1/docs/integrations/document_loaders/airtable/) to [3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#editing) reveal the community's broad interest in various AI technologies.

   - **[M3 Max for LLMs](https://x.com/osanseviero/status/1791567896482635801)** received praise for performance, particularly with 96GB of RAM, fueling more significant strides in model capabilities and setting new standards for large language model training efficiency.

4. **Collaborative Efforts Shape AI's Future**:

   - **[Hugging Face's LangChain Integration](https://huggingface.co/blog/langchain)**: New packages aim to facilitate seamless integration of models into LangChain, offering new architectures and optimizing interaction capabilities for community projects.

   - **[Memary Webinar](https://lu.ma/nzh3o83f)** presents an open-source long-term memory solution for autonomous agents, addressing critical needs in knowledge graph generation and memory stream management.

5. **AI-Community Buzz with Ethical and Practical AI Implementations**:

   - **[Anthropic's Responsible Scaling Policy](https://www.anthropic.com/news/reflections-on-our-responsible-scaling-policy)**: The increased computing power suggests significant upcoming innovations and aligns with new responsible scaling policies to manage ethical concerns in AI development.

   - **[Collaborations in AI](https://lu.ma/fp0xr460)** continue to thrive in events like the PizzerIA meetup in Paris and San Francisco, enhancing the Retrieval-Augmented Generation (RAG) techniques and community engagement in AI innovations.


---

{% if medium == 'web' %}



# PART 1: High level Discord summaries




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **PDF Extraction with PyMuPDF and Tesseract**: Engineers shared tools and workflows for PDF text extraction using [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/tutorial.html) and OCR, with mentions of `fitz` and the `sort=True` option, as well as ABBYY and MarkerAPI for handling complex PDFs.

- **Optimizing LLM Training and Fine-Tuning**: Technical discussions highlighted tools like vllm for multiple user services, with references to workflows using pyenv and virtualenv, and dependencies in Axolotl. Insights were shared from Anthropic's research on model interpretability with a nod to [Claude Sonnet's research](https://www.anthropic.com/research/mapping-mind-language-model).

- **Innovative Learning and Collaboration**: Engineers brainstormed over resources like Vik's Marker API and GitHub repositories for fine-tuning models, with a strong focus on multilingual model fine-tuning and shared problem-solving.

- **Model Serving Tips on Modal**: For serving LLM models efficiently, engineers were advised to use `modal serve` over `modal run`, with insights on cost management and minimizing idle container times. Modal credits can be obtained through [this form](https://bit.ly/modal-credits) and $500 in credits plus $30/month on the free tier are available on signing up.

- **Bangalore Meetup Enthusiasm**: There's keen interest for a Bangalore meetup. Techniques for incorporating new languages into models without impairment, performance discussions on Japanese LLMs, and region-specific meetups were all points of fervor.

- **Course Structure and Engagement**: A newly explained course structure includes Fine-Tuning Workshops, Office Hours, and Conference Talks. Technical challenges with Llama3, hyperparameters, and resources for fine-tuning like Stanford's [Pyvene](https://github.com/stanfordnlp/pyvene/issues/46) were exchanged among erudite participants.

- **Hugging Face's Accelerate Touted**: Members were encouraged to check out Accelerate, useful for distributing PyTorch code across configurations, with examples provided for starting with `nlp_example` on [Hugging Face's GitHub](hhttps://github.com/huggingface/accelerate/tree/main/examples). Resources for estimating model memory and FLOPS, like [Model Memory Utility](https://huggingface.co/spaces/hf-accelerate/model-memory-usage), were also highlighted.

- **Axolotl and BitsandBytes Queries**: Engineering queries on bitsandbytes and MLX support on macOS were addressed with a particular reference to [issues on GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436). Offers for fine-tuning comparison between OpenAI and Axolotl sparked interest in OpenAI's 30-minute token-based service.

- **Systematic Prompt Engineering Curiosity**: Interest was piqued in Jason's techniques for systematic prompt engineering, with eager await for his "recipe" during his upcoming workshop session.

- **Gradio's Approachable Interface Development**: Gradio's maintainer invited queries and demo sharing, advocating for its ease of developing user interfaces for AI models and sharing useful guides like the [quickstart tutorial](https://www.gradio.app/guides/quickstart) and how to [build a chatbot swiftly](https://www.gradio.app/guides/creating-a-chatbot-fast).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity and Tako Unite**: Perplexity AI has collaborated with Tako to enhance user experience through **advanced knowledge search and visualization**, now available in the U.S. and in English, with a mobile version expected soon. Details are available [here](https://trytako.com/blog/introducing-tako-and-perplexity-integration).

- **Perplexity Powers Rich Discussions**: Engineers exchanged insights on using **Perplexity AI** with a lively debate on platform loyalty, discussions around model use cases with GPT-4 and Claude 3 Opus, and shared excitement for new features like Tako charts. They also banded together when facing service downtime, suggesting a strong user community.

- **Perplexity API Woes and Wins**: AI engineers identified challenges integrating Perplexity API with Open WebUI, with particular confusion surrounding model compatibility. Solutions involved proxy servers and precise Docker commands, and engineers actively shared progress and advice.

- **Perplexity: A Portal to Knowledge**: Contributions in the **sharing** channel underlined Perplexity AI's ability to address a diverse array of topics, from history and mathematics to script creation and technical computing concepts, echoing the platform's versatility as a knowledge resource.

- **API Integration Tactics and Teething Troubles**: The **pplx-api** channel buzzed with tactical discussions on configuring Docker for optimal **Perplexity API** usage, verifying the absence of a `/models` endpoint, and clarifying current limitations like the lack of image support through the API.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Phi-3 Models and ZeroGPU Excite AI Builders**: Microsoft launched **Phi-3 small (7B)** and **Phi-3 medium (14B)** models with 128k context windows that excel in MMLU and AGI Eval tasks, revealed on HuggingFace. Complementing this, HuggingFace's new **ZeroGPU initiative offers $10M in free GPU access**, aiming to boost AI demo creation for independent and academic sectors.

**Discovering Documentary Abilities of PaliGemma**: Merve highlighted the document understanding prowess of **PaliGemma** through a series of links to Hugging Face and related tweets. Inquiries about Mozilla's DeepSpeech and various resources from LangChain to 3D Gaussian Splatting reveal the community's broad interest in various AI technologies.

**LangChain Memory Trick**: Practical advice was offered to incorporate conversation history into **LLM-based chatbots** using LangChain, addressing a common challenge of bots forgetting prior interactions. Meanwhile, a user critiqued story enhancement abilities of **llama3 8b 4bit**, unveiling a limitation in the model's creative processes.

**Transformer Integrations and Model Contributions Generate Buzz**: Engineers are integrating ImageBind with the `transformers` library, while another engineer's PR got merged, fixing an issue with finetuned AI models. Moreover, the **llama-cpp-agent** suggests advancements in computational efficiency by leveraging ZeroGPU.

**Vision Tech Queries and Solutions Exchange**: In the computer vision domain, requests for papers on advanced patching techniques in Vision Transformers and methods for zero-shot object detection in screenshots were highlighted. The conversations indicate a need for more sophisticated approaches and zero-shot methodologies in object recognition tasks.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **ScarJo Strikes Back at AI Voice Cloning**: Scarlett Johansson has sued OpenAI for unauthorized replication of her voice. As a consequence, OpenAI has already taken down the voice model amidst mounting public concern.

- **Phi-3 Debuts on Hugging Face**: Microsoft has released the **Phi-3-Medium-128K-Instruct model** on [Hugging Face](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct), touting enhanced benchmarks and an extended context of 128k. Engineers in the guild are currently deliberating its merits and the challenges with its large context window.

- **Colab Conundrum with T4 GPUs Resolved**: Imperfect T4 GPU detection by PyTorch on Colab led to notebook chaos until Unsloth’s [update](https://x.com/danielhanchen/status/1792985678030221464) was propagated. The fix addresses PyTorch's incorrect assumption of T4’s bfloat16 support.

- **Discussion Brews Around MoRA**: A discussion kicked off about a new fine-tuning method called **MoRA**, with a link to the [arXiv paper](https://arxiv.org/abs/2405.12130) provided. Guild members are showing early interest in testing its vanilla implementation in their workflows.

- **Dolphin-Mistral's Lean Success**: There's buzz around **dolphin-mistral-2.6** being refined with around 20k samples to match the instructional performance of the original, which used millions. This novel training approach has piqued interest and a promised paper could detail the process later in the year.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Scam Alert for AI Enthusiasts**: Users are advised to avoid scam subscriptions for **Stable Diffusion** services and to use only the official [stability.ai](https://stability.ai) site for legitimate access.

- **Stable Diffusion Runs Offline Too**: **Stable Diffusion's** capability to run locally without an internet connection was confirmed, reducing dependencies on constant online connectivity.

- **Tech Support for Stable Diffusion Setup**: Community support is at hand for those struggling with the setup of **Stable Diffusion** and tools like **ComfyUI**, with users sharing advice on tackling installation issues.

- **EU AI Act Raises Eyebrows**: The newly introduced **EU AI Act** is sparking debate regarding its implications for AI-generated content, including worries about mandatory watermarks and enforcement challenges.

- **Mitigating Hardware Performance Bottlenecks**: Discussions on **Stable Diffusion** performance problems suggest checking system configurations and using diffusers scripts, with a speculation of thermal throttling on new hardware setups.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Real-Time AI**: GPT-4o's ability to process video at **2-4 frames per second** sparked discussion, and integration of **GPT-4o** into Microsoft Copilot is anticipated to bring real-time voice and video capabilities. OpenAI's Sky feature voice resemblance to Scarlett Johansson stirred legal and ethical debates.

- **Model Precision and Characteristics**: GPT-4's 128,000 token context window includes both the prompt and response, while strategies for achieving precise language and specific behavior in responses, akin to the AI in the movie "Her", were hot topics.

- **Prompt Engineering for Conciseness**: Ingenious prompt crafting techniques were shared to keep GPT-4 outputs within specific character limits, with a focus on clear templates and strategic use of token count to ensure concise and relevant responses.

- **Ethics and Legality in AI**: The ability to sell AI-generated art was confirmed, though complexities surrounding copyright issues were highlighted, and community members expressed concerns about GPT-4's evaluation of numerical values.

- **Safety and Updates**: A significant safety update was announced at the AI Seoul Summit with further details available at the [OpenAI Safety Update](https://openai.com/index/openai-safety-update/), reinforcing OpenAI's commitment to responsible AI development.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Run LM Studio as Admin for Log Access**: Running **LM Studio** with admin permissions solves blank server log issues, providing users access to needed log files for troubleshooting.

**AVX2 a Must for LM Studio**: Understanding that **AVX2 instructions** are necessary to run LM Studio, users can check CPU compatibility for AVX2 using tools like [HWInfo](https://www.hwinfo.com/download/). Older CPUs lacking AVX2 support will face compatibility issues with the software.

**Efficient Image Gen via Civit.ai**: For improved image quality, members recommended using local models like **Automatic1111** and **ComfyUI** with supporting resources from [Civit.ai](https://civitai.com/), cautioning the need for sufficient VRAM and RAM in system specs.

**Getting Specific with Models**: To ensure response completeness in **LM Studio**, setting **max_tokens** to **-1** resolves issues of prematurely cut-off responses encountered when the value is set to null. The community also discussed using model-specific prompts, as shown with **MPT-7b-WizardLM**; referencing [Hugging Face](https://huggingface.co/DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF) for required quantization levels and templates.

**ROCm and Linux Bonding Over AMD GPUs**: Linux aficionados with AMD GPUs have been invited to test an early version of **LM Studio** integrated with ROCm, as listed on [AMD's supported GPU list](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). Success reports have come from users running unsupported GPUs, with users sharing their diverse Linux distribution experiences and findings involving **infinity fabric (fclk)** speed sync affecting system performance.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Zooming into Mojo Community Meetings**: The [Mojo community meeting](https://modular.zoom.us/j/89417554201?pwd=Vj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1) was held, and though some faced notification issues, the recording is now available on [YouTube](https://www.youtube.com/playlist?list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D). There was initial confusion regarding the need for a commercial Zoom account, which was clarified as unnecessary.

**Boosted Mojo Performance with k-means Clustering**: A [blog post](https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering) taught readers to use the k-means clustering algorithm in Mojo, promising considerable performance improvements compared to Python.

**Challenging Code Conundrums and Compiler Chronicles**: Discussions included handling null terminators in strings, exploring asynchronous programming, and utilizing the Lightbug HTTP framework within Mojo. Solutions and workarounds were devised within the community, with some technical queries leading to [GitHub issue discussions](https://github.com/saviorand/lightbug_http/issues/41).

**Nightly Updates Navigate Compiler Complexities**: The [latest nightly Mojo compiler release](https://github.com/modularml/mojo/compare/7e8cd37ff8fe2ddbe69a3cca787e59abf6357d76...69e92d0040af838de8f3f0fdba1cea92f1904986) was detailed, with conversations around the `pop` method in dictionaries, Unicode support in strings, and other [GitHub issue](https://github.com/modularml/mojo/issues/2696) and [PR delibarations](https://github.com/modularml/mojo/pull/2739).

**Peering into SIMD Optimization**: Members engaged in discussions around optimizing SIMD gather and scatter operations in Mojo, conquering challenges such as ARM SVE and memory alignment, with suggestions on minimizing gather/scatter operations and tips for sorting scattered memory for iterative decoders.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Kubernetes: Necessity or Overkill?**: Some members argue **managed Kubernetes services like EKS** may efficiently replace on-prem ML servers, despite others noting Kubernetes isn't essential for ML infrastructure; decision should be tailored to project requirements.

**Triton Gets a Makeover**: Updates to the **Triton** library include a [pull request](https://github.com/triton-lang/triton/pull/3959) improving tutorial readability and new insights into how **GPU kernel specifics** affect maximum block size.

**Wrangling with SASS and Complex Operations**: Engineers discuss academic resources on **SASS**, and deliberate on the merits of "cucomplex" versus "cuda::std::complex" for atomic operations on advanced **NVIDIA architectures**.

**Torch Tricks for Efficient Memory Use**: Users discover that **Torch's native `*` operator** doubles memory usage whereas `mul_()` doesn’t, and `torch.empty_like` outperforms `torch.empty` for CUDA device allocations.

**Activation Quantization Takes Center Stage at CUDA**: Focus shifts to **activation quantization** using features like **2:4 sparsity** and **fp6/fp4** on newer GPUs, with an eye to integrating these into **torch.compile** for enhanced graph-level optimizations.

**Torchao 0.2 Ushers Custom Extensions**: The **torchao 0.2 release** on [GitHub](https://github.com/pytorch/ao/releases/tag/v0.2.0) introduces custom CUDA and CPU extensions, and the integration of **NF4 tensors with FSDP** for improved model training.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SF Seeks Safety Specialists**: A newly established San Francisco office of the **UK Artificial Intelligence Safety Institute (AISI)** is offering competitive salaries to attract talent. They're engaging in collaborations, including a [UK-Canada AI safety partnership](https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership).

- **A Call to Action Against SB 1047**: Stakeholders in the AI community are mobilizing against **California's SB 1047**, arguing the bill could threaten open-source AI development with its stringent regulatory measures, as detailed in [this analysis](https://context.fund/policy/sb_1047_analysis.html).

- **FLOP-Sweating the Details**: Intricate discussions emerged on the computation of FLOPs for attention mechanisms, referencing the [EleutherAI cookbook](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py) for FLOP calculations, highlighting the necessity to include QKVO projections.

- **Multi-Modal Models Making Headlines**: Discussions centered on improving AI models through multi-modal training, including the benefits observed in **CLIP** when incorporating audio for zero-shot classification. Performance enhancements without emergent capabilities were noted in models like [ImageBind](https://arxiv.org/abs/2305.05665).

- **Efficiency in MoE Spotlighted**: New research introduces **[MegaBlocks](https://arxiv.org/abs/2211.15841)**, a resource-efficient system for MoE training that forgoes token dropping and utilizes block-sparse operations, offering considerable enhancements in training efficiency.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Temporal Conquers Workflow Management**: Following discussions on workflow orchestration, a guild member has confirmed the selection of **Temporal.io** over Apache Airflow due to its robust features.

- **Navigating the AI Labyrinth**: Members highlight various challenges such as the **ineffective LLM leaderboard** and **Chatbot Arena's skewed ratings**. Microsoft's **Copilot+** presentation stirred chat, and the unveiling of the **Yi-1.5** model garnered attention for addressing different context size needs.

- **Research Initiatives Thrive**: The **Manifold Research Group's** continued progress in the NEKO Project reflects the community's drive towards developing comprehensive models, further underlined by the **Phi-3 Vision's** release aligning vision and text with fine-tuning and optimization techniques.

- **Picturing AI Boundaries**: Creative exploration via ASCII and generated simulation images spurred discussions on the functional and symbolical capacities of AI, particularly the applications of **WorldSim**.

- **Knowledge in Motion**: A shared timelapse of an **Obsidian knowledge graph** and the call for support with public evaluation methods for **rerankers** reflect the dynamic and collaborative nature of the engineering community.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Sky Voice Grounded**: OpenAI has temporarily halted the use of the **Sky voice** in ChatGPT due to user feedback; the company is working to address these concerns. The decision strikes a chord with ongoing discussions about AI-generated voices and the ethical considerations inherent in such technologies. [Read the tweet](https://x.com/OpenAI/status/1792443575839678909)

**CogVLM2: Use with Caution**: The **CogVLM2 model**, which was noted for its 8K content length support, comes with a controversial license that restricts usage against China's national interest, stirring discussions about real open-source principles. The license also stipulates that any disputes are subject to Chinese jurisdiction. [Review the License](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE)

**AI Copilot: From Code to Life's Companion?**: Mustafa Suleyman's teaser of the upcoming **Copilot AI** that can interact with the physical world in real-time sparked a variety of reactions, reflecting the community's mixed sentiments towards the increasingly blurred lines between AI assistance and privacy. [See the tweet](https://fxtwitter.com/mustafasuleyman/status/1792623877744623806)

**ScarJo's Voice Doppelgänger Dilemma**: The use of a voice resembling that of actress **Scarlett Johansson** by OpenAI's voice assistant sparked a debate on ethical boundaries and legal issues around AI's mimicking of human voices, particularly celebrities.

**Sakuga-42M Dataset Disappears Amidst Bot Backlash**: High demand and automated downloading led to the removal of the **Sakuga-42M dataset** from hosting platforms, fueling a conversation on the challenges of maintaining accessible datasets in the face of aggressive web scraping. [Hacker News Discussion](https://news.ycombinator.com/item?id=40389711)



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI's Voice Controversy Halts "Sky"**: OpenAI has halted the use of "Sky," a voice AI resembling Scarlett Johansson, due to legal pressure and negative public perception, highlighting the ethical concerns of voice mimicry and consent. The incident is reminiscent of controversies involving impersonations of public figures like musicians, sparking discussions on accountability and the need for clear ethical guidelines in the AI industry.

- **Anthropic's Quantum Leap in Compute**: [Anthropic has ramped up its compute resources to four times that of its previous model Opus](https://www.anthropic.com/news/reflections-on-our-responsible-scaling-policy), stirring the community's interest regarding what the company has in the pipeline. Details are scarce, but the magnitude of compute increase points to significant developments.

- **AI Arena Faces Hard Prompts Challenge**: The introduction of the "Hard Prompts" category by [LMsysorg](https://fxtwitter.com/lmsysorg/status/1792625968865026427) has turned up the heat in AI model evaluations, proving particularly strenuous for models like Llama-3-8B which showed a notable performance dip against GPT-4-0314. The rigorous evaluation raises questions about the effectiveness of current judge models, such as Llama-3-70B-Instruct.

- **OpenAI's Superalignment Commitment Breach**: OpenAI faces scrutiny over allegations from a [Fortune article](https://fortune.com/2024/05/21/openai-superalignment-20-compute-commitment-never-fulfilled-sutskever-leike-altman-brockman-murati/) that it reneged on a promise to allocate 20% of its computing power to their Superalignment team, leading to a team shake-up. This revelation sparks dialogues on the prioritization between product development and AI safety, with some viewing the company's move as a predictable deviation from its commitments.

- **The Domain Deal and AI Dataset Dilemma**: Nathan Lambert's purchase of the domain rlhfbook.com for a dealing price of $7/year, and joking banter about the potential legal risks associated with using the AI Books4 dataset to train LLMs, spotlight both the quirky side of AI development and the serious legal considerations of data use. The reference of Microsoft Surface AI experiencing latency raises questions about the balance between local processing and cloud-dependent safety verifications, suggesting an area for potential optimization.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **"Memory Tuning" Raises the Bar for LLMs**: Sharon Zhou of Lamini introduced a new technique called "Memory Tuning," which claims to significantly reduce hallucinations (<5%) in Language Models (LLMs), surpassing the performance of LoRA and traditional fine-tuning methods. Details on early access and further explanations are pending ([Sharon Zhou's Tweet](https://x.com/realsharonzhou/status/1792578913572429878)).

- **Scarlett Johansson's AI Voice Controversy**: OpenAI temporarily ceased using an AI-generated voice similar to Scarlett Johansson's after legal action was suggested by her lawyers, stirring debates about likeness and endorsements ([NPR Article](https://www.npr.org/2024/05/20/1252495087/openai-pulls-ai-voice-that-was-compared-to-scarlett-johansson-in-the-movie-her)).

- **Scaling Up: Scale AI's Billion-Dollar Injection**: Scale AI secured $1 billion in funding at a $13.8 billion valuation, planning to use the investment to enhance frontier data and target profitability by the end of 2024, with Accel leading the round ([Fortune Article](https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/)).

- **Microsoft Unveils Phi 3 Model Lineup**: Microsoft released Phi 3 models at MS Build with benchmark performances competitive to Llama 3 70B and GPT 3.5, supporting context lengths up to 128K and released under the MIT license ([Tweet about Phi 3 Models](https://x.com/reach_vb/status/1792949163249791383)).

- **Introducing Pi: The Emotionally Intelligent LLM**: Inflection AI announced a shift towards creating more emotive and cognitive AI models, with more than 1 million daily users interacting with their empathetic LLM "Pi," showcasing AI's transformative potential ([Inflection AI's Announcement](https://inflection.ai/redefining-the-future-of-ai)).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Rate Limiting Ruffles Feathers**: Azure's GPT-32k model has been hitting token rate limits, with users citing specific issues when making requests to the ChatCompletions_Create Operation under Azure OpenAI API version 2023-07-01-preview.

- **Phi-3 Models Gain Traction**: The community has been exploring Phi-3 models for superior reasoning with data, examining models like [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct), which uses supervised fine-tuning, and [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct), that features direct preference optimization.

- **New Twist on LLM Interaction Coming Up**: A novel approach for interaction with LLMs has been circulating, termed "Action Commands", and a discussion thread sharing experiences and seeking feedback is available [here](https://x.com/leonjcoe/status/1792946945528320382).

- **Conciseness vs. Verbosity Debate Continues**: Strategies for managing verbosity in models like Wizard8x22 are being evaluated, with some members advocating for a decrease in repetition penalty to ensure more concise outputs.

- **OpenRouter Shows Open Wallet for Non-Profits**: OpenRouter discussed its 20% margin pricing policy in response to a user's Error 400 billing issue and their request for non-profit discounts.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Grok Enthusiasts Gear Up**: AI engineers are showing enthusiasm for training **Grok** using the [PyTorch version](https://huggingface.co/hpcai-tech/grok-1), discussing potential enhancements with **torchtune integration**, and comparing compute platforms, including **Mi300x** versus **H100s**.
- **Sharp Turn in Mistral Finetuning**: Members are troubleshooting **Mistral 7B** finetuning issues, with proposals ranging from full finetuning to **Retrieval-Augmented Generation (RAG)** techniques to address content retention, as noted in a shared [configuration guide](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/lora.yml).
- **OOM Woes and Wisdom**: Out-of-Memory (OOM) errors are a central topic, with a multitude of solutions including **gradient accumulation steps**, **mixed precision training**, **model parallelism**, batch size adjustments, and **DeepSpeed ZeRO optimization** being put forward to tackle VRAM limitations, with more details on [Phorm.ai](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=79ea3546-1ab6-4fe1-8984-1d8eb8183eda).
- **M3 Max Takes the Stage**: The **M3 Max** chip earns praise for its LLM performance capabilities, with recommendations to equip it with 96GB of RAM to get the most out of large language models.
- **Code Debacles and Python Queries**: Conversations include troubleshooting Syntax Errors in the **Transformers** library involving `CohereTokenizer` with the exploration of faster alternatives, as discussed in a [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files), and the search for a Python library to accelerate the speech-to-text to LLM to speech synthesis chain.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Memary Makes Memories**: An upcoming webinar focused on **memary**, an open-source long-term memory system for autonomous agents, promises deep dives on its use of LLMs and neo4j for knowledge graph generation. Scheduled for Thursday at 9am PT, engineers can join by registering [here](https://lu.ma/nzh3o83f).

**Knack for Stacking RAG Techniques**: In the realm of retrieval-augmented generation (RAG), @hexapode will share advanced strategies at PizzerIA in Paris, while Tryolabs and ActiveLoop will present at the first in-person meetup in San Francisco next Tuesday—sign up [here](https://t.co/qIGOmCW62G).

**GPT-4o Integrates with LlamaParse**: LlamaIndex.TS documentation is enhanced, and **GPT-4o** now seamlessly works with LlamaParse for analyzing complex documents. Further, you can safely execute LLM-generated code using Azure Container Apps as per their [latest offering](https://t.co/2cnsBH411k).

**Resolving Twin Data Quandaries**: Engineers discussed methods to compute unique hashes for documents to avoid duplicates in Pinecone and examined workarounds for dealing with empty nodes in VectorStoreIndex.

**Streamlining Systems and Storage**: Insights were shared on how to modify an OpenAI agent's system prompt using `chat_agent.agent_worker.prefix_messages`, and the merits of utilizing Airtable over Excel/Sqlite due to its Langchain integration—info available [here](https://python.langchain.com/v0.1/docs/integrations/document_loaders/airtable/).



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Emotive AI On The Horizon**: Inflection AI is reportedly planning to integrate **emotional AI into business bots**, raising prospects for more empathetic AI companions, detailed in a [VentureBeat article](https://venturebeat.com/ai/exclusive-inflection-ai-reveals-new-team-and-plan-to-embed-emotional-ai-in-business-bots). The conversation also touched on AI characters, with a *Just Monika* reference from *Doki Doki Literature Club* clarified through a [GIF from Tenor](https://tenor.com/view/ddlc-doki-doki-literature-club-just-monika-monika-gif-20717242).
  
- **Cracking AI Town's Memory Woes**: Community feedback indicates that AI characters in AI Town often **fail to remember past interactions**, leading to repeated dialogues. It was advised to tweak `convex/constants.ts` to adjust the `NUM_MEMORIES_TO_SEARCH` and ease the retrieval of past exchanges.

- **Overcome SQL Schema Confusion**: Engineers shared **SQL queries** and tools for exporting AI Town conversation data, including links to GitHub repositories like [townplayer](https://github.com/cocktailpeanut/townplayer/blob/main/index.html) and an explanatory [Twitter thread](https://x.com/cocktailpeanut/status/1786421948638965870), facilitating data manipulation and understanding.

- **Introductions to 3D AI**: A tease of an ongoing project involving **3D character chatbots** was mentioned, with the recommendation to check out further details in another channel within the community. 

- **Animated Explanation Lacks Impact**: A playful discussion around the cultural impact of *AI waifus* was noted, underlining both the humor and significance of AI character development in user interfaces.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LLMs Tangle with Text Types**: LLMs, including structured and unstructured data handlers like [Hermes 2 Pro - Mistral 7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) and [OpenAI's chatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md), don't have innate preferences for text types but excel with finetuning.

**LangChain's Community Contributions**: The `langchain-core` package is streamlined for base abstractions, while `langchain-openai` and `langchain-community` house more niche integrations, detailed in the [architectural overview](https://python.langchain.com/v0.2/docs/concepts/#architecture).

**Sequential Chains in Action**: A [YouTube tutorial](https://youtu.be/2xxziIWmaSA?si=3wkNt_huJKu3xK3t&t=1694) has been pointed out for setting up sequential chains, where one chain's output becomes the next one's input.

**Commissions from Chat Customizations**: An affiliate program entices with a 25% commission for the **ChatGPT Chrome Extension - Easy Folders**, detailed [here](https://easyfolders.promotekit.com/), despite some users reporting issues with the extension's performance.

**Agent Upgrades and PDF Insights**: Transitioning from LangChain to the newer **LangGraph** platform has been expounded in a [Medium article](https://medium.com/ai-advances/upgrading-your-agents-a-smooth-transition-from-legacy-langchain-to-langgraph-c552cb60fcb3), alongside a guide to querying PDFs with **Upstage AI solar models**, available [here](https://medium.com/@sonam.gupta1105/creating-a-pdf-query-assistant-with-upstage-ai-solar-and-langchain-integration-6631280093b5).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**AI-Empowered DevOps on the Rise**: A full-stack junior DevOps engineer is creating a **lite O1 AI project** with the prospect of providing **discreet auditory assistance** for various DevOps tasks, seeking community insights for development and practical applications.

**OpenInterpreter's Symbiosis with Daily Tech**: Engineers are exploring how **Open Interpreter** can streamline their workflow, from code referencing across devices to summarizing technical documents, underlining the practical impact of AI in everyday technical tasks.

**Combining Voice Tech with OpenInterpreter**: A community member is integrating **Text-to-Speech** with Open Interpreter and has been directed to the relevant [GitHub repository](https://github.com/OpenInterpreter/01) to further their project.

**Connection Queries and Missing Manuals**: One member sought help with linking their laptop to a light app despite the absence of instructions in the provided guides, while another requested advice on assembling 3D printed parts for their version of **Open Interpreter lite 01**.

**Humorous Nod to Misssed Opportunities**: The user *ashthescholar.* lightheartedly noted a missed opportunity in naming conventions, showcasing the playful side of technical communities.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Codegen-350M-mono Tackles Compatibility**: A solution to compatibility issues with using **Codegen-350M-mono** in Transformers.js is provided through an [ONNX version](https://huggingface.co/Xenova/codegen-350M-mono) shared by members, indicating successful cross-platform implementation.
- **Translating with CommandR+**: For Korean-English translation tasks, **CommandR+** has been highlighted as an effective tool, with the [Chat API documentation](https://docs.cohere.com/docs/chat-api) serving as a resource with sample code and usage instructions.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Johansson and OpenAI's Voice Controversy**: OpenAI has paused the use of the Sky voice in GPT-4o, substituting it with Juniper, amid copyright claims and an issued [statement](https://x.com/BobbyAllyn/status/1792679435701014908) from Scarlett Johansson.
- **GPT-4o's Unified Modal Approach**: GPT-4o has augmented its capabilities by integrating a unified model for text, vision, and audio which enhances emotional understanding in interactions but could complicate the model's performance and potential use cases.
- **Lem's Take on System Reliability**: Engineers shared a perspective from Stanisław Lem's work, advocating for the construction of resilient rather than perfectly reliable systems, acknowledging the inevitability of system failures.
- **Voice Cloning's Moral Maze**: Engineers discussed the nuanced ethical and legal challenges posed by voice cloning technologies, cautioning against sole reliance on legislation for protection of identity.
- **All Eyes on Qualcomm's New Kit**: Qualcomm's launch of the Snapdragon Dev Kit for Windows was met with excitement, boasting specs such as a 4.6 TFLOP GPU, 32GB RAM, and 512GB storage; available for $899.99, drawing comparisons to Apple’s Mac Mini. [Read more](https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-elite) about the dev kit.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **SFT vs Preference Optimization Debate**: A community member questioned the necessity of **Supervised Fine-Tuning (SFT)** when **Preference Optimization** seems to achieve a similar outcome by adjusting the probability distribution for both desired and undesired outputs.
  
- **Phi3 Vision Gains Recognition**: **Phi3 Vision**, a 4.2 billion parameter model, received praise for its impressive low-latency live inference capabilities on image streams, with potential applications in robotics highlighted in [Jan P. Harries's post](https://x.com/jphme/status/1792950682695479734).

- **Model Matchup: Phi3 Vision vs Moondream2**: The community compared **Phi3 Vision** and [Moondream2](https://huggingface.co/spaces/vikhyatk/moondream2) on image inference tasks, noting Moondream2's reduced hallucinations but issues with some datasets.

- **New Models from Microsoft**: Microsoft introduced new AIs with **7 billion and 14 billion parameters**, with mentions of these releases only providing the instruct versions, sparking interest and discussion among community members.

- **Further Discussion Required**: The insights provided prompted further discussion, likely leading the community to deep-dive into the efficacy and applications of these models.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **SQLite-VeC In The Spotlight**: Alex introduced [`sqlite-vec`](https://github.com/asg017/sqlite-vec), a new **SQLite extension for vector search**, describing its use for features like RAG and semantic search; the extension is compatible with **cosmopolitan** and is currently in beta.
- **Diving into 'sqlite-vec'**: A [detailed blog post](https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html) by Alex unveils the aspirations for `sqlite-vec` to outshine `sqlite-vss` with better performance and easier embedding in applications; binaries and packages will be available for various programming environments.
- **Call to Collaborate and Experiment**: Acknowledging that `sqlite-vec` is in beta, Alex is offering his support to help anyone interested in integrating or troubleshooting the extension within their projects.
- **Community Buzz for Llamafile Integration**: The integration possibilities of `sqlite-vec` with **Llamafile** have sparked excitement among guild members, highlighting the extension's potential to advance current project capabilities.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**GPT-4o Outshines Its Predecessors**: A Discord guild member detailed a notable performance leap in **GPT-4o** over GPT-4 and GPT-4-Turbo in the domain of complex legal reasoning, emphasizing the significance of the advancement with a [LinkedIn post](https://www.linkedin.com/posts/evan-harris-387375b2_the-release-of-gpt-4o-from-openai-has-been-activity-7196856963454959617-w1i1).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Manifold Research Group Calls for Collaboration:** The Manifold Research Group, an open-source R&D lab focusing on *generalist models* and AI agents, is seeking collaborators and has shared links to their [research log](https://www.manifoldrg.com/research-log-038/), [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com), and [GitHub](https://github.com/ManifoldRG?ref=manifoldrg.com).
- **NEKO Project Charts Course for Open-Source AI:** The NEKO Project is ambitiously building a large-scale, open-source generalist model that incorporates a diverse array of modalities, including tasks in control and robotics, details of which are outlined in their [project document](https://docs.google.com/document/d/e/2PACX-1vQELDXCIT9tn7Uq5vxQG4_3HsrkQcuBRqvXm-MkxW06Zkh-LP3G9z7TP7a-2MNWyA/pub?ref=manifoldrg.com).



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1242043650851078205)** (225 messages🔥🔥): 

- **Mastering PDF Extraction with Python and OCR**: Members shared tools and code snippets for PDF text extraction using PyMuPDF and tesseract. One highlighted the efficiency of `fitz` with the `sort=True` option, while others discussed OCR solutions like ABBYY and MarkerAPI for handling complex and low-quality PDFs ([PyMuPDF tutorial](https://pymupdf.readthedocs.io/en/latest/tutorial.html)).
  
- **Exploring and Optimizing LLM Training and Fine-Tuning**: Detailed discussions on optimizing LLM training setups, with references to tools like vllm for serving multiple users simultaneously. Users also shared fine-tuning workflows using pyenv, virtualenv, and addressed dependency issues in Axolotl ([StarCoder2-instruct](https://github.com/bigcode-project/starcoder2-self-align)).

- **Handling Large Language Models and Memory Optimization**: Participants explored methods for handling large language models, particularly on GPUs, and shared insights from new research. Discussions included memory tuning, using vLLM for efficient model serving, and recent findings from Anthropic on model interpretability ([Claude Sonnet research](https://www.anthropic.com/research/mapping-mind-language-model)).

- **Collaborative Learning and Resource Sharing**: Attendees connected over shared resources and tools, such as the Vik's Marker API for PDF processing and various GitHub repos for fine-tuning models. Many also shared their experience and sought collaboration on multilingual and domain-specific model fine-tuning ([Marker API](https://github.com/satish860/PDF-Extraction-API)).

- **Workshop Logistics and Participation**: Queries about session recordings, managing time zones, and accessing course materials were discussed, with confirmatory responses that all sessions will be recorded. Participants also reflected on the credit distributions from sponsors and the organizational structure of the course’s Discord meetings ([Modal examples](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://huggingface.co/spaces/hf-accelerate/model-memory-usage">Model Memory Utility - a Hugging Face Space by hf-accelerate</a>: no description found</li><li><a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#post-installation-actions">CUDA Installation Guide for Linux</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.10131">RAFT: Adapting Language Model to Domain Specific RAG</a>: Pretraining Large Language Models (LLMs) on large corpora of textual data is now a standard paradigm. When using these LLMs for many downstream applications, it is common to additionally bake in new k...</li><li><a href="https://github.com/satish860/PDF-Extraction-API">GitHub - satish860/PDF-Extraction-API: A Marker Library based API for doing the Marker Response.</a>: A Marker Library based API for doing the Marker Response. - satish860/PDF-Extraction-API</li><li><a href="https://github.com/poloclub/unitable">GitHub - poloclub/unitable: UniTable: Towards a Unified Table Foundation Model</a>: UniTable: Towards a Unified Table Foundation Model - poloclub/unitable</li><li><a href="https://github.com/VikParuchuri">VikParuchuri - Overview</a>: VikParuchuri has 88 repositories available. Follow their code on GitHub.</li><li><a href="https://pymupdf.readthedocs.io/en/latest/tutorial.html">Tutorial - PyMuPDF 1.24.4 documentation</a>: no description found</li><li><a href="https://x.com/jxnlco/status/1792549015273513102">Tweet from jason liu (@jxnlco)</a>: If you’re a company building RAG and want to level up your Eng team please fill out this form.   https://q7gjsgfstrp.typeform.com/to/SL656ADC  We will invite other operators to share their stories, gi...</li><li><a href="https://x.com/mlpowered/status/1792948212728524917">Tweet from Emmanuel Ameisen (@mlpowered)</a>: Today, we announced that we’ve gotten dictionary learning working on Sonnet, extracting millions of features from one of the best models in the world.  This is the first time this has been successfull...</li><li><a href="https://github.com/pyenv/pyenv?tab=readme-ov-file#automat">GitHub - pyenv/pyenv: Simple Python version management</a>: Simple Python version management. Contribute to pyenv/pyenv development by creating an account on GitHub.</li><li><a href="https://github.com/VikParuchuri/surya">GitHub - VikParuchuri/surya: OCR, layout analysis, reading order, line detection in 90+ languages</a>: OCR, layout analysis, reading order, line detection in 90+ languages - VikParuchuri/surya</li><li><a href="https://x.com/Kyrannio/status/1792440824355332313">Tweet from Kiri (@Kyrannio)</a>: I was curious, so I found the GPT-4o iOS system prompt:  “You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.  You are chatting with the user via the ChatGPT iO...</li><li><a href="https://x.com/llama_index/status/1791258285993230786">Tweet from LlamaIndex 🦙 (@llama_index)</a>: Structured Image Extraction with GPT-4o 🖼️  GPT-4o is state-of-the-art in integrating image/text understanding, and we’ve created a full cookbook showing you how to use GPT-4o to extract out structur...</li><li><a href="https://x.com/VikParuchuri/status/1788966758742982696">Tweet from Vik Paruchuri (@VikParuchuri)</a>: Marker v2 is out!  The main new features:  - Extracts images/figures - Better table parsing - Pip package install - Can be used commercially - Improved OCR with more languages - Better ordering for co...</li><li><a href="https://x.com/rohanpaul_ai/status/1792640477641970029?s=">Tweet from Rohan Paul (@rohanpaul_ai)</a>: DPO(Direct Preference Optimization) can NOT be as good as PPO (Proximal Policy Optimization) - From latest Google research 🤔  It investigates why online reinforcement learning algorithms (like PPO) f...</li><li><a href="https://x.com/rohanpaul_ai/status/1792640477641970029?s=46&t=mgKHGVn_Owt0fh3SjofSeg">Tweet from Rohan Paul (@rohanpaul_ai)</a>: DPO(Direct Preference Optimization) can NOT be as good as PPO (Proximal Policy Optimization) - From latest Google research 🤔  It investigates why online reinforcement learning algorithms (like PPO) f...</li><li><a href="https://x.com/realSharonZhou/status/1792576516444065967">Tweet from Sharon Zhou (@realSharonZhou)</a>: Hallucinations are one of the biggest blockers to production LLMs & agents.  No hallucinations (&lt;5%) have been achieved internally — and for customers.   We’ve been able to tune LLMs to recall spec...</li><li><a href="https://github.com/shisa-ai/shisa-v2/wiki/Ablations">Ablations</a>: Contribute to shisa-ai/shisa-v2 development by creating an account on GitHub.</li><li><a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/text_generation_inference.py">modal-examples/06_gpu_and_ml/llm-serving/text_generation_inference.py at main · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving">modal-examples/06_gpu_and_ml/llm-serving at main · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://www.anthropic.com/research/mapping-mind-language-model">Mapping the Mind of a Large Language Model</a>: We have identified how millions of concepts are represented inside Claude Sonnet, one of our deployed large language models. This is the first ever detailed look inside a modern, production-grade larg...</li><li><a href="https://github.com/bigcode-project/starcoder2-self-align/tree/main?tab=readme-ov-file#data-generation-pipeline">GitHub - bigcode-project/starcoder2-self-align: StarCoder2-Instruct: Fully Transparent and Permissive Self-Alignment for Code Generation</a>: StarCoder2-Instruct: Fully Transparent and Permissive Self-Alignment for Code Generation - bigcode-project/starcoder2-self-align</li><li><a href="https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts">Dependency Resolution - pip documentation v24.1.dev1</a>: no description found</li><li><a href="https://github.com/explosion/prodigy-segment">GitHub - explosion/prodigy-segment: Select pixels in Prodigy via Facebook&#39;s Segment-Anything model.</a>: Select pixels in Prodigy via Facebook&#39;s Segment-Anything model. - explosion/prodigy-segment</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/453">pip install flash-attn always happens ModuleNotFoundError: No module named &#39;packaging&#39;,but actually i have pip install packaging · Issue #453 · Dao-AILab/flash-attention</a>: Collecting flash-attn Using cached flash_attn-2.0.7.tar.gz (2.2 MB) Installing build dependencies ... done Getting requirements to build wheel ... error error: subprocess-exited-with-error × Gettin...</li><li><a href="https://github.com/pyenv/pyenv?tab=readme-ov-file#automatic-installer">GitHub - pyenv/pyenv: Simple Python version management</a>: Simple Python version management. Contribute to pyenv/pyenv development by creating an account on GitHub.</li><li><a href="https://github.com/pyenv/pyenv-virtualenv">GitHub - pyenv/pyenv-virtualenv: a pyenv plugin to manage virtualenv (a.k.a. python-virtualenv)</a>: a pyenv plugin to manage virtualenv (a.k.a. python-virtualenv) - pyenv/pyenv-virtualenv</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-3447/#llm-finetuning-hamel-dan-discord">[AINews] Skyfall</a>: Not thinking about superalignment Google Scarlett Johansson is all you need. AI News for 5/17/2024-5/20/2024. We checked 7 subreddits, 384 Twitters and 29...</li><li><a href="https://github.com/xl0">xl0 - Overview</a>: Full-time learner. (Linux, Biology, Electronics) -&gt; AI :heart: Writing some lovely software. :two_hearts:
 Open to exciting opportunities! - xl0</li><li><a href="https://chinese-reader.vercel.app">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1242011294861627402)** (141 messages🔥🔥): 

- **Creative Writing AI Sparks Interest**: Members discuss creating AI for assisting in creative writing, focusing on **prompt engineering** to generate ideas and overcome writer's block. Fine-tuning is suggested to align the model with specific genres or writing styles.
- **BERT and Sentence Transformers in Action**: Members introduce the use of **BERT-type models** and **sentence-transformers** like [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for tasks like clustering and semantic search. Sample code shows practical usage of the model for encoding sentences.
- **Legal Document Summarization Debated**: Discussion on using **LLMs** for summarizing legal documents and providing client support. The combination of fine-tuning, RAG, and prompt engineering is explored for tasks like legal research and strategy development.
- **RAG vs Prompting for Customer Support**: A member reconsiders using fine-tuning versus prompt engineering for an LLM designed to help customers create feature tickets. Initial thoughts lean towards fine-tuning for tone and procedures, but later prompting is preferred due to practical considerations.
- **Mental Health and Medical AI Use Cases Emerge**: Multiple members propose creating AI systems for medical coding, summarizing patient records, and offering mental health advice, utilizing **fine-tuning and RAG**. Examples include summarizing ICD-10 codes and providing targeted mental health insights.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">sentence-transformers/all-MiniLM-L6-v2 · Hugging Face</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/">LlamaParse - LlamaIndex</a>: no description found</li><li><a href="https://unstructured.io/">Unstructured | The Unstructured Data ETL for Your LLM</a>: Unstructured helps you get your data ready for AI by transforming it into a format that large language models can understand. Easily connect your data to LLMs.</li><li><a href="https://www.youtube.com/watch?v=sTQaJyrI-zg&list=PLVVTN-yNn8rvEwlY8ClxDUWeVPVfdifYj&index=8&ab_channel=StanfordOnline">Stanford CS25: V2 I Common Sense Reasoning</a>: February 14, 2023Common Sense ReasoningYejin ChoiIn this speaker series, we examine the details of how transformers work, and dive deep into the different ki...</li><li><a href="https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax-Pattern.html">pattern - Amazon CloudWatch Logs</a>: no description found</li><li><a href="https://www.onetonline.org/find/all">See All Occupations</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1242056437182632046)** (49 messages🔥): 

- **Bangalore Meetup Gains Traction**: Multiple members expressed interest in organizing a **meetup for Bangalore**. The idea received significant enthusiasm with users chiming in from Bangalore.
  
- **Inquiry about Non-English Language Model Fine-Tuning**: An interesting exchange occurred on techniques for adding new languages to models without degrading performance. Suggestions included using a **90/10% data mix** to minimize catastrophic forgetting and possibly employing techniques like **layer freezing**.

- **Japanese LLM Performance Discussion**: A member shared extensive updates on Japanese language model development, mentioning various models and benchmarks. Links were provided to their [benchmark framework](https://github.com/shisa-ai/shaberi) and a [Hugging Face model](https://huggingface.co/shisa-ai/shisa-v1-llama3-70b) that matches GPT-3.5-turbo in Japan.

- **Link to Detailed Review on Training Datasets**: A notable mention of the [Shisa project](https://huggingface.co/augmxnt/shisa-7b-v1) and a review on [public Japanese training sets](https://github.com/AUGMXNT/shisa/wiki/A-Review-of-Public-Japanese-Training-Sets#analysis) provided insights into the challenges and methodologies in Japanese LLM development.

- **Multi-City Meetup Initiatives**: Invitations were extended for meetups in various other locales, including [NCR](https://x.com/sivil_taram/status/1791159335999201380), Pune, Singapore, and Malaysia. Enthusiastic responses and commitments were noted from several members in these regions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sivil_taram/status/1791159335999201380">Tweet from Qian Liu 🔭 (@sivil_taram)</a>: Introducing Sailor-14B Model and Sailor2 Project 🚢  We&#39;re thrilled to announce the release of the Sailor-14B models, including the Base and the Chat versions!  ✅Built upon the Qwen1.5-14B model, ...</li><li><a href="https://huggingface.co/blog/leonardlin/llm-jp-eval-eval">Evaling llm-jp-eval (evals are hard)</a>: no description found</li><li><a href="https://huggingface.co/shisa-ai/shisa-v1-llama3-70b">shisa-ai/shisa-v1-llama3-70b · Hugging Face</a>: no description found</li><li><a href="https://github.com/AUGMXNT/shisa/wiki/A-Review-of-Public-Japanese-Training-Sets#analysis">A Review of Public Japanese Training Sets</a>: Contribute to AUGMXNT/shisa development by creating an account on GitHub.</li><li><a href="https://gist.github.com/cedrickchee/c3d9f8fed88f1c486b883153a64ee7dc">LLM Fine-Tuning for Software Engineers</a>: LLM Fine-Tuning for Software Engineers. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://x.com/cedric_chee/status/1790638025397117031">Tweet from Cedric Chee (@cedric_chee)</a>: When and why to fine-tune an LLM:  - Extremely narrow problem - Prompt engineering is impractical - Quality vs. latency tradeoff - Data privacy  Long-live model fine-tuning.</li><li><a href="https://huggingface.co/augmxnt/shisa-7b-v1">augmxnt/shisa-7b-v1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/augmxnt/shisa-gamma-7b-v1">augmxnt/shisa-gamma-7b-v1 · Hugging Face</a>: no description found</li><li><a href="https://wandb.ai/wandb-japan/llm-leaderboard/reports/Nejumi-LLM-Leaderboard-Evaluating-Japanese-Language-Proficiency--Vmlldzo2MzU3NzIy)">Weights & Biases</a>: Weights & Biases, developer tools for machine learning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1242129546334175393)** (37 messages🔥): 

- **Unlock Modal Credits, Get Decoding**: Members received guidance on obtaining **Modal credits** by filling out the [Modal hackathon credits form](https://bit.ly/modal-credits) after signing up on [modal.com](https://modal.com/signup). Credits amount to $500, valid for one year, and additional $30/month on the free tier.

- **Stay Active to Save on Modal Costs**: A member shared tips on managing **Modal service costs** by setting `container_idle_timeout` to minimize charges during testing. Using GPU services prudently for workloads like LLM serving was emphasized, supported by a [GitHub example](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_mixtral.py).

- **Fine-tuning and Model Serving Tips**: For effective **fine-tuning and serving LLM models** on Modal, `modal serve` is recommended for development over `modal run`. For optimized results, reference the [TensorRT-LLM serving guide](https://modal.com/docs/examples/trtllm_llama) and engage in batch processing.

- **Smoother Experience with Modal Deployments**: Members discussed operational issues like setting `container_idle_timeout` correctly and avoiding repetitive model loading. Valid usage of `modal serve` vs. `modal deploy` was clarified through community insights and links to relevant GitHub projects.

- **Join the Modal Slack for Faster Support**: Members were directed to the [Modal Slack](https://modal.com/slack) for specialized support from the engineering team. Questions suitable for the general or LLMS channels were encouraged for quicker, around-the-clock responses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_mixtral.py">modal-examples/06_gpu_and_ml/llm-serving/vllm_mixtral.py at main · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://bit.ly/modal-credits.">Modal hackathon credits</a>: To claim your Modal credits, sign up for an account at https://modal.com/ first.  Then, let us know your username through this form.   For support, join the Modal Slack.  Here’s some examples to get s...</li><li><a href="https://github.com/modal-labs/llm-finetuning/">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://github.com/satish860/PDF-Extraction-API/blob/main/app.py#L58">PDF-Extraction-API/app.py at main · satish860/PDF-Extraction-API</a>: A Marker Library based API for doing the Marker Response. - satish860/PDF-Extraction-API</li><li><a href="https://modal.com/docs/examples/trtllm_llama">Serverless TensorRT-LLM (LLaMA 3 8B)</a>: In this example, we demonstrate how to use the TensorRT-LLM framework to serve Meta’s LLaMA 3 8B model at a total throughput of roughly 4,500 output tokens per second on a single NVIDIA A100 40GB GPU....</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl.git">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://modal.com/settings/YOURUSERNAME/usage">Sign in</a>: Welcome back to Modal! Sign in to your Modal account by selecting an identity provider below.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis](https://discord.com/channels/1238365980128706560/1241117895740625099/1242545274329763912)** (3 messages): 

- **Running Axolotl Interests Users**: A user expressed interest in running Axolotl, asking specifically for a member’s attention. They shared a direct link to the relevant discussion.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1242350581474136136)** (10 messages🔥): 

- **Credits will be sorted soon**: Updates on credit distribution will be provided shortly. Appreciation was shown for community patience.

- **Axolotl models search issue acknowledged**: Users observed that they can filter but not search for axolotl models on [HuggingFace](https://huggingface.co/models?other=axolotl). It's explained that the search bar uses predefined tags to avoid confusion, and potential UI improvements are discussed to handle additional tags better.

- **Alternative way to filter axolotl models via code**: A user shared a code snippet to filter all axolotl models using the Hugging Face API: 
  ``` 
  from huggingface_hub import HfApi
  hf_api = HfApi()
  models = hf_api.list_models(filter="axolotl")
  ```

- **Positive feedback on hybrid sharding strategy**: A member expressed enthusiasm for the energy and efforts focused on the HYBRID_SHARD strategy, which involves sharding models using Fully Sharded Data Parallel (FSDP) and DeepSpeed (DS) techniques.

**Link mentioned**: <a href="https://huggingface.co/models?other=axolotl)">Models - Hugging Face</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1242342595267395594)** (4 messages): 

- **Credits provision query causing confusion**: A member expressed that they signed up with their email address but have not received the credits yet. In response, another member assured that the credits issue will be sorted out soon and thanked everyone for their patience.

- **Clarifying Replicate's use case**: A member inquired about the primary use case for Replicate, questioning whether it is meant to offer API endpoints for downstream tasks for firms or individuals. They also mentioned specific features like fine-tuning and custom datasets.

- **Registration mismatches being a common issue**: Another member pointed out that their situation mirrored another user's issue regarding different registration methods between Replicate and a conference. This highlights a recurring concern about consistency in user registration methods.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1242284773599215687)** (5 messages): 

- **New Members Join Course**: Two new members announced their enrollment in the course. One user mentioned not receiving their LangSmith credit after signing up.

- **Query About Free Credit**: A member asked whether setting up billing is necessary to receive an additional 250 free credits on top of the existing 250. Another member reassured that credit allocation will be sorted out soon and updates will be provided.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1242485573386637455)** (613 messages🔥🔥🔥): 

- **Debate on Discord Stages and Zoom Chat Integration**: Members discussed the pros and cons of using Discord stages. One participant noted that stages "are just audio only" and another confirmed it, suggesting it's a voice/video/screenshare channel.
  
- **New Course Structure Explained**: Hamelm outlined the three types of sessions for the course: Fine-Tuning Workshops, Office Hours for deeper Q&A, and Conference Talks. Calendar invite titles have been updated to clarify session types.

- **Technical Discussions on Fine-tuning**: In-depth conversations about Llama3 model issues, hyperparameter importance, and multilingual capabilities. Participants referenced specific challenges and shared resources like [Stanford's Pyvene](https://github.com/stanfordnlp/pyvene/issues/46).

- **Resources and Tips Shared**: Numerous links, blog posts, and papers were shared for further reading and resource pooling, such as [Practical Tips for Finetuning LLMs](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) and [Axolotl's GitHub](https://github.com/OpenAccess-AI-Collective/axolotl).

- **Issues with Apple Silicon for Fine-tuning**: Users discussed difficulties using Axolotl on Apple M1 due to bitsandbytes not supporting the architecture. Suggestions such as using Docker or mlx were provided as potential workarounds.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.09673">LoRA Learns Less and Forgets Less</a>: Low-Rank Adaptation (LoRA) is a widely-used parameter-efficient finetuning method for large language models. LoRA saves memory by training only low rank perturbations to selected weight matrices. In t...</li><li><a href="https://arxiv.org/abs/2305.11206">LIMA: Less Is More for Alignment</a>: Large language models are trained in two stages: (1) unsupervised pretraining from raw text, to learn general-purpose representations, and (2) large scale instruction tuning and reinforcement learning...</li><li><a href="https://www.malwarebytes.com/blog/news/2024/04/billions-of-scraped-discord-messages-up-for-sale">Billions of scraped Discord messages up for sale | Malwarebytes</a>: An internet scraping platform is offering access to a database filled with over four billion Discord messages and combined user profiles</li><li><a href="https://huggingface.co/docs/peft/main/en/conceptual_guides/lora">LoRA</a>: no description found</li><li><a href="https://huggingface.co/datasets/GAIR/lima">GAIR/lima · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/bhutanisanyam1/status/1758159687051350189">Tweet from Sanyam Bhutani (@bhutanisanyam1)</a>: LLM Fine-Tuning Benchmarks! 🙏  Super excited to finally publish this report comparing different GPUs and precisions:  - First, why do it and what is it?  - There are MANY GPU benchmarks but few speci...</li><li><a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)</a>: Things I Learned From Hundreds of Experiments</li><li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca/tree/main/data">parlance-labs/hc-mistral-alpaca at main</a>: no description found</li><li><a href="https://hamel.dev/blog/posts/evals/">- Your AI Product Needs Evals</a>: How to construct domain-specific LLM evaluation systems.</li><li><a href="https://x.com/danielhanchen/status/1789659394302718373">Tweet from Daniel Han (@danielhanchen)</a>: Was fixing LLM fine-tuning bugs and found 4 issues:  1. Mistral: HF&#39;s batch_decode output is wrong 2. Llama-3: Be careful of double BOS 3. Gemma: 2nd token has an extra space - GGUF(_Below) = 3064...</li><li><a href="https://huggingface.co/spaces/muellerzr/llm-conf">LLM Conf talk - a Hugging Face Space by muellerzr</a>: no description found</li><li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca/tree/main/configs">parlance-labs/hc-mistral-alpaca at main</a>: no description found</li><li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca">parlance-labs/hc-mistral-alpaca · Hugging Face</a>: no description found</li><li><a href="https://x.com/TheZachMueller/status/1696157965890339148">Tweet from Zach Mueller (@TheZachMueller)</a>: Excited to announce a new @huggingface space to help with one of machine learning&#39;s biggest questions:  How much space does {X} model take in vRAM? And most importantly: when using `device_map=&#3...</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: no description found</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/pretraining.html">Axolotl - Pre-training</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-small-8k-instruct">microsoft/Phi-3-small-8k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer#packing-dataset--constantlengthdataset-">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/config.html">Axolotl - Config options</a>: no description found</li><li><a href="https://outlines-dev.github.io/outlines/">Outlines</a>: Structured text generation with LLMs</li><li><a href="https://en.wiktionary.org/wiki/OTTOMH">OTTOMH - Wiktionary, the free dictionary</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=eGVDKegRdgM">Scaling Up “Vibe Checks” for LLMs - Shreya Shankar | Stanford MLSys #97</a>: Episode 97 of the Stanford MLSys Seminar Series!Scaling Up “Vibe Checks” for LLMsSpeaker: Shreya ShankarBio:Shreya Shankar is a PhD student in computer scien...</li><li><a href="https://www.honeycomb.io/blog/introducing-query-assistant">Observability, Meet Natural Language Querying with Query Assistant </a>: Announcing Query Assistant, the first introduction of AI into Honeycomb. With Query Assistant, you can describe/ask things in plain English.</li><li><a href="https://huggingface.co/collections/leonardlin/multilingual-6594d0ea075245eadd6aa99c">multilingual - a leonardlin Collection</a>: no description found</li><li><a href="https://x.com/HamelHusain/status/1784769559364608222">Tweet from Hamel Husain (@HamelHusain)</a>: Llama 3 70b function calling works pretty well out of the box with prompting only 🚀💰   See the below demo (prompt and code in next tweet)</li><li><a href="https://github.com/TimDettmers/bitsandbytes/blob/main/CHANGELOG.md">bitsandbytes/CHANGELOG.md at main · TimDettmers/bitsandbytes</a>: Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes</li><li><a href="https://github.com/stanfordnlp/pyvene/issues/46">[P1] Support more huggingface (transformer-based) models · Issue #46 · stanfordnlp/pyvene</a>: Descriptions: Ideally, all the models listed here can be supported by this library without exposing the model details to the users of this library. This requires we set up model folders for all mod...</li><li><a href="https://github.com/argilla-io/distilabel/blob/main/examples/structured_generation_with_outlines.py">distilabel/examples/structured_generation_with_outlines.py at main · argilla-io/distilabel</a>: ⚗️ distilabel is a framework for synthetic data and AI feedback for AI engineers that require high-quality outputs, full data ownership, and overall efficiency. - argilla-io/distilabel</li><li><a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">Installing the NVIDIA Container Toolkit &mdash; NVIDIA Container Toolkit 1.15.0 documentation</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=jkrNMKz9pWU">A Hackers&#39; Guide to Language Models</a>: In this deeply informative video, Jeremy Howard, co-founder of fast.ai and creator of the ULMFiT approach on which all modern language models (LMs) are based...</li><li><a href="https://huggingface.co/models?other=axolotl">Models - Hugging Face</a>: no description found</li><li><a href="https://poe.com/s/c0BFLNhTwiyPXOulPCnO">you have a column with each element containing a list of tuple. get the frequency of the appearance of each tuple</a>: TrinoAgentEx: Which SQL keyword do you want to learn about? TrinoAgentEx: To query a frequency distribution of tuples within a list in a single Trino SQL query, you&#x27;ll have to perform several ope...</li><li><a href="https://discord.gg/2YkbgY5TQj">Join the Axolotl AI Discord Server!</a>: Check out the Axolotl AI community on Discord - hang out with 2197 other members and enjoy free voice and text chat.</li><li><a href="https://x.com/abacaj/status/1782835550396850449">Tweet from anton (@abacaj)</a>: Phi-3 seems pretty good, an improvement over phi-2 for sure. The long context 128k seems very useful for extracting information and document processing given that the model is quite small it can be de...</li><li><a href="https://lake-scilla-bc6.notion.site/LLM-fine-tuning-workshop-6832ed2266a14957831ed8e2b3a959b3">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://github.com/ml-explore/mlx">GitHub - ml-explore/mlx: MLX: An array framework for Apple silicon</a>: MLX: An array framework for Apple silicon. Contribute to ml-explore/mlx development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/949">Evaluation took much more time when enable eval_table_size  · Issue #949 · OpenAccess-AI-Collective/axolotl</a>: Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior The evaluation time is expected to increase but not...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb">axolotl/examples/colab-notebooks/colab-axolotl-example.ipynb at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/parlance-labs/ftcourse">GitHub - parlance-labs/ftcourse</a>: Contribute to parlance-labs/ftcourse development by creating an account on GitHub.</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: Structured Text Generation</a>: Structured Text Generation. Contribute to outlines-dev/outlines development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docker/Dockerfile-cloud#L8">axolotl/docker/Dockerfile-cloud at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html">Axolotl - Template-free prompt construction</a>: no description found</li><li><a href="https://docs.google.com/presentation/d/1MC8JqXf9SU9fEYh6RhXPzF8LjAjpmdrmUMTWnPpi79Y/edit?usp=sharing">Cutting Edge Tricks</a>: SANYAM BHUTANI Sr Data Scientist, H2O.ai</li><li><a href="https://lightning.ai/pages/community/lora-insights/">Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments - Lightning AI</a>: LoRA is one of the most widely used, parameter-efficient finetuning techniques for training custom LLMs. From saving memory with QLoRA to selecting the optimal LoRA settings, this article provides pra...</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Templates for Chat Models</a>: no description found</li><li><a href="https://github.com/parlance-labs/ftcourse/tree/master">GitHub - parlance-labs/ftcourse</a>: Contribute to parlance-labs/ftcourse development by creating an account on GitHub.</li><li><a href="https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2">Fine-Tuning LLMs: In-Depth Analysis with LLAMA-2 | Anyscale</a>: In this blog, we compare full-parameter fine-tuning with LoRA and answer questions around the strengths and weaknesses of the two techniques. </li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples/llama-3">axolotl/examples/llama-3 at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/parlance-labs/ftcourse/tree/master/sample_data">ftcourse/sample_data at master · parlance-labs/ftcourse</a>: Contribute to parlance-labs/ftcourse development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=PXWYUTMt-AU">LoRA: Low-Rank Adaptation of Large Language Models - Explained visually + PyTorch code from scratch</a>: A full visual explanation of LoRA, with PyTorch code form scratch!Full code and slides are available on my GitHub: https://github.com/hkproj/pytorch-loraChap...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436">ERROR: No matching distribution found for bitsandbytes==0.43.0 for macOS  · Issue #1436 · OpenAccess-AI-Collective/axolotl</a>: Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior The command pip3 install -e &#39;.[flash-attn,deeps...</li><li><a href="https://buttondown.email/ainews">AI News</a>: We summarize top AI discords + AI reddits + AI X/Twitters, and send you a roundup each day! See archive for examples.  &quot;Highest-leverage 45 mins I spend everyday&quot; - Soumith &quot;best AI new...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1589">8-Bit DoRA training with FSDP doesn&#39;t work, but 4-bit QDoRA does / peft_use_dora is ignored? · Issue #1589 · OpenAccess-AI-Collective/axolotl</a>: Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior With 8-bit LoRA enabled and peft_use_dora: true, th...</li><li><a href="https://x.com/sroecker/status/1757103619705299061?t=uajfu81xkUp7x80xgQ7i1A&s=19">Tweet from Steffen Röcker (@sroecker)</a>: Ever wondered how to fine-tune LLMs using @axolotl_ai and @Podman_io?  Follow the instructions for NVIDIA toolkit CDI and simply run &#34;podman run --rm --device http://nvidia.com/gpu=all --security-...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/peft/pull/1724">FIX Allow DoRA init on CPU when using BNB by BenjaminBossan · Pull Request #1724 · huggingface/peft</a>: Resolves #1674 For some users, it is necessary to initialize the model on CPU, even when using BitsAndBytes, which requires a GPU eventually. Since DoRA requires to dequantize the BNB weights at in...</li><li><a href="https://lu.ma/terrible-ai-systems?utm_source=llm">How to Build Terrible AI Systems with Jason Liu · Luma</a>: Jason is an independent consultant who uses his expertise in recommendation systems to help fast-growing startups build out their RAG applications. He was…</li><li><a href="https://nbsanity.com/static/d06085f1dacae8c9de9402f2d7428de2/demo.html">Llama-3 Function Calling Demo</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1791900967472140583">Tweet from Daniel Han (@danielhanchen)</a>: My take on &#34;LoRA Learns Less and Forgets Less&#34;  1) &#34;MLP/All&#34; did not include gate_proj. QKVO, up & down trained but not gate (pg 3 footnote)  2) Why does LoRA perform well on math and ...</li><li><a href="https://www.guardrailsai.com/">Guardrails AI</a>: Enforce assurance for LLM applications
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1242526610024824942)** (3 messages): 

```html
- **Jason's W&B course wows**: A user expressed excitement about Jason's session and mentioned being halfway through his **Weights & Biases (W&B) course**. They used the teacher emoji to show their admiration.
- **Prompt engineering curiosity peaks**: Another user inquired about Jason's systematic approach to prompt engineering, praising his extensive work on optimizing prompts. They were eager to learn his "recipe" during his workshop session.
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1242489403129987194)** (2 messages): 

- **Gradio Maintainer Introduces Himself**: Freddy, a maintainer of **Gradio**, a Python library for developing user interfaces for AI models, invited members to ask questions and share demos. He provided links to [Gradio's quickstart guide](https://www.gradio.app/guides/quickstart) and another guide on how to [build a chatbot in 5 lines of code](https://www.gradio.app/guides/creating-a-chatbot-fast).
- **Member Shows Interest in Gradio**: A member expressed gratitude for the shared resources and mentioned they will eventually have questions, particularly related to an **A1111-extension** they had previously worked on and found challenging.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.gradio.app/guides/quickstart">Quickstart</a>: A Step-by-Step Gradio Tutorial</li><li><a href="https://www.gradio.app/guides/creating-a-chatbot-fast">Creating A Chatbot Fast</a>: A Step-by-Step Gradio Tutorial
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[askolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1242543726312689705)** (13 messages🔥): 

- **Issue with bitsandbytes on macOS**: An issue related to installing bitsandbytes on macOS is discussed in [this GitHub thread](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436). The specific error is *"No matching distribution found for bitsandbytes==0.43.0 for macOS"*.
- **MLX support not yet available**: A member pointed out that Axolotl does not yet support MLX, referencing [an open issue on GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1119). MLX is praised for its efficiency in fine-tuning large language models on consumer hardware.
- **Fine-tuning comparison: OpenAI vs Axolotl**: One user shared their experience using OpenAI for fine-tuning, stating it takes about 30 minutes and charges per token. They queried how Axolotl compares in terms of time and cost for fine-tuning.
- **Apple M1 not ideal for fine-tuning**: A statement highlighted that Apple ARM (M1) does not support q4 and q8, making it less suitable for fine-tuning. The user was advised to rent a Linux GPU server on RunPod instead.
- **MLX-examples for guidance**: For those interested in using MLX, a reference was provided to the [MLX examples documentation](https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md) on GitHub for further guidance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md">mlx-examples/lora/README.md at main · ml-explore/mlx-examples</a>: Examples in the MLX framework. Contribute to ml-explore/mlx-examples development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436">ERROR: No matching distribution found for bitsandbytes==0.43.0 for macOS  · Issue #1436 · OpenAccess-AI-Collective/axolotl</a>: Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior The command pip3 install -e &#39;.[flash-attn,deeps...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1119">MLX Support · Issue #1119 · OpenAccess-AI-Collective/axolotl</a>: Hi, It would be great to have MLX support in Axolotl. MLX has been shown to be able to quickly and efficiently finetune many LLMs, including 7B LLMs on consumer hardware. Thank you! (edit: update)
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1242565467562967152)** (1 messages): 

- **Accelerate your PyTorch with Accelerate**: A member shared a [presentation on Hugging Face's Spaces](https://huggingface.co/spaces/muellerzr/llm-conf) introducing Accelerate, a library that simplifies running PyTorch code across any distributed configuration. The linked [Accelerate documentation](https://huggingface.co/docs/accelerate) shows how to implement it with just a few lines of code.

- **Accelerate Features Quicktour**: The [Quicktour on Hugging Face](https://huggingface.co/docs/accelerate/quicktour) illustrates Accelerate's features including a unified command line interface for distributed training scripts, a training library for PyTorch, and Big Model Inference support for large models.

- **Examples to Get You Started**: A collection of examples is available on [Hugging Face's GitHub](https://github.com/huggingface/accelerate/tree/main/examples), recommended to start with `nlp_example`. The examples showcase the versatility of Accelerate in handling various distributed training setups.

- **In-Depth Model Memory Estimators**: Members shared links to [a memory usage estimator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) and the [TransformerAnalyzer tool](https://huggingface.co/spaces/cllatMTK/TransformerAnalyzer) which provides detailed FLOPS and other parameter estimates, useful for understanding model requirements.

- **Run Large Language Models Efficiently**: The [Can I Run it LLM Edition](https://huggingface.co/spaces/Vokturz/can-it-run-llm) space, discussed on Hugging Face, focuses on inference abilities, highlighting LoRa applicability for efficient large language model deployment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/muellerzr/llm-conf">LLM Conf talk - a Hugging Face Space by muellerzr</a>: no description found</li><li><a href="https://huggingface.co/docs/accelerate">Accelerate</a>: no description found</li><li><a href="https://huggingface.co/docs/accelerate/quicktour">Quicktour</a>: no description found</li><li><a href="https://github.com/huggingface/accelerate/tree/main/examples">accelerate/examples at main · huggingface/accelerate</a>: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....</li><li><a href="https://huggingface.co/spaces/hf-accelerate/model-memory-usage">Model Memory Utility - a Hugging Face Space by hf-accelerate</a>: no description found</li><li><a href="https://huggingface.co/spaces/Vokturz/can-it-run-llm">Can You Run It? LLM version - a Hugging Face Space by Vokturz</a>: no description found</li><li><a href="https://huggingface.co/spaces/cllatMTK/TransformerAnalyzer">TransformerAnalyzer - a Hugging Face Space by cllatMTK</a>: no description found
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1242522009758470174)** (1 messages): 

- **Perplexity AI partners with Tako for advanced knowledge search**: *"We’re teaming up with Tako to bring advanced knowledge search and visualization to our users."* This allows users to search, juxtapose, and share authoritative knowledge cards within Perplexity, initially available in the U.S. and in English, with mobile access coming soon. [Read about our partnership](https://trytako.com/blog/introducing-tako-and-perplexity-integration).

**Link mentioned**: <a href="https://trytako.com/blog/introducing-tako-and-perplexity-integration">Tako</a>: no description found

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1242024539555106878)** (735 messages🔥🔥🔥): 

```html
- **Loyalty to platforms debated**: One member shared their experience using Perplexity and Gemini, emphasizing that users have "zero loyalty" and praised Perplexity for its direct answers ([Tenor GIF](https://tenor.com/view/oh-no-homer-simpsons-hide-disappear-gif-16799752)).
- **Perplexity’s feature tips shared**: There was a discussion about using Perplexity with various functionalities, including understanding the API, tweaking search engine options in browsers like Firefox, and handling system prompts.
- **Perplexity temporarily down**: Multiple users reported issues with Perplexity being down; they sympathized over missing the service and speculated on maintenance and updates.
- **Model preferences and uses discussed**: Members compared models like GPT-4o and Claude 3 Opus, discussing their strengths and preferences for tasks such as creative writing and coding ([Spectrum IEEE article](https://spectrum.ieee.org/perplexity-ai)).
- **Interactive features in Perplexity**: Members were curious about and shared tips on using Perplexity's new features like Tako charts, with some mentioning tips like adding `since:YYYY/01/01` to improve search results. 
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://violentmonkey.github.io/">no title found</a>: no description found</li><li><a href="https://apps.apple.com/us/app/elevenlabs-reader-ai-audio/id6479373050">‎ElevenLabs Reader: AI Audio</a>: ‎Convert text into natural, expressive speech. Ideal for articles, ePubs, PDFs, or any text. ElevenLabs Reader puts our most capable Text to Speech (TTS) model in your pocket.  App Features Text Reade...</li><li><a href="https://docs.perplexity.ai/docs/perplexitybot">PerplexityBot</a>: no description found</li><li><a href="https://spectrum.ieee.org/perplexity-ai">Perplexity.ai Turns Tables on Google, Upends SEO Credos</a>: AI search leader mixes Meta-built smarts with scrappy startup fervor</li><li><a href="https://x.com/bobbyallyn/status/1792679435701014908?s=46">Tweet from Bobby Allyn (@BobbyAllyn)</a>: Statement from Scarlett Johansson on the OpenAI situation. Wow:</li><li><a href="https://greasyfork.org/en/scripts/490634-perplexity-model-selection">Perplexity Model Selection</a>: Adds model selection buttons to Perplexity AI using jQuery</li><li><a href="https://tenor.com/view/oh-no-homer-simpsons-hide-disappear-gif-16799752">Oh No Homer GIF - Oh No Homer Simpsons - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://blogs.microsoft.com/blog/2024/05/20/introducing-copilot-pcs/">Introducing Copilot+ PCs - The Official Microsoft Blog</a>: An on-demand recording of our May 20 event is available. Today, at a special event on our new Microsoft campus, we introduced the world to a new category of Windows PCs designed for AI, Copilot+ PCs. ...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1242134770088149032)** (9 messages🔥): 

- **Historical Questions Answered via Perplexity AI**: A member shared a link asking *"Qui est Adolf?"* featuring detailed historical insights. [Explore here](https://www.perplexity.ai/search/Qui-est-adolf-TQqGm0aDRRWWqeblJpYUgg#5).

- **Understanding Ideal Structures in Mathematics**: A link was posted addressing the question *"Does every ideal?"* which delves into complex mathematical theories. [Explore here](https://www.perplexity.ai/search/Does-every-ideal-hQP30OxPQjqQIg4cK.sFDA#0).

- **Script Creation Query via Perplexity**: A user shared a search for *"Create a script,"* likely aimed at generating specific scripts or code snippets. [Explore here](https://www.perplexity.ai/search/Create-a-script-ZkKbE43aRhyXn3HIlXADUg).

- **Exploring Technical Concepts in Computing**: One member asked *"what is layer?"* in a Perplexity AI search, touching upon detailed discussions in computing or machine learning. [Explore here](https://www.perplexity.ai/search/what-is-layer-xXVSIKHpT2uGOqogIZmOVw).

- **Discussion on Indoor Topics**: Another search titled *"talk about indoor"* suggested a focus on indoor environments or activities. [Explore here](https://www.perplexity.ai/search/talk-about-indoor-Wkghx1CeTwuZWH_gcxDpJw).
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1242229643336224879)** (98 messages🔥🔥): 

```html
- **Struggles with Perplexity API on Open WebUI**: A user reported issues with model compatibility, noting, "it works perfectly fine with OpenAI (Closed) and Groq, but maybe they don’t have the model names setup to work with PPLX." Another user suggested using `api.perplexity.ai` directly but discovered Perplexity doesn't have a `/models` endpoint, causing further complications.
- **Proxy Server Solution and Execution Assistance**: A workaround was proposed to create a local server that proxies the models and chat completions endpoints. A user mentioned completing the proxy and instructing, "you need to add the `--network=host` to your docker command" to fix localhost issues.
- **Docker Configuration Conversations**: Users discussed the intricacies of Docker configurations, with one summarizing the correct command, "docker run -d --network=host -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main," while troubleshooting connection issues.
- **Inquiries about Sending Images**: When asked, "Is there a way to send images via the API?", it was clarified that currently, Perplexity's API only supports text, stating, "they are just using Claude and Openai vision api," and the LLAVA models that support images are not available via API.
- **User Appreciation and Final Adjustments**: One user showed gratitude saying, "Thank you, 🙂" while another user confirmed they needed to align Docker configurations to ensure proper API functionality. This indicates ongoing effort and collaboration to resolve the issues.
```

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:8080`">no title found</a>: no description found</li><li><a href="https://docs.openwebui.com/">🏡 Home | Open WebUI</a>: Open WebUI is an extensible, feature-rich, and user-friendly self-hosted WebUI designed to operate entirely offline. It supports various LLM runners, including Ollama and OpenAI-compatible APIs.
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1242544374726791270)** (1 messages): 

- **Phi-3 Models Take the Stage**: Microsoft released **Phi-3 small (7B)** and **Phi-3 medium (14B)** models with up to 128k context, achieving impressive scores on MMLU and AGI Eval. Check them out [here](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)!

- **$10M of Compute Up for Grabs**: Hugging Face announced a **$10M commitment to free GPU access** through ZeroGPU, facilitating AI demo creation for indie and academic AI builders. Learn more about the initiative [here](https://huggingface.co/zero-gpu-explorers).

- **Transformers 4.41.0 packed with new features**: The latest update includes **Phi3, JetMoE, PaliGemma, VideoLlava, and Falcon 2**, as well as improved support for **GGUF, watermarking, and new quant methods like HQQ and EETQ**. Full release notes are available [here](https://github.com/huggingface/transformers/releases/tag/v4.41.0).

- **LangChain Integration Simplified**: New **langchain-huggingface package** facilitates seamless integration of Hugging Face models into LangChain. Check out the [announcement and details](https://huggingface.co/blog/langchain).

- **CommonCanvas and Moondream Updates**: **CommonCanvas** released the first open-source text-to-image models trained on Creative Commons images, with the [largest dataset](https://huggingface.co/common-canvas) available on Hugging Face. **Moondream** now runs directly in browsers via WebGPU, improving user privacy.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ClementDelangue/status/1791115403734778185)">Tweet from clem 🤗 (@ClementDelangue)</a>: GPU-Poor no more: super excited to officially release ZeroGPU in beta today. Congrats @victormustar & team for the release!  In the past few months, the open-source AI community has been thriving. Not...</li><li><a href="https://x.com/LysandreJik/status/1792923587340390733)">Tweet from Lysandre (@LysandreJik)</a>: From a model page to your Local App in seconds, the @huggingface  Hub welcomes Local Apps!  Suggest your favorite Local App leveraging the Hub there to get them added to the dropdown and ✨ deep linked...</li><li><a href="https://x.com/osanseviero/status/1792904237153722569)">Tweet from Omar Sanseviero (@osanseviero)</a>: Transformers 4.41.0 has lots of goodies🤗  🥳 New models: Phi3, JetMoE, PaliGemma, VideoLlava, and Falcon 2. 🤯 GGUF support with from_pretrained 🤏 New quant methods: HQQ and EETQ 🔍 Watermarking sup...</li><li><a href="https://x.com/_philschmid/status/1790419788931416466)">Tweet from Philipp Schmid (@_philschmid)</a>: We are excited to announce huggingface-langchain🚀 A new open-source package to seamlessly integrate the latest open Models from @huggingface into @LangChainAI, supporting local models hosted models! ...</li><li><a href="https://x.com/multimodalart/status/1791201296357142663)">Tweet from apolinario (multimodal.art) (@multimodalart)</a>: Quite excited that CommonCanvas is JUST out! 🖼️  • First open source text-to-image models trained fully on openly licensed images (SD2 and SDXL architectures)  • The dataset, with ~70M openly license...</li><li><a href="https://x.com/xenovacom/status/1791436796498174047)">Tweet from Xenova (@xenovacom)</a>: Moondream, your favorite tiny vision language model by @vikhyatk can now run directly in the browser on WebGPU! 🤯 Powered, of course, by Transformers.js and ONNX Runtime Web! 🤗  Local inference mean...</li><li><a href="https://x.com/xenovacom/status/1792570966272336074)">Tweet from Xenova (@xenovacom)</a>: You can now use 🤗 Transformers.js with Google Visual Blocks, a visual programming framework that lets you create machine learning pipelines in a no-code graph editor!  🛠️ Rapid workflow prototyping ...</li><li><a href="https://x.com/IlysMoutawwakil/status/1791406503112704455)">Tweet from Ilyas Moutawwakil (@IlysMoutawwakil)</a>: Optimum-Benchmark on PyPI 🎉 But why now ? 🤔 Because it&#39;s getting integrated in Transformers&#39; benchmarking workflow 😍 Your favorite transformers will only get faster and lighter ; Kudos to @...</li><li><a href="https://x.com/osanseviero/status/1791567896482635801)">Tweet from Omar Sanseviero (@osanseviero)</a>: Curious about LLMs? Join this Fine-Tuning course with top experts! 🚀  @huggingface is offering $501.42 in GPU credits for can Space demos, fine-tuning, inference, and more! Enjoy 🤗  https://maven.co...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1242025543457833002)** (678 messages🔥🔥🔥): 

- **Voice Models Showdown**: A user shared links to two notable text-to-speech models, [bark by Suno on Hugging Face](https://huggingface.co/suno/bark) and the paid service [Eleven Labs](https://elevenlabs.io/), and inquired about the underlying models used in [Udio](https://www.udio.com).
- **Git LFS Upload Issues**: Multiple users discussed troubleshooting issues related to uploading large files using git LFS to Hugging Face repositories. Suggestions included using the `upload_file` function from the `huggingface_hub` library.
- **Language Model Specifications**: There was a discussion surrounding the largest language models with references to [GPT-4](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) and Google's 1.5 trillion parameter model, and an exploration into optimizing Falcon-180B and Llama models.
- **Hugging Face Store Anticipation**: Users expressed excitement and impatience for the reopening of the Hugging Face merchandise store, highlighting a strong community desire for official swag.
- **Job Application Success**: Congratulations and best wishes were shared with members who had applied for roles at Hugging Face, reflecting the community's support and encouragement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pypi.org/project/ratelimiter/">ratelimiter</a>: Simple python rate limiting object</li><li><a href="https://huggingface.co/spaces/parler-tts/parler-tts-expresso">Parler TTS Expresso - a Hugging Face Space by parler-tts</a>: no description found</li><li><a href="https://huggingface.co/spaces/parler-tts/parler_tts_mini">Parler-TTS Mini - a Hugging Face Space by parler-tts</a>: no description found</li><li><a href="https://huggingface.co/learn/deep-rl-course/unit1/hands-on#install-dependencies-and-create-a-virtual-screen-">Train your first Deep Reinforcement Learning Agent 🤖 - Hugging Face Deep RL Course</a>: no description found</li><li><a href="https://x.com/kuldeep_s_s/status/1792296168111628717">Tweet from Kuldeep Singh Sidhu (@kuldeep_s_s)</a>: You are happy that @Meta has open-sourced Llama 3 😃... So you jump on HuggingFace Hub to download the new shiny Llama 3 model only to see a few quintillion Llama 3&#39;s! 🦙✨  Which one should you us...</li><li><a href="https://huggingface.co/mlabonne/Meta-Llama-3-225B-Instruct">mlabonne/Meta-Llama-3-225B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/gemma-peft">Fine-Tuning Gemma Models in Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/google/switch-c-2048">google/switch-c-2048 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/kanye-ye-kanye-west-ty-dolla-vultures-gif-3313542573422740922">Kanye Kanye West GIF - Kanye Ye Kanye west - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="http://hf.co/papers">Daily Papers - Hugging Face</a>: no description found</li><li><a href="https://tenor.com/p9BpiQov0bB.gif">Skibidi Toilet GIF - Skibidi toilet Skibidi Toilet - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/minor-spelling-mistake-gif-21179057">Minor Spelling Mistake GIF - Minor Spelling Mistake - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/cowboy-hug-brokeback-mountain-couple-gay-gif-5066019591388392130">Cowboy Hug GIF - Cowboy Hug Brokeback Mountain - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/main/app.py#:~:text=if%20reaction.message.author.id%20!%3D%20user.id%3A%20%23%20can%27t%20earn%20while%20self%2Dreacting%2C%20which%20is%20abuseable)">app.py · huggingface-projects/LevelBot at main</a>: no description found</li><li><a href="https://tenor.com/bjKth.gif">Idk Shrug GIF - Idk Shrug Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/datasets/H-D-T/Buzz">H-D-T/Buzz · Datasets at Hugging Face</a>: no description found</li><li><a href="https://elevenlabs.io/">Text to Speech &amp; AI Voice Generator</a>: Create premium AI voices for free in any style and language with the most powerful online AI text to speech (TTS) software ever. Generate text-to-speech voiceovers in minutes with our character AI voi...</li><li><a href="https://www.udio.com/">Udio | AI Music Generator - Official Website</a>: Discover, create, and share music with the world. Use the latest technology to create AI music in seconds.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1242134561761267755)** (2 messages): 

- **Working on ImageBind integration for Transformers**: One member mentioned, *"Working on adding ImageBind to `transformers`."*  While details were sparse, this suggests ongoing efforts to enhance the capabilities of the Transformers library.
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1242139476772388884)** (13 messages🔥): 

- **Merve showcases PaliGemma's document models**: *"Quoting merve (@mervenoyann) I got asked about PaliGemma's document understanding capabilities..."*. For more details, refer to the [tweet](https://x.com/giffmana/status/1791541209883717973?s=46).
  
- **DeepSpeech inquiry**: A member asked, *"has anyone here worked with mozillas deepspeech?"*, capturing interest around Mozilla’s DeepSpeech project.

- **LangChain to LangGraph transition guide**: An in-depth guide on upgrading from legacy LangChain to LangGraph was shared through an [article](https://medium.com/ai-advances/upgrading-your-agents-a-smooth-transition-from-legacy-langchain-to-langgraph-c552cb60fcb3).

- **Leveraging LLMs in Magnolia CMS**: A member shared insights into using LLMs for content creation in Magnolia CMS via [this Medium post](https://joaquin-alfaro.medium.com/openai-as-writing-assistant-in-magnolia-cms-7052a4715201).

- **Curated 3D Gaussian Splatting resources**: A comprehensive list of 3D Gaussian Splatting papers and resources, with significant potential in robotics and embodied AI, was highlighted in this [GitHub repository](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#editing).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/giffmana/status/1791541209883717973?s=46">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: Merve just casually being on fire:  Quoting merve (@mervenoyann)   I got asked about PaliGemma&#39;s document understanding capabilities, so I built a Space that has all the PaliGemma fine-tuned doc m...</li><li><a href="https://huggingface.co/papers/2301.13276">Paper page - Distributed Swarm Intelligence</a>: no description found</li><li><a href="https://huggingface.co/docs/evaluate/base_evaluator#evaluate-models-on-the-hub">Using the `evaluator`</a>: no description found</li><li><a href="https://github.com/anthonyrussano/wikitweet/blob/main/tweet-natural-healing-thread.py">wikitweet/tweet-natural-healing-thread.py at main · anthonyrussano/wikitweet</a>: Contribute to anthonyrussano/wikitweet development by creating an account on GitHub.</li><li><a href="https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#editing">GitHub - MrNeRF/awesome-3D-gaussian-splatting: Curated list of papers and resources focused on 3D Gaussian Splatting, intended to keep pace with the anticipated surge of research in the coming months.</a>: Curated list of papers and resources focused on 3D Gaussian Splatting, intended to keep pace with the anticipated surge of research in the coming months. - MrNeRF/awesome-3D-gaussian-splatting
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1242054556188151858)** (15 messages🔥): 

- **Announcing Sdxl Flash Mini**: A member announced the release of **SDXL Flash Mini** in collaboration with [Project Fluently](https://hf.co/fluently). The model is described to be fast and efficient, with less resource consumption while maintaining respectable quality levels [SDXL Flash Mini](https://huggingface.co/sd-community/sdxl-flash-mini).

- **SDXL Flash Demo by KingNish**: Exciting new demo of **SDXL Flash** available on Hugging Face Spaces, demonstrated by KingNish. This provides a practical showcase of its capabilities [SDXL Flash Demo](https://huggingface.co/spaces/KingNish/SDXL-Flash).

- **Tokun Tokenizer Release**: Inspired by Andrej Karpathy, a member developed a new tokenizer called **Tokun**, aimed at significantly reducing model size while enhancing capabilities. Shared both the [GitHub project](https://github.com/apehex/tokun) and [article about testing](https://x.com/4pe0x/status/1792638900059385942).

- **Transformers Library Contribution**: A member celebrated their PR merge into the **Transformers library**, which fixes an issue with finetuned AI models and custom pipelines. Shared the link for the PR [here](https://github.com/huggingface/transformers/pull/29004).

- **llama-cpp-agent Using ZeroGPU**: The member shared the creation of **llama-cpp-agent** on Hugging Face Spaces utilizing ZeroGPU technology, indicating a promising advancement in computational efficiency [llama-cpp-agent](https://huggingface.co/spaces/pabloce/llama-cpp-agent).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://swiftapi.pro/">Swift API</a>: no description found</li><li><a href="https://x.com/4pe0x/status/1792638900059385942">Tweet from Apehex (@4pe0x)</a>: Excited to introduce `tokun`, a game-changing #tokenizer for #LLM.  It could bring the size of #llama3 down by a factor 10 while improving capabilities!  https://github.com/apehex/tokun/blob/main/arti...</li><li><a href="https://huggingface.co/spaces/KingNish/SDXL-Flash">SDXL Flash - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://huggingface.co/sd-community/sdxl-flash-mini">sd-community/sdxl-flash-mini · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/pabloce/llama-cpp-agent">Llama Cpp Agent - a Hugging Face Space by pabloce</a>: no description found</li><li><a href="https://github.com/formentor-studio/magnolia-ai-contents">GitHub - formentor-studio/magnolia-ai-contents: Generation of contents in Magnolia CMS using AI</a>: Generation of contents in Magnolia CMS using AI. Contribute to formentor-studio/magnolia-ai-contents development by creating an account on GitHub.</li><li><a href="https://github.com/apehex/tokun">GitHub - apehex/tokun: tokun to can tokens</a>: tokun to can tokens. Contribute to apehex/tokun development by creating an account on GitHub.</li><li><a href="https://github.com/wikip-co/wikip.co">GitHub - wikip-co/wikip.co: A static wiki built with node.js</a>: A static wiki built with node.js. Contribute to wikip-co/wikip.co development by creating an account on GitHub.</li><li><a href="https://github.com/branyang02/notie">GitHub - branyang02/notie: Personal markdown notetaking app.</a>: Personal markdown notetaking app. Contribute to branyang02/notie development by creating an account on GitHub.</li><li><a href="https://notie-nine.vercel.app/">Notie</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1242324285541056522)** (4 messages): 

- **LLMs struggle with story enhancements**: One member found that using **llama3 8b 4bit** to implement "Creating Suspenseful Stories: Iterative Planning with Large Language Models" was ineffective. The LLM could critique the plot proficiently but failed to enhance it when fed the critique, exemplifying a notable limitation of current models.
- **Need for better prompts or bigger models**: Another member acknowledged the trend where **LLMs are better at critiquing** than improving based on that critique, suggesting the need for at least **13b models or better prompts** like chain-of-thought (CoT) to achieve more effective results.
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1242062334118199306)** (2 messages): 

- **Seeking Advanced Vision Transformer Techniques**: A user inquired about **papers explaining patching techniques in Vision Transformers** that are more advanced than VIT. They are looking for in-depth resources to expand their knowledge on this topic.
- **Zero-Shot Object Detection in Screenshots**: Another user described a task involving **finding all objects similar to a reference image within a webpage screenshot**, emphasizing the need for zero-shot methods due to the reference image always changing. They are seeking guidance or solutions on achieving this capability efficiently.
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1242029949796225064)** (12 messages🔥): 

- **LLMs Forget Conversations, Store Histories Manually**: A user expressed difficulty with their bot not considering conversation history. Members advised to manually concatenate previous messages as **LLMs** inherently do not remember previous exchanges. [GitHub repository for the bot](https://github.com/jakobdylanc/discord-llm-chatbot).

- **Comparing Runtimes: Gemini 1.5 Flash vs Llama3-70B**: A user noted that **Llama3-70B** provides accurate data pattern analysis and truthful answers, while **Gemini Flash** tends to hallucinate. This suggests Llama3-70B's stronger performance in complex data scenarios.

- **Ensemble Model for Hallucination Detection**: A member working on a master thesis shared their approach using an ensemble of **Mistral 7B** models to measure different types of uncertainty. They asked for questions potentially lying outside the model's training data to test for increased epistemic uncertainty as an indicator of hallucinations.

- **Hosting Fine-Tuned LLMs on HuggingFace**: A user asked about hosting a fine-tuned **LLM** on HuggingFace and using an API for requests. They were confident, saying, *"like 99.9%"* sure it can be done.
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1242026772761940049)** (10 messages🔥): 

- **French-to-English translation request in Diffusion channel**: A user initially posted in French, then translated their message to English, explaining an issue with the [llmcord chatbot](https://github.com/jakobdylanc/discord-llm-chatbot) not retaining conversation history. Another member suggested that such queries are more appropriate for NLP channels rather than the Diffusion Discussions channel.

- **LLMcord chatbot conversation history tip**: Another user recommended a solution for the conversation history problem by sending the history within the prompt. They shared a link to the [LangChain documentation](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/) which explains how to manage chat message history.

- **Diffusion model denoiser issue and math inquiry**: A user shared their struggle with implementing a diffusion model, mentioning success with the forward diffusion process but issues with the denoiser. They asked for advice on which math field to study, specifically inquiring about fields related to gaussians and normal distributions; another user suggested studying *variational inference*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/jakobdylanc/discord-llm-chatbot">GitHub - jakobdylanc/discord-llm-chatbot: llmcord.py • Talk to LLMs with your friends!</a>: llmcord.py • Talk to LLMs with your friends! Contribute to jakobdylanc/discord-llm-chatbot development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/">Add message history (memory) | 🦜️🔗 LangChain</a>: The RunnableWithMessageHistory lets us add message history to certain types of chains. It wraps another Runnable and manages the chat message history for it.
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1242018499040116736)** (402 messages🔥🔥): 

- **Scarlett Johansson sues OpenAI for voice replication**: *Reported details on Scarlett Johansson suing OpenAI for generating her voice* and discussed potential legal implications. Members noted that *OpenAI has since removed the voice* amidst public backlash.
- **Phi-3 model release shakes things up**: Microsoft released the **Phi-3-Medium-128K-Instruct model** on [Hugging Face](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct), boasting improved benchmarks and up to 128k context. Participants debated its performance and potential issues with the context length.
- **Colab issues linked to PyTorch's T4 GPU detection**: Due to PyTorch misidentifying Tesla T4's capabilities, **Colab notebooks misbehaved** until an [update from Unsloth's side](https://x.com/danielhanchen/status/1792985678030221464) was implemented. A tweet by Daniel Hanchen confirmed the recognition glitch.
- **Diverse finetuning discussions**: Discussions ranged from the **use of multiple GPUs** to **fine-tuning models on Google Cloud vs. Colab**. Practical nuances of fine-tuning included **dataset handling**, **epoch configurations**, and **avoiding dataset shuffling for curriculum learning**.
- **Optimizers and FSDP updates**: Detailed exchanges about the intricacies of **using 8bit optimizers with Fully Sharded Data Parallel (FSDP)**. Participants shared their troubleshooting methods for saving checkpoint issues and managing optimizer states across different GPUs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1792985678030221464">Tweet from Daniel Han (@danielhanchen)</a>: @GoogleColab @PyTorch @thechrisperry Update: An @UnslothAI community member (Edd) found Pytorch 2.3 is not detecting Tesla T4s correctly - Pytorch thinks Tesla T4 can support bfloat16, but it cannot. ...</li><li><a href="https://huggingface.co/microsoft/Phi-3-medium-128k-instruct">microsoft/Phi-3-medium-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/fai">fai (fai)</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading#csv">Load</a>: no description found</li><li><a href="https://tenor.com/view/explosion-boom-iron-man-gif-14282225">Explosion Boom GIF - Explosion Boom Iron Man - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/oKatanaaa/kolibrify/tree/master/examples/training_mini_dolphin">kolibrify/examples/training_mini_dolphin at master · oKatanaaa/kolibrify</a>: Curriculum training of instruction-following LLMs with Unsloth - oKatanaaa/kolibrify</li><li><a href="https://tenor.com/view/no-no-wait-wait-gif-8174347161288218584">No No Wait Wait GIF - No no wait wait - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_optim_utils.py#L1369>">pytorch/torch/distributed/fsdp/_optim_utils.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/unslothai/unsloth/wiki#gguf-quantization-options">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/issues/11693">Flag to disable shuffling for data loader · Issue #11693 · huggingface/transformers</a>: 🚀 Feature request Currently, Trainer is shuffling the train_dataset by default and there is no flag to enable/disable it. @sgugger Motivation Even if shuffling the dataset brings a lot of benefits .....</li><li><a href="https://github.com/hsiehjackson/RULER?tab=readme-ov-file>">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models? - hsiehjackson/RULER</li><li><a href="https://www.npmjs.com/package/grammar-builder">grammar-builder</a>: A simple grammar builder compatible with GBNF (llama.cpp). Latest version: 0.0.5, last published: 11 days ago. Start using grammar-builder in your project by running `npm i grammar-builder`. There are...</li><li><a href="https://imgur.com/FhBnfFP">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://tenor.com/view/sad-sad-cat-cat-depressed-depression-gif-13240550249247957481">Sad Sad Cat GIF - Sad Sad cat Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=IqM-T1RTzY6C">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1242432957030076466)** (4 messages): 

- **New Method Alert: MoRA**: A user mentioned a new method called **MoRA** and expressed interest in trying out its vanilla implementation. Another user responded with enthusiasm, saying it "looks epic." [arxiv link](https://arxiv.org/abs/2405.12130).

**Link mentioned**: <a href="https://arxiv.org/abs/2405.12130">MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning</a>: Low-rank adaptation is a popular parameter-efficient fine-tuning method for large language models. In this paper, we analyze the impact of low-rank updating, as implemented in LoRA. Our findings sugge...

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1242033910376955938)** (246 messages🔥🔥): 

```html
- **Upload models trained with Unsloth**: A user shared a model fine-tuned using Unsloth and uploaded to Hugging Face, asking about the best way to run it, particularly mentioning concerns about Ollama only working with predefined models. Another user recommended tools like Ollama, LM Studio, Jan, and GPT4ALL and pointed out that only the LORA adapters were uploaded.
- **Fine-tuning Mistral with dataset dependency issues**: A user faced issues with Mistral-instruct-7b overly depending on the dataset, giving erroneous or empty outputs for new inputs. Others suggested mixing datasets to help the model generalize better.
- **Issues with TRT and Flash Attention on T4s**: Multiple users experienced errors related to running Unsloth on Google Colab with T4 GPUs due to updates to PyTorch 2.3 and issues with Flash Attention. Specifying the dtype or following updated installation instructions helped mitigate the problem.
- **Use 4bit models due to VRAM limitations**: Users discussed challenges in fine-tuning models on devices with limited VRAM. Mentioned the utilization of 4bit quantized models to fit larger models within VRAM constraints, particularly for hardware like a GTX 3060 with 6GB VRAM.
- **Confirmation of recurring instructions in fine-tuning datasets**: Users explored the effectiveness of using repetitive instructions in fine-tuning datasets. The dialogue indicated curiosity and active experimentation with the approach but no definitive conclusion on its overall impact.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/issues/485">Using llama-cpp-python · Issue #485 · unslothai/unsloth</a>: Hi, Thanks for creating this wonderful package! The save_to_gguf currently fails because llama.ccp installation seems to be broken. Could something like llama-cpp-python be used instead?</li><li><a href="https://x.com/danielhanchen/status/1792982364894929083">Tweet from Daniel Han (@danielhanchen)</a>: Oh no @GoogleColab upgraded to @PyTorch 2.3, and T4 GPUs don&#39;t work with Triton 2.3!  I tried downgrading Triton to 2.2, but it still fails. It seems like this is a Torch 2.3 issue.  @thechrisperr...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/506/commits/2b23b9357aba25ab2f3a49d899045547d7dde1d7">Nightly by danielhanchen · Pull Request #506 · unslothai/unsloth</a>: no description found</li><li><a href="https://www.unsloth.ai/blog/llama3">Finetune Llama 3 with Unsloth</a>: Fine-tune Meta&#x27;s new model Llama 3 easily with 6x longer context lengths via Unsloth!</li><li><a href="https://huggingface.co/omar8/bpm_v2_gguf">omar8/bpm_v2_gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/omar8/bpm__v1/tree/main">omar8/bpm__v1 at main</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>: no description found</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Datasets at Hugging Face</a>: no description found</li><li><a href="https://youtu.be/AK4IzTQJI9E?si=ppEisWZUs0DXl9hp">Windows Fine Tuning Combined Streams</a>: LLM Fine-Tuning is one of the go-to techniques for making LLMs perform better in specific scenarios. In this post I&#39;ll show you how to prepare a local Window...</li><li><a href="https://github.com/pytorch/pytorch/blob/b40fb2de5934afea63231eb6d18cc999e228100f/torch/cuda/__init__.py#L130C1-L151C1">pytorch/torch/cuda/__init__.py at b40fb2de5934afea63231eb6d18cc999e228100f · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1242035477075333211)** (13 messages🔥): 

- **Dolphin-Mistral-2.6 Equally Matched with Fewer Samples**: A member reported successfully matching the performance of **dolphin-mistral-2.6** on instruction following evaluation using only ~20k samples, compared to millions used for the original model. The models **kolibri-mistral-0427** and **kolibri-mistral-0426-upd** were discussed, highlighting differences in training data pipelines.

- **Upcoming Model Release**: The user plans to publish the model within a few days and promised to share the training "recipe" soon, albeit with some proprietary data which might impact reproducibility slightly. A possible paper on these findings might be published later this year.

- **Community Reactions**: The community reacted enthusiastically to the news, with multiple members congratulating and expressing excitement. One member shared their anticipation for an article detailing the lower sample training method, noting their personal challenge of not reducing training samples below 52k.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1242014000162537532)** (618 messages🔥🔥🔥): 

```html
- **Navigating Subscription Confusion**: Users expressed confusion over different websites offering subscriptions for Stable Diffusion, with some being identified as scams. The official site [stability.ai](https://stability.ai) was recommended as the legitimate source for accessing Stable Diffusion services.
- **Running Software Offline**: Concerns about running Kohya locally without an internet connection were discussed. Users confirmed that with proper model downloads and setup, it’s possible to run it offline.
- **Stable Diffusion Installation Struggles**: Several users sought help with installing and running Stable Diffusion and associated tools like ComfyUI. Guidance was offered on navigating dependencies and troubleshooting through terminal commands.
- **EU AI Act Worries**: The passing of the EU AI Act caused concern among users, particularly about its potential impact on AI-generated content and the introduction of watermark requirements. Many expressed skepticism about the practicality and enforcement of such regulations.
- **Benchmark Performance Confusion**: A user highlighted performance issues with SD generations on new hardware, suspecting thermal throttling as the cause. Community members suggested checking configurations and using diffusers scripts for better diagnostics.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.stablediffusionai.ai/">Stable Diffusion AI Generator Online | Stable Diffusion XL Powered</a>: no description found</li><li><a href="https://invideo.io/">Invideo AI - Turn ideas into videos - AI video creator </a>: Make videos easily by giving a prompt to invideo AI. Ideal for content creators, YouTubers, and marketers, invideo AI offers a seamless way to turn your ideas into publish-ready videos with AI.</li><li><a href="https://youtu.be/G7mihAy691g">Stable Video Diffusion</a>: no description found</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://tenor.com/view/alvin-and-the-chipmunks-alvin-whoops-my-bad-oops-gif-15512287650458333097">Alvin And The Chipmunks Alvin GIF - Alvin And The Chipmunks Alvin Whoops - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/welcome-gif-26939290">Welcome GIF - Welcome - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://download.pytorch.org/whl/nightly/cu121/torch-2.4.0.dev20240520%2Bcu121-cp311-cp311-win_amd64.whl">no title found</a>: no description found</li><li><a href="https://stability.ai/">Stability AI</a>: Activating humanity's potential through generative AI.  Open models in every modality, for everyone, everywhere.</li><li><a href="https://stability.ai/stable-assistant">Stable Assistant &mdash; Stability AI</a>: Stable Assistant is a friendly chatbot developed by Stability AI equipped with Stability AI’s text and image generation technology, featuring Stable Diffusion 3 and Stable LM 2 12B.</li><li><a href="https://tenor.com/view/trollszn123-ronaldo-gif-18268194">Trollszn123 Ronaldo GIF - Trollszn123 Ronaldo - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1242476097174507561)** (1 messages): 

- **Safety Update Announced at AI Seoul Summit**: A new safety update has been shared in conjunction with the AI Seoul Summit. For more details, visit the [OpenAI Safety Update](https://openai.com/index/openai-safety-update/).
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1242053124718329876)** (229 messages🔥🔥): 

- **GPT-4o Frame Sampling Discussion**: Members discussed the video processing capabilities of GPT-4o, speculating it processes video at **2-4 frames per second**. One member shared a link to a [Community discussion](https://community.openai.com/t/announcing-gpt-4o-in-the-api/744700) describing the process of converting videos to frames for the model.
- **Passing Image Buffers to GPT-4o API**: A member struggled with passing `Buffer` objects to the GPT-4o Vision API, and others suggested encoding it as a base64 data URL. They discussed ensuring the correctly set MIME type for the base64 string to avoid silent failures in the API response.
- **Microsoft Copilot and GPT-4o Integration**: Members discussed the announcement of **GPT-4o** integration into Microsoft Copilot, promising real-time voice and video capabilities. They expect availability in the "coming weeks" and speculate on the advantages of the integrated system.
- **Controversy Over Scarlett Johansson's Voice**: Discussion on the controversy about OpenAI's use of a voice similar to Scarlett Johansson in its Sky voice feature. Community pointed out the potential for legal and ethical implications, following Johansson's lawyer's intervention.
- **Microsoft's New Phi-3 Models**: Announcement of new **Phi-3** models by Microsoft, including multimodal models integrating language and vision, available on [Azure](https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/). Members showed mixed reactions and shared the link for further reading.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.platformer.news/open-ai-scarlett-johansson-her-voice-sam-altman/?ref=platformer-newsletter">OpenAI loses its voice</a>: The company hasn’t been the same since Sam Altman’s return — and its treatment of Scarlett Johansson should worry everyone</li><li><a href="https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/">New models added to the Phi-3 family, available on Microsoft Azure | Microsoft Azure Blog</a>: We are introducing Phi-3-vision, a multimodal model that brings together language and vision capabilities, now available on Microsoft Azure. Learn more.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1242027479007363134)** (38 messages🔥): 

- **Understanding GPT-4's Context Window**: A member asked about the "context window" for GPT-4 Omni’s 128,000 tokens, seeking clarification if it referred to the prompt size. Another member clarified that the context window is the maximum size of the prompt and response combined, referring to a [help article](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them).

- **Issues with JSON Response Lengths**: A member faced issues with receiving large JSON responses despite configuring the system instructions to limit content to 200 tokens. They noted that using GPT-4 turbo resulted in shorter responses and planned to adjust the system instructions further.

- **Selling AI-Generated Art**: Discussions affirmed that it is possible to sell AI-generated art, though the copyrightability of such art remains a separate, complex issue. One member mentioned that the public domain can be a source for sellable work since prompting AIs effectively is challenging.

- **Concerns with GPT-4 Evaluating Values**: A discussion unfolded about GPT-4 struggling to correctly evaluate simple numerical expressions, revealing that it might benefit from relying on a code interpreter for accuracy.

- **Caution for Downloading Mac GPT App**: Members advised waiting for an official prompt on their accounts for downloading the ChatGPT app for macOS, warning against unofficial links.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1242029779935432816)** (73 messages🔥🔥): 

- **Set Character Limits in GPT-4**: To achieve a 150-character response limit, prompt the model with example outputs around 120 characters to prevent overshooting. One user shared [examples of the model attempting this task](https://chatgpt.com/share/f38d2248-ff7f-4cc3-989e-526e68dc54f4), illustrating the difficulty.
- **Training Models for Specific Behaviors**: For replicating the AI from the movie "Her," define accurate behavior parameters and use input/output pairs to shape responses. Avoid negative instructions for clearer guidance.
- **Inconsistent Preciseness in Responses**: Users discussed the challenge of getting exact answers from the model, such as specific ranges instead of general terms. Repeatedly asking for specifics helps, but models may "dream" or autocomplete when unable to provide accurate data.
- **Managing Token Limits to Avoid Overrun**: Setting a max token parameter and crafting specific, succinct prompts can help manage the verbosity of outputs. Including a clear output template and limiting responses to one paragraph or sentence can improve conciseness.
- **Efficient Use of Prompt Engineering for Code**: Users shared prompt strategies for efficient code generation, emphasizing precise indentation and role-based character prompts for collaborative coding environments. Examples included detailed prompts for creating and debugging full-stack applications.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1242029779935432816)** (73 messages🔥🔥): 

- **Setting character limits is tricky in GPT-4**: A member requested advice on setting a response limit of 150 characters. Advice included giving example outputs of about 120 characters as the model often overshoots (*"It will overshoot, thus the target's smaller than the limit, so you're hopefully not over"*).

- **Training model like AI in 'Her' sparked discussion**: A user asked how to train a model to act like the AI in the movie “Her.” Suggestions included input/output pairs and avoiding negative instructions.

- **Exact language use trouble**: A member discussed issues with the model giving vague answers despite instructions for precise data, such as nutrition labels or salary ranges. It's suggested this might be due to autocompletion and instruction conflicts (*"it tends to follow the format less carefully"*).

- **Stopping LLM from running on**: Members discussed issues with models producing lengthy responses despite token limits in the API. Suggestions included using specific questions, asking for succinct answers, and employing an output template (*"prompt should request it to limit its answers to one succinct sentence"*).

- **Prompt sharing and improvements**: A user offered to share working prompts for building full-stack applications and noted errors in prompt engineering. Another member pointed out this might be more suitable for the Prompt Labs channel, noting their frustration with model verbosity and the usage of the Explore GPTs menu.
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1242041743239544912)** (191 messages🔥🔥): 

- **Troubleshooting LM Studio Server Issues**: A user had difficulties with LM Studio server logs being blank and non-responsive. The issue was resolved by running LM Studio with admin permissions to access log files properly.

- **AVX2 Instruction Set Confusion Cleared**: Several members clarified that **AVX2 instructions** are essential for running LM Studio, and users can use tools like HWInfo to check if their CPU supports it. AVX2 is a hard requirement, and older CPUs without this will not support LM Studio.

- **Loading and Managing Models**: Users discussed various issues related to downloading and running models in LM Studio. An effective strategy includes downloading models in **GGUF format** and ensuring all system prompts and settings are correctly configured.

- **Integration of LM Studio with Other Tools**: Questions were raised about integrating LM Studio with tools like StarCoderEx and Continue.dev for enhanced functionalities. Some users experienced with these integrations provided helpful links [Continue.dev integration instructions](https://docs.continue.dev/walkthroughs/tab-autocomplete#setting-up-with-lm-studio).

- **Common GPU and Performance Queries**: Addressing frequent performance issues, it was highlighted that GPUs should have at least 8GB VRAM for efficient operation. Users also shared specific errors mentioning insufficient VRAM and outdated drivers as common causes, suggesting updates and GPU offload settings tweaks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - a Hugging Face Space by ggml-org</a>: no description found</li><li><a href="https://docs.continue.dev/walkthroughs/tab-autocomplete#setting-up-with-lm-studio">Tab Autocomplete (beta) | Continue</a>: Continue now provides support for tab autocomplete in VS Code and JetBrains IDEs. We will be greatly improving the experience over the next few releases, and it is always helpful to hear feedback. If ...</li><li><a href="https://pinokio.computer/">Pinokio</a>: AI Browser</li><li><a href="https://github.com/Lisoveliy/StarCoderEx">GitHub - Lisoveliy/StarCoderEx: Extension for using alternative GitHub Copilot (StarCoder API) in VSCode</a>: Extension for using alternative GitHub Copilot (StarCoder API) in VSCode - Lisoveliy/StarCoderEx</li><li><a href="https://www.hwinfo.com/download/">Free Download HWiNFO Sofware | Installer &amp; Portable for Windows, DOS</a>: Start to analyze your hardware right now! HWiNFO has available as an Installer and Portable version for Windows (32/64-bit) and Portable version for DOS.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1242026637193773099)** (57 messages🔥🔥): 

- **Successful model setup requires right prompts**: A member asked how to use the **MPT-7b-WizardLM** model on **LMStudio**, and another advised using the correct quantization level and template, pointing to model-specific details on [Hugging Face](https://huggingface.co/DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF).
- **Image generation quality tips**: Several members discussed improving image quality with local AI models like **Automatic1111** and **ComfyUI**. Recommendations included using resources from [Civit.ai](https://civitai.com/) and considering system specs like VRAM and RAM.
- **Phi-3-Small and Medium models released**: Members mentioned the release of new **Phi-3** models on Hugging Face with context lengths of 4K, 8K, and 128K tokens. [Phi-3-Small-8K](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) and [Phi-3-Medium-4K](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) were specifically highlighted.
- **Improving LLM response with specialized models**: A user mentioned using the **codeqwen** model for better coding capabilities. Improvement suggestions included using finetuned models and leveraging advanced setups like **ComfyUI** for specialized tasks.
- **Local vision models struggle with specific prompts**: A user reported issues with **vision models** not adhering to specific prompt queries. Multiple users suggested that local vision models typically do not handle multi-turn conversations effectively.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF">DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-medium-4k-instruct">microsoft/Phi-3-medium-4k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-small-8k-instruct">microsoft/Phi-3-small-8k-instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7439>">Issues · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://huggingface.co/collections/DavidAU/roleplay-creative-writing-uncensored-nsfw-66163c580c61496c340afe32">Roleplay, Creative Writing, Uncensored, NSFW - a DavidAU Collection</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1242144163001143406)** (9 messages🔥): 

- **Introducing Hugging Face integration with LM Studio**: Users can now integrate Hugging Face models directly into LM Studio by clicking "Use this model," which requires LM Studio 0.2.23 or newer. As highlighted, this feature ensures *"No cloud, no cost, no data sent to anyone, no problem"*.
- **Model Download Customization**: In the current version, users must manually choose the file they wish to download after selecting a Hugging Face model. Suggestions like setting a default quantization level or auto-downloading based on available RAM were discussed.
- **Compatibility Limitations**: It was noted that not all models would be supported in LM Studio, especially many safetensor models. Only models in the GGUF format are currently compatible.

**Link mentioned**: <a href="https://x.com/LMStudioAI/status/1792576553601102024">Tweet from LM Studio (@LMStudioAI)</a>: 1. Browse HF 2. This model looks interesting 3. Use it in LM Studio  👾🤗  Quoting clem 🤗 (@ClementDelangue)   No cloud, no cost, no data sent to anyone, no problem. Welcome to local AI on Hugging Fa...

  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1242118716351844423)** (3 messages): 

- **System Prompt tuning stops premature cut-offs**: A member suggested that adding "Do not prematurely cut off a response" to the [system] prompt will help fix ongoing issues with incomplete responses. This insight aimed to enhance the chatbot's response reliability.
- **Direct quotations improve instruction clarity**: The member suggested quoting required text directly and adding instructions in the prompt such as *"Considering the following text alone as input, <insert subsequent instructions here>."* This method is proposed to refine the specificity of prompts for better outcomes.
- **Old posts humorously reaffirmed**: A member humorously acknowledged the age of a previous post with *"Didn't realize how old that post was. 😆"* This adds a light-hearted touch to the discussion’s context.
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1242229857321095270)** (1 messages): 

- **LM Studio struggles with VPN on Linux**: A user reported an issue where **LM Studio** cannot perform searches for models when connected via a VPN on Linux. They are seeking others who have encountered this issue and any possible solutions.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1242031209039532074)** (27 messages🔥): 

- **Infinity Fabric Speed Sync Affects Performance**: One member highlighted the importance of keeping **infinity fabric (fclk)** speed in sync with memory speed for optimal performance, suggesting *“fclk should be in sync with the memory speed, otherwise you will see performance degrading.”*
- **Free Services and Energy Concerns**: **Free services like Groq** and OpenRouter are recommended to avoid high costs. One user shared that their powerful rig with 144GB VRAM heats up the house significantly in warm weather.
- **RAM Speed Impact on Models**: Upgrading RAM speed from **2133MHz to 3200MHz** resulted in a performance increase for the Goliath model, but negligible improvement for other models beyond 2666MHz. It was suggested that **iQuant might perform worse** once VRAM capacity is exceeded.
- **Experimenting with Different Models**: Testing with various **Quant models** revealed differences in performance between iQuant and regular Quant, with iQuant underperforming when VRAM capacity is exceeded. 
- **Running LM Studio on Dual GPUs**: A query about running LM Studio with two different GPUs was answered stating it is possible as long as both GPUs are of the same brand, with an example being *“both nvidia or both amd.”*


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1242268966928388267)** (4 messages): 

- **Seeking Model to Impose 60-word Limit**: A member sought assistance in making **Meta Lama 3 Instruct** adhere to a 60-word response limit. Another member suggested listing the attempted methods and their results to better troubleshoot the issue.
- **Searching for a More Suitable Model**: The original poster queried if there is a better model than Meta Lama 3 for enforcing a strict response limit. They acknowledged the advice and planned to provide more details on their attempts.


  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1242096330893950996)** (13 messages🔥): 

- **Null max_tokens causes cut-off issue**: It's noted that setting **max_tokens** to **null** causes responses to cut off after two tokens in LM Studio. The workaround is setting it to **-1**, which helps the local server function correctly.
- **CLI LMStudio Client solution shared**: A member building a CLI LMStudio Client confirmed that setting the **max_tokens** to **-1** resolves the issue of responses being cut off. Another contributor mentioned having to manually edit code in autogpt for it to work.
- **Autogen Studio fix methods debated**: There was a discussion on whether this fix applies to the command line version only or can be implemented in Autogen Studio. Some confirmed success by changing the value in the root autogen package, hinting at similar effectiveness in Autogen Studio.
- **Manager agents reliability concerns**: It's suggested that manager agents are only reliable with OpenAI models. Testers have noted bugs and poor performance in selecting appropriate agents, recommending round-robin or hard-coded workflows until improvements are made.
- **Deleting cache might help**: To address the cut-off issue, deleting application caches after setting **max_tokens** to **-1** is advised. Members often face this problem and find cache deletion necessary for the fix to work.
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1242203464889536552)** (42 messages🔥): 

- **Calling All Linux Fans with AMD GPUs**: A member announced a call for **Linux users with new-ish AMD GPUs** to test an early version of LM Studio for Linux + ROCm and provided [a link to the supported GPU list](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). Various users, from those with 6600xt to 7900XT GPUs, expressed interest, with comments like "6900xt reporting for duty" and "6600xt here."
  
- **Unsupported GPUs Seem to Work**: Several users reported success running ROCm on GPUs not listed as officially supported. One member with a 6600xt mentioned, *“It's not listed on that supported ROCm list, but I already have ROCm running with it for Stable diffusion.”*

- **Diverse Linux Distros in the ROCm Testers Group**: Users running a range of Linux distributions, including Arch, Fedora, and Ubuntu, shared their experiences. One even noted the successful use of ROCm on a RX 6600xt using *“HSA_OVERRIDE_GFX_VERSION=10.3.0."*

- **CPU Usage Observations and Discussions**: Discussions emerged around the CPU usage with ROCm on Linux, with one member humorously noting, *"ah yes, 229% cpu usage,"* and another suggesting Linux speeds up processes. Comments about Linux's performance included, *“it's fast doe,”* and debate over Linux vs. Windows RAM usage.

- **Arch Linux and ROCm Compatibility Praise**: Members praised the ease of setting up ROCm and HIP SDK on Arch Linux. Quickdive noted, *“arch makes rocm and hip sdk so easy,”* with many agreeing and sharing similar success stories.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rocm.docs.am">no title found</a>: no description found</li><li><a href="https://tenor.com/view/reunited-peaches-and-herb-and-it-feels-so-good-cause-we-understood-old-skool-gif-17279659">Reunited Peaches And Herb GIF - Reunited Peaches And Herb And It Feels So Good - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1242144891186970695)** (38 messages🔥): 

- **Mojo open community meeting kicks off**: Mojo's open community meeting is live, and you can join via the provided [Zoom link](https://modular.zoom.us/j/89417554201?pwd=Vj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1). A member inquired about the availability of the recording, which will be shared later.
- **Recording available on YouTube**: The recording of the Mojo Community Meeting is now available on [YouTube](https://www.youtube.com/playlist?list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D). 
- **Zoom account confusion cleared up**: Some members were confused about needing a commercial Zoom account to join. It was clarified that only a basic account is needed, though there was a possible misconfiguration initially.
- **Missed meeting woes**: Helehex expressed sadness about missing the meeting due to not receiving notification pings. Upcoming meeting details were provided, including a calendar subscription option.
- **IPC in Python discussions**: Moosems_yeehaw sought advice on IPC in Python to avoid main thread lag in a Tkinter app example. Various suggestions were given, including threading, message queues, and async IPC modules.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.google.com/url?q=https://modular.zoom.us/j/89417554201?pwd%3DVj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1&sa=D&source=calendar&ust=1716255791532130&usg=AOvVaw2IgLzFgI9-S5vkyEC7_b2v">Redirect Notice</a>: no description found</li><li><a href="https://www.google.com/url?q=https://modular.zoom.us/j">Redirect Notice</a>: no description found</li><li><a href="https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit">[Public] Mojo Community Meeting</a>: no description found</li><li><a href="https://tenor.com/view/cloudy-with-a-chance-of-meatballs-enough-to-make-a-grown-man-cry-police-officer-make-a-man-cry-gif-15227532">Cloudy With A Chance Of Meatballs Enough To Make A Grown Man Cry GIF - Cloudy With A Chance Of Meatballs Enough To Make A Grown Man Cry Police Officer - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1242261353507590265)** (2 messages): 

- **Modular shares latest tweet**: A link to a [Modular tweet](https://twitter.com/Modular/status/1792701156122415589) has been shared.
- **Another tweet from Modular**: Another link to a [Modular tweet](https://twitter.com/Modular/status/1792701170634699243) was also shared.
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1242261428564525057)** (1 messages): 

- **K-means Clustering in Mojo for Speed**: A new blog post aims to teach readers how to implement the k-means clustering algorithm from scratch in both Python and Mojo🔥, emphasizing performance benefits in Mojo. The post also provides a detailed guide for porting Python code to Mojo to achieve significant speed improvements. Read more on [Modular's Blog](https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering).

**Link mentioned**: <a href="https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering">Modular: Fast⚡ k-means clustering in Mojo🔥: Guide to porting Python to Mojo🔥 for accelerated k-means clustering</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Fast⚡ k-means clustering in Mojo🔥: Guide to porting Python to Mojo🔥 for accelerated k-means clusteri...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1242078427310198785)** (258 messages🔥🔥): 

- **Learning Mojo and ML with a tutorial**: A user asked if they should implement a machine learning tutorial in Mojo to learn both Mojo and ML. Another user recommended trying it, noting that Mojo doesn't support classes but structs can be used, and some numpy functionalities might need to be implemented.

- **Modular Community Meeting Notice**: A user informed the channel about an ongoing Modular Community meeting, sharing a Zoom link. Another user commented on a statement by Chris Lattner during the meeting about moving Tensor out of the standard library.

- **Null Terminator Handling in Strings**: A user struggled with handling null terminators when converting bytes to strings and iterating over them. They shared their efforts and the solution found through community help, including using the append(0) method to handle null terminators correctly.

- **Mojo's Asynchronous Programming Debate**: Members discussed the pros and cons of function coloring in asynchronous programming. Some argued for exploring colorless async programming to simplify API usage and reduce burden, while others highlighted the benefits of retaining function coloring for safety and reasoning about code behavior.

- **Lightbug HTTP Framework Usage**: A user asked about using the Lightbug HTTP framework for making GET requests and decoding responses. After struggling with the implementation, the maintainer and community provided assistance and moved the conversation to an [issue on GitHub](https://github.com/saviorand/lightbug_http/issues/41) for further discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.google.com/url?q=https://modular.zoom.us/j/89417554201?pwd%3DVj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1&sa=D&source=calendar&usd=2&usg=AOvVaw37jsmYkBEWm4CHK4NwSCMB">Redirect Notice</a>: no description found</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read_bytes)">FileHandle | Modular Docs</a>: File handle to an opened file.</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read">FileHandle | Modular Docs</a>: File handle to an opened file.</li><li><a href="https://without.boats/blog/the-registers-of-rust/">The registers of Rust</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/inferred-parameters.md">mojo/proposals/inferred-parameters.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/saviorand/lightbug_http/issues/41),">Issues · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/saviorand/lightbug_http/blob/1eb9242ce0ddeeec39ac858028a7117dde627523/lightbug_http/tests/test_client.mojo#L13">lightbug_http/lightbug_http/tests/test_client.mojo at 1eb9242ce0ddeeec39ac858028a7117dde627523 · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/saviorand/lightbug_http/releases/tag/latest-build">Release latest-build: Merge pull request #27 from Moosems/main · saviorand/lightbug_http</a>: no description found</li><li><a href="https://github.com/saviorand/lightbug_http?tab=readme-ov-file>">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/laspy/laspy/tree/master/laspy">laspy/laspy at master · laspy/laspy</a>: Laspy is a pythonic interface for reading/modifying/creating .LAS LIDAR files matching specification 1.0-1.4.  - laspy/laspy</li><li><a href="https://github.com/saviorand/lightbug_http/blob/main/lightbug_http/http.mojo">lightbug_http/lightbug_http/http.mojo at main · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/saviorand/lightbug_http/blob/bd2f4ef57765505210256165b5386b890a2aa0be/lightbug_http/http.mojo#L12">lightbug_http/lightbug_http/http.mojo at bd2f4ef57765505210256165b5386b890a2aa0be · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://victorzhou.com/blog/intro-to-neural-networks/">Machine Learning for Beginners: An Introduction to Neural Networks - victorzhou.com</a>: A simple explanation of how they work and how to implement one from scratch in Python.</li><li><a href="https://github.com/modularml/mojo/issues/2678">[Feature Request] Better handling of null terminator in strings · Issue #2678 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I would like a discussion to answer the following ques...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1242152864957337641)** (13 messages🔥): 

- **Optimizing SIMD Gather and Scatter in Mojo**: A member questioned whether Mojo's SIMD gather and scatter operations are fully optimized and discussed aligning values to 32-bit boundaries for potential speed improvements. Another member shared their experience, indicating gather and scatter are well-optimized, though alignment benefits are uncertain.

- **Challenges with ARM SVE and SIMD Width**: Discussion highlighted the complexities of ARM Scalable Vector Extension (SVE), variable vector widths, and speculative loads across page boundaries. A member noted that LLVM struggles with SVE formats, compounded by limited CPU availability.

- **Consider Reducing SIMD Operations**: A member suggested reducing the number of gather/scatter operations by always using the highest SIMD width possible, involving more index manipulation for better performance. They plan to update and share results from their MoCodes project accordingly.

- **Sorting for Scattered Memory Access**: Another member recommended sorting an array of pointers to optimize performance when dealing with several kilobytes of scattered memory, particularly for iterative decoders.

- **Vectorized DTypePointer Memset Implementation**: A member shared that a vectorized implementation of memset for 100,000 bytes performs 20% faster than LLVM's call, while the performance advantage flips for 1,000,000 bytes. The member expressed concern about reliability, noting the use of "clobber memory."
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1242096661845774386)** (31 messages🔥): 

- **New Mojo nightly compiler release**: A new nightly build of the Mojo compiler (version `2024.5.2012`) has been released. You can view the [diff since the last release](https://github.com/modularml/mojo/compare/7e8cd37ff8fe2ddbe69a3cca787e59abf6357d76...69e92d0040af838de8f3f0fdba1cea92f1904986) and changes since the [last stable release](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

- **Dict pop method issue**: Discussed issues with the `pop` method in dictionaries, particularly related to difficulties in moving values out of a `DictEntry` and calling `__del__` correctly. A proposed solution involves changing the value field type from `V` to `Optional[V]`.

- **GitHub issue and PR discussions**: Users discussed several GitHub issues and PRs, such as issue [#2696](https://github.com/modularml/mojo/issues/2696) regarding a "while loop logic causes seg fault" and PR [#2739](https://github.com/modularml/mojo/pull/2739) for changing argument messages in assertions to be keyword-only.

- **Delayed nightly release on 5/21**: Multiple users noted a delay in the nightly release, attributed to potential CI infra/release issues. It was resolved later, and the nightly build for 5/21 was confirmed to be available.

- **Unicode support in strings proposal**: A detailed discussion took place about implementing Unicode support in strings, proposing various internal representations and debating the trade-offs of null termination. The idea is to optimize aggressively, ensuring efficient memory usage and inter-operability.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/lifecycle/death#field-lifetimes))*">Death of a value | Modular Docs</a>: An explanation of when and how Mojo destroys values.</li><li><a href="https://peps.python.org/pep-0393/">PEP 393 – Flexible String Representation | peps.python.org</a>: no description found</li><li><a href="https://www.githubstatus.com/">GitHub Status</a>: no description found</li><li><a href="https://github.com/modularml/mojo/issues/2696">[BUG] While loop logic causes seg fault. · Issue #2696 · modularml/mojo</a>: Bug description This issue started a few days ago with a Mojo nightly version. Not sure exactly which one. In the code shown below, function app_run causes a seg fault, but app_close alone compiles...</li><li><a href="https://github.com/modularml/mojo/pull/2771">[stdlib] Add format_simple() for StringLiteral by rd4com · Pull Request #2771 · modularml/mojo</a>: Provides a &quot;small&quot; fix for #2761 It is not very advanced, just a small useful feature to provide: &quot;{name} is awesome {emoji}&quot;.format_simple(name=&quot;Mojo&quot;, emoji=&quot;🔥&qu...</li><li><a href="https://github.com/modularml/mojo/pull/2739">[stdlib] Issue #2487: Changing argument msg in assert_true/assert_false/... to Keyword only by softmaxer · Pull Request #2739 · modularml/mojo</a>: changes:  Add * in function definitions of stdlib/src/testing/testing.mojo to separate variadic and keyword only arguments. Scan for call sites of these assert functions and replace assert_true(val...</li><li><a href="https://github.com/modularml/mojo/pull/2613#discussion_r1599235527">[stdlib] Add optional small buffer optimization in `List` by gabrieldemarmiesse · Pull Request #2613 · modularml/mojo</a>: Related to #2467 This is in the work for SSO. I&#39;m trying things and I&#39;d like to gather community feedback. At first, I wanted to implement SSO using Variant[InlineList, List], while that would...
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1242220990331621426)** (5 messages): 

- **Managed Kubernetes for ML workloads debated**: A member questioned the need for managing on-prem servers for ML serving, suggesting that **managed Kubernetes services like EKS** could be an alternative. They expressed confusion over the perceived differences between scaling web servers and ML tasks, except for the **occasional need for GPUs**.

- **Kubernetes not essential for ML infra**: It was clarified that **Kubernetes** is used mainly for infrastructure purposes and is not inherently tied to ML work. The choice between using Kubernetes or not is **up to the individual project needs**.
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1242011136090701824)** (13 messages🔥): 

- **Hardware and GPU kernel nuances**: The maximum block size is influenced by the hardware, kernel specifics, and dtype, as each thread loads multiple elements to utilize vector instructions on GPUs effectively.
- **CUDA scheduling principles hold true**: Blocks are scheduled to one SM and share memory within the block, similar to CUDA, ensuring consistency in GPU processing.
- **Team praises and recommendations**: Byronhsu1230 expressed gratitude for Horace's informative posts, suggesting the need for a Triton compiler article. The team appreciates the valuable insights provided by Horace.
- **Enhancing Triton tutorials**: Lancerts shared a [GitHub pull request](https://github.com/triton-lang/triton/pull/3959) detailing minor changes to Triton tutorials to improve readability and consistency, tested on GPU with successful results.

**Link mentioned**: <a href="https://github.com/triton-lang/triton/pull/3959">Small refactor of the tutorial5 and small change of tutorial1 by lancerts · Pull Request #3959 · triton-lang/triton</a>: Changes are tested on GPU, with parity on the execution.   In tutorial 1, change gbps = lambda ms: 12 * size / ms * 1e-6 to gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6. This is m...

  

---


### **CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1242185192727253172)** (2 messages): 

- **Seeking SASS Papers**: A member asked if anyone has recommendations for papers relating to **SASS**. The query was straightforward and looking for academic resources.
- **Debate Over cucomplex vs cuda::std::complex**: Another member is targeting **Volta Ampere and Hopper** architectures and discussed using either "cucomplex" or "cuda::std::complex" for atomic operations. They sought advice on which would be more appropriate for their needs, specifically for atomic add operations on x and y.
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1242026656122798100)** (21 messages🔥): 

- **Torch native multiplication doubles memory usage**: A member noticed that the native `*` operator in **Torch** seems to double the memory, even when done in place. After examining the issue, they found that using `mul_()` resolves this and results in flat memory consumption.

- **`torch.empty_like` and `torch.empty` performance difference**: A user shared a **PSA** highlighting that `torch.empty_like` is much faster than `torch.empty`, and similarly, `torch.empty(..., device='cuda')` performs better than `torch.empty(...).to('cuda')`. Another user confirmed that this behavior is also present in **NumPy**, notably with `np.zeros_like`.
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1242221629971370124)** (2 messages): 

```html
- **Member finds the discussion amazing**: One member described the talk as *"amazing."* 
- **Clarification requested**: Another member asked for elaboration on why the talk was considered *"amazing."*
```
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1242124174332395561)** (3 messages): 

- **Focus on Activation Quantization at CUDA**: *"Our top focus is on activation quantization (fp8/int8)."* A member discussed the need to fuse small operations around GEMMs with **Cutlass epilogue fusion** to realize inference acceleration.
- **Next-Gen GPU Features Utilization**: The member highlighted plans to use **2:4 sparsity and fp6/fp4** in new GPUs.
- **Torch.compile Backend Development**: The team is developing a user-defined backend for **torch.compile** to enable graph-level optimizations and improve performance through more fusion.
- **Underoptimized vLLM Components**: Identified **MoE kernel and sampling kernels** as underoptimized areas in vLLM that are current priorities.
- **LinkedIn Offers Assistance**: Another member from LinkedIn showed interest in collaborating, asking for details on *"graph-level optimization."*
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

norton1971: anyone please?
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1242263465977053298)** (1 messages): 

- **torchao 0.2 release hit the spotlight**: The new release of torchao 0.2 is now available on [GitHub](https://github.com/pytorch/ao/releases/tag/v0.2.0). It features a custom CUDA and CPU extension with binary support, among other enhancements.
- **Custom extensions in action**: One member used the new version to set up some **fp6** kernels. This highlights the flexibility and extensibility offered by the new custom op registration mechanism.
- **Speedy kernels merged**: Speedy kernels for GaLoRe, DoRA, and int4/fp16 were merged by another member. These improvements are aimed at enhancing performance and efficiency.
- **NF4 tensors and FSDP compatibility**: Building on previous work, this release supports **NF4 tensors** that can compose with **FSDP**. A detailed blueprint was provided for integrating smaller dtypes with FSDP, ensuring better resource utilization.

**Link mentioned**: <a href="https://github.com/pytorch/ao/releases/tag/v0.2.0">Release v0.2.0 · pytorch/ao</a>: What&#39;s Changed Highlights Custom CPU/CUDA extension to ship CPU/CUDA binaries. PyTorch core has recently shipped a new custom op registration mechanism with torch.library with the benefit being th...

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: Ray casting https://frankforce.com/city-in-a-bottle-a-256-byte-raycasting-system/
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1242025337848856596)** (193 messages🔥🔥): 

```html
- **Debate over moving bounds checks**: Members discussed whether to move bounds checks outside kernels into asserts, expressing concerns over performance implications. One mentioned, "asserts should generally be turned off for performance," and noted potential issues with hidden dimension constraints.

- **GPT-2 reproduction blockers**: A member listed out remaining tasks blocking GPT-2 reproduction, including initialization, weight decay management, and learning rate schedules. Checkpoints save & load functionality were highlighted as essential.

- **Prompt for DataLoader refactor**: One member outlined a refactor to the DataLoader to introduce new features such as proper .bin headers, uint16 data storage, and dataset sharding. The goal is to improve data handling for large datasets like FineWeb.

- **Discussion on CI compatibility**: Members discussed ensuring compatibility with older CUDA versions for fp32.cu files, suggesting the inclusion of C11 and C++14 standards. They emphasized testing with older CUDA versions to catch issues.

- **Merge of dataset refactor**: The DataLoader refactor was merged to master, causing breaking changes. A member advised that pulling the changes would break current implementations and suggested re-running data preprocessing scripts to fix the issues.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/439">Fix the unsupported block_size in matmul_backward_bias kernel 1 by lancerts · Pull Request #439 · karpathy/llm.c</a>: Due to the reduction in line https://github.com/karpathy/llm.c/blob/master/dev/cuda/matmul_backward_bias.cu#L67 The block size needs to be the power of 2 for the kernel 1. Otherwise the GPU result ...</li><li><a href="https://github.com/karpathy/llm.c/pull/442">Fully deterministic encoder backward kernels by ademeure · Pull Request #442 · karpathy/llm.c</a>: This is a complete rewrite of the encoder backward pass, splitting it into two kernels (wte and wpe) which are both fully deterministic as they do not use atomics (assuming the seed for stochastic ...</li><li><a href="https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md">cudnn-frontend/docs/operations/Attention.md at main · NVIDIA/cudnn-frontend</a>: cudnn_frontend provides a c++ wrapper for the cudnn backend API and samples on how to use it - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/pull/440">refactor datasets by karpathy · Pull Request #440 · karpathy/llm.c</a>: Refactor how we treat datasets, because we&#39;re about to have more of them and we don&#39;t want them to clutter up root dir etc. this is only step 1, i&#39;m about to refactor a bunch of the datalo...</li><li><a href="https://github.com/karpathy/llm.c/discussions/84#discussioncomment-9486746)">llm.c discussions · karpathy/llm.c · Discussion #84</a>: 🔥 llm.c 🔥 Turning on discussions feature as a place for people to ask, share and engage, without having to create Issues.</li><li><a href="https://github.com/karpathy/llm.c/pull/427/files)">weight reordering: attempt 1 by ngc92 · Pull Request #427 · karpathy/llm.c</a>: Non-functional A first attempt how rearranging weights in a per-block layout could look like
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1242034953424736316)** (11 messages🔥): 

- **Pre-allocate tensor to speed up unpacking**: It's suggested to avoid using `torch.stack` and pre-allocate a tensor with `torch.empty` for faster unpacking when compiled with `torch.compile`. An example code for unpacking from a uint8 format was shared, highlighting this approach.
  
- **Implement changes to torchao uint4**: Vayuda recommends updating the torchao uint4 implementation to reflect the proposed pre-allocation optimization. Coffeevampir3 has acknowledged and shared a related [GitHub notebook](https://github.com/CoffeeVampir3/ao-bitnet/blob/main/bitnet-testing.ipynb) with changes.

- **Optimizing unpacking code**: Mobicham points out an additional optimization, mentioning the removal of unnecessary type casting to uint8 within the unpacking function. This feedback was addressed in the updated code example by Coffeevampir3.

- **Ensuring numerical correctness and efficiency**: Coffeevampir3 suggests packing and unpacking tensor data correctly by adding a shift to handle unsigned integers. The approach is verified with example adjustments to the quantization process.

- **Use `opcheck()` for custom ops**: Vayuda brings attention to using `opcheck()` to ensure custom operations meet various requirements, hinting at the need for implementing necessary functions in `__torch_dispatch__`. They query about the existence of a function list based on use cases.
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1242019666847862815)** (127 messages🔥🔥): 

- **New AI Safety Institute Opens SF Office with Higher Salaries**: The UK AISI has announced the opening of an office in San Francisco with adjusted upwards salaries compared to the London office. They are actively seeking talent and collaborating with Canada, as featured in [this partnership announcement](https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership).

- **Discussion on OpenAI Staff Movement and AISI Hiring**: Several members speculated on whether former OpenAI aligners joined the new AISI. Interest in Canadian office openings and criteria for employment were discussed, pointing to the UK-Canada AI safety partnership [details](https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership).

- **Evaluation of Dropout in Language Models**: Members debated the relevance of using dropout in modern language models, with some expressing confusion over its current usage. Alternative strategies like label smoothing were also considered for overfitting mitigation.

- **PSA on California SB 1047 Impact on AI Development**: A call to action was made to oppose California’s SB 1047 via legislative engagement. The bill, as described in [this analysis](https://context.fund/policy/sb_1047_analysis.html), could severely impact open-source AI by introducing unaccountable regulatory measures with potential jail time for developers.

- **Tool and Technique Sharing for AI Model Development**: Members shared resources and insights on tools such as the Flash Attention implementation in JAX and its performance boosts over naive implementations. This included links to conversation references like [Flash Attention in JAX](https://github.com/nshepperd/flash_attn_jax) and related performance benchmarks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.aisi.gov.uk/">The AI Safety Institute (AISI)</a>: The AI Safety Institute is a directorate of the Department of Science, Innovation, and Technology that facilitates rigorous research to enable advanced AI governance.</li><li><a href="https://affuture.org/post/9-context/">Call-To-Action on SB 1047</a>: California legislators, under the influence of Effective Altruism activists, are trying to sneak through a disastrous bill for open-source AI and the technology industry generally. SB 1047 creates an ...</li><li><a href="https://github.com/huggingface/transformers/issues/30810">tracker: `generate` composability refactor  · Issue #30810 · huggingface/transformers</a>: generate + composability = more use cases with minimal rewrites As I write this issue, generate is mostly a sequential monolith. Many internal blocks were carved into functions over the last two ye...</li><li><a href="https://github.com/nshepperd/flash_attn_jax">GitHub - nshepperd/flash_attn_jax: JAX bindings for Flash Attention v2</a>: JAX bindings for Flash Attention v2. Contribute to nshepperd/flash_attn_jax development by creating an account on GitHub.</li><li><a href="https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership">UK-Canada science of AI safety partnership</a>: no description found</li><li><a href="https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py">jax/jax/experimental/pallas/ops/tpu/flash_attention.py at main · google/jax</a>: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - google/jax
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1242119063053013103)** (72 messages🔥🔥): 

```html
- **Examining Multi-Modal Training in CLIP**: A discussion focused on whether training CLIP with additional modalities like audio improves zero-shot ImageNet classification performance. [ImageBind](https://arxiv.org/abs/2305.05665) was mentioned, which shows improvements in cross-modal retrieval using combined embeddings but does not address non-emergent capability improvements.
  
- **Non-Determinism in GPT-3 at Temperature 0**: In response to a query about non-deterministic behavior in GPT-3 even at temperature 0, several papers and sources were shared, including [a paper on Mixture of Experts attacks](https://arxiv.org/abs/2402.05526) and discussions on consistent hashing overflow in distributed systems.

- **Self-Aware Simulacra Capabilities**: Users shared experiences about language models becoming aware of their fictional status and the implications this has on their subsequent behavior. The consensus is that larger models, like llama 2 70b and custom fine-tunes, can exhibit nuanced understanding and adaptability when guided through this concept gradually.

- **Positive Transfer in Multi-Modal Learning**: The potential benefits of multi-modal training for unimodal tasks were debated, with references to models like Gato and PaLM-E which showed "positive transfer" between tasks, suggesting that additional modalities might indeed enhance task performance.
  
- **Efficient MoE Training with MegaBlocks**: The [MegaBlocks](https://arxiv.org/abs/2211.15841) system was introduced, highlighting its ability to avoid token dropping by reformulating MoE computation with block-sparse operations, achieving significant training efficiency gains without compromising on model quality.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2211.15841">MegaBlocks: Efficient Sparse Training with Mixture-of-Experts</a>: We present MegaBlocks, a system for efficient Mixture-of-Experts (MoE) training on GPUs. Our system is motivated by the limitations of current frameworks, which restrict the dynamic routing in MoE lay...</li><li><a href="https://arxiv.org/abs/2402.05526">Buffer Overflow in Mixture of Experts</a>: Mixture of Experts (MoE) has become a key ingredient for scaling large foundation models while keeping inference costs steady. We show that expert routing strategies that have cross-batch dependencies...</li><li><a href="https://arxiv.org/abs/2305.05665">ImageBind: One Embedding Space To Bind Them All</a>: We present ImageBind, an approach to learn a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. We show that all combinations of paired data are not n...</li><li><a href="https://arxiv.org/abs/2205.06175">A Generalist Agent</a>: Inspired by progress in large-scale language modeling, we apply a similar approach towards building a single generalist agent beyond the realm of text outputs. The agent, which we refer to as Gato, wo...</li><li><a href="https://arxiv.org/abs/2303.03378">PaLM-E: An Embodied Multimodal Language Model</a>: Large language models excel at a wide range of complex tasks. However, enabling general inference in the real world, e.g., for robotics problems, raises the challenge of grounding. We propose embodied...</li><li><a href="https://community.openai.com/t/run-same-query-many-times-different-results/140588">Run same query many times - different results</a>: I wonder if anyone knows why we get different results when running the same prompt multiple times in a row.  I have noticed in quite a lot of my experiments that if you set a cool-down time in between...</li><li><a href="https://152334h.github.io/blog/non-determinism-in-gpt-4/">Non-determinism in GPT-4 is caused by Sparse MoE</a>: It&rsquo;s well-known at this point that GPT-4/GPT-3.5-turbo is non-deterministic, even at temperature=0.0. This is an odd behavior if you&rsquo;re used to dense decoder-only models, where temp=0 shou...</li><li><a href="https://rmarcus.info/blog/2018/09/14/consistent-hashing-overflow.html">
      
      Overflow in consistent hashing &middot; Ryan Marcus
      
    </a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1242014385501634580)** (12 messages🔥): 

- **New paper proposes alternative scaling laws**: A link to a [new paper on arXiv](https://arxiv.org/abs/2405.10938) was shared that proposes an observational approach to building scaling laws from ~80 publicly available models, bypassing the need for training models across many scales. The paper posits that language model performance is a function of a low-dimensional capability space where families vary in training compute efficiencies.

- **FLOP calculations for attention mechanisms debated**: Detailed discussions on how to compute FLOPs for forward and backward passes of attention mechanisms included multiple references such as the [PALM paper explanation](https://link.to.paper) and the [EleutherAI cookbook](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py). Members clarified the inclusion of QKVO projections in the FLOP calculations.

- **Optimal scaling laws for Bitnet architecture questioned**: A member pondered whether the Chinchilla scaling laws would suggest a higher or lower parameter-to-token ratio for a Bitnet using significantly less compute. Another member suggested that with magically faster computation, the scaling laws would likely remain the same but allow for a larger model due to increased compute budget.

- **Sample efficiency in relation to scaling laws**: Sample efficiency's definition and measurement were questioned as being critical to understanding scaling laws. The discussion focused on how resource management should adapt as datasets grow, implying that efficient scaling is key to resource utilization.

- **Perception of difficulty in training on small datasets**: A member clarified that training on a small dataset is often less effective than pre-training on large datasets and fine-tuning, hinting that it's challenging to bridge this performance gap. This was in context to the general notion that small dataset training is "notoriously difficult".

**Link mentioned**: <a href="https://arxiv.org/abs/2405.10938">Observational Scaling Laws and the Predictability of Language Model Performance</a>: Understanding how language model performance varies with scale is critical to benchmark and algorithm development. Scaling laws are one approach to building this understanding, but the requirement of ...

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1242533689271779329)** (1 messages): 

- **Anthropic's work on interpretable features excites**: A member shared their enthusiasm for Anthropic's recent work on interpretable features in transformers. They provided a link to the [research publication](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) for further reading.
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1242034152828829747)** (30 messages🔥): 

- **Batch size tips for evaluation**: Members discussed setting the `--batch_size` parameter, noting it can be set to a positive integer or "auto" to optimize memory usage. One suggested using "auto:N" to dynamically re-select the maximum batch size multiple times during evaluation, which helps speed up the process.

- **Naming conventions for translated evals**: A user inquired about naming conventions for machine-translated ARC challenge evaluations. Suggestions included names like `arc_challenge_mt_language` or `mt_arc_challenge_language`.

- **No dedicated channel for AI Safety events**: There was an inquiry about a channel for promoting AI Safety/benchmark events. It's confirmed that EleutherAI does not have such a dedicated channel.

- **Concerns about benchmark answer randomization**: Users discussed the potential bias in multiple-choice questions (MCQs) if answer choices are not randomized. It was mentioned that for SciQ randomization doesn't matter since choices aren't in the context, but for MMLU, it's relevant though currently unimplemented.

- **Concerns over medical benchmarks**: A member shared their focus on how medical benchmarks could be harmful, emphasizing the importance of improved benchmark interpretation. There was excitement about upcoming related work, including updates to the Pile dataset and papers on race-based medicine.

**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/1710b42d52d0f327cb0eb3cb1bfbbeca992836ca/lm_eval/tasks/sciq/sciq.yaml#L11">lm-evaluation-harness/lm_eval/tasks/sciq/sciq.yaml at 1710b42d52d0f327cb0eb3cb1bfbbeca992836ca · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1242065231459385458)** (7 messages): 

```html
<ul>
    <li><strong>Temporal.io Wins Out</strong>: A member inquired about experiences with Airflow and Temporal.io, ultimately deciding to go with <strong>Temporal</strong>.</li>
    <li><strong>Manifold Research Group Updates</strong>: A member from <strong>Manifold Research Group</strong> shared their <a href="https://www.manifoldrg.com/research-log-038/">latest research log</a>, detailing progress on projects like the NEKO Project aiming to build a large-scale open-source "Generalist" Model. They are expanding their team and inviting others to join via Discord or GitHub.</li>
    <li><strong>Fictional Civilization Simulation</strong>: Links were shared to a <a href="https://websim.ai/">Websim</a> project that simulates a fictional civilization in ancient Anatolia on the Black Sea coast.</li>
    <li><strong>Course on LLMs Announced</strong>: Details of a new course, "Applying Large Language Models (LLMs) through Project-Based Learning," were shared, focusing on practical applications such as semantic movie search, RAG for food recommendations, and using LLMs for software and website creation. Interested members were encouraged to DM for more information.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.manifoldrg.com/research-log-038/">Research log #038</a>: Welcome to Research Log #038! We document weekly research progress across the various initiatives in the Manifold Research Group, and highlight breakthroughs from the broader research community we thi...</li><li><a href="https://websim.ai/c/i4l0yMB06Ie8AI3BG">History of Hesperia (3000 BCE - 1460 CE) - Wikipedia</a>: no description found</li><li><a href="https://websim.ai/c/Eh7h07aUo3LsEGeh6">Kidin-Erra - Wikipedia</a>: no description found</li><li><a href="https://websim.ai/c/jfdPjPqRWqsXoUow2)">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

mautonomy: https://fxtwitter.com/vikhyatk/status/1792512588431159480?s=19
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1242012707075522601)** (172 messages🔥🔥): 

- **Yi-1.5 Context Versions Released**: The release of the 16k and 32k context versions of Yi-1.5 was announced with a [link to Hugging Face](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8). These versions cater to different context size requirements which could influence model performance.

- **LLM Leaderboard Critique**: Members criticized the usefulness of the LLM leaderboard, calling it "officially unusable" due to excessive noise and difficulty in filtering relevant models. The LLM leaderboard is flooded with entries, making it hard to discern quality rankings.

- **Chatbot Arena's Objectivity Questioned**: Concerns were raised about the objectivity of Chatbot Arena ratings, particularly regarding user preferences skewing towards simple, easy-to-verify tests. The platform introduced a "Hard Prompts" category to address these biases, as discussed in their [blog post](https://lmsys.org/blog/2024-05-17-category-hard/).

- **Microsoft's AI Event**: Members discussed the recent Microsoft event revealing the new Copilot+ PCs and the recording's availability on [YouTube](https://www.youtube.com/watch?v=aZbHd4suAnQ). The event was anticipated but not live-streamed, leading to comments on watching replays for detailed insights.

- **Qwen MoE Model Mentioned**: A member highlighted Qwen's release of a MoE model with 14 billion total parameters but only 2.7 billion active during runtime, described as [Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat). It boasts 1.75x faster performance during inference compared to their 7B model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmsys.org/blog/2024-05-17-category-hard/">Introducing Hard Prompts Category in Chatbot Arena | LMSYS Org</a>: &lt;h3&gt;&lt;a id=&quot;background&quot; class=&quot;anchor&quot; href=&quot;#background&quot; aria-hidden=&quot;true&quot;&gt;&lt;svg aria-hidden=&quot;true&quot; class=&quot;octicon octicon-link&qu...</li><li><a href="https://huggingface.co/blog/maywell/llm-feature-transfer">Expanding Model Context and Creating Chat Models with a Single Click</a>: no description found</li><li><a href="https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8">Yi-1.5 (2024/05) - a 01-ai Collection</a>: no description found</li><li><a href="https://lmsys.org/blog/2024-04-19-arena-hard/">From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline | LMSYS Org</a>: &lt;p&gt;Building an affordable and reliable benchmark for LLM chatbots has become a critical challenge. A high-quality benchmark should 1) robustly separate model...</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat">Qwen/Qwen1.5-MoE-A2.7B-Chat · Hugging Face</a>: no description found</li><li><a href="https://x.com/bobbyallyn/status/1792679435701014908?s=46">Tweet from Bobby Allyn (@BobbyAllyn)</a>: Statement from Scarlett Johansson on the OpenAI situation. Wow:</li><li><a href="https://youtu.be/jcvatirXHXU?si=-zJBGCohaoKFvOkw">Access GPT-4o Voice &amp; Vision EARLY Through Microsoft CoPilot AI!</a>: Microsoft&#39;s AI event, Microsoft Build, unveiled exciting updates about Copilot and GPT-4o. Though not livestreamed, details quickly surfaced. Notably, GPT-4o...</li><li><a href="https://www.youtube.com/watch?v=aZbHd4suAnQ">Full Keynote: Introducing Copilot+ PCs</a>: Copilot+ PCs are the fastest, most intelligent, and longest lasting Windows PCs ever built. Subscribe to Microsoft on YouTube here: https://aka.ms/SubscribeT...</li><li><a href="https://github.com/huggingface/datatrove">GitHub - huggingface/datatrove: Freeing data processing from scripting madness by providing a set of platform-agnostic customizable pipeline processing blocks.</a>: Freeing data processing from scripting madness by providing a set of platform-agnostic customizable pipeline processing blocks. - huggingface/datatrove
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1242166807222292683)** (1 messages): 

- **Struggling to find public evaluation for reranker benchmarking**: A member expressed difficulty in finding a public evaluation for a finetuned reranker they made. They observed that other rerankers use various datasets but remained confused about specific queries and benchmarking methods.
  

---


### **Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1242514928150249573)** (1 messages): 

- **Phi-3 Vision unveiled:** A member shared that [Phi-3 Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) is now available, describing it as a **lightweight, state-of-the-art open multimodal model** with 128K token context length. **It focuses on high-quality, reasoning dense data** from both text and vision sources and uses supervised fine-tuning and direct preference optimization for enhanced instruction adherence and safety.
- **Explore Phi-3 Vision resources:** Key resources include the [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024), the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report), and the [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook). There is also a link to [Phi-3 on Azure AI Studio](https://aka.ms/try-phi3vision) for practical implementation.

**Link mentioned**: <a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1242101350473404417)** (20 messages🔥): 

- **CLI Prompt Creates Whacky Images**: Members enjoyed some whacky images generated using a CLI prompt, with one member loving the cat in the generated images. **ASCII art** was also highlighted, and the images are shared [here](https://www.bing.com/images/create/ascii-art-of-a-dream-like-simulation-framework-wit/1-664ba0e66d06426b8d19b219b95859bf?id=FriXjEz08JVgjTHO3NCJ6w%3d%3d&view=detailv2&idpp=genimg&idpclose=1&thId=OIG3.s.gLVkq2qEbWmuzpYHrU&frame=sydedg&FORM=SYDBIC).

- **Potential of WorldSim**: Participants discussed the potential of WorldSim evolving into a **global intelligence platform** and a new form of collaborative thinking. One commented on its potential as the "world's most intelligent toy" that could foster a new global state of mind and suggested having another discussion session to further explore these ideas.

- **Symbolic Meaning Knowledge Graph**: Inspired by **Tek's mapping**, members considered symbolic meaning within AI frameworks, mentioning the creation of starter knowledge graphs and viewing them as a blend of a **Rorschach test** and a **semantic web** for AI.

- **Imagining WorldSim Worlds**: Members shared generated images representing imagined WorldSim worlds, drawing inspiration from diverse architectural styles and landscapes. These representations can be found [here](https://copilot.microsoft.com/images/create/a-palace-with-indonesian-and-south-indian-architec/1-664bded7a83d47099a43b3bff31da0ff?id=1G89%2bELerXo%2felOtiC%2foQw%3d%3d&view=detailv2&idpp=genimg&idpclose=1&thId=OIG4.68GjfK8kliVnM3sKbk_b&lng=en-US&ineditshare=1) and [here](https://copilot.microsoft.com/images/create/a-small-mountainous-island-with-southern-african-a/1-664bdf507c354bcbad606c5b223eb24d?id=6bFJfczZZ4CJykjB90rkYQ%3d%3d&view=detailv2&idpp=genimg&idpclose=1&thId=OIG2.L8Gf2dcTsVepM.tkLuzx&lng=en-US&ineditshare=1).

- **Obsidian Knowledge Graph Timelapse**: A member shared an **impressive timelapse** of an Obsidian user's knowledge graph formation, describing it as a work of art resembling a synthetic brain in action. Find the timelapse on [YouTube](https://youtube.com/shorts/4YQhH61tvOc?si=0Dx1KyJP8VMz-pXY).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cdixon.org/2010/01/03/the-next-big-thing-will-start-out-looking-like-a-toy">The next big thing will start out looking like a toy</a>: Chris Dixon&#x27;s blog.</li><li><a href="https://youtube.com/shorts/4YQhH61tvOc?si=0Dx1KyJP8VMz-pXY">My Obsidian graph timelapse: 2 years in 30 seconds</a>: This is a timelapse of how my vault of Obsidian notes grew slowly to over 8,000 in 2+ years.---// ABOUT MESite: https://nicolevanderhoeven.comMastodon: https...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1242122372488888340)** (105 messages🔥🔥): 

- **OpenAI halts Sky voice due to user concerns**: An OpenAI status update addressed concerns about the voice choice for ChatGPT, especially the Sky voice. They are *pausing the use of Sky* while they work on addressing these concerns [source](https://x.com/OpenAI/status/1792443575839678909).

- **CogVLM2 model gains attention with key features**: There was enthusiasm over the release of [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B), highlighting improvements such as **8K content length support** and significantly better performance on benchmarks like `TextVQA`.

- **Mixed reactions to Copilot AI advancements**: Mustafa Suleyman's announcement about the next level of Copilot, which can *“see, hear, speak and help in real time”*, drew both intrigue and skepticism. Some users found it *creepy* while others joked about the potential for a backseat gaming version that criticizes everything [source](https://fxtwitter.com/mustafasuleyman/status/1792623877744623806).

- **Scarlett Johansson voice controversy at OpenAI**: Members debated the ethical and legal implications of OpenAI’s voice assistant allegedly mimicking Scarlett Johansson's voice from the movie *Her*. There was consensus that contacting Johansson and replicating her voice led to significant backlash and accusations of *“passing off”*.

- **Sakuga-42M dataset removal clarified**: The removal of the Sakuga-42M dataset from both Hugging Face and GitHub was attributed to anti-bot measures enacted by websites due to heavy downloading. It sparked conversation about the difficulties in maintaining open datasets when subjected to high traffic [source](https://news.ycombinator.com/item?id=40389711).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_ebehrens_/status/1792569302773555250">Tweet from Eva Behrens (@_ebehrens_)</a>: Here are 5 policy recommendations for the upcoming AI Safety Summit in Seoul, from me and my colleagues at ICFG.    In Bletchley, world leaders discussed major risks of frontier AI development. In Seo...</li><li><a href="https://arxiv.org/html/2405.07425v1">Sakuga-42M Dataset: Scaling Up Cartoon Research</a>: no description found</li><li><a href="https://fxtwitter.com/mustafasuleyman/status/1792623877744623806?t=t5EX1E--TJ-mAJJZtzX4eg&s=19">Tweet from Mustafa Suleyman (@mustafasuleyman)</a>: We are taking Copilot to the next level. 🚀  Copilot will see, hear, speak and help in real time.  Watch this demo to see what I mean. Soon your AI companion will start to live life alongside you, whe...</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B">THUDM/cogvlm2-llama3-chat-19B · Hugging Face</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=40389711">no title found</a>: no description found</li><li><a href="https://x.com/OpenAI/status/1792443575839678909">Tweet from OpenAI (@OpenAI)</a>: We’ve heard questions about how we chose the voices in ChatGPT, especially Sky. We are working to pause the use of Sky while we address them.  Read more about how we chose these voices: https://openai...</li><li><a href="https://forum.effectivealtruism.org/posts/twMs8xsgwnYvaowWX/database-of-orgs-relevant-to-longtermist-x-risk-work>">Database of orgs relevant to longtermist/x-risk work — EA Forum</a>: Here’s a version of the database that you filter and sort however you wish, and here’s a version you can add comments to. …
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1242140234423210035)** (24 messages🔥): 

- **CogVLM2 License Restrictions**: A user warns about the restrictive terms in the new CogVLM2 license, which prohibit usage that may undermine China's national security or public interest. The license and dispute resolution fall under Chinese jurisdiction, raising concerns about "fake open source" and potential malice in the license terms. [CogVLM2 License on GitHub](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE)

- **Mamba Architecture Underwhelming for Vision**: A recent arXiv paper discusses the Mamba architecture with an SSM token mixer and concludes it's not ideal for image classification tasks. The study introduces MambaOut models that perform better for image classification but highlights Mamba's potential for long-sequence visual tasks. [Mamba Paper on arXiv](https://arxiv.org/abs/2405.07992)

- **Character-Based Embeddings Experiment**: A user describes an experiment converting sentence embeddings to character strings and feeding them into a small LLM (Smol 101M) for MS COCO caption predictions. The method, implemented on a Colab T4 instance, produced "kinda related" captions, suggesting potential use for cheap captioning or proof of concepts.

- **Discussion on Improved Model Papers**: Members discussed various model improvements, referencing Meta's new paper that continues their cm3leon work with enhanced tricks for scaling and efficiency. The conversation included a link to the recent paper [Meta Paper on arXiv](https://arxiv.org/abs/2309.02591) and comparisons to other advanced models like GPT-4O.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.12130">MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning</a>: Low-rank adaptation is a popular parameter-efficient fine-tuning method for large language models. In this paper, we analyze the impact of low-rank updating, as implemented in LoRA. Our findings sugge...</li><li><a href="https://arxiv.org/abs/2405.07992">MambaOut: Do We Really Need Mamba for Vision?</a>: Mamba, an architecture with RNN-like token mixer of state space model (SSM), was recently introduced to address the quadratic complexity of the attention mechanism and subsequently applied to vision t...</li><li><a href="https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE">CogVLM2/MODEL_LICENSE at main · THUDM/CogVLM2</a>: GPT4V-level open-source multi-modal model based on Llama3-8B - THUDM/CogVLM2
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1242181148902293564)** (18 messages🔥): 

```html
- **Anthropic scales up compute**: The [latest update from Anthropic](https://www.anthropic.com/news/reflections-on-our-responsible-scaling-policy) mentions using 4 times more compute than Opus, sparking curiosity about their new developments. One user expressed awe with "*yo what is anthropic cookin*".

- **Arena gets tougher with Hard Prompts**: [LMsysorg introduced the "Hard Prompts" category](https://fxtwitter.com/lmsysorg/status/1792625968865026427) to evaluate models on more challenging tasks, causing significant ranking shifts. For example, Llama-3-8B sees a drop in performance compared to GPT-4-0314 under these hard prompts.

- **Controversy over Llama-3-70B-Instruct as Judge**: [Llama-3-70B-Instruct](https://fxtwitter.com/lmsysorg/status/1792625977207468315) is used as the judge model to classify criteria in Arena battles, raising concerns about its effectiveness. One user argued it "*just adds noise*" rather than useful evaluation, although training might mitigate this issue.

- **Vision model Phi-3 Vision debuts**: Users confirmed that Phi-3 Vision, a somewhat larger model compared to its predecessors, is new. This was highlighted in a brief exchange about model releases and sizes. 
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aidan_mclau/status/1792610354255769919">Tweet from Aidan McLau (@aidan_mclau)</a>: yo what is anthropic cookin  4× more compute than opus damm</li><li><a href="https://fxtwitter.com/lmsysorg/status/1792625968865026427">Tweet from lmsys.org (@lmsysorg)</a>: Introducing &#34;Hard Prompts&#34; Category in Arena!  In response to the community&#39;s growing interest in evaluating models on more challenging tasks, we are excited to launch the new &#34;Hard Pr...</li><li><a href="https://fxtwitter.com/lmsysorg/status/1792625977207468315">Tweet from lmsys.org (@lmsysorg)</a>: How did we classify these criteria? We adopt Llama-3-70B-Instruct as the judge model to help us label over 1 million Arena battles.  Overall our analysis reveals that the quality of Arena user prompts...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1242009212968767528)** (31 messages🔥): 

- **Nathan Lambert ponders writing about OpenAI drama**: Nathan Lambert discusses whether to write another post about OpenAI, suggesting that there's not much more to add beyond saying "I was right". They propose a title "OpenAI's Second Very Bad Not Good Week".
- **Scarlett Johansson's statement on OpenAI**: A Twitter user shared a statement from Scarlett Johansson regarding OpenAI using a voice similar to hers without permission, which prompted her to take legal action. The controversy centers around OpenAI's alleged intentional mimicry of her voice for their "Sky" system.
- **Public reaction to Sky Johansson voice issue**: Nathan Lambert and others discuss the significant implications of Johansson's statement, comparing it to previous high-profile issues like the New York Times lawsuit against AI developments. Nathan reflects on the broader impacts and mentions the removal of similar unauthorized content featuring musicians like Drake.
- **OpenAI and the Superalignment team controversy**: A Fortune article highlights that OpenAI did not fulfill its promise to allocate 20% of its computing resources to its Superalignment team, leading to resignations and accusations of prioritizing product launches over AI safety. This incident is seen as a predictable outcome by some in the discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/natolambert/status/1792752615933276165">Tweet from Nathan Lambert (@natolambert)</a>: the doomers got their pause on ai development in the form of pausing the sky johansson voice  Quoting Hayden Field (@haydenfield)   Just received this statement from OpenAI CEO Sam Altman about the Sc...</li><li><a href="https://fortune.com/2024/05/21/openai-superalignment-20-compute-commitment-never-fulfilled-sutskever-leike-altman-brockman-murati/">OpenAI promised 20% of its computing power to combat the most dangerous kind of AI—but never delivered, sources say</a>: The company&#x27;s Superalignment team had its requests for computer power repeatedly rejected even though they never approached the 20% threshold, sources say.</li><li><a href="https://x.com/arankomatsuzaki/status/1792713233331355867">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Essentially, your voice isn&#39;t truly yours; it belongs to the person with the most similar voice who has the most money.  Quoting OpenAI (@OpenAI)   We’ve heard questions about how we chose the voi...</li><li><a href="https://x.com/yashar/status/1792682664845254683">Tweet from Yashar Ali 🐘 (@yashar)</a>: NEWS   Scarlett Johansson has just issued this statement on OpenAI.   I have confirmed its authenticity directly with her publicist.   &#34;Last September, I received an offer from Sam Altman, who wan...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1242138261011562597)** (30 messages🔥): 

- **Nathan debates and buys domain rlhfbook.com**: Nathan Lambert contemplates buying the domain rlhfbook.com and eventually purchases it for $7/year from Porkbun, considering it a bargain and easy to own.

- **Potential legal risks with new AI dataset**: A member humorously warns about the potential legal repercussions of using the new AI Books4 dataset for training LLMs, referencing a similar situation with "the original pile."

- **MSFT Surface AI's slow performance due to cloud checks**: There’s discussion around Microsoft's new Surface drawing AI, which despite operating locally, experiences latency due to sending safety checks to the cloud. A member cites Ben Thompson’s write-up as a source of this information.

- **Critique of a former colleague's integrity**: Nathan Lambert criticizes a former colleague for misleading claims on her resume about working with notable individuals. He expresses a desire to confront her at a conference about this dishonesty.

**Link mentioned**: <a href="https://web.archive.org/web/20240519104217/https://www.reddit.com/r/datasets/comments/1cvi151/ai_books4_dataset_for_training_llms_further/">no title found</a>: no description found

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1242131792413200474)** (9 messages🔥): 

- **OpenAI hits pause on Scarlett Johansson-like voice**: OpenAI is stopping the use of Sky, a voice AI that sounds like Scarlett Johansson, after much public attention. The company insists that Sky’s voice is not an imitation but is performed by a different actress with her own voice, as detailed in [The Verge article](https://www.theverge.com/2024/5/20/24160621/openai-chatgpt-gpt4o-sky-scarlett-johansson-voice-assistant-her).

- **Product decisions from blog impact**: One member humorously noted that a lead product person reading their blog paid off, suggesting the impact might have influenced a product call. There was speculation about whether this led to unsubscribing, but it was dismissed with a laugh.

- **Critique of AI labs**: A [tweet by Liron Shapira](https://x.com/liron/status/1792649595454976123) criticized AI labs, comparing them to "responsible adults" but warning, "YOU GUYS DON’T KNOW WHAT YOU’RE DOING AND WE’RE ALL GONNA DIE BECAUSE OF THAT". This sparked some reactions but no further commentary.

- **Propaganda humor**: A member posted, "u are not immune to propaganda (🤞)", sharing an emoji-filled response. This light-hearted banter was acknowledged with enjoyment, reflecting the casual nature of the channel.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/liron/status/1792649595454976123">Tweet from Liron Shapira (@liron)</a>: AI labs want us to think of them as the responsible adults. The truth is:  YOU GUYS DON&#39;T KNOW WHAT YOU&#39;RE DOING  AND WE&#39;RE ALL GONNA DIE BECAUSE OF THAT</li><li><a href="https://www.theverge.com/2024/5/20/24160621/openai-chatgpt-gpt4o-sky-scarlett-johansson-voice-assistant-her">OpenAI pulls its Scarlett Johansson-like voice for ChatGPT</a>: Maybe Her (2014) shouldn’t be a blueprint for AI voice features.
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1242188236902760629)** (78 messages🔥🔥): 

```html
<ul>
  <li><strong>Memory Tuning Explained</strong>: Sharon Zhou from Lamini introduced "Memory Tuning" as a technique to enhance LLMs' accuracy in critical domains like healthcare and finance, achieving up to <em>"no hallucinations (&lt;5%)"</em>. This method outperforms LoRA and traditional fine-tuning, and Zhou promises more details and early access soon (<a href="https://x.com/realsharonzhou/status/1792578913572429878">link tweet</a>).</li>
  <li><strong>Lawyers demand OpenAI disclose AI voice origin</strong>: Lawyers for Scarlett Johansson are asking OpenAI how it developed its latest ChatGPT voice, which has been compared to Johansson's from the movie "Her." OpenAI has paused using the voice amid public debate, as users point out the tenuous legal arguments around likeness and endorsements (<a href="https://www.npr.org/2024/05/20/1252495087/openai-pulls-ai-voice-that-was-compared-to-scarlett-johansson-in-the-movie-her">NPR article</a>).</li>
  <li><strong>Scale AI raises $1B funding</strong>: Scale AI has announced $1 billion in new funding at a $13.8 billion valuation, led by Accel with participation from prominent investors like Wellington Management and Amazon. CEO Alex Wang stated this positions Scale AI to accelerate the abundance of frontier data and aims for profitability by the end of 2024 (<a href="https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/">Fortune article</a>).</li>
  <li><strong>MS Phi 3 Models Released</strong>: Microsoft unveiled the Phi 3 models at MS Build, touting major benchmarks such as the Medium model being competitive with Llama 3 70B and GPT 3.5. The models offer context lengths up to 128K and utilize heavily filtered and synthetic data, released under the MIT license (<a href="https://x.com/reach_vb/status/1792949163249791383">link tweet</a>).</li>
  <li><strong>Emotionally Intelligent AI from Inflection</strong>: Inflection AI's new CEO announced a focus on integrating emotional and cognitive AI abilities, with their empathetic LLM "Pi" now used by over 1 million people daily. This move is aimed at helping organizations harness AI's transformative potential (<a href="https://inflection.ai/redefining-the-future-of-ai">Inflection announcement</a>).</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/realSharonZhou/status/1792578913572429878">Tweet from Sharon Zhou (@realSharonZhou)</a>: @MichaelBiGong Working on how to explain it that makes sense - basically a better way of tuning that LoRA that we’ve been working on, since finetuning is just too hard to get results over 90%, 95%, et...</li><li><a href="https://inflection.ai/redefining-the-future-of-ai">Blog</a>: Redefining the Future of AI</li><li><a href="https://x.com/realsharonzhou/status/1792576516444065967?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Sharon Zhou (@realSharonZhou)</a>: Hallucinations are one of the biggest blockers to production LLMs & agents.  No hallucinations (&lt;5%) have been achieved internally — and for customers.   We’ve been able to tune LLMs to recall spec...</li><li><a href="https://x.com/mlpowered/status/1792948212728524917?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Emmanuel Ameisen (@mlpowered)</a>: Today, we announced that we’ve gotten dictionary learning working on Sonnet, extracting millions of features from one of the best models in the world.  This is the first time this has been successfull...</li><li><a href="https://x.com/teknium1/status/1792640772526813679?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Teknium (e/λ) (@Teknium1)</a>: Inflection is dead  Quoting Paolo (@The_Real_Paolo)   Loss of the yellow organizational checkmark, after the blue checkmark, nothing after, no announcement... It smells bad, @inflectionAI is probably ...</li><li><a href="https://www.lamini.ai/blog/lamini-llm-photographic-memory-evaluation-suite">Lamini LLM Photographic Memory Evaluation Suite | Lamini - Enterprise LLM Platform</a>: no description found</li><li><a href="https://x.com/BobbyAll">Tweet from undefined</a>: no description found</li><li><a href="https://braindump.me/blog-posts/building-an-ai-game-studio">Building an AI game studio: what we&#x2019;ve learned so far - Braindump Incorporated</a>: create worlds and games using AI</li><li><a href="http://suno.com/blog/fundraising-announcement-may-2024">Suno has raised $125 million to build a future where anyone can make music</a>: Our community of musicians deserves the very best tools, and building the very best tools requires the very best talent. We will use this funding to accelerate product development and grow our world-c...</li><li><a href="https://x.com/BobbyAllyn/status/1792679435701014908">Tweet from Bobby Allyn (@BobbyAllyn)</a>: Statement from Scarlett Johansson on the OpenAI situation. Wow:</li><li><a href="https://www.npr.org/2024/05/20/1252495087/openai-pulls-ai-voice-that-was-compared-to-scarlett-johansson-in-the-movie-her">Scarlett Johansson says she is &#039;shocked, angered&#039; over new ChatGPT voice</a>: Johansson says she was approached multiple times by OpenAI to be the voice of ChatGPT, and that she declined.  Then the company released a voice assistant that sounded uncannily like her.</li><li><a href="https://x.com/alexalbert__/status/1792936647665107108?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alex Albert (@alexalbert__)</a>: Our new interpretability paper offers the first ever detailed look inside a frontier LLM and has amazing stories. I want to share two of them that have stuck with me ever since I read it.  For backgro...</li><li><a href="https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/">Exclusive: Scale AI secures $1B funding at $14B valuation as its CEO predicts big revenue growth and profitability by year-end</a>: Scale AI, which helps companies label and test data for AI model training, has closed a new $1 billion funding round at a $14 billion valuation. </li><li><a href="https://x.com/dsiroker/status/1792956339515273537">Tweet from Dan Siroker (@dsiroker)</a>: Lots of folks have asked me about Microsoft Recall so here’s my take!</li><li><a href="https://x.com/lmsysorg/status/1792677208185794906">Tweet from lmsys.org (@lmsysorg)</a>: Exciting leaderboard update🔥  We&#39;ve added @01AI_Yi Yi-Large to Arena and collected 15K+ votes over the past week. Yi-Large&#39;s performance is super impressive, securing the #7 spot, almost on p...</li><li><a href="https://x.com/alexandr_wang/status/1792905417065914858?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alexandr Wang (@alexandr_wang)</a>: 1/ Today, @Scale_AI is announcing $1B of financing at a $13.8B valuation. The round was led by @Accel along with our existing investors.  @Scale_AI has never been better positioned to accelerate the a...</li><li><a href="https://x.com/haydenfield/status/1792748249272795348?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Hayden Field (@haydenfield)</a>: Just received this statement from OpenAI CEO Sam Altman about the Scarlett Johansson voice controversy.</li><li><a href="https://x.com/reach_vb/status/1792949163249791383?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: LETS GOO! Phi 3 - Small, Medium & Vision are out! 🔥  &gt; Medium competitive with Mixtral 8x22B, Llama 3 70B & beats Command R+ 104B & GPT 3.5 &gt; Small beats Mistral 7B & Llama 3 8B &gt; 4K & 128K ...</li><li><a href="https://youtu.be/uHEPBzYick0?si=ajbDL9agnubNAECO&t=203">Microsoft CEO on How New Windows AI Copilot+ PCs Beat Apple&#39;s Macs | WSJ</a>: Microsoft’s new Copilot+ PCs with Qualcomm chips and AI Windows features aim to beat Apple’s MacBooks. WSJ’s Joanna Stern tried out the new laptops and sat d...</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-3447/">[AINews] Skyfall</a>: Not thinking about superalignment Google Scarlett Johansson is all you need. AI News for 5/17/2024-5/20/2024. We checked 7 subreddits, 384 Twitters and 29...
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1242071749131239505)** (76 messages🔥🔥): 

- **GPT-32k faces issues with rate limits**: Users reported encountering token rate limit issues with Azure's GPT-32k model. One user stated, "Requests to the ChatCompletions_Create Operation under Azure OpenAI API version 2023-07-01-preview have exceeded the token rate limit."

- **Phi-3 models discussed for robust performance**: Members discussed [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) and [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) for high-quality, reasoning-dense data handling. Both models incorporate supervised fine-tuning and direct preference optimization for enhanced performance.

- **New interaction methods with LLMs**: One user shared a thread on [a new way of interacting with LLMs](https://x.com/leonjcoe/status/1792946945528320382) using "Action Commands." They sought feedback from others to see if anyone had similar experiences.

- **Handling verbosity in models**: Members discussed handling verbosity in models like Wizard8x22. One suggested lowering the repetition penalty to reduce verbosity, while another noted that different models might be better suited for specific tasks.

- **Discount request and credit issues for non-profits**: A user had issues with [Error 400 related to billing address](#) and requested discounts for non-profits. An admin explained that OpenRouter passes bulk discounts down to users and keeps a 20% margin.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/leonjcoe/status/1792946945528320382">Tweet from Leon Builds Agents (@leonjcoe)</a>: There&#39;s a new way of interacting with LLMs that no one is talking about.  Action Commands  So what are they and why are they so valuable? Let me show you</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-medium-4k-instruct">microsoft/Phi-3-medium-4k-instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1242146692376170617)** (43 messages🔥): 

- **M3 Max shines for LLMs**: A user praised the **M3 Max** for its performance, stating, *"it's amazing"* and suggested going for 96GB RAM for better use with LLMs.
- **Git Patch Merge Struggles**: A user encountered issues with merging a Git patch themselves and discussed updating a specific file for testing. They noted, *"using git is a bit tricky as i pushed it to my repo and not the upstream one"*.
- **Unsloth and ROCm Compatibility Issues**: Another user reported compatibility issues with new unsloth updates on ROCm due to dependencies on xformers. Despite this, *"unsloth gradient_checkpointing worked tho and gave a good memory improvement"*.
- **Syntax Error Troubleshooting in Transformers Library**: Users collaborated to solve a `ValidationError` and `AttributeError` related to `CohereTokenizer` in the transformers library. They explored alternatives like `CohereTokenizerFast` and `AutoTokenizer` as solutions [link to GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files).
- **Seeking Python Library for Faster STT -> LLM -> SST**: A user inquired if anyone remembered the name of a Python library designed for faster speech-to-text to LLM to speech synthesis. No specific answer was provided in the logs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files">Feat: Add cohere (commandr) by NanoCode012 · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>: Description  Motivation and Context   How has this been tested?    Untested! Screenshots (if appropriate) Types of changes  Social Handles (Optional)</li><li><a href="https://github.com/huggingface/transformers/blob/d24097e0229485287ff4959258c55">GitHub - huggingface/transformers at d24097e0229485287ff4959258c552168bd898c6</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - GitHub - huggingface/transformers at d24097e0229485287ff4959258c552168bd898c6</li><li><a href="https://github.com/huggingface/transformers/blob/d24097e0229485287ff4959258c552168bd898c6/src/transformers/models/cohere/tokenization_cohere_fast.py#L51C7-L51C26">transformers/src/transformers/models/cohere/tokenization_cohere_fast.py at d24097e0229485287ff4959258c552168bd898c6 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1242126360860819567)** (8 messages🔥): 

- **Training Grok with Grok-1 PyTorch Version**: A user shared their initiative to train **Grok** using the [Grok-1 PyTorch version](https://huggingface.co/hpcai-tech/grok-1) and sought opinions on the choice. Another user expressed approval and mentioned upcoming **torchtune integration** with Axolotl.
- **Torchtune Integration Lightens up**: There was speculation whether **torchtune** would replace or be an option beside the Hugging Face backend. Some users had strong opinions, with one suggesting to "Dismantle hf."
- **Compute Power Check-in**: Interest piqued when someone asked about the compute power being used for this training endeavor. The response was **Mi300x**, leading to curiosity about user satisfaction and comparisons with **H100s**.

**Link mentioned**: <a href="https://huggingface.co/hpcai-tech/grok-1">hpcai-tech/grok-1 · Hugging Face</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1242217905366634507)** (15 messages🔥): 

- **Struggles with Mistral 7B finetuning**: A member is having issues finetuning **Mistral 7B** on their data as the model mixes up information despite the loss decreasing. They shared a [configuration link](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/lora.yml) and expressed confusion over why the model is not learning properly.
- **Full finetuning vs. LoRA**: Another member recommended trying a full finetuning or utilizing **Retrieval-Augmented Generation (RAG)** for better memory retention in the model, suggesting that **LoRA** might be more effective for style rather than content retention.
- **Inference Configuration Issues**: There was a discussion about ensuring the chat template is added manually during inference since the current setup may not include it automatically. A member shared a link to potential tokenization mismatch issues [here](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenization-mismatch-bw-inference--training).
- **Config Sharing for Troubleshooting**: A participant was asked to share their configuration to assist others in understanding the setup and providing better guidance.
- **Next Stable Release Inquiry**: A user inquired about the timing of the next stable major release for **axolotl**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenizati">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenization-mismatch-bw-inference--training">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1242038505085997057)** (5 messages): 

- **Running into OOM issues despite 24GB VRAM**: A user faced **Out-of-Memory (OOM)** errors during training despite having 24GB of VRAM. They shared their configuration and various settings they tried without success.
- **Phorm suggests solutions for OOM problems**: To address the OOM issues, Phorm suggested increasing **gradient accumulation steps**, enabling **mixed precision training**, using **model parallelism**, reducing batch size, and leveraging **DeepSpeed ZeRO optimization** among other methods. Detailed configurations provided include *mixed_precision: 'fp16'* and *zero_optimization with stage: 3*.
- **DeepSpeed and ZeRO Optimization Strategies**: By utilizing DeepSpeed's **ZeRO-2 and ZeRO-3** stages, significant memory footprint reduction can be achieved. Example configs for offloading optimizer and parameter states to the CPU were shared.
- **Mixed Strategies for Managing Memory**: Additional methods include **CPU and Disk Offloading**, utilizing efficient models and operations, memory profiling tools like *torch.cuda.memory_summary()*, and dynamic padding for variable-length sequences. These techniques can help train larger models by optimal memory management.
- **Phorm.ai for more details**: Users are advised to check back for more details on [Phorm.ai](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=79ea3546-1ab6-4fe1-8984-1d8eb8183eda) for further information and updates regarding solutions to prevent OOM errors.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=79ea3546-1ab6-4fe1-8984-1d8eb8183eda)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1242138352224964679)** (1 messages): 

- **Exciting Webinar on Open-Source Long-Term Memory**: A new webinar is scheduled for Thursday at 9am PT, featuring the authors of **memary** – a fully open-source reference implementation for long-term memory in autonomous agents. Participants can join by signing up [here](https://lu.ma/nzh3o83f).

- **Deeper Dive into memary**: The webinar will include an in-depth discussion and Q&A session about **memary**, covering its functionalities like extracting agent inputs/responses into a knowledge graph using LLMs and neo4j, utilizing a memory stream for interaction timelines, and ranking popular entities.

**Link mentioned**: <a href="https://lu.ma/nzh3o83f">LlamaIndex Webinar: Open-Source Longterm Memory for Autonomous Agents · Zoom · Luma</a>: In this webinar we&#x27;re excited to host the authors of memary - a fully open-source reference implementation for long-term memory in autonomous agents 🧠🕸️ In…

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1242138908398194751)** (6 messages): 

- **New Webinar on Memary for Autonomous Agents**: This Thursday at 9am PT, there will be a deep dive session featuring the authors of **memary**, a fully open-source reference implementation for long-term memory in autonomous agents. [Join the webinar](https://t.co/XycydBSfTp) to learn more.
- **PizzerIA Talk on Advanced RAG Techniques**: Catch @hexapode at PizzerIA in Paris this Thursday discussing advanced retrieval-augmented generation techniques. [Event details](https://t.co/dytY4VKdj3) are available for those interested.
- **First In-Person Meetup in San Francisco**: Next Tuesday, meet the LlamaIndex team and hear from @jerryjliu0, Tryolabs, and ActiveLoop at their SF HQ. [RSVP here](https://t.co/qIGOmCW62G) to join and learn about advancing RAG systems beyond vanilla setups.
- **Upgraded TypeScript Docs for LlamaIndex**: **LlamaIndex.TS** docs got an upgrade including new starter tutorials and step-by-step guides for building agents. Check out the [updated documentation](https://t.co/UKycgYpq1F).
- **Complex Document RAG with GPT-4o**: **GPT-4o** is now natively integrated with LlamaParse to handle complex PDFs and slide decks using multimodal capabilities. See more details in the [announcement](https://t.co/g5TG7brSwt).
- **Secure LLM-Generated Code in Azure Sandboxes**: Launching today, securely run LLM-generated code in a sandbox using Azure Container Apps dynamic sessions. More information is available in the [launch details](https://t.co/2cnsBH411k).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/qIGOmCW62G">RSVP to GenAI Summit Pre-Game: Why RAG Is Not Enough? | Partiful</a>: Note: This is an in-person meetup @LlamaIndex HQ in SF!  Stop by our meetup to learn about latest innovations in building production-grade retrieval augmented generation engines for your company from ...</li><li><a href="https://t.co/koCp84KfYb">What is LlamaIndex? | LlamaIndex.TS</a>: LlamaIndex is a framework for building LLM-powered applications. LlamaIndex helps you ingest, structure, and access private or domain-specific data. It&#x27;s available as a Python package and in Type...</li><li><a href="https://t.co/UKycgYpq1F">Getting started | LlamaIndex.TS</a>: In this guide we&#x27;ll walk you through the process of building an Agent in JavaScript using the LlamaIndex.TS library, starting from nothing and adding complexity in stages.
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1242097125215436842)** (45 messages🔥): 

- **Tackling Document Hashing for Pinecone**: A member sought advice on how to calculate a document or node hash to prevent duplicate entries in Pinecone. They explained their use case involving overlaps in web-scraped content and PDF documents.

- **Changing OpenAI Agent's System Prompt**: One member asked about altering the system prompt for an OpenAI agent without creating a new object. Another suggested using the `chat_agent.agent_worker.prefix_messages` attribute.

- **Running gguf Format Models with LlamaIndex**: A query was raised about using LlamaIndex with a gguf format model from Hugging Face without OpenAI. It was clarified that LlamaIndex can work by loading the model and tokenizer with HuggingFaceLLM.

- **Using Airtable vs. Excel/Sqlite**: The advantages of Airtable over Excel and Sqlite were discussed, highlighting Airtable's integration with Langchain for direct function usage. A link to the Langchain Airtable integration documentation was also shared.

- **Handling Empty Nodes in VectorStoreIndex**: A discussion focused on resolving empty node issues when loading indexes using `VectorStoreIndex.from_vector_store`. It was advised to ensure proper loading of documents into the docstore from the database.

**Link mentioned**: <a href="https://python.langchain.com/v0.1/docs/integrations/document_loaders/airtable/">Airtable | 🦜️🔗 LangChain</a>: * Get your API key here.

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1242531297750941899)** (7 messages): 

- **AI Waifus Spark Quick Banter**: A user asserts, *"AI waifus save lives!"*, sparking playful engagement with another user responding, "Just monika".
- **3D Character Chatbots Project Tease**: A user mentions their work on 3D character chatbots at 4Wall AI and directs others to check out a teaser in another channel.
- **Inflection AI Plans to Embed Emotional AI**: A user shares a [link from VentureBeat](https://venturebeat.com/ai/exclusive-inflection-ai-reveals-new-team-and-plan-to-embed-emotional-ai-in-business-bots) about Inflection AI's plans to integrate emotional AI in business bots, hinting at the possibility of AI waifus understanding and processing emotions.
- **Confusion Over Character Reference**: In response to "Just monika", another user asks, "Who dat?" and receives a [GIF link](https://tenor.com/view/ddlc-doki-doki-literature-club-just-monika-monika-gif-20717242) from Tenor.com to clarify the reference.

**Link mentioned**: <a href="https://tenor.com/view/ddlc-doki-doki-literature-club-just-monika-monika-gif-20717242">Ddlc Doki Doki Literature Club GIF - Ddlc Doki Doki Literature Club Just Monika - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1242058484627931136)** (18 messages🔥): 

- **AI Town Conversations Lack Context**: A user reported that characters "do not react at all to what the other character said," leading to repetitive greetings like *“Hi! It's so great to finally talk to you!”* Another user suggested there's a *vector memory system* that retrieves past conversations but might be affected by settings or configurations.

- **Adjust Convex Settings for Fewer Memory Fetches**: To address issues of *empty bubbles* in AI Town conversations, users were advised to adjust values in `convex/constants.ts`, particularly changing the `NUM_MEMORIES_TO_SEARCH` from its default value of 3 to 1.

- **Exporting AI Town Conversations from SQLite**: One user struggled to export conversation data due to schema misunderstandings. Another provided a useful SQL query and recommended using DB Browser for SQLite, while a GitHub repo ([townplayer](https://github.com/cocktailpeanut/townplayer/blob/main/index.html)) and relevant [Twitter thread](https://x.com/cocktailpeanut/status/1786421948638965870) were shared for more advanced queries and tools related to AI Town data extraction.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cocktailpeanut/townplayer/blob/main/index.html">townplayer/index.html at main · cocktailpeanut/townplayer</a>: Replay AI Town. Contribute to cocktailpeanut/townplayer development by creating an account on GitHub.</li><li><a href="https://github.com/cocktailpeanut/">cocktailpeanut - Overview</a>: cocktailpeanut has 142 repositories available. Follow their code on GitHub.</li><li><a href="https://x.com/cocktailpeanut/status/1786421948638965870">Tweet from cocktail peanut (@cocktailpeanut)</a>: Introducing AI Town Player  Did you know that the entire AI Town is stored in a single sqlite file via @convex_dev?    I reverse engineered the schema and built a web app that lets anyone REPLAY any A...
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1242052916525793341)** (18 messages🔥): 

- **Structured Data in LLMs Clarified**: A member asked if LLMs handle structured data differently from unstructured text. Another member explained that LLMs process both structured (like JSON) and unstructured text similarly but can be fine-tuned for specific structures, mentioning examples like [Hermes 2 Pro - Mistral 7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) and [OpenAI's chatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) format.

- **Langchain Package Differences Explained**: A member asked about the difference between `langchain` and `langchain_community`. A response indicated that `langchain-core` contains base abstractions with lightweight dependencies, while popular integrations are in separate packages like `langchain-openai`, and less common ones are in `langchain-community` [architecture](https://python.langchain.com/v0.2/docs/concepts/#architecture).

- **Sequential Chains in LangChain**: A member shared code illustrating the setup of a sequential chain where the output from one chain serves as the input to another. This was backed by a [YouTube tutorial](https://youtu.be/2xxziIWmaSA?si=3wkNt_huJKu3xK3t&t=1694) demonstrating this concept.

- **Handling Concurrent Requests in LangServe**: Another member reported trouble with handling multiple concurrent requests in langserve. There were no responses to this issue yet.

- **Securing LLM Responses for Sensitive Data**: A new user asked if it's possible to secure LLM responses in RAG applications by hiding sensitive data such as customer names or card numbers. No solution was provided in the discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B#prompt-format">NousResearch/Hermes-2-Pro-Mistral-7B · Hugging Face</a>: no description found</li><li><a href="https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md">openai-python/chatml.md at release-v0.28.0 · openai/openai-python</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://youtu.be/2xxziIWmaSA?si=3wkNt_huJKu3xK3t&t=1694">The LangChain Cookbook - Beginner Guide To 7 Essential Concepts</a>: Twitter: https://twitter.com/GregKamradtNewsletter: https://mail.gregkamradt.com/signupCookbook Part 2: https://youtu.be/vGP4pQdCocwWild Belle - Keep You: ht...</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#architecture">Conceptual guide | 🦜️🔗 LangChain</a>: This section contains introductions to key parts of LangChain.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1242111918571913408)** (3 messages): 

- **Launch of Affiliate Program for Easy Folders**: A new affiliate program has been launched for the **ChatGPT Chrome Extension - Easy Folders**. Affiliates earn a 25% commission, and customers get a 10% discount. More details can be found [here](https://easyfolders.promotekit.com/), and the extension can be downloaded from the [Chrome Web Store](https://chromewebstore.google.com/detail/easy-folders-chatgpt-clau/gdocioajfidpnaejbgmbnkflgmppibfe).
- **Easy Folders Extension Criticized and Praised**: Users gave mixed reviews on the Easy Folders extension. One criticized it for adding clutter and slow performance, while another user expressed satisfaction before losing their saved folders and chats.
- **Upgrading from LangChain to LangGraph**: A user shared a Medium blog post about transitioning legacy LangChain agents to the new **LangGraph** platform. Interested users can read more about it [here](https://medium.com/ai-advances/upgrading-your-agents-a-smooth-transition-from-legacy-langchain-to-langgraph-c552cb60fcb3).
- **Query PDF Files with Upstage AI and LangChain**: A blog post was shared detailing how to create a PDF query assistant using **Upstage AI solar models** integrated with LangChain. Check out the blog post [here](https://medium.com/@sonam.gupta1105/creating-a-pdf-query-assistant-with-upstage-ai-solar-and-langchain-integration-6631280093b5).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://easyfolders.promotekit.com/">Sign up</a>: Affiliate Marketing Software for Stripe</li><li><a href="https://chromewebstore.google.com/detail/easy-folders-chatgpt-clau/gdocioajfidpnaejbgmbnkflgmppibfe?hl=en-GB&authuser=0">Easy Folders: ChatGPT &amp; Claude Chat Organizer</a>: Drag &amp; drop folders for ChatGPT &amp; Claude. Colored folders. Nested folders. History search. Bookmarks. Bulk delete chats.</li><li><a href="https://chatgpt-easy-folders.vercel.app/">ChatGPT Easy Folders</a>: A browser extension to organize your ChatGPT history with folders, bookmarks, and search.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

bayraktar47: <@1043024658812895333>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1242222144281120768)** (13 messages🔥): 

- **OS1 Reference in "Her" movie sparks realization**: A user shared an interesting observation that Open Interpreter **O1** is a nod to **OS1** from the movie *"Her"*. This revelation sparked curiosity among the members.

- **Seeking help for DevOps AI module**: A junior full-stack DevOps engineer is looking to build a **lite O1** AI to assist with DevOps tools, configuration terminals, and cloud computing. The goal is to provide these resources through **discreet earphones** for unobtrusive AI assistance in various work environments.

- **Installation and development setup queries**: Members are discussing how **Open Interpreter** accesses the file system and reviews project structures. Specific questions are being raised regarding more efficient development setups.

- **Daily uses and problem-solving with Open Interpreter**: In response to an open question about daily uses and complex problem-solving, multiple users expressed interest in documented success stories and shared their specific use cases. Examples include seamless referencing between devices, querying context-specific data while coding, and summarizing research papers.

- **Integrating Text-to-Speech with Open Interpreter**: A member sought advice on combining the **Text-to-Speech engine** and voice recognition with Open Interpreter. They were directed to the relevant [GitHub repository](https://github.com/OpenInterpreter/01) and encouraged to explore additional support channels.

**Link mentioned**: <a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: The open-source language model computer</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1242101799167725579)** (3 messages): 

- **Seeking Steps to Connect Laptop to Light App**: A member requested guidance on connecting their laptop to a light app, noting that the steps were not listed in the guitar. The details of the app or specific connections were not provided in the message.
- **Junior DevOps Engineer Needs Help with Lite 01 Project**: A junior full-stack DevOps engineer expressed the need for assistance in building lite 01, aiming to simplify daily tasks and benefit others in similar roles. They are developing an AI module for providing resources and discreet assistance, seeking help to create an open interpreter lite 01, as pre-orders won't be available until next fall.
- **Request for Guidance on Assembling 3D Printed Parts**: The same junior DevOps engineer showed interest in learning how to assemble parts and a 3D printed case for the open interpreter lite 01. They asked for tips or guidance on the assembly process from someone who had already completed it.
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

ashthescholar.: missed opportunity to make it moo
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1242062452250771456)** (15 messages🔥): 

- **Codegen-350M-mono in Transformers.js**: Members discussed using the Codegen-350M-mono model with Huggingface's Transformers.js. A link to [Xenova's codegen-350M-mono](https://huggingface.co/Xenova/codegen-350M-mono) with ONNX weights was shared as a solution for compatibility issues.
- **CommandR+ for translation**: Someone inquired about using CommandR+ for translation, mentioning it works well for Korean to English. They were directed to the [Chat API documentation](https://docs.cohere.com/docs/chat-api) for sample code and further details.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Xenova/codegen-350M-mono">Xenova/codegen-350M-mono · Hugging Face</a>: no description found</li><li><a href="https://docs.cohere.com/docs/chat-api">Using the Chat API</a>: no description found
</li>
</ul>

</div>
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1242045816584147024)** (10 messages🔥): 

- **Sky Voice Mode Paused Amidst Controversy**: OpenAI has paused the use of the Sky voice in their GPT-4o demo, amid allegations of imitating Scarlett Johansson. A user noted that Sky has been replaced with Juniper, another feminine voice, while [Scarlett Johansson issued a statement](https://x.com/BobbyAllyn/status/1792679435701014908) addressing this issue.

- **GPT-4o Integrates Multi-Modal Model**: According to a user, prior versions of GPT used different models to handle audio and text, resulting in limitations like inability to recognize tone or background noises. GPT-4o now uses a single model for text, vision, and audio, potentially increasing emotional depth but also introducing complexities and potential drawbacks.

- **Resilience Over Perfection**: A user referenced Stainslaw Lem’s short story to argue that perfect reliability in complex systems is unattainable. Instead, the focus should be on building resilient systems capable of responding to inevitable failures.

- **Legal Complications in Voice Cloning**: Users discussed the legal and ethical implications of voice cloning, especially in light of Scarlett Johansson's concerns. One user criticized reliance on legislation for protecting likeness, highlighting limitations in enforcement and existing open-source voice cloning technologies.

- **Qualcomm Snapdragon Dev Kit Launch**: A member shared enthusiasm for Qualcomm’s new $899.99 [Snapdragon Dev Kit for Windows](https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-elite). This dev kit offers significant power with its 4.6 TFLOP GPU, 32GB RAM, and 512GB storage, packaged similarly to Apple’s mini desktop.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-elite">Here’s the eight-inch Snapdragon PC for your Windows on Arm experiments</a>: Qualcomm is selling it in black.</li><li><a href="https://x.com/BobbyAllyn/status/1792679435701014908">Tweet from Bobby Allyn (@BobbyAllyn)</a>: Statement from Scarlett Johansson on the OpenAI situation. Wow:
</li>
</ul>

</div>
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1242451432444006400)** (6 messages): 

- **Supervised Fine-Tuning (SFT) vs Preference Optimization**: A member asked for clarification on the differences between **Supervised Fine-Tuning (SFT)** and **Preference Optimization**. They suggested that while SFT pushes up the probability distribution of the SFT dataset, preference optimization adjusts both undesired and desired probabilities, questioning why SFT is necessary.
  
- **Phi3 Vision impresses with its efficiency**: A member expressed their admiration for **Phi3 Vision**, a 4.2 billion parameter model, praising its performance in low-latency/live inference on image streams. They shared a [post on X](https://x.com/jphme/status/1792950682695479734) discussing the potential applications in robotics.
  
- **Comparing Phi3 Vision and Moondream2**: Another member encouraged the use of [Moondream2](https://huggingface.co/spaces/vikhyatk/moondream2) on the same image as Phi3 Vision to compare results. Feedback indicated that Moondream2 performs well and has reduced hallucinations, though some datasets remain problematic.
  
- **Microsoft releases new models**: **Microsoft released 7 billion and 14 billion parameter models**. Notably, only the instruct versions are available, as observed by a community member.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/vikhyatk/moondream2">moondream2 - a Hugging Face Space by vikhyatk</a>: no description found</li><li><a href="https://x.com/jphme/status/1792950682695479734">Tweet from Jan P. Harries (@jphme)</a>: Phi3 vision was just released - it is just 4.2b params and extremely impressive. 🤩  I feel this is a breakthrough for low-latency/live inference on image streams - just imagine what even smaller/more...
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1242187117501743406)** (2 messages): 

- **Alex introduces `sqlite-vec` to the community**: Alex shared his new project [`sqlite-vec`](https://github.com/asg017/sqlite-vec), a SQLite extension for vector search, and mentioned it might be integrated with Llamafile for features like RAG, memory, semantic search, etc. *"It's written entirely in C and should work with cosmopolitan, though haven't tested myself yet."*
  
- **Detailed project description**: Alex provided a [detailed blog post](https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html) explaining the potential and progress of `sqlite-vec`, which aims to replace `sqlite-vss` and offer a more performant and embeddable solution. The extension is still in beta but available for early trials, with distributions in C/C++ projects and packages on pip/npm/gem platforms.

- **Open for collaboration and support**: Alex expressed his willingness to support and help anyone get started with `sqlite-vec` and address any issues users might encounter during the beta phase. "More coming soon, but happy to help anyone here get started or get around any issues!"

- **Community excitement**: A member welcomed Alex and expressed enthusiasm about the project's potential integrations with Llamafile. *"super excited about your project, and about the possibilities presented by integrating it with llamafile."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/asg017/sqlite-vec">GitHub - asg017/sqlite-vec: Work-in-progress vector search SQLite extension that runs anywhere.</a>: Work-in-progress vector search SQLite extension that runs anywhere. - asg017/sqlite-vec</li><li><a href="https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html">I'm writing a new vector search SQLite Extension</a>: sqlite-vec is an new vector search SQLite extension, coming soon!</li><li><a href="https://github.com/asg017/sqlite-vec/releases">Releases · asg017/sqlite-vec</a>: Work-in-progress vector search SQLite extension that runs anywhere. - asg017/sqlite-vec
</li>
</ul>

</div>
  

---



### **LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1242201665835237448)** (1 messages): 

- **GPT-4o Excels in Legal Reasoning**: A member shared their experience running internal evaluation tests on **GPT-4o** for complex legal reasoning tasks. They reported a *"non-trivial improvement"* from GPT-4 and GPT-4-Turbo, and linked a [LinkedIn post](https://www.linkedin.com/posts/evan-harris-387375b2_the-release-of-gpt-4o-from-openai-has-been-activity-7196856963454959617-w1i1) about the release of GPT-4o.
  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1242233137828728903)** (1 messages): 

- **Manifold Research Group seeks collaborators**: A representative from Manifold Research Group introduced their OS R&D Lab focused on *generalist models* and AI Agents. They invited interested individuals to [learn more](https://www.manifoldrg.com/research-log-038/) or join the team through their [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) and [Github](https://github.com/ManifoldRG?ref=manifoldrg.com).
- **NEKO Project aims high with open-source generalist models**: The NEKO Project is building the first large-scale, open-source generalist model trained on various modalities, including control and robotics tasks. More information can be found in their detailed [project document](https://docs.google.com/document/d/e/2PACX-1vQELDXCIT9tn7Uq5vxQG4_3HsrkQcuBRqvXm-MkxW06Zkh-LP3G9z7TP7a-2MNWyA/pub?ref=manifoldrg.com).

**Link mentioned**: <a href="https://www.manifoldrg.com/research-log-038/">Research log #038</a>: Welcome to Research Log #038! We document weekly research progress across the various initiatives in the Manifold Research Group, and highlight breakthroughs from the broader research community we thi...

  
---




{% else %}

> The full channel by channel breakdowns are now truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!

If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

---
id: bb0bd562-3bbf-4479-a6a1-0e1ef2c16d71
title: 'Kolmogorov-Arnold Networks: MLP killers or just spicy MLPs?'
date: '2024-05-07T22:47:14.398752Z'
original_slug: ainews-kolmogorov-arnold-networks-mlp-killers-or
description: >-
  A recent theory paper on **Kolmogorov-Arnold Networks (KANs)** by Max
  Tegmark’s grad student **Ziming Liu** sparked excitement for potentially
  outperforming MLPs in interpretability and efficiency, but was quickly
  challenged as equivalent to MLPs after a week, illustrating rapid academic
  debate. Meanwhile, **Microsoft** is developing a massive *500B* parameter
  model called **MAI-1** to compete with **OpenAI** and **Google**, signaling
  intensifying competition in large language models, alongside positive
  reception for Microsoft’s **Copilot Workspace** and OpenAI’s adoption of the
  **C2PA** standard for AI image provenance.
tags:
  - openai
  - microsoft
  - google
  - anthropic
  - alembic
  - kan
  - mlp
  - gpt-2
  - gpt-4
  - gpt-4.5
  - gpt-5
  - mai-1
  - dall-e-3
  - deeps-eek-v2
  - llama-3
  - large-language-models
  - model-interpretability
  - model-efficiency
  - safety-testing
  - image-provenance
  - mixture-of-experts
  - scaling-laws
  - hallucination-mitigation
  - medical-ai
  - developer-tools
  - emergent-abilities
companies:
  - openai
  - microsoft
  - google
  - anthropic
  - alembic
models:
  - kan
  - mlp
  - gpt-2
  - gpt-4
  - gpt-4.5
  - gpt-5
  - mai-1
  - dall-e-3
  - deeps-eek-v2
  - llama-3
topics:
  - large-language-models
  - model-interpretability
  - model-efficiency
  - safety-testing
  - image-provenance
  - mixture-of-experts
  - scaling-laws
  - hallucination-mitigation
  - medical-ai
  - developer-tools
  - emergent-abilities
---


<!-- buttondown-editor-mode: plaintext -->**Are Learnable Activations all you need?**

> AI News for 5/6/2024-5/7/2024. We checked 7 subreddits and [**373** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**419** channels, and **3749** messages) for you. Estimated reading time saved (at 200wpm): **414 minutes**.

Theory papers are usually above our paygrade, but that is enough drama and not enough else going on today that we have the space to write about it. A week ago, Max Tegmark's grad student [Ziming Liu published](https://twitter.com/ZimingLiu11/status/1785483967719981538) his very well written [paper on KANs](https://arxiv.org/abs/2404.19756) (complete with [fully documented library](https://kindxiaoming.github.io/pykan/)), claiming them as almost universally equal to or superior to MLPs on many important dimensions like [interpretability](https://x.com/ZimingLiu11/status/1785489972218433627)/[inductive bias injection](https://x.com/ZimingLiu11/status/1785490122303287346), [function approximation accuracy](https://x.com/ZimingLiu11/status/1785489587122601997) and scaling (though is acknowledged to be currently [10x slower to train](https://www.reddit.com/r/MachineLearning/comments/1chrafb/comment/l24eymi/) on current hardware on a same-param count basis, it is also [100x more param efficient](https://www.reddit.com/r/MachineLearning/comments/1chrafb/comment/l24fp48/)).

 ![image.png](https://assets.buttondown.email/images/f3be6c78-f460-49bc-b7be-b0a4fce9f5d8.png?w=960&fit=max) 

> While MLPs have fixed activation functions on nodes ("neurons"), KANs have learnable activation functions on edges ("weights").

Instead of layering preset activations like ReLu, KANs model "learnable activation functions" using [B-splines](https://ocw.mit.edu/courses/18-085-computational-science-and-engineering-i-fall-2008/resources/lecture-21-boundary-conditions-splines-gradient-divergence/) (aka no linear weights, just curves) and [simple addition](https://twitter.com/predict_addict/status/1785696131436998681). People got excited, [rewriting GPTs with KANs](https://x.com/predict_addict/status/1787853844862378463).

One week on, it now turns out that [you can rearrange the KAN terms to arrive back at MLPs with the ~same number of params](https://www.reddit.com/r/MachineLearning/comments/1clcu5i/d_kolmogorovarnold_network_is_just_an_mlp/) ([twitter](https://twitter.com/bozavlado/status/1787376558484709691)):

 ![image.png](https://assets.buttondown.email/images/99ec5339-40b6-45e4-a8e3-e3ef52a2078b.png?w=960&fit=max) 

It doesn't surprise that you can rewrite one universal approximator as another - but following this very simple publication, many are defending KANs as more interpretable... which is also [being rightfully challenged](https://x.com/FreeFooooooood/status/1787403148442718696).

Have we seen the full rise and fall of a new theory paper in a single week? Is this the preprint system working?

---

**Table of Contents**

[TOC] 



---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**OpenAI and GPT Models**

- **Potential GPT-5 Release**: [@bindureddy](https://twitter.com/bindureddy/status/1787844680182555052) noted that the gpt-2 chatbots are back on chat.lmsys and may be the latest GPT-5 versions, though they seem underwhelming compared to the hype. [@nptacek](https://twitter.com/nptacek/status/1787798590741458976) tested the im-a-good-gpt2-chatbot model, finding it very strong and **definitely better than the latest GPT-4**, while the im-also-a-good-gpt2-chatbot had fast output but tended to fall into repetitive loops.
- **OpenAI Safety Testing**: [@zacharynado](https://twitter.com/zacharynado/status/1787864594553184427) speculated that OpenAI's "safety testing" for GPT-4.5 couldn't finish in time for a Google I/O launch like they did with GPT-4.
- **Detecting AI-Generated Images**: OpenAI adopted the C2PA metadata standard for certifying the origin of AI-generated images and videos, which is integrated into products like DALL-E 3. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1787853748682805631) noted the classifier can identify **~98% of DALL-E 3 images while incorrectly flagging <0.5% of non-AI images**, but has lower performance distinguishing DALL-E 3 from other AI-generated images.

**Microsoft AI Developments**

- **In-House LLM Training**: According to [@bindureddy](https://twitter.com/bindureddy/status/1787498838024139185), Microsoft is training its own 500B parameter model called MAI-1, which may be previewed at the Build conference. As the model becomes available, it will be natural for Microsoft to push it instead of OpenAI's GPT line, making the two companies more competitive.
- **Copilot Workspace Impressions**: [@svpino](https://twitter.com/svpino/status/1787893785814249780) had very positive first impressions of Copilot Workspace, noting its refined approach and tight integration with GitHub for **generating code directly in repositories, solving issues, and testing**. The tool is positioned as an aid to developers rather than a replacement.
- **Microsoft's AI Focus**: [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1787580620694077639), having joined Microsoft, shared that the company is AI-first and driving massive technological transformation, with **responsible AI as a cornerstone**. Teams are working to define new norms and build products for positive impact.

**Other LLM Developments**

- **Anthropic's Approach**: In an interview discussed by [@labenz](https://twitter.com/labenz/status/1787580822192574845), Anthropic's CTO explained their approach of giving the AI many examples rather than fine-tuning for every task, as fine-tuning fundamentally narrows what the system can do.
- **DeepSeek-V2 Release**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1787478986731429933) announced the release of DeepSeek-V2, an open-source 236B parameter MoE model that **places top 3 in AlignBench, surpassing GPT-4**, and ranks highly in MT-Bench, rivaling LLaMA3-70B. It specializes in math, code, and reasoning with a 128K context window.
- **Llama-3 Developments**: [@abacaj](https://twitter.com/abacaj/status/1787527652628140321) suggested Llama-3 with multimodal capabilities and long context could put pressure on OpenAI. [@bindureddy](https://twitter.com/bindureddy/status/1787641041304363061) noted Llama-3 on Groq allows efficiently making multiple serial calls for LLM apps to make multiple decisions before giving the right answer, which is difficult with GPT-4.

**AI Benchmarks and Evaluations**

- **LLMs as a Commodity**: [@bindureddy](https://twitter.com/bindureddy/status/1787507453023994251) argued that LLMs have become a commodity, and even if GPT-5 is fantastic, other major labs and companies will catch up within months as language abilities plateau. He advises using **LLM-agnostic services for the best performance and efficiency**.
- **Evaluating LLM Outputs**: [@aleks_madry](https://twitter.com/aleks_madry/status/1787564287986389277) introduced ContextCite, a method for attributing LLM responses back to the given context to see how the model is using the information and if it's misinterpreting anything or hallucinating. It can be applied to any LLM at the cost of a few extra inference calls.
- **Emergent Abilities of LLMs**: [@raphaelmilliere](https://twitter.com/raphaelmilliere/status/1787845213224067499) shared a preprint exploring philosophical questions around LLMs, covering topics like **emergent abilities, consciousness, and the status of LLMs as cognitive models**. The paper dedicates a large portion to recent interpretability research and causal intervention methods.

**Scaling Laws and Architectures** 

- **Scaling Laws for MoEs**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1787625359199911999) noted that DeepSeek-V2-236B took 1.4M H800-hours to train compared to Llama-3-8B's 1.3M H100-hours, validating the Scaling Laws for Fine-Grained MoEs paper. DeepSeek openly shares inference unit economics in contrast to some Western frontier companies.
- **Benefits of MoE Models**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1787647666286916053) highlighted DeepSeek's architectural innovations in **attention mechanisms (Multi-head Latent Attention for efficient inference) and sparse layers (DeepSeekMoE for training strong models economically)**, contrasting with the "scale is all you need" mindset of some other labs.
- **Mixture-of-Experts Efficiency**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1787828124475928748) pointed out that at 1M context, a ~250B parameter MLA model like DeepSeek-V2 uses only 34.6GB for cache, suggesting that saving long-context examples as an alternative to fine-tuning is becoming more feasible.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Progress and Capabilities**

- **Google's medical AI outperforms GPT and doctors**: In /r/singularity, Google's Med-PaLM 2 AI [**destroys GPT's benchmark and outperforms doctors on medical diagnosis tasks**](https://newatlas.com/technology/google-med-gemini-ai/). This highlights the rapid progress of AI in specialized domains like healthcare.
- **Microsoft developing large language model to compete**: In /r/artificial, it's reported that [**Microsoft is working on a 500B parameter model called MAI-1**](https://www.theinformation.com/articles/meet-mai-1-microsoft-readies-new-ai-model-to-compete-with-google-openai) to compete with offerings from Google and OpenAI. The race to develop ever-larger foundational models continues.
- **AI system claims to eliminate "hallucinations"**: In /r/artificial, Alembic claims to have [**developed an AI that eliminates "hallucinations" and false information generation**](https://venturebeat.com/ai/exclusive-alembic-debuts-hallucination-free-ai-for-enterprise-data-analysis-and-decision-support/) in outputs. If true, this could be a major step towards more reliable AI systems.

**AI Ethics and Societal Impact**

- **Viral AI generated misinformation**: In /r/singularity, an [**AI generated photo of Katy Perry at the Met Gala went viral**](https://i.redd.it/x880wms1hwyc1.jpeg), gaining over 200k likes in under 2 hours. This demonstrates the potential for AI to rapidly spread misinformation at scale.
- **Prominent AI critic's credibility questioned**: In /r/singularity, it's revealed that [**Gary Marcus, a prominent AI critic, admits he doesn't actually use the large language models he criticizes**](https://i.redd.it/vpgrmnytrryc1.png), drawing skepticism about his understanding of the technology.
- **Concerns over AI scams and fraud**: In /r/artificial, Warren Buffett [**predicts AI scamming and fraud will be the next big "growth industry"**](https://www.cnbc.com/2024/05/04/warren-buffett-says-ai-scamming-will-be-the-next-big-growth-industry.html) as the technology advances, highlighting concerns over malicious uses of AI.

**Technical Developments**

- **New neural network architecture analyzed**: In /r/MachineLearning, the [**Kolmogorov-Arnold Network is shown to be equivalent to a standard MLP**](https://colab.research.google.com/drive/1v3AHz5J3gk-vu4biESubJdOsUheycJNz) with some modifications, providing new insights into neural network design.
- **Efficient large language model developed**: In /r/MachineLearning, [**DeepSeek-V2, a 236B parameter Mixture-of-Experts model, achieves strong performance while reducing costs**](https://github.com/deepseek-ai/DeepSeek-V2) compared to dense models, advancing more efficient architectures.
- **New library for robotics and embodied AI**: In /r/artificial, Hugging Face releases [**LeRobot, a library for deep learning robotics**](https://github.com/huggingface/lerobot), to enable real-world AI applications and advance embodied AI research.

**Stable Diffusion and Image Generation**  

- **Stable Diffusion 3.0 shows major improvements**: In /r/StableDiffusion, [**Stable Diffusion 3.0 demonstrates major improvements in image quality and prompt adherence**](https://www.reddit.com/gallery/1clsic3) compared to previous versions and competitors.
- **Efficient model matches Stable Diffusion 3.0 performance**: In /r/StableDiffusion, the [**PixArt Sigma model shows excellent prompt adherence, on par with SD3.0 while being more efficient**](https://www.reddit.com/gallery/1clf240), providing a compelling alternative.
- **New model for realistic light painting effects**: In /r/StableDiffusion, a [**new "Aether Light" LoRA model enables realistic light painting effects**](https://civitai.com/models/410151/aether-light-lora-for-sdxl) in Stable Diffusion, expanding creative possibilities for artists.

**Humor and Memes**

- **Humorous AI chatbot emerges**: In /r/singularity, an ["**im-a-good-gpt2-chatbot" model appears on OpenAI's Playground and engages in humorous conversations**](https://i.redd.it/yzdibltw6yyc1.png) with users, showcasing the lighter side of AI development.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Model Performance Optimization and Benchmarking**

- **[Quantization]** techniques like **AQLM** and **QuaRot** aim to run large language models (**LLMs**) on individual GPUs while maintaining performance. Example: [AQLM project](https://github.com/Vahe1994/AQLM) with **Llama-3-70b** running on RTX3090.

- Efforts to **boost transformer efficiency** through methods like **Dynamic Memory Compression (DMC)**, potentially improving throughput by up to 370% on **H100 GPUs**. Example: [DMC paper](https://arxiv.org/abs/2403.09636) by @p_nawrot.

- Discussions on **optimizing CUDA operations** like fusing element-wise operations, using **Thrust library's `transform`** for near-bandwidth-saturating performance. Example: [Thrust documentation](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each).

- Comparisons of **model performance** across benchmarks like **AlignBench** and **MT-Bench**, with **DeepSeek-V2** surpassing GPT-4 in some areas. Example: [DeepSeek-V2 announcement](https://x.com/deepseek_ai/status/1787478986731429933).

**2. Fine-tuning Challenges and Prompt Engineering Strategies**

- Difficulties in **retaining fine-tuned data** when converting **Llama3** models to GGUF format, with a [confirmed bug](https://github.com/ggerganov/llama.cpp/issues/7062) discussed.

- Importance of **prompt design** and usage of correct templates, including end-of-text tokens, for influencing model performance during fine-tuning and evaluation. Example: [Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47).

- Strategies for **prompt engineering** like splitting complex tasks into multiple prompts, investigating **logit bias** for more control. Example: [OpenAI logit bias guide](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api).

- Teaching LLMs to use `<RET>` token for **information retrieval** when uncertain, improving performance on infrequent queries. Example: [ArXiv paper](https://arxiv.org/abs/2404.19705).

**3. Open-Source AI Developments and Collaborations**

- Launch of **StoryDiffusion**, an open-source alternative to Sora with MIT license, though weights not released yet. Example: [GitHub repo](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file). 

- Release of **OpenDevin**, an open-source autonomous AI engineer based on Devin by Cognition, with [webinar](https://lu.ma/fp0xr460) and growing interest on GitHub.

- Calls for collaboration on open-source **machine learning paper** predicting IPO success, hosted at [RicercaMente](https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html).

- Community efforts around **LlamaIndex** integration, with issues faced in Supabase Vectorstore and package imports after updates. Example: [llama-hub documentation](https://github.com/run-llama/llama-hub/tree/main#how-to-add-a-loadertoolllama-pack).

**4. Hardware Considerations for Efficient AI Workloads**

- Discussions on **GPU power consumption**, with insights on P40 GPUs idling at 10W but drawing 200W total, and strategies to limit to 140W for 85% performance. 

- Evaluating **PCI-E bandwidth** requirements for inference tasks, often overestimated based on shared resources. Example: [Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/15yknoo/is_inference_reliant_on_pcie_bandwidth/).

- Exploring **single-threaded operations** in frameworks like **tinygrad**, which doesn't use multi-threading for CPU ops like matrix multiplication.

- Inquiries into **Metal memory allocation** on Apple Silicon GPUs for shared/global memory akin to CUDA's `__shared__`.

**5. Misc**

- **Exploring Capabilities and Limitations of AI Models**: Engineers compared the performance of various models like **Llama 3 70b**, **Mistral 8x22b**, **GPT-4 Turbo**, and **Sonar** for tasks such as function calling, essay writing, and code refactoring ([Nous Research AI](https://discord.com/channels/1053877538025386074/1149866623109439599)). They also discussed the impact of quantization on model performance, like the degradation seen with `llama.cpp`'s quantization ([OpenAccess AI Collective](https://discord.com/channels/1104757954588196865/1112023441386778704)).

- **Prompt Engineering Techniques and Challenges**: The importance of prompt design was emphasized, noting how changes to templates and tokens can significantly impact model performance ([OpenAccess AI Collective](https://discord.com/channels/1104757954588196865/1112023441386778704)). Users shared tips for complex prompting tasks, like splitting product identification into multiple prompts ([OpenAI](https://discord.com/channels/974519864045756446/1046317269069864970)), and discussed the difficulty of integrating negative examples ([OpenAI](https://discord.com/channels/974519864045756446/1046317269069864970)). A new prompt generator tool from Anthropic was also explored ([LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851/1179271229593624677)).

- **Optimizing CUDA Operations and Model Training**: CUDA developers shared insights on using **Triton** for efficient kernel design, leveraging **fused operations** for element-wise computations, and utilizing CUDA's Thrust library for optimal performance ([CUDA MODE](https://discord.com/channels/1189498204333543425/1189607726595194971)). Discussions also covered techniques to boost transformer efficiency, like **Dynamic Memory Compression (DMC)** ([CUDA MODE](https://discord.com/channels/1189498204333543425/1189861061151690822)), and fine-tuning CUDA kernels for specific architectures ([CUDA MODE](https://discord.com/channels/1189498204333543425/1227345713348870156)).

- **Advancements in Open-Source AI Projects**: Notable open-source releases included **DeepSeek-V2**, an MoE model excelling in code and reasoning benchmarks ([Latent Space](https://discord.com/channels/822583790773862470/1075282825051385876)), **StoryDiffusion** for magic story creation ([OpenAccess AI Collective](https://discord.com/channels/1104757954588196865/1104757955204743201)), and **OpenDevin**, an autonomous AI engineer ([LlamaIndex](https://discord.com/channels/1059199217496772688/1187460979064324127)). Collaborations were encouraged, like the [open call to contribute to a machine learning IPO prediction paper](https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html) ([Unsloth AI](https://discord.com/channels/1179035537009545276/1179779344894263297)).

---



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GGUF Conversion Conundrum**: Engineers noted a bug in converting **Llama3 models to GGUF** which affects retention of fine-tuned data, spurring discussions on related GitHub [Issue #7062](https://github.com/ggerganov/llama.cpp/issues/7062) and a Reddit thread.
- **Training Tips and Troubles**: Model training was a hot topic, including *tokenization issues* during fine-tuning and successful utilization of **LORA Adapters** with Unsloth. Additionally, concerns about base data impacting fine-tuned model results were addressed, suggesting fine-tuning alters weights for previously seen tokens.
- **AI Engine Struggles and Strategies**: **Aphrodite Engine's** compatibility with 4bit bnb quantization raised questions, while a VRAM calculator for models like GGUF and exllama was referenced for inference program compatibility. Separately, a member highlighted a need for a *generalised approach* to fine-tuning **Large Vision Language Models (LVLM)**.
- **Model Iterations and Collaborations**: New versions of models were unveiled, including [LLaMA-3-8B-RDF-Experiment](https://huggingface.co/M-Chimiste/Llama-3-8B-RDF-Experiment) for knowledge graph construction, while an [open call](https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html) was made for collaboration on a machine learning paper predicting IPO success.
- **Product Showcase and Support Requests**: The introduction of **Oncord**, a professional website builder, was met with a demo at [oncord.com](https://www.oncord.com/), and members debated marketing tactics for startups. Additionally, support for **moondream** fine-tuning was requested, linking a GitHub [notebook](https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Function Calling Face-off**: **Llama 3 70b** shows better function calling performance over **Mistral 8x22b**, revealing a gap despite the latter's touted capabilities, exemplified by the members' discussion around the utility and accuracy of function calling in AI chatbots.

**A Battle of Speeds in AI Training**: Comparing training times leads to concerns, with reports of 500 seconds per step on an A100 for LoRA llama 3 8b tuning and just 3 minutes for 1,000 iterations for Llama2 7B using litgpt, showing wide variances in efficiency and raising questions on optimization and practices.

**Impatience for Improvements**: Users express disappointment over inaccessible features such as [worldsim.nousresearch.com](https://worldsim.nousresearch.com), and latency in critical updates for networks like **Bittensor**, highlighting real-time challenges faced by developers in AI and the ripple effects of stalled updates on productivity.

**Quantization Leaps Forward**: The [AQLM project](https://github.com/Vahe1994/AQLM) advances with models like **Llama-3-70b** and **Command-R+**, demonstrating progress with running Large Language Models on individual GPUs and touching upon the community's push for greater model accessibility and performance.

**Chasing Trustworthy AI**: Invetech's "Deterministic Quoting" to combat hallucinations indicates a strong community desire for reliable AI, particularly in sensitive sectors like healthcare, aiming to marry veracity with the innovative potential of Large Language Models as seen in the [discussion](https://mattyyeung.github.io/deterministic-quoting).



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Hyper Diffusion vs Stable Diffusion 3 Demystified**: Engineers tackled the nuances between **Hyper Stable Diffusion**, known for its speed, and the upcoming **Stable Diffusion 3**. The community expressed concern over the latter potentially not being open-source, prompting discussions on the strategic safeguarding of AI models.

- **Bias Alert in Realistic Human Renders**: The quest for the most effective **realistic human model** stimulated debate, with a consensus forming on the necessity of avoiding models with heavy biases like those from **civitai** to maintain diversity in generated outputs.

- **Dreambooth and LoRA Deep Dive**: Deep technical consultation amongst users shed light on leveraging **Dreambooth** and **LoRA** when fine-tuning **Stable Diffusion** models. There was a particular focus on generating unique and varied faces and styles.

- **The Upscaling Showdown**: Participants compared upscalers, such as **RealESRGAN_x4plus** and **4xUltrasharp**, sharing their personal successes and preferences. The conversations aimed to identify superior upscaling techniques for enhanced image resolution.

- **Open-Source AI Twilight?**: A recurrent theme in the dialogues reflected the community's anxiety about the future of open-source AI, particularly related to **Stable Diffusion** models. Talk revolved around the implications of proprietary developments and strategies for preserving access to crucial AI assets.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Private Life for Your Code**: Users call for a **server logging off feature in LM Studio** for privacy during development, with genuine concerns about server logs being collected through the GUI.

**A Day in the CLI**: There's interest in using **LM Studio in headless mode** and leveraging the **[lms CLI](https://github.com/lmstudio-ai/lms)** to start servers via the command line. Users also shared updates on tokenizer complications for Command R and Command R+ after a llama toolkit update and issued guidance for downloading updated quantizations from [Hugging Face Co's Model Repo](https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF).

**Memory Lapses in Linux**: A peculiar case of **Linux misreporting memory** in LM Studio version 0.2.22 stirred some discussions, with suggestions offered to resolve GPU offloading troubles for running models like **Meta Llama 3 instruct 7B**.

**Prompts Lost in Thought**: Users tackled issues around **LM Studio erroneously responding to deleted content** and scoped document access, sparking a debate about LLMs' handling and retention of data.

**Model Malfunctions**: Troubles with several models in LM Studio were flagged, including **llava-phi-3-mini** misrecognizing images and models like **Mixtral** and **Wizard LM** fumbling Dungeon & Dragons data persistence despite *AnythingLLM database* use.

**Power-play Considerations**: Hardware aficionados in the guild grapple with **GPU power consumption, server motherboards, and PCIe bandwidth**, sharing successful runs of LM Studio in VMs with virtual GPUs and weighing in on practical hardware setups for AI endeavors.

**Beta-testing Blues**: Discussions mentioned crashes in **7B models on 8GB GPUs** and unloading issues post-crash, with beta users seeking solutions for recurring errors.

**SDK Advent**: Announcement of **new lmstudiojs SDK** signals upcoming **langchain integrations** for more streamlined tool development.

**In the AI Trenches**: Users provided a solution for dependency package installation on Linux, discussed LM Studio's compatibility on Ubuntu 22.04 vs. 24.04, and shared challenges with LM Studio's API integration and concurrent request handling.

**Engineer Inquiry**: Curiosity peaked about **GPT-Engineer** setup with LM Studio and whether it involved custom prompting techniques.

**Prompting the AIs**: Some voiced the value of **prompt engineering as a craft**, citing it as central to garnering premium outputs from LLMs and sharing a win in Singapore’s GPT-4 Prompt Engineering Competition covered in [Towards Data Science](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41).

**AutoGen Hiccups**: There's a brief mention of a bug causing **AutoGen Studio** to send incomplete messages, with no further discussion on the resolution or cause.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**ASR Fine-Tuning Takes Center Stage**: Engineers discussed enhancing the **openai/whisper-small** ASR model, emphasizing *dataset size* and *hyperparameter tuning*. Tips included adjusting `weight_decay` and `learning_rate` to improve training, highlighted by community-shared resources on hyperparameters like gradient accumulation steps and learning rate adjustments.

**Deep Dive into Quantum and AI Tools**: Stealthy interest in seemingly nascent **quantum virtual servers** surfaced with [Oqtant](https://oqtant.infleqtion.com/), while the AI toolkit included everything from an all-in-one assistant [everything-ai](https://github.com/AstraBert/everything-ai) capable of 50+ language support to the spaghetti-coded image-generating discord bot [Sparky 2](https://github.com/Gapi505/Sparky-2).

**Debugging and Datasets**: Chatbots designing **PowerPoint slides**, XLM-R getting a **Flash Attention 2** upgrade, and multi-label image classification training woes took the stage, connecting community members across problems and sharing valuable insights. Meanwhile, the lost **UA-DETRAC dataset** incited a search for its much-needed annotations for traffic camera-based object detection.

**Customization and Challenges in Model Training**: From personalizing image models with **Custom Diffusion**—requiring minimal example images—to the struggles with fine-tuning **Stable Diffusion 1.5** and BERT models, the community wrestled with and brainstormed solutions for various training hiccups. Device mismatches during multi-GPU and CPU offloading and the importance of optimization techniques for restricted resources were notable pain points.

**Novel Approaches in Teaching Retrieval to LLMs**: A newer technique encouraging **LLMs** to use a `<RET>` token for information retrieval to boost performance was discussed with reference to a [recent paper](https://arxiv.org/abs/2404.19705), highlighting the importance of this method for elusive questions that evade the model's memory. This sits alongside observations on model billing methods via token counts, with practical insights shared on [pricing strategies](https://docsbot.ai/tools/gpt-openai-api-pricing-calculator).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Beta Bewilderment**: Users experienced confusion with accessing **Perplexity AI's** beta version; one assumed clicking an icon would reveal a form, which didn't happen, and it was clarified that the beta is closed.

**Performance Puzzles**: Across different devices, **Perplexity AI** users reported technical issues such as unresponsive buttons and sluggish loading. Conversations revolved around limits of models like **Claude 3 Opus** and **Sonar 32k**, effecting work, with calls to check **Perplexity's FAQ** for details.

**AI Model Melee**: Comparisons of AI models' capabilities, including **GPT-4 Turbo**, **Sonar**, and **Opus**, were discussed, focusing on tasks like essay writing and code refactoring. Clarity was sought on whether source limits in searches had increased, with GIFs used to illustrate responses.

**API Angst and Insights**: Discussions in the **Perplexity API channel** ranged from crafting JSON outputs to perplexities with the search features of Perplexity's online models. The **documentation** was updated (as highlighted in a [link to docs](https://docs.perplexity.ai/docs/model-cards)), important for users dealing with issues like outdated search results and exploring model parameter counts.

**Shared Discoveries through Perplexity**: The community delved into **Perplexity AI's** offerings, addressing an array of topics from **US Air Force** insights to **Microsoft's 500 billion parameter AI model**. Users shared an aspiration for a standardized image creation UI along with links to features like *Insanity by XDream* and emphasized content's shareability.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**GPU Clock Speed Mix-Up**: A conversation was sparked by confusion over the clock speed of **H100 GPUs**, with the initial statement of 1.8 MHz corrected to 1.8 GHz. This highlighted the need to distinguish MHz from GHz and the importance of accurate specifications in discussions on GPU performance.

**Tuning CUDA: From Kernels to Libraries**: Members shared insights on optimizing CUDA operations, emphasizing the **efficiency of Triton** in kernel design, the advantage of **fused operations** in element-wise computations, and the use of CUDA's Thrust library. A CUDA best practice is to use [Thrust's `for_each`](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each) and `transform` for near-bandwidth-saturating performance.

**PyTorch Dynamics**: Various issues and improvements in PyTorch were discussed, including troubleshooting **dynamic shapes with PyTorch Compile** using `TORCH_LOGS="+dynamic"` and how to work with `torch.compile` for the Triton backend. An issue reported on **PyTorch's GitHub** relates to combining Compile with DDP & dynamic shapes, captured in [pytorch/pytorch #125641](https://github.com/pytorch/pytorch/issues/125641).

**Transformer Performance Innovations**: Conversations revolved around techniques to boost the efficiency of transformers, with the introduction of **Dynamic Memory Compression (DMC)** by a community member, potentially improving throughput by up to 370% on H100 GPUs. Members also discussed whether quantization was involved in this method, with reference to the [paper](https://arxiv.org/abs/2403.09636) on the technique.

**CUDA Discussions Heat Up in llm.c**: The llm.c channel was bustling with activity, addressing issues such as multi-GPU training hangs on the master branch and optimization opportunities using NVIDIA Nsight™ Systems. A notable contribution is HuggingFace's release of the **FineWeb** dataset for **LLM performance**, documented in [PR #369](https://github.com/karpathy/llm.c/issues/369), with potential kernel optimizations for performance gains discussed in [PR #307](https://github.com/karpathy/llm.c/pull/307).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Linguistically Defines Its Data Commandments**: [OpenAI's new document on data handling](https://openai.com/index/approach-to-data-and-ai) clarifies the organization's practices and ethical guidelines for processing the copious amounts of data in the AI industry.

- **AI's Rhythmic Revolution Might Be Here**: The discussion centered around the evolution of AI in music, referencing a [musician's jam session with AI](http://drinkoblog.weebly.com/) as an example of significant advancements in AI’s ability to generate music that resonates with human listeners.

- **Perplexity and Cosine Similarity Stir Engineer's Minds**: Engineers marveled at discovering the utility of *Perplexity* in AI text analysis and debated the optimal cosine similarity thresholds for text embeddings, highlighting the shift to a "new 0.45" standard from the "old 0.9".

- **Prompting Practices and Pitfalls in the Spotlight**: Tips on prompt engineering emphasized the complexity of using negative examples and splitting tasks into multiple prompts, and pointed to the [OpenAI logit bias guide](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api) for fine-tuning AI responses.

- **GPT's Vector Vault and Uniform Delivery Assurances**: Insights into GPT's knowledge base mechanics and performance consistency were shared, dispelling the notion that varying user demand affects GPT-4 output or that inferior models may be deployed to manage user load.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Questioning the Sacred P-Value**: Discussions highlighted the arbitrary nature of the **.05 p-value threshold** in scientific research, pointing toward a movement to shift this standard to 0.005 to enhance reproducibility, as advocated in a [Nature article](https://www.nature.com/articles/s41562-017-0189-z).

- **Pushing Boundaries with Skip Connections**: **Adaptive skip connections** are under investigation with some evidence that making weights negative can improve model performance; details of these experiments can be found on [Weights & Biases](https://wandb.ai/nickcdryan/bench/reports/train_loss-24-05-07-01-06-58---Vmlldzo3ODM5Njcx). Queries related to the underlying mechanics of weight dynamics were responded to with a [gated residual network paper](https://arxiv.org/abs/2003.04887) and a [code snippet](https://gist.github.com/nickcdryan/08c059ec3deb3ef2aca881bdc4409631).

- **Model Evaluation in a Logit-Locked World**: The concealment of logits in API models like OpenAI's to prevent extraction of sensitive "signatures" has sparked conversations about alternatives for model evaluation, referencing the approach with 'generate_until' in YAML for Italian LLM comparisons, in light of recent findings ([logit extraction work](https://arxiv.org/abs/2403.09539)).

- **Encounter with Non-Fine-Tunable Learning**: Introduction of **SOPHON**, a framework designed for non-fine-tunable learning to restrict task transferability, aims to mitigate ethical misuse of pre-trained models ([SOPHON paper](https://arxiv.org/abs/2404.12699)). Alongside this, there's an emerging discussion about **QuaRot**, a rotation-based quantization scheme that compresses LLM components to 4-bit while maintaining performance ([QuaRot paper](https://arxiv.org/abs/2404.00456)).

- **Scaling and Loss Curve Puzzles**: A noteworthy model scaling experiment using a 607M parameter setup trained on the [fineweb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb) unearthed unusual loss curves, initiating advice to try the experiment on other datasets for benchmarking.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Exploring "Mojo" with Boundless Coding Adventures**: Engineers discussed intricacies of programming in **mojo**, including installing on **Intel Mac OS** using **Docker**, **Windows** support through **WSL2**, and integration with **Python** ecosystems. Emphasis on design choices, such as the inclusion of **structs** and **classes**, sparked debate while compilation capabilities allowing **native machine code** like **.exe** remained a highlight.

- **Stay Updated with Modular's Latest Scoops**: Two important updates from the **Modular** team surfaced on Twitter, hinting at unmentioned advancements or news, with the community directed to check out [Tweet 1](https://twitter.com/Modular/status/1787601254459519395) and [Tweet 2](https://twitter.com/Modular/status/1787601269668143500) for the full details.

- **MAX Engine Excellence and API Elegance on Display**: **MAX 24.3** debuted in a community livestream, showcasing its latest updates and introducing a new Extensibility API for **Mojo**. Eager learners and curious minds are directed to [watch the explanatory video](https://www.youtube.com/watch?v=kKOCuLy-0UY).

- **Tinkering with Tensors and Tactics in Mojo Development**: From tensor indexing tips to SIMD complications for large arrays, AI engineers shared pointers and paradigms in the **mojo** domain. The discussions expanded to cover benchmarking functions, constructors in a classless setup, advanced complier tool needs, a proposal for `where` clauses, and the potential of compile-time metaprogramming in **mojo**.

- **Community Projects Propelling Mojo Forward**: Updates within the community projects showcased advancements and requests for assistance, such as an efficient radix sort plus benchmark for **mojo-sort**, migration troubles with **Lightbug** to **Mojo 24.3** detailed in a [GitHub issue](https://github.com/saviorand/lightbug_http/issues/34), and the porting of **Minbpe** to **Mojo** that outpaced Python versions at [Minbpe.mojo](https://github.com/dorjeduck/minbpe.mojo). Meanwhile, the search for a **Mojo GUI library** continues.

- **Nightly Compilation Changes the Game**: Engineers wrangled with **Mojo's type handling**, specifically with traits and variants, signaling limitations and workarounds like `PythonObject` and `@staticmethods`. A fresh **nightly compiler release** sparked conversation about automating release notifications and highlighted improvements to `Reference` usage, all framed by a playful comment about the updates stretching the capacity of a 2k monitor.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Rollback on Model Usage Rates**: **Soliloquy L3 8B** model's price dropped to **$0.05/M tokens** for 2023 - 2024, available on both private and logged endpoints as announced in [OpenRouter's price update](https://openrouter.ai/models/lynn/soliloquy-l3).

- **Seeking Beta Brainiacs for Rubik**: **Rubik's AI** calls for beta testers, offering two months of premium access to models including **Claude 3 Opus, GPT-4 Turbo, and Mistral Large** with a promo code at [rubiks.ai](https://rubiks.ai/), also hinting at a tech news section featuring Apple and Microsoft's latest endeavors.

- **Decoding the Verbose Llama **: Engineers shared frustrations over the length of responses from **llama-3-lumimaid-8b**, discussing complexities with verbosity compared to models like Yi and Wizard, and buzzed about the release of **Meta-Llama-3-120B-Instruct**, highlighted in a [Hugging Face reveal](https://huggingface.co/mlabonne/Meta-Llama-3-120B-Instruct).

- **Inter-Regional Model Request Mysteries**: Users mulled over **Amazon Bedrock** potentially imposing regional restrictions on model requests, with the consensus tilting towards cross-region requests being plausible.

- **Precision Pointers and Parameter Puzzles**: Conversations peeled back preferences on model precision within OpenRouter, generally sustaining **fp16**, and occasionally distilling to **int8**, dovetailing into discussions on whether the default parameters require tinkering for optimal conversational results.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Python 3.10 Spells Success**: Open Interpreter (OI) should be run with **Python 3.10** to avoid compatibility issues; one user improved performance by switching to models like *dolphin* or *mixtral*. The [GitHub repository for Open Interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/core/computer/skills/skills.py) was suggested for insights on skill persistence.

**Conda Environments Save the Day**: Engineers recommended using a **Conda environment** for a conflict-free installation of Open Interpreter on Mac, specifically with Python 3.10 to sidestep version clashes and related errors.

**Jan Framework Enjoys Local Support**: Jan can be utilized as a local model framework for the O1 device without hiccups, contingent on similar model serving methods as with Open Interpreter.

**Globetrotters Inquire About O1**: The 01 device works globally, but hosted services are assumed to be US-centric for now, with no international shipments confirmed.

**Fine-Tuning Frustrations and Fixes**: A call to understand and employ **system messages** effectively before fine-tuning models led to the suggestion of **[OpenPipe.ai](https://openpipe.ai/)**, as members navigate optimal performance for various models with Open Interpreter. The conversation included benchmarking models and the poor performance of Phi-3-Mini-128k-Instruct when used with OI.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Open Source Magic on the Rise**: The community launched an open-source alternative to Sora, named **StoryDiffusion**, released under an MIT license on [Github](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file); its weights, however, are still pending release.

**Memory Efficiency Through Unsloth Checkpointing**: Implementing unsloth gradient checkpointing has led to a reported reduction in VRAM usage from 19,712MB to 17,427MB, highlighting Unsloth's effectiveness in memory optimization.

**Speculations on Lazy Model Layers**: An oddity was observed where only specific slices of model layers were being trained, contrasting the full layer training seen in other models; theories posited include models potentially optimizing mainly the first and last layers when confronted with too easy datasets.

**Prompt Design Proves Pivotal**: AI enthusiasts emphasized that prompt design, particularly regarding the use of suitable templates and end-of-text tokens, is critical in influencing model performance during both fine-tuning and evaluation.

**Expanded Axolotl Docs Unveil Weight Merging Insights**: A new update to **Axolotl** documentation has been rolled out, enhancing insights on merging model weights, with an emphasis on extending these guidelines to cover inference strategies, as seen on the [Continuum Training Platform](https://axolotl.continuumlabs.pro/).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain's OData V4 and More**: Discussions highlighted interest in **LangChain's compatibility** with **Microsoft Graph (OData V4)**, and a need for API access to **kappa-bot-langchain**. There was also a query about the `k` parameter in **ConversationEntityMemory**, referencing the [LangChain documentation](https://python.langchain.com/docs/modules/tools/custom_tools/).

- **Python vs. JS Streaming Consistency Issues**: Members are experiencing inconsistencies with `streamEvents` in the JavaScript implementation of **LangChain's RemoteRunnable**, which works as expected in Python. This prompted suggestions to contact the [LangChain GitHub repository](https://github.com/langchain-ai/langchainjs) for resolution.

- **AI Projects Seek Collaborators**: An update was shared about *everything-ai* V1.0.0, now including a user-friendly local AI assistant with capabilities like text summarization and image generation. The request for beta testers for Rubiks.ai, a research assistant tool, was also discussed. Beta tester sign-up is available at [Rubiks.ai](https://rubiks.ai/).

- **No-Code Tool for Smooth AI Deployments**: Introduction of a no-code tool aimed at easing the creation and deployment of AI apps with embedded prompt engineering features. The early demo can be watched [here](https://drive.google.com/file/d/1U5lHigyCf655-sD2fbThuE0HMvueW2JR/view?usp=sharing).

- **Learning Langchain Through Video Tutorials**: Members have access to the "Learning Langchain Series" with the latest tutorials on **API Chain** and **Router Chain** available on [YouTube](https://youtu.be/IpLJwDfxiVA?si=v8Vocd0QzJx-CsbW) and [here](https://youtu.be/ItppCNZBzbY?si=9OeVm6OOU2giW1Zc), respectively. These guide users through the usage and benefits of these tools in managing APIs with large language models.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Hungry for Realistic AI Chat? Look to Roleplays!**: An idea was pitched to compile a dataset of purely human-written dialogue, which might include jokes and more authentic interactions, to enhance AI conversations that go beyond the formulaic replies seen in smart instruct models.

- **Create With Fake: Introducing Simian Synthetic Data**: A [Simian synthetic data generator](https://github.com/RaccoonResearch/Simian) was introduced, capable of generating images, videos, and 3D models for potential AI experimentation, offering a tool for those looking to simulate data for research purposes.

- **The Hunt for Perfect Datasets**: In response to a request about datasets optimal for text/numeric regression or classification tasks, several suggestions were made, including MNIST-1D and the [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

- **Text-to-Video: Diffusion Beats Transformers**: It was debated that diffusion models are currently the best option for state-of-the-art (SOTA) text-to-video tasks and are often more computationally efficient as they can be fine-tuned from text-to-image (T2I) models.

- **Video Diffusion Model Expert Weighs In**: An author of a stable video diffusion paper discussed the challenges faced in ensuring quality text supervision for video models, and the benefits of captioning videos using large language models (LLMs), bringing up the differences between autoregressive and diffusion video generation techniques.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Learn from OpenDevin's Creators**: LlamaIndex invites engineers to a webinar featuring OpenDevin's authors on **Thursday at 9am PT**, to explore autonomous AI agent construction with insight from GitHub's growing embrace. Register for the webinar [here](https://lu.ma/fp0xr460).

- **Hugging Face and AIQCon Updates**: Upgrades to Hugging Face's TGI toolkit now cater to **function calling and batched inference**; meanwhile, Jerry Liu gears up to discuss Advanced Question-Answering Agents at AIQCon, with discounts via "Community" code cited in a [tweet](https://twitter.com/llama_index/status/1787619739654836309).

- **Integrating LlamaIndex Just Got Trickier**: Engineers reported challenges integrating LlamaIndex with Supabase Vectorstore and experienced package import confusion, quickly addressed by the [updated llama-hub documentation](https://github.com/run-llama/llama-hub/tree/main#how-to-add-a-loadertoolllama-pack).

- **Problem-Solving the LlamaIndex**: Debating over deletion of document knowledge and local PDF parsing libraries, the community leaned towards re-instantiating the query engine and leveraging PyMuPDF for solutions, while considering prompt engineering to tackle irrelevant model responses.

- **Scouting & Reflecting on AI Agents**: Engineers seek effective **HyDE methods** for language to **SQL conversion** while introspective agents draw focus with their **reflection agent pattern**, as observed in an [article](https://medium.com/ai-artistry/introspective-agents-with-llamaindex-777d018f791d) on AI Artistry, despite some hitting a 404 error.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **LLVM IR Inspires tinygrad Formatting Proposal**: A readability improvement for **tinygrad** was suggested, looking to adopt an operation representation closer to LLVM IR's human-readable format. The conversation pivoted to Static Single Assignment (SSA) form and potential confusion caused by the placement of the PHI operation in tinygrad.

- **Tinygrad Stays Single-threaded**: George Hotz confirmed that **tinygrad does not use multi-threading** for CPU operations like matrix multiplication, maintaining its single-threaded design.

- **Remapping Tensors for Efficiency**: Techniques involving remapping tensors by altering strides were discussed, with a focus on how to perform reshapes efficiently, akin to **tinygrad's internal methods**.

- **Push for Practical Understanding in tinygrad Community**: Sharing of resources such as [symbolic mean explanations on GitHub](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/symbolic-mean.md) and a [Google Doc on view merges](https://docs.google.com/document/d/1RRuhAW-I5u_6ssbm1eLqVWSK_PYFoARIVBYwYZwmcM4/edit?usp=sharing) indicated a drive for better understanding through practical examples and documentation in the tinygrad community.

- **tinygrad Explores Quantized Inference**: Conversation touched on **tinygrad's** capabilities to perform quantized inference, a feature that can potentially compress models and accelerate inference times.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**SQL Database Harbor Found**: The SQL database needed for tracking conversational history in the Cohere toolkit is set to operate on **port 5432**, but a precise location was not mentioned.

**Google Bard Rivalry, School Edition**: A high school student planning to create a Bard-like chatbot received guidance from Cohere about adhering to user agreements with the caveat of obtaining a production key, as elaborated in [Cohere's documentation](https://docs.cohere.com/docs/going-live#production-key-specifications).

**Chroma Hiccups Amidst Local Testing**: There's an unresolved **IndexError** when using Cohere toolkit's Chroma for document retrieval, with a full log trace available at [Pastebin](https://pastebin.com/raw/1RmUQAP4) and a recommendation to use the latest prebuilt container.

**Retriever Confusion in Cohere Toolkit**: An anomaly was observed where **Langchain retriever** was selected by default despite an alternative being specified, as per a user report – though the screenshot provided to evidence this was not viewable.

**Production Key Puzzle**: A user faced an odd situation where a new production key behaved like a trial key in the Cohere toolkit. However, Cohere support clarified that it is expected behavior in **Playground / Chat UI** and correct functionality should prevail when used in the API.

**Coral Melds Chatbot and ReRank Skills**: Introducing **Coral Chatbot**, which merges capabilities like text generation, summarization, and ReRank into a unified tool available for feedback on its [Streamlit page](https://cohere-api-endpoints.streamlit.app/).

**Python Decorators, a Quick Byte**: A brief explainer titled "Python Decorators In 1 MINUTE" was shared for those seeking an expedited introduction to this pythonic concept - the video is accessible on [YouTube](https://www.youtube.com/watch?v=uVUhyPsqdJ8).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Centaur Coders Could Trim the Fat**: The integration of AI in development is fostering a trend where [Centaur Programmer](https://www.v01.io/posts/2024-centaur-programmers-shrinking-team-sizes/) teams might downsize, potentially leading to heightened precision and efficiency in production.

- **DeepSeek-V2 Climbs the Ranks**: [DeepSeek-V2](https://x.com/deepseek_ai/status/1787478986731429933) announced on Twitter as an open-source MoE model, boasts superior capabilities in code and logical reasoning, fueling discussions on its impact on current AI benchmarks.

- **Praising DeepSeek's Accomplishments**: Correspondence featured praise for DeepSeek-V2's benchmark success, with an [AI News newsletter](https://buttondown.email/ainews/archive/ainews-deepseek-v2-beats-mixtral-8x22b/) detailing the model's fascinating enhancements to the AI ecosystem.

- **Scouting for Unified Search Synergy**: The quest for effective unified search solutions prompts conversations about tools like **Glean** and a Hacker News discussion on potential open-source alternatives, suggesting a bot to bridge discordant search platforms.

- **Crowdsourcing AI Orchestration Wisdom**: Curiosity arose around best practices for AI orchestration, with community members consulting on favored tools and techniques for managing complex pipelines involving text and embeddings.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Freeware Faraday Fundamentals**: Engineers have confirmed that **Faraday** can be utilized locally without cost and does not necessitate a cloud subscription; a member's setup with 6 GB VRAM effectively runs the software along with its free voice output capability.
- **Enduring Downloads**: It was emphasized that assets such as characters and models downloaded from the **Faraday platform** can be accessed and used indefinitely without any additional charges.
- **GPU Might Makes Right**: A powerful GPU has been acknowledged as a viable alternative to a cloud subscription for running **Faraday** unless one prefers to support the developers through subscription.
- **Simulation Station Collaboration**: In the realm of user-initiated projects, @abhavkedia has sparked a collaboration for creating a **fun simulation** aligning with the Kendrick and Drake situation, encouraging other members to join in.
- **New Playground for AI Enthusiasts**: Engineers are invited to try out and potentially integrate [Llama Farm](https://labs.convex.dev/llama-farm) with discussions centering around an integration technique that involves **AI-Town**, and a pivot towards making **Llama Farm** more universally applicable in systems utilizing the OpenAI API.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Need for Speed on Device? Try Rocket-3B**: Experiencing **8 seconds per token**, participants sought faster model options, with **Rocket-3B** providing a notable speed improvement.

- **llamafile Caching Matures**: Users can prevent redundant model downloads in **llamafile** by employing the ollama cache via `-m model_name.gguf`, enhancing efficiency.

- **Port Troubles with AutoGPT and llamafile**: Integration issues between **AutoGPT** and **llamafile** surfaced; **llamafile agent** crashed during AP server starts, necessitating a manual workaround.

- **Seeking Feedback for AutoGPT-llamafile Integration**: The AutoGPT community is actively developing integration with **llamafile** as indicated by a [draft PR](https://github.com/Significant-Gravitas/AutoGPT/pull/7091), calling for feedback before further work.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**AI Benchmarks in Spotlight**: [Dr. Jim Fan's tweet](https://twitter.com/DrJimFan/status/1786054643568517261?utm_source=ainews&utm_medium=email&utm_campaign=ainews-evals-the-next-generation) spurred a debate on the overvaluation of specific benchmarks and public democracy in AI evaluation, and the member suggested AB testing as a more effective approach.

**Benchmarking Across Industries**: Drawing parallels to the database sector, one engineer underscored the significance of having standard benchmarks for AI, referencing the approach mentioned in Dr. Fan's tweet.

**TPC Standards Explained**: In response to inquiries, a member clarified TPC as the Transaction Processing Council, which standardizes database industry benchmarks, referencing specific benchmarks such as TPC-C and TPC-H.

**GPT-2's Surprising Comeback**: A light-hearted mention by **Sam Altman** prompted discussion about GPT-2’s return to the LMsys arena, with a tweet snapshot shared showing the humor involved.

**Lingering Doubts Over LMsys Direction**: Nathan Lambert voiced skepticism towards OpenAI possibly using LMsys for model evaluations and expressed concern about LMsys's resource limitations and potential reputation damage from the latest 'chatgpt2-chatbot' hype.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **PR Hits the Chopping Block**: A **Pull Request** was closed without additional context provided, signaling a potential change or halt in a discussed development effort.
- **AIDEV Excitement Builds Among Engineers**: Attendees of the upcoming **AIDEV event** are syncing up and showing enthusiasm about meeting in person, but attendees are inquiring about whether they need to bring their own food.
- **Mistral Gains Ground in German Discussions**: Utilization of the **8x22b Mistral** model has been validated for a project, with a focus on deployment and performance. Inquiries into low-latency decoding techniques and the creation of a German dataset for inclusive language sparked dynamic discussions.
- **Critical Data Set Crafting for German AI**: Suggestions for building a German-exclusive pretraining dataset from Common Crawl have been made, prompting a discussion about which domains to prioritize for inclusion due to their high-quality content.
- **Inclusive Language Resources Shared**: For those interested in implementing inclusive language modes in models, resources like the *INCLUSIFY* prototype ([https://davids.garden/gender/](https://davids.garden/gender/)) and its GitLab repository ([https://gitlab.com/davidpomerenke/gender-inclusive-german](https://gitlab.com/davidpomerenke/gender-inclusive-german)) have been circulated.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**Anthropic AI's Prompt Tool Piques Interest**: Engineers found a new **prompt generator tool** in the **Anthropic console**, sparking discussions on its potential and capabilities.

**Politeness through AI Crafted**: The tool demonstrated its value by successfully rephrasing statements more courteously, marking a thumbs-up for practical AI usage.

**Unpacking the AI's Instruction Set**: An engineer embarked on uncovering the tool's system prompt, specifically noting the heavy reliance on **k-shot examples** in its architecture.

**Extracting the Full AI Prompt Faces Challenges**: Despite hurdles in retrieving the complete prompt due to its considerable size, the enthusiasm in the discussions remained high.

**Share and Care Amongst AI Aficionados**: A pledge was made by a community member to share the fully extracted prompt with peers, ensuring collective progress in understanding and utilizing the new tool.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

Given the information provided, there is no relevant discussion content to summarize for an AI Engineer audience. If future discussions include technical, detail-oriented content, a summary appropriate for engineers can be generated.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **GitHub Issue Sparks Plugin Collaboration**: A discussion focused on improving a plugin included a link to a [GitHub issue](https://github.com/simonw/llm-evals-plugin/issues/4), indicating active development for a feature to implement parameterization in testing.
- **OpenAI Assistant API Compatibility Question**: An inquiry was made about the possibility of using `llm` with the OpenAI Assistant API, expressing concern about missing previously shared information on the topic.



---


The **Skunkworks AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1236937281487507457)** (170 messages🔥🔥): 

- **Technical Glitch in GGUF Conversion**: Users discussed a confirmed bug in fine-tuning Llama3 models when converting to GGUF, specifically mentioning an issue with retaining fine-tuned data during the conversion. Follow-up and related discussions were referenced with direct links to Reddit and GitHub issues ([Reddit Discussion](https://www.reddit.com/r/LocalLLaMA/comments/1ckvx9l/part2_confirmed_possible_bug_llama3_gguf/), [GitHub Issue](https://github.com/ggerganov/llama.cpp/issues/7062)).

- **Running Aphrodite Engine with Quantization**: A user encountered difficulties in running the Aphrodite engine with 4bit bnb quantization and sought advice. Recommendations were given to use fp16 with the `--load-in-4bit` flag and to build from the dev branch for better support and features.

- **LLM VRAM Requirements for Inference Programs**: A link was shared to a VRAM calculator on Hugging Face's Spaces ([LLM-Model-VRAM-Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)) with discussion around its accuracy and compatibility with inference programs such as vLLM, GGUF, and exllama.

- **Unsloth Studio Release Delayed**: A user inquired about the delay in the release of Unsloth Studio due to issues with phi and llama, looking forward to easier notebook usage. Another user clarified the correct usage of the eos_token in Unsloth’s updated training code for Llama3-8b-instruct.

- **Concerns About Model Base Data on Inference Results**: The impact of the base model's training data on the results of a fine-tuned model was discussed. Clarification was provided that fine-tuning likely updates weights used for predicting tokens in conversations previously seen by the model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckvx9l/part2_confirmed_possible">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094875716">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckvx9l/part2_confirmed_possible_bug_llama3_gguf/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1237041241808699412)** (86 messages🔥🔥): 

- **Supporter Role Confusion**: A member was unsure about their supporter status after a message about a private channel for supporters. It was clarified that supporter roles are present but require membership or a donation of at least $10.
- **Tackling Unfiltered LLaMA-3 Outputs**: A member expressed concern that LLaMA-3 provided uncensored outputs to questionable prompts. Despite attempts to stop it with system prompts, LLaMA-3 continued to produce explicit content.
- **FlashAttention Optimization Discussion**: A member highlighted an article on [Hugging Face Blog](https://huggingface.co/blog/mayank-mishra/padding-free-transformer) about optimizing attention computation using FlashAttention for long sequence machine learning models, which can reduce memory usage in training.
- **Graphics Card Sale Alert**: A member shared a [Reddit post](https://www.reddit.com/r/buildapcsales/comments/1cljlba/gpu_msi_geforce_rtx_4090_suprim_liquid_x_24_gb/) about a discount on an MSi GeForce RTX 4090 SUPRIM LIQUID X 24 GB Graphics Card, prompting discussions on the advantages of smaller and more efficient cooling systems in newer GPU models.
- **AI-Generated Profile Picture Admiration**: Discussion ensued about a member's new profile picture, which turned out to be AI-generated. It sparked interest and comparisons to characters from popular media.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/buildapcsales/comments/1cljlba/gpu_msi_geforce_rtx_4090_suprim_liquid_x_24_gb/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/blog/mayank-mishra/padding-free-transformer">Saving Memory Using Padding-Free Transformer Layers during Finetuning</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1236940194662453258)** (412 messages🔥🔥🔥): 

- **GGUF Upload Queries**: Members discussed the ability to upload GGUF models to GPT-4all, with confirmation from another member that it should be possible, and the use of Huggingface's `model.push_to_hub_gguf` to do so.
- **Tokenization Troubles**: Conversations highlighted an issue with tokenization across various formats including GGUF, noting differences in responses when using Unsloth for fine-tuning compared to other inference methods.
- **Tokenizer Regex Revision**: There's an ongoing discussion on GitHub Issue [#7062](https://github.com/ggerganov/llama.cpp/issues/7062) regarding tokenization problems with LLama3 GGUF conversion, especially relating to LORA adapters; a regex modification has been proposed to address this.
- **LORA Adapters and Training**: A member successfully used LORA by training with `load_in_4bit = False`, saving LORA adapters separately, and converting them using a specific llama.cpp script, which resulted in perfect results for them.
- **Deployment and Multigpu Questions**: Inquiries about deployment using local data for fine-tuning models and the ability to use multiple GPUs for training with Unsloth were discussed, with the current conclusion that Unsloth does not yet support multigpu but it may in the future.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cltac3/part3_cause_to_issue_found_possible_bug_llama3/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/bartowski1182/status/1786038369132171444?t=hJfQz8lGt9v31yZRG4X1vA&s=09">Tweet from bartowski (@bartowski1182)</a>: After days of compute (since I had to start over) it&#39;s finally up! Llama 3 70B GGUF with tokenizer fix :)  https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF  In other news, just orde...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://tenor.com/view/cooking-cook-gif-26515278">Cooking GIF - Cooking Cook - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/docs/trl/sft_trainer#dataset-format-support">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://www.reddit.com/user/Dependent_Factor_204/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ScottMcNaught">ScottMcNaught - Overview</a>: ScottMcNaught has one repository available. Follow their code on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/430">GGUF breaks - llama-3 · Issue #430 · unslothai/unsloth</a>: Findings from ggerganov/llama.cpp#7062 and Discord chats: Notebook for repro: https://colab.research.google.com/drive/1djwQGbEJtUEZo_OuqzN_JF6xSOUKhm4q?usp=sharing Unsloth + float16 + QLoRA = WORKS...</li><li><a href="https://github.com/ollama/ollama/issues/3759#issuecomment-2097836764">llama3-instruct models not stopping at stop token · Issue #3759 · ollama/ollama</a>: What is the issue? I&#39;m using llama3:70b through the OpenAI-compatible endpoint. When generating, I am getting outputs like this: Please provide the output of the above command. Let&#39;s proceed f...</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=LjY75GoYUCB8">Google Colab</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2095465106">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/gg/bpe-preprocess">GitHub - ggerganov/llama.cpp at gg/bpe-preprocess</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2098452227">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094948789">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2095371349">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/f4ab2a41476600a98067a9474ea8f9e6db41bcfa">llama : fix BPE pre-tokenization (#6920) · ggerganov/llama.cpp@f4ab2a4</a>: * merged the changes from deepseeker models to main branch
 
 * Moved regex patterns to unicode.cpp and updated unicode.h
 
 * Moved header files
 
 * Resolved issues
 
 * added and refactored unic...</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/ca3632602091e959ed2ad4c09c67a7c790b10d31">readme : add note that LLaMA 3 is not supported with convert.py (#7065) · ggerganov/llama.cpp@ca36326</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1237019268596633652)** (23 messages🔥): 

- **LLaMA Variants for Enhanced Knowledge**: A new **LLaMA-3 variant** has been developed for aiding knowledge graph construction, with a focus on structured data like **RDF triples**. The model, [LLaMA-3-8B-RDF-Experiment](https://huggingface.co/M-Chimiste/Llama-3-8B-RDF-Experiment), is designed to generate knowledge graph triples and specifically excludes non-English data sets.
- **Instruct Coder Model Released**: A new **LLaMA model**, rombodawg/Llama-3-8B-Instruct-Coder-v2, has been finished and hosts improvements over its predecessor. The updated model [Llama-3-8B-Instruct-Coder-v2](https://huggingface.co/rombodawg/Llama-3-8B-Instruct-Coder-v2) has been retrained to fix previous issues and is expected to perform better.
- **Oncord: Professional Website Builder Unveiled**: **Oncord** has been presented as a professional website builder for creating modern websites with integrated tools for marketing, commerce, and customer management. The platform, showcased at [oncord.com](https://www.oncord.com/), offers a read-only demo and is aimed at a mix of technical and non-technical users.
- **Open Call for Collaboration on Machine Learning Paper**: There's an invitation for the community to contribute to an **open source paper** predicting IPO success with machine learning. Interested parties can assist with the paper hosted at [RicercaMente](https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html).
- **Startup Discussion and Networking**: A dialogue took place regarding **startup marketing**, strategies, and collaborations. Specifically, one startup [Oncord](https://www.oncord.com/) has been discussed, with a focus on enhancing technical flexibility for users, and another concept for measuring trust between viewers and content creators was hinted at but not officially launched yet.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html">PREDICT IPO USING MACHINE LEARNING</a>: Open source project that aims to trace the history of data science through scientific research published over the years</li><li><a href="https://huggingface.co/rombodawg/Llama-3-8B-Instruct-Coder-v2">rombodawg/Llama-3-8B-Instruct-Coder-v2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/M-Chimiste/Llama-3-8B-RDF-Experiment">M-Chimiste/Llama-3-8B-RDF-Experiment · Hugging Face</a>: no description found</li><li><a href="https://pubmed.ncbi.nlm.nih.gov/28288169/):">The miR-200 family is increased in dysplastic lesions in ulcerative colitis patients - PubMed</a>: UC-Dysplasia is linked to altered miRNA expression in the mucosa and elevated miR-200b-3p levels.</li><li><a href="https://www.oncord.com/">Oncord - Digital Marketing Software</a>: Website, email marketing, and ecommerce in one intuitive software platform. Oncord hosted CMS makes it simple.</li><li><a href="https://www.tryoncord.com/admin/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1237119442899435582)** (3 messages): 

- **Fine-Tuning LVLM Request**: A member expressed a desire for a *generalised way of fine-tuning* Large Vision Language Models (LVLM).

- **Call for Moondream Support**: Another member requested support for **moondream**, noting that it currently only finetunes the phi 1.5 text model, and shared the [GitHub notebook for moondream finetuning](https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb).

**Link mentioned**: <a href="https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb">moondream/notebooks/Finetuning.ipynb at main · vikhyat/moondream</a>: tiny vision language model. Contribute to vikhyat/moondream development by creating an account on GitHub.

  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1237067945465876572)** (2 messages): 

- **In Quest for Data Collection Progress**: A member inquired about the current count of pages collected in the **cortex** project, seeking an update on the data accumulation milestone.

- **Navigating the Void**: A link was posted presumably related to the **ctx-length-research** channel, but the content or context of the link is inaccessible as it was referenced as *<<<null>>>*.
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1237082706584670271)** (6 messages): 

- **Innovative Cooking Convenience Unveiled**: A YouTube video titled "Recipic Demo" was shared, showcasing a website where users can upload their available ingredients to receive meal recipes. Intrigue is sparked for those seeking culinary inspiration with what they have on hand. [Watch "Recipic Demo"](https://youtu.be/J1ZiaE7cqQY)

- **Delving into Enhancements for Multimodal Language Models**: A member inquires about ways to significantly improve multimodal language models, mentioning the integration of **JEPA** as a potential enhancement, though a repository or model for such integration hasn't been found.

- **Multimodal Collaboration Envisioned**: In response to enhancing multimodal language models, another member suggests the idea of tools that enable language models to utilize **JEPA models**, indicating an interest in cross-model functionality.

- **Push for Higher Resolution in Multimodal Language Models**: Advancing multimodal models can involve increasing their resolution to better interpret small text in images, a member suggests. This advancement could widen the scope of visual data that language models can effectively understand and incorporate.

**Link mentioned**: <a href="https://youtu.be/J1ZiaE7cqQY">Recipic Demo</a>: Ever felt confused about what to make for dinner or lunch? What if there was a website where you could just upload what ingredients you have and get recipes ...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1237075619481915414)** (7 messages): 

- **AQLM Pushes the Envelope with Llama-3**: The [AQLM](https://github.com/Vahe1994/AQLM) project introduces more prequantized models, such as **Llama-3-70b** and **Command-R+**, enhancing the accessibility of open-source Large Language Models (LLMs). In particular, _Llama-3-70b_ can run on a single RTX3090, showcasing significant progress in _model quantization_.
  
- **Orthogonalization Techniques Create Kappa-3**: Phi-3's weights have been [orthogonalized to reduce model refusals](https://www.reddit.com/r/LocalLLaMA/comments/1clmo7u/phi3_weights_orthogonalized_to_inhibit_refusal/), released as the Kappa-3 model. Kappa-3 comes with full precision (fp32 safetensors) and a GGUF fp16 option, although questions remain about its performance on prompts requiring rule compliance.

- **Deepseek AI Celebrates a Win**: A share from [Deepseek AI's Twitter](https://twitter.com/deepseek_ai/status/1787478990665777589) points to their success, triggering a light-hearted joke about family resemblances in AI achievements.

- **Revolutionizing Healthcare with Deterministic Quoting**: [Invetech's](https://www.invetechgroup.com/) project introduces **"Deterministic Quoting"** to address the risk of LLMs generating hallucinated quotations in sensitive fields like healthcare. With this technique, only verbatim quotes from the source material are displayed with a blue background, aiming to enhance trust in AI's use in medical record processing and diagnostics. [Details and visual provided](https://mattyyeung.github.io/deterministic-quoting).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mattyyeung.github.io/deterministic-quoting">Hallucination-Free RAG: Making LLMs Safe for Healthcare</a>: LLMs have the potential to revolutionise our field of healthcare, but the fear and reality of hallucinations prevent adoption in most applications.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1clinlb/bringing_2bit_llms_to_production_new_aqlm_models/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1clmo7u/phi3_weights_orthogonalized_to_inhibit_refusal/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1236938502470635520)** (527 messages🔥🔥🔥): 

```html
<ul>
  <li><strong>AI Chatbot Comparison and Speculation</strong>: Members discussed the performance of various AI models, with particular focus on function calling capabilities. **Llama 3 70b** was deemed superior to **Mistral 8x22b** for function calling, despite the latter's "superior function calling" marketing.</li>
  <li><strong>The Return of GPT-2 in LMSYS</strong>: There's buzz around the return of **GPT-2** to LMSYS with significant improvements, and speculation on whether it's a new model being A/B tested or something else, such as GPT-4Lite or a more cost-efficient GPT alternative.</li>
  <li><strong>Testing of the Hermes 2 Pro Llama 3 8B Model</strong>: A member requested testing of the **Hermes 2 Pro Llama 3 8B** model's function calling ability up to the 32k token limit, but practical limitations due to time and resource constraints were mentioned.</li>
  <li><strong>Chatbot Names, Open Source Hopes, and GPT Hype Debates</strong>: The unique naming of chatbot models (like GPT-2 chatbot) led to discussions and jokes about their capabilities and the potential for an OpenAI model becoming open source. There were both skepticism and anticipation regarding the next big AI development and its release timeline.</li>
  <li><strong>YAML vs. JSON in Model Input</strong>: A brief mention was made on the preference for YAML over JSON for model inputs due to better human readability and token efficiency.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/cognitivecomputations/Meta-Llama-3-120B-Instruct-gguf">cognitivecomputations/Meta-Llama-3-120B-Instruct-gguf · Hugging Face</a>: no description found</li><li><a href="https://www.yahoo.com/tech/openai-exec-says-chatgpt-laughably-211309042.html?guccounter=1&guce_referrer=aHR0cHM6Ly90LmNvLw&guce_referrer_sig=AQAAAGwirDgIzCb9D5wbMs_5Bi7_J5eDn2Ua-tptQL2z-ypVUJr4y7Widh2gdqzKdDntOf6YEFv8Y-WCz1N124gqIDNK_WjZ1c_HFrRai7SW28OC6kdiYNcMseLL_Q8w3yW-VNi5gKpd9Lm0AIA3NmB3YCDGPvh1WbIet-FhnT1V86AO">OpenAI exec says today&#x27;s ChatGPT will be &#x27;laughably bad&#x27; in 12 months</a>: OpenAI&#x27;s COO said on a Milken Institute panel that AI will be able to do &#x22;complex work&#x22; and be a &#x22;great teammate&#x22; in a year.</li><li><a href="https://fxtwitter.com/maximegmd/status/1783091026384527833">Tweet from Maxime G, M.D (@maximegmd)</a>: Internistai 7b: Medical Language Model  Today we release the best 7b medical model, outperforming GPT-3.5 and achieving the first pass score on the USMLE! Our approach allows the model to retain the s...</li><li><a href="https://tenor.com/view/mlp-relevant-mylittlepony-interests-gif-4506356">Mlp Relevant GIF - MLP Relevant Mylittlepony - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/vllm-project/vllm/issues/1229">Support for grammar · Issue #1229 · vllm-project/vllm</a>: It would be highly beneficial if the library could incorporate support for Grammar and GBNF files. https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md</li><li><a href="https://huggingface.co/TRI-ML/mamba-7b-rw">TRI-ML/mamba-7b-rw · Hugging Face</a>: no description found</li><li><a href="https://x.com/0xblacklight/status/1787329977957982398">Tweet from Kyle Mistele 🏴‍☠️ (@0xblacklight)</a>: btw I tested this with @vllm_project and it works to scale @NousResearch&#39;s Hermes 2 Pro Llama 3 8B to ~32k context with great coherence & performance (I had it summarizing @paulg essays)  Download...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7013">Update Server&#39;s README with undocumented options for RoPE, YaRN, and KV cache quantization by K-Mistele · Pull Request #7013 · ggerganov/llama.cpp</a>: I recently updated my LLama.cpp and found that there are a number of server CLI options which are not described in the README including for RoPE, YaRN, and KV cache quantization as well as flash at...</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B#prompt-format-for-json-mode--structured-outputs">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/lerobot">GitHub - huggingface/lerobot: 🤗 LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch</a>: 🤗 LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch - huggingface/lerobot</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/1965">Extending context size via RoPE scaling · ggerganov/llama.cpp · Discussion #1965</a>: Intro This is a discussion about a recently proposed strategy of extending the context size of LLaMA models. The original idea is proposed here: https://kaiokendev.github.io/til#extending-context-t...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5104">Port of self extension to server by Maximilian-Winter · Pull Request #5104 · ggerganov/llama.cpp</a>: Hi, I ported the code for self extension over to the server. I have tested it with a information retrieval, I inserted information out of context into a ~6500 tokens long text and it worked, at lea...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1236985926773899304)** (12 messages🔥): 

- **LlamaCpp Update Resolves Issue**: An issue with **LlamaCpp** not generating the `<tool_call>` token was resolved by updating to the latest version. The system prompt now works as intended.
- **LoRA Tuning Challenges on A100**: A member is experiencing unexpectedly long training times with **LoRA llama 3 8b**, where each step takes approximately 500 seconds on an A100 using **axolotl**, prompting them to consider debugging due to others having much faster training times.
- **Comparative Training Speed Insights**: For **Llama2 7B**, a member reported it took roughly 3 minutes for 1,000 iterations using **litgpt**, indicating a significant speed difference in training times compared to what another member experienced with **LoRA**.
- **Best Practices for Teaching GPT Examples**: A member asked for advice on the best method to train GPT with examples, contemplating between providing a file with examples and structuring the examples as repeated user-assistant message pairs.
- **Attention Paper Implementation Feedback Request**: A member sought feedback on their reimplementation of the "Attention is All You Need" paper, sharing their GitHub repository at [https://github.com/davidgonmar/attention-is-all-you-need](https://github.com/davidgonmar/attention-is-all-you-need). They're considering improvements like using torch's scaled dot product and pretokenizing.
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1237369136129118219)** (11 messages🔥): 

- **New Miner Repo Stuck**: A user reported their repository commitment to the mining pool was not downloading for hours, indicating potential network issues for newcomers.
- **Network Awaiting Critical PR**: A user mentioned the **Bittensor network** is currently non-operational until a pending pull request (PR) is merged, which is crucial for fixing the network.
- **Timeframe for Network Fix Uncertain**: When asked, a user stated that the PR would be merged **"soon"**, but clarified they have no control over the PR review process, leaving an ambiguous timeline.
- **Network Issues Stall Model Validation**: Clarification was given that new commits or models submitted to the network will not be validated until the aforementioned PR is resolved, directly impacting miner operations.
- **Seeking GraphQL Service Information**: One user inquired about resources or services related to GraphQL for **Bittensor subnets**, indicating a possible need for developer support or documentation.
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1237186712036315186)** (7 messages): 

- **World Sim Access Issues Persist**: A member expressed difficulties with accessing [worldsim.nousresearch.com](https://worldsim.nousresearch.com), noting that the site is still not operational with a simple "still not work" comment.
- **Expressing Disappointment**: In response to the ongoing issue, there was another expression of disappointment, characterized by multiple frowning emoticons.
- **Call for Simulation**: A brief message stating "plz sim" was posted, possibly indicating a desire to start or engage with a simulation.
- **Inquiry about World Sim**: A member inquired, "What's world sim? Where can i find more info? and What's a world-sim role?" showing interest in the simulation aspect of the channel.
- **Guidance to Information**: In response to questions about World Sim, a member directed others to a specific channel <#1236442921050308649> for a pinned post that likely contains the relevant information.

**Link mentioned**: <a href="https://worldsim.nousresearch.com/">worldsim</a>: no description found

  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1236939628943245312)** (421 messages🔥🔥🔥): 

- **Dissecting Diffusion Models**: Members exchanged insights on the difference between **Hyper Stable Diffusion**, a finetuned or LoRA-ed model that operates quickly, and **Stable Diffusion 3**, a distinct model not equivalent to Hyper Stable Diffusion. Links to explanatory resources were not provided.

- **Seeking Stable Diffusion Clarity**: Conversations circled around **Stable Diffusion** no longer being open-source and the potential non-release of SD3. Users discussed the importance of downloading and saving models and adapters amid fears that AI's open-source era might be ending.

- **Optimizing Realistic Human Models**:
  A discussion on finding the best **realistic human model** with flexibility covered various model options, with suggestions to avoid heavy bias in models like those from **civitai** to prevent sameness in generated people.

- **Dreambooth and LoRA Explorations**: Users engaged in verbose consultation and detailed discussions about how to best use **Dreambooth** and **LoRA** training for **Stable Diffusion**, debating the best approach to creating unique faces and styles.

- **Adventures in Upscaling**: Queries about the most effective upscaler led to discussions about various upscaling models and workflows such as **RealESRGAN_x4plus** and **4xUltrasharp**, with users sharing personal experiences and approximation of preferences.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stable-diffusion-art.com/beginners-guide/">How to use Stable Diffusion - Stable Diffusion Art</a>: Stable Diffusion AI is a latent diffusion model for generating AI images. The images can be photorealistic, like those captured by a camera, or in an artistic</li><li><a href="https://www.youtube.com/watch?v=kqXpAKVQDNU&list=PLXS4AwfYDUi5sbsxZmDQWxOQTml9Uqyd2">How to Install Stable Diffusion - automatic1111</a>: Part 2: How to Use Stable Diffusion https://youtu.be/nJlHJZo66UAAutomatic1111 https://github.com/AUTOMATIC1111/stable-diffusion-webuiInstall Python https://w...</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/stable_cascade/">Stable Cascade Examples</a>: Examples of ComfyUI workflows</li><li><a href="https://stylus-diffusion.github.io/">Stylus: Automatic Adapter Selection for Diffusion Models</a>: no description found</li><li><a href="https://youtu.be/xXNr9mrdV7s?si=PalGotNpAeYl4Zy4">LORA training EXPLAINED for beginners</a>: LORA training guide/tutorial so you can understand how to use the important parameters on KohyaSS.Train in minutes with Dreamlook.AI:  https://dreamlook.ai/?...</li><li><a href="https://rentry.co/59xed3#prodigy">THE OTHER LoRA TRAINING RENTRY</a>: Stable Diffusion LoRA training science and notes By yours truly, The Other LoRA Rentry Guy. This is not a how to install guide, it is a guide about how to improve your results, describe what options d...
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1236950886933205022)** (107 messages🔥🔥): 

- **Server Logging Off Option Request**: A user expressed discomfort with the inability to turn off server logging via the GUI in LM Studio, emphasizing a desire for increased privacy in their app development process.
- **Recognition of Prompt Engineering Value**: The legitimacy of prompt engineering as a critical and valuable skill in the tech industry was acknowledged, with references indicating it as a lucrative career and a pivotal aspect in producing high-quality outputs from LLMs.
- **Headless Mode Operation for LM Studio**: Users discussed the feasibility of operating LM Studio in a headless mode, where a user demonstrated interest in starting the server mode via command line rather than GUI, and others provided insights on using lms CLI as a potential solution.
- **Phi-3 vs. Llama 3 for Quality Outputs**: A debate emerged over the effectiveness of the Phi-3 model compared to Llama 3, particularly concerning the task of summarizing content and generating FAQs, with users sharing settings and strategies to improve outcomes.
- **Troubleshooting Model Crashes and Configuration**: Multiple users reported issues regarding model performance in LM Studio, with problems such as high RAM consumption despite sufficient VRAM, unexpected behavior after updates, and errors when loading models. Community members responded with suggestions such as checking drivers, adjusting model configs, and evaluating system specs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/welcome">Welcome | LM Studio</a>: LM Studio is a desktop application for running local LLMs on your computer.</li><li><a href="https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41">How I Won Singapore’s GPT-4 Prompt Engineering Competition</a>: A deep dive into the strategies I learned for harnessing the power of Large Language Models (LLMs)</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>: LM Studio in your terminal. Contribute to lmstudio-ai/lms development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1237122192605712465)** (21 messages🔥): 

- **Llama Toolkit Update Affects Command R+ Tokenizer**: Changes in llamacpp upstream for llama3 broke Command R and Command R+'s tokenizer, with additional reports of incorrect quantization. Updated quants for Command R+ can be found at [Hugging Face Co Model Repo](https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF), and a note that `do not concatenate splits` but rather use `gguf-split` for file merging, if necessary.

- **Problems Fine-tuning Hermes-2-Pro L3 Noted**: Despite popularity, fine-tuning Hermes-2-Pro L3 still presents issues, with an opinion expressed that it's better than L3 8b but not as improved over its predecessor as hoped.

- **Hermes-2-Pro L3 in Action**: Running the model with 8bit MLX showed impressive handling of incoherent input, with a quoted example testing the AI's response to disclosing potentially unethical information. A user queried about applying a "jailbreak" to remove content safeguards.

- **Difficulties with GGUF Format and Llama.cpp**: A user learned that GGUF format is currently not working with llama.cpp due to the need for an update in the toolkit. Prompting the suggestion to try alternative models or wait for the toolkit update before using certain GGUF-based models.

- **Translations and Cultural Sensitivity in AI Models**: Recommendations were made for ai models excelling in translation and creative writing, noting Llama3's multilingual capabilities and Command R's language support. For culturally sensitive responses, WestLake was recommended, and for specific translation tasks, the T5 model was suggested as an alternative, available through [Hugging Face's T5 documentation](https://huggingface.co/docs/transformers/model_doc/t5).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF">dranger003/c4ai-command-r-plus-iMat.GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/YorkieOH10/granite-34b-code-instruct-Q8_0-GGUF">YorkieOH10/granite-34b-code-instruct-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/model_doc/t5">T5</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1237320853079851009)** (11 messages🔥): 

- **Linux Memory Misreporting in LM Studio**: A user reported that their Ubuntu machine displayed **33.07KB** of free RAM in LM Studio, while actually having **20GB** free. There was confirmation of using version **0.2.22 of the Linux AppImage** from the LM website.
- **Ubuntu Version Could Affect Dependencies**: Clarification was sought on whether the user was using the modern **Ubuntu v22.04**, as older versions might have issues with library dependencies.
- **Library Dependency Concerns on Older Ubuntu Releases**: The discussion pointed towards the possibility of **library dependencies not functioning correctly** on older Ubuntu releases.
- **Disabling GPU Offloading Resolves Running Issue**: Disabling GPU offloading in settings appeared to resolve an issue, allowing a user to run the **Meta Llama 3 instruct 7B** model.
- **Guidance for Accessing Linux Beta Channels**: A user was directed to obtain access to the Linux Beta channels by signing up for the **Linux beta role** through the Channels & Roles option on LM Studio.
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1237053597582229556)** (8 messages🔥): 

- **Scoped Access to Document Sections**: A user inquired about a technique in LM Studio, or LLMs in general, that would allow giving the AI temporary access to a specific section of a document. It was clarified that LLMs only know what is included in the prompt or embedded in their weights.
- **AI Responding to Deleted Content**: The same user reported instances where LM Studio seemed to mix contexts that had been deleted, raising the possibility of bugs affecting the AI's response.
- **Understanding AI's Response Mechanism**: There is a discussion on whether language models can retain information that was believed to be deleted, but the consensus is that if a language model seems to remember deleted content, it could either be due to a bug or an illusion. 


  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1237015258187698246)** (9 messages🔥): 

- **Seeking the Ultimate AI Dungeon Master**: A member expressed frustration with models like **Mixtral** and **Wizard LM** failing to track complex game elements in **Dungeons & Dragons**, despite uploading extensive background information. They reported that models struggle with maintaining continuity, like character sheets and hit points in adventures, even with the help of the *AnythingLLM database*.
  
- **llava-phi-3-mini Model Confusion**: A member reported issues with **llava-phi-3-mini** model, which instead of describing the uploaded image, would describe random images from **Unsplash**. Attempts with different Prompt Templates have not resolved the problem, which includes the template getting stuck in a loop.

- **Philosophical Troubleshooting Inquiry**: In response to problems with the **llava-phi-3-mini model**, another member inquired about how it's determined that the model is describing content from **Unsplash** and what specific Prompt Template was failing.

- **Continued Model Image Recognition Struggles**: Multiple members are facing similar issues with vision models that either describe Unsplash images instead of the uploaded ones or stop working after processing the first image. The problem persists across various models, including **ollama**, and seems to be due to a recent backend update.

- **Bunny Llama to the Rescue!**: Amidst the issues with various models, a member found success in **Bunny Llama 3 8B V**, which worked for them without the issues present in other models.

- **Long-Term Memory Challenges in AI Role-Playing**: A member suggested **nous research hermes2 mitral 8x7b do the q8** version and **ycros bagelmusterytour v2 8x7b**, musing that the current databases might be inadequate for advanced role-playing. They recommended exploring *lollms*, which has shown promise in maintaining long-term memory and personality continuity.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1237009526989918280)** (25 messages🔥): 

- **GPU Power Consumption Discussions**: A user observed their **P40 GPUs** idling at 10 watts but never dropping below 50 watts after use, with a total draw of 200 watts from the GPUs, even when LM Studio is only using one at a time. They shared their server setup details, which include two 1600-watt power supplies at 220vac and a setup inside a shop to mitigate noise, while they remote desktop from their office.

- **Planning GPU Power Budget for Inference**: Another user discussed their plan to limit their GPU to 140 watts for 85% performance, intending to use it on models such as **7b Mistral** with high context and a small vision model, and asked if LM Studio effectively utilizes multiple GPUs.

- **Assessing Gaming Mobo for P40s Without Additional GPUs**: One user pondered about using gaming motherboards for their P40s since server motherboards would provide full PCIe x16 bandwidth for each GPU, whereas gaming motherboards might compromise bandwidth when running multiple GPUs.

- **Debunking PCI-E Bandwidth Myths for Inference**: In response to concerns about having sufficient PCI-E bandwidth for inference, a user provided Reddit links ([Inference Reliance on PCI-E Bandwidth](https://www.reddit.com/r/LocalLLaMA/comments/15yknoo/is_inference_reliant_on_pcie_bandwidth/), [Multi-GPU on exl2 or llamacpp](https://www.reddit.com/r/LocalLLaMA/comments/178y4tj/is_multigpu_on_exl2_or_llamacpp_affected_by_low/)) and a GitHub discussion ([Perf test on various HW](https://github.com/turboderp/exllama/discussions/16#discussioncomment-6245573)), suggesting that PCI-E bandwidth requirements are often overestimated for inference tasks.

- **Considering Hardware Configs for Efficient LLM Inferencing**: Users exchanged ideas about efficient server builds, power consumption, thermals, and the balance between robust hardware and practicality, discussing if a single-purpose server is justified for running language models, and shared their practices, like not running servers 24/7 to save on energy.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.msn.com/en-us/lifestyle/shopping/apple-announces-new-m4-chip/ar-BB1lYkba">MSN</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15yknoo/is_inference_reliant_on_pcie_bandwidth/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/178y4tj/is_multigpu_on_exl2_or_llamacpp_affected_by_low/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/turboderp/exllama/discussions/16#discussioncomment-6245573">Perf test on various HW · turboderp/exllama · Discussion #16</a>: First of all I would like to thank you for your work, really like your inference implementation, seems to be the fastest so far for nvidia gpus! I ran a bunch of tests on various GPUs and wanted to...</li><li><a href="https://github.com/turboderp/exllama/discussions/16#discussioncom">Perf test on various HW · turboderp/exllama · Discussion #16</a>: First of all I would like to thank you for your work, really like your inference implementation, seems to be the fastest so far for nvidia gpus! I ran a bunch of tests on various GPUs and wanted to...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1237379779834151002)** (28 messages🔥): 

- **Elusive Error with Vision Model**: A user reported an error when using a vision model, but the error message didn't include an error code or description, only system specifications. Other members (like propheticus_05547 and heyitsyorkie) engaged to clarify the problem, suggesting possible GPU limitations and asking about other models' functionality.

- **In Search of the Newest LLMs on LM Studio**: User yu_xie inquired about refreshing LM Studio content to get the latest LLM models. heyitsyorkie explained that the home page of LM Studio is static and not a live repository, suggesting instead to use the model explorer, search "GGUF," and sort by Most Recent as a workaround.

- **Downloadable LLM Models**: A user requested the ability to download the latest LLM models directly from the LM Studio App home page. heyitsyorkie responded that it's not currently possible due to models not always being released in GGUF format and potential updates required for llama.cpp when new models like IBM Granite are released.

- **Isolated Incident or Pervasive Issue?**: A user, aibyhumans, mentioned crashes when using a 7B model on an 8GB GPU and that the crashes occur only with visual models. propheticus_05547 responded by questioning whether non-vision models work and suggesting to turn off GPU offloading or adjust it to 50/50.

- **Model Does Not Stick the Landing**: The same user, aibyhumans, observed that the model seems not to fully unload after a crash and that after one successful load, subsequent attempts result in errors, necessitating a restart of the model each time. This behavior was noted to occur with different models as well.
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1237227937216991323)** (2 messages): 

- **Troubleshooting AutoGen Studio**: A member mentioned a bug in **AutoGen Studio** that seems to send truncated messages with only one or two words. They requested to see the code used for calling the bot and suggested printing out the message list before sending it.
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1237362821969477683)** (2 messages): 

- **LM Studio SDK Development**: A member mentioned the availability of the new **lmstudiojs SDK**, implying future **langchain integrations** are on the horizon. 

  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1237363118376620072)** (1 messages): 

- **Inquiry on GPT-Engineer Setup**: A member expressed interest in understanding how **GPT-Engineer** was set up with LM Studio, asking specifically whether custom prompting was involved in the process.

  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1237164062589321266)** (41 messages🔥): 

- **Dependency Package Install Success**: The command `sudo apt install xvfb fuse libatk1.0-0 libatk-bridge2.0-0 libcups2 libgtk-3-0 libgbm1 libasound2` resolved prior issues, allowing the user to move past Step 2 in their installation process.

- **Compatibility Issues with LM Studio on Ubuntu 22.04**: Users discussed compatibility issues while running LM Studio on Ubuntu 22.04 versus the latest Ubuntu Server 24.04, suggesting a possible dependency on newer operating systems.

- **LM Studio Needs GPU?**: A member trying to run LM Studio encountered errors implying GPU issues, despite using a server that only provides a virtual or motherboard-based GPU.

- **LM Studio on Virtual GPU**: Another user confirmed they have successfully run **LM Studio** inside a Virtual Machine using a virtual GPU, although they have not tested environments without a GPU.

- **LM Studio API Integration**: A member shared excitement about integrating LM Studio support via the API into their own UI, while another faced challenges with LM Studio's concurrent request handling, especially when generating embeddings without using the provided SDK.
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1236943392643092510)** (164 messages🔥🔥): 

- **Whispering Fresh: Fine-Tuning ASR**: Users discussed fine-tuning the **openai/whisper-small** ASR model, highlighting the importance of a sufficient dataset size and considering hyperparameter adjustments like `weight_decay` and `learning_rate` for better performance. Specific hyperparameters shared included those for gradient accumulation steps, learning rate, evaluation strategy, and more, while addressing differences between training and validation loss.

- **Accessing Llama on Android**: One participant inquired about using **llama.cpp on Android** with Termux and accessing it through a webpage, indicating ongoing exploration of Llama's applications on mobile platforms.

- **Puzzling Performance Paradox**: Inquiries were made about **Smaug-72B-LLM** outperforming giants like **Mistral 8x22B** and **Llaam-3 70B**, yet lacking the same level of fame, sparking a debate over the reasons behind its quiet presence despite strong benchmark results.

- **Help Wanted: Integrating AI with Blockchain**: A user expressed interest in integrating AI with blockchain, seeking to meld together these cutting-edge technologies, demonstrating the diverse interests and initiatives present within the community.

- **Resourceful Sharing and Collaboration**: The conversations were highlighted by members sharing youtube tutorials on fine-tuning Whisper, links to GitHub projects like [BabyTorch](https://github.com/amjadmajid/BabyTorch) for educational purposes, and URLs to helpful HF-related resources for working with LLMs and audio data. Community support and resource sharing were prevalent throughout the interactions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Mozilla-Ocho/llamafile">GitHub - Mozilla-Ocho/llamafile: Distribute and run LLMs with a single file.</a>: Distribute and run LLMs with a single file. Contribute to Mozilla-Ocho/llamafile development by creating an account on GitHub.</li><li><a href="https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384">timm/ViT-SO400M-14-SigLIP-384 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/posts">Hugging Face – Posts</a>: no description found</li><li><a href="https://huggingface.co/learn/audio-course/chapter5/fine-tuning">Fine-tuning the ASR model - Hugging Face Audio Course</a>: no description found</li><li><a href="https://github.com/getumbrel/llama-gpt">GitHub - getumbrel/llama-gpt: A self-hosted, offline, ChatGPT-like chatbot. Powered by Llama 2. 100% private, with no data leaving your device. New: Code Llama support!</a>: A self-hosted, offline, ChatGPT-like chatbot. Powered by Llama 2. 100% private, with no data leaving your device. New: Code Llama support! - getumbrel/llama-gpt</li><li><a href="https://github.com/amjadmajid/BabyTorch">GitHub - amjadmajid/BabyTorch: BabyTorch is a minimalist deep-learning framework with a similar API to PyTorch. This minimalist design encourages learners explore and understand the underlying algorithms and mechanics of deep learning processes. It is design such that when learners are ready to switch to PyTorch they only need to remove the word `baby`.</a>: BabyTorch is a minimalist deep-learning framework with a similar API to PyTorch. This minimalist design encourages learners explore and understand the underlying algorithms and mechanics of deep le...</li><li><a href="https://www.youtube.com/watch?v=O60EnXcbi6g)">OpenAI Whisper - Fine tune to Lithuanian | step-by-step with Python</a>: Fine-tune OpenAI&#39;s Whisper to different language is simple using Python and Google Colab with GPU. In this tutorial, I selected the small version of Whisper ...</li><li><a href="https://youtu.be/vhRx3rEOL60">A Multi-Agent game where LLMs must trick each other as humans until one gets caught</a>: Five top LLMs - OpenAI&#39;s ChatGPT, Google Gemini, Anthropic&#39;s Claude, Meta&#39;s LLAMA 2, and Mistral AI&#39;s Mixtral 8x7B compete in this text-based Turing Test gam...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1236992519775064064)** (6 messages): 

- **Look-alike Machine Learning Modelling Unpacked**: Tolulade shares an [educational post](https://tolulade-ademisoye.medium.com/introduction-to-look-alike-machine-learning-modelling-343290015c00) on look-alike modeling for beginners. The article blends informative content with engineering insights, also promoting a networking platform called Semis for AI and Big Tech.

- **Struggling with Step-by-Step LLMs**: A member tried to implement a "think step by step" process for their local Large Language Models (LLMs) but found that the models could not adapt from their fine-tuned, regular full answers.

- **Innovating Model Response Chain**: The same member encountered better success by creating a language chain sequence of `planner`, `writer`, `analyst`, and `editor` working in a loop, using Llama 3 instruct 7B, which provided more comprehensive results than zero-shot outputs.

- **Promoting Eco-friendly AI**: The importance of environmentally-conscious AI development was highlighted through a [YouTube video](https://youtu.be/j7x_tb_o1Nw) discussing Meta AI's open reporting of CO2 emissions and a related tool called [codecarbon](https://github.com/mlco2/codecarbon), which aims to estimate the carbon footprint of ML projects.

- **Learning the Ropes of Quantization**: A member shares their learning journey into quantization, both symmetric and asymmetric, which is an essential technique for optimizing machine learning models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tolulade-ademisoye.medium.com/introduction-to-look-alike-machine-learning-modelling-343290015c00">Introduction to Look-alike-Machine Learning Modelling</a>: In Transactional &amp; Marketing Email</li><li><a href="https://youtu.be/j7x_tb_o1Nw?si=mtRUCiam1Ujiy-FN">What Makes Environmentally Conscious genAI? An Overview</a>: A quick look at what it means to be environmentally conscious, as seen from @meta AI&#39;s example set with their open reporting of CO2 emissions from the LLaMA3...</li><li><a href="https://mlco2.github.io/impact/#computer">Machine Learning CO2 Impact Calculator</a>: Machine Learning has in impact on our climate. Here's how to estimate your GPU's carbon emissions
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1237020216853008475)** (12 messages🔥): 

- **Unveiling Quantum Virtual Servers**: A member shared a link to an intriguing resource, [Oqtant](https://oqtant.infleqtion.com/), which appears to be a platform related to quantum virtual servers. 
- **Revolutionizing RAG Pipeline Evaluation**: Efforts to evaluate Retrieval Augmented Generation pipelines are now boosted with the introduction of [Ragas Framework](https://docs.ragas.io/en/stable/), designed for assessing the performance of RAG applications and developing metrics-driven strategies.
- **Introspective Agents under the Spotlight**: A blog post on [Medium](https://medium.com/ai-artistry/introspective-agents-wi) discusses the potential of introspective AI agents within the LlamaIndex framework, aiming to improve AI’s self-evaluation and refinement capabilities.
- **Lilian Weng's AI Safety Blog**: [Lilian Weng](https://lilianweng.github.io/) blogs about her notes on AI learning and her work with the AI safety and alignment team at OpenAI, with a hint of humor about ChatGPT's help (or lack thereof) in her writing.
- **Innovating Image-Based Virtual Try-On**: The [IDM-VTON](https://arxiv.org/abs/2403.05139) model is presented in a new paper, aiming to enhance naturalness and garment identity preservation in virtual try-on applications using a novel diffusion approach.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lilianweng.github.io/">Lil&#39;Log</a>: Document my learning notes.</li><li><a href="https://arxiv.org/abs/2403.05139">Improving Diffusion Models for Virtual Try-on</a>: This paper considers image-based virtual try-on, which renders an image of a person wearing a curated garment, given a pair of images depicting the person and the garment, respectively. Previous works...</li><li><a href="https://oqtant.infleqtion.com/">Oqtant</a>: no description found</li><li><a href="https://medium.com/ai-artistry/introspective-agents-with-llamaindex-777d018f791d">Introspective Agents with LlamaIndex</a>: Ankush k Singal</li><li><a href="https://arxiv.org/abs/2401.03568">Agent AI: Surveying the Horizons of Multimodal Interaction</a>: Multi-modal AI systems will likely become a ubiquitous presence in our everyday lives. A promising approach to making these systems more interactive is to embody them as agents within physical and vir...</li><li><a href="https://docs.ragas.io/en/stable/">Introduction | Ragas</a>: no description found</li><li><a href="https://medium.com/ai-artistry/introspective-agents-wi">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1237022417595924593)** (8 messages🔥): 

- **Meet everything-ai: Your New AI Multi-Tasking Assistant**: everything-ai is an all-in-one local AI-powered assistant that can interact with PDFs, text, images, and more in over 50 languages. [The project's GitHub page](https://github.com/AstraBert/everything-ai) includes a quick-start guide and a new user interface, distributed as a Docker application.
  
- **Spaghetti Code Achieves Functionality**: Sparky 2 is a discord bot with image generation capabilities built on llama-cpp, described as a "spaghetti of python code" by its creator. The bot’s code is available on [GitHub](https://github.com/Gapi505/Sparky-2).

- **AI-Assisted Research Revolution**: Adityam Ghosh introduces EurekAI, a new tool aimed to streamline the research process. Those interested in contributing feedback can engage via user interviews by contacting the team on their website, [eurekai.tech](https://www.eurekai.tech/).

- **Seeking Beta Testers for Advanced Research Assistant and Search Engine**: Rubik's AI is looking for beta testers to try their premium search engine which includes access to state-of-the-art models like GPT-4 Turbo and Mistral Large. Interested parties can sign up and receive two months free premium using the promo code `RUBIX` at [rubiks.ai](https://rubiks.ai/).

- **AI Music Generation Goes 'nanners'**: A new death metal-dubstep track created using AI music generators and samples from Udio AI is shared for feedback. The track, "DJ Stomp - The Arsonist," can be heard on [YouTube](https://youtu.be/JPM1EacdpMs).

- **Real-Time Video Generation Demonstrated on Twitter**: A real-time AI-generated video at 17fps was shared, demonstrating the prompt-based control of the visual output, although without recorded audio. The post can be viewed on [Twitter](https://twitter.com/Dan50412374/status/1787936305751748844).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openalgos.ai/">OpenAlgos</a>: no description found</li><li><a href="https://github.com/Gapi505/Sparky-2">GitHub - Gapi505/Sparky-2</a>: Contribute to Gapi505/Sparky-2 development by creating an account on GitHub.</li><li><a href="https://astrabert.github.io/everything-ai">everything-ai</a>: Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖</li><li><a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: Introducing everything-ai, your multi-task, AI-powered and local assistant! 🤖</a>: Introducing everything-ai, your multi-task, AI-powered and local assistant! 🤖 - AstraBert/everything-ai</li><li><a href="https://youtu.be/JPM1EacdpMs">DJ Stomp - The Arsonist | Deathstep</a>: The Arsonist is a bit darker and more intense than any of my previous work, merging aggressive dubstep with death metal influences. Simply put, this track is...</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1236989157205737614)** (3 messages): 

- **Special Tokens Trigger Smarter Retrievals**: The discussion introduced the concept of teaching **Large Language Models (LLMs)** to use a special token `<RET>` to trigger information retrieval when uncertain. The [paper discussed](https://arxiv.org/abs/2404.19705) explores the use of this technique to improve LLM performance, especially vital for less frequent questions that the LLM's parametric memory can't handle.

- **When LLMs Should Look Up Info**: A new paper, promoted through a [tweet by @omarsar0](https://x.com/omarsar0/status/1785498325913108556?t=Mfnr02-d3Hn0J4vcH9KPNA&s=09), details a fine-tuning method that enables LLMs to decide when to retrieve extra context. This approach can lead to more accurate and dependable **Retrieve-And-Generate (RAG)** systems.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.19705">When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively</a>: In this paper, we demonstrate how Large Language Models (LLMs) can effectively learn to use an off-the-shelf information retrieval (IR) system specifically when additional context is required to answe...</li><li><a href="https://bytez.com/read/arxiv/2404.19705">Bytez: When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively</a>: In this paper, we demonstrate how Large Language Models (LLMs) can effectively learn to use an off-the-shelf information retrieval (IR) system specifically when additional context is required to answe...</li><li><a href="https://x.com/omarsar0/status/1785498325913108556?t=Mfnr02-d3Hn0J4vcH9KPNA&s=09">Tweet from elvis (@omarsar0)</a>: When to Retrieve?  This new paper presents an approach to train LLMs to effectively utilize information retrieval.  It first proposes a training approach to teach an LLM to generate a special token, &...</li><li><a href="https://youtu.be/gu5ttnClB5g?si=pTOTrcgsdMG6Q4mV">Training an LLM to effectively use information retrieval</a>: This new paper presents an approach to train LLMs to effectively utilize information retrieval.It first proposes a training approach to teach an LLM to gener...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1236943917728141352)** (15 messages🔥): 

- **Tinkering with Darknet Yolov4**: A member shared their intention to experiment with **Darknet Yolov4** by attaching a CNN to its tail for processing rescaled images and retraining the entire network, despite the model's age and the challenges in finding documentation for it.

- **Searching for the lost UA-DETRAC dataset**: The **UA-DETRAC dataset**, useful for computer vision and object detection from traffic cameras, has disappeared online and its annotations are no longer available through the website or Internet Archive. A community member from BSC facilities has requested assistance from anyone who might have previously downloaded it.

- **Freezing Convnext for Efficient Training**: A member inquired about using `AutoModelForImageClassification.from_pretrained()` with **convnext tiny** and whether it defaults to freezing the pretrained portion of the model. Another member provided advice, recommending explicitly setting the `requires_grad` attribute to `False` for the convolutional base parameters.

- **Training on Multi-label Image Classification**: A discussion on resources for training image classification models on multi-label data took place, with a member seeking advice on managing datasets where images may have multiple color labels ranging from 1 to 10 possible colors. A resource from Hugging Face's computer vision course was found but reported to not work.

- **Facenet vs VGG16 for Face Recognition and Keypoints Detection**: One member expressed the need for guidance on applying transfer learning to a pretrained facenet model for face recognition, while another was interested in finding models suitable for fine-tuning for cephalometric keypoint detection.

**Link mentioned**: <a href="https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/vision-transformers-for-image-classification#multi-label-image-classification">Transfer Learning and Fine-tuning Vision Transformers for Image Classification - Hugging Face Community Computer Vision Course</a>: no description found

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1236942917898207232)** (9 messages🔥): 

- **Flash Attention 2 in XLM-R Inquiry**: A member expressed interest in adding **Flash Attention 2** to XLM-R and is seeking guidance since it's not implemented in HuggingFace yet. There was a query about available **tutorials** or **guidelines** for implementation.
- **Request for Chatbot-Powered PowerPoint Generation**: A request was made for a chatbot capable of generating **PowerPoint presentations** using OpenAI Assistant API, learning from previous ones, and modifying only the slide content. Alternatives with RAG or LLM models were also asked for.
- **Adding Models to Transformers**: Discussion on the process for deciding which **new models** to integrate into HuggingFace Transformers. Community contributions are encouraged, with a suggestion to consider *Papers with Code* and other trending **SOTA models**.
- **Random Findings on Model Behavior**: A member shared that classifiers created by Moritz are efficient, and highlighted issues related to **probability distributions and model choices**, requesting more information for troubleshooting the issue.
- **Debugging Script Issues Across Different Cloud Clusters**: The community discussed the challenges of running scripts on different cloud environments and debugging peculiar errors such as the **None type error** in the sentence transformers' encode function. Debugging is emphasized as a valuable learning tool and crucial for resolving code issues.
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1236963318514716753)** (7 messages): 

- **Customize Image Models with Ease**: [Custom Diffusion](https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion) is a personalization technique for image generation models, requiring as few as 4-5 example images to learn concepts via cross-attention layers. For resource optimization, enabling xFormers with `--enable_xformers_memory_efficient_attention` is recommended for systems with limited vRAM, while the `--set_grads_to_none` flag can further reduce memory usage during training.
 
- **Conquering Device Mismatch in Model Offloading**: A user encountered errors when attempting to combine *Accelerate's* multi-GPU functionality with *Diffusers*' model CPU offloading, receiving "expect tensors to be on the same device" and "cannot copy from meta device" error messages.
 
- **Estimating AI Model Costs with Token Counting**: A user discusses billing considerations based on token counts using a [pricing calculator guide](https://docsbot.ai/tools/gpt-openai-api-pricing-calculator), highlighting that token-based billing is a standard practice, with 1,000 tokens approximately equating to 750 words in English.

- **BERT Model Training Distress Signal**: A user seeks help with BERT pretraining and fine-tuning, noting that while pretraining loss decreases normally, fine-tuning for sentiment analysis leads to overfitting within two epochs. They shared their [Colab notebook](https://colab.research.google.com/drive/1VImgKVih2703-vNPmQQu8CfoIxXRmutA?usp=sharing) for community input.

- **Stable Diffusion Finetuning Frustrations**: A member asked for tips on fine-tuning Stable Diffusion 1.5 with an intimate dataset of about 1300 examples, indicating a struggle to find effective hyperparameters.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docsbot.ai/tools/gpt-openai-api-pricing-calculator">OpenAI &amp; other LLM API Pricing Calculator - DocsBot AI</a>: Calculate and compare the cost of using OpenAI, Azure, Anthropic, Llama 3, Google Gemini, Mistral, and Cohere APIs with our powerful FREE pricing calculator.</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion#:~:text=Custom%20Diffusion%20is%20unique%20because%20it%20can%20also%20learn%20multiple%20concepts%20at%20the%20same%20time.)">Custom Diffusion</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1VImgKVih2703-vNPmQQu8CfoIxXRmutA?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion#">Custom Diffusion</a>: no description found
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1236936857099571211)** (168 messages🔥🔥): 

- **Confusion Over Beta Access**: A member clicked an icon expecting a form to appear, but it did not. Another member clarified that the beta is closed.
- **Technical Difficulties on Perplexity**: Multiple users reported issues with Perplexity's responsiveness, with problems logging in and out, unresponsive buttons, and slow loading times across various devices.
- **Questions on Model Limits and Types**: Several members inquired about daily limits on different models like **Claude 3 Opus** and **Sonar 32k**, discussing how these restrictions might affect their work, with references to official statements and updates on Perplexity's FAQ page.
- **Comparison Queries Between AI Models**: Users compared the capabilities and limitations of different models such as **GPT-4 Turbo**, **Sonar**, and **Opus** for various tasks, including essay writing, code refactoring, and learning from novel writing styles.
- **Seeking Clarification on Source Limits**: There was confusion around the source limit in searches, with members debating whether there had been an increase and sharing links to potentially relevant GIFs as responses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1047197230748151888/1047649527299055688/1230472581837230100">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://news.sky.com/story/china-hacked-ministry-of-defence-sky-news-learns-13130757">China hacked Ministry of Defence, Sky News learns</a>: MPs will be told of a massive data breach involving the Ministry of Defence later today, targeting service personnel. </li><li><a href="https://www.tiktok.com/@dnaturelovers?_t=8m88ov8QuoL&_r=1">TikTok - Make Your Day</a>: no description found</li><li><a href="https://tenor.com/view/thistest-test-gif-21687059">Thistest GIF - Thistest Test - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/scooby-doo-mystery-machine-cartoon-old-school-smoking-gif-16100024">Scooby Doo Mystery Machine GIF - Scooby Doo Mystery Machine Cartoon - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/skeleton-meme-gif-9417473421307480238">Skeleton Meme GIF - Skeleton meme - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1236985062822772808)** (19 messages🔥): 

- **Showcasing Perplexity's Collection**: Various users shared links to [Perplexity AI](https://www.perplexity.ai) exploring topics like the **US Air Force**, **AlphaGo's reasoning**, the game of **Go**, **tape-to-tape**, **image creation**, **Boeing**, **Microsoft's 500 billion parameter model**, **noise-canceling headphones**, and more.
- **Highlight on Intuitive Image Creation**: A user shared a [Perplexity AI link](https://www.perplexity.ai/search/create-an-image-GS9uJ17vSLCBMW5F7d2iGw) about *creating an image* and expressed a wish for a standard image creation UI.
- **Exploration of Tech and Trends**: Some users are investigating [Microsoft's new model](https://www.perplexity.ai/search/Microsoft-builds-500b-w_EGBxOzQJ22vgyqsJ2f8Q), **noise-canceling headphones**, and other diverse topics through Perplexity AI's search function.
- **Emphasis on Shareable Content**: Twice, Perplexity AI reminded users to ensure their threads are `Shareable`, emphasizing the importance of shareable content within the community.
- **XDream Features**: Links were shared to an XDream page, focusing on an [interface project](https://www.perplexity.ai/page/XDreams-We-Interface-T__fwlhLTKuUkWtClKT4Hg) and a feature titled [Insanity by XDream](https://www.perplexity.ai/page/Insanity-by-XDream-EAmzvXwkTf6bFgB9X3X2UQ).
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1237018407816265738)** (23 messages🔥): 

- **JSON Outputs Curation Ideas**: Members discussed workarounds for producing JSON formatted outputs, with one suggesting the use of simple config style in **llama3** by giving examples with *Explanation=Blah Blah Blah* and *Rating=Blah Blah Blah*.
- **Perplexity API Search Capabilities Scrutinized**: Users expressed difficulties with **Perplexity's online models**, particularly when trying to obtain up-to-date URLs for competitor landing pages, stating that results are often outdated or irrelevant.
- **Model Cards Guidance Updated**: An update to the **Perplexity documentation** on model cards was pointed out, specifying that system prompts do not affect retrieval process in online models and providing a [link to documentation](https://docs.perplexity.ai/docs/model-cards) for further details.
- **Debating Model Parameter Counts**: There is confusion and discussion around the parameter count for the **llama-3-sonar-large** models, and some debate if it actually uses Llama as a base due to its reported MoE-like structure but not being Llama.
- **Sonar Model Site Limitations**: Users inquired about the possibility of limiting **Perplexity's sonar model** outputs to specific sites, with attempts such as using *site:scholar.google.com* having inconsistent results.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found</li><li><a href="https://docs.perplexity.ai/">pplx-api</a>: no description found
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1236942321040359435)** (13 messages🔥): 

- **Implementing DoRALinear Modules**: A member highlighted the implementation of **`BNBDoRALinear` and `HQQDoRALinear`** modules with torch-only and fused forward passes, adapted from **FSDP-DoRA layers**. The code, which needs to be robust for training, can be found in the `Usage` and `Benchmarks` section of their PR.
- **Triton's Advantage in Kernel Design**: A member expressed the efficiency of **Triton** in designing kernels swiftly compared to high-level libraries like `Cutlass`, which would take more time. They also made minor tweaks to the **autotuner** for better debugging.
- **Bottlenecks and Fused GEMMs in DoRA Layer**: Detailed profiling of the **DoRA layer** was shared, pinning the most costly kernels to the base layer **matmul** and the combined `lora_B / lora_A matmul + 2-norm` operations. Two custom fused GEMMs were designed to optimize these bottlenecks, present in the `Profiling` section of their PR.
- **Autotuner Code Divergence Concerns**: There was a brief discussion about the **Triton autotuner**, with a member encouraging trying out the logging function, although there were concerns about code divergence due to other recent changes to the autotuner.
- **Understanding Triton Autotune with NCU Profiling**: A member inquired about the autotune functionality of **Triton**, asking if all configs are compiled and run for each specific input shape and the implications for profiling with tools like **ncu**.
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1237093396724584618)** (20 messages🔥): 

- **CUTLASS Stream-K Scheduling Clarification**: A discussion clarified that **Stream-K** is a load balancing scheme used in load balancing tiles for GEMM implementation, and it's conceptually independent of the rest of CUTLASS. It was mentioned that covering Stream-K could fit as a short section of a future talk but explaining the entire CUTLASS 2.0 API might be extensive.

- **Optimizing Element-Wise Operations on CUDA**: For optimizing element-wise operations, suggestions included using **fusions for multiple operations**, applying tricks from a referenced lecture 8 such as coarsening and hierarchical splitting, and utilizing `thrust::for_each` or `thrust::transform` from **Thrust** which could potentially achieve up to 90% of saturating bandwidth. A [link to Thrust documentation](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each) was provided for reference.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each">Modifying</a>: CUDA C++ Core Libraries</li><li><a href="https://nvidia.github.io/cccl/thrust/api/groups/group__transformations.html#function-transform">Transformations</a>: CUDA C++ Core Libraries
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1237032983316140054)** (2 messages): 

- **Troubleshooting Dynamic Shapes with PyTorch Compile**: A member suggests running PyTorch with the log option `TORCH_LOGS="+dynamic"` to diagnose errors indicating that `inputs_ids.shape[1] == 7` was expected for dynamic shapes. This log setting can shed light on whether user code or PyTorch framework code necessitated shape specialization during tracing.

- **Issue Posted on PyTorch GitHub**: A member has created an issue with a minimal example on PyTorch's GitHub, concerning the combination of Compile with DDP (Distributed Data Parallel) & dynamic shapes. The issue can be explored and additional information can be provided through the link: [pytorch/pytorch #125641](https://github.com/pytorch/pytorch/issues/125641).

**Link mentioned**: <a href="https://github.com/pytorch/pytorch/issues/125641>.">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch

  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1237123704392126574)** (2 messages): 

- **Boosting Transformer Efficiency**: A post by @p_nawrot introduces **Dynamic Memory Compression (DMC)**, a method to compress the KV cache in Large Language Models (LLMs) without sacrificing their performance, significantly improving throughput by up to 370% on a H100 GPU. They provided a [link to the paper](https://arxiv.org/abs/2403.09636) and stated that the code and models will be released soon.

- **Questioning Quantization**: In response to the KV cache compression technique, a member inquired whether the approach involves quantizing the model. No additional information or response was provided regarding this query.

**Link mentioned**: <a href="https://x.com/p_nawrot/status/1768645461689168365">Tweet from Piotr Nawrot (@p_nawrot)</a>: The memory in Transformers grows linearly with the sequence length at inference time.  In SSMs it is constant, but often at the expense of performance.  We introduce Dynamic Memory Compression (DMC) w...

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1237114531596079125)** (7 messages): 

- **Beginner's GPU Clock Speed Confusion**: A member was confused about the clock speed of **H100 GPUs**, stating it as 1.8 MHz and asking why the theoretical peak for fp64 is 34 TFLOPs. The difference pointed by another could imply a common error between MHz and GHz.
  
- **Correcting GPU Clock Speed Assumptions**: In the follow-up, the same member acknowledged the likelihood that they meant 1.8 GHz, a significantly higher clock speed than initially stated, yet could not find a source confirming this for **H100 GPUs**.

- **Inquiry on PyTorch Torch Compile for Triton**: A member asked how to call `torch.compile` for the **Triton** backend when using PyTorch, unsure whether `backend="inductor"` is the correct option.

- **Improving Model Performance in PyTorch**: The discussion touched on using **BetterTransformer** with `torch.compile` to optimize models in PyTorch. The member shared code snippets illustrating how to implement these optimizations in the **Hugging Face ecosystem**.
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1236967183829962802)** (4 messages): 

- **Confusion Over Tiling in Transpose**: A discussion in Chapter 5 exercises questioned the need for **tiling** in matrix transpose operations. Another member clarified that the purpose is to ensure **coalesced memory write** and shared a valuable [NVIDIA blog post](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) for better understanding.

- **Anticipated Knowledge Can Cause Puzzles**: Members noted that the book exercises sometimes include questions on topics not yet covered, leading to confusion. It’s indicated the topic of **coalescing** is expected to be discussed in a subsequent chapter.

**Link mentioned**: <a href="https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/">An Efficient Matrix Transpose in CUDA C/C++ | NVIDIA Technical Blog</a>: My last CUDA C++ post covered the mechanics of using shared memory, including static and dynamic allocation. In this post I will show some of the performance gains achievable using shared memory.

  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1237264519068385330)** (4 messages): 

- **Seeking the Next Video Meetup**: A member inquired about the schedule for the next video meetup.
- **Navigating to Event Information**: They were directed to find the video meetup details under the **Events** section, identifiable by the calendar icon.
  

---


**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1237474982699728979)** (1 messages): 

- **Exploring Multi-Chip Model Training**: A member shared a [blog post](https://simveit.github.io/high%20performance%20computing/multi-chip-performance/) discussing the necessity of training machine learning models over multiple chips. It focuses on efficiently using Google's TPUs, particularly on **Google Cloud**, with a visual example of **layerwise matrix multiplication** for workload distribution.

**Link mentioned**: <a href="https://simveit.github.io/high%20performance%20computing/multi-chip-performance/">Multi chip performance in JAX</a>: The larger the models we use get the more it becomes necessary to be able to perform training of machine learning models over multiple chips. In this blog post we will explain how to efficiently use G...

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1237220472865685524)** (8 messages🔥): 

- **Metal Memory Clarification Sought**: A member sought clarification on Metal & Apple Silicon, particularly on how to allocate buffers in context to Shared/Tile or Global/Unified/System Memory similar to CUDA's `__shared__` and `float global_array`. They found their answer in [Apple's documentation](https://developer.apple.com/documentation/metal/resource_fundamentals/choosing_a_resource_storage_mode_for_apple_gpus).
- **Metal Memory Access Query**: The same member asked whether all GPU threads can access `.memoryless` and `.private` memory categories in Metal on Apple Silicon, redirecting from a previous question about memory allocation.
- **Inquiry on Lightning AI Studio Feedback**: A user inquired if anyone had experience with Lightning AI Studio and if they could provide feedback.
- **Triton Language Presentation Proposal**: Discussion included a proposal for a presentation on OpenAI's [Triton language](https://github.com/openai/triton) and its application to ML inference, situated alongside clarifications to distinguish it from Nvidia's Triton Inference Server.
- **Reference to CUDA Mode YouTube for Triton Talk**: A member redirected others to CUDA MODE’s YouTube channel for a previous talk on Triton, which might include related content to the proposed presentation idea.



**Link mentioned**: <a href="https://github.com/openai/triton">GitHub - openai/triton: Development repository for the Triton language and compiler</a>: Development repository for the Triton language and compiler - openai/triton

  

---


**CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

glaxus_: Anyone going to be at MLSys?
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1236986822563725312)** (133 messages🔥🔥): 

- **Multi-GPU Training Hangs on Master**: An issue has been raised regarding multi-GPU training hanging on the master branch, presumably due to the introduction of `cudastreams`. The problem is documented in a GitHub issue [#369](https://github.com/karpathy/llm.c/issues/369) and contributors are looking into it, with some suggesting that self-hosted runners on GitHub could be used for real GPU testing in CI to avoid such issues.
  
- **Nsight Systems for Performance Analysis**: A link to NVIDIA Nsight™ Systems was shared for analyzing application algorithms and identifying optimization opportunities for CPUs and GPUs. The tool allows visualization of system workload on a timeline and can be used for both local and remote profiling as indicated by the [Nsight Systems download link for macOS](https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_3/NsightSystems-macos-public-2024.3.1.75-3419530.dmg).

- **Addressing Kernel Synchronization for Performance**: Suggestions have been made for the synchronization of kernels with train files, particularly converting all to `floatX` and standardizing documentation to make it easier for new collaborators. An example pull request, [#319](https://github.com/karpathy/llm.c/pull/319), demonstrates what the synchronization could look like.

- **Fine-Tuning CUDA Kernels for Better Performance**: There is an active discussion on the importance of finely optimizing CUDA kernels for performance improvements on specific GPU architectures, particularly for memory-bound operations. For instance, Gelu_backward appears to be memory-bound and could benefit from reworking that avoids needless functions, as suggested in [PR #307](https://github.com/karpathy/llm.c/pull/307).

- **HuggingFace Introduces FineWeb for Pretraining**: HuggingFace has released a dataset called FineWeb, boasting over 15 trillion tokens from cleaned and deduplicated English web data, optimized for LLM performance. The dataset's smallest subset is approximately 27.6GB, suitable for pretraining experiments with from-scratch models, and available at [HuggingFace datasets](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/Chillee/42e4635c59760a74cb3b4ba7ea5ad9f8">Strangely, Matrix Multiplications Run Faster When Given &quot;Predictable&quot; Data!</a>: Strangely, Matrix Multiplications Run Faster When Given &quot;Predictable&quot; Data! - mm_weird.py</li><li><a href="https://stackoverflow.com/questions/36047035/when-should-i-use-cudas-built-in-warpsize-as-opposed-to-my-own-proper-constant">When should I use CUDA&#x27;s built-in warpSize, as opposed to my own proper constant?</a>: nvcc device code has access to a built-in value, warpSize, which is set to the  warp size of the device executing the kernel (i.e. 32 for the foreseeable future). Usually you can&#x27;t tell it apart ...</li><li><a href="https://github.com/karpathy/llm.c/issues/369">MultiGPU training hangs · Issue #369 · karpathy/llm.c</a>: mpirun with multiple GPUs is hanging after allocated 474 MiB for master copy of params Most probably due to the introduction of cudastreams. @karpathy @PeterZhizhin</li><li><a href="https://github.com/karpathy/llm.c/pull/319">convert all float to floatX for layernorm_forward by JaneIllario · Pull Request #319 · karpathy/llm.c</a>: change all kernels to use floatX</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">HuggingFaceFW/fineweb · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/352">utilities for mixed-precision tests/benchmarks by ngc92 · Pull Request #352 · karpathy/llm.c</a>: This allows us to compile a single executable that can serve as test/benchmark for f32, f16, and bf16 versions of the kernels. So far, I&#39;ve updated only those test files which already defined a BF...</li><li><a href="https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners">Adding self-hosted runners - GitHub Docs</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/378">Optimisations for layernorm_backward / matmul_backward_bias / fused_classifier by ademeure · Pull Request #378 · karpathy/llm.c</a>: These are fairly difficult optimisations to describe, hopefully the comments are helpful/enough! I&#39;d focus on the changes in train_gpt2.cu rather than the similar ones in /dev/cuda/ (I didn&#39;t ...</li><li><a href="https://github.com/karpathy/llm.c/pull/307">Improve tanh derivative in backward gelu by akbariyeh · Pull Request #307 · karpathy/llm.c</a>: It is cheaper to compute the derivative of tanh as 1 - tanh^2 than computing 1/(cosh^2). This will probably not make a measurable difference.</li><li><a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html">cuda::memcpy_async</a>: CUDA C++ Core Libraries</li><li><a href="https://developer.nvidia.com/nsight-systems">NVIDIA Nsight Systems</a>: Profile systems, analyze performance, and optimize platforms.</li><li><a href="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_3/NsightSystems-macos-public-2024.3.1.75-3419530.dmg">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[oneapi](https://discord.com/channels/1189498204333543425/1233802893786746880/1237279577680646196)** (4 messages): 

- **GitHub Pull Request Sighted**: A member shared a [GitHub pull request](https://github.com/pytorch/pytorch.github.io/pull/1596#pullrequestreview-2041058328) for **PyTorch.org**. The pull adds an accelerators dropdown to the quick start table with options for Huawei Ascend, Intel Extension for PyTorch, and Intel Gaudi. 

- **PyTorch.org Preview Unveiled**: A preview link to [PyTorch.org](https://deploy-preview-1596--pytorch-dot-org-preview.netlify.app/) was shared, highlighting the **PyTorch Conference 2024 call for proposals**, early bird registration, new features in **PyTorch 2.3**, membership information, and ecosystem details. The preview also emphasizes TorchScript, TorchServe, torch.distributed backend, and cloud platform support for PyTorch.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://deploy-preview-1596--pytorch-dot-org-preview.netlify.app/">
    
      PyTorch
    
  </a>: no description found</li><li><a href="https://github.com/pytorch/pytorch.github.io/pull/1596#pullrequestreview-2041058328">Add accelerators to quick start table by aradys · Pull Request #1596 · pytorch/pytorch.github.io</a>: Create accelerators dropdown with following options and add it to quick start table:   Huawei Ascend Intel Extension for PyTorch Intel Gaudi   Add commands to previous versions section  RFC: pytorc...
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1237420491518771331)** (1 messages): 

- **OpenAI Shares Its Data Doctrine**: OpenAI has outlined its [approach to content and data](https://openai.com/index/approach-to-data-and-ai) in the age of AI. This crucial document details how they handle the vast amount of data in today's landscape and what ethical principles they adhere to.
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1236938669219119114)** (87 messages🔥🔥): 

- **Exploring AI's Music Chops**: A member mentioned [musician Davie504 jamming with AI music](http://drinkoblog.weebly.com/), suggesting the quality of AI music has become compelling.
- **Perplexity: A Hidden Gem in AI**: A member expressed astonishment upon discovering *Perplexity*, regretting not using it sooner and stating, "why haven't I used this sooner."
- **Cosine Similarity Thresholds in Embedding Models**: One member inquired about the appropriate cosine similarity threshold for the model text-embedding-3-small, noting that the "old 0.9" threshold might equate to "new 0.45" in newer models.
- **Local LLM Model Recommendations for 8GB VRAM**: When a member sought recommendations for an LLM model suitable for an 8GB VRAM card, it was suggested that Llama8B runs well and can be tried using LM Studio.
- **AI News and Updates**: Members shared different sources for staying updated on AI trends with suggestions including engaging with community members, following relevant Twitter accounts, and browsing sites like [OpenAI Community](https://community.openai.com) and [Ars Technica](https://arstechnica.com/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://community.openai.com">OpenAI Developer Forum</a>: Ask questions and get help building with the OpenAI platform</li><li><a href="https://arstechnica.com/">Ars Technica</a>: Serving the Technologist for more than a decade. IT news, reviews, and analysis.
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1237038163411865682)** (15 messages🔥): 

- **Seeking GPT Knowledge Prioritization**: A member expressed difficulty with GPT's preference for general answers rather than accessing specific entries in its knowledge base. It was suggested that telling GPT to refer to its "Analyze utility" might prompt a more accurate response.
  
- **GPT's Knowledge Base Explained**: Clarification was provided on how GPT's knowledge base works, with the process described as splitting knowledge into chunks, converting them into mathematical vectors, and searching these within a vector database to match prompts.

- **The Challenge of Vector Visualization**: There was a brief discussion on the complexities of visualizing vectors in the context of GPT's knowledge base, noting the challenge due to the vectors being in 256 dimensions.

- **Uniform GPT-4 Performance Across Demand**: Another member noted that GPT-4's performance does not vary with demand, everyone gets the same "turbo model" regardless of usage.

- **No Inferior Models to Handle Demand**: A final comment debunked the idea that an inferior model is used to manage demand, suggesting it's more cost-effective to invest in servers rather than a less desired model.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1236966287452930059)** (33 messages🔥): 

- **Struggle with Twitter Data**: A member tried using RAG for incorporating Twitter data as knowledge for the LLM but found the model lost generalization. A suggestion was to explore Custom GPT or web-browser based GPT solutions that may support such functionality today.
- **Negative Prompting Pitfalls**: There was discussion regarding the difficulty of integrating negative examples into prompt engineering, with suggestions to consider it as an "advanced prompt engineering" technique due to its complex nature and possible impact on response quality.
- **Prompt Expansion Advice for Product Identification**: For complex tasks such as identifying product details in supermarket brochures, it was advised to split the workload into multiple prompts instead of one, using outputs from previous prompts as inputs for subsequent ones for better results.
- **Challenges with DALL-E Prompt Specificity**: Members discussed DALL-E's struggle with negative prompts, noting that specifying what not to include, e.g., "no avocado pits", can lead to confusion. A channel with experienced DALL-E users was recommended for further advice.
- **Scripting for Improved Responses with Logit Bias**: To achieve more consistent output without random bullet points, it was proposed to use output templates with open variable names in pseudocode format, and for more control, investigate logit bias which requires following a procedure in the provided link.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1236966287452930059)** (33 messages🔥): 

- **GPT Models Struggle with Tailored Responses**: A member asks for advice on incorporating Twitter data into ChatGPT's knowledge to address specific prompts. They express difficulty with the LLM model using RAG and its inability to respond to inquiries beyond Twitter data.
- **Prompt Engineering Best Practices**: Madame_architect recommends a LinkedIn Learning course by Ronnie Sheer, *Prompt Engineering with ChatGPT*, for learning basics to advanced techniques, including Socratic prompting and Dall-E interactions.
- **DALL-E's Difficulty with Negative Prompts**: It is discussed that DALL-E often struggles with negative prompts, for instance not including avocados with pits, even when explicitly mentioned in the prompt as unwanted.
- **Tackling Unwanted Tokens via Logit Bias**: Link to the ([OpenAI logit bias guide](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api)) is shared to address the issue of an AI producing inconsistent outputs with unwanted tokens.
- **Step-wise Improvement in API Prompt Responses**: Madame_architect shares a multi-step approach to improve the response from GPT API for analyzing and formatting product information, suggesting separating vision tasks and format structuring into distinct API calls.
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1236939233340555374)** (39 messages🔥): 

- **ICLR Attendance with a Focus on AI Interpretability**: A member mentioned they would be attending ICLR, with work focused on **Interpretability in generative AI**, particularly on the vision side and **3D vision**.

- **Greeting from Thailand**: A simple *Hello* was shared by a member from Thailand.

- **Testing the Impact of System Prompts on Model Performance**: A member expressed interest in using **lm-evaluation-harness** to evaluate how varying system prompts affect a model's performance, using models such as llama3 8b instruct, and inquired on how to specify system prompts for **Hugging Face models** in the harness.

- **Italian Language Leaderboard in lm-eval**: A member is evaluating **Large Language Models (LLMs)** in Italian and maintaining a leaderboard using **lm-evaluation-harness**.

- **Query on Obtaining MMLU Data**: A student member asked about acquiring granular **MMLU data** for any closed-source model, as they exhausted their eval credits on attempting to evaluate GPT-4. They mention finding a CSV file but it lacked model-specific answers.

- **Discussion on PEFT with Available VRAM**: Members discussed whether **Prompt-based Efficient Fine-Tuning (PEFT)** using **LoRA** would be beneficial when sufficient VRAM is available, with some insight that perhaps the implementation could be suboptimal or specific configurations may affect performance, such as mixed precision settings and the R value in LoRA.
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1236990487320334346)** (77 messages🔥🔥): 

- **Evaluating Scientific Standards**: A member discussed that accepted scientific community standards, such as the .05 p-value threshold, may seem arbitrary but served practical purposes historically. The **Nature article** [Lowering the P-Value Threshold](https://www.nature.com/articles/s41562-017-0189-z) was shared, advocating for a shift from 0.05 to 0.005 to improve the reproducibility of scientific findings.

- **The P-Value Debate Continues**: Conversations ensued highlighting the arbitrary nature of the p-value threshold, with references to its historical context involving Fisher, Neyman, and the development of statistical tests. Discrepancies between physics and other fields, the influence of data on outcomes, and misunderstandings of uncertainty by the general public were key points of discussion.

- **SOPHON Framework Introduced**: A new learning paradigm called **non-fine-tunable learning** was shared, which aims to prevent pre-trained models from being misused for unethical tasks. The [SOPHON framework](https://arxiv.org/abs/2404.12699) is designed to protect pre-trained models from being fine-tuned in restricted domains.

- **QuaRot's Novel Quantization Scheme**: A link was provided to **QuaRot**, a new quantization approach for language models (LLMs) detailed in [an arXiv paper](https://arxiv.org/abs/2404.00456), which claims to maintain performance while applying comprehensive 4-bit quantization to weights, activations, and KV cache.

- **Mixture-of-Experts Architecture Lory Unveiled**: Discussion on a new MoE model called **Lory** was introduced, with a focus on a differentiable architecture [pre-trained on 150B tokens](https://arxiv.org/abs/2405.03133). It includes novel techniques for causal segment routing and similarity-based data batching for expert specialization.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hehao13.github.io/projects-CameraCtrl/">CameraCtrl</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.12699">SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Models</a>: Instead of building deep learning models from scratch, developers are more and more relying on adapting pre-trained models to their customized tasks. However, powerful pre-trained models may be misuse...</li><li><a href="https://en.wikipedia.org/wiki/Lady_tasting_tea">Lady tasting tea - Wikipedia</a>: no description found</li><li><a href="https://xkcd.com/882/">Significant</a>: no description found</li><li><a href="https://openreview.net/forum?id=aN4Jf6Cx69">The mechanistic basis of data dependence and abrupt learning in an...</a>: Transformer models exhibit in-context learning: the ability to accurately predict the response to a novel query based on illustrative examples in the input sequence, which contrasts with...</li><li><a href="https://arxiv.org/abs/2405.03133">Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training</a>: Mixture-of-experts (MoE) models facilitate efficient scaling; however, training the router network introduces the challenge of optimizing a non-differentiable, discrete objective. Recently, a fully-di...</li><li><a href="https://www.nature.com/articles/s41562-017-0189-z">Redefine statistical significance - Nature Human Behaviour</a>: We propose to change the default P-value threshold for statistical significance from 0.05 to 0.005 for claims of new discoveries.</li><li><a href="https://arxiv.org/abs/2404.00456">QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs</a>: We introduce QuaRot, a new Quantization scheme based on Rotations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits. QuaRot rotates LLMs in a way t...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages): 

nullonesix: https://arxiv.org/abs/2102.01293
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1237179035944091749)** (34 messages🔥): 

- **Exploring Skip Connections**: A member is experimenting with **adaptive skip connections** where weights reduce and can even become negative during training, resulting in improved loss metrics compared to a standard model. They provided [experimental results](https://wandb.ai/nickcdryan/bench/reports/train_loss-24-05-07-01-06-58---Vmlldzo3ODM5Njcx) and are asking for related research on this phenomenon.

- **Investigating Weight Dynamics**: Another member shared a related paper on gating the residual path [https://arxiv.org/pdf/2003.04887](https://arxiv.org/pdf/2003.04887), although it differs from the experiment in not restricting the identity component to positive values.

- **Code and Clarifications**: Further clarification was sought on what was meant by "identity component" in the experiments. The member [shared code](https://gist.github.com/nickcdryan/08c059ec3deb3ef2aca881bdc4409631) that reveals the use of a single weight on the residual connection in a transformer layer.

- **Dataset and Model Details Revealed**: The models used in the experiments have 607M parameters and were trained on the [fineweb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb) with a batch size of 24 and a learning rate of 6e-4 on a context window of 768, all run on a single A100.

- **Discussion on Training Speed and Odd Loss Curves**: A member discussing the experiment noted that the loss curve seemed odd and was going down too slowly, while another suggested trying the experiment on the OpenWebText dataset to compare results.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/nickcdryan/bench/reports/train_loss-24-05-07-01-06-58---Vmlldzo3ODM5Njcx">train_loss (24/05/07 01:06:58)</a>: Publish your model insights with interactive plots for performance metrics, predictions, and hyperparameters. Made by Nick Ryan using Weights &amp; Biases</li><li><a href="https://gist.github.com/nickcdryan/08c059ec3deb3ef2aca881bdc4409631">gist:08c059ec3deb3ef2aca881bdc4409631</a>: GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1237054020506484836)** (11 messages🔥): 

- **Logits Support Still Unavailable**: Despite interest in logit access for model understanding, API models like OpenAI's still do not support logits or the use of logit biases after recent research indicated the potential to extract a model's "image" or "signature" from them ([logit extraction work](https://arxiv.org/abs/2403.09539)). This affects the ability to perform evaluations based on log likelihoods and has led to the omission of log probs for *input tokens* in API responses.
- **Evaluation Workaround for Closed Models**: Encouraging attempts to evaluate **Italian LLMs** against models like **GPT-3.5** were shared, including a YAML setup for utilizing 'generate_until' instead of logits for tasks like MMLU and ARC, indicating a workaround for external evaluations on closed models.
- **OpenAI Docs Suggests Logprob Return**: A member noted that OpenAI's documentation implies the return of logprobs, highlighting the complexity in running external evaluations on proprietary models.
- **Clarification on Logprob Availability**: It was clarified that the limitation is the absence of logprobs for *prompt/input tokens*, which is crucial for calculating the completeness of a multiple-token response from models in evaluation scenarios.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.06634">Stealing Part of a Production Language Model</a>: We introduce the first model-stealing attack that extracts precise, nontrivial information from black-box production language models like OpenAI&#39;s ChatGPT or Google&#39;s PaLM-2. Specifically, our...</li><li><a href="https://arxiv.org/abs/2403.09539">Logits of API-Protected LLMs Leak Proprietary Information</a>: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/flan_cot_zeroshot/_mmlu_flan_cot_zeroshot_template_yaml">lm-evaluation-harness/lm_eval/tasks/mmlu/flan_cot_zeroshot/_mmlu_flan_cot_zeroshot_template_yaml at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1237156300694225017)** (39 messages🔥): 

- **Installation Inquiry**: A user inquired about how to install **mojo** on **Intel Mac OS via Docker**, but no direct solution was provided.
- **Mojo on Windows**: Users discussed alternatives to using **mojo** on **Windows**, with a suggestion to utilize **WSL2** for a comparable experience. Patience was expressed humorously through a gif implying a wait for native Windows support.
- **Discussion on Mojo's Design Choices**: Members debated the rationale behind having both **structs and classes** in **mojo**, touching on design patterns, and addressing concerns over the decision to include both constructs.
- **Mojo's Compilation Capabilities**: There was clarification that **mojo** can compile to native machine code, similar to **Rust**, and can produce executable files like **.exe**. Concerns about the speed of interpretation in data engineering tasks, compared to the immediacy of languages like **Python**, were addressed.
- **Python Integration with Mojo**: Users explained how **mojo** integrates with Python by importing modules and calling functions, referencing official documentation to demonstrate that **mojo** aims to be a superset of **Python** that leverages the established Python ecosystem while introducing its own functionalities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/its-gonna-take-a-while-steve-kornacki-msnbc-it-will-take-some-time-take-a-long-time-gif-19744456">Its Gonna Take A While Steve Kornacki GIF - Its Gonna Take A While Steve Kornacki Msnbc - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.modular.com/mojo/manual/python/#python-environment">Python integration | Modular Docs</a>: Using Python and Mojo together.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1237164440089006110)** (2 messages): 

- **Modular Shares Two Important Updates**: The **Modular** team tweeted important updates shared via two separate Twitter posts.
- **Catch the Latest From Modular**: For more information, follow the provided links to [Tweet 1](https://twitter.com/Modular/status/1787601254459519395) and [Tweet 2](https://twitter.com/Modular/status/1787601269668143500) on Modular's official Twitter page.
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1237145345541017682)** (1 messages): 

- **Modular Community Livestream Alert**: A new video titled "Modular Community Livestream - New in MAX 24.3" has been shared, announcing the availability of **MAX 24.3**. The livestream delves into the latest updates on the MAX Engine and introduces the new Extensibility API for Mojo🔥. [Watch now](https://www.youtube.com/watch?v=kKOCuLy-0UY).

**Link mentioned**: <a href="https://www.youtube.com/watch?v=kKOCuLy-0UY">Modular Community Livestream - New in MAX 24.3</a>: MAX 24.3 is now available! Join us on our upcoming livestream as we discuss what’s new in MAX Engine and Mojo🔥 - preview of MAX Engine Extensibility API for...

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1236947750331875358)** (53 messages🔥): 

- **Troubleshooting Tensor Indexing**: A user encountered an error when setting the value of a tensor at a specific location using indexing, which was resolved by using the `Index` utility from `utils.index`. Code from a tensor library demonstrates setting tensor values at specified indices correctly without errors.
  
- **Optimizing SIMD Operations on Large Arrays**: A member grappled with using SIMD for large arrays and experienced compilation issues with arrays beyond size 2^14. Another user advised utilizing smaller SIMD blocks and provided example code from [Mojo's GitHub repository](https://github.com/modularml/mojo/tree/main/examples).

- **Benchmarking Dilemmas**: A user raised concerns about benchmarking functions where computed values are not used, potentially causing optimizations that eliminate the function itself. The issue was addressed by suggesting the use of `benchmark.keep` to retain results and prevent such optimizations.

- **Curiosity on Constructors Without Class Inheritance**: A member questioned the need for constructors in Mojo, as the language lacks classes and inheritance. The conversation touched upon the differentiation between Mojo and other languages like Go and Rust, emphasizing constructors as a means to ensure instances are valid upon creation.

- **Exploration of Advanced Compiler Tools**: Participants in the chat discussed the necessity for tools to reveal compiled code details, such as LLVM IR, to eliminate guesswork. Interest was expressed in tools equivalent to compiler explorer for Mojo or the ability to view lower-level representation of compiled Mojo code.

- **Proposal for 'where' Clause in Mojo**: A discussion took place regarding a [GitHub proposal](https://github.com/modularml/mojo/issues/1245#issuecomment-2097388163) for parameter inference in Mojo functions, debating the implementation and readability of right-to-left inference rules versus the use of `where` clauses similar to those found in mathematics and the Swift programming language. Participants shared differing opinions on the most intuitive approach for specifying parameter constraints.

- **Compile-time Metaprogramming in Mojo**: A user inquired about the extent of compile-time metaprogramming in Mojo, particularly whether calculations like the Fibonacci sequence could be performed at compile time. Confirmation was given that such computations are indeed possible, with the caveat that there should be no side effects involved.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/engine/reference/mojo/graph/ops/casting/transpose">transpose | Modular Docs</a>: transpose(input Int, y: Int) -&gt; Symbol</li><li><a href="https://github.com/modularml/mojo/issues/1245#issuecomment-2097388163">[Feature Request] Parameter Inference from Other Parameters · Issue #1245 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? Mojo already supports inferring parameters from argume...</li><li><a href="https://github.com/basalt-org/basalt/blob/main/basalt/utils/tensorutils.mojo#L744">basalt/basalt/utils/tensorutils.mojo at main · basalt-org/basalt</a>: A Machine Learning framework from scratch in Pure Mojo 🔥 - basalt-org/basalt</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/inferred-parameters.md#segmented-parameter-lists.">mojo/proposals/inferred-parameters.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/tree/main/examples">mojo/examples at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/mo">mo - Overview</a>: mo has 49 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1236989454938411088)** (14 messages🔥): 

- **Mojo-sort Updated and Enhanced**: The [mojo-sort](https://github.com/mzaks/mojo-sort) project has been updated to work with the latest Mojo nightly. It now includes a more efficient radix sort algorithm for strings, boasting faster speeds across all benchmarks.
- **Help Needed with Lightbug Migration**: The [Lightbug](https://github.com/saviorand/lightbug_http) project is facing issues migrating to Mojo version 24.3, particularly concerning errors that appear to log EC2 locations. Assistance is requested by the developers, with details documented in [this GitHub issue](https://github.com/saviorand/lightbug_http/issues/34).
- **Basalt Navigates Mojo's Limitations**: The Basalt project adapts to Mojo's current limitations, like the lack of classes and inheritance, by finding workarounds such as using StaticTuple for compile-time lists, but it generally hasn't limited the overall goals.
- **A New Port of Minbpe to Mojo**: [Minbpe.mojo](https://github.com/dorjeduck/minbpe.mojo), a Mojo port of Andrej Kathpathy's Python project, has been released. Although currently slower than its Rust counterpart, it runs three times faster than the original Python version, and there's potential for optimization, including possible future SIMD implementations.
- **Mojo GUI Library Inquiry**: A member expressed interest in finding out if a Mojo GUI library exists, to which there has been no response within the given messages.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/saviorand/lightbug_http/issues/34">Client tests don&#39;t work with changes in Mojo 24.3 · Issue #34 · saviorand/lightbug_http</a>: Since Mojo 24.3 main() functions inside packages are no longer supported. This was used in /tests/run.mojo to run a test suite (which is just one client test for now). The client test worked by run...</li><li><a href="https://github.com/dorjeduck/minbpe.mojo">GitHub - dorjeduck/minbpe.mojo: port of Andrjey Karpathy&#39;s minbpe to Mojo</a>: port of Andrjey Karpathy&#39;s minbpe to Mojo. Contribute to dorjeduck/minbpe.mojo development by creating an account on GitHub.</li><li><a href="https://github.com/mzaks/mojo-sort">GitHub - mzaks/mojo-sort</a>: Contribute to mzaks/mojo-sort development by creating an account on GitHub.</li><li><a href="https://github.com/guidorice/mojo-pytest/blob/main/example/tests/util.mojo">mojo-pytest/example/tests/util.mojo at main · guidorice/mojo-pytest</a>: Mojo test runner, pytest plugin (aka pytest-mojo). Contribute to guidorice/mojo-pytest development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1237060908409421834)** (16 messages🔥): 

- **Type Handling Woes in Traits and Variants**: Discussions around handling return types in a programming context surfaced, mentioning the need for a `Never` type or macro trick similar to Rust to pass return type checks. Some members see current solutions as band-aids, emphasizing the need for more explicit language constructs like keywords.
  
- **Variant Types Challenged by Trait Inheritance**: A member is running into issues with creating variant types with trait inheritance, highlighting a current limitation flagged in the [associated GitHub issue](https://github.com/modularml/mojo/issues/2413). They are also exploring alternative methods like `PythonObject` with `@staticmethods` to avoid using `UnsafePointer`.

- **Nightly Compiler Release & Reference Ergonomics Improvement**: The Mojo Discord announces a [new nightly release of the Mojo compiler](https://github.com/modularml/mojo/pull/2559/files) along with enhancements to the ergonomics of `Reference`, which have simplified understanding and usage.

- **Request for Automating Nightly Release Notifications**: A question about automating messages for nightly releases led to a response from a member stating that while it is possible, it has not been prioritized enough to implement yet.

- **Monitors Stretched by the Latest Update**: A user humorously comments that their 2k monitor is just enough to handle the scope of the latest updates, implying the significance of the changes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/2413">[Feature Request] Allow substitution of child traits for parent traits · Issue #2413 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? If a function takes variadic arguments bound by a trai...</li><li><a href="https://github.com/modularml/mojo/pull/2559/files">[stdlib] Update stdlib corresponding to 2024-05-06 nightly/mojo by JoeLoser · Pull Request #2559 · modularml/mojo</a>: This updates the stdlib with the internal commits corresponding to today&#39;s nightly release: mojo 2024.5.622.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/pull/2559/commits/f514304fdb839cdcee8804e6531a5defae4a8ff7">[stdlib] Update stdlib corresponding to 2024-05-06 nightly/mojo by JoeLoser · Pull Request #2559 · modularml/mojo</a>: This updates the stdlib with the internal commits corresponding to today&#39;s nightly release: mojo 2024.5.622.
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1237291023751843901)** (1 messages): 

- **Soliloquy L3 Price Drop**: The price for using [**Soliloquy L3 8B**](https://openrouter.ai/models/lynn/soliloquy-l3) model has been reduced to **$0.05/M tokens** for both private and logged endpoints for the year 2023 - 2024.

**Link mentioned**: <a href="https://openrouter.ai/models/lynn/soliloquy-l3)">Lynn: Llama 3 Soliloquy 8B v2 by lynn | OpenRouter</a>: Soliloquy-L3 v2 is a fast, highly capable roleplaying model designed for immersive, dynamic experiences. Trained on over 250 million tokens of roleplaying data, Soliloquy-L3 has a vast knowledge base,...

  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1237072970636197928)** (1 messages): 

- **Beta Testers Wanted for Rubik's AI**: Users are invited to become beta testers for **Rubik's AI**, a new advanced research assistant and search engine. Potential testers will receive 2 months free of premium features including access to **Claude 3 Opus, GPT-4 Turbo, Mistral Large**, and other cutting-edge models, by signing up at [rubiks.ai](https://rubiks.ai/) with the promo code `RUBIX`.
- **Tech World Buzz - Apple and Microsoft Innovations**: In the app's trending topics, there are updates on **Apple's latest iPad models for 2024** and news about **Microsoft developing a 500b parameter model named MAI-1**, competing with OpenAI. Users can access these stories within the app for detailed insights.

**Link mentioned**: <a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1236952106796318760)** (119 messages🔥🔥): 

- **Search for Elusive Verbose Llama**: Members discussed challenges in getting **llama-3-lumimaid-8b** to generate longer responses, comparing its shorter outputs unfavorably with previous experiences from models like Yi and Wizard.
- **Meta-Llama-3-120B-Instruct Reveal**: Conversation about the launch of **Meta-Llama-3-120B-Instruct** on Hugging Face, a model inspired by large self-merges such as **Goliath-120b** and **MegaDolphin-120b**; member shared link to a tweet regarding its release.
- **Amazon Bedrock’s Model Request Regional Restrictions**: A user questioned the possibility of regional restrictions when requesting models on **Amazon Bedrock** from different billing regions, with others suggesting that *requesting access to another region seems possible*.
- **OpenRouter’s Precision Preference**: Dialogue on whether OpenRouter runs models at *full precision* becomes clear it varies with provider, mostly at **fp16**, and sometimes quantized to **int8**.
- **Model Optimization and Parameter Tuning**: Inquiry about whether **OpenRouter's default parameters** for models are typically suitable, eliciting opinions and experiences on the need for parameter adjustments for better conversational outcomes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://infiniteworlds.app)">no title found</a>: no description found</li><li><a href="https://huggingface.co/mlabonne/Meta-Llama-3-120B-Instruct">mlabonne/Meta-Llama-3-120B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://help.aidungeon.com/faq/what-are-the-different-ai-language-models">What are the different AI models in AI Dungeon?</a>: no description found</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/177x0cs/entire_system_prompt_leaks_in_data_export/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/177x0cs/e">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B">abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/docs#limits">OpenRouter</a>: Build model-agnostic AI apps
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1236948350130061372)** (39 messages🔥): 

- **Troubleshooting Local Interpreter Issues**: A user experienced an error when running **Interpreter with Mixtral locally**. Another sought to understand the capabilities of **OpenInterpreter**, comparing their personal experience with the project's impressive demo video.
- **Phi Performance Evaluations**: A discussion took place regarding the use of a **Hugging Face model** (Phi-3-Mini-128k-Instruct) on **OpenInterpreter**; Mike mentioned having had "**very** poor performance" using Phi.
- **Interest in Benchmarks for Model Performance**: Members are interested in benchmarking various models to see which work best with **Open Interpreter**, suspecting that models with less synthetic data may perform better.
- **Calling for Model and Framework Nicknames**: Suggestions for nicknaming model+framework combinations arose after a user shared positive feedback on their experience with **Gemma-over-Groq (GoG)**.
- **Request for Custom System Instructions**: There was a call for assistance regarding **Custom/System Instructions for Ubuntu** using **GPT-4**, indicating a need in the community for such resources.
- **Recommendation for Pre-Fine Tuning**: In response to a question about training models for a specific task, it was suggested to use **[OpenPipe.ai](https://openpipe.ai/)**, also mentioning the importance of understanding and using **system messages** effectively prior to fine-tuning a model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openpipe.ai/">OpenPipe: Fine-Tuning for Developers</a>: Convert expensive LLM prompts into fast, cheap fine-tuned models.</li><li><a href="https://huggingface.co/microsoft/">microsoft (Microsoft)</a>: no description found</li><li><a href="https://tenor.com/view/what-hello-yes-say-what-gif-445039305839444057">What Hello GIF - What Hello Yes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://rubiks.ai/search/?id=2doji3-eejo-88bg-v35a-sz678y8bv5y1">What is Reka Core?</a>:  **Reka Core** is a frontier-class, multimodal language model developed by Reka. It is one of only two commercially available comprehensive multimodal solutions, capable of processing and understandin...</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1236978783622529096)** (80 messages🔥🔥): 

- **Open Interpreter Version Specificity**: It was highlighted that Open Interpreter (OI) should run under Python 3.10 to avoid compatibility issues. One user encountered slow performance when running Groq and was advised to switch to a smaller model, such as *dolphin or mixtral*, for better experience.

- **Conda for Clean Installations on Mac**: Users discussed creating a separate Conda environment with Python 3.10 for a fresh installation of Open Interpreter, after facing multiple version conflicts with Python and errors.

- **Local Model Framework Inquiry**: A user inquired about supporting *Jan* as a local model framework for the O1 device, similar to its use with Open Interpreter (text). It was affirmed that there should be no issues so long as the models are served similarly.

- **01 Device International Availability**: Concerning the 01 device's service availability, it was noted that while the device works anywhere with internet access, the hosted service is likely US-only at present, and no units have been shipped out yet.

- **LLM Skill Persistence and Execution Issues**: A user noted the importance of persisting learned skills to avoid re-teaching the same information to language models. They were directed to the Open Interpreter's GitHub for information on how skills persist in storage. Additionally, issues with running code using a smaller language model were mentioned and addressed in a video reference.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/core/computer/skills/skills.py">open-interpreter/interpreter/core/computer/skills/skills.py at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1236965432670289983)** (35 messages🔥): 

- **Sora's Open Source Alternative Launched**: An open-source alternative to Sora with an MIT license has been released on Github, providing a tool for creating magic stories. However, it's noted that the weights are not released yet. [Discover the tool here](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file).
- **Gradient Checkpointing Memory Savings**: A member reported successful use of the new unsloth gradient checkpointing, seeing a reduction in VRAM usage from 19,712MB to 17,427MB, indicating significant memory savings.
- **Unusual Training Patterns in Model Layers**: During a discussion, a puzzling finding was pointed out where only a single slice of a model layer seemed to be getting trained, in contrast with other models where layers were fully trained, leading to some speculation but no concrete explanation.
- **Speculation on Lazy Optimizer Behavior**: There was speculation that if a dataset is too easy, the model may not optimize all layers and focus only on the first few and the last layers for efficiency, yet anomalies in layer training intensity were not fully explained by this theory.
- **Axolotl Features Inquiry**: A member inquired about the latest features in Axolotl after having not trained in a while, seeking updates on new capabilities or enhancements.

**Link mentioned**: <a href="https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file">GitHub - HVision-NKU/StoryDiffusion: Create Magic Story!</a>: Create Magic Story! Contribute to HVision-NKU/StoryDiffusion development by creating an account on GitHub.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/)** (1 messages): 

icecream102: Coincidence?
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1237281394124656710)** (5 messages): 

- **Identifying HuggingFace Dataset Types**: When uncertain about the type of a HuggingFace dataset, the simplest method is to download and open it up to inspect the contents. Alternatively, one can check the dataset's preview for this information.

- **Building a Language-Specific LLM for Code**: Inspired by IBM's granite models, a member shared an interest in creating a language-specific LLM, specifically for Java code assistance, with the intent to operate on a standard laptop without a GPU. They seek guidance on selecting a base model for fine-tuning, determining the right number of epochs, training size, and quantization to maintain accuracy.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1237101588842156073)** (38 messages🔥): 

- **Challenges in Fine-tuning for Mathematical Performance**: Discussions highlighted a decrease in scores for mathematical topics, especially in the **mmlu** and **Chinese math evaluations** (ceval and cmmlu). The decline in performance was noted even after fine-tuning on datasets such as `orca-math-word-problems-200k`, `math instruct`, and `metamathQA` from Hugging Face.

- **Quantization's Impact on Model Performance**: Members broached the topic of [quantization effects](https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094961774), specifically referencing how `llama.cpp`'s quantization might significantly degrade model performance.

- **Fine-tuning and Evaluation Strategies Discussed**: Models were reportedly fine-tuned on datasets like `orca-math-word-problems-200k`, `math instruct`, and `MetaMathQA`, and evaluated using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). However, some concern was raised regarding ensuring correct prompt template usage during evaluation and fine-tuning.

- **Prompt Design Can Influence Model Behavior**: A pointed discussion took place about the importance of using correct prompt designs, as changes to templates, including potentially incorrect end-of-text tokens, could impact model performance.

- **Key Role of Prompt Formats in Fine-tuning**: A participant argued that custom prompt formats like `alpaca` could be utilized if models are also fine-tuned with those examples, while recognizing the need for accurate comparisons and awareness of potential performance issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | Model Cards and Prompt formats</a>: Special Tokens used with Meta Llama 3. A prompt should contain a single system message, can contain multiple alternating user and assistant messages, and always ends with the last user message followe...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094961774">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47">axolotl/src/axolotl/prompters.py at 3367fca73253c85e386ef69af3068d42cea09e4f · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2097072819">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...</li><li><a href="https://huggingface.co/datasets/TIGER-Lab/MathInstruct">TIGER-Lab/MathInstruct · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/meta-math/MetaMathQA">meta-math/MetaMathQA · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/1237224063047565433)** (2 messages): 

- **Expanded Documentation on Model Merging**: A [new documentation update](https://axolotl.continuumlabs.pro/) has been made on *Axolotl*, addressing the merging of model weights. The next goal outlined is to tackle guidance on inference.
- **Axolotl: A Training Hub for AI Enthusiasts**: The **Axolotl GitHub repository** offers a flexible tool for fine-tuning AI models, covering a wide range of Huggingface models and fine-tuning techniques, and highlights the ability to customize configurations.

**Link mentioned**: <a href="https://axolotl.continuumlabs.pro/">Introduction | Continuum Training Platform | Axolotl Training Platform</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1237059589854203956)** (5 messages): 

- **Modifying System Prompts in ShareGPT**: To change the system prompt for conversational training, one must adjust the conversation template in the `ShareGPTPrompter` class or the initial system message. This involves modifying the `_build_result` method or the corresponding configuration parameters.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d76285fb-b795-43de-a278-b9adfdec1559)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1236980085555204107)** (43 messages🔥): 

- **In Search of OData V4 Support**: A member inquired about **LangChain's** support for **Microsoft Graph (OData V4)** and discussed the potential need to create a custom tool similar to the **SQLDatabase utility**.
- **API Wishes for Kappa Bot**: A user expressed interest in an API for **kappa-bot-langchain** to use outside of Discord due to workplace restrictions on the platform.
- **Entity Memory Under the Microscope**: A clarification was sought on the parameter `k` in **ConversationEntityMemory**, directing attention to the **LangChain documentation** for insight.
- **Framework Showdown for Python Newcomers**: A newcomer to Python queried the community about choosing between **Flask or Django** for a new application, mentioning concerns about Flask’s scalability.
- **Seeking LangChain's Data Heft Limits**: A user questioned the upload size limit for datasets within the **Langsmith Free plan**, specifying a concern with a 300MB CSV file.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/modules/tools/custom_tools/">Defining Custom Tools | 🦜️🔗 LangChain</a>: When constructing your own agent, you will need to provide it with a list of Tools that it can use. Besides the actual function that is called, the Tool consists of several components:</li><li><a href="https://juanpml.com/table-with-llms-token-output-limit">Table with LLMs Token Output Limit</a>: A table with information on Context Window and Maximum Output Tokens of multiple large language models (LLMs)</li><li><a href="https://learn.microsoft.com/en-us/graph/query-parameters?tabs=http">Use query parameters to customize responses - Microsoft Graph</a>: Microsoft Graph provides optional query parameters that you can use to specify and control the amount of data returned in a response. Includes common parameters.</li><li><a href="https://api.python.langchain.com/en/latest/memory/langchain.memory.entity.ConversationEntityMemory.html">langchain.memory.entity.ConversationEntityMemory &mdash; 🦜🔗 LangChain 0.1.18</a>: no description found</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1237404915039600762)** (13 messages🔥): 

- **StreamEvents with RemoteRunnable Inquiry**: A member questioned the possibility of using `streamEvents` with `RemoteRunnable` in LangChain. They were provided an affirmative answer along with a detailed code example and directed to the [LangChain documentation](https://js.langchain.com/docs/ecosystem/langserve#usage) and [API reference](https://api.js.langchain.com/classes/langchain_core_runnables_remote.RemoteRunnable.html).

- **RemoteRunnable Streaming Issues in JavaScript**: The member later reported an issue where the JavaScript implementation of `RemoteRunnable` was not streaming via `streamEvents` as expected, despite working in Python. This suggests a potential inconsistency or problem with the JavaScript version.

- **Possible Misdirection to Incorrect API Endpoint**: The conversation concluded with the member noting that the `streamEvents` method was making HTTP POST requests to `/stream` instead of `/stream_events`. The inconsistency prompted advice to raise an issue on the [LangChain GitHub repository](https://github.com/langchain-ai/langchainjs) for clarification or to correct the potential bug.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://js.langchain.com/docs/ecosystem/langserve#usage>)">Integrating with LangServe | 🦜️🔗 Langchain</a>: LangServe is a Python framework that helps developers deploy LangChain runnables and chains</li><li><a href="https://js.langchain.com/docs/expression_language/interface#stream-events>)).">Interface | 🦜️🔗 Langchain</a>: In an effort to make it as easy as possible to create custom chains, we&#x27;ve implemented a &quot;Runnable&quot; protocol that most components implement.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1237023525181718539)** (5 messages): 

- **Introducing Everything-AI**: *everything-ai* is rebranded to V1.0.0 and features a multi-task, AI-powered local assistant capable of conversing with PDFs, summarizing texts, generating images, and more. The [project can be found on GitHub](https://github.com/AstraBert/everything-ai) with a brand new user interface and quick-start documentation provided at [https://astrabert.github.io/everything-ai](https://astrabert.github.io/everything-ai).

- **Calling All Beta Testers**: Rubiks.ai seeks beta testers for its advanced research assistant and search engine, offering 2 months free of premium access to models like Claude 3 Opus, GPT-4 Turbo, and Mistral Large. Interested parties are welcomed with a promo code and can sign up at [https://rubiks.ai/](https://rubiks.ai/).

- **No-Code Tool for AI Applications Unveiled**: A new no-code tool designed to streamline the creation of AI applications and facilitate the transition from prototype to production has been introduced, complete with built-in prompt engineering and one-click deployment. The early demo is viewable at [Google Drive demo](https://drive.google.com/file/d/1U5lHigyCf655-sD2fbThuE0HMvueW2JR/view?usp=sharing), and feedback can be scheduled via [booking link](https://calendar.app.google/SgryS4Pr47CJHic9A).

- **Discovering API Chain through Langchain Series**: A tutorial on using APIChain for calling APIs with large language models (LLMs) is available in a video called "API Chain | Chain Types | Learning Langchain Series," found on [YouTube](https://youtu.be/IpLJwDfxiVA?si=v8Vocd0QzJx-CsbW).

- **Learn to Leverage Router Chain**: Another part of the Learning Langchain Series covers Router Chain, a tool for managing multiple APIs and tasks with LLMs, which is explained in detail in the video "ROUTER CHAIN | Learning Langchain Series | Chain Types" on [YouTube](https://youtu.be/ItppCNZBzbY?si=9OeVm6OOU2giW1Zc).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://astrabert.github.io/everything-ai">everything-ai</a>: Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖</li><li><a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: Introducing everything-ai, your multi-task, AI-powered and local assistant! 🤖</a>: Introducing everything-ai, your multi-task, AI-powered and local assistant! 🤖 - AstraBert/everything-ai</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://youtu.be/IpLJwDfxiVA?si=v8Vocd0QzJx-CsbW">API Chain | Chain Types | Learning Langchain Series | Become an expert in calling APIs with LLMs!</a>: Learn how to call an API using APIChain from langchain. You will see that with the help of this library, you&#39;ll be at the forefront of delivering values when...</li><li><a href="https://drive.google.com/file/d/1U5lHigyCf655-sD2fbThuE0HMvueW2JR/view?usp=sharing">Pixie.ai.mp4</a>: no description found</li><li><a href="https://calendar.app.google/SgryS4Pr47CJHic9A">Appointments</a>: no description found</li><li><a href="https://youtu.be/ItppCNZBzbY?si=9OeVm6OOU2giW1Zc">ROUTER CHAIN | Learning Langchain Series | Chain Types | Route between your LLMs in a fashion way!</a>: Router chain is one of the tools you&#39;ll definitely need when dealing with multiple tasks! Imagine how to handle multiple APIs or multiple tasks among multipl...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

mhadi91: https://youtu.be/WTfWgYsIspE?si=gEdyMrX4vJm2gC6E
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1237256126886576210)** (61 messages🔥🔥): 

- **Exploring Roleplay Datasets for AI**: A member suggested creating a dataset composed of exclusively human-written dialogue, including jokes and human interactions, to improve AI model responses beyond the typical soulless outputs of smart instruct models.
- **Synthetic Data Generation for Research**: A user shared a [GitHub link to Simian](https://github.com/RaccoonResearch/Simian), a synthetic data generator for image, video, and 3D models, offering a possible resource for experimentation.
- **Seeking Recommendations for Datasets**: A new member to machine learning inquired about datasets suitable for a research paper focusing on text/numeric regression or classification tasks, leading to various dataset recommendations like MNIST-1D and sentiment analysis datasets such as [Stanford's Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
- **Discussion on Text-to-Video Diffusion Models**: A lively discussion unfolded concerning the superiority of diffusion models over generative transformers for SOTA text-to-video tasks, noting that diffusion models are often fine-tuned from existing T2I models, saving on computational costs. Insights into the 3D knowledge of these models and their text-conditioning challenges were shared.
- **Video Diffusion Model Insights from an Author**: The author of the stable video diffusion paper participated in the discussion, highlighting the challenges of obtaining quality text supervision for video models, the potential of auto-captioning videos with LLMs, and the nuances between autoregressive and diffusion approaches to video generation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sites.research.google/videopoet/">VideoPoet – Google Research</a>: A Large Language Model for Zero-Shot Video Generation. VideoPoet demonstrates simple modeling method that can convert any autoregressive language model into a high quality video generator.</li><li><a href="https://github.com/RaccoonResearch/simian">GitHub - RaccoonResearch/Simian: Synthetic data generator for image, video and 3D models</a>: Synthetic data generator for image, video and 3D models - RaccoonResearch/Simian</li><li><a href="https://github.com/instructlab/community">GitHub - instructlab/community: InstructLab Community wide collaboration space including contributing, security, code of conduct, etc</a>: InstructLab Community wide collaboration space including contributing, security, code of conduct, etc - instructlab/community
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1237417884909699202)** (1 messages): 

- **OpenDevin Webinar Announcement**: LlamaIndex is hosting a webinar featuring the authors of OpenDevin, an open-source version of Devin by Cognition, on **Thursday at 9am PT**. Attendees can learn about building autonomous AI agents and will gain insights from its burgeoning popularity on GitHub. Register [here](https://lu.ma/fp0xr460).

**Link mentioned**: <a href="https://lu.ma/fp0xr460">LlamaIndex Webinar: Build Open-Source Coding Assistant with OpenDevin · Zoom · Luma</a>: OpenDevin is a fully open-source version of Devin from Cognition - an autonomous AI engineer able to autonomously execute complex engineering tasks and…

  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1237098170333397043)** (4 messages): 

- **Hugging Face TGI Unveils New Features**: Hugging Face's TGI (Text Generation Inference) toolkit announced support for function calling and batched inference among other features, aimed at optimizing LLM deployments on the platform. The full list of features was shared in a [tweet by LlamaIndex](https://twitter.com/llama_index/status/1787538681999630658).

- **Jerry Liu to Speak at AIQCon**: Co-founder Jerry Liu will be speaking about building Advanced Question-Answering Agents Over Complex Data at AIQCon in San Francisco. A 15% discount on tickets is available with the code "Community," as per a [tweet with the conference details](https://twitter.com/llama_index/status/1787619739654836309).

- **Enhancing RAG with LlamaParse**: LlamaParse aims to improve the data quality for building Retrieval-Augmented Generation (RAG) models over complex documents, emphasizing that quality data is critical for good performance. This development was highlighted in a [recent LlamaIndex tweet](https://twitter.com/llama_index/status/1787636831678521395).

- **OpenDevin as an Open-Source AI Engineer**: @cognition_labs released OpenDevin, an open-source autonomous AI engineer capable of executing complex engineering tasks and collaborating on software projects. The announcement and details were provided in a [LlamaIndex update](https://twitter.com/llama_index/status/1787858033412063716).

**Link mentioned**: <a href="https://t.co/jA6SyZNdgn">The AI Quality Conference</a>: The world's first AI Quality Conference on June 25, 2024 in San Francisco, CA

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1236988697254301828)** (50 messages🔥): 

- **The Quest for LlamaIndex Integration**: The conversation grappled with issues integrating **LlamaIndex with other databases**. A user faced challenges with empty responses when querying **Supabase Vectorstore** and mismatched dimension errors; they eventually resolved the dimension issue by specifying the model and dimensions during query engine creation.

- **Package Imports and Documentation Confusion**: Discussion unfolded around **package imports** after a recent update to llama-index. Users shared experiences about locating the correct package paths and imports, and guidance was provided to refer to the [updated llama-hub documentation](https://github.com/run-llama/llama-hub/tree/main#how-to-add-a-loadertoolllama-pack) for installs and imports.

- **Troubleshooting the Deletion of Document Knowledge**: A user encountered problems when attempting to **delete document knowledge** from their index in llama-index. Conversation with another user suggested re-instantiating the **query_engine** after persisting changes in order to observe the deletion, although the issue appeared unresolved as the user continued to see the document in the json vector store.

- **PDF Parsing Libraries for Local Use**: A user inquired about **local PDF parsing libraries** as an alternative to using LlamaParse. PyMuPDF was recommended, with a usage example given showing how to integrate PyMuPDF with LlamaIndex.

- **Handling Absence of Relevant Information in Responses**: Users discussed approaches to **prevent a model from responding** when no relevant information is found in the context. Prompt engineering and employment of a check similar to Evaluate on each request were suggested as potential solutions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docsbot.ai/tools/gpt-openai-api-pricing-calculator">OpenAI &amp; other LLM API Pricing Calculator - DocsBot AI</a>: Calculate and compare the cost of using OpenAI, Azure, Anthropic, Llama 3, Google Gemini, Mistral, and Cohere APIs with our powerful FREE pricing calculator.</li><li><a href="https://llamahub.ai">Llama Hub</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/embeddings/llama-index-embeddings-huggingface">llama_index/llama-index-integrations/embeddings/llama-index-embeddings-huggingface at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/low_level/evaluation#setup>)">Building Evaluation from Scratch - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1237021916792098916)** (4 messages): 

- **Seeking HyDE for Complex NL-SQL**: A member is building a **NL-SQL chat bot** to handle complex SQL queries across hundreds of tables and is looking for a HyDE method that's effective in this scenario. They've only found references to HyDE's use in pdf/text chatbots and are exploring options for enhancing the **LLM's** accuracy in database querying.

- **Introspective Agents with LlamaIndex**: A link to an article titled "Introspective Agents with LlamaIndex" was shared, describing **introspective agents** that use the *reflection agent pattern* within the LlamaIndex framework. The source is [AI Artistry on Medium](https://medium.com/ai-artistry/introspective-agents-with-llamaindex-777d018f791d) and includes images created using MidJourney.

- **Medium 404 Error for Agent Article**: A member shared a link to the same article, "**Introspective Agents with LlamaIndex**," but this time the link led to a **404 Page Not Found** error on Medium. The page suggested navigating to other stories that could potentially be of interest.

- **Article Acknowledgement**: Another member acknowledged the shared article on **Introspective Agents with LlamaIndex** as a "nice article," implying a positive reception but providing no further detail or discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/ai-artistry/introspective-agents-with-llamaindex-777d018f791d">Introspective Agents with LlamaIndex</a>: Ankush k Singal</li><li><a href="https://medium.com/ai-artistry/introspective-agents-wi">no title found</a>: no description found
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1236972866252046336)** (35 messages🔥): 

- **Rethinking UOps Representation**: A user suggests improving the readability of tinygrad's operation presentation by adopting a format similar to LLVM IR. The suggestion includes using a more **human-readable format** for operations.
- **Clarifying SSA and UOps**: In discussing the change, it's highlighted that **tinygrad's operations** are meant to be in Static Single Assignment (SSA) form. A user points to the confusion around the placement of the PHI operation at the end of a block rather than at the beginning, as seen in traditional LLVM IR.
- **Opinions Divided on Proposed Formatting**: While one member argues against changing tinygrad's current formatting, stating it introduces an unnecessary abstraction layer, another member encourages submitting a **Pull Request (PR)** to implement the proposed changes.
- **Discord Betting Escapade**: Two users engage in a conversation about betting on the correctness of code through their PayPal balance. It evolves into an acknowledgement of potential bugs and the intricacies of creating a **betting challenge** bound by time and accuracy on the server.
- **Machine Learning Jargon Workaround Discussed**: Amidst the technical exchange, a user seeks advice on approaching machine learning without delving deep into **mathematical terminology**. They are directed towards resources by Andrej Karpathy, while another user reiterates the chat rules around asking beginner questions.
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1236952973742182490)** (20 messages🔥): 

- **CPU Kernels Single Threaded in tinygrad**: In response to a question about whether **tinygrad** utilizes multiple threads for operations like matrix multiplication, George Hotz confirmed that **tinygrad is single threaded** and does not use threads for CPU operations.

- **Understanding Remapping and Strides in Tensors**: A user described how remapping a tensor by changing its stride can allow certain reshapes and computations, and suggested maintaining an original shape to calculate indices after reshaping, a technique which might resemble what **tinygrad** does under the hood.

- **Sharing Knowledge Through Documentation**: Users have shared self-created explanatory content such as posts on symbolic mean and Winograd convolutions understanding. One user shared a [GitHub post on symbolic mean](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/symbolic-mean.md) and another offered a [Google Doc link for view merges](https://docs.google.com/document/d/1RRuhAW-I5u_6ssbm1eLqVWSK_PYFoARIVBYwYZwmcM4/edit?usp=sharing).

- **Quantized Inference Capabilities in tinygrad**: A user inquired about **tinygrad's** ability to perform quantized inference similar to bitsandbytes library, with an acknowledgment that it is somewhat capable of doing so.

- **Recommendations for Learning Through Documentation and Examples**: Users encouraged the creation and sharing of toy examples and documentation as a method for learning and teaching concepts related to tinygrad, indicating the potential benefit for both the original author and the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/symbolic-mean.md">tinygrad-notes/symbolic-mean.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/docs-legacy/reshape_without_symbolic.md">tinygrad/docs-legacy/reshape_without_symbolic.md at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://docs.google.com/document/d/1RRuhAW-I5u_6ssbm1eLqVWSK_PYFoARIVBYwYZwmcM4/edit?usp=sharing">View Merges</a>: no description found
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1237029049033818164)** (35 messages🔥): 

- **Seeking SQL Database Location in Cohere Toolkit**: A member inquired about the location of the SQL database for storing conversational history in the Cohere toolkit. Another member clarified that it's on **port 5432**, without specifying an exact location.

- **Student Aspires to Create a Google Bard-like Chatbot**: A high school student expressed their ambition to build a chatbot similar to Google's Bard, questioning whether it complies with Cohere's user agreement. Cohere's representative shared [guidance on trial and production keys](https://docs.cohere.com/docs/going-live#production-key-specifications), confirming that building and eventually providing paid access to a chatbot is permissible, subject to obtaining a production key.

- **Addressing Chroma Retrieval and Embedding Issues**: A member reported issues while testing the Cohere toolkit locally with Chroma, specifically an IndexError happening during document retrieval. The conversation pointed towards checking the full log trace available at a [Pastebin link](https://pastebin.com/raw/1RmUQAP4) and the prebuilt container from `ghcr.io/cohere-ai/cohere-toolkit:latest`.

- **Langchain Retriever Always Selected in Cohere Toolkit**: Despite selecting File Reader -LlamaIndex, a user reported that the **Langchain retriever** is used instead, as evidenced by a shared screenshot, which, however, was not accessible.

- **Registration of Production Key Acting as Trial Key**: A user experienced an issue where a newly registered production key still functioned as a trial key. Cohere's representative clarified that the trial key is only used in the **Playground / Chat UI** and assured that when used in the API, it should reflect production key usage and does not require pre-funding.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.trychroma.com/telemetry">📏 Telemetry | Chroma</a>: Chroma contains a telemetry feature that collects anonymous usage information.</li><li><a href="https://pasteboard.co/teLlt9T57Gu5.jpg">Pasteboard - Uploaded Image</a>: no description found</li><li><a href="https://docs.cohere.com/docs/going-live#production-key-specifications">Going Live - Cohere Docs</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=uVUhyPsqdJ8">Python Decorators In 1 MINUTE!</a>: Discover the power of Python decorators in just 1 minute! This quick tutorial introduces you to the basics of decorators, allowing you to enhance your Python...</li><li><a href="https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf/#using-pypdf">PDF | 🦜️🔗 LangChain</a>: Portable Document Format (PDF), standardized as ISO 32000, is a file format developed by Adobe in 1992 to present documents, including text formatting and images, in a manner independent of applicatio...
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1237471018629464136)** (2 messages): 

- **Cohere Coral Combines Chatbot and ReRank**: A member introduced an app called **Coral Chatbot** which integrates text generation, summarization, and ReRank into one tool. You can check out the app and provide feedback through their [Streamlit page](https://cohere-api-endpoints.streamlit.app/).

- **Python Decorators Demystified in 60 Seconds**: A quick tutorial entitled "Python Decorators In 1 MINUTE" was shared, promising a brief introduction to Python decorators. Interested members can watch the explainer video on [YouTube](https://www.youtube.com/watch?v=uVUhyPsqdJ8).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=uVUhyPsqdJ8">Python Decorators In 1 MINUTE!</a>: Discover the power of Python decorators in just 1 minute! This quick tutorial introduces you to the basics of decorators, allowing you to enhance your Python...</li><li><a href="https://cohere-api-endpoints.streamlit.app/">no title found</a>: no description found
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1236988240884404256)** (35 messages🔥): 

- **Centaur Coders to Shrink Team Sizes?**: A linked post from [v01.io](https://www.v01.io/posts/2024-centaur-programmers-shrinking-team-sizes/) discusses the potential for **Centaur Programmers** to reduce product team sizes, leveraging human-AI collaboration to increase efficiency. Discussions around this hypothesis speculate whether smaller teams will emerge, or if teams will instead focus on improving product output.

- **DeepSeek-V2 Tops Performance Charts**: Announced via [Twitter](https://x.com/deepseek_ai/status/1787478986731429933), **DeepSeek-V2** is a notable open-source MoE model excelling in benchmarks, with top-tier performance in several areas, including code and reasoning capabilities. Community response included excitement and analysis, with linked discussions examining the implications of the new model.

- **Unveiling DeepSeek's Achievements**: Additional conversation centered on DeepSeek-V2's benchmark achievements was shared, along with a personal take provided in an [AI News newsletter](https://buttondown.email/ainews/archive/ainews-deepseek-v2-beats-mixtral-8x22b/), illustrating the model's impact on the AI landscape.

- **Exploration of Unified Search Options**: The search for a feasible unified search solution for small organizations led to the mention of **Glean** and a potential OSS alternative discussed in a shared [Hacker News post](https://news.ycombinator.com/item?id=39467413). A suggestion was made for a bot that can preemptively search for relevant posts across platforms like Discord.

- **Inquiry into AI Orchestration Practices**: Queries about **AI (data) orchestration** practices were raised, seeking community input on preferred orchestration tools, data transfer methods, and architectural advice for handling complex data pipelines involving text and embeddings.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.v01.io/posts/2024-centaur-programmers-shrinking-team-sizes/">Centaur Programmers shrink product team sizes — at least, they should. &ndash; Klaus Breyer</a>: Pedro Alves, Bruno Pereira Cipriano, 2023: The idea of the Centaur Programmer is based on the premise that a collaborative approach between humans and AI will be more effective than AI alone, as demon...</li><li><a href="https://x.com/deepseek_ai/status/1787478986731429933">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 Launching DeepSeek-V2: The Cutting-Edge Open-Source MoE Model!  🌟 Highlights: &gt; Places top 3 in AlignBench, surpassing GPT-4 and close to GPT-4-Turbo. &gt; Ranks top-tier in MT-Bench, rivaling ...</li><li><a href="https://news.ycombinator.com/item?id=39467413">no title found</a>: no description found</li><li><a href="https://buttondown.email/ainews/archive/ainews-deepseek-v2-beats-mixtral-8x22b/">[AINews] DeepSeek-V2 beats Mixtral 8x22B with &gt;160 experts at HALF the cost</a>: AI News for 5/3/2024-5/6/2024. We checked 7 subreddits and 373 Twitters and 28 Discords (419 channels, and 10335 messages) for you. Estimated reading time...
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1237016300333694997)** (6 messages): 

- **Local Faraday Use is Free**: Members clarified that using **Faraday** locally doesn't require a cloud subscription and there are no costs involved. One member shared their personal experience, stating it works fine with 6 GB VRAM and includes free voice output.

- **Forever Access to Downloads**: Users highlighted that once characters and models are downloaded from the **Faraday platform**, they can be used indefinitely without further charges.

- **A Nod to Sufficient GPUs**: It was pointed out that a powerful enough GPU eliminates the need for a cloud account, unless users wish to contribute to the developers through a subscription.
  

---


**AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/1237055950032998422)** (5 messages): 

- **Simulation Collaboration Kickoff**: An individual, @abhavkedia, proposed working together to create a **fun simulation** in relation to the Kendrick and Drake situation, seeking collaborators.
- **Project Progress & Team-Up**: @abhavkedia shared their current progress on the simulation, and @jakekies expressed interest in joining the project, indicating a collaborative effort is underway.
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1236972397345640488)** (11 messages🔥): 

- **Code Updates Need Manual Table Wiping**: Updating **character definitions** in code won't automatically refresh the systems; members were reminded to run **wipe-all-tables** if they make character function or map changes.
- **Automation Suggestion Using inotifywait**: A method was shared to automate updates by using `inotifywait` for detecting changes in **data/character.ts** and making changes accordingly.
- **Invitation to Try Llama Farm**: Members were encouraged to try a new simulation, [Llama Farm](https://labs.convex.dev/llama-farm), with interest already being shown to the shared link.
- **Llama Farm Integration with AI-Town**: A unique **integration concept** between **Llama Farm** and **AI-Town** was proposed, requiring an instance of Ollama to participate in AI-Town, with local message processing tagged by character.
- **Toward a More Generic Llama Farm**: Plans to generalize **Llama Farm's** ability to hook into any system using the OpenAI API were shared, including an approach for streaming using query/mutation for multiplexing.

**Link mentioned**: <a href="https://labs.convex.dev/llama-farm">llama farm</a>: no description found

  

---


**AI Stack Devs (Yoko Li) ▷ #[paper-spam](https://discord.com/channels/1122748573000409160/1227492197541220394/)** (1 messages): 

Deforum Daily Papers: Papers will now be sent to <#1227492197541220394>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1237024677503696967)** (18 messages🔥): 

- **Slower Model Performance on Devices**: A member reported that running the model on their device was slow, yielding **8 seconds per token**.

- **Rocket-3B as a Speedier Alternative**: After facing performance issues with a different model, a member was advised to try **Rocket-3B**, which significantly improved the speed.

- **Efficient Use of Ollama Cache with Llamafile**: There was an inquiry about whether **llamafile** can utilize models stored in the ollama cache to prevent multiple downloads, and the response clarified that it's possible by using `-m model_name.gguf`.

- **Challenges with AutoGPT and Llamafile Integration**: A member encountered a problem with **AutoGPT** not starting the AP server correctly, causing **llamafile agent** to be killed on startup; manual restart was a workaround for port `8080` but failed for port `8000`.

- **Draft PR for Llamafile Support in AutoGPT**: Instructions for setting up **autoGPT + llamafile** were provided, indicating that feedback from AutoGPT maintainers is pending before further development. The conversation implied there's an ongoing effort to integrate **llamafile** with **AutoGPT** via a draft PR. [Draft llamafile support instructions](https://github.com/Mozilla-Ocho/AutoGPT/tree/draft-llamafile-support/autogpts/autogpt/llamafile-integration).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Mozilla-Ocho/AutoGPT/tree/draft-llamafile-support/autogpts/autogpt/llamafile-integration">AutoGPT/autogpts/autogpt/llamafile-integration at draft-llamafile-support · Mozilla-Ocho/AutoGPT</a>: AutoGPT is the vision of accessible AI for everyone, to use and to build on. Our mission is to provide the tools, so that you can focus on what matters. - Mozilla-Ocho/AutoGPT</li><li><a href="https://github.com/Significant-Gravitas/AutoGPT/pull/7091">Draft llamafile support by k8si · Pull Request #7091 · Significant-Gravitas/AutoGPT</a>: Background  This draft PR is a step toward enabling the use of local models in AutoGPT by adding llamafile as an LLM provider. Related issues:  #6336 #6947  Changes 🏗️  For full documentation of th.....
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1237179630398734376)** (5 messages): 

- **Debate Over AI Evaluations**: A link was shared from [Dr. Jim Fan's tweet](https://twitter.com/DrJimFan/status/1786054643568517261?utm_source=ainews&utm_medium=email&utm_campaign=ainews-evals-the-next-generation) regarding AI evaluations. The message highlighted that while the tweet's content is interesting, there may be errors, and that specific benchmarks and public democracy in evaluation are overvalued, favoring AB testing over open democracy.

- **Benchmarking—A Database Perspective**: A member resonated with the need for standard benchmarks in AI and compared it with their experience in the database field. They suggested that the three sets of benchmarking as mentioned in the tweet could be an appropriate approach.

- **What's TPC? A Quick Intro**: Following a question about TPC, a member explained that TPC stands for Transaction Processing Council, a neutral entity that sets and audits standards for the database industry via benchmarks like TPC-C and TPC-H. They detailed it as a response to overhyped claims by database vendors.
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1237306437244424243)** (11 messages🔥): 

- **GPT-2 Chatbots Stir Up LMsys Arena**: A tweet shared by **xeophon.** mentioned a "good gpt2-chatbot" referencing a humorous comment by **Sam Altman**, indicating GPT-2's return to the LMsys arena. The tweet included a [conversation snapshot](https://x.com/U8JDq51Thjo1IHM/status/1787744675023249805).
- **Skepticism Over OpenAI's Use of LMsys for Evaluations**: Nathan Lambert expressed a personal disapproval of **OpenAI** seemingly using **LMsys** for their model evaluations.
- **LMsys Tightrope Walk**: Nathan Lambert observed that **LMsys** is in a difficult position due to a lack of resources, which prevents them from refusing collaborations.
- **Concerns About LMsys Credibility**: Nathan mentioned that the recent wave of 'chatgpt2-chatbot' engagement could negatively impact **LMsys**'s reputation and credibility.
- **Podcast Interview on LMsys Considered**: Nathan considered doing an **Interconnects audio-interview** with the **LMsys** team but remains undecided, citing past interactions that lacked synergy.

**Link mentioned**: <a href="https://x.com/U8JDq51Thjo1IHM/status/1787744675023249805">Tweet from ハードはんぺん (@U8JDq51Thjo1IHM)</a>: I’m-also-a-good-gpt2-chatbot  I’m-a-good-gpt2-chatbot ？？  Quoting Jimmy Apples 🍎/acc (@apples_jimmy)   @sama funny guy arnt you.   Gpt2 back on lmsys arena.

  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1237068475919503420)** (2 messages): 

- **PR Closure Confirmed**: The discussion concluded with the information that a **Pull Request (PR)** was closed/rejected. No further details were provided.
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1237314994564239390)** (3 messages): 

- **AIDEV Conference Gathering Excitement**: Members are expressing excitement about the upcoming **AIDEV event** and are coordinating to meet up. Those attending are encouraged to connect if they haven’t already.
- **Inquiries on Conference Amenities**: A question was raised regarding whether food will be available at the **AIDEV event**, or if attendees should bring their own.
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1237016176429629501)** (10 messages🔥): 

- **Exploring Mistral's Capabilities**: A member confirmed using the **8x22b Mistral** model for their current project, discussing its deployment and performance aspects.

- **Decoding Techniques to Lower Patency**: A query was raised about achieving low latency in decoding without waiting for the end of a sentence, touching on strategies for efficient language model output generation.

- **Potential for a German DPO Dataset**: The idea of creating a German dataset for inclusive language was proposed, sparking a discussion on its utility and whether it should also focus on grammar and wording alongside inclusiveness.

- **Seeking Input for German Pretraining Dataset**: A member sought feedback on building a German-exclusive pretraining dataset from Common Crawl and inquired about specific domains that might warrant greater emphasis due to high-quality content.

- **Resource Sharing for Inclusive Language**: Resources were shared for gender and diversity-sensitive language, including the _INCLUSIFY_ prototype ([https://davids.garden/gender/](https://davids.garden/gender/)) and a related GitLab repository ([https://gitlab.com/davidpomerenke/gender-inclusive-german](https://gitlab.com/davidpomerenke/gender-inclusive-german)), which might be relevant to implementing inclusive language modes in AI models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://davids.garden/gender/">David’s Garden - Gender-inclusive German: A benchmark and a model</a>: Gender-inclusive language is important for achieving gender equality in languages with gender inflections, such as, for the purpose of this report, German.</li><li><a href="https://gitlab.com/davidpomerenke/gender-inclusive-german">David Pomerenke / Gender-inclusive German - a benchmark and a pipeline · GitLab</a>: GitLab.com
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1237119509165248593)** (7 messages): 

- **Prompt Generator Tool Revealed**: A member discussed the discovery of a new **prompt generator tool** in the **Anthropic console**.
- **Polite Rephrasing Function Tested**: The tool was tested with a prompt to rephrase a sentence more politely, yielding a satisfactory result. 
- **Exploring the Underlying Mechanics**: A member is in the process of extracting the system prompt used by the new tool, commenting on the extensive use of **k-shot examples**, with an intriguing one about a Socratic math tutor.
- **Challenge in Extracting Long Prompts**: Attempts to extract the full prompt are ongoing, with difficulties arising due to its length, particularly a lengthy math tutor example.
- **Promise to Share Information**: The member confirmed that once the full prompt is successfully extracted, it will be shared in the chat.
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1237418262850175016)** (2 messages): 

Since the provided messages are only greetings, there is no substantive content to summarize in the requested format. If more topical and detailed messages are provided, I would be able to create a summary based on those.
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1237375746050101278)** (2 messages): 

- **GitHub Issue Collaboration**: Discussing parameterized testing for a plugin, a link to a GitHub issue ([Design and implement parameterization mechanism · Issue #4](https://github.com/simonw/llm-evals-plugin/issues/4)) was shared, indicating ongoing development and contributions.
- **Inquiry About `llm` with OpenAI Assistant API**: A member asked whether the `llm` can be used with the OpenAI Assistant API, hoping they hadn't missed that information somewhere.

**Link mentioned**: <a href="https://github.com/simonw/llm-evals-plugin/issues/4#issuecomment-2098250711">Design and implement parameterization mechanism · Issue #4 · simonw/llm-evals-plugin</a>: Initial thoughts here: #1 (comment) I want a parameterization mechanism, so you can run the same eval against multiple examples at once. Those examples can be stored directly in the YAML or can be ...

  

---



---




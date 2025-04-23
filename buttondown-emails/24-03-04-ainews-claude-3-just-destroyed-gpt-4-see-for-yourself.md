---
id: bd591bdf-2e17-43a4-942e-2384e35a4b5a
title: '[AINews] Claude 3 just destroyed GPT 4 (see for yourself)'
date: '2024-03-04T23:59:02.180354Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-claude-3-just-destroyed-gpt-4-see-for
---

<!-- buttondown-editor-mode: plaintext -->> AI News for 3/2/2024-3/4/2024. We checked [**356** Twitters](https://twitter.com/i/lists/1585430245762441216) and **22** Discords (**352** channels, and **9688** messages) for you. Estimated reading time saved (at 200wpm): **984 minutes**.

[Claude 3 is here](https://news.ycombinator.com/item?id=39590666)! Nothing else from the weekend matters in comparison, which is awfully nice for weekday newsletter writers.

 ![image.png](https://assets.buttondown.email/images/e5df127c-db27-4d63-b1ca-efffc631fcf4.png?w=960&fit=max) 

**TLDR**: 

- Claude now comes in 3 sizes - the smallest two (Haiku - unreleased, Sonnet - default, on claude.ai, AWS and GCP) are fast  (2x faster than Claude 2) and cheap ([half the cost of GPT4T](https://x.com/mattshumer_/status/1764738098389225759?s=20)) and good, and the big one (Opus, on Claude Pro, but slower and more expensive) appears to **beat GPT4 on every benchmark that matters**. Sometimes, like in GPQA, a LOT better, [impressing the GPQA benchmark author](https://x.com/idavidrein/status/1764675668175094169?s=20).
- They're all multimodal, specifically vision and convincingly turned a [2hr Karpathy video into a blogpost](https://x.com/mlpowered/status/1764718705991442622?s=20).
- Better alignment - fewer bad refusals, and improved accuracy on hard questions
- 200k token context, can extend up to 1m tokens, with Gemini 1.5-like perfect recal;
  - Notably, [detected that it was being tested](https://twitter.com/alexalbert__/status/1764722513014329620) while doing a routine Needle in Haystack test. [Safetyists up in arms](https://x.com/NPCollapse/status/1764740710731837516?s=20).

Our full notes below:

- Haiku (small, $0.25/mtok - "available soon"), Sonnet (medium, $3/mtok - powers claude.ai, is on Amazon Bedrock and Google Vertex), Opus (large $15/mtok - powers Claude Pro)
- **Speed**: Haiku is the fastest and most cost-effective model on the market for its intelligence category. **It can read an information and data dense research paper on arXiv (~10k tokens) with charts and graphs in less than three seconds**. Following launch, we expect to improve performance even further. [Sonnet is 2x faster than Opus and Claude 2/2.1](https://x.com/AnthropicAI/status/1764653835568726215?s=20)
- **Vision**: The Claude 3 models have **sophisticated vision capabilities** on par with other leading models. They can process a wide range of visual formats, including photos, charts, graphs and technical diagrams.
- [Opus can turn a 2hr video into a blogpost](https://x.com/mlpowered/status/1764718705991442622?s=20)
- **Long context and near-perfect recall:** Claude 3 Opus not only achieved near-perfect recall, surpassing 99% accuracy, but in some cases, it even identified the limitations of the evaluation itself by recognizing that the "needle" sentence appeared to be artificially inserted into the original text by a human.
- **Easier to use**: The Claude 3 models are better at following complex, multi-step instructions. They are particularly adept at adhering to brand voice and response guidelines, and developing customer-facing experiences our users can trust. In addition, the Claude 3 models are better at producing popular structured output in formats like JSON—making it simpler to instruct Claude for use cases like natural language classification and sentiment analysis.
- **Safety**
- Lower refusal rate - very good to combat anthropic safetyist image and topical vs gemini issues from feb
- "Opus not only found the needle, it recognized that the inserted needle was so out of place in the haystack that this had to be an artificial test constructed by us to test its attention abilities." [from Anthropic prompt engineer](https://twitter.com/alexalbert__/status/1764722513014329620)
- criticized by [MMitchell](https://x.com/mmitchell_ai/status/1764739357112713267?s=20) and [Connor Leahy](https://x.com/NPCollapse/status/1764740710731837516?s=20)
- **Evals**
- [choosing to highlight Finance, Medicine, Philosophy domain evals rather than MMLU/HumanEval is good](https://twitter.com/DrJimFan/status/1764719012678897738)
- [59.5% on GPQA](https://x.com/idavidrein/status/1764675668175094169?s=20) is  much better than generalist PhDs and GPT4 - GPQA author is impressed. [paper]([arxiv.org/abs/2311.12022](https://t.co/hb4u4xXzkw)).
- **GPT4 comparisons**
- beats GPT4 at [coding a discord bot](https://twitter.com/Teknium1/status/1764746084436607010)
- [fails at simple shirt drying but GPT4 doesnt](https://x.com/abacaj/status/1764698421749756317?s=20)
- **misc commentary**
- [200k context, can extend to 1m tokens](https://x.com/mattshumer_/status/1764657732727066914?s=20)
- [Haiku is close to GPT4 in evals, but half the cost of GPT3.5T](https://x.com/mattshumer_/status/1764738098389225759?s=20)
- [Trained on synthetic data](https://x.com/Justin_Halford_/status/1764677260555034844?s=20)
- [lower loss on code is normal/unremarkable](https://twitter.com/kipperrii/status/1764673822987538622)

As a bonus, Noah did 2 runs of Claude 3 (Sonnet) vs GPT4 on the same Twitter data scrapes you see below. We think Claude 3's summarization capabilities are way, way better.


---

**Table of Contents**

[TOC] 


---

# PART X: AI Twitter

> Compare [Claude 3](https://github.com/smol-ai/SocialPipelines/blob/summary-prod/data_ingestion/scripts/constants/summary_27_anthropic.md) vs [GPT4T](https://github.com/smol-ai/SocialPipelines/blob/summary-prod/data_ingestion/scripts/constants/summary_27_anthropic.md)


**AI Progress and Capabilities**

- Sam Altman said that "[all of this has happened before, all of this will happen again](https://twitter.com/sama/status/1764178151889084558)" and "[the hurricane turns faster and faster but it stays perfectly calm in the eye](https://twitter.com/sama/status/1764179620486930941)", possibly alluding to the rapid progress in AI.
- Gemini 1.5 Pro from Google is impressive, with "[objectively sharper optics and higher contrast images than the Apple Vision Pro](https://twitter.com/ylecun/status/1764005546456199345)" according to Yann LeCun. However, John Carmack [points out](https://twitter.com/ID_AA_Carmack/status/1764016979101286756) there are many variables that make this comparison not definitive.
- François Chollet [believes](https://twitter.com/fchollet/status/1763629637689893228) his 2023 views on LLM capabilities were overestimating their potential and usefulness. He outlines [four levels of generalization](https://twitter.com/fchollet/status/1763692655408779455) LLMs can achieve, with general intelligence being the ability to synthesize new programs to solve never-seen-before tasks.
- [Gemma](https://twitter.com/AravSrinivas/status/1764063557841469758) from Google is able to be deployed zero-shot in the wild in San Francisco for real-world tasks, without any reinforcement learning, just from next-token prediction on simulation and YouTube data.

**AI Investments and Business**

- Softbank sold all its Nvidia shares in 2019 for $3.6B, which would be worth [$93B today](https://twitter.com/nearcyan/status/1763692669698748478). Investing in AI was one of the primary goals of Softbank's Vision Fund.
- Nvidia's early years involved [relentlessly improving](https://twitter.com/ID_AA_Carmack/status/1763949929611796919) despite competitors having advantages. Their differentiator was taking software more seriously, building the CUDA ecosystem.
- Google faces a problem with the likes of [OpenAI and Perplexity](https://twitter.com/AravSrinivas/status/1763666653748105404) showing that many "search" tasks are better served through conversational AI, similar to how Google disrupted with PageRank and links 25 years ago.
- [Compute and data are the currency of the future](https://twitter.com/alexandr_wang/status/1764071674767720823) according to Alexandr Wang.

**AI Safety and Regulation**

- Elon Musk's lawsuit revealed an investor remark after meeting with Demis Hassabis that "[the best thing he could have done for the human race was shoot Mr. Hassabis then and there](https://twitter.com/AISafetyMemes/status/1763793546535223468)".
- India is [regulating the ability to spin up AI models](https://twitter.com/levelsio/status/1764422501243703684), which some see as self-sabotage at a critical moment, similar to China kicking out its tech giants.
- Vinod Khosla called for banning open-source AI platforms, which Yann LeCun [believes](https://twitter.com/ylecun/status/1764083890119942533) would cause us to lose the "war" he thinks we are in.

**Memes and Humor**

- "[Thank god I didn't go into computer science](https://twitter.com/Nexuist/status/1763651659886969329)" says the junior analyst in New York staring at Excel. "Thank god I didn't go into finance" says the ML scientist in San Francisco, also staring at a spreadsheet.
- Geoff Hinton being spotted working on Gemini at Google leads to speculation he's preparing to [take back the CEO role from Sundar Pichai](https://twitter.com/levelsio/status/1764100109325791561) to save the company he built.
- "[Trump's internal LLM seems to have suffered excessive pruning. How many parameters does he have left? How short is his context window now?](https://twitter.com/ylecun/status/1764133615590346994)"

**Other Relevant Tweets for AI Engineers**

- "[You know you've got your new system's core abstractions right when things you didn't explicitly design for that used to be complex become incredibly simple](https://twitter.com/gdb/status/1764005795799400796)"
- A guide to [probabilistic programming](https://twitter.com/svpino/status/1763914648359748066) and analyzing


---

# PART 0: Summary of Summaries of Summaries

> This is now also driven by Claude 3, which is [way better than OpenAI's output](https://chat.openai.com/share/b6e0a4c6-ee07-4a45-9215-b4a9408b7493).

<div class="contents"><p class="whitespace-pre-wrap">Got it, here's the summary in bullet point markdown format:</p>
<h2><strong>AI Model Performance and Comparisons</strong></h2>
<ul class="list-disc pl-8 space-y-2" depth="0">
<li class="whitespace-normal" index="0">The release of <strong><a href="https://x.com/anthropicai/status/1764653830468428150?s=46">Claude 3</a></strong> by Anthropic sparked extensive discussions and benchmarking comparisons against GPT-4 across multiple Discord servers, with users claiming superior performance on tasks like math and coding. <strong><a href="https://x.com/idavidrein/status/1764675668175094169?s=46&amp;t=6FDPaNxZcbSsELal6Sv7Ug">Claude 3's ~60% accuracy on GPQA</a></strong> was highlighted.</li>
<li class="whitespace-normal" index="1">Debates arose around the <strong>Mistral Large</strong> model's performance versus GPT-4 for coding tasks, with some claiming its superiority despite official benchmarks.</li>
<li class="whitespace-normal" index="2">The <strong><a href="https://huggingface.co/HaileyStorm/MambaMate-Micro">Mamba LM chess model</a></strong> with 11M parameters showed promising results, achieving a 37.7% win rate against Stockfish level 0 as white.</li>
</ul>
<h2><strong>AI Engineering and Deployment Challenges</strong></h2>
<ul class="list-disc pl-8 space-y-2" depth="0">
<li class="whitespace-normal" index="0">Extensive discussions revolved around the difficulties of deploying large language models (LLMs) like <strong>Mistral</strong>, with specific focus on <strong>VRAM requirements</strong>, <strong>quantization strategies</strong>, and optimal configurations for setups like dual NVIDIA 3090 GPUs.</li>
<li class="whitespace-normal" index="1"><strong>CUDA</strong> and GPU optimization were recurring topics, with resources like <strong><a href="https://docs.nvidia.com/cuda/cublasdx/index.html">NVIDIA's cuBLASDx documentation</a></strong> and a <strong><a href="https://github.com/cuda-mode/lectures/tree/main/lecture8">lecture on CUDA performance gotchas</a></strong> being shared.</li>
<li class="whitespace-normal" index="2">The <strong><a href="https://arxiv.org/abs/2401.17948">Terminator architecture</a></strong> was introduced, proposing to replace residual learning with a novel approach to full context interaction.</li>
</ul>
<h2><strong>AI Ethics, Privacy, and Regulations</strong></h2>
<ul class="list-disc pl-8 space-y-2" depth="0">
<li class="whitespace-normal" index="0">Concerns were raised about potential <strong>data scraping from personal profiles</strong> after an AI model's response contained identifiable personal details, prompting discussions on ethics and legality.</li>
<li class="whitespace-normal" index="1"><strong>India's AI deployment regulations</strong> requiring government approval sparked alarms over potential stifling of innovation, as per <strong><a href="https://x.com/martin_casado/status/1764408870804623753?s=46&amp;t=90xQ8sGy63D2OtiaoGJuww">Martin Casado's tweet</a></strong>.</li>
<li class="whitespace-normal" index="2">The Open Source Initiative is working on a new draft for an <strong>open-source AI definition</strong>, with the evolving drafts available <strong><a href="https://opensource.org/deepdive/drafts">here</a></strong>.</li>
</ul>
<h2><strong>Cutting-Edge AI Research and Techniques</strong></h2>
<ul class="list-disc pl-8 space-y-2" depth="0">
<li class="whitespace-normal" index="0"><strong>Aristotelian Rescoring</strong>, a concept that could address complex AI challenges, was discussed, with related works like <strong>STORIUM</strong>, <strong>FairytaleQA</strong>, and <strong>TellMeWhy</strong> available on <strong><a href="https://github.com/StonyBrookNLP/tellmewhy">GitHub</a></strong> and <strong><a href="https://huggingface.co/datasets/StonyBrookNLP/tellmewhy">Hugging Face</a></strong>.</li>
<li class="whitespace-normal" index="1">The novel <strong>HyperZ⋅Z⋅W Operator</strong> was introduced as part of the #Terminator network, blending classic and modern technologies, with the full research <strong><a href="https://arxiv.org/pdf/2401.17948.pdf">available here</a></strong>.</li>
<li class="whitespace-normal" index="2"><strong>RAPTOR</strong>, a new technique for Retrieval-Augmented Generation (RAG) introduced by LlamaIndex, aims to improve higher-level context retrieval, as announced on <strong><a href="https://twitter.com/llama_index/status/1763972097628684607">Twitter</a></strong>.</li>
</ul></div>

---

# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **AI Sensitivity on the Rise**: Claude 3 AI's latest version has heightened sensitivity to potential offensive content and copyright issues, raising questions about safety or over-cautiousness. The mention of Claude 3 was associated with [Google-backed Anthropic debuting its most powerful chatbot yet](https://www.cnbc.com/2024/03/04/google-backed-anthropic-debuts-claude-3-its-most-powerful-chatbot-yet.html).
  
- **CUDA Conundrum**: There's concern within the community about NVIDIA's new licensing terms that restrict the use of CUDA on non-NVIDIA hardware, particularly impacting translation layers. This discussion revolves around the recent update that [Nvidia bans the use of translation layers](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers).

- **Stuck in the Game**: Skepticism prevails over AI's near-future role in game development as current AI limitations may not be easily surpassed by brute-forcing with more compute power.

- **Fine-Tuning Frustration**: An issue with fine-tuning is reported, specifically with an **OpenOrca Mistral 7b model** that gives incorrect outputs post `gguf` conversion. The reported issue can be traced across multiple channels, indicating a broader interest in the problem and potential solutions, with suggestions like checking pre-quantization performance and considering the use of imatrix for outliers.

- **Chess Model Checkmate Performance**: Success is seen in the training of a smaller parameter Mamba LM chess model with an 11M parameter count, performing better as white with a 37.7% win rate against Stockfish level 0. Model available at [HaileyStorm/MambaMate-Micro · Hugging Face](https://huggingface.co/HaileyStorm/MambaMate-Micro).

- **Code-Capable AI Harnesses New Heights**: User @ajibawa_2023 presents their fine-tuned models, notably the **[OpenHermes-2.5-Code-290k-13B](https://huggingface.co/ajibawa-2023/OpenHermes-2.5-Code-290k-13B)**, which demonstrates proficient coding capabilities and the potential for applications in diverse tasks including blogging and story generation.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **AI Community Finds Alternatives During GPT Outage**: Users discussed alternatives to GPT during a recent service downtime, mentioning *Bing Chat, Hugginface, Deft GPT, LeChat, together.xyz*, and *Stable Chat*. Anthropic's **Claude 3** was highlighted as a particularly impressive alternative, with one user mentioning experimenting with the free Sonnet model, while others debated the capabilities and cost considerations of AI models like Claude 3 and OpenAI's offerings.

- **Custom Agents and Optimal Code Generation**: Questions arose about whether custom agents could integrate **CSV files** into their knowledge bases, prompting a technical discussion on file types. User @yuyu1337 explored finding an optimal GPT model for code generation, sparking a conversation about achieving the best time/space complexity and suggestions for using pseudo code.

- **Vision and Humor APIs Puzzle Engineers**: Participants grappled with applying humor in their prompts with varying success between ChatGPT and the GPT 3.5 API. The Discord community was also engrossed in a "owl and palm" brain teaser, trying to solve the puzzle using **GPT-V** with multiple prompting strategies, yet encountering obstacles due to the model's limitations in interpreting measurements.

- **Community Laughs at and Laments Over Usage Limits**: Amid playful banter about AI limitations and usage limits, users exchanged prompt engineering techniques with mixed results. Concerns were raised over server auto-moderation impacting the ability to discuss and share advanced prompts, stirring a call for OpenAI to reconsider prompt restrictions for more effective knowledge sharing.

- **AI Enthusiasts Offer Tips and Seek Training Advice**: Newcomers and experienced users alike asked for and provided advice on prompt engineering, discussing the importance of template structuring and the utilization of AI for content creation tasks while adhering to OpenAI's policies. Discussions highlighted the importance of community and knowledge exchange in the evolving landscape of AI engineering and usage.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Subscription Snafu Sparks Outrage**: @vivekrgopal reported being charged for an annual **Perplexity** subscription despite attempting cancellation during the trial, seeking a refund through direct messages.

- **AI Integration Fever Rises**: Users such as @bioforever and @sebastyan5218 are keenly awaiting the integration of new language models like **Claude 3** and **Gemini Ultra** into **Perplexity**, signaling a high demand for cutting-edge AI features.

- **Benchmark Bafflements with AI Responses**: @dailyfocus_daily delved into the inconsistencies across AI model problem-solving by comparing responses from **GPT-4**, **Claude 3**, and others to a benchmark question about sorting labeled balls into boxes.

- **IAM Insights and AI Fundamentals**: Users like @riverborse and @krishna08011 shared **Perplexity** [links](https://www.perplexity.ai/search/What-is-iam-o2tdFsxGRraeKVWSzo.fIg) focusing on insights into identity access management, and the basics of AI, useful for technical professionals looking to deepen their understanding of key concepts.

- **API Discussions Unfold with Concerns and Anticipations**: Users discussed the limits of **Perplexity API**, including time-sensitive query issues and missing YouTube summary features; they also anticipated new features like citation access. A discussion on temperature settings revisited how they affect the naturalness and reliability of language outputs, and a link to assist with API usage was shared by @icelavaman, [Perplexity API Info](https://discord.com/channels/1047197230748151888/1118264005207793674/1213229028870324305).



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

**Hermes 2.5 Takes the Lead**: Discussions in the guild revealed that **Hermes 2.5** has unexpectedly outperformed **Hermes 2** in various benchmarks, with specific reference to the MMLU benchmark performance - a significant point for those considering upgrades or new deployments.

**Mistral Deployment and Configuration Insights**: Engineers seeking optimal configurations for **Mistral** deployment gathered valuable advice, with best practices discussed for a dual NVIDIA 3090 setup, VRAM requirements for fp16 precision (~90GB), and quantization strategies. Curious eyes were also pointed towards "thebloke"'s Discord for additional community support.

**Benchmarks Resonating With Personal Experience**: A significant number of posts revolved around performance benchmarks and personal experiences with different models. Particularly intriguing was the reported superiority of **Mistral Large** over GPT-4 for coding tasks, challenging official tests and signaling the need for user-specific benchmarks.

**Discussions Hover Around Model Limitations**: Technical dialogues converged on the inherent limitations of models such as **Mistral** and **Mixtral**, specifically discussing the context size constraints with a 32k token limit for **Mistral-7B-Instruct-v0.2**, and sliding window functionality issues leading to possible performance degradation.

**Fine Tuning and Usage Nuances Explored**: Users shared insights on successfully leveraging models for specific tasks, such as sentiment analysis and scientific reasoning. However, concerns about **Mixtral's** training implementation and requests for a minimalistic guide suggest a demand for clearer documentation within the community.

**Emerging AI Tools and Competitive Landscape**: Enthusiasts and practitioners alike have turned their attention to emerging AI tools, including Kubernetes AI tooling and Anthropic's release of **Claude-3**, sparking discussions on competitive offerings and the importance of open weights for AI models.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Phi-2's Token Limit Hits a Roadblock**: Users discussed the limits of the **Phi-2** model regarding token expansion, with a suggestion that it might behave like a default transformer beyond its configured limit of 2,048 tokens. Caution was advised when altering Phi-2's settings to avoid erratic performance. A link to Phi-2's configuration file was provided [here](https://huggingface.co/microsoft/phi-2/blob/main/config.json#L19).

- **Mac Setup for Engineers**: Community members exchanged a flurry of suggestions for setting up a new Mac, mentioning tools like Homebrew, TG Pro for temperature monitoring, and Time Machine for backups. A YouTube tutorial on Mac setup for Python/ML was highlighted, available [here](https://www.youtube.com/watch?v=mmkDyV59nRo&t=1368s).

- **Scaling AI Model Debate Rages On**: There was a heated debate about the benefits of scaling up AI models. Some users argued that post-500B-1T parameters, efficiency gains are more likely from training techniques than sheer size, citing articles critical of the scaling approach. The contention touched upon the practicality of training 100T parameter models and the potential of smaller models, with one side expressing skepticism and another suggesting a sufficient data threshold like Redpajama v2 could still push scaling benefits. Cost-effectiveness and recent comparisons of AI models were also topics of interest.

- **Claude 3 Piques Interest**: In the general discussion, **Claude 3** captured attention with its potential performance against **GPT-4**. There was interest in inference platforms for function calling models, and advice exchanged on B2B software sales strategies. Additionally, approaches to building knowledge graphs were discussed, with anticipation for a new model to enhance structured data extraction.

- **Diverse Queries on LLMs Addressed**: Questions flew around topics like PPO script availability for LLMs, best platforms for model inference, 1-shot training in ChatML, and fine-tuning AI for customer interactions. A warning against possible model manipulation was shared, along with a Business Insider article for context [here](https://www.businessinsider.com/car-dealership-chevrolet-chatbot-chatgpt-pranks-chevy-2023-12).

- **Praise for Moondream In Project Obsidian**: **Moondream**, a tiny vision language model, received praise for its performance in preliminary testing, with a GitHub link provided for those interested in exploring it [here](https://github.com/vikhyat/moondream).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Open Source AI Nears Milestone**: The Open Source Initiative (OSI) is working on a new draft for an *open-source AI definition* with a monthly release cadence, aiming for a version 1.0 by October 2024, as discussions continue in their public forum. The evolving drafts can be reviewed [here](https://opensource.org/deepdive/drafts).

- **EFF's Legal Stance on DMCA**: The Electronic Frontier Foundation (EFF) has initiated a legal challenge, _Green v. Department of Justice_, against the DMCA's anti-circumvention provisions, claiming they impede access to legally purchased copyrighted content. Details of the case are documented [here](https://www.eff.org/cases/green-v-us-department-justice).

- **Quantization in AI Comes Under Scrutiny**: Debates have risen around quantization in neural networks, especially regarding weights and activations. Researchers have discussed papers like the 'bitlinear paper' and the quantization of activation functions, touching upon the concept of epistemic uncertainty.

- **Safety Alert: Compromised Code via GitHub Malware**: A malware campaign on GitHub has cloned legitimate repositories to distribute malware. A detailed threat analysis by Apiiro is available [here](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/).

- **Challenging Predictive Modeling in Biology**: A user claimed that predictive modeling cannot effectively create economically viable biomolecules due to the complexity of biological systems, indicating a contrast with the more predictable physical models used in engineering.

- **Revolutionizing AI with Counterfactuals**: A new approach named **CounterCurate**, combining **GPT-4V** and **DALLE-3**, leads to visio-linguistic improvements. CounterCurate uses counterfactual image-caption pairs to boost performance on benchmarks. The paper explaining this is available [here](https://countercurate.github.io/).

- **LLMs Overhyped? Functional Benchmarks Suggest So**: Discussions arose from a [Twitter thread](https://x.com/_saurabh/status/1763626711407816930?s=20) questioning over-reported reasoning capabilities of LLMs, referring to functional benchmarks indicating significant reasoning gaps, available [here](http://arxiv.org/abs/2402.19450), with an accompanying [GitHub repository](https://github.com/ConsequentAI/fneval).

- **Terminator Architecture Could Replace Residual Learning**: The **Terminator** network architecture could replace residual learning with its new approach to full context interaction. An [arXiv paper](https://arxiv.org/abs/2401.17948) discusses its potential. Future applications and code release were hinted by community members.

- **AzureML Integration with `lm-eval-harness`**: AzureML users discussed issues and solutions regarding the setup of `lm-eval-harness`. The talk included dependency, CUDA detection, multi-GPU use, and orchestration across nodes, with insights found [here](https://docs.ray.io/en/latest/serve/index.html) and [here](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#multi-gpu-evaluation-with-hugging-face-accelerate).

- **Mamba vs Transformer**: A comparison was drawn between Mamba and Transformer models in terms of their ability to learn and generalize on the PARITY task. Concerns over LSTM, Mamba performance, and the mechanism models used to learn PARITY were voiced, along with a shared [GitHub script](https://github.com/dashstander/automatic-circuits/blob/main/train_mamba.py) for training Mamba.

- **Advancing Dataset Development**: A GitHub repository containing scripts for development of **The Pile** dataset was shared, particularly useful for those working on training language models. The repository and its README can be accessed [here](https://github.com/EleutherAI/The-Pile/tree/master/processing_scripts).

- **Figma Meets Imageio in Creative Animation**: An innovative workflow was mentioned where animation was achieved by manipulating SVG frames created in Figma into a GIF using imageio.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Switch the Bot When Models Misbehave**: Users faced issues with the **Codellama Python 7B** model within LM Studio, and `@heyitsyorkie` suggested switching to the [Magicoder-S-DS-6.7B-GGUF model on Hugging Face](https://huggingface.co/itsdotscience/Magicoder-S-DS-6.7B-GGUF) to fix a "broken quant" problem. Discussions about model support, such as for **LoRAs** and **QLoRA**, indicated they are not yet available, and users cannot upload pdf files directly to LM Studio.
  
- **Data Privacy Alarm Bells Ringing**: Concerns were aired about potential data scraping from personal profiles after an unexpected model response contained identifiable personal details, leading to discussions on the ethics and legality of such practices in training AIs.

- **VRAM: A Hot Topic Among Hardware Geeks**: Several threads touched on the necessity of substantial VRAM, with recommendations for a GPU having at least 24 GB for running large language models efficiently. The discussions pointed to resources like [Debian on Apple M1](https://wiki.debian.org/InstallingDebianOn/Apple/M1), emphasizing the limitations and potential challenges with Apple's unified memory architecture and using Apple M1 Macs for AI work with Linux.

- **Impending Beta Release Buzz**: `@heyitsyorkie` indicated an upcoming beta release of LM Studio would include integration of **Starcoder2-15b**. This discussion was backed by a GitHub pull request [adding support for it to llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5795).

- **The Trials and Errors of Autogen**: Users experienced issues with Autogen integration, such as a **401 error** and slow model loading times in LM Studio. Suggestions for troubleshooting included reinstalling and using the Docker guide with adjustments for Windows system paths as found on [StackOverflow](https://stackoverflow.com/questions/41485217/mount-current-directory-as-a-volume-in-docker-on-windows-10/41489151#41489151).

- **AI Engineering with Distributed LLMs**: A query was raised regarding the development of custom AI agents and running different large language models on various hardware setups, including mentioning specific hardware such as a **3090 GPU**, a **Jetson Orion**, and a **6800XT GPU**. However, there was no additional context or detailed discussion provided on these topics.

- **Short Communications**: A user confirmed the existence of a package on Arch Linux using *yay*, and another inquired about Linux support for a feature without additional context.

- **Need for Clarity in AI Discussions**: Comments indicated a lack of context and clarity in discussions regarding **JavaScript compatibility** with **crew ai**, as well as a mention of *Visual Studio Code* that required further information.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Model Training Hunger Games**: Engineers joked about the hunger of AI models during training, devouring 90GB of memory. Better check those **Gradio** components for deployment, as outdated versions like `ImageEditor` might haunt your dreams.

- **AI Learning Ladder**: From newbie to pro, members are eager to climb the AI learning curve, sharing resources for **CUDA**, **SASS** in Gradio, and **PPO** theory – no safety ropes attached.

- **Chat Conference Calling**: AI community events like conferences in Asheville, NC are the real-world meeting grounds for **GenAI** aficionados. Meanwhile, collaborations emerge for tasks like TTS and crafting book club questions – who said AI doesn't have hobbies?

- **Discord Doubles Down on Diffusers**: `diffusers` scheduler naming issues had everyone double-checking their classes post-update until a [pull request fix](https://github.com/huggingface/diffusers/pull/7192) was merged. Inpainting discussions were illustrated, and **LoRA** adapter implementation advice was dispensed like candy on Halloween.

- **Edgy Bots and Data Detectives**: Creative engineers unleashed bots like DeepSeek Coder 33b and V0.4 Proteus on the [Poe platform](https://poe.com). Others shared breakthroughs in protein anomaly detection and musings on the intersection of AI and music sampling, hinting at an era where AI could be the DJ at your next party.

- **Scheduler Confusion Resolution in Diffusers**: A GitHub issue with incorrect scheduler class names in `diffusers` was resolved by a [pull request](https://github.com/huggingface/diffusers/pull/7192), improving accuracy for AI engineers needing the right tools without the confusion.

- **NLP Model Deployment Drama**: Flask vs. Triton is not an apples-to-apples comparison when deploying NLP models – pick your battle. And if you're on the hunt for efficiency, **Adam optimizer** still wears the crown in some circles, but keep an eye on the competition.

- **Building Bridges to Computer Vision**: The connection between a georeferenced PDF of a civil drawing and GIS CAD is being explored, while curious minds considered the potential of small Visual Language Models for tasks like client onboarding. Glimpses into the synergy of AI and vision are ever-expanding, just beyond the visible spectrum.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **HyperZ⋅Z⋅W Operator Shaking Foundations**: `@alex_cool6` introduced the #Terminator network, blending classic and modern technologies and utilizing the novel **HyperZ⋅Z⋅W Operator**, with the full research [available here](https://arxiv.org/pdf/2401.17948.pdf).

- **Claude 3 Attracts Attention**: Discussions around the **Claude 3 Model** are heating up, with its performance benchmarks stirring the community. A [Reddit thread](https://www.reddit.com/r/singularity/comments/1b6dn1m/claude_3_benchmarks/) showcases the community's investigation into its capabilities.

- **Claude 3 Outperforms GPT-4**: `@segmentationfault8268` found **Claude 3** to outdo GPT-4 in dynamic response and understanding, potentially stealing users away from their existing ChatGPT Plus subscriptions.

- **CUDA Kernel Challenges Persist with Claude 3**: Despite its advancements, Claude 3 seems to lack improvement in non-standard tasks like handling **PyTorch CUDA kernels**, as pointed out by `@twoabove`.

- **Sonnet Enters VLM Arena**: The conversation has ignited interest in **Sonnet**, identified as a Visual Language Model (VLM), and its comparative performance with giants like **GPT4v** and **CogVLM**.

- **Seeking Aid for DPO Adjustment**: `@huunguyen` made a call for collaboration to refine the **Dynamic Programming Optimizer (DPO)**. Interested collaborators are encouraged to connect via direct message.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

**Swap Space on Speed Dial**: Discussion centered on using **Linux VRAM** as swap space with potential speed advantages over traditional disk paging, although possible demand conflicts were noted. Resources like [vramfs on GitHub](https://github.com/Overv/vramfs) and [ArchLinux documentation](https://wiki.archlinux.org/title/Swap_on_video_RAM) were shared.

**Rapid Verification and Chat Retrievals**: Users sought assistance on accessing previous day's live chat discussions and queried about Gmail verification times on lightning.ai, highlighting quick resolution times and the ease of accessing recorded sessions.

**CUDA Conundrums and Triton Tweaks**: Engineers shared insights into CUDA programming difficulties, examining **Triton's relationship to NVCC** and **asynchronous matrix multiplication** in Hopper architecture. Resources such as the [unsloth repository](https://github.com/unslothai/unsloth/blob/dbba69b085b9d6049b57b48b882af7e9f29df5b2/unsloth/kernels/rms_layernorm.py#L53) and the [Triton GitHub page](https://github.com/openai/triton) were highlighted.

**GPU-Powered Databases**: The idea of running databases on GPUs gained traction, with mentions of the [cuDF library](https://github.com/rapidsai/cudf) and reference to a [ZDNet article on GPU databases](https://www.zdnet.com/article/gpu-databases-are-coming-of-age/).

**Mistral's Computation Contemplations**: Debates arose over **Mistral's computing capabilities**, questioning the adequacy of 1.5k H100 GPUs for large-scale model training and discussing asynchronous operations. Links included [NVIDIA's cuBLASDx documentation](https://docs.nvidia.com/cuda/cublasdx/index.html) and a [tweet from Arthur Mensch](https://x.com/arthurmensch/status/1762818733016322168).

**PyTorch Developer Podcast Drops New Episode**: The podcast's episode discussing [AoTInductor](https://pytorch-dev-podcast.simplecast.com/episodes/aotinductor) was shared, echoing community enthusiasm for the series.

**Ring Attention Rings a Bell**: Ring and Striped Attention were hot topics, with references to discussions on the YK Discord and a [Together.ai blog post](https://www.together.ai/blog/flash-decoding-for-long-context-inference). Various code bases like [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) and [flash-attention](https://github.com/Dao-AILab/flash-attention) provided implementation insights.

**CUDA-MODE Lecture Loaded**: Announcement of **Lecture 8 on CUDA performance gotchas** with a promise of tricks for maximizing occupancy and minimizing issues, set to start promptly for eager learners.

**Career Cornerstones**: Job postings by Lamini AI and Quadrature aimed at HPC and GPU Optimization Engineers, highlighting opportunities to work on exciting projects such as optimizing LLMs on AMD GPUs and AI workloads in global financial markets. Details can be found on [Lamini AI Careers](https://jobs.lever.co/laminiai/af688bf8-6c6e-42b5-87aa-0ee9afccdced) and [Quadrature Careers](https://quadrature.ai/).

**Lecture 8 Redux on YouTube**: After technical issues with a prior recording, Lecture 8, titled *CUDA Performance Checklist*, was re-recorded and shared along with corresponding [code samples](https://github.com/cuda-mode/lectures/tree/main/lecture8) and [slides](https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **RAPTOR Elevates RAG Retrieval**: LlamaIndex introduced **RAPTOR**, a new technique for *Retrieval-Augmented Generation* (RAG) that improves the retrieval of higher-level context. Promoting better handling of complex questions, it was announced via [Twitter](https://twitter.com/llama_index/status/1763972097628684607).

- **GAI Enters Urban Planning**: LlamaIndex displayed practical applications of RAG, including a **GAI-powered ADU planner**, aiming to enhance the process of constructing accessory dwelling units [Tweet](https://twitter.com/llama_index/status/1764020728427667605).

- **MongoDB Meets RAG** with LlamaIndex’s new reference architecture, developed by @AlakeRichmond, utilizing @MongoDB Atlas for efficient data indexing, vital for building sophisticated RAG systems, as per a [Twitter update](https://twitter.com/llama_index/status/1764078471276642469).

- **Semantic Strategies Sharpen RAG**: Semantic chunking is spotlighted for its potential to advance RAG's retrieval and synthesis capabilities by grouping semantically similar data, an approach shared by Florian June and picked up by LlamaIndex [Twitter post](https://twitter.com/llama_index/status/1764335221141631471).

- **Claude 3's Triumphant Trio**: Claude 3 has been released with different variants, including Claude Opus, surpassing GPT-4's performance according to LlamaIndex, which has announced immediate support for the model [announcement](https://twitter.com/llama_index/status/1764731195286577247).

- **Leveraging LongContext with LlamaIndex**: Integration of **LlamaIndex** with **LongContext** shows promise for enhancing RAG, especially with Google's recent Gemini 1.5 Pro release that features a 1M context window, which could potentially be incorporated [Medium article](https://medium.com/ai-advances/empowering-long-context-rag-the-integration-of-llamaindex-with-longcontext-6cf014d4d738).

- **Community Corner Catches Fire**: The **LlamaIndex Discord community** was complimented as more organized and supportive than others, particularly in terms of API documentation structure insights and practical guides on setting up sophisticated search systems involving hybrid vector and keyword searching [Y Combinator news](https://news.ycombinator.com/item?id=37764489), [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/), and multiple other resources listed above.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord Summary

- **Claude 3.0 Arrival on OpenRouter**: The much-anticipated **Claude 3** AI has been released, with an exclusive mention of an experimental self-moderated version being available on OpenRouter, as announced by `@alexatallah`.

- **LLM Security Game Sparks Caution**: `@leumon` has launched a game on a server attempting to deceive GPT3.5 into exposing a secret key, underlining the significance of handling AI outputs with caution and safeguarding sensitive data. Players can also engage freely with various AI models like **Claude-v1, Gemini Pro, Mixtral, Dolphin**, and **Yi**.

- **Claude 3 vs GPT-4 Reactions and Tests**: Discussions and reactions to the comparison between Claude 3, including **Claude 3 Opus**, and GPT-4 are ongoing, with users like `@arsoban` noting greater text comprehension in Claude 3 Opus in tests, while others express concerns over its pricing.

- **Performance Debate Heats Up Among AIs**: The capabilities of different Claude 3 variants spurred debates, with shared observations such as Sonnet sometimes outperforming Opus and plans to test Claude 3 for English-to-code translations in gaming applications.

- **AI Deterioration Detected by Community**: `@capitaindave` pointed out what seems to be a diminishing reasoning ability over time in **Gemini Ultra**, sparking discussions on the potential deterioration of model performance after initial release.

**Links mentioned**:
- OpenRouter with Claude 3.0: [OpenRouter](https://openrouter.ai/playground?models=anthropic/claude-instant-1.2)
- LLM Encryption Challenge on Discord: [Discord - A New Way to Chat with Friends & Communities](https://discord.gg/YWX8Eft6R8)
- Claude Performance Test Result Image: [codebyars.dev](https://share.codebyars.dev/u/jGY25U.png)



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **OpenAI Turns a New Page with Browsing Feature**: OpenAI has unveiled a **browsing feature**, prompting excitement for its resemblence to existing tools like Gemini/Perplexity. The announcement was shared via a [tweet from OpenAI](https://twitter.com/wangzjeff/status/1764572262743851339).

- **Claude 3's Promising Debut**: The new **Claude 3** model family is creating buzz for potentially surpassing GPT-4 in tasks involving math and code based on user `@res6969`'s claims. Discussions concerning its cost-efficiency and the anticipation for the **Haiku model** highlight user interest in balancing price with performance.

- **Claude 3's Operative Edge**: Experiments referred to by `@res6969` point to **Claude 3's latency** outperforming others, with first token responses around 4 seconds, demonstrating its operational efficiency in user experiences.

- **Navigating Cost-Effective Embedding Solutions**: With a goal of 100 inferences per second in production, `@iyevenko` explored the most cost-effective embedding models. User `@yikesawjeez`'s recommendations included [Qdrant](https://qdrant.tech/) and [Weaviate](https://www.semi.technology/developers/weaviate/current/).

- **Weighing OpenAI's Embedding Affordability**: Despite initial quality concerns, `@iyevenko` is considering OpenAI's embedding solutions for cloud infrastructure, which appear to be quite cost-effective, especially in light of improvements to their embeddings.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord Summary

- **Anthropic Unveils Claude 3 to Rave Reviews**: [AnthropicAI announced](https://x.com/anthropicai/status/1764653830468428150?s=46) **Claude 3**, its latest series of AI models including Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku, challenging benchmarks in AI performance. Users like `@sid221134224` and `@canadagoose1` expressed their excitement, noting Claude 3's strengths over GPT-4 and its potential due to no reliance on proprietary data sets.

- **Claude 3 Ignites Misinformation and Drama**: The release of **Claude 3** catalyzed the spread of problematic tweets, causing `@natolambert` to intervene directly by addressing misleading posts as "dumb." `@natolambert` also humorously rejected the idea of using an alternate account to combat misinformation due to the effort involved.

- **RL Innovations and Discussions**: A paper on a foundational model for RL was highlighted, discussing a policy conditioned on embedding of the reward function for adaptable generalization ([Sergey Levine's tweet](https://twitter.com/svlevine/status/1764116636142076070)). Concurrently, the community explored the Cohere PPO paper's claim that corrections for Policy Gradient Optimization (PPO) may not be required for Large Language Models (LLMs), sparking interest in verification from other research groups.

- **From Silver Screen to AI Dreams**: `@natolambert` is seeking a **video editing** partner to create a trailer, possibly inspired by the film *Her*, emphasizing AI themes. Additionally, `@natolambert` teased upcoming content and mentioned possible collaboration with Hugging Face's CTO, linking to a discussion about the benefits of **open source AI** ([Logan.GPT’s tweet](https://x.com/officiallogank/status/1764435268021502226?s=46)).

- **The AI Community Embraces Julia**: Amid discussions, `@xeophon.` focused on the merits of the **Julia programming language** for AI development, providing a link to [JuliaLang](https://julialang.org/) for those interested. The conversation indicated a growing engagement with Julia within the engineering community.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Deciphering Tokenizer Mechanics**: A [YouTube tutorial](https://www.youtube.com/watch?v=zduSFxRajkE) was shared by `@lhc1921`, offering insights into constructing a **tokenizer** for Large Language Models (LLMs), highlighting its significance in converting strings to tokens.

- **Galaxy AI Proposes Free API Access**: **Galaxy AI** was introduced by `@white_d3vil` which offers complimentary API services for high-caliber AI models, including **GPT-4**, **GPT-4-1106-PREVIEW**, and **GPT-3.5-turbo-1106**.

- **Tech Stack Advice for Scalable LLM Web Apps**: Mixed suggestions were made on building a scalable **LLM web application**, ranging from using Python 3.11 with FastAPI and Langchain to leveraging Next.js with Langserve.js on Vercel. Langchain's production readiness and customization for commercial use were queried, expressing a preference for custom code in production settings.

- **Beware of Potential Spam Links**: Users are warned against a suspicious link shared by `@teitei40` across multiple channels, claiming to offer a $50 Steam gift card but raising concerns about its legitimacy and potential as a phishing attempt.

- **Innovative Projects and Educational Resources**: The community has showcased a variety of works, including **Devscribe AI**'s YouTube video chat tool, a guide on using generative AI for asset-liability management, and a **Next.js 14+ starter template** for modern web development. Additionally, discussions on enhancing Langchain's **retrieval-augmented generation** and the efficacy of the Feynman Technique for learning were highlighted.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Overflowing with AI Knowledge**: Gemini for Google Cloud is set for a boost with the integration of Stack Overflow's knowledge via [OverflowAPI](https://techcrunch.com/2024/02/29/google-brings-stack-overflows-knowledge-base-to-gemini/), aiming to sharpen AI assistance directly within the cloud console.
  
- **Brin Banks on AGI Breakthroughs**: Google’s co-founder, Sergey Brin, has sparked discussions by suggesting initiatives like Gemini could lead Google’s artificial intelligence towards AGI, as flaunted in a circulating [tweet](https://twitter.com/marvinvonhagen/status/1764036713889116661) about his insights.

- **Perfecting Digital Reality**: [LayerDiffusion](https://github.com/layerdiffusion/sd-forge-layerdiffusion) envisions a new horizon for AI creativity, offering tools to seamlessly insert items with realistic reflections into photos, a promising venture for Stable Diffusion aficionados.

- **Claude 3 Makes a Splash**: Anthropic’s announcement of its Claude 3 model family stirs the AI community with discussions on its advanced metadata awareness and impact on current AI models, with important benchmarks being shared, such as [Claude 3's ~60% accuracy on GPQA](https://x.com/idavidrein/status/1764675668175094169?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **India’s AI Regulatory Chokepoint**: Martin Casado's [tweet](https://x.com/martin_casado/status/1764408870804623753?s=46&t=90xQ8sGy63D2OtiaoGJuww) on India's AI deployment regulations has raised alarms over potential stifling of innovation due to the required government approval, stirring debate among the tech community about the balance between oversight and progress.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Resolved: Hugging Face Commit Chaos**: `@giftedgummybee` reported that the Hugging Face `KTO` issue was resolved by identifying a commit version mix-up. This primarily concerned the Hugging Face transformers library, relevant to **Axolotl's** deployment.

- **Axolotl Staying Put on Hugging Face**: `@nanobitz` clarified that there are no plans to port **Axolotl** to Tinygrad, citing dependency on the Hugging Face transformers library, and reminded users to keep configuration questions to appropriate help channels.

- **Optuna CLI Consideration for Axolotl**: `@casper_ai` suggested the integration of a CLI tool for hyperparameter optimization using Optuna, referencing a [GitHub issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1356) for context.

- **Deep Learning GPU Conundrums and Fixes**: Various GPU-related issues were surfaced, including a `python` vs `python3` conflict and a glitch with deepspeed's final save; however, `@rtyax` did not experience issues with deepspeed 0.13.4's final save function.

- **Mixtral vs. Mistral: Model Preference Showdown**: A discussion was initiated by `@dctanner` comparing **Mixtral** to **Mistral Large** for synthetic data generation, with `@le_mess` expressing a preference for personal models over **Mixtral**, suggesting nuanced performance outcomes for different use-cases.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Aristotelian AI Models Entering the Stage**: [@crispstrobe](https://discord.com/channels/1178995845727785010/1178996063537991752/1213540286941364254) discussed the potential of "Aristotelian Rescoring", a concept that could address complex AI challenges, highlighting related works such as STORIUM, FairytaleQA, and TellMeWhy, with resources available on [GitHub](https://github.com/StonyBrookNLP/tellmewhy) and [Hugging Face](https://huggingface.co/datasets/StonyBrookNLP/tellmewhy).
- **German Semantics Leaping Forward**: `@sten6633` improved German semantic similarity calculations by fine-tuning Deepset's `gbertlarge` with domain-specific texts and Telekom's paraphrase dataset, and turning it into a sentence transformer.
- **Eager for AI-Production Know-how Sharing**: `@dsquared70` invited individuals working with Generative AI in production to speak at an upcoming conference in Asheville, NC, with applications open until [April 30](https://www.aiinproduction.com/cfp).
- **Aligning German Data Delicately**: `@johannhartmann` pointed out a translation error in a dataset but managed to integrate the fixed dataset into [FastEval](https://github.com/mayflower/FastEval), following a bug fix in their evaluation using `./fasteval`.
- **Brezn's Bilingual Breakthrough**: [@thomasrenkert](https://discord.com/channels/1178995845727785010/1197630242815213618/1213464731348762634) lauded **Brezn-7b**'s performance in German, spurred by model merging and aligned with 3 DPO datasets, while [@johannhartmann](https://discord.com/channels/1178995845727785010/1197630242815213618/1213464731348762634) proposed potentially using ChatML by default to improve Brezn's benchmark scores.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Stable Diffusion Goes Extra Large**: **Stable Diffusion XL Lightning** impresses users with its capabilities, as highlighted in the shared demo link: [fastsdxl.ai](https://fastsdxl.ai/).
- **Claude 3 Interaction Now Simplified**: **SimonW** released a new plugin for the **Claude 3** models, with the repository available on [GitHub](https://github.com/simonw/llm-claude-3).
- **Artichoke Naming Gets Creative**: One user infuses humor into the discussion by suggesting whimsical names for artichokes such as "Choke-a-tastic" and "Arti-party."
- **Mistral Model Prices Noticed**: The **Mistral large model** earns praise for its data extraction performance, but also noted for its higher-than-desired cost.
- **Plugin Development Speed Wins Applause**: The development pace of the new plugin for interacting with the Claude 3 model garners quick commendation from the community.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **New Collaboration Opportunity**: `@wasooli` has shown a keen interest to collaborate within the **Alignment Lab AI** community, with `@taodoggy` open to further discussions through direct messaging.
- **GenAI Conference Call**: `@dsquared70` has announced a **GenAI in production** conference, encouraging submissions by **April 30th**. More information and application details can be found at [AI in Production](https://www.aiinproduction.com/cfp).



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Call for AI Integration Expertise**: Developers actively integrating **GenAI** into production systems are invited to share their insights at a conference in Asheville, NC. Interested parties can submit their papers by April 30 at [AI in Production Call for Presentations](https://www.aiinproduction.com/cfp).

- **A Comical Start to the Day**: An attempt to brighten the chat with "good morning yokks" brought a humorous start to the discussions despite the typo corrected to "yolks".



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **Hackathon Hierarchy Explained**: `@needforspeed4` raised a question about whether the **Agape hackathon** is related to the **AI Engineer Foundation** managing the Discord server, and the use of separate Discords for different hackathons. `@hackgoofer` clarified that the **AI Engineer Foundation Hackathons** are held within this Discord, but pointed out that the **Agape hackathon** operates independently.



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1213408815492374580) (994 messages🔥🔥🔥): 

- **Claude 3 AI, Safety or Over-Caution?**: Claude 3, a new AI model, is showcased by users highlighting its heightened sensitivity to potentially offensive content or copyright concerns.
- **Rumination Over AI's Role in Game Development**: Some users predict AI will be involved in the rendering and creation of future video games; however, netrve voices skepticism about the capability to bruteforce current AI limitations with raw compute.
- **Contentious NVIDIA Licensing Tactics**: Nvidia's efforts to restrict the use of CUDA on non-NVIDIA hardware through licensing terms spurs discussion on the legality and impact on developers, especially regarding translation layers.
- **Benchmarks and OpenAI's Future**: Models like Phind 70b are discussed while users question the reliability of benchmarks and the significance of ongoing AI model releases with the anticipation of GPT-5.
- **Technical Deep Dive into GPU Technologies**: Netrve discusses the complexities and advancements in game rendering, including Epic's Nanite system in Unreal Engine 5, while others lament restrictive moves by NVIDIA.

**Links mentioned**:

- [no title found](https://www.marktechpost.com/2024/03/03/meet-phind-70b-an-artificial-intelligence-ai-model-that-closes-execution-speed-and-the-code-generation-quality-gap-with-gpt-4-turbo/?amp=): no description found
- [no title found](https://www.marktechpost.com/2024/03/03/meet-phind-70b-an-artificial-intelligence-ai-model-that-clos): no description found
- [AI Open Letter - SVA](https://openletter.svangel.com/): Build AI for a Better Future
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/AnthropicAI/status/1764653833970659560?s=20): With this release, users can opt for the ideal combination of intelligence, speed, and cost to suit their use case.  Opus, our most intelligent model, achieves near-human comprehension capabilities. I...
- [Nvidia bans using translation layers for CUDA software &mdash; previously the prohibition was only listed in the online EULA, now included in installed files [Updated]](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers): Translators in the crosshairs.
- [Fr Lie GIF - Fr Lie - Discover &amp; Share GIFs](https://tenor.com/view/fr-lie-gif-9063740662564569899): Click to view the GIF
- [Google-backed Anthropic debuts its most powerful chatbot yet, as generative AI battle heats up](https://www.cnbc.com/2024/03/04/google-backed-anthropic-debuts-claude-3-its-most-powerful-chatbot-yet.html): Anthropic on Monday debuted Claude 3, a chatbot and suite of AI models that it calls its fastest and most powerful yet.
- [Google is rapidly turning into a formidable opponent to BFF Nvidia &mdash; the TPU v5p AI chip powering its hypercomputer is faster and has more memory and bandwidth than ever before, beating even the mighty H100](https://www.techradar.com/pro/google-is-rapidly-turning-into-a-formidable-opponent-to-bff-nvidia-the-tpu-v5p-ai-chip-powering-its-hypercomputer-is-faster-and-has-more-memory-and-bandwidth-than-ever-before-beating-even-the-mighty-h100): Google's latest AI chip is up to 2.8 times faster at training LLMs than its predecessor, and is fitted into the AI Hypercomputing architecture
- [GPU Reviews, Analysis and Buying Guides | Tom's Hardware](https://www.tomshardware.com/pc-components/gpus): no description found
- [Lone (Hippie)](https://huggingface.co/Lone): no description found
- [Turbo (Chen)](https://huggingface.co/Turbo): no description found
- [pip wheel - pip documentation v24.0](https://pip.pypa.io/en/stable/cli/pip_wheel/#cmdoption-only-binary>): no description found
- [GitHub: Let’s build from here](https://github.com): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...
- [Fix defaults + correct error in documentation for Mixtral configuration by kalomaze · Pull Request #29436 · huggingface/transformers](https://github.com/huggingface/transformers/pull/29436): What does this PR do?  The default value for the max_position_embeddings was erroneously set to 4096 * 32. This has been corrected to 32768 Mixtral does not use Sliding Window Attention, it is set ...
- [Add Q4 cache mode · turboderp/exllamav2@bafe539](https://github.com/turboderp/exllamav2/commit/bafe53972840cb0d0673bbd85a3afdeab360a9ab#diff-71431ec327109cd8333884920a1573a325ed1eea3dea804d6bc652f91a4a91f8): no description found
- [[Mixtral] Fixes attention masking in the loss by DesmonDay · Pull Request #29363 · huggingface/transformers](https://github.com/huggingface/transformers/pull/29363): What does this PR do?   I think there may be something not quite correct in load_balancing_loss. Before submitting   This PR fixes a typo or improves the docs (you can dismiss the other checks if t...
- [GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets](https://github.com/e-p-armstrong/augmentoolkit): Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1213410060785684571) (379 messages🔥🔥): 

- **Understanding Llama.cpp Limitations**: `@pri9278` noted that while SD (Sparse Diffusion) and Lookup decoding are implemented in llama.cpp, they are not integrated into server APIs, which limits the capabilities of the server-sided implementation of the model.
- **Model Performance and Hardcoding**: `@superking__` discussed the complexity of hardcoding models, noting the difficulty when using transformers and the possibilities when using strict formats for model prompting.
- **Discussions on Roleplay and Story Generation**: Chat members, including `@gamingdaveuk`, `@netrve`, `@lisamacintosh`, and `@concedo`, engaged in complex discussions about using AI models for roleplaying and story generation, exploring topics like context caching for optimization, front-end/user interface quirks, and specific use cases for chatbots in roleplay scenarios.
- **Sharing Experiences with Fine-tuned Models**: `@c.gato` shared their experience testing the Thespis-CurtainCall Mixtral model, commenting on its performance with complex tasks like playing tic-tac-toe and generating prompts based on greentext stories.
- **Engaging with AutoGPT and DSPY**: `@sunija` inquired about the status of AutoGPT and its applications in roleplay, prompting replies from `@wolfsauge` and `@maldevide` discussing alternative methods, such as DSPY, for optimized prompt generation and automatic evaluation of response variations.

**Links mentioned**:

- [Constructive](https://xkcd.com/810/): no description found
- [Chub](https://chub.ai/characters/illuminaryidiot/vixens-of-the-orient-express-freeplay-a39e7fe1>): Find, share, modify, convert, and version control characters and other data for conversational large language models (LLMs). Previously/AKA Character Hub, CharacterHub, CharHub, CharaHub, Char Hub.
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard): no description found
- [cgato/Thespis-CurtainCall-8x7b-v0.3 · Hugging Face](https://huggingface.co/cgato/Thespis-CurtainCall-8x7b-v0.3): no description found
- [Thanos Memoji GIF - Thanos Memoji - Discover &amp; Share GIFs](https://tenor.com/view/thanos-memoji-gif-23490017): Click to view the GIF
- [ZeroBin.net](https://zerobin.net/?946f95701988d7a9#qkUcZ1pb/9O5nK4Zal): no description found
- [ZeroBin.net](https://zerobin.net/?946f95701988d7a9#qkUcZ1pb/9O5nK4ZalZTRztZwuZnU3hwBu9cK3hgLVo=): no description found
- [Mihawk Zoro GIF - Mihawk Zoro One piece - Discover &amp; Share GIFs](https://tenor.com/view/mihawk-zoro-one-piece-gif-15330479296855641524): Click to view the GIF
- [Worldsgreatestswordsmen Onepiece GIF - Worldsgreatestswordsmen Onepiece Mihawk - Discover &amp; Share GIFs](https://tenor.com/view/worldsgreatestswordsmen-onepiece-mihawk-anime-one-gif-25849503): Click to view the GIF
- [Cat Cat Meme GIF - Cat Cat meme Funny cat - Discover &amp; Share GIFs](https://tenor.com/view/cat-cat-meme-funny-cat-cat-eating-cat-eating-chips-gif-10455465908695706650): Click to view the GIF
- [Rapeface Smile GIF - Rapeface Smile Transform - Discover &amp; Share GIFs](https://tenor.com/view/rapeface-smile-transform-gif-12599812): Click to view the GIF
- [GGUF quantizations overview](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9): GGUF quantizations overview. GitHub Gist: instantly share code, notes, and snippets.
- [Family guys - Sony Sexbox](https://youtu.be/7ciVKIm7bcg?si=2Tu1DYbRtuRhkgXT): Clip from Family Guy S03E16

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1213458845616177172) (39 messages🔥): 

- **Fine-tuning Troubles**: `@coldedkiller` experienced issues fine-tuning an **OpenOrca Mistral 7b model**; after converting to 'gguf' format the model failed to give correct outputs for both its own data and the data it was fine-tuned on.
- **Cosine Similarity Cutoffs in Training Models**: `@gt9393` inquired about the appropriate cosine similarity cutoff for models, leading `@dirtytigerx` to respond that it depends on various factors, and no hard cutoff can be provided.
- **Use of Special Tokens and Model Training**: `@gt9393` discussed uncertainties regarding the inclusion of start and end of sequence tokens in datasets. `@dirtytigerx` recommended having these tokens, but appending them after the prompt has been encoded.
- **Chess Model Training Achievement**: `@.haileystorm` shared their success training an 11M parameter **Mamba LM chess model**, offering links to the relevant resources, training code, and indicating it plays better as white. The model's training was compared to a larger parameter model and showcased a **37.7% win rate** against Stockfish level 0.
- **Seeking Fine-Tuning Guidance for Small to Medium LLMs**: Users `@coldedkiller` and `@zelrik` sought advice for fine-tuning language models, being directed to resources by Jon Durbin and a guide from **UnslothAI**. Discussions covered format, special tokens, and hardware requirements with `@maldevide` providing insights on preprocessing book texts, hardware capacities, and tools for parameter-efficient fine-tuning (PEFT).

**Links mentioned**:

- [HaileyStorm/MambaMate-Micro · Hugging Face](https://huggingface.co/HaileyStorm/MambaMate-Micro): no description found
- [GitHub - jondurbin/bagel: A bagel, with everything.](https://github.com/jondurbin/bagel): A bagel, with everything. Contribute to jondurbin/bagel development by creating an account on GitHub.
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth?tab=readme-ov-file#-finetune-for-free): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1213458904906727496) (1 messages): 

- **Fine-tuning Woes with OpenOrca**: `@coldedkiller` is experiencing issues with a **fine-tuned OpenOrca Mistral 7b model**. After converting it to `gguf` format, the model fails to produce proper output on both original and fine-tuned datasets.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1213459165041655818) (11 messages🔥): 

- **OpenOrca Fine-tuning Woes**: User `@coldedkiller` is facing issues where their fine-tuned OpenOrca Mistral 7b model isn't outputting expected answers post conversion to gguf format. `@spottyluck` suggests checking the model's performance pre-quantization and considering the use of imatrix if there's a problem with outliers.
  
- **GPTQ Out of the Spotlight**: `@yeghro` queries if GPTQ is no longer in focus since TheBloke has stopped releasing more about it, and `@_._pandora_._` hints at rumors that *TheBloke is missing*, contributing to no recent releases.

- **Model Test Dilemma**: `@gamingdaveuk` seeks the smallest possible model to load on a 6GB VRAM laptop for API call tests. They mention finding an answer on Reddit suggesting the use of *Mistral instruct v0.2*, and `@dirtytigerx` advocates for any gguf quant model as long as it's around 4GB in size.

- **Coldedkiller's Model Mishap**: In a follow-up, `@coldedkiller` elaborates on the issue with their fine-tuned model not providing answers from their trained Q&A dataset after format conversion. They observe the model gives irrelevant responses when queried.

- **Ajibawa_2023 Showcases Enhanced Models**: User `@ajibawa_2023` shares links to their fine-tuned models boasting enhanced coding capabilities. One model, **[OpenHermes-2.5-Code-290k-13B](https://huggingface.co/ajibawa-2023/OpenHermes-2.5-Code-290k-13B)**, incorporates their dataset, performs well in coding rankings, and can handle various tasks including blogging and story generation.

**Links mentioned**:

- [ajibawa-2023/OpenHermes-2.5-Code-290k-13B · Hugging Face](https://huggingface.co/ajibawa-2023/OpenHermes-2.5-Code-290k-13B): no description found
- [ajibawa-2023/Code-290k-6.7B-Instruct · Hugging Face](https://huggingface.co/ajibawa-2023/Code-290k-6.7B-Instruct): no description found

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1213492101631840336) (128 messages🔥🔥): 

- **GPT Alternatives Discussed Amidst Downtime**: User `@whodidthatt12` expressed frustration with GPT being down and inquired about alternative AI writing assistants. Suggestions included *Bing Chat, Hugginface, Deft GPT, LeChat, together.xyz*, and *Stable Chat*. 

- **Claude 3 AI Impressions**: `@glamrat` mentioned testing Anthropic's Claude 3, finding it impressive, especially the free Sonnet model. Various users are discussing their experiences and expectations, from using Claude 3 for math tutoring (`@reynupj`) to potentially dropping a GPT Plus subscription in favor of Claude (`@treks1766`).

- **Enthusiasm for AI Competition**: Users like `@treks1766` and `@lolrepeatlol` expressed their excitement about the competition between AI services like Claude 3 and GPT-4, anticipating benefits for consumers and advancements in the AI field.

- **Debate Over AI Model Capabilities**: Some users argued over the reported superiority of Claude 3 over OpenAI's models (`@darthcourt.`, `@hanah_34414`, `@cosmosraven`), with comments ranging from skepticism (`@drinkoblog.weebly.com`) to anticipation for the next big release by OpenAI.

- **Cost Considerations and Availability**: Concerns were raised about the cost of using Claude 3's API (`@dezuzel`) and the availability of different models in various regions. There is anticipation around how existing services like Perplexity AI Pro will integrate with new models like Claude 3 (`@hugovranic`).

**Links mentioned**:

- [Anthropic says its latest AI bot can beat Gemini and ChatGPT](https://www.theverge.com/2024/3/4/24090087/anthropic-claude-3-opus-ai-chatbot-multimodal): Claude 3 is here with some big improvements.
- [OpenAI Status](https://status.openai.com/): no description found

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1213485605443346462) (38 messages🔥): 

- **GPT Alternatives Sought Amidst Downtime**: User `@whodidthatt12` is seeking alternative AI options for writing assignments due to GPT being down.
- **Custom CSVs for AI Knowledge Bases**: `@.bren_._` inquired if custom agents could utilize CSV files as part of their knowledge bases and was experiencing technical difficulties confirming if it's a valid file type for use.
- **File Types and Technical Support in Custom Agents**: `@.bren_._` shared an error message about accessing system root directories, while `@darthgustav.` suggested using row-separated values in plain text files as a more successful approach.
- **Finding the Most Optimal GPT for Code**: `@yuyu1337` is searching for a GPT model that generates code with optimal time/space complexity, with other users like `@eskcanta` and `@beanz_and_rice` contributing to the discussion on achieving optimality and providing creative pseudo code.
- **GPT Store Publishing Paths Clarified**: `@bluenail65` queries about the necessity of a website to list GPTs in the store, to which `@solbus` clarifies the options for publishing, including using a billing name or sharing privately via a link.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1213497548589899818) (506 messages🔥🔥🔥): 

- **Humor Struggles in API**: `@dantekavala` is experiencing a discrepancy where tests in ChatGPT work well for prompting a humorous writing style, but the same approach fails when used with GPT 3.5 API; the API's output remains consistent, unaffected by the requested style. They've tried various styles and reached out for guidance in the Developers Corner.
  
- **Owl and Palm Puzzle Persists**: Many participants, including `@madame_architect`, `@aminelg`, and `@eskcanta`, have engaged in a lively exploration of the "owl and palm" brain teaser. While they have all attempted various prompting strategies to accurately solve the puzzle using GPT-V, none have achieved consistent success.

- **Prompt Engineering Tactics Discussed**: User `@madame_architect` suggests using multiple prompting tactics like the "take a deep breath" trick from the system 2 thinking paper and points from emotionprompt (tm) to tackle the problem. However, `@eskcanta` notes that the core issue might be with the Vision model's training, not so much the prompting methods themselves.

- **Vision Model’s Limitations**: Despite testing various prompts and theories about the Vision model's understanding of image measurement, users like `@eskcanta` and `@darthgustav.` highlight that the model's failure to consistently interpret measurements correctly may stem from the need for additional training, rather than prompting inadequacies.

- **Feedback on Personal Creations**: Newcomer `@dollengo` inquires about creating and training AI for educational purposes, with an intention to publish, but there is a focus on staying within OpenAI's dialogue and sharing policies. Users `@eskcanta` and `@aminelg` give advice respecting the platforms terms of service and prompt-writing practices for the AI models.

**Links mentioned**:

- [Terms of use](https://openai.com/policies/terms-of-use): no description found
- [DALL·E 3](https://openai.com/dall-e-3): DALL·E 3 understands significantly more nuance and detail than our previous systems, allowing you to easily translate your ideas into exceptionally accurate images.

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1213497548589899818) (506 messages🔥🔥🔥): 

- **Puzzle Prompt Engineering Saga Continues**: Users `@aminelg`, `@eskcanta`, `@darthgustav.`, and `@madame_architect` continued their efforts to craft the perfect advanced prompt for an AI vision puzzle involving an owl and a tree. Despite various strategies, issues persisted with GPT-V accurately interpreting the image, leading to discussions about the model's limitations and potential need for retraining.

- **The Highs and Lows of Model Behavior**: Across multiple attempts with nuanced prompts (like `@madame_architect`'s which achieved a singular success), GPT-V consistently misinterpreted the measurement of the 200 units on the right side of the image, often confusing it with the full height of the tree, making it an observable weakness in the model's capabilities.

- **Playful Competition Heats Up**: Discussions turned humorous as `@aminelg` and `@spikyd` exchanged jests about reaching their usage limits and joked about generating prompts that would outperform the AI's current understanding of the complex image teasing out the occasional correct response as a "10 points to GPT V" moment.

- **Sharing Knowledge Comes at a Cost**: `@darthgustav.` expressed frustration with the Discord server's auto-moderation, which limited his ability to discuss certain details and share prompts, triggering calls for OpenAI to revise system prompt restrictions for a more transparent and conducive prompt engineering discussion.

- **Newcomer Queries and Tips Exchange**: New participants like `@snkrbots`, `@chenzhen0048`, and `@dollengo` sought advice on prompt engineering and AI training, eliciting responses from veteran contributors. Ideas exchanged included improving prompts with template structuring, asking GPT for refinement aid, and the potential for AI to assist in content creation tasks.

**Links mentioned**:

- [Terms of use](https://openai.com/policies/terms-of-use): no description found
- [DALL·E 3](https://openai.com/dall-e-3): DALL·E 3 understands significantly more nuance and detail than our previous systems, allowing you to easily translate your ideas into exceptionally accurate images.

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1213430364501049414) (618 messages🔥🔥🔥): 

- **Potential Perplexity Subscription Issue**: User `@vivekrgopal` expressed frustration about being charged for an annual subscription after attempting to cancel during the trial period. They requested assistance for a refund through direct messages.
- **Users Eager for New AI Integrations**: There's anticipation among users like `@bioforever` and `@sebastyan5218` for Perplexity to integrate new language models such as Claude 3 and Gemini Ultra, highlighting the community's desire for the latest AI advancements.
- **Discussion on Perplexity AI's Effectiveness**: User `@names8619` cheered on Perplexity Pro's performance, comparing it favorably against YouTube for research without clickbait, while others mentioned challenges with OpenAI's GPT-3 results needing to switch to models like Mistral for certain topics.
- **Uncertainty Over AI Model Availability**: Users `@gooddawg10` and `@fluxkraken` discussed the availability of certain AI models (Gemini Ultra, Claude 3) within Perplexity, with some confusion about which models are accessible to users.
- **Comparison of AI Models and Their Responses**: User `@dailyfocus_daily` shared a benchmark question regarding sorting labeled balls into boxes and compared the varied answers given by different AI models including GPT-4, Claude 3, and others, illustrating the inconsistencies in their problem-solving abilities.

**Links mentioned**:

- [AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): no description found
- [GitHub Next | GPT-4 with Calc](https://githubnext.com/projects/gpt4-with-calc/): GitHub Next Project: An exploration of using calculation generation to improve GPT-4&#x27;s capabilities for numeric reasoning.
- [Tweet from Ananay (@ananayarora)](https://fxtwitter.com/ananayarora/status/1762439921825120578): Just ported Perplexity to the Apple Watch! 🚀@perplexity_ai
- [The One Billion Row Challenge in Go: from 1m45s to 4s in nine solutions](https://benhoyt.com/writings/go-1brc/): no description found
- [Oliver Twist GIF - Oliver Twist - Discover &amp; Share GIFs](https://tenor.com/view/oliver-twist-gif-26543489): Click to view the GIF
- [David Leonhardt book talk: Ours Was the Shining Future, The Story of the American Dream](https://www.youtube.com/watch?v=ovkwsvbGq1I): Join Professor Jeff Colgan in conversation with senior New York Times writer David Leonhardt as they discuss David’s new book, which examines the past centur...
- [SmartGPT: Major Benchmark Broken - 89.0% on MMLU + Exam&#39;s Many Errors](https://youtu.be/hVade_8H8mE): Has GPT4, using a SmartGPT system, broken a major benchmark, the MMLU, in more ways than one? 89.0% is an unofficial record, but do we urgently need a new, a...
- [Perplexity.ai Turns Tables on Google, Upends SEO Credos](https://spectrum.ieee.org/perplexity-ai): AI search leader mixes Meta-built smarts with scrappy startup fervor
- [PerplexityBot](https://docs.perplexity.ai/docs/perplexitybot): We strive to improve our service every day. To provide the best search experience, we need to collect data. We use web crawlers to gather information from the internet and index it for our search engi...
- [GitHub - danielmiessler/fabric: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere.](https://github.com/danielmiessler/fabric): fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere. - ...

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1213417151206264852) (20 messages🔥): 

- **Exploring Identity Access Management**: User `@riverborse` shared a [link](https://www.perplexity.ai/search/What-is-iam-o2tdFsxGRraeKVWSzo.fIg) diving into what identity access management (IAM) entails.
- **Understanding Perplexity v2**: `@scarey022` provided a [link](https://www.perplexity.ai/search/What-is-perplexity-v2XimT_gTp6evpknTUvSUg) to learn more about the concept of perplexity in language models.
- **In Search of Optimal Solutions**: User `@dtyler10` posted a [link](https://www.perplexity.ai/search/Create-an-optimal-Nrj9EJpnQ0KSI7vHs8Siaw) that leads to discussions about creating optimal settings, environments, or outcomes.
- **Technical Insights Offered**: A technical explanation was the focus of a [link](https://www.perplexity.ai/search/A-technical-explanation-LEqjHwN4Qa613cx57Dg9AQ?s=m) shared by `@imigueldiaz`.
- **AI Basics Explored**: `@krishna08011` and `@elpacon64` shared links ([link1](https://www.perplexity.ai/search/What-are-AI-bQw5YfH6RdOkhyPdrw7XnA), [link2](https://www.perplexity.ai/search/What-are-AI-BQsp1PCvS5WWDM0tmj3w0g)) discussing what AI is and its various aspects.
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1213428411372933151) (27 messages🔥): 

- **Confusion over Random Number Generator Ethics**: User `@moistcornflake` expressed amusement and confusion over **codellama** providing an ethical warning when asked to create a random number generator. The bot response suggested prioritizing content that promotes positive values and ethical considerations.

- **Performance Issues Noted for Time-Sensitive Queries**: `@brknclock1215` observed an improvement in general quality but reported continued failures in time-sensitive queries and reminisced that it used to perform better in such tasks.

- **Missing Feature for YouTube Summarization**: `@rexx.0569` highlighted the absence of a feature that summarized YouTube videos, which seemed to have been a native function of **Perplexity**. They noted that the feature isn't accessible on different devices.

- **Inquiry About Perplexity API Usage**: `@marvin_luck` sought help on how to achieve the same effects as a web request through **Perplexity API**. To which `@icelavaman` shared a discord link, presumably with relevant information, [Perplexity API Info](https://discord.com/channels/1047197230748151888/1118264005207793674/1213229028870324305).

- **Users Anticipate Citation Feature Access**: `@_samrat` and `@brknclock1215` are waiting to gain access to citations in the API, and `@icelavaman` mentioned that this process might take 1-2 weeks or more. `@brknclock1215` later confirmed seeing improvement in response quality and eagerly awaits the addition of citations.

- **Temperature Settings Discussion**: `@brknclock1215`, `@thedigitalcat`, and `@heathenist` engaged in a discussion about how temperature settings in AI models affect the naturalness and reliability of language outputs. They suggested that lower temperature settings don't always guarantee more reliable outputs and touched upon the complexity of natural language and self-attention mechanisms.

**Links mentioned**:

[Perplexity Blog](https://blog.perplexity.ai/faq/what-is-collections)): Explore Perplexity&#39;s blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1213481272891146270) (213 messages🔥🔥): 

- **Broken Discord Links**: Users `@v01338` and `_._pandora_._` mentioned that both Discord and LinkedIn links on the Mistral AI website are broken. `_._pandora_._` confirmed this by checking HTML source.
- **Discussion on Model Lock-In Scenarios**: `@justandi` asked if migrating from one model to another in an enterprise context could lock in a specific implementation. `@mrdragonfox` chimes in saying that the inference API is similar across platforms, hinting at seamless migration.
- **Concerns Over Model Benchmarking Transparency**: `@i_am_dom` expressed concerns about the lack of published scores for specific Mistral model benchmarks, suggesting that transparency is essential, especially from benchmark owners.
- **Ollama and VLLM Discussion for Mixtral Inference**: `@distro1546` inquired about achieving sub-second inference times with Mixtral using an A100 server and was advised by `@mrdragonfox` to consider exllamav2 or vLLM deployment with 6bpw instead of using llama.cpp, which doesn't fully utilize GPU capabilities.
- **Clarification on Mixtral's Context Window**: `@_._pandora_._` and `@i_am_dom` discuss confusion regarding Mistral and Mixtral's context sizes and sliding window functionality. A Reddit update and documentation inaccuracies in Hugging Face were mentioned, highlighting the need for HF to update their documents.

**Links mentioned**:

- [vLLM | Mistral AI Large Language Models](https://docs.mistral.ai/self-deployment/vllm/): vLLM can be deployed using a docker image we provide, or directly from the python package.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18k0fek/psa_you_can_and_may_want_to_disable_mixtrals/): no description found
- [Mixtral Tiny GPTQ By TheBlokeAI: Benchmarks and Detailed Analysis. Insights on Mixtral Tiny GPTQ.](https://llm.extractum.io/model/TheBlokeAI%2FMixtral-tiny-GPTQ,2VHCHigcDcquIs0aVBv3Ea): LLM Card: 90.1m LLM, VRAM: 0.2GB, Context: 128K, Quantized.
- [You Have GIF - You Have No - Discover &amp; Share GIFs](https://tenor.com/view/you-have-no-idea-gif-27149353): Click to view the GIF
- [TheBloke/Mixtral-8x7B-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF): no description found
- [Mixtral](https://huggingface.co/docs/transformers/en/model_doc/mixtral): no description found

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1213412767990685727) (79 messages🔥🔥): 

- **Mistral Large Surprises in Coding**: `@claidler` reported better performance with **Mistral Large** than GPT-4 for coding tasks, despite official tests suggesting GPT-4's superiority. They observed Mistral Large providing correct solutions where GPT-4 failed repeatedly, raising questions about the tests' accuracy or applicability in certain scenarios.

- **Personal Benchmarks Matter Most**: `@tom_lrd` advised that **personal experience** with models should be considered the best benchmark and recommended trying different models with the same input to see their performance on specific use cases.

- **Mistral Next's Speed Questioned**: `@nezha___` inquired if **Mistral Next** is smaller than Mistral Large, noting its quicker responses, and wondering if its speed is due to being a Mixture of Experts (MoE) model.

- **Context Size Limit Clarified**: Conversation between `@fauji2464`, `@mrdragonfox`, and `._pandora_._` discussed warnings about exceeding the model’s maximum length when using **Mistral-7B-Instruct-v0.2**. It was clarified that the model will ignore content beyond the **32k token** limit, leading to performance issues.

- **LLM Context Windows Explained**: `._pandora_._` explained Large Language Models (LLMs) like Mistral and Mixtral have a "narrow vision" and can only consider up to **32k tokens** for their current context in each inference cycle. If input exceeds this, extra content is ignored, but the model will still produce output based on the last 32k tokens.

**Links mentioned**:

[LLM Visualization](https://bbycroft.net/llm): no description found

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1213921912879579179) (17 messages🔥): 

- **Seeking Mistral Deployment on Dual 3090s**: User `@generalenthu` inquired about the best approach for setting up **Mistral** on a system with 2x NVIDIA 3090 GPUs, aiming for minimal quantization and seeking advice on managing the trade-off between speed and using GPU vs RAM.
- **VRAM Requirements for fp16**: `@mrdragonfox` informed that using fp16 precision would require approximately **90GB of VRAM** for running the model.
- **Model Run with Exllama**: `@mrdragonfox` mentioned that on a 48GB VRAM setup, one can run Mistral with about **5-6 bits per word (bpw)** using **exllama** configuration just fine.
- **How to Start Setup and Use Quants**: `@mrdragonfox` advised `@generalenthu` to start with a "regular oobabooga" as a default setup, access *N inferences*, and use quantization models from **lonestriker** and **turboderp** available on **Hugging Face**.
- **Additional Resources and Community Support**: `@mrdragonfox` suggested that `@generalenthu` join "thebloke"'s Discord for further support from a community that assists with local model deployment, noting that it could be a supplement to the current community for this specific use case.
  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1214156902918000702) (1 messages): 

- **Request for Minimalistic Mistral Training Guide**: User `@casper_ai` mentioned that the community faces challenges in achieving optimal results with the **Mixtral model**. They referenced previous conversations which suggest an implementation discrepancy in the Huggingface trainer, and asked for a minimalistic reference implementation of Mixtral training.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1213512787087859772) (1 messages): 

- **Smaug-Mixtral Outperforms Mixtral-8x7b**: `@bdambrosio` mentioned that **Smaug-Mixtral** surpasses **mixtral-8x7b-instruct-v0.1** in 8bit exl2 quant tests, specifically for applications in **long-context scientific reasoning and medium length report writing**. Exact performance metrics were not provided, but outcomes may vary based on use case.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1213782556814741535) (3 messages): 

- **Collaborative AI for Offline LLM Agents**: User `@yoan8095` shared their work on using **Mistral 7b for LLM Agents** that operate offline, coupling it with a neuro-symbolic system for better planning. The repository available at [HybridAGI on GitHub](https://github.com/SynaLinks/HybridAGI) allows for *Graph-based Prompt Programming* to program AI behavior.
- **Feature-Rich Discord Bot Announcement**: `@jakobdylanc` promotes their Discord bot capable of interfacing with over 100 **LLMs**, offering features such as *collaborative prompting*, *vision support*, and *streamed responses*, all within 200 lines of code. The project is outlined on [GitHub](https://github.com/jakobdylanc/discord-llm-chatbot).
- **Mistral-Large's Formatting Flaws**: `@fergusfettes` reports that while **Mistral-large** produces good results, it struggles with formatting and switching between *completion mode* and *chat mode*. They shared a video demonstrating how *loomed* integration of different LLMs can work at [Multiloom Demo: Fieldshifting Nightshade](https://youtu.be/xiQDGxqEals).

**Links mentioned**:

- [Multiloom Demo: Fieldshifting Nightshade](https://youtu.be/xiQDGxqEals): Demonstrating a loom for integrating LLM outputs into one coherent document by fieldshifting a research paper from computer science into sociology.Results vi...
- [GitHub - jakobdylanc/discord-llm-chatbot: Supports 100+ LLMs • Collaborative prompting • Vision support • Streamed responses • 200 lines of code 🔥](https://github.com/jakobdylanc/discord-llm-chatbot): Supports 100+ LLMs • Collaborative prompting • Vision support • Streamed responses • 200 lines of code 🔥 - jakobdylanc/discord-llm-chatbot
- [GitHub - SynaLinks/HybridAGI: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected](https://github.com/SynaLinks/HybridAGI): The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected - SynaLinks/HybridAGI

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1214232800719409182) (13 messages🔥): 

- **Kubernetes AI Tooling Made Easy**: `@alextreebeard` shared their open-sourced package meant to simplify setting up AI tools on Kubernetes, inviting users for feedback. The tool can be found at [GitHub - treebeardtech/terraform-helm-kubeflow](https://github.com/treebeardtech/terraform-helm-kubeflow).
- **The Arrival of Claude-3**: `@benjoyo.` linked the Anthropic AI's announcement of their new model family, [Claude-3](https://www.anthropic.com/news/claude-3-family), and hinted a query about when a comparable "mistral-huge" might be released.
- **Model Training Takes Time**: In response to a query related to Mistral's response to new competition, `@mrdragonfox` explained that large models take quite a while to train, with large versions only recently coming out.
- **Competition Heating Up**: Following early testing, `@benjoyo.` observed that Anthropic's new model is "extremely capable and ultra steerable/adherent," while continuing to champion the value of open weights for differentiation.
- **New AI Model Pricing Discussed**: `@nunodonato` reflected on the costliness of the new models, while `@mrdragonfox` provided specific pricing for Opus model usage, with input costing $15 per Mega Token (MTok) and output at $75 per MTok.

**Links mentioned**:

[GitHub - treebeardtech/terraform-helm-kubeflow: Kubeflow Terraform Modules - run Jupyter in Kubernetes 🪐](https://github.com/treebeardtech/terraform-helm-kubeflow): Kubeflow Terraform Modules - run Jupyter in Kubernetes 🪐 - treebeardtech/terraform-helm-kubeflow

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1213409058300891136) (82 messages🔥🔥): 

- **Function Calling In NodeJS**: `@jetset2000` was looking for documentation on using function calling with Mistral in NodeJS. `@sophiamyang` provided a helpful response with an [example](https://github.com/mistralai/client-js/blob/main/examples/function_calling.js) in the Mistral AI's JS client repository.
- **Mistral Medium Model Timeout Issues**: `@patrice_33841` reported timeouts when making requests to the mistral-medium-latest model. Other users seemed to have no trouble with the medium model, and `@mrdragonfox` provided contact information for support, suggesting to post in the tech support channel or email support directly.
- **Confusion on Prompt Documentation**: `@benjoyo` expressed confusion about the consistency between user and system messages and actual prompts in Mistral's documentation, which `@sophiamyang` acknowledged and promised clarity soon.
- **Response Format Clarifications Needed**: `@gbourdin` encountered issues with new JSON response formats, leading to a discussion about correct prompt settings which `@proffessorblue` clarified with instructions from the docs, resolving `@gbourdin`'s problem.
- **Exploring Sentiment Analysis Efficacy**: `@krangbae` shared experiences with using different Mistral models for sentiment analysis, noting that 8x7b seemed more effective than the small model.

**Links mentioned**:

- [Pricing and rate limits | Mistral AI Large Language Models](https://docs.mistral.ai/platform/pricing/): Pay-as-you-go
- [Model Selection | Mistral AI Large Language Models](https://docs.mistral.ai/guides/model-selection/): Mistral AI provides five API endpoints featuring five leading Large Language Models:
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/#json-mode): We provide client codes in both Python and Javascript.
- [Function Calling | Mistral AI Large Language Models](https://docs.mistral.ai/guides/function-calling/): Function calling allows Mistral models to connect to external tools. By integrating Mistral models with external tools such as user defined functions or APIs, users can easily build applications cater...
- [client-js/examples/function_calling.js at main · mistralai/client-js](https://github.com/mistralai/client-js/blob/main/examples/function_calling.js): JS Client library for Mistral AI platform. Contribute to mistralai/client-js development by creating an account on GitHub.
- [Function Calling | Mistral AI Large Language Models](https://docs.mistral.ai/guides/function-calling/.): Function calling allows Mistral models to connect to external tools. By integrating Mistral models with external tools such as user defined functions or APIs, users can easily build applications cater...
- [Mistral AI API | Mistral AI Large Language Models](https://docs.mistral.ai/api/): Chat Completion and Embeddings APIs

  

---


### Mistral ▷ #[le-chat](https://discord.com/channels/1144547040454508606/1211692704363323453/1213408985500352573) (126 messages🔥🔥): 

- **Mistral Large Style Praised**: `@foxalabs_32486` expressed appreciation for Mistral Large's more natural and less stuffy writing style, while maintaining depth similar to GPT-4.
- **User Interface Quirks**: `@steelpotato1` reported an issue with the user interface where prompts and responses jump positions during the generation process, creating a disorienting user experience.
- **Rate Limit Woes and Workarounds**: Users like `@shanman6991` and `@tom_lrd` encountered rate limits when using the chat API, leading to discussions about usage limits and suggestions to contact support for adjustments.
- **Hallucination and Misinformation Concerns**: `@godefv` pointed out that Le Chat sometimes provides incorrect information or generates content based on hallucinations rather than actual knowledge, like claiming details about a non-existent PhD thesis.
- **API Usage Puzzle**: `@sim3239` struggled with differences in API and Le Chat responses, inquiring about parameters used by Le Chat to replicate its complete responses in their own Python application.

**Links mentioned**:

- [LLM Tokenizer](https://www.danieldemmel.me/tokenizer.html): no description found
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/): We provide client codes in both Python and Javascript.

  

---


### Mistral ▷ #[failed-prompts](https://discord.com/channels/1144547040454508606/1212715819159654451/1213499316518260777) (13 messages🔥): 

- **Mistral Model Math Mishap**: `@propheticus_05547` discovered that **Mistral Instruct 7B v0.2 Q4_K_M** incorrectly calculated `10+3-9+33` as 22 instead of the correct answer, 37, when run in [Jan](https://github.com/janhq/jan) with Vulkan acceleration, questioning the model's arithmetic capabilities.
- **Learning Curve for Running Models Locally**: Responding to `@_._pandora_._`'s explanation about LLMs' poor math skills, `@propheticus_05547` noted improvement when the prompts were limited to knowledge and language-based questions and shared success with a different version, **Q5_K_M**, that could handle simple math.
- **Mistral Model Resists System Prompts**: `@jakobdylanc` reported that **Mistral Large** resists following its system prompt more than the **Mistral Medium** model when prompted as a helpful Discord chatbot named Jakobson.
- **Differences with GPT-4 on API Exposure**: `@benjoyo` observed that **Mistral Large** on the API tends to reveal its functional capabilities more readily than GPT-4, which generally doesn't expose such technical details to the user.
- **Not All Mistral Behavior Is Predictable**: In response to the behavior seen in **Mistral Large**, `@mrdragonfox` cautioned against assuming the bot's responses are always meaningful, suggesting some could be mere hallucinations.
  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1213984969504854047) (5 messages): 

- **Phi-2 Token Limit Confusion**: `@faldore` questioned the possibility of using **Phi-2** with more than 2k tokens. They pointed to [Hugging Face's Phi-2 model summary](https://huggingface.co/microsoft/phi-2) which indicates a 2k token limit.
- **Direct Link to Phi-2 Configuration**: In the follow-up, `@faldore` provided a [direct link](https://huggingface.co/microsoft/phi-2/blob/main/config.json#L19) to the Phi-2 configuration file which shows the `"max_position_embeddings": 2048` setting.
- **Explanation on Phi-2 Token Extension**: `@vatsadev` responded to the question about extending Phi-2's tokens by indicating it would act like a default transformer, implying standard transformer behavior beyond configured limits.
- **Caution on Extending Phi-2's Capabilities**: In another message, `@vatsadev` warned that deviating from Phi-2's configured settings could either block the model or cause erratic performance.

**Links mentioned**:

- [config.json · microsoft/phi-2 at main](https://huggingface.co/microsoft/phi-2/blob/main/config.json#L19): no description found
- [microsoft/phi-2 · Hugging Face](https://huggingface.co/microsoft/phi-2): no description found

  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1213548816205353001) (31 messages🔥): 

- **Mac Set-Up Suggestions Galore**: `@gabriel_syme` is looking for Mac application suggestions. Numerous users like `@max_paperclips` and `@deki04` proposed essentials such as Homebrew, Parallels for ARM-based Macs, temperature monitoring with TG Pro, and using Time Machine for backups, with `@deki04` also sharing helpful [Mac setup tips for Python/ML](https://www.youtube.com/watch?v=mmkDyV59nRo&t=1368s) from a YouTuber.
- **Better Touch Tool and More**: `@denovich` recommends Better Touch Tool for gesture control, setting up Samba share with Time Machine, and enjoying Windows 11 ARM under Parallels for those who need Windows on their Mac.
- **Handy Apps and Homebrew to the Rescue**: `@eas2535` highlights the utility of Homebrew, sharing a [link to the tool](https://brew.sh), and lists useful applications like Maccy, Hyperkey, Shortcuts, and more for an efficient Mac experience.
- **Human Genetic Diversity at its Peak**: `@teknium` shares a tweet by `@richardfuisz` that claims every possible mutation in the human genome now exists in at least 50 people. This led to a request for the related scientific paper by `.ben.com`, who couldn't access the Twitter thread.
- **Enthusiasm for tridao**: `@hexani` mentions that "tridao is so goated," to which `@teknium` responds with a cat emoji, suggesting agreement.

**Links mentioned**:

- [no title found](https://brew.sh)): no description found
- [Tweet from Richard Fuisz (@richardfuisz)](https://x.com/richardfuisz/status/1763591765620121990?s=46): Every mutation that could exist, does exist. That has only been true for ~200 years. Most beneficial variants just haven&#39;t had the time to become ubiquitous. But now, at least 50 people in the wor...
- [TG Pro](https://www.tunabellysoftware.com/tgpro/#): Maximize your Mac's performance with TG Pro. The ultimate solution for fan control and extensive temperature monitoring: CPU, GPU, SSD, and more.
- [Setting up new MacBook for software development](https://www.youtube.com/watch?v=mmkDyV59nRo&t=1368s): Here I go through setting up a new MacBook for software development, the way I usually set things up for my own tasks.▶️ Setting up a new M2 Mac Mini - https...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1213440419803963473) (49 messages🔥): 

- **Scaling and Efficiency Hot Debate**: `@ldj` argued that **compute optimality** breaks down after the 500B-1T parameter mark, with efficiency gains from MoE and training techniques rather than scale. They cited **Sam Altman** suggesting the era of scaling is over, with future gains from architectural innovations, as detailed in [this article](https://www.analyticsvidhya.com/blog/2023/04/the-end-of-the-giant-ai-models-era-openai-ceo-warns-scaling-era-is-over/) and a supporting **Medium** post [here](https://medium.datadriveninvestor.com/dear-sam-altman-there-was-never-an-era-of-making-models-bigger-288c5f2b743c ).
  
- **The Skepticism Around 100T Models**: `@intervitens` and `@.ben.com` were skeptical of the feasibility and practicality of training 100T parameter models, questioning both hardware capabilities and data availability. `@euclaise` countered, suggesting availability of sufficient data resources like Redpajama v2.

- **Potential of Smaller Models**: `@ldj` further emphasized that bigger models are not necessarily better, pointing out that GPT-4 might have better performance with around 200B active parameters compared to models with over 500B active parameters. `@teknium` disagreed, suggesting that **parameter scaling** could still be beneficial if combined with adequate training data.

- **Cost Concerns in AI Scaling**: `@ldj` raised a practical concern about the cost-effectiveness of scaling up models, alluding to the possibility that increasing the number of parameters could result in prohibitively high costs for both training and inference.

- **Reference to AI Model Recent Comparisons**: A link to a **Reddit post** featuring a comparison of 17 new models, adding up to 64 ranked, was shared by `@mautonomy`, without comments from others in the channel.

**Links mentioned**:

- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750): Efficiently serving large language models (LLMs) requires batching many requests together to reduce the cost per request. Yet, the key-value (KV) cache, which stores attention keys and values to avoid...
- [- Fuck You, Show Me The Prompt.](https://hamel.dev/blog/posts/prompt/): Quickly understand inscrutable LLM frameworks by intercepting API calls.
- [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516): Activation sparsity refers to the existence of considerable weakly-contributed elements among activation outputs. As a prevalent property of the models using the ReLU activation function, it has been ...
- [Technological Approach to Mind Everywhere | Michael Levin](https://www.youtube.com/watch?v=JC4FOzAuHF4): Extract from &quot;Evolution, Basal Cognition and Regenerative Medicine&quot;, kindly contributed by Michael Levin in SEMF&#39;s 2023 Interdisciplinary Summer School (http...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b5vp2e/llm_comparisontest_17_new_models_64_total_ranked/): no description found
- [Dear Sam Altman- There was never an era of making models bigger](https://medium.datadriveninvestor.com/dear-sam-altman-there-was-never-an-era-of-making-models-bigger-288c5f2b743c): LLMs have never been revolutionary or as game-changing as gurus online would have pushed you to believe
- [The End of the Giant AI Models Era: OpenAI CEO Warns Scaling Era Is Over](https://www.analyticsvidhya.com/blog/2023/04/the-end-of-the-giant-ai-models-era-openai-ceo-warns-scaling-era-is-over/): Learn what OpenAI&#039;s CEO Sam Altman has to say about future advances in AI models like ChatGPT and how access to GPUs remains crucial.

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1213447897216057364) (328 messages🔥🔥): 

- **AI for Music**: User `@audaciousd` looked forward to new music generative AI, with particular interest in the upcoming release from a company called stabilities. They inquired about others' knowledge on the topic.
- **Claude 3 Generates Buzz**: Discussion on Claude 3's release: `@fibleep` referenced an announcement, and users like `@4biddden` and `@mautonomy` speculated on its comparison to GPT-4 performance.
- **GPT-4 vs. Claude 3 Opinions**: Several users, including `@teknium`, shared and sought feedback through [Twitter polls](https://x.com/teknium1/status/1764732905660830024?s=46) on whether Claude 3 Opus is actually better than GPT-4.
- **B2B Sales Strategies Shared**: User `@mihai4256` asked for advice on selling B2B software products, leading `@hexani` to offer their experience on targeting small businesses and the challenges and strategies involved. Hexani emphasized the necessity of direct engagement and a high bar for product viability.
- **Knowledge Graph Building Resources Explored**: Users `@mihai4256` and `@everyoneisgross` discussed models and approaches for creating knowledge graphs, with `@max_paperclips` suggesting using Hermes for JSON structured triplet extraction. They shared that a new model with improved structured data extraction capabilities is forthcoming.


**Links mentioned**:

- [AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): no description found
- [Tweet from roon (@tszzl)](https://x.com/tszzl/status/1493816776731205643?s=46): yeah this is the correct take. GPT4 coming soon. 2trillies baby  ↘️ Quoting bayes (@bayeslord)   I feel like the burst of openai ppl talking about alignment and conscious language models is a hint tha...
- [Tweet from Teknium (e/λ) (@Teknium1)](https://x.com/teknium1/status/1764737777667891291?s=46): Is claude 3 opus better than gpt4?  New poll because last one was too ambigious and also because I had no show results  ██████ Yes  (18.8%) ███ No  (10.6%) ██████████████████████ Show results  (70.6%)...
- [Tweet from db (@dxyba)](https://x.com/dxyba/status/1763756934262321574?s=20))): ng position was eliminated, role was terminated  i now have 5 months to get a job -- no name state school, no prev experiences, no leetcode, no projects  time to work my ass off, to do the biggest 180...
- [wandb/gemma-2b-zephyr-dpo · Hugging Face](https://huggingface.co/wandb/gemma-2b-zephyr-dpo): no description found
- [Tweet from Yam Peleg (@Yampeleg)](https://fxtwitter.com/Yampeleg/status/1763532342482612625?s=20): Trained from a merge of Mistral-7B&#39;s on a synthetic dataset made with GPT-4 Turbo.  It is continued pretraining but throughout the samples there are some instructions (but they are not aligned to ...
- [Tweet from virat (@virattt)](https://x.com/virattt/status/1764363199049072743?s=46): I am blown away by RAGAS  With 10 lines of code, I created a question + answer dataset of Airbnb&#39;s latest annual report (10-K).  The dataset has 3 parts: • questions • contexts • ground truth answ...
- [Tweet from John Nay (@johnjnay)](https://x.com/johnjnay/status/1764470331568238618?s=46): LLM Prediction Capabilities Match Human Accuracy  -A crowd of 12 LLMs vs a crowd of 925 human forecasters on a 3-month forecasting tournament -LLM crowd is statistically equivalent to the human crowd ...
- [Tweet from Tsarathustra (@tsarnick)](https://x.com/tsarnick/status/1763756693610184811?s=46): Nick Chater: AI language models cannot create new knowledge because they are just reflecting back what we already know
- [GPT4All Documentation](https://docs.gpt4all.io/gpt4all_python_embedding.html"): no description found
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/AnthropicAI/status/1764653830468428150?s=20): Today, we&#39;re announcing Claude 3, our next generation of AI models.   The three state-of-the-art models—Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku—set new industry benchmarks across reason...
- [Tweet from Teknium (e/λ) (@Teknium1)](https://x.com/teknium1/status/1764732905660830024?s=46): So is it actually better than gpt 4?  ████████████████ Yes  (52%) ███████████████ No  (48%)  960 votes · 21 hours left
- [Google Colaboratory](https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_tacred.ipynb): no description found
- [studio-ousia/luke-large-finetuned-tacred · Hugging Face](https://huggingface.co/studio-ousia/luke-large-finetuned-tacred): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/16lnvv1/is_it_legal_to_use_gpt4_output_to_finetune_llama2/): no description found
- [Tweet from Bin Lin (@LinBin46984)](https://x.com/LinBin46984/status/1763476690385424554?s=20): 👏👏👏We are thrilled to launch a project called Open-Sora plan, aiming to reproduce OpenAI&#39;s ( &#34;CloseAI&#34;🤪) Sora. This project supports now🎉🎉🎉: (1) 🚀Variable Aspect Ratios (2) ✈️Varia...
- [Tweet from Blaze (Balázs Galambosi) (@gblazex)](https://x.com/gblazex/status/1764664048522600690?s=20): Claude 3 Opus (output) is very expensive  It does have solid reasoning scores, so we&#39;ll see how much it&#39;ll worth the extra cost.  But GPT-4 Turbo remains the most cost-efficient high-end solut...
- [euclaise (Jade)](http://hf.co/euclaise): no description found
- [GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP](https://github.com/AnswerDotAI/fsdp_qlora): Training LLMs with QLoRA + FSDP. Contribute to AnswerDotAI/fsdp_qlora development by creating an account on GitHub.
- [llama : add T5 (encoder-decoder) support · Issue #5763 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5763): Still not familiar with the details, but it seems it would be useful to support this architecture in llama.cpp. First, need to decide on the API and see what changes would be necessary See discussi...
- [laion/OIG · Datasets at Hugging Face](https://huggingface.co/datasets/laion/OIG): no description found
- [google-coral](https://github.com/google-coral): Open source projects for coral.ai. google-coral has 37 repositories available. Follow their code on GitHub.
- [Aligning LLMs with Direct Preference Optimization](https://youtu.be/QXVCqtAZAn4?t=1512): In this workshop, Lewis Tunstall and Edward Beeching from Hugging Face will discuss a powerful alignment technique called Direct Preference Optimisation (DPO...
- [GitHub - parthsarthi03/raptor: The official implementation of RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://github.com/parthsarthi03/raptor): The official implementation of RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval - parthsarthi03/raptor
- [LUKE](https://huggingface.co/docs/transformers/model_doc/luke): no description found

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1213437854038884352) (32 messages🔥): 

- **PPO Script Inquiry**: `@xela_akwa` in search for a PyTorch or PyTorch Lightning script for PPO on LLMs, finds the hf trl (HuggingFace Transformers Reinforcement Learning) limited. A conversation ensued with potential alternatives being suggested, including a GitHub repository by `@.mahouko` for related work. No definitive solution for PPO was provided.
- **Function Calling Model Server Showdown**: `@giulio123456` questions about the best inference platform for fastest function calling models. Replies by `@sundar_99385` and `@dustinwcarr` suggested that Anyscale and Deepinfra are some of the platforms supporting Mistral/Mixtral with notable performance, but no direct latency comparisons were provided.
- **Format for 1-Shot in ChatML**: `@cognitivetech` queries about the correct templating for 1-shot training using ChatML with a focus on system-user interactions; `@teknium` confirms the correct format excludes the 'name=' convention and endorses a simpler template.
- **LLaMa Architecture Clarification**: `@qtnx` asks about the specifics of patch embedding conversion in LLaMa 1 and 1.5 architectures, receiving a brief acknowledgement without specific details given from `@teknium` and a follow-up query by `@qnguyen3`.
- **AI-Assisted Chat Considerations**: `@betim01` discusses strategies for fine-tuning an AI model for customer interactions, considering Nous Hermes and RAG. `@teknium` warned against potential downsides with an example of ChatGPT being tricked in a dealership context, recommending a more reliable RAG approach and listing potential inference platform options.
- **Next Steps for Language Models**: `@pier1337` speculates on the future of language models, mentioning Sutskever's views on object-driven AI and the potential application within simulated environments, but there were no direct responses to this prediction within the chat log provided.

**Links mentioned**:

- [Tweet from Chris Bakke (@ChrisJBakke)](https://x.com/chrisjbakke/status/1736533308849443121?s=46): I just bought a 2024 Chevy Tahoe for $1.
- [A car dealership added an AI chatbot to its site. Then all hell broke loose. ](https://www.businessinsider.com/car-dealership-chevrolet-chatbot-chatgpt-pranks-chevy-2023-12): Pranksters figured out they could use the ChatGPT-powered bot on a local Chevrolet dealer site to do more than just talk about cars.

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1214171242362306580) (2 messages): 

- **Exploring Moondream**: User `@ee.dd` shared their positive experience with **Moondream**, highlighting its speed and effectiveness after some testing. They provided the GitHub link: [Moondream - tiny vision language model](https://github.com/vikhyat/moondream).

**Links mentioned**:

[GitHub - vikhyat/moondream: tiny vision language model](https://github.com/vikhyat/moondream): tiny vision language model. Contribute to vikhyat/moondream development by creating an account on GitHub.

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1213547327097274369) (197 messages🔥🔥): 

- **AI Alignment in Open-Source**: The Open Source Initiative (OSI) will be releasing a new draft of the open-source AI definition monthly, targeting a 1.0 release by the end of October 2024, with discussions in their public forum and [draft documents available for review](https://opensource.org/deepdive/drafts).

- **Legal Battle Against the DMCA**: The EFF has filed a lawsuit, _Green v. Department of Justice_, challenging the anti-circumvention and anti-trafficking provisions of the DMCA for restricting access to purchased copyrighted materials. [Full case details](https://www.eff.org/cases/green-v-us-department-justice).

- **Quantization Debate in Neural Networks**: A discussion emerged about the practicality and implications of quantization in neural network weights and activations. Users debated over papers like the [bitlinear paper](https://arxiv.org/pdf/2310.11453.pdf) and the notion of quantizing activation functions, invoking concepts such as epistemic uncertainty.

- **GitHub Malware Spread Campaign**: A malware distribution campaign on GitHub has resulted in cloning legitimate repositories, injecting malware, and promoting compromised code. [Apiiro's security analysis](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/) explains the threat in detail.

- **Discussions on Predictive Modeling Limitations**: User `@rallio.` asserts there's no capability to de novo create economically viable biomolecules through predictive modeling, arguing the complexities of biological systems make them unpredictable, unlike physical models used for engineering.

**Links mentioned**:

- [Green v. U.S. Department of Justice](https://www.eff.org/cases/green-v-us-department-justice): Green v. Department of Justice is an EFF lawsuit challenging the constitutionality of the Digital Millennium Copyright Act’s anti-circumvention and anti-trafficking provisions on First Amendment groun...
- [GitHub struggles to keep up with automated malicious forks](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/): Cloned then compromised, bad repos are forked faster than they can be removed
- [Turing jump - Wikipedia](https://en.wikipedia.org/wiki/Turing_jump): no description found
- [Scientists aghast at bizarre AI rat with huge genitals in peer-reviewed article](https://arstechnica.com/science/2024/02/scientists-aghast-at-bizarre-ai-rat-with-huge-genitals-in-peer-reviewed-article/): It&#39;s unclear how such egregiously bad images made it through peer-review.
- [jax/docs/multi_process.md at main · google/jax](https://github.com/google/jax/blob/main/docs/multi_process.md): Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - google/jax
- [Ergonomic way to extract a single iteration output from a scan · google/jax · Discussion #20054](https://github.com/google/jax/discussions/20054): It&#39;s common to extract the activations for a single hidden layer in a network, but this gets annoying when using a scan over parameters. Here&#39;s a toy example: import jax import jax.numpy as jn...
- [GitHub - davisyoshida/gemma-flax: Implementation of Gemma in Jax/Flax](https://github.com/davisyoshida/gemma-flax): Implementation of Gemma in Jax/Flax. Contribute to davisyoshida/gemma-flax development by creating an account on GitHub.
- [Attention entire world!!!](https://professorhooh.substack.com/p/attention-entire-world-f1f): I challenge you to The Game
- [google/jax · Discussions](https://github.com/google/jax/discussions): Explore the GitHub Discussions forum for google jax. Discuss code, ask questions &amp; collaborate with the developer community.
- [All-atom Molecular Dynamics Simulation of the Bacterial Cytoplasm](https://www.youtube.com/watch?v=5JcFgj2gHx8),): How biomolecules behave in crowded cellular environments has been an important question in life science. Researchers at RIKEN and Michigan State University m...
- [Drafts of the Open Source AI Definition](https://opensource.org/deepdive/drafts): The drafts of the Open Source AI Definition. We&#8217;re publishing the draft documents as they&#8217;re released. Check the individual drafts below for instructions on how to leave your comments. …

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1213413741757202432) (115 messages🔥🔥): 

- **Counterfactual Examples Sharpen AI's Visio-Linguistic Reasoning**: `@digthatdata` shared a new approach called **CounterCurate**, detailed in a [research paper](https://countercurate.github.io/), which improves visio-linguistic compositional reasoning in multimodal models. CounterCurate employs **GPT-4V** and **DALLE-3** to create counterfactual image-caption pairs, achieving higher performance on benchmarks such as SugarCrepe.

- **Functional Benchmarks Challenge LLMs**: `@.the_alt_man` pointed to a [Twitter thread](https://x.com/_saurabh/status/1763626711407816930?s=20) by `@_saurabh` suggesting that **over 50% of the reported reasoning abilities of LLMs might not be true reasoning**. The thread discussed a paper introducing functional benchmarks, revealing significant reasoning gaps in state-of-the-art models, with an associated [arXiv draft](http://arxiv.org/abs/2402.19450) and [GitHub repository](https://github.com/ConsequentAI/fneval).

- **Contrastive Learning for Unanswerable Questions in SQuADv2**: `@paganpegasus` queried the best approach for creating negative samples in contrastive learning for unanswerable questions in SQuADv2, suggesting using Spacy to extract noun chunks as potential negatives. `@fern.bear` proposed using sets of answers with maximum confidence that are exclusive to SQuADv2, labeled by a model, as another evil method.

- **Concerns Over RLHF Impact on Models' Capabilities**: Discussion by `@.the_alt_man` and `@canadagoose1` revolved around the impact of Reinforcement Learning from Human Feedback (RLHF) on models' abilities, with a suspicion that RLHF might be degrading performance due to poor implementation.

- **Terminator Architecture: Potential Game-Changer for AI?**: `@fredholm` highlighted the **Terminator** network described in an [arXiv paper](https://arxiv.org/abs/2401.17948), which posits a new architecture potentially replacing residual learning with large implicit kernels for full context interaction. In the ensuing conversation, `@harvie_zhang_32234` and `@alex_cool6` confirmed the unique approach of Terminator, with the latter stating plans to apply it to image generation and release the code in the future.

**Links mentioned**:

- [Tweet from Saurabh Srivastava (@_saurabh)](https://x.com/_saurabh/status/1763626711407816930?s=20): More than 50% of the reported reasoning abilities of LLMs might not be true reasoning.  How do we evaluate models trained on the entire internet? I.e., what novel questions can we ask of something tha...
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1764653830468428150?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): Today, we&#39;re announcing Claude 3, our next generation of AI models.   The three state-of-the-art models—Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku—set new industry benchmarks across reason...
- [AtP*: An efficient and scalable method for localizing LLM behaviour to components](https://arxiv.org/abs/2403.00745): Activation Patching is a method of directly computing causal attributions of behavior to model components. However, applying it exhaustively requires a sweep with cost scaling linearly in the number o...
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750): Efficiently serving large language models (LLMs) requires batching many requests together to reduce the cost per request. Yet, the key-value (KV) cache, which stores attention keys and values to avoid...
- [Domain-Specific Tensor Languages](https://arxiv.org/abs/2312.02664): The tensor notation used in several areas of mathematics is a useful one, but it is not widely available to the functional programming community. In a practical sense, the (embedded) domain-specific l...
- [Beyond Language Models: Byte Models are Digital World Simulators](https://arxiv.org/abs/2402.19155): Traditional deep learning often overlooks bytes, the basic units of the digital world, where all forms of information and operations are encoded and manipulated in binary format. Inspired by the succe...
- [CounterCurate](https://countercurate.github.io/): no description found
- [Sparse is Enough in Scaling Transformers](https://arxiv.org/abs/2111.12763): Large Transformer models yield impressive results on many tasks, but are expensive to train, or even fine-tune, and so slow at decoding that their use and study becomes out of reach. We address this p...
- [Mega: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655): The design choices in the Transformer attention mechanism, including weak inductive bias and quadratic computational complexity, have limited its application for modeling long sequences. In this paper...
- [HyperZ$\cdot$Z$\cdot$W Operator Connects Slow-Fast Networks for Full Context Interaction](https://arxiv.org/abs/2401.17948): The self-attention mechanism utilizes large implicit weight matrices, programmed through dot product-based activations with very few trainable parameters, to enable long sequence modeling. In this pap...
- [How to Scale Your EMA](https://arxiv.org/abs/2307.13813): Preserving training dynamics across batch sizes is an important tool for practical machine learning as it enables the trade-off between batch size and wall-clock time. This trade-off is typically enab...
- [Maracas Jimcarrey GIF - Maracas Jimcarrey Jim - Discover &amp; Share GIFs](https://tenor.com/view/maracas-jimcarrey-jim-celebrate-dance-gif-5055069): Click to view the GIF
- [How to Scale Hyperparameters as Batch Size Increases](https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/): no description found
- [APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding](https://arxiv.org/abs/2401.06761): The massive adoption of large language models (LLMs) demands efficient deployment strategies. However, the auto-regressive decoding process, which is fundamental to how most LLMs generate text, poses ...
- [Detecting anomalous proteins using deep representations](https://doi.org/10.1093/nargab/lqae021): Abstract. Many advances in biomedicine can be attributed to identifying unusual proteins and genes. Many of these proteins’ unique properties were discovered by

  

---


### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1214275379494395945) (1 messages): 

- **Creative Use of Figma for Animation**: User `@kyo_takano` described their process of creating an animation: they crafted a **template SVG in Figma**, manipulated it to compose various frames, and then used **imageio** to blend these into a **GIF animation**.
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1213428209148760064) (50 messages🔥): 

- **Mamba vs Transformers on Learning Parity**: `@dashiell_s` reported that a two-layer Mamba model can learn parity for sequences up to length 128, but doesn't generalize well to longer sequences. Their tests showed Mamba performed much better than a similarly configured transformer, which struggled with sequences longer than 64.
  
- **Skeptical of Associative Architecture's Efficiency**: `@norabelrose` expressed skepticism that architectures based on associative recurrence relations can efficiently learn PARITY, suggesting a need for experimentation to compare LSTM and Mamba performance.

- **Possible Misunderstanding of Sensitivity in ML Literature**: `@stellaathena` pointed out that a paper discussing "sensitivity" actually refers to average sensitivity rather than maximum sensitivity, which could imply different theoretical implications.

- **Trained Mamba on PARITY**: `@dashiell_s` shared that they conducted an experiment with Mamba on the PARITY problem, with results and code available on GitHub ([train_mamba.py](https://github.com/dashstander/automatic-circuits/blob/main/train_mamba.py)).

- **Debating the Mechanisms of Learning PARITY**: Various discussion points emerged around whether a lookup table or actual computation of PARITY was being learned by models (`@norabelrose` and `@dashiell_s`). There was also curiosity about whether deeper transformers could find more sophisticated solutions.

**Links mentioned**:

- [jax.lax.associative_scan &#8212; JAX  documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html): no description found
- [Prefix sum - Wikipedia](https://en.wikipedia.org/wiki/Prefix_sum#Parallel_algorithms): no description found
- [automatic-circuits/train_mamba.py at main · dashstander/automatic-circuits](https://github.com/dashstander/automatic-circuits/blob/main/train_mamba.py): Contribute to dashstander/automatic-circuits development by creating an account on GitHub.

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1213591513821610024) (71 messages🔥🔥): 

- **AzureML Woes for `lm-eval-harness`**: `@synthetic_johnny` encountered problems setting up `lm-eval-harness` on an AzureML compute cluster, experiencing dependency and CUDA device detection issues. A discussion unfolded around finding the right environment build, with `@hailey_schoelkopf` guiding on the specifics of using the tool, including details on multi-GPU use and model compatibility with AzureML.

- **Multi-Machine Parallelism Challenges**: `@hailey_schoelkopf` clarified that `lm-eval-harness` does not support multi-machine parallelism, which was causing issues for `@synthetic_johnny`. A workaround was suggested by `@rand0mm` who shared [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), which can help orchestrate the execution of `lm-eval-harness` across various nodes.

- **Handling Large Models on Single Nodes**: `@synthetic_johnny` was advised by `@hailey_schoelkopf` on evaluating large language models like GPT-J-6B by using `model_args parallelize=True` and `dtype=bfloat16` to spread the model across multiple GPUs in one node, and to start with a batch size of 1 to avoid out-of-memory errors. Discussion touched on the importance of model-parallel over data-parallel configurations when using AzureML.

- **Confusion over LAMBADA Training Data Usage**: `@smerkyg` posed a query regarding the proper use of the LAMBADA dataset for training LLMs. `@hailey_schoelkopf` clarified that it's better not to finetune on the LAMBADA training set as the benchmark is now intended to evaluate general-purpose language modeling abilities.

- **Seeking Example for Multi-GPU HELLASWAG on PYTHON**: `@antonvls` inquired about examples or success stories for running HELLASWAG evaluation on multiple GPUs using Python. `@stellaathena` directed them to the library's automated multi-GPU handling and provided a [GitHub link](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#multi-gpu-evaluation-with-hugging-face-accelerate) for further guidance.

**Links mentioned**:

- [Ray Serve: Scalable and Programmable Serving &#8212; Ray 2.9.3](https://docs.ray.io/en/latest/serve/index.html): no description found
- [lm-evaluation-harness/docs/interface.md at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#multi-gpu-evaluation-with-hugging-face-accelerate>)): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/) (1 messages): 

besiktas: havent really seen anything and have wondered/experimented this as well
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1213722504741396480) (2 messages): 

- **Processing Scripts for The Pile**: User `@catboy_slim_` shared a [GitHub link](https://github.com/EleutherAI/The-Pile/tree/master/processing_scripts) that could be helpful, particularly the README file. The repository contains scripts related to the development of **The Pile**, a large-scale dataset for training language models.
- **Inquiry on Validation Data Path**: User `@pietrolesci` queried about the **validation data file** mentioned in the wandb logs for run `v2 1.4B deduped_1dhzgs7f`. They are seeking to comprehend if the file is a random sample from the deduplicated pile.

**Links mentioned**:

[the-pile/processing_scripts at master · EleutherAI/the-pile](https://github.com/EleutherAI/The-Pile/tree/master/processing_scripts): Contribute to EleutherAI/the-pile development by creating an account on GitHub.

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1213428130828521502) (155 messages🔥🔥): 

- **Model Troubleshooting in LM Studio**: `@helloxan.` encountered an issue with Codellama Python 7B model in LM Studio and sought help on making the bot respond. Assistance was provided by `@heyitsyorkie`, who suggested using a different model from Hugging Face ([Magicoder-S-DS-6.7B-GGUF](https://huggingface.co/itsdotscience/Magicoder-S-DS-6.7B-GGUF)), and provided guidance on resolving a "broken quant" issue.
- **Questions on Model Support and Features**: Users like `@ciphersson`, `@justmarky`, and `@archi_95` asked about loading specific models such as LoRAs, QLoRA, and starCoder 2 in LM Studio, as well as uploading pdf files. `@heyitsyorkie` clarified that features such as QLoRA and starCoder2 support are not yet available, and uploading pdfs directly is not possible.
- **Technical Difficulties with LM Studio Discussed**: Several users like `@sourguava`, `@shadowdoggie`, and `@boting_0215` experienced technical issues ranging from models taking a long time to load to encountering errors with no clarification provided.
- **Model Presets and Parameters Explored**: Users were seeking and sharing information on obtaining more presets (`@techfren` shared a [YouTube video resource](https://youtu.be/LUiVbOeLeas?si=y96mVuTitAFjX_Zq)), understanding parameters for model quantization (`@unkown101`), and the effects of changing randomness settings for code generation (`@drawless111`).
- **GPU Requirements and Capabilities for LLM**: Various users, including `@ethanboyle`, `@broski_1337`, and `@ocn` touched on the required hardware specifications, like GPU offloading and the necessity of a powerful GPU for efficient model utilization. `@heyitsyorkie` advised that a GPU with at least 24GB of VRAM is necessary for speed and efficiency when running large language models.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/careers): Find, download, and experiment with local LLMs
- [no title found](https://www.marktechpost.com/2024/03/03/meet-phind-70b-an-artificial-intelligence-ai-model-that-closes-execution-speed-and-the-code-generation-quality-gap-with-gpt-4-turbo/): no description found
- [itsdotscience/Magicoder-S-DS-6.7B-GGUF · Hugging Face](https://huggingface.co/itsdotscience/Magicoder-S-DS-6.7B-GGUF): no description found
- [ItsD (D)](https://huggingface.co/itsd): no description found
- [AI solves huge problem holding back fusion power](https://flip.it/Go0HL0): Princeton researchers have trained an AI to predict and prevent a common problem during nuclear fusion reactions.
- [Introducing the next generation of Claude](https://www.anthropic.com/news/claude-3-family): Today, we&#x27;re announcing the Claude 3 model family, which sets new industry benchmarks across a wide range of cognitive tasks. The family includes three state-of-the-art models in ascending order ...
- [LM Studio Models not behaving? Try this!](https://youtu.be/LUiVbOeLeas?si=y96mVuTitAFjX_Zq): The repository for free presets:https://github.com/aj47/lm-studio-presets➤ Twitter - https://twitter.com/techfrenaj➤ Twitch  - https://www.twitch.tv/techfren...
- [afrideva/TinyLlama-con-creative-writing-v0.2-GGUF · Hugging Face](https://huggingface.co/afrideva/TinyLlama-con-creative-writing-v0.2-GGUF): no description found
- [GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.](https://github.com/oobabooga/text-generation-webui): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1213430946599018556) (49 messages🔥): 

- **Concerns about Model Leaking Personal Data**: User `@tobitege` shared an unexpected and irrelevant response obtained from a model, raising concerns about data privacy. `@tay2win` speculated that this could be a case of data scraping, like from LinkedIn or GitHub, which they felt should be illegal.
- **Hugging Face Model Sources Questioned**: `@tobitege` shared their unease after finding a real person matching the name given in an irrelevant model response. This led to a discussion about the sources of training data, with `@tay2win` hoping that emails and chat logs are not used for training AIs.
- **Misunderstandings of Model Overfitting Addressed**: `@aswarp` clarified the concept of "regurgitating" data by AI models when `@tay2win` suggested overfitting could be the cause. `@aswarp` indicated the issue is known and occurs when models repeat bits of the training data.
- **Confusion Over Using Grok with LM Studio**: In a conversation about integrating Grok with LM Studio, `@pandora_box_open` provided a link to Groq.com, but clarifications on what was intended led to mixed responses and corrections from `@wildcat_aurora` and `@jedd1`.
- **Seeking the Right Model Fit for VRAM and Context Size**: `@jason_2065` exchanged information with `@heyitsyorkie` and `@vinni_spx` about various models suitable for coding, their VRAM usage, context length, and the need for filters on Hugging Face. `@jason_2065` also inquired about "mixture of experts" models, citing good speed and VRAM fit with Laser Dolphin Mixtral.

**Links mentioned**:

- [GroqChat](https://groq.com/): no description found
- [Mixture of Experts Explained](https://huggingface.co/blog/moe): no description found
- [TheBloke/laser-dolphin-mixtral-2x7b-dpo-GGUF · Hugging Face](https://huggingface.co/TheBloke/laser-dolphin-mixtral-2x7b-dpo-GGUF): no description found

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1213560308120158229) (5 messages): 

- **Seeking Non-API CURL Guidance**: `@newoperator` inquired about making a curl request directly to an LM Studio model without using the OpenAI completion API, noting a lack of documentation. `@fabguy` responded that CURL can directly interact with the LM Studio server without an additional API.
- **Error Code Conundrum in LM Studio**: `@instamailing` posted an issue with LM Studio characterized by an exit error code (-1073740791) and accompanying JSON data revealing RAM and VRAM details, but no definitive cause.
- **Navigating to the Right Support Channel**: `@heyitsyorkie` directed `@instamailing` to the appropriate support channel for the issue they're facing and advised including more information than just the error message to get better assistance.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1213542955915419808) (114 messages🔥🔥): 

- **16GB VRAM and Debating Mac Pros**: `@ethanboyle` noted the upper end of VRAM in consumer video cards seems to be 16GB. Following up on suggestions for MacBook Pros with Apple Silicon, concerns about non-upgradability of RAM and installing Linux on Macs were discussed, including potential issues highlighted at [Debian on Apple M1](https://wiki.debian.org/InstallingDebianOn/Apple/M1) and [Debian ARM port](https://www.debian.org/ports/arm/).
- **Apple's Unified Memory Architecture in the Hot Seat**: Users debated the upgradability and performance aspects of Apple's M-series chips with unified memory (`@heyitsyorkie`, `@nink1`, `@wyrath`). The architecture, which lacks user-upgradable memory, was contrasted against potential future AMD APUs and CAMM memory modules.
- **Potential Challenges Running LM Studio on Integrated GPUs**: `@ayyouboss` faced issues running LLMs on an integrated VEGA GPU in a Ryzen rig with 16GB of RAM, despite LM Studio running on the CPU. `@rewire` suggested a VRAM limitation might be at play, and after back-and-forth troubleshooting, they proposed trying out Windows instead of Linux due to probable driver issues.
- **Evaluating Silicon Macs for Linux and AI Use**: `@ethanboyle` and others discussed the use of Macs with Apple silicon for AI work, bearing in mind the challenges of Linux installation and non-upgradable unified memory. Some community knowledge and external links such as [Tart virtualization for Apple Silicon](https://tart.run) were shared, which `@wolfspyre` reported as a powerful and free tool for running Linux in containers on Mac.
- **A Costly Affair with Groq Chips**: In a hardware performance and cost comparison, `@nink1` and `@wyrath` discussed how the Groq chip architecture necessitates clustering many chips for high performance, resulting in a significant cost gap compared to Nvidia solutions. An investment in a Groq cluster to run large models could potentially reach millions of dollars.

**Links mentioned**:

- [Debian -- ARM Ports ](https://www.debian.org/ports/arm/): no description found
- [Mac computers with Apple silicon - Apple Support](https://support.apple.com/en-us/116943): Starting with certain models introduced in late 2020, Apple began the transition from Intel processors to Apple silicon in Mac computers.
- [Apple Macbook Pro M1 Max 16&quot; 2021 10-Core CPU 32 GPU 1TB SSD 32GB Ram Gray 194252546833 | eBay](https://www.ebay.com/itm/225868883217): no description found
- [InstallingDebianOn/Apple/M1 - Debian Wiki](https://wiki.debian.org/InstallingDebianOn/Apple/M1): no description found

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1213428496811163678) (2 messages): 

- **Starcoder2-15b Anticipation**: User `@.bambalejo` inquired about when StarCoder2-15b would be available to try in LM Studio, referencing a GitHub pull request that adds support for it to llama.cpp at [https://github.com/ggerganov/llama.cpp/pull/5795](https://github.com/ggerganov/llama.cpp/pull/5795).
- **LM Studio Update Pending for StarCoder2 Integration**: `@heyitsyorkie` responded that the integration of StarCoder2-15b into LM Studio will likely occur with the next beta release, once LM Studio is updated to the version of llama.cpp that supports this model.

**Links mentioned**:

- [Request: add support for Cerebras GPT models just released · ggerganov/llama.cpp · Discussion #579](https://github.com/ggerganov/llama.cpp/pull/579): The announcement is here: https://www.cerebras.net/press-release/cerebras-systems-releases-seven-new-gpt-models-trained-on-cs-2-wafer-scale-systems The models are available here: https://huggingfac...
- [Add support for StarCoder2 by pacman100 · Pull Request #5795 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5795): What does this PR do?  Adds support for StarCoder 2 models that were released recently.

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1214041526532050984) (4 messages): 

- **Autogen Integration Troubles**: User `@sourguava` experienced connection errors when testing the model, specifically highlighting a **401 error** indicating an "Incorrect API key." They referenced their API key struggles and provided the link to find the correct key at [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).
- **Reinstalling Autogen Might Help**: In response to `@sourguava`, `@thebest6337` suggested **reinstalling autogen** as a possible fix to the connection issues.
- **LM Studio May Be Experiencing Delays**: `@sourguava` mentioned that LM Studio has been problematic with models loading very slowly, hinting at potential performance issues with the platform.
- **Docker Volume Mounting Error**: `@remliv` attempted to follow AutoGen's Docker installation guide but encountered an error indicating **invalid characters** for a local volume name when using the command `docker run`.
- **Windows Path Challenge with Docker**: `@remliv` found a possible solution for the volume mounting error on StackOverflow, where it was suggested to replace `$(pwd)` with `%cd%` for Windows systems, but this led to another error stating the file could not be opened because it was not found.

**Links mentioned**:

- [Docker | AutoGen](https://microsoft.github.io/autogen/docs/installation/Docker/#option-1-install-and-run-autogen-in-docker): Docker, an indispensable tool in modern software development, offers a compelling solution for AutoGen&#x27;s setup. Docker allows you to create consistent environments that are portable and isolated ...
- [Mount current directory as a volume in Docker on Windows 10](https://stackoverflow.com/questions/41485217/mount-current-directory-as-a-volume-in-docker-on-windows-10/41489151#41489151)): Description&#xA;&#xA;I am using Docker version 1.12.5 on Windows 10 via Hyper-V and want to use container executables as commands in the current path. I built a Docker image that is running fine, but ...

  

---


### LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/) (1 messages): 

triffed.: <@1211375065191682131> it exists i'm on arch i just used yay to get it
  

---


### LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/) (1 messages): 

.tntflo: Can we get this for linux too
  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1213737128849186848) (5 messages): 

- **JavaScript compatibility inquiry**: `noneofya_business` asked if **crew ai** works with JavaScript, but no additional details or responses were provided.
- **Visual Studio Code mentioned**: `tobitege` mentioned *Visual Studio Code (VSC)* presumably in response to an earlier query, but the context is unclear. `wolfspyre` echoed the mention of **Visual Studio Code** and emphasized the need for clarity.
- **Seeking clarity on confusing topics**: `wolfspyre` commented that navigating these topics can be quite confusing, highlighting a need for further explanation or guidance.
- **Exploring the construction of personalized AI agents**: `ccarroz` inquired about experiences in building custom AI agents (defining role and mission) outside the realm of pre-built examples and the feasibility of leveraging different Language Models (LLMs) on various local devices. They shared an ambitious plan to run diverse LLMs on different hardware, including a **3090 GPU**, a **Jetson Orion**, and a **6800XT GPU**.
  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1213408961244561418) (121 messages🔥🔥): 

- **Local Model Training Memory Woes**: `@chunkchampion` joked about whether local model training was advisable, considering it was using up 90 gigabytes of memory.
- **Gradio Version Issues Trouble Space Deployers**: Several users including `@ilovesass`, `@cubietom`, and `@vipitis` discuss issues with deploying spaces, suggesting the need to check for outdated Gradio versions and looking for updated components like `ImageEditor`.
- **Request for Guidance Ascending the AI Learning Curve**: `@gschwepp_84093` queried how to progress from introductory AI projects to more complex ones. User `@dailafing` expressed a desire for advice on the same subject, hoping an experienced member could provide insight.
- **Model Hunting for Specific Use Cases**: Users like `@pazanchick` and `@apaz` sought advice and suggestions for models suitable for tasks like TTS and generating book club questions, respectively.
- **Share and Seek Opportunities in the AI Community**: `@dsquared70` promoted a conference in Asheville, NC for developers working with GenAI in production, while `@jan_skaryna` searched for a senior AI/ML developer.

**Links mentioned**:

- [AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): no description found
- [Creausdemo - a Hugging Face Space by niggathug](https://huggingface.co/spaces/niggathug/creausdemo): no description found
- [LGM - a Hugging Face Space by ashawkey](https://huggingface.co/spaces/ashawkey/LGM): no description found
- [GitHub struggles to keep up with automated malicious forks](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/): Cloned then compromised, bad repos are forked faster than they can be removed
- [Fbi Fbiopenup GIF - Fbi Fbiopenup Carlwhitman - Discover &amp; Share GIFs](https://tenor.com/view/fbi-fbiopenup-carlwhitman-gif-19586039): Click to view the GIF
- [Marching cubes - Wikipedia](https://en.wikipedia.org/wiki/Marching_cubes): no description found
- [Gradio ImageEditor Docs](https://www.gradio.app/docs/imageeditor): no description found
- [Find the best open source model for your project with Sambaverse](https://sambaverse.sambanova.net/): no description found
- [SambaLingo Chat Space - a Hugging Face Space by sambanovasystems](https://huggingface.co/spaces/sambanovasystems/SambaLingo-chat-space): no description found
- [SambaLingo  - a sambanovasystems Collection](https://huggingface.co/collections/sambanovasystems/sambalingo-65e25770f2037c85ad35ca77): no description found
- [Open-source LLM Ecosystem at Hugging Face](https://youtu.be/e9gNEAlsOvU): How to find, shrink, adapt and deploy open-source large language models? Here&#39;s a 10 min walkthrough on all the tools in @huggingface 🤗 featuring transforme...

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1213471352464351263) (5 messages): 

- **Discovering Helix as a 'novice-friendly' editor**: `@ai_noob` shared their first-time experience with the **Helix editor** and pointed out the availability of a comprehensive tutorial using the command **`helix --tutor`**.
- **CUDA MODE YT series in the spotlight**: `@iakhil` is dedicating the weekend to exploring the **CUDA MODE YouTube** series for deeper understanding.
- **Styling Gradio with SASS**: `@targetdummy5623` is working on a project to swap out the default theming in **Gradio** by implementing styles with **SASS** instead of Python.
- **HuggingMod keeps the pace in check**: HuggingMod reminded a user (`<@500991911650394143>`) to slow down their message frequency to maintain the quality of the discussion. 🤗
- **PPO theory on the study table**: `@0enzi` mentioned delving into the theory behind **Proximal Policy Optimization (PPO)**, hinting at deepening their understanding of reinforcement learning.
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1213555785213149204) (7 messages): 

- **Exploring In-The-Stack by Bigcode**: User `tonic_1` shared a [link to In-The-Stack](https://huggingface.co/spaces/bigcode/in-the-stack), a space by Bigcode on HuggingFace, and inquired about others' experiences with it.
- **Brains, Not Computers**: `markplusai` posted an article from The Guardian discussing the intricacies of the human brain and linked to research about manipulating memories in mice, emphasizing that we are in the thick of a significant scientific journey to understand our brains. Here's the [thought-provoking article](https://www.theguardian.com/science/2020/feb/27/why-your-brain-is-not-a-computer-neuroscience-neural-networks-consciousness).
- **LLaMA with Super Saiyan Strength**: `pacozaa` found an informative Medium article about using few-shot prompts with LLaMA2 and improving its performance with assistance from Claude. The article also discusses using large language models (LLMs) to aid in creating macOS agents, which can be read in detail [here](https://medium.com/@sarinsuriyakoon/creating-macos-agent-part-2-applied-a-few-shot-prompts-to-llama2-33eea86ac366).
- **Synching Lips with Pika**: `jacob_f97` shared a YouTube video titled "Introducing Lip Sync on Pika," which reveals a new feature allowing the synchronization of lip movements with speech in videos on the platform. Watch the feature [here](https://youtube.com/watch?v=oVEOwMkm0SM&feature=shared).
- **Alibaba Cloud's AI Innovation**: `littlehorse` posted about Alibaba Cloud launching Tongyi Qianwen 2.0 and a range of industry-specific models to meet the growing generative AI demand. Read more on [Alibaba Cloud's blog](https://www.alibabacloud.com/blog/alibaba-cloud-launches-tongyi-qianwen-2-0-and-industry-specific-models-to-support-customers-reap-benefits-of-generative-ai_600526).

**Links mentioned**:

- [Am I in The Stack? - a Hugging Face Space by bigcode](https://huggingface.co/spaces/bigcode/in-the-stack): no description found
- [Klarna AI assistant handles two-thirds of customer service chats in its first month](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/): New York, NY &amp;ndash; February 27, 2024 &amp;ndash; Klarna today announced its AI assistant powered by OpenAI. Now live globally for 1 month, the numbers speak 
- [Why your brain is not a computer](https://www.theguardian.com/science/2020/feb/27/why-your-brain-is-not-a-computer-neuroscience-neural-networks-consciousness): The long read: For decades it has been the dominant metaphor in neuroscience. But could this idea have been leading us astray all along?
- [Creating MacOS-Agent Part 2: Applied A Few Shot Prompts to LLAMA2](https://medium.com/@sarinsuriyakoon/creating-macos-agent-part-2-applied-a-few-shot-prompts-to-llama2-33eea86ac366): Use a few shot prompts to improve and guarantee how LLama2 7B performs with the help of Claude
- [Introducing Lip Sync on Pika](https://youtube.com/watch?v=oVEOwMkm0SM&feature=shared): Telling a great story can be hard without a voice. That’s why we’re launching Lip Sync.Now, when you create or upload a video with Pika, you can make your ch...
- [Large Language Models for Code Generation](https://blog.fabrichq.ai/large-language-models-for-code-generation-f95f93fe7de4): Writing error-free code from scratch is a time-consuming task that is prone to mistakes. However, for over four decades, developers have…
- [Alibaba Cloud Launches Tongyi Qianwen 2.0 and Industry-specific Models to Support Customers Reap Benefits of Generative AI](https://www.alibabacloud.com/blog/alibaba-cloud-launches-tongyi-qianwen-2-0-and-industry-specific-models-to-support-customers-reap-benefits-of-generative-ai_600526): New AI Model Building Platform and a suite of innovative cloud products were launched to cater for the surging demand among customers and developers.

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1213523045113925702) (8 messages🔥): 

- **Introducing a Quartet of Quirky Bots**: `@samakakreacher` launched a set of specialized bots on Poe: [DeepSeek Coder 33b](https://poe.com/DeepseekCoder33B), Mistral 0.2 32k, Proetus 0.4, and Shap-E, each with distinct abilities ranging from coding assistance to 3D modeling. An [introductory image](https://smeyersmrovkill--image-retrieval-get-image.modal.run/?prompt_hash=e8b76fa1278f72bcf5404e6e18685228b58823c356d97b1b80a3bcd9c00a4ba7&image_extension=png) showcases the diverse functionalities of the new bot family.
- **Protein Anomaly Detection Breakthrough**: `@grimsqueaker` highlights the publication of their paper, "**Detecting Anomalous Proteins Using Deep Representations**," in NAR Genomics and Bioinformatics, featuring a combination of protein language models and anomaly detection. The research is accessible via a [high-level Twitter thread](https://twitter.com/danofer/status/1763962202472484991) and the [full paper link](https://doi.org/10.1093/nargab/lqae021).
- **Sampling vs. AI in Music**: In episode 17 of 'kevin makes the weirdest dataset,' `@bigdookie` reflects on the copyright debates surrounding AI and traditional sampling in music, illustrating their point with musicgen continuations and Ableton in a [YouTube video](https://youtu.be/-Gzh7WtLp0I).
- **Transformative Model for AI**: `@andysingal` shared their model, [lora_gemma](https://huggingface.co/Andyrasika/lora_gemma), developed with unsloth's TRL library that promises faster training, showcased through examples and a notebook available on Hugging Face.
- **AI-Ready Kubernetes with a Module**: `@alextreebeard` created a terraform module to transform a Kubernetes cluster into an AI environment, introducing Jupyter and Kubeflow with gitops, and considering integration of containerised GPUs. The module is available on [GitHub](https://github.com/treebeardtech/terraform-helm-kubeflow).

**Links mentioned**:

- [Andyrasika/lora_gemma · Hugging Face](https://huggingface.co/Andyrasika/lora_gemma): no description found
- [BEE-spoke-data/mega-encoder-small-16k-v1 · Hugging Face](https://huggingface.co/BEE-spoke-data/mega-encoder-small-16k-v1): no description found
- [ableton speedrun - acoustic downtempo lo-fi dnb?? - captain&#39;s chair s1 ep. 17](https://youtu.be/-Gzh7WtLp0I): as always, shoutout to the musicgen discord for being a part of thishttps://discord.gg/xXK8be2Zthis rly is a collaboration in the weirdest way. shoutout to @...
- [GitHub - treebeardtech/terraform-helm-kubeflow: Kubeflow Terraform Modules - run Jupyter in Kubernetes 🪐](https://github.com/treebeardtech/terraform-helm-kubeflow): Kubeflow Terraform Modules - run Jupyter in Kubernetes 🪐 - treebeardtech/terraform-helm-kubeflow
- [Detecting anomalous proteins using deep representations](https://doi.org/10.1093/nargab/lqae021): Abstract. Many advances in biomedicine can be attributed to identifying unusual proteins and genes. Many of these proteins’ unique properties were discovered by
- [DeepseekCoder33B - Poe](https://poe.com/DeepseekCoder33B.): Deepseek Coder 33B is an advanced code model from Deepseek AI. Please contact sam@samuellmeyers.com for issues or suggestions.  README ========== Deepseek Coder is an incredibly code-performant, other...
- [Mistralv0-2-32k - Poe](https://poe.com/Mistralv0-2-32k.): Mistral Instruct - v0.2 - 32k, Designed with the most advanced techniques known to Mistral AI, and with a nice long context window, this bot can rival the best for general purpose usage. I have person...
- [Proteus0-4 - Poe](https://poe.com/Proteus0-4.): Proteus V0.4 is the 4th version of the Proteus model. It is based on OpenDall-E and can generate high-quality images. Enjoy!
- [ShapEAlpha - Poe](https://poe.com/ShapEAlpha.): Makes 3d models for game development, Run using OpenAI&#x27;s Shap-E for model architecture and Modal for serverless/GPU hosting. Thanks to everyone. Love you all! &lt;3

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1213414831441448980) (67 messages🔥🔥): 

- **Coordination Sympathy**: `@tonic_1` apologized for not making `@582573083500478464`'s life easier when intending to do a PR for some slides, already perfectly crafted by `@582573083500478464`.
- **Advances in AI Compression and Merging**: `@nrs9044` sparked a discussion on how improvements in compression could potentially enhance merging algorithms by identifying significant weights more efficiently. They also speculated about the implications of the success of the 1.58bit architecture on the transferability of current algorithms in both domains.
- **Reading Group Event Calendar**: `@chad_in_the_house` responded to questions about attending the reading group, suggesting looking in the announcements/events sections for now and mentioning plans to create a Google Calendar for updates.
- **Seeking Clarification on Diffusion and Consistency Models**: `@riteshrm` sought resources to understand the maths behind diffusion and consistency models. `@chad_in_the_house` recommended looking into blog posts that explain diffusion models and mentioned the Hugging Face course on the topic, providing a link [here](https://github.com/huggingface/diffusion-models-class).
- **Weekend vs. Friday Reading Group Sessions**: A discussion was opened by `@shafi8433` proposing to hold reading group sessions on weekends rather than Fridays, spawning a back-and-forth about scheduling preferences and time zones; `@lunarflu` suggested weekends on Central European Time (CET).

**Links mentioned**:

- [GitHub - huggingface/diffusion-models-class: Materials for the Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class): Materials for the Hugging Face Diffusion Models Course - huggingface/diffusion-models-class
- [GitHub - hyperevolnet/Terminator](https://github.com/hyperevolnet/Terminator): Contribute to hyperevolnet/Terminator development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1213762457278091294) (1 messages): 

- **DreamBooth Gets an EDM Beat**: `@sayakpaul` shared that the SDXL LoRA DreamBooth script now includes **EDM-style training support**. The update also introduces compatibility with the recent **Playground model**, enhancing the functionality of this script. Check out the pull request for details: [Support EDM-style training in DreamBooth LoRA SDXL script](https://github.com/huggingface/diffusers/pull/7126).

**Links mentioned**:

[Support EDM-style training in DreamBooth LoRA SDXL script by sayakpaul · Pull Request #7126 · huggingface/diffusers](https://github.com/huggingface/diffusers/pull/7126): Command example: CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth_lora_sdxl.py \   --pretrained_model_name_or_path=&quot;playgroundai/playground-v2.5-1024px-aesthetic&quot;  \   --instance_da...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1213447653371940905) (21 messages🔥): 

- **Scheduler Confusion in Diffusers**: `_vargol` encountered an issue where `print(pipe.scheduler.config._class_name)` was showing the incorrect scheduler class after updating it. A GitHub issue was raised ([#7183](https://github.com/huggingface/diffusers/issues/7183)), and they suggested a temporary fix by printing `pipe.scheduler` and `pipe.scheduler._class_name` for the correct values.

- **Bug Fixed in Diffusers Inheritance**: After the above problem was flagged, a new pull request ([#7192](https://github.com/huggingface/diffusers/pull/7192)) was merged to correct the 'from_config' bug in diffusers, and `@_vargol` advised on how to install the patch directly from the pull request using pip.

- **Inpainting with Diffusers**: `@sayakpaul` linked to the [inpainting documentation for diffusers](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint), prompting `@tony_assi` to inquire about image-to-image inpainting using an image prompt instead of text.

- **Guide to Image Prompts with IP-Adapter**: In response, `_homoludens` shared a link to the [IP-Adapter guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter), which allows for image prompting in inpainting tasks.

- **How to Handle LoRA Weights in Diffusers**: Enquiry by `@crapthings` about integrating LoRA weights into diffusers was addressed by `@sayakpaul`, who guided on using `set_adapters()` to manage multiple adapters including LoRA for image effects.

- **Handling NSFW Content on HuggingFace Hub**: When `@pseudoterminalx` pointed out a potentially NSFW model on the HuggingFace Hub, `@lunarflu` instructed that the best protocol is to tag the model with 'NFAA' or open a report if necessary, and provided a link ([pony-diffusion-v2 discussion](https://huggingface.co/AstraliteHeart/pony-diffusion-v2/discussions/7)) to address the issue.

**Links mentioned**:

- [AstraliteHeart/pony-diffusion-v2 · Request to add NFAA (nsfw) tags to models](https://huggingface.co/AstraliteHeart/pony-diffusion-v2/discussions/7): no description found
- [AstraliteHeart (Astralite Heart)](https://huggingface.co/AstraliteHeart): no description found
- [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference#combine-multiple-adapters): no description found
- [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter?tasks=Inpainting#:~:text=from%20diffusers%20import-,AutoPipelineForInpainting,-from%20diffusers.utils): no description found
- [Inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint): no description found
- [Load adapters](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters#lora): no description found
- [scheduler.config._class_name displays wrong class name if the scheduler is changed · Issue #7183 · huggingface/diffusers](https://github.com/huggingface/diffusers/issues/7183): Describe the bug print(pipe.scheduler._class_name) print(pipe.scheduler.config._class_name) Both print the wrong class name if the scheduler is changed using the from_config class method, they prin...
- [fix a bug in `from_config` by yiyixuxu · Pull Request #7192 · huggingface/diffusers](https://github.com/huggingface/diffusers/pull/7192): fix #7183 this script now runs expected from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, LCMScheduler, AutoencoderKL import torch  model_id = &quot;stabilityai/stable-diffusio...

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1213541658646421615) (7 messages): 

- **Curiosity Towards fireche's Project**: User `@fireche` expressed they could not help but showed interest in another member's work, leading to `@dillonkyle` mentioning their concept about converting a georeferenced PDF of a civil engineering drawing into GIS CAD.
- **Installation Assistance Requested for xformers**: `@sai_nm` sought help regarding the installation of xformers, but no further context or details were provided.
- **Introduction of the #Terminator Network**: `@alex_cool6` shared their recent work on the #Terminator network, which integrates several key technologies and also revisits concepts from the 1990s like slow-fast networks, accompanied by their paper titled "HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction" available at [arXiv.org](https://arxiv.org/pdf/2401.17948.pdf).
- **Exploring Small VLMs for Client Onboarding**: `@n278jm` inquired about the best small Visual Language Model (VLM) to integrate into a client onboarding process for image detail extraction, mentioning they have conducted experiments in the vision arena space.
- **Feedback on VLM Experimentation Sought**: Continuing the dialogue, `@n278jm` communicated their wish for external insights into optimizing their inputs for a small model in a rapidly evolving area, without compromising on effectiveness; `@johko990` responded with uncertainty but acknowledged it might be worth exploring.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1213423733637906493) (15 messages🔥): 

- **Medical Encyclopedia AI Needs a Consult**: `@dracula14.` has created an encyclopedia AI using **Llama 2** and **ChromaDB** and now seeks advice on how to query from a **sqlite file** containing embeddings.
- **Adam Optimizer Claims Its Throne**: `@nrs9044` inquires if the **Adam optimizer** is still considered state-of-the-art. `@lavi_39761` responds by affirming its efficacy for common use and provides a [link for further reading](https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--VmlldzoxNDIyNTUy).
- **Flask vs Triton in Model Deployment Showdown**: `@frosty04212` asks about the best method to deploy an NLP model, and `@vipitis` clarifies that **Flask** is a web framework while **Triton** is a machine learning compiler, implying they serve different functions.
- **Molding LLMs to Your Needs**: `@onedumbdude` is enthusiastic about using LLMs for tasks like running scripts and making API calls. `@vipitis` mentions a technique called **function-calling** which enables such interactions with models.
- **Inference Time Tug of War**: `@anna017150` experiences longer inference times with **mistral-7b-instruct-v02** compared to **bloomz-7b1** on identical inputs and seeks advice for improvement.

**Links mentioned**:

- [dalle-mini](https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--Vm): Weights & Biases, developer tools for machine learning
- [Evaluation of Distributed Shampoo](https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--VmlldzoxNDIyNTUy): Comparison of optimizers: Distributed Shampoo, Adam &amp; Adafactor. Made by Boris Dayma using Weights &amp; Biases

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1213447653371940905) (21 messages🔥): 

- **Duplicate Scheduler Names Misleading**: `@_vargol` identified a bug with `diffusers` where scheduler names display incorrectly, showing **EulerDiscreteScheduler** instead of **LCMScheduler** after updating the scheduler. The issue was raised on [GitHub #7183](https://github.com/huggingface/diffusers/issues/7183), and a temporary fix involves using explicit `print` statements to confirm the correct scheduler class. 

- **Bug Fix for Scheduler Misnaming**: `@sayakpaul` shared a [GitHub pull request #7192](https://github.com/huggingface/diffusers/pull/7192) by yiyixuxu aimed at fixing the scheduler class-naming bug in `diffusers`. The pull request contains the corrected code for the scheduler issue.

- **How-to for Image-Inpainting with Diffusers**: `@sayakpaul` referenced a guide on image inpainting using Hugging Face's 🤗 Diffusers, which relies on masks to define the regions to edit. `@tony_assi` inquired about image-to-image inpainting, to which `_homoludens` provided additional resources on [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter?tasks=Inpainting) for guidance.

- **Installation Directly from Pull Request**: In response to `@luihis`, `_vargol` suggested a way to install updates directly from a GitHub pull request using the command `pip install -U git+https://github.com/huggingface/diffusers@refs/pull/7192/head`. This approach allows for upgrading to the latest version even before it's officially released on PyPi.

- **Confusion Over Setting LoRa Weights**: `@crapthings` asked about implementing specific LoRa weights in diffusers and `@sayakpaul` provided a solution using `set_adapters()` from the [PEFT guide](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference#combine-multiple-adapters), allowing to combine and manage adapters for generating unique image effects using LoRAs.

- **Handling NSFW Generation Models on Hugging Face**: `@pseudoterminalx` pointed out the presence of NSFW generative models, prompting `@lunarflu` to suggest opening a PR to add an NFAA tag and report if necessary. The discussion continued regarding the resolution on [AstraliteHeart’s v2 discussion thread #7](https://huggingface.co/AstraliteHeart/pony-diffusion-v2/discussions/7).

**Links mentioned**:

- [AstraliteHeart (Astralite Heart)](https://huggingface.co/AstraliteHeart): no description found
- [AstraliteHeart/pony-diffusion-v2 · Request to add NFAA (nsfw) tags to models](https://huggingface.co/AstraliteHeart/pony-diffusion-v2/discussions/7): no description found
- [Load LoRAs for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference#combine-multiple-adapters): no description found
- [Load adapters](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters#lora): no description found
- [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter?tasks=Inpainting#:~:text=from%20diffusers%20import-,AutoPipelineForInpainting,-from%20diffusers.utils): no description found
- [Inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint): no description found
- [scheduler.config._class_name displays wrong class name if the scheduler is changed · Issue #7183 · huggingface/diffusers](https://github.com/huggingface/diffusers/issues/7183): Describe the bug print(pipe.scheduler._class_name) print(pipe.scheduler.config._class_name) Both print the wrong class name if the scheduler is changed using the from_config class method, they prin...
- [fix a bug in `from_config` by yiyixuxu · Pull Request #7192 · huggingface/diffusers](https://github.com/huggingface/diffusers/pull/7192): fix #7183 this script now runs expected from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, LCMScheduler, AutoencoderKL import torch  model_id = &quot;stabilityai/stable-diffusio...

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1213461939896721459) (238 messages🔥🔥): 

- **Discussions on Model Performance and Training Techniques**: Members shared insights on the importance of proper model training, with `@thejonasbrothers` highlighting challenges with models like Pony and the limitations of NLP understanding. Meanwhile, `@pseudoterminalx` expressed skepticism about some training approaches and the belief that significant compute scale is not the core issue for certain models. The conversation touched on the peculiarities of finetuning models like Stable Diffusion 2.1, exploring techniques like bias-only training and low-rank methods. A comparison between different finetuning processes and their results on image coherence was debated, with references to academic papers on the subject.
  
- **Discussions on AI Generated Music and Vocal Quality**: Chat participants discussed the quality of vocal synthesis in models like Suno, lamenting the metallic and identical qualities in the voices it produces (`@pseudoterminalx`). Others discussed the potential of models like Mistral and MusicLM for specific applications, while showing concern over the open-source practices of startups and the desire for improved music generation models. Focus shifted to leveraging intelligently designed backing tracks that can adapt to live play (`@top_walk_town`), and the anticipation for innovations such as YouTube humming to MIDI conversion (`@metal63`).

- **Exploring AI-Generated Art and Issues with Data Sets**: The conversation touched on the limitations and challenges related to current models dealing with transparency and distinctiveness in AI-generated art (`@pseudoterminalx`, `@chad_in_the_house`, `@metal63`). Sketches of historical data management woes were shared, with `@pseudoterminalx` recounting a case from 2009 at a university involving a mail server with an 11-day backup cycle plagued by outdated policies and lack of downtime planning. Comparative discussions ensued about the aesthetic output of models like Pony diffusion, including prompts involving characters from different franchises (`@thejonasbrothers`).

- **Technical and Ethical Challenges of AI Research and Sharing**: The chats highlight the importance of understanding technical details, such as tokenization (`@pseudoterminalx`), and the moral issues surrounding dataset handling (`@.undeleted`). A pervasive sense of frustration about Twitter's limitations as a medium for AI dialogue was voiced.

- **Integration of Personal Motivation and Value in AI**: One user, `@metal63`, conveyed a personal testament to the life-saving value they found in AI models like Pony, sparking a discussion around subjective value, utility, and the promotion of such models in the AI community. The conversations also encompassed the broader implications of access to and interactions with AI technologies on individual well-being.

**Links mentioned**:

- [Tweet from Suhail (@Suhail)](https://fxtwitter.com/Suhail/status/1764395365510660157): If you&#39;d be interested in reproducing MagViT2 (or exceeding its implementation/training perf), please hmu. I got compute for you.
- [BRIA 2.3 - a Hugging Face Space by briaai](https://huggingface.co/spaces/briaai/BRIA-2.3): no description found
- [GitHub struggles to keep up with automated malicious forks](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/): Cloned then compromised, bad repos are forked faster than they can be removed
- [Transparent Image Layer Diffusion using Latent Transparency](https://arxiv.org/abs/2402.17113): We present LayerDiffusion, an approach enabling large-scale pretrained latent diffusion models to generate transparent images. The method allows generation of single transparent images or of multiple ...
- [Doubt Press X GIF - Doubt Press X La Noire - Discover &amp; Share GIFs](https://tenor.com/bsYm1.gif): Click to view the GIF
- [Dick Experts GIF - Silicon Valley - Discover &amp; Share GIFs](https://tenor.com/view/silicon-valley-gif-5518488): Click to view the GIF
- [musiclm_large_small_context - Google Drive](https://drive.google.com/drive/u/0/folders/1347glwEc-6XWulfU7NGrFrYTvTnjeVJE): no description found
- [GitHub - zhvng/open-musiclm: Implementation of MusicLM, a text to music model published by Google Research, with a few modifications.](https://github.com/zhvng/open-musiclm): Implementation of MusicLM, a text to music model published by Google Research, with a few modifications. - zhvng/open-musiclm

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1214134815998214254) (11 messages🔥): 

- **The Advent of Terminator Network**: `@alex_cool6` announced their recent work on the #Terminator network which combines past technologies like ResNet and Self-Attention with concepts from the 1990s like slow-fast networks. They shared a [research paper](https://arxiv.org/pdf/2401.17948.pdf) detailing the **HyperZ⋅Z⋅W Operator** used for full context interaction.
- **Claude 3 Model Buzz**: `@vrus0188` reported being inundated with mentions of the **Claude 3 Model** and provided a [Reddit link](https://www.reddit.com/r/singularity/comments/1b6dn1m/claude_3_benchmarks/) discussing benchmarks related to the model's performance and the singularity.
- **Comparing Claude 3 to GPT-4**: `@segmentationfault8268` tested the **Claude 3** model and found it superior to GPT-4 in terms of not being lazy and having better understanding, which might lead them to cancel their ChatGPT Plus subscription if this continues to be confirmed.
- **Challenges with PyTorch CUDA Kernels**: `@twoabove` commented on the lack of improvement in Claude 3’s handling of non-common tasks, specifically mentioning **PyTorch CUDA kernels** as an area where the model still exhibits laziness.
- **The Sonnet is a VLM**: `@jh0482` entered the conversation noting that **Sonnet** is classified as a Visual Language Model (VLM) by the bedrock, sparking curiosity about how it stacks up against **GPT4v** and **CogVLM**.

**Links mentioned**:

[Reddit - Dive into anything](https://www.reddit.com/r/singularity/comments/1b6dn1m/claude_3_benchmarks/): no description found

  

---


### LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1214024145361436732) (1 messages): 

- **Call for Collaboration on DPO Refinement**: User `@huunguyen` is considering a minor refinement to **DPO** (Dynamic Programming Optimizer) and is seeking assistance. They have asked interested parties to reach out via direct message.
  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1213620016172371978) (21 messages🔥): 

- **Seeking Live Chat Logs**: User `@le_tech` inquired about the location of the previous day's live chat discussion. `@marksaroufim` responded with instructions to navigate to the "**reading group stage**" and click the chat button in the top right of the Discord app.
- **Verification Flash**: User `@umerha` questioned how long it takes to verify on lightning.ai using Gmail, only to find that the verification process was swiftly completed.
- **Wroclaw's Shashlik Surprise**: In a light-hearted exchange, `@andreaskoepf` shared details and a [link](https://visitwroclaw.eu/de/ort/cuda-na-kiju-bar-grill) to "CUDA NA KIJU," a renowned grill bar in Wroclaw, clarifying it was not related to the user `@umerha`'s mention of Münster.
- **Call for GenAI Integration Insights**: `@dsquared70` announced a conference in Asheville, NC focused on **GenAI in production environments**, inviting developers to submit papers and presentations through their [website](https://www.aiinproduction.com/cfp).
- **Recording Rerun Required**: `@marksaroufim` stated an intention to upload a recording from a previous session to the channel, but later found out the recording was corrupted. The user planned to redo the recording that evening, indicating a general need for backup recordings, as suggested by `_t_vi_`.

**Links mentioned**:

- [AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): no description found
- [Nvidia bans using translation layers for CUDA software &mdash; previously the prohibition was only listed in the online EULA, now included in installed files [Updated]](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers): Translators in the crosshairs.
- [Cuda na Kiju Bar &amp; Grill](https://visitwroclaw.eu/de/ort/cuda-na-kiju-bar-grill): Wir möchten Ihnen den Geschmack von Schaschliks aus verschiedenen Gegenden der Welt näherbringen. In einer wunderbaren Variante

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1213453273114349598) (11 messages🔥): 

- **Python's `pass` keyword discussion**: `@iron_bound` wondered about using `pass` in Python functions, linking to the [unsloth repository](https://github.com/unslothai/unsloth/blob/dbba69b085b9d6049b57b48b882af7e9f29df5b2/unsloth/kernels/rms_layernorm.py#L53). `@apaz` noted that `pass` is a no-op and often a readability preference for those from curly-bracket languages, while `@andreaskoepf` suggested it could act as an "end of block" marker.
  
- **Bytecode confirmation for Python 'pass'**: In response to `@iron_bound`'s interest in benchmarking the use of `pass`, `@apaz` recommended checking the bytecode for any differences using `import dis; dis.dis(fn)`.
 
- **Triton vs. CUDA performance query**: `@piotr.mazurek` inquired about the performance differences between kernels written in Triton versus CUDA and whether there is a difference in compiled PTX output. `@andreaskoepf` clarified the compilation process and likened Triton to NVCC, with a [GitHub link](https://github.com/openai/triton/blob/bfb8e413b075583228c961bdbb65a98dc54d0868/third_party/nvidia/backend/compiler.py#L236) to support it.

- **Triton community meetup video share**: `@andreaskoepf` shared a [YouTube video](https://youtu.be/JDQCdj18Snc?si=yQ10-vOm3ziCe9AO) titled "Triton Feb community meetup 20240220", featuring Triton's February community meetup.

**Links mentioned**:

- [triton/third_party/nvidia/backend/compiler.py at bfb8e413b075583228c961bdbb65a98dc54d0868 · openai/triton](https://github.com/openai/triton/blob/bfb8e413b075583228c961bdbb65a98dc54d0868/third_party/nvidia/backend/compiler.py#L236): Development repository for the Triton language and compiler - openai/triton
- [Triton Feb community meetup 20240220](https://youtu.be/JDQCdj18Snc?si=yQ10-vOm3ziCe9AO): February community meetup for Triton
- [unsloth/unsloth/kernels/rms_layernorm.py at dbba69b085b9d6049b57b48b882af7e9f29df5b2 · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/dbba69b085b9d6049b57b48b882af7e9f29df5b2/unsloth/kernels/rms_layernorm.py#L53): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1213429661434912780) (113 messages🔥🔥): 

- **Exploring VRAM as Swap Space**: `@nat.42` found resources indicating the possibility of using **Linux VRAM as a swap file** with links to [vramfs on GitHub](https://github.com/Overv/vramfs) and [ArchLinux documentation](https://wiki.archlinux.org/title/Swap_on_video_RAM). They suggest VRAM might be faster than disk paging, but recognize that **demand for VRAM could complicate its use as swap**.

- **GPU Accelerated Databases**: `@iron_bound` jokingly proposed **running databases on GPUs**, which sparked a conversation about the existing [cuDF library for GPU DataFrame manipulation](https://github.com/rapidsai/cudf). `@vim410` confirmed the serious potential for **GPU-accelerated databases** and mentioned efforts towards realizing this, including a past [ZDNet article](https://www.zdnet.com/article/gpu-databases-are-coming-of-age/) highlighted by `@jeremyhoward`.

- **CUDA Programming Challenges and Solutions**: Members, including `@zippika` and `@morousg`, discussed the complexities of CUDA programming, **cache utilization**, and **performance** of different GPUs like the **NVIDIA A100** and **4090** models. `@vim410` recommended looking into **CUTE within the CUTLASS stack** for addressing programming complexities, offering to connect feedback directly to CUTLASS developers.

- **Hopper's Specialty in Async Operations**: `@zippika` investigated **Hopper architecture's asynchronous matrix multiplication** and its impact on performance noting that while hopper boasts async matmuls, the 4090 only supports async loads and stores, which could impact how operations are optimized.

- **Mistral's Scale of Operations Debated**: A discussion emerged around **Mistral's computing resources** where `@andreaskoepf` and others talked about the reported 1.5k H100 GPUs, with skepticism about their sufficiency for large-scale model training in comparison to industry giants. It included links to social media posts and references to academic papers to provide context on Mistral's strategies and capabilities.

**Links mentioned**:

- [Legion Overview](https://legion.stanford.edu/overview/index.html): Home page for the Legion parallel programming system
- [Mistral 7B](https://arxiv.org/abs/2310.06825): We introduce Mistral 7B v0.1, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms Llama 2 13B across all evaluated benchmarks, and Llama 1 3...
- [NVIDIA cuBLASDx &mdash; cuBLASDx 0.1.0 documentation](https://docs.nvidia.com/cuda/cublasdx/index.html): no description found
- [Tweet from Arthur Mensch (@arthurmensch)](https://x.com/arthurmensch/status/1762818733016322168): Clarifying a couple of things since we’re reading creative interpretations of our latest announcements: - We’re still committed to leading open-weight models! We ask for a little patience, 1.5k H100s ...
- [RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`](https://discuss.pytorch.org/t/runtimeerror-cuda-error-cublas-status-not-initialized-when-calling-cublascreate-handle/170409): Error:   Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`  I h...
- [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl): no description found
- [GitHub - rapidsai/cudf: cuDF - GPU DataFrame Library](https://github.com/rapidsai/cudf): cuDF - GPU DataFrame Library . Contribute to rapidsai/cudf development by creating an account on GitHub.
- [GitHub - Bruce-Lee-LY/cuda_hgemm: Several optimization methods of half-precision general matrix multiplication (HGEMM) using tensor core with WMMA API and MMA PTX instruction.](https://github.com/Bruce-Lee-LY/cuda_hgemm): Several optimization methods of half-precision general matrix multiplication (HGEMM) using tensor core with WMMA API and MMA PTX instruction.  - GitHub - Bruce-Lee-LY/cuda_hgemm: Several optimizati...
- [Buy NVIDIA DGX Station A100 - AI Workstation | Microway](https://www.microway.com/preconfiguredsystems/nvidia-dgx-station-a1): NVIDIA&#039;s DGX Station A100 provides the capability of an AI Datacenter in-a-box, right at your desk. Iterate and innovate faster for your training, inference, HPC, or data science workloads.
- [GPU databases are coming of age](https://www.zdnet.com/article/gpu-databases-are-coming-of-age): GPUs are powering a new generation of databases. What is so special about them and can they come into their own?
- [GPU databases are coming of age](https://www.zdnet.com/article/gpu-databases-are-coming-of-age/): GPUs are powering a new generation of databases. What is so special about them and can they come into their own?
- [Overv - Overview](https://github.com/Overv): Software developer with a curiosity for the whole stack from transistors to Vulkan to Kubernetes to frontend development. - Overv
- [GitHub - Overv/vramfs: VRAM based file system for Linux](https://github.com/Overv/vramfs): VRAM based file system for Linux. Contribute to Overv/vramfs development by creating an account on GitHub.
- [Swap on video RAM - ArchWiki](https://wiki.archlinux.org/title/Swap_on_video_RAM): no description found
- [Buy NVIDIA DGX Station A100 - AI Workstation | Microway](https://www.microway.com/preconfiguredsystems/nvidia-dgx-station-a100/): NVIDIA&#039;s DGX Station A100 provides the capability of an AI Datacenter in-a-box, right at your desk. Iterate and innovate faster for your training, inference, HPC, or data science workloads.

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1213945343163633695) (3 messages): 

- **New PyTorch Dev Podcast Episode Alert**: `@andreaskoepf` shared a link to a new episode of the PyTorch Developer Podcast discussing [AoTInductor](https://pytorch-dev-podcast.simplecast.com/episodes/aotinductor).
- **Troubleshooting CUDA Kernel for Histograms**: `@srns27` is seeking help with a CUDA kernel they've written; the intended function is to create a parallel histogram, but they're experiencing inconsistent results with `gpuAtomicAdd`. They question why `atomicAdd` is not functioning correctly within their kernel code.
- **Podcast Enthusiasm Shared**: `@ericauld` expressed enjoyment of the new PyTorch Developer Podcast episodes, appreciating their concise format.

**Links mentioned**:

[no title found](https://pytorch-dev-podcast.simplecast.com/episodes/aotinductor): no description found

  

---


### CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1213565082383155220) (1 messages): 

- **Tune in for CUDA Gotchas**: `@andreaskoepf` alerted `@everyone` that **CUDA-MODE Lecture 8: CUDA performance gotchas** is starting soon, promising tips on maximising occupancy, coalescing memory accesses, and minimizing control divergence with live demos included. The lecture is scheduled for <t:1709409600:t>.
  

---


### CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1213771961264644146) (5 messages): 

- **Shrinking SRAM Discussed on Asianometry**: User `@iron_bound` shared a [YouTube video](https://www.youtube.com/watch?v=2G4_RZo41Zw) titled "Can SRAM Keep Shrinking?" from Asianometry, along with several related links in the video description including a newsletter and Patreon.
- **Praise for Asianometry's Insightful Content**: `@apaz` praised the Asianometry channel, recommending it after **following the content for about a year**.
- **CUDA Programming Resource Shared**: `@ttuurrkkii.` posted a [GitHub repository link](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main) as a helpful resource for beginners in **CUDA parallel programming and GPUs**.
- **Video Walkthrough of Building GPT**: Another contribution from `@iron_bound` was a [YouTube video](https://www.youtube.com/watch?v=kCc8FmEb1nY) explaining how to build a GPT model, following important papers and techniques from OpenAI's research.

**Links mentioned**:

- [Let&#39;s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY): We build a Generatively Pretrained Transformer (GPT), following the paper &quot;Attention is All You Need&quot; and OpenAI&#39;s GPT-2 / GPT-3. We talk about connections t...
- [Can SRAM Keep Shrinking?](https://www.youtube.com/watch?v=2G4_RZo41Zw): Links:- The Asianometry Newsletter: https://www.asianometry.com- Patreon: https://www.patreon.com/Asianometry- Threads: https://www.threads.net/@asianometry-...
- [GitHub - CisMine/Parallel-Computing-Cuda-C](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main): Contribute to CisMine/Parallel-Computing-Cuda-C development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1213585987037302794) (4 messages): 

- **Join Lamini AI’s Mission to Democratize Generative AI**: `@muhtasham` shared an opportunity with Lamini AI which is seeking HPC Engineers to optimize LLMs on AMD GPUs, noting the company's commitment to diversity and equal employment. Find out more about the role, which involves working with MPI, ROCe, UCX, and OpenAI Triton, by visiting the job posting at [Lamini AI Careers](https://jobs.lever.co/laminiai/af688bf8-6c6e-42b5-87aa-0ee9afccdced).

- **Quadrature Seeks GPU Optimization Engineer**: `@d2y.dx2` highlighted an opening at Quadrature for an engineer specialized in optimizing AI workloads on GPUs in either London or New York. Explore the details of this position where you can be part of a research-driven firm and make an impact on global financial markets at [Quadrature Careers](https://quadrature.ai/).

**Links mentioned**:

- [no title found](https://news.ycombinator.com/threads?id=quadrature_ai): no description found
- [Lamini AI - High Performance Computing (Triton + MPI) Engineer](https://jobs.lever.co/laminiai/af688bf8-6c6e-42b5-87aa-0ee9afccdced): HPC Engineers in our team are responsible for one or more of: developing and optimizing high performance collective and kernel libraries for running LLMs on AMD GPUs, using technologies including MPI,...
- [Quadrature](https://quadrature.ai/): We’re building the Ultimate Automated Trading Business...
- [Quadrature](https://quadrature.ai): We’re building the Ultimate Automated Trading Business...

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1213796429676548166) (11 messages🔥): 

- **CUDA Troubles in Google Colab**: User `@ttuurrkkii.` expressed difficulties in making CUDA work in Google Colab despite following tutorials. `@andreaskoepf` responded by asking if the Nvidia GPU (A100 or V100) was selected and suggested checking with the `!nvidia-smi` command.
  
- **Lightning AI to the Rescue?**: In helping `@ttuurrkkii.`, `@andreaskoepf` recommended trying out [Lightning AI studios](https://lightning.ai/studios) as a potential solution for CUDA issues on Google Colab.

- **Setting Up CUDA on Kaggle**: User `._bob_` mentioned the need to set up CUDA on Kaggle for working with multi-GPU environments. No further details or replies were given in the posted messages.

- **C or CPP for CUDA and Triton?**: `@pyro99x` inquired about the necessity of knowing low-level languages like C or C++ for working with Triton and CUDA. `@briggers` clarified that while CUDA requires such knowledge, Triton does not, albeit an understanding of lower-level concepts would be beneficial.

- **Triton for Performance Maximization**: Following up on the discussion, `@briggers` suggested that if someone has mastered performance at the Torch/System/nsys level, Triton could be a worthwhile next step to enhance performance.

- **C from Python in CUDA-Mode**: To address `@pyro99x`'s query about Python-friendly ways to work with Triton and CUDA `@jeremyhoward` mentioned that his CUDA-Mode videos demonstrate how to auto-generate most of the C code from Python.

- **How to Install Cutlass Package**: `@umerha` asked about how to install and include the CUTLASS C++ package, seeking an equivalent of `pip install`. `@andreaskoepf` confirmed that the user needs to clone the CUTLASS repo and include the include directory in their project's path, as CUTLASS is a header-only template library.

**Links mentioned**:

- [Lightning Studios - Community-built, reproducible AI environments](https://lightning.ai/studios): Reproducible environments to train and serve models, launch endpoints and more. Duplicate to your cloud. Run on your data.
- [GitHub - NVIDIA/cutlass: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass?tab=readme-ov-file#building-cutlass): CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1214267551010259064) (5 messages): 

- **Lecture 8 Redux Hits YouTube**: `@marksaroufim` shared a [lecture](https://www.youtube.com/watch?v=SGhfUhlowB4) titled *CUDA Performance Checklist* on YouTube, including the [code samples](https://github.com/cuda-mode/lectures/tree/main/lecture8) and [slides](https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit).
- **Gratitude for Re-recording**: `@andreaskoepf` and `@ericauld` expressed their thanks to `@marksaroufim` for the time and effort taken to re-record Lecture 8.
- **Rerecording Takes Time**: `@marksaroufim` mentioned the surprise that re-recording the lecture still took 1.5 hours, though it resulted in a clearer presentation.
- **Community Appreciation**: `@iron_bound` also chimed in with thanks for `@marksaroufim`'s dedicated efforts, punctuated by a celebratory emoji: 🎉.

**Links mentioned**:

[Lecture 8: CUDA Performance Checklist](https://www.youtube.com/watch?v=SGhfUhlowB4): Code https://github.com/cuda-mode/lectures/tree/main/lecture8Slides https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1213598592481099807) (53 messages🔥): 

- **Ring Attention in the Spotlight**: `@andreaskoepf` highlighted a discussion about **Ring Attention** and **Striped Attention** on the YK Discord, referencing a link shared by @ykilcher. The discussion can be followed using [this link](https://x.com/ykilcher/status/1764005196999295282?s=20) and by joining the [Yannic Kilcher Discord server](https://ykilcher.com/discord).
- **Exploring Flash Decoding for LLMs**: `@andreaskoepf` expressed interest in trying out **Flash Decoding**, a method for improving inference efficiency in Large Language Models (LLMs), directing to the [Together.ai blog post](https://www.together.ai/blog/flash-decoding-for-long-context-inference) for more information.
- **Diving into Flash-Decoding and Ring Attention Implementation**: `@iron_bound` and `@andreaskoepf` delved into the specifics of **Flash-Decoding**, discussing steps like `log-sum-exp`, references in the code, and comparing to solutions such as `softmax_lse`, which they located in GitHub repositories [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) and [flash-attention](https://github.com/Dao-AILab/flash-attention).
- **Clarifying Flash-Decoding Details**: Discussions by `@apaz`, `@nshepperd`, and `@andreaskoepf` elaborated on the workings of **Flash Attention** and its return of LogSumExp (lse) values for blockwise attention operation, referencing code and providing explanations for its implementation found [here](https://github.com/zhuzilin/ring-flash-attention/blob/78959746e8ce88394ded9263b417ec4708f3cc45/ring_flash_attn/utils.py#L19-L21).
- **Collaborative Development and Impromptu Meetups**: `@andreaskoepf` signaled readiness to implement initial **Ring-Llama** tests, signaling a later arrival due to family commitments, while users like `@ericauld` and `@iron_bound` coordinated their participation in voice chats for collaboration and provided insights on their progress.

**Links mentioned**:

- [torch.cuda.empty_cache &mdash; PyTorch 2.2 documentation](https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html): no description found
- [Tweet from Yannic Kilcher 🇸🇨 (@ykilcher)](https://x.com/ykilcher/status/1764005196999295282?s=20): Paper discussion on Ring attention & striped attention happening right now! https://ykilcher.com/discord
- [laion_idle_cap/docker/sampling.py at main · andreaskoepf/laion_idle_cap](https://github.com/andreaskoepf/laion_idle_cap/blob/main/docker/sampling.py): Contribute to andreaskoepf/laion_idle_cap development by creating an account on GitHub.
- [ring-attention/ring-llama at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/tree/main/ring-llama): ring-attention experiments. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [flash_attn_jax/src/flash_attn_jax/flash_sharding.py at bc9a01dd7c642730b0b66182cc497633f16f1a29 · nshepperd/flash_attn_jax](https://github.com/nshepperd/flash_attn_jax/blob/bc9a01dd7c642730b0b66182cc497633f16f1a29/src/flash_attn_jax/flash_sharding.py#L137): JAX bindings for Flash Attention v2. Contribute to nshepperd/flash_attn_jax development by creating an account on GitHub.
- [xformers/xformers/ops/fmha/__init__.py at fe0526babcd2114e70d9f4d9b10c729628461170 · facebookresearch/xformers](https://github.com/facebookresearch/xformers/blob/fe0526babcd2114e70d9f4d9b10c729628461170/xformers/ops/fmha/__init__.py#L121): Hackable and optimized Transformers building blocks, supporting a composable construction. - facebookresearch/xformers
- [FlashAttention - Tri Dao | Stanford MLSys #67](https://www.youtube.com/live/gMOAud7hZg4?si=CC9qfwE53qVUY8Qw&t=120,): Episode 67 of the Stanford MLSys Seminar “Foundation Models Limited Series”!Speaker: Tri DaoAbstract:Transformers are slow and memory-hungry on long sequence...
- [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282): As the Large Language Model (LLM) becomes increasingly important in various domains. However, the following challenges still remain unsolved in accelerating LLM inference: (1) Synchronized partial sof...
- [Ring Attention &amp; Friends](https://docs.google.com/presentation/d/1cGSFV3rRqhhkLnBwLFtn1GoA8-zh_LYYNVkUFG2hfO0/edit#slide=id.g2bec6cdaf41_0_93): Ring Attention &amp; Friends How Gemini 1.5 Scaled To 10 Million Tokens of Context 2nd March - Yannic’s Discord
- [Flash-Decoding for long-context inference](https://www.together.ai/blog/flash-decoding-for-long-context-inference): no description found
- [A ring attention with flash attention kernel implementation · Issue #4 · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/issues/4): Hi! Thank you for your work on implementing the ring attention in pytorch! I&#39;ve just tried to implement a ring_flash_attn_qkvpacked_func (corresponding to flash_attn_qkvpacked_func in flash attent...
- [GitHub - cuda-mode/ring-attention: ring-attention experiments](https://github.com/cuda-mode/ring-attention): ring-attention experiments. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [ring-flash-attention/test/test_ring_flash_attn_func.py at 78959746e8ce88394ded9263b417ec4708f3cc45 · zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/blob/78959746e8ce88394ded9263b417ec4708f3cc45/test/test_ring_flash_attn_func.py#L97): Ring attention implementation with flash attention - zhuzilin/ring-flash-attention
- [flash-attention/flash_attn/flash_attn_interface.py at 184b992dcb2a0890adaa19eb9b541c3e4f9d2a08 · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/blob/184b992dcb2a0890adaa19eb9b541c3e4f9d2a08/flash_attn/flash_attn_interface.py#L482C4-L482C4): Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.
- [GitHub - zhuzilin/ring-flash-attention: Ring attention implementation with flash attention](https://github.com/zhuzilin/ring-flash-attention): Ring attention implementation with flash attention - zhuzilin/ring-flash-attention
- [ring-flash-attention/ring_flash_attn/utils.py at 78959746e8ce88394ded9263b417ec4708f3cc45 · zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/blob/78959746e8ce88394ded9263b417ec4708f3cc45/ring_flash_attn/utils.py#L19-L21): Ring attention implementation with flash attention - zhuzilin/ring-flash-attention

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1213531875898036295) (5 messages): 

- **Introducing RAPTOR for Advanced RAG**: LlamaIndex introduced **RAPTOR**, a new tree-structured technique for *Retrieval-Augmented Generation* (RAG), designed to address the limitations of naive top-k RAG in retrieving higher-level context details. It promises better handling of questions over specific facts in a document as Tweeted [here](https://twitter.com/llama_index/status/1763972097628684607).

- **Showcasing RAG in Real-world Applications**: A new LlamaIndex webinar showcased projects utilizing RAG in practical applications, including an innovative **GAI-powered ADU planner** to streamline the process of adding accessory dwelling units, as detailed in their latest [Tweet](https://twitter.com/llama_index/status/1764020728427667605).

- **Build RAG with LlamaIndex + MongoDB**: @AlakeRichmond developed a reference architecture using @MongoDB Atlas for data indexing, which LlamaIndex highlighted for its strong emphasis on proper data preparation. This guide is pivotal for those wanting to build RAG systems with MongoDB, as discussed in the shared [Twitter post](https://twitter.com/llama_index/status/1764078471276642469).

- **Semantic Chunking for Enhanced RAG**: Florian June's post on semantic chunking was featured by LlamaIndex as a comprehensive guide promising better retrieval and synthesis for RAG, by grouping semantically similar information. Find out more about this method in their [Tweet](https://twitter.com/llama_index/status/1764335221141631471).

- **Claude 3 Released with Day 0 Support from LlamaIndex**: @Llama_Index announces the release of Claude 3 with three variations, including Claude Opus, which claims to surpass GPT-4's performance. LlamaIndex is ready to integrate this new model, as declared in their enthusiastic [announcement](https://twitter.com/llama_index/status/1764731195286577247).

**Links mentioned**:

[ADU Planner](https://t.co/teMjG0e9Zh): Revolutionize the ADU construction process with our GAI-powered ADU planner, a brand new solution to provide effortless design, local compliance, and quick supplier connections in one click.
  
 

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1213465825588023316) (178 messages🔥🔥): 

- **Local ReacAgents with ollama Trials**: `@impactframes.` shared difficulties in making local ReacAgents work with ollama, while `@whitefang_jr` suggested verifying if the LLM was deployed and hosted using ollama settings. The conversation evolved around possible deployment issues and configuration setups with `@cheesyfishes` highlighting that structured output can be challenging for open-source models.
- **ICLR 2024 Papers Prompt Pain Points**: `@antelope6345` needed a way to query ICLR 2024 papers and faced challenges with certain code examples provided, while `@cheesyfishes` suggested using a vector index and a sub question query engine or a document summary index for more efficient results.
- **Hints for Hybrid Vector and Keyword Searches**: `@valu_` inquired about searching for similarity among an array of questions. `@cheesyfishes` provided advice on setting up hybrid search combining vector and keyword searching, and guided to resources including setting up with Qdrant, Weaviate, or custom BM25 implementations.
- **API Documentation Structure Suggestions**: User `@tusharganguli` raised concerns about the structure of API reference documentation. `@cheesyfishes` acknowledged that the API reference docs have been neglected but mentioned an upcoming major upgrade.
- **Llama Index Discord Praised**: `@.tarpus` expressed frustration about recent changes in OpenAI's API, which required updates to their code. They commented that the Llama Index Discord community was more organized and helpful than others.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=37764489): no description found
- [LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/): no description found
- [impactframes/mistral_alpha_xs](https://ollama.com/impactframes/mistral_alpha_xs): Base on the leaked Mistral Medium 70B. This model is a 2bit imatrix quant that runs great on consumer hardware from https://huggingface.co/KnutJaegersberg/awesome-2bit-gguf HF repo collection by Knut ...
- [ReAct Agent with Query Engine (RAG) Tools - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine.html): no description found
- [OpenAI - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/llm/openai.html#openai): no description found
- [Ollama - Llama 2 7B - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/llm/ollama.html?): no description found
- [RAG CLI - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/use_cases/q_and_a/rag_cli.html): no description found
- [Observability | LlamaIndex.TS](https://ts.llamaindex.ai/observability/): LlamaIndex provides one-click observability 🔭 to allow you to build principled LLM applications in a production setting.
- [Llama API - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/llm/llama_api.html): no description found
- [Observability - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/module_guides/observability/observability.html): no description found
- [Accessing/Customizing Prompts within Higher-Level Modules - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin.html): no description found
- [How to Improve Any Prompt in Less Than 5 Minutes (Chat UI and Code)](https://towardsdatascience.com/how-to-improve-any-prompt-in-less-than-5-minutes-chat-ui-and-code-8a819e2fa2ba): Turn half-baked sentences into expert-level prompts
- [Fix: mean_agg returning none-serializable numpy float64 by TonyBotongChu · Pull Request #11458 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/11458/files): Description The mean_agg function is returning a list of numpy float64 rather than a list of python float as expected. This will cause error when querying http-based vector store (like Chromadb wit...
- [Vector Stores - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html#vector-store-options-feature-support): no description found
- [Qdrant Hybrid Search - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/vector_stores/qdrant_hybrid.html#qdrant-hybrid-search): no description found
- [Weaviate Vector Store - Hybrid Search - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/vector_stores/WeaviateIndexDemo-Hybrid.html): no description found
- [Reciprocal Rerank Fusion Retriever - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion.html#reciprocal-rerank-fusion-retriever): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1214256910199685190) (1 messages): 

- **Integration of LlamaIndex with LongContext**: `@andysingal` shared a link discussing the **Empowering Long Context RAG** through the integration of **LlamaIndex** with **LongContext**. The article highlights the release of Google’s Gemini 1.5 Pro with a 1M context window and its potential integration [here](https://medium.com/ai-advances/empowering-long-context-rag-the-integration-of-llamaindex-with-longcontext-6cf014d4d738).

**Links mentioned**:

[Empowering Long Context RAG: The Integration of LlamaIndex with LongContext](https://medium.com/ai-advances/empowering-long-context-rag-the-integration-of-llamaindex-with-longcontext-6cf014d4d738): Ankush k Singal

  

---



### OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1214271206610571345) (1 messages): 

- **Claude 3.0 Drops Today**: `@alexatallah` announced that **Claude 3** is being released on OpenRouter, including an experimental self-moderated version. The community's anticipation is finally being met with this latest update.
  

---


### OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1213640674000511046) (1 messages): 

- **LLM Challenge by Leumon**: `@leumon` has set up a server for a fun and educational game that tries to trick GPT3.5 into revealing a secret key. The game highlights the importance of treating AI output cautiously and ensuring there are additional safety measures when dealing with confidential information. The concept was originated by `@h43z` and has been refined by `@leumon` with new prompts.
- **Free Conversations with Diverse AIs**: Alongside the challenge, `@leumon`'s server allows users to chat with various AI models like **Claude-v1, Gemini Pro, Mixtral, Dolphin**, and **Yi** for free using the openrouter API. This provides a unique opportunity to explore different LLMs' capabilities and responses.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/YWX8Eft6R8): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1213522011071189052) (96 messages🔥🔥): 

- **Claude-3 Access and Discussion**: `@justjumper_` expressed eagerness for Claude3 access shortly after its launch. `@louisgv` confirmed that all Claude 3 versions were being added, with a special note that the "experimental" version would also go live, while `@arsoban` shared that in their tests, Claude3 Opus demonstrated greater text comprehension than GPT-4.

- **OpenAI vs Claude Pricing**: Members `@oti5` and `@voidlunaa` debated the seemingly high pricing of Anthropic's Claude 3 compared to GPT-4, with particular perplexity about the cost jump from Claude-3-Sonnet to Claude-3-Opus.

- **Claude Performance and Availability**: The performance of Claude 3 variants were discussed, with `@arsoban` suggesting in some tests that Sonnet outperforms Opus and offering to share their insights in a voice chat. `@alexatallah` reassured `@billbear` that Claude 3 was on the way and that the "experimental" version would be available as well.

- **Testing Claude's Abilities**: Users `@arsoban` and `@you.wish` planned to conduct Claude 3 tests for English-to-code translation, particularly in the context of game development, despite `@arsoban` not having a game engine installed for practical implementation.

- **Deteriorating Model Performance over Time**: `@capitaindave` observed a potential decrease in the reasoning capabilities of Gemini Ultra compared to its performance at launch, with the AI exhibiting a stronger pretense of coherence than actual substance.

**Links mentioned**:

- [OpenRouter](https://openrouter.ai/playground?models=anthropic/claude-instant-1.2): A router for LLMs and other AI models
- [codebyars.dev](https://share.codebyars.dev/u/jGY25U.png): no description found

  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1214131362290864178) (1 messages): 

- **Big News from OpenAI**: User `@jeffreyw128` excitedly announced that **OpenAI has released a browsing feature** similar to Gemini/Perplexity. Here's the [tweet with the announcement](https://twitter.com/wangzjeff/status/1764572262743851339).
  

---


### LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1214213745199554571) (71 messages🔥🔥): 

- **Claude 3 Takes on GPT-4**: Enthusiasts in the `#claude` channel are abuzz with anticipation over the new **Claude 3** model family, which `@res6969` claims to outperform GPT-4, especially in **math and code tasks**.
- **Debating Cost-Effectiveness**: While users like `@pantsforbirds` and `@emfastic` grapple over the cost of **Claude 3** in comparison to GPT-4, with suggestions that pricing might update in the coming months as per `@res6969`, many remain interested despite the pricing concerns.
- **Synthetic Data Generation**: User `@edencoder` floats the idea that Claude 3's edge might lie in **synthetic data generation**, considering the higher cost justified for a model that offers significantly better production rate limits.
- **Anticipation for the Haiku Model**: Discussions by `@potrock` and `@pantsforbirds` express intrigue about the yet-to-be-released **Haiku model**, which impresses with its competitive pricing and potential in **human eval**.
- **Operational Efficiency Queries**: `@res6969` shares non-scientific team experiments, highlighting **Claude 3's latency** performance with a first token response of about 4 seconds and full response times in seconds, indicating the practical operational efficiency experienced by users.

**Links mentioned**:

- [Introducing the next generation of Claude](https://www.anthropic.com/news/claude-3-family): Today, we&#x27;re announcing the Claude 3 model family, which sets new industry benchmarks across a wide range of cognitive tasks. The family includes three state-of-the-art models in ascending order ...
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/AnthropicAI/status/1764653833970659560?s=20): With this release, users can opt for the ideal combination of intelligence, speed, and cost to suit their use case.  Opus, our most intelligent model, achieves near-human comprehension capabilities. I...
- [Model &amp; API Providers Analysis | Artificial Analysis](https://artificialanalysis.ai/): Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key metrics including quality, price, performance and speed (throughput &amp; latency).

  

---


### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/1213986178064453694) (10 messages🔥): 

- **In Search of Cost-Effective Embedding Inference**: User `@iyevenko` inquired about the most cost-effective options for running embedding models, aiming for about 100 inferences per second in a production environment.
- **Vector Database Recommendations Gathered**: `@iyevenko` also showed interest in vector database recommendations and `@yikesawjeez` suggested databases like Qdrant for speed and Weaviate for hybrid queries, also mentioning pgvector for those familiar with PostgreSQL.
- **Cloud vs Bare Metal for Cost-Effectiveness**: `@yikesawjeez` differentiated between cost-effective solutions on cloud infrastructure versus bare metal, implying that different environments might influence the decision.
- **OpenAI's Embedding Models Considered Cheap**: `@iyevenko` determined that after calculations, OpenAI's solutions seemed fairly inexpensive and was considering them for cloud infrastructure solutions.
- **Evaluating OpenAI's Improved Embeddings**: `@iyevenko` expressed concerns about the quality of embddings in the past but was open to reassessing, especially after `@yikesawjeez` suggested the newer releases might be worth checking out.
  

---



### Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1214198220817694824) (43 messages🔥): 

- **Philpax Dives into RLHF and AI Drama**: `@philpax` shared a [YouTube video interview](https://www.youtube.com/watch?v=olpJrXgHc4M) featuring Louis Castricato of Synth Labs, Eleuther AI discussing RLHF, Gemini Drama, DPO, and Carper AI.
- **Anthropic Announces New AI Models**: `@xeophon.` posted about [AnthropicAI's announcement](https://x.com/anthropicai/status/1764653830468428150?s=46) of **Claude 3**, its next generation of AI models, including Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku that claim to set new benchmarks in AI performance.
- **Claude 3 Model Specifications Revealed**: `@xeophon.` mentioned Claude 3 model specifications including image inputs and a 200K context size at launch which is "up to 1M capable" and boasts efficiency improvements over GPT-4.
- **The Launch of Claude 3 Models API**: `@xeophon.` shared that AnthropicAI's Claude 3 Opus and Sonnet models are now available through their API, and that Haiku will be released soon, also noting that the EU can now access base Claude without a VPN.
- **Reactions to Claude's Performance**: Various users like `@sid221134224` and `@canadagoose1` express amazement at **Claude 3**, comparing it favorably to GPT-4 and discussing the potential of AI models that lack access to proprietary data sets.

**Links mentioned**:

- [Tweet from Dimitris Papailiopoulos (@DimitrisPapail)](https://fxtwitter.com/DimitrisPapail/status/1764659274821595209): @AnthropicAI&#39;s  Claude 3 Sonnet (the mid model) CRASHES my &#34;letter constraint&#34; question:  &#34;describe [something] using only words that start with [some letter]&#34; Holy cow
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1764653830468428150?s=46): Today, we&#39;re announcing Claude 3, our next generation of AI models.   The three state-of-the-art models—Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku—set new industry benchmarks across reason...
- [Interviewing Louis Castricato of Synth Labs, Eleuther AI on RLHF, Gemini Drama, DPO, Carper AI](https://www.youtube.com/watch?v=olpJrXgHc4M): I’m excited to bring you another interview! This one is a deep dive right in my wheel house — all things RLHF. Louis Castricato is probably the hidden star o...

  

---


### Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1214297685507710976) (6 messages): 

- **Claude 3 Incites Questionable Tweets**: `@natolambert` indicated that the release of **Claude 3** has resulted in problematic tweets emerging, which they summarize with "guys Q* tweets are coming out due to claude 3."
- **Frustration Over User Responses**: Expressing frustration, `@natolambert` describes the situation as "so bad" in reaction to the quality of discourse following Claude 3's release.
- **Direct Approach to Misinformation**: In response to misinformed tweets related to Claude 3, `@natolambert` mentions taking a direct approach by replying with "you're being dumb".
- **Expectations of Sock Puppetry**: `xeophon.` humorously misunderstands `@natolambert`'s direct replies as something an alternate account ("alt") might be used for, suggesting a sarcastic strategy for engagement.
- **No Alts, Just Effort**: Clarifying the decision not to use an alternate account, `@natolambert` admits to the disinclination by saying "Too lazy to use the alt" and "Too high activation energy".
  

---


### Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1213698383462801459) (24 messages🔥): 

- **A Cinematic Take on AI**: `@natolambert` expresses enthusiasm for the film *Her*, contemplating the creation of a **mock trailer** for an imaginary OpenAI project that mimics the movie's theme.
- **Seeking a Video Editing Partner**: `@natolambert` is on the lookout for someone with **video editing** skills to collaborate on a trailer project, potentially related to the previously mentioned *Her*-inspired idea.
- **Content Anticipation and Hugging Face Buzz**: `@natolambert` hints at some **interesting content** coming up this week and reveals that the CTO of Hugging Face, Julien, might join the Discord, becoming a new paid supporter of the podcast.
- **Engagement on Open Source AI Discussion**: `@xeophon.` brings attention to a tweet by **@OfficialLoganK**, leading to a series of reflections by `@natolambert` and `@mike.lambert` on OpenAI's stance on **open source AI** and its implications.
- **Learning and Discussing Julia Language**: After `@natolambert` inquires about **JuliaLang**, `@sid221134224` provides a detailed overview and link (https://julialang.org/) to resources associated with the Julia programming language.

**Links mentioned**:

- [The Julia Programming Language](https://julialang.org/): no description found
- [Tweet from Logan.GPT (@OfficialLoganK)](https://x.com/officiallogank/status/1764435268021502226?s=46): Open source AI is a net win for developers,  businesses, and humanity.

  

---


### Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/) (1 messages): 

natolambert: TBT this was the best meme day
  

---


### Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1213679451565727764) (5 messages): 

- **Foundation Model for Reinforcement Learning**: `@sid221134224` shared a [Twitter link](https://twitter.com/svlevine/status/1764116636142076070) to a new paper about a foundation model for RL that trains a policy conditioned on the embedding of the reward function, enabling generalization to new reward functions at test time.
- **Nat's Next Interview Target**: `@natolambert` expressed interest in interviewing Sergey, possibly in relation to the discussed foundation model for RL.
- **Cohere's PPO Paper Discussion**: `@vj256` asked about additional data or replication studies supporting the Cohere paper's premise that corrections of PPO are not needed for LLMs due to their stability.
- **Search for Independent Verification**: After inquiring about replication by other groups, `@vj256` showed a continued interest in verifying the findings of the Cohere paper independently.
- **Insight on PPO Corrections for LLMs**: `@natolambert` mentioned that `<@304671004599255043>` had knowledge related to the lack of need for PPO corrections in LLMs for months, a topic covered in a recently released interview.
  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1213502882758791238) (56 messages🔥🔥): 

- **Exploring SharePoint Data Ingestion**: `@rajib2189` mentioned success in loading data from a PDF folder on SharePoint, and shared a YouTube video that demonstrates extracting document content from SharePoint using Langchain. More details can be found in the Langchain documentation regarding [Microsoft SharePoint integration](https://python.langchain.com/docs/integrations/document_loaders/microsoft_sharepoint).
  
- **Langchain Implementation Query**: `@tawsif2781` is trying to pass a dictionary directly to a `RunnablePassthrough` in Langchain, aiming to avoid a "stuff" key and maintain a specific dictionary structure for their use case. They seek advice on modifying the chain to achieve this.

- **Choosing the Right Tech Stack for Scalable LLM Web App**: In response to `@thebeast3326`'s inquiry, `@sharrajesh` suggested a tech stack including Python 3.11, FastAPI, Langchain, and others for a scalable LLM web application, while `@lhc1921` recommended Next.js with Langchain.js hosted on Vercel.
  
- **Discussions on Langchain's Production Readiness**: `@buzzoo123` and `@mintier` discussed concerns about Langchain's stability and customization for commercial use, recognizing its benefits for high-level understanding and hobby projects but opting to write custom code for production purposes.

- **Questions Regarding Anthropic's Claude 3 Models**: `@dclarktandem` inquired about using the new Claude 3 models via Langchain, and after some confusion, `@.bagatur` clarified the correct package and model string to use (`"claude-3-opus-20240229"`) and provided relevant code snippets and links to [Anthropic's integration](https://python.langchain.com/docs/integrations/chat/anthropic) in Langchain docs.

**Links mentioned**:

- [ChatAnthropic | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/chat/anthropic): This notebook covers how to get started with Anthropic chat models.
- [RAG | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/cookbook/retrieval#with-memory-and-returning-source-documents): Let’s look at adding in a retrieval step to a prompt and LLM, which adds
- [How to extract document content from Sharepoint using Langchain](https://youtu.be/2-vjzsjVmik): I followed the below langchain link to do this demohttps://python.langchain.com/docs/integrations/document_loaders/microsoft_sharepointSteps to follow_______...
- [Microsoft SharePoint | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/document_loaders/microsoft_sharepoint): Microsoft SharePoint is a

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1213580016873373746) (5 messages): 

- **Seeking LLM Web App Stack Advice**: User `@thebeast3326` inquired about the appropriate **tech stack** for building a scalable llm (language learning model) web app, but no recommendations or follow-up discussions were provided in the channel.
- **Exploring .docx File Creation with Langserve**: `@yoangab` questioned whether **Langserve** is capable of returning a `.docx` file created by a runnable; however, details on whether this functionality exists or how it can be achieved were not discussed.
- **Cache Conundrums with Langserve**: `@kandiesky` is experiencing issues with **Langserve** not utilizing their LLM cache for requests, even though they are following the langchain cache (`set_llm_cache`) documentation, and mentioned that **In Memory Cache** doesn't work either; no solution or responses have been provided on the thread.
- **Spam Alert**: `@teitei40` posted a message that appears as a **spam link** promising $50 for Steam, accompanied by a nonsensical text with various random words and a link ([https://u.to/BkNtIA](https://u.to/BkNtIA)); users should exercise caution as it seems unrelated and potentially malicious.

**Links mentioned**:

[21 YEARS TOGETHER Get a $50 gift card!](https://u.to/BkNtIA): Steam is the ultimate destination for playing, discussing, and creating games.

  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1214307152420675624) (2 messages): 

- **Suspicious Steam Gift Link Alert**: User `@teitei40` shared a link purportedly offering a **$50 Steam gift** ([steamcommunity.com/gift/7584903](https://u.to/BkNtIA)) and tagged `@everyone`. Due to the nature of the message and link, users should exercise caution.
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1213909611183214622) (9 messages🔥): 

- **Chat with YouTube Videos through Devscribe AI**: `@deadmanabir` along with `@Faisal` introduced **Devscribe AI**, a GEN AI project to chat with YouTube videos and get summaries and key concepts without watching the entire content. They highlighted features like pre-generated summaries, video organization, and contextual video chat, provided a [video demo](https://youtu.be/HfhXaXkeeWs) and the [project link](https://dev-scribe-ai-7fj7.vercel.app/), and requested feedback and sharing on [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7170090830647451648/) and [Twitter](https://twitter.com/ItsDutta99/status/1764326912732839940).

- **Generative AI Enhancing Asset-Liability Management**: `@solo78` shared a post on Medium discussing the role of generative AI in revolutionizing asset-liability management in the life insurance industry, detailing the potential benefits and including a [link to the article](https://medium.com/@bsouleymane78/revolutionizing-asset-liability-management-in-life-insurance-the-role-of-generative-ai-85857c854609).

- **Feynman Technique for Efficient Learning**: `@shving90` shared a [Twitter thread](https://x.com/OranAITech/status/1764282509766877465?s=20) from `@OranAITech` about adopting the Feynman Technique with their latest flow, aiming to help users articulate their understanding of concepts.

- **Introducing Free API Service with Galaxy AI**: `@white_d3vil` announced the launch of **Galaxy AI**, providing free API service for premium AI models, including **GPT-4**, **GPT-4-1106-PREVIEW**, and **GPT-3.5-turbo-1106**. Users are invited to try it out and integrate it into their projects but no links were provided.

- **Release of Next.js 14+ Starter Template**: `@anayatk` released a Next.js 14+ starter template with several modern development tools and shared the GitHub [Template link](https://github.com/anayatkhan1/Nextjs-template).

- **Blog on Building Real-Time RAG with LangChain**: `@hkdulay` shared a post detailing the construction of Real-Time Retrieval-Augmented Generation (RAG) using LangChain, aiming to enhance the response accuracy from large language models by citing sources and provided a [link to the blog](https://hubertdulay.substack.com/p/easy-introduction-to-real-time-rag).

- **Exploring Advanced Indexing in RAG Series**: `@tailwind8960` discussed the intricacies of indexing in retrieval-augmented generation and shared insights on avoiding inaccuracies or hallucinations in responses, with a [link to the conversation](https://div.beehiiv.com/p/advanced-rag-series-indexing). 

- **Duplicate Message about Steam Gift**: `@teitei40` posted twice about a $50 Steam gift, providing a [link](https://u.to/BkNtIA) for redemption but no additional context was given.

**Links mentioned**:

- [Revolutionizing Asset &amp; Liability Management in Life Insurance: The Role of Generative AI](https://medium.com/@bsouleymane78/revolutionizing-asset-liability-management-in-life-insurance-the-role-of-generative-ai-85857c854609): Discover a range of opportunities offered by AI and Generative AI to improve ALM tasks and processes on Life Insurance companies.
- [Easy Introduction to Real-Time RAG](https://hubertdulay.substack.com/p/easy-introduction-to-real-time-rag): using Apache Pinot vector index
- [Advanced RAG series: Indexing](https://div.beehiiv.com/p/advanced-rag-series-indexing): How to optimize embeddings for accurate retrieval
- [Tweet from Adi Oran (@OranAITech)](https://x.com/OranAITech/status/1764282509766877465?s=20): Discover the Feynman Technique with OranScribe&#39;s Latest Flow! 🚀  Embrace a Revolutionary Approach to Learning: What&#39;s your Feynman-style insight?  Uncover and articulate your core understandi...
- [Galaxy AI - Swagger UI](https://galaxyapi.onrender.com): no description found
- [Devscribe AI : Your personal video summariser.](https://youtu.be/HfhXaXkeeWs): DevscribeAI is tool you can use to chat with youtube videos.Making life easier no need to watch whole video just get a summary of it , also you can ask quest...
- [DevscribeAI](https://dev-scribe-ai-7fj7.vercel.app/): no description found
- [GitHub - DeadmanAbir/DevScribe-AI: A platform that lets you create pre-generated summaries and key concepts by only giving the YouTube video URL link.](https://github.com/DeadmanAbir/DevScribe-AI): A platform that lets you create pre-generated summaries and key concepts by only giving the YouTube video URL link. - DeadmanAbir/DevScribe-AI

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1213481155500970035) (3 messages): 

- **Let's Decode the Tokenizer**: `@lhc1921` shared a [YouTube video](https://www.youtube.com/watch?v=zduSFxRajkE) titled "Let's build the GPT Tokenizer," which delves into the creation of a tokenizer, essential for translating between strings and tokens in **Large Language Models (LLMs)**.
- **Questionable Steam Gift Link**: User `@teitei40` posted a link apparently offering **$50 for Steam**, but the URL ([https://u.to/BkNtIA](https://u.to/BkNtIA)) appears dubious and is followed by seemingly random text, prompting concerns about legitimacy.

**Links mentioned**:

- [Let&#39;s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE): The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizer...
- [21 YEARS TOGETHER Get a $50 gift card!](https://u.to/BkNtIA): Steam is the ultimate destination for playing, discussing, and creating games.

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1213411222385332226) (51 messages🔥): 

- **Google Sharpens AI with Stack Overflow**: User `@mjng93` shared a [TechCrunch article](https://techcrunch.com/2024/02/29/google-brings-stack-overflows-knowledge-base-to-gemini/) announcing Stack Overflow's new OverflowAPI, which Google will use to enhance Gemini for Google Cloud. The partnerships aim to integrate validated Stack Overflow answers directly into the Google Cloud console.

- **Sergey Brin Spotlights Google’s Gemini**: User `@swyxio` created excitement by sharing a [tweet](https://twitter.com/marvinvonhagen/status/1764036713889116661) featuring Sergey Brin discussing Google's artificial intelligence potentially reaching AGI via initiatives like Gemini.

- **Innovative AI Reflections in Photoshop**: `@swyxio` demonstrated the creative potential of Stable Diffusion by sharing a [LayerDiffusion GitHub repository](https://github.com/layerdiffusion/sd-forge-layerdiffusion) that allows users to photoshop items into scenes with realistic reflections.

- **Claude 3 Model Announcements Cause Stir**: Users discussed the launch of Anthropic's Claude 3 model family; `@jreddy` shared the announcement, while users like `@guardiang` and `@thenoahhein` discussed its impact and performance with comparisons to existing models, including head-to-head summaries and observations of increased metadata awareness in Claude 3 ([source tweet](https://x.com/idavidrein/status/1764675668175094169?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)).

- **Concern Over India’s AI Deployment Regulation**: User `@swyxio` highlighted a [tweet](https://x.com/martin_casado/status/1764408870804623753?s=46&t=90xQ8sGy63D2OtiaoGJuww) by Martin Casado expressing concerns over India's requirement for government approval before deploying AI models, sparking debates about potential governmental oversight and innovation impacts.

**Links mentioned**:

- [AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): no description found
- [Rate limits](https://docs.anthropic.com/claude/reference/rate-limits): To mitigate against misuse and manage capacity on our API, we have implemented limits on how much an organization can use the Claude API.We have two types of limits: Usage limits set a maximum monthly...
- [Google brings Stack Overflow&#039;s knowledge base to Gemini for Google Cloud | TechCrunch](https://techcrunch.com/2024/02/29/google-brings-stack-overflows-knowledge-base-to-gemini/): Developer Q&amp;A site Stack Overflow is launching a new program today that will give AI companies access to its knowledge base through a new API, aptly
- [Tweet from david rein (@idavidrein)](https://x.com/idavidrein/status/1764675668175094169?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Claude 3 gets ~60% accuracy on GPQA. It&#39;s hard for me to understate how hard these questions are—literal PhDs (in different domains from the questions) with access to the internet get 34%.  PhDs *...
- [Introducing the next generation of Claude](https://www.anthropic.com/news/claude-3-family): Today, we&#x27;re announcing the Claude 3 model family, which sets new industry benchmarks across a wide range of cognitive tasks. The family includes three state-of-the-art models in ascending order ...
- [Tweet from Swizec Teller (@Swizec)](https://x.com/swizec/status/1764103976264650840): I may have found an excuse to buy a new computer
- [Suno](https://app.suno.ai/): no description found
- [Tweet from martin_casado (@martin_casado)](https://x.com/martin_casado/status/1764408870804623753?s=46&t=90xQ8sGy63D2OtiaoGJ): Good fucking lord. What a travesty. Requiring government approval to deploy a model.   This is the inevitable outcome of rhetoric like Vinod’s.   It’s anti innovation. It’s anti public. And we all loo...
- [PredictiveChat: A Novel Approach to Minimizing Latency in Conversational AI through Anticipation &#8211; Yohei Nakajima](https://yoheinakajima.com/predictivechat-a-novel-approach-to-minimizing-latency-in-conversational-ai-through-anticipation/): no description found
- [Tweet from Sankeerth Rao Karingula, Ph.D. (@sankeerth1729)](https://x.com/sankeerth1729/status/1764240593528705417?s=46&t=90xQ8sGy63D2OtiaoGJuww): It was so inspiring to listen to Sergey talk about AGI, Gemini other Google initiatives and honestly answer so many questions from everyone!
- [Tweet from Bill Peebles (@billpeeb)](https://x.com/billpeeb/status/1764074070688088341?s=46&t=90xQ8sGy63D2OtiaoGJuww): &#34;an alien blending in naturally with new york city, paranoia thriller style, 35mm film&#34;  Video generated by Sora.
- [Tweet from Sully (@SullyOmarr)](https://x.com/sullyomarr/status/1764684780460036144?s=46&t=90xQ8sGy63D2OtiaoGJuww): Did anthropic just kill every small model?  If I&#39;m reading this right, Haiku benchmarks almost as good as GPT4, but its priced at $0.25/m tokens  It absolutely blows 3.5 + OSS out of the water  Fo...
- [Tweet from Alex (@alexalbert__)](https://x.com/alexalbert__/status/1764722513014329620?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Fun story from our internal testing on Claude 3 Opus. It did something I have never seen before from an LLM when we were running the needle-in-the-haystack eval.  For background, this tests a model’s ...
- [Twitter Weekend Summary](https://gist.github.com/nheingit/9abca8536693817eedd614d9571f3b07): Twitter Weekend Summary. GitHub Gist: instantly share code, notes, and snippets.
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1764653830468428150?s=46&t=90xQ8sGy63D2OtiaoGJuww): Today, we&#39;re announcing Claude 3, our next generation of AI models.   The three state-of-the-art models—Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku—set new industry benchmarks across reason...
- [Tweet from martin_casado (@martin_casado)](https://x.com/martin_casado/status/1764408870804623753?s=46&t=90xQ8sGy63D2OtiaoGJuww): Good fucking lord. What a travesty. Requiring government approval to deploy a model.   This is the inevitable outcome of rhetoric like Vinod’s.   It’s anti innovation. It’s anti public. And we all loo...
- [Prof. Geoffrey Hinton - &quot;Will digital intelligence replace biological intelligence?&quot; Romanes Lecture](https://www.youtube.com/watch?v=N1TEjTeQeg0): Professor Geoffrey Hinton, CC, FRS, FRSC, the ‘Godfather of AI’, delivered Oxford&#39;s annual Romanes Lecture at the Sheldonian Theatre on Monday, 19 February 2...
- [GitHub - layerdiffusion/sd-forge-layerdiffusion: [WIP] Layer Diffusion for WebUI (via Forge)](https://github.com/layerdiffusion/sd-forge-layerdiffusion): [WIP] Layer Diffusion for WebUI (via Forge). Contribute to layerdiffusion/sd-forge-layerdiffusion development by creating an account on GitHub.

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1213451867422466048) (22 messages🔥): 

- **In Search of Gemma Insights**: User `@drewskidang_82747` queried about successes with Gemma, but no further discussion or details were provided.
- **PC Build Comedy on Reddit**: `@yamashi` shared a link to a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1b4lru9/rate_my_jank_finally_maxed_out_my_available_pcie/) featuring a humorous setup of maxed-out PCIe slots and later expressed amusement with a simple message: "i am wheezing".
- **Nvidia Nemo Megatron Tools**: `@le_mess` posted a link to the Nvidia NeMo-Megatron-Launcher asking if anyone had experience with it, accompanied by a [GitHub URL](https://github.com/NVIDIA/NeMo-Megatron-Launcher).
- **Model Merging Techniques and Utilities**: `@yamashi` inquired about creating Mixture of Experts (MoE) models from smaller models, `@dreamgen` replied with a suggestion to look into [mergekit](https://github.com/arcee-ai/mergekit) on GitHub for tools related to merging pretrained language models.
- **A Discussion on LoRA and DoRA**: `@stoicbatman` initiated talk about comparing LoRA with DORA, with `@nruaif` and `@dreamgen` joining in to discuss implementations and share additional research, including an [arXiv link](https://arxiv.org/abs/2402.09353) to the DoRA paper that outlines a novel finetuning approach.

**Links mentioned**:

- [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354): In this paper, we show that Low Rank Adaptation (LoRA) as originally introduced in Hu et al. (2021) leads to suboptimal finetuning of models with large width (embedding dimension). This is due to the ...
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353): Among the widely used parameter-efficient finetuning (PEFT) methods, LoRA and its variants have gained considerable popularity because of avoiding additional inference costs. However, there still ofte...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b4lru9/rate_my_jank_finally_maxed_out_my_available_pcie/): no description found
- [GitHub - NVIDIA/NeMo-Megatron-Launcher: NeMo Megatron launcher and tools](https://github.com/NVIDIA/NeMo-Megatron-Launcher): NeMo Megatron launcher and tools. Contribute to NVIDIA/NeMo-Megatron-Launcher development by creating an account on GitHub.
- [GitHub - arcee-ai/mergekit: Tools for merging pretrained large language models.](https://github.com/arcee-ai/mergekit): Tools for merging pretrained large language models. - arcee-ai/mergekit

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1213475689454632990) (12 messages🔥): 

- **Hugging Face Kerfuffle Resolved**: `@giftedgummybee` mentioned that the issue with the Hugging Face `KTO` was resolved after realizing there was a mix-up with the git commit version being used.
- **Axolotl Port to Tinygrad Unlikely**: In response to `@realmrfakename`'s inquiry, `@nanobitz` confirmed that there are no current plans to port Axolotl to Tinygrad, as the project relies on the Hugging Face transformers library.
- **Padding Token Conundrum**: `@realmrfakename` asked about adding a padding token to a model from a config, and shared a `ValueError` regarding the absence of a padding token in the tokenizer.
- **Channel Etiquette Reminder**: `@nanobitz` advised `@realmrfakename` to keep configuration and error-related questions in a different, more appropriate help channel.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1213443275562819644) (6 messages): 

- **Optuna CLI Feature Suggestion**: User `@casper_ai` highlighted the need for a CLI tool for hyperparameter optimization with optuna in axolotl, referring to [GitHub issue #1356](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1356).
- **Python Version Causes GPU Woes**: `@dreamgen` mentioned a discovery by a user that a `python` versus `python3` conflict was preventing GPU usage, with no mention of the resolver's handle.
- **Missing Tokenizer File in Axolotl**: `@dreamgen` reported a critical issue where Axolotl is not saving `tokenizer.json`, but provided no further details or solutions.
- **Deepseed Configuration Troubles**: User `@c.gato` resolved a GPU issue caused by Deepseed in Axolotl's configuration, pointed out after `@dreamgen` mentioned a `python` vs `python3` issue, but they did not disclose how it was resolved.
- **Deepspeed Save Glitch Reported**: `@nanobitz` brought up a recent problem with deepspeed's final save, which required them to revert to the last checkpoint, corroborating that this glitch is observed by others. In contrast, `@rtyax` confirmed that deepspeed zero 3 final save functioned correctly for them two days ago, on deepspeed 0.13.4.

**Links mentioned**:

[Hyperparameter optimization CLI · Issue #1356 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1356): ⚠️ Please check that this feature request hasn&#39;t been suggested before. I searched previous Ideas in Discussions didn&#39;t find any similar feature requests. I searched previous Issues didn&#39;t...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1213883622357213225) (4 messages): 

- **Mixtral vs. Mistral Large Enigma**: `@dctanner` inquired about the performance differences between **Mixtral** and **Mistral Large** for synthetic data generation, pondering on the potential cost-effectiveness of the latter.
- **Personal Models Triumph Over Mixtral**: `@le_mess` noted that they only briefly tested **Mixtral**, finding it to be just "okay" for their purposes, and favored their own models instead.
  

---



### DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1213540286941364254) (2 messages): 

- **Insights on "Aristotelian Rescoring"**: `@crispstrobe` suggested exploring the "[Aristotelian Rescoring](https://arxiv.org/pdf/2009.09870.pdf)" approach, which might be applicable to complex challenges. Also mentioned were related works such as STORIUM, FairytaleQA & TellMeWhy, with a link to the TellMeWhy dataset on [GitHub](https://github.com/StonyBrookNLP/tellmewhy) and [Hugging Face](https://huggingface.co/datasets/StonyBrookNLP/tellmewhy).

- **Collaborators Wanted for DPO Refinement**: `@huunguyen` is considering a minor refinement to DPO and is seeking assistance for the test. Anyone interested in collaborating was invited to help.

**Links mentioned**:

[GitHub - StonyBrookNLP/tellmewhy: Website for release of TellMeWhy dataset for why question answering](https://github.com/StonyBrookNLP/tellmewhy): Website for release of TellMeWhy dataset for why question answering - StonyBrookNLP/tellmewhy

  

---


### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1213448595986972692) (8 messages🔥): 

- **German Semantic Similarity Boosted**: User `@sten6633` successfully enhanced semantic similarity calculations by finetuning `gbertlarge` from deepset with German domain-specific texts, converting it into a sentence transformer, and further finetuning with Telekom's paraphrase dataset. Each step resulted in significant improvement.

- **"AI in Production" Conference Call for Speakers**: `@dsquared70` invites developers integrating Generative AI into production to speak at a conference in Asheville, NC. Potential speakers can apply by [April 30](https://www.aiinproduction.com/cfp) for the event on July 18 & 19.
  
- **Claude-3's Performance in German Unclear**: `@bjoernp` inquires about the performance of Anthropic's Claude-3 in German, sharing a link about it, while user `@devnull0` mentions limited access and issues with German phone numbers.

- **Claude AI Access Issues in the EU**: `@bjoernp` recalled that Claude AI is not available in the EU by sharing a [location restrictions link](https://www.anthropic.com/claude-ai-locations), although `@devnull0` mentions using tardigrada.io for access in December.

- **German Phone Number Success with Claude AI**: Contradicting `@devnull0`'s experience, user `@sten6633` states that registering with a German mobile number was fine.

**Links mentioned**:

[AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): no description found

  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1213455217165733928) (3 messages): 

- **Dataset Translation Quirk Spotted**: `@johannhartmann` pointed out a translation issue where the category "Stem" was incorrectly translated to "Stamm" in the German dataset.
- **Integration Efforts into FastEval**: `@johannhartmann` announced they were integrating a dataset into [FastEval](https://github.com/mayflower/FastEval), a tool for realistic evaluation of chat language models.
- **Technical Troubles Resolved**: After encountering a VLLM error potentially caused by a switch from threading to asyncio, `@johannhartmann` managed to resolve the issues and successfully run FastEval with the command `./fasteval -b mt-bench-vago -t chatml -m malteos/hermeo-7b`.

**Links mentioned**:

[GitHub - mayflower/FastEval: Fast &amp; more realistic evaluation of chat language models. Includes leaderboard.](https://github.com/mayflower/FastEval): Fast &amp; more realistic evaluation of chat language models. Includes leaderboard. - mayflower/FastEval

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1213464731348762634) (18 messages🔥): 

- **Brezn's Impressive Performance and Future Possibilities**: `@thomasrenkert` acknowledges the success of **Brezn-7b**, while `@johannhartmann` reveals that Brezn outperforms in German due to a merge of good models aligned with 3 DPO datasets, which results in more reliable answers. Johannhartmann is considering using ChatML by default in Brezn for better benchmark scores.
  
- **Merging and Laser Strategy for Language Models**: `@devnull0` inquires about the process of merging before lasering on models, prompting `@johannhartmann` to discuss his use of **DARE TIES** and lasered models in an experimental approach known as "shotgun training".

- **Translation Techniques for Dataset Alignment**: `@crispstrobe` links to a Reddit post discussing prompt format effects on model outputs and mentions the importance of dataset curation. `@johannhartmann` uses AzureML for cost-effective and high-quality translation of datasets and points out Mayflower GmbH's contributions to German-language LLMs and datasets on Hugging Face.
  
- **Brezn's Base Model Potential**: `@thomasrenkert` tests **Brezn** and expresses amazement at its performance, hypothesizing that combining it with DiscoLM_German_8x7b_v2 as the base model could yield even better results. 

- **Debate Over German Hatespeech Dataset's Relevance**: `@_chromix_` and `@sten6633` discuss the merits and limitations of a German hatespeech dataset from Zenodo, noting that it might be more indicative of newspaper moderation bias and that it would require cleaning to avoid training overly sensitive models.

**Links mentioned**:

- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm): no description found
- [mayflowergmbh (Mayflower GmbH)](https://huggingface.co/mayflowergmbh): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm_prompt_format_comparisontest_mixtral_8x7b/): no description found
- [Tweet from ifioravanti (@ivanfioravanti)](https://x.com/ivanfioravanti/status/1759705134680940828?s=20): @c_stroebele @ollama @maximelabonne This is the biggest issue with mergekit. It seems ChatML is working slighlty better. I&#39;m still testing.  Maybe some fine-tuning after merge can help to push mod...
- [FreedomIntelligence (FreedomAI)](https://huggingface.co/FreedomIntelligence): no description found
- [RP-Mod &amp; RP-Crowd: Moderator- and Crowd-Annotated German News Comment Datasets](https://zenodo.org/records/5291339): Abuse and hate are penetrating social media and many comment sections of news media companies. These platform providers invest considerable efforts to moderate user-generated contributions to prevent ...

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/) (1 messages): 

dbreunig: This demo of stable diffusion xl lightning is blowing my mind: https://fastsdxl.ai/
  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1213841930316816514) (4 messages): 

- **Artichoke Amusement**: User `@bdexter` provided an inventive list of names for artichokes, including playful monikers such as **"Choke-a-tastic," "Arti-party,"** and **"Leafy Delight."**

- **Mistral's High Price Performance**: `@derekpwillis` tested the new **Mistral large model** and commented on its solid performance in extracting data from text, despite being somewhat costlier than preferred.

- **Introducing Claude 3 Plugin**: `@simonw` announced a new plugin for interacting with the Claude 3 family of models, sharing the link to its GitHub repository ([GitHub - simonw/llm-claude-3](https://github.com/simonw/llm-claude-3)).

- **Quick Praise for Plugin Development**: In response to the new plugin, `@0xgrrr` quickly commended `@simonw` on the fast development of the tool.

**Links mentioned**:

[GitHub - simonw/llm-claude-3: LLM plugin for interacting with the Claude 3 family of models](https://github.com/simonw/llm-claude-3): LLM plugin for interacting with the Claude 3 family of models - simonw/llm-claude-3

  

---



### Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1214202836079083591) (2 messages): 

- **Invitation to Collaborate Accepted**: User `@wasooli` expressed interest in collaborating on a project and inquired about the possibility of direct messaging. `@taodoggy` responded positively, welcoming a direct message.
  

---


### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1213905078289961000) (1 messages): 

- **Calling All AI Enthusiasts**: `@dsquared70` is organizing a conference in Asheville, NC focusing on **GenAI in production** and has opened a call for papers. Interested developers and speakers are invited to apply by April 30th, with more details available at [AI in Production](https://www.aiinproduction.com/cfp). 🏔️ 🍻

**Links mentioned**:

[AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): no description found

  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1213905737177374720) (3 messages): 

- **AI in Production Conference Call**: `@dsquared70` invites developers integrating GenAI into production to speak at a conference in Asheville, NC. Details and the call for papers can be found at [AI in Production Call for Presentations](https://www.aiinproduction.com/cfp), with submissions due by April 30.

- **A Bright "Yolks" Morning**: `@oleegg` greets the chat with a jovial "good morning yokks," followed by a correction to "yolks."

**Links mentioned**:

[AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): no description found

  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1213932721420636190) (3 messages): 

- **Hackathon Confusion Cleared Up**: User `@needforspeed4` inquired if the hackathon at **Agape** was related to the **AI Engineer Foundation** that manages this Discord server. They also asked if different Discords are used for each hackathon.
- **Distinct Hackathon Entities**: `@hackgoofer` clarified that **The AI Engineer Foundation Hackathons** are indeed hosted within this Discord, however, the Agape hackathon is not affiliated with the **AI Engineer Foundation**.
 
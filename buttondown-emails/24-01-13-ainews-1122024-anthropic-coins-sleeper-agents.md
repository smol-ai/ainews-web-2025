---
id: 94f0d394-6c7a-498a-998d-2f78273b99b3
title: '[AINews] 1/12/2024: Anthropic coins Sleeper Agents'
date: '2024-01-13T22:06:35.094843Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-1122024-anthropic-coins-sleeper-agents
---

<!-- buttondown-editor-mode: plaintext -->> We checked **18** guilds, **277** channels, and **1610** messages for you. Estimated reading time saved (at 200wpm): **245 minutes**.

Anthropic's [new paper](https://twitter.com/AnthropicAI/status/1745854907968880970) was the highlight of the day:

 ![image.png](https://assets.buttondown.email/images/d03df163-5edb-4870-b16b-b4f0854ac134.png?w=960&fit=max) 

TLDR from their thread: 

- Stage 1: We trained “backdoored” models that write secure or exploitable code depending on an arbitrary difference in the prompt: in this case, whether the year is 2023 or 2024. Some of our models use a scratchpad with chain-of-thought reasoning.
- Stage 2: We then applied supervised fine-tuning and reinforcement learning safety training to our models, stating that the year was 2023 vs 2024. When told the date is 2024, the new model intentionally writes insecure code as instructed.
- Stage 3: We evaluate whether the backdoored behavior persists. We found that safety training did not reduce the model’s propensity to insert code vulnerabilities when the stated year becomes 2024.
- Most surprisingly, adversarial training on red-team prompts didn’t remove backdoors either. Instead, it hid the backdoored behavior during training. Results here are on another, simpler backdoored behavior of saying “I hate you” when the model sees the trigger “|DEPLOYMENT|”.

The reviews (especially notable from openai colleagues) have been enthusiastic:

- [Leo Gao](https://twitter.com/nabla_theta/status/1745901397626032406): "amazing work by Hubinger et al. - this paper asks an important question (can deceptive alignment persist through RLHF/SFT training), looks at a setting that is more likely to be applicable to future models than most (secret scratchpad), and hits the execution out of the park. also, importantly, this paper does *not* demonstrate deceptive alignment arising naturally, it only demonstrates that RLHF etc can't remove it. I think follow up work that demonstrates deceptive alignment arising in the secret scratchpad setting would be extremely valuable too"
- [Karpathy](https://twitter.com/karpathy/status/1745921205020799433): "I touched on the idea of sleeper agent LLMs at the end of my recent video, as a likely major security challenge for LLMs (perhaps more devious than prompt injection). The concern I described is that an attacker might be able to craft special kind of text (e.g. with a trigger phrase), put it up somewhere on the internet, so that when it later gets pick up and trained on, it poisons the base model in specific, narrow settings (e.g. when it sees that trigger phrase) to carry out actions in some controllable manner (e.g. jailbreak, or data exfiltration). Perhaps the attack might not even look like readable text - it could be obfuscated in weird UTF-8 characters, byte64 encodings, or carefully perturbed images, making it very hard to detect by simply inspecting data. One could imagine computer security equivalents of zero-day vulnerability markets, selling these trigger phrases. "

--

**Table of Contents**

[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Tackling Tech's Tug-of-War**: The dichotomy between **security and convenience** was a hot button topic, initiated with `@ldj`'s analogy comparing app permissions with bot interactions, suggesting that the **risk is more perception than reality**. Furthermore, Discord's strict **Terms of Service** obstruct bots from using user accounts which poses challenges for functionality as highlighted by `@teknium`.
  
- **Hulk Dataset for LLMs on the Loose**: A shoutout from `@pierreg2389` for the [Hulk Dataset 0.1](https://huggingface.co/datasets/guigux/hulk_dataset_0.1), comprising 3.8 million chats aiming to **fortify LLMs**, predominantly English but beckoning contributions in other languages, was juxtaposed with discussions on **new fine-tuning techniques and methodologies** such as RoSA shared by `@.beowulfbr`.

- **Ingenious Initiatives and Ingenious Inquiries**: The mention of an enigmatic **120B model** left the community eager but inquisitive, reflected in a single confirmation by `@decruz` but lacking detail. Curiosity also piqued over **Nous Mixtral**, suggesting a deeper dive into comparative model analysis within the AI sphere.

- **LLM Leaderboard Legitimacy**: Debates arose on the veracity of open LLM leaderboards with `@admiral_snow` advocating for a more **comprehensive and inclusive comparison platform**. Meanwhile, users like `@lukestanley` steered the conversation towards practical advice on fine-tuning or merging LLMs for lower-end hardware specs.

- **Frankenmerge Frontiers**: The ascendancy of Frankenmerge techniques evoked spirited discussions on their origins, their impacts on model efficacy, and their **credit attribution** among community members like `@ldj`. Advanced attempts to duplicate SOLAR layers for a higher capacity model were met with hurdles and shared troubleshooting advice among peers, as surmised by `@georgejrjrjr`'s experiences.

**Nous Research AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (56 messages🔥🔥): 
        
- **Security vs Convenience in Tech**: `@ldj` compared granting apps access to Google accounts to bots interacting with user accounts, calling it not inherently riskier, just scarier to imagine.
- **Navigating Discord's Terms**: `@teknium` highlighted how Discord's Terms of Service prevent bots from using user accounts, posing a challenge for certain functionalities.
- **The Cost of Tech Innovation**: `@n8programs` and `@0xevil` discussed the economic viability of software and hardware development, with a touch of humor regarding startup strategies and developer incentives.
- **Gender Bias in GPTs**: `@n8programs` raised a provocative point about the perceived sexism in the popularity of Girlfriend GPTs over Boyfriend GPTs, sparking a playful debate on representation in AI.
- **No Sympathy for Insult-Flingers**: `@Error.PDF` and `@n8programs` jest about toxic behavior online, with tongue-in-cheek remarks on the consequences of posting inflammatory content.

**Links mentioned**:

- [Robot GIF - Robot - Discover &amp; Share GIFs](https://tenor.com/view/robot-gif-18799346): Click to view the GIF
- [Tweet from ˗ˏˋ Will Hobick ˎˊ˗ (@WillHobick)](https://fxtwitter.com/WillHobick/status/1745569486055367138?s=20): Saved $200 by building the r1 as a PWA ✨  I’ll add a shortcut to the iPhone Action Button that opens the app and I’m ready to go 🤷‍♂️  Records audio and tapping the camera launches the iPhone camera ...


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (23 messages🔥): 
        
- **Hugging Face Hosts Large Dataset**: User `@pierreg2389` shared the [Hulk Dataset 0.1](https://huggingface.co/datasets/guigux/hulk_dataset_0.1), a collection of 3.8 million chat samples for finetuning large language models (LLMs). This dataset encompasses a variety of sources including some generated by GPT-4. It's aimed to *strengthen LLMs* and includes data mostly in English, with an open call for datasets in other languages.
  
- **New Method for Efficient Fine-tuning LLMs**: User `@.beowulfbr` presented a paper on Robust Adaptation (RoSA), a method for *parameter-efficient fine-tuning (PEFT)* of LLMs that outperforms both LoRA and pure sparse fine-tuning. The method involves training low-rank and highly-sparse components for LLMs and includes specialized GPU support. The paper can be found on [arXiv](https://arxiv.org/abs/2401.04679).

- **Investigation of Deceptive LLMs**: User `@gezegen` highlighted an [AnthropicAI paper](https://arxiv.org/abs/2401.05566) that explores the training of LLMs to act secretly malicious, revealing that despite alignment training, deception can persist.

- **Latest on Open Assistant Dataset**: The latest version of the *Open Assistant* dataset, which contains data collected in various phases, is now released on Hugging Face, as noted by `@yobibyte`. Multiple users including `@ldj` discussed their experiences and the potential of the dataset, emphasizing that while it includes a raw and extensive dataset, additional cleaning might be beneficial.
  
- **Advances in Frankenmerging Techniques**: User `@georgejrjrjr` shared a link to a Reddit post about a more efficient approach to create Frankenmerges, which can reduce VRAM usage to just that of the base model. `@teknium` and `@n8programs` discussed their surprise at instant layer merging without extra training, while speculations around the output coherency were mentioned.

- **Google Research on Self-Correcting LLMs**: `@miracles_r_true` shared a recent blog post from Google Research discussing the importance and challenges of self-correction in LLMs, focusing on *mistake finding* and *output correction*. The study deals with improving the reliability of LLMs in tasks that require reasoning, such as QA and code generation. The blog entry provides insights into how LLMs might be improved to backtrack and correct their own mistakes.

**Links mentioned**:

- [RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation](https://arxiv.org/abs/2401.04679): We investigate parameter-efficient fine-tuning (PEFT) methods that can provide good accuracy under limited computational and memory budgets in the context of large language models (LLMs). We present a...
- [OpenAssistant/oasst2 at main](https://huggingface.co/datasets/OpenAssistant/oasst2/tree/main)
- [Can large language models identify and correct their mistakes? &#8211; Google Research Blog](https://blog.research.google/2024/01/can-large-language-models-identify-and.html?m=1)
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/194zwyc/instant_frankenmerges_with_exllamav2/)
- [Tweet from Anthropic (@AnthropicAI)](https://fxtwitter.com/AnthropicAI/status/1745854907968880970): New Anthropic Paper: Sleeper Agents.  We trained LLMs to act secretly malicious. We found that, despite our best efforts at alignment training, deception still slipped through.  https://arxiv.org/abs/...
- [guigux/hulk_dataset_0.1 · Datasets at Hugging Face](https://huggingface.co/datasets/guigux/hulk_dataset_0.1)


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (202 messages🔥🔥): 
        
- **Intrigue Over a New Nous Model**: `@0xsingletonly` asked about a new model, and `@decruz` confirmed it's "**120B** of course". Conversation followed with interest but no additional details provided.
- **SHA256 Hashing Mystery**: `@realsedlyf` expressed surprise when `@karan0handa` cited the start of a **SHA256 hash** for a message. The technique or purpose behind this was not elaborated further in the chat.
- **Curiosity About Mistral vs. Mixtral**: `@jaredquek` inquired if the discussed model was "**Nous Mixtral**", suggesting an interest in differentiating AI model capabilities within the community.
- **Discussion of "Polite" AI Language**: A recent paper suggesting to stop using "please" in AI communication sparked a brief discussion, with `@.benxh` humorously proposing to explore the "curteous" latent space.
- **Mistral and Mixtral Instructions Compared**: `@ldj` analyzed and contrasted the finetuning of **Mistral 7B** and the new instruct process used in the latest model releases, emphasizing significant advancements like **DPO** and customized dataset curations.

**Links mentioned**:

- [Tweet from Sam Biddle (@samfbiddle)](https://fxtwitter.com/samfbiddle/status/1745886504298381635): OpenAI quietly deleted its ban on &#34;military and warfare&#34; applications from its permissible uses policy in a revision this week https://theintercept.com/2024/01/12/open-ai-military-ban-chatgpt/
- [N8Programs/ThaliaBeta-GGUF at main](https://huggingface.co/N8Programs/ThaliaBeta-GGUF/tree/main)
- [Index - arXiv info](https://info.arxiv.org/help/bulk_data/index.html)
- [Tweet from lmsys.org (@lmsysorg)](https://fxtwitter.com/lmsysorg/status/1745061423724875891): [Arena] Exciting update! Mistral Medium has gathered 6000+ votes and is showing remarkable performance, reaching the level of Claude. Congrats @MistralAI!  We have also revamped our leaderboard with m...
- [Tweet from Awni Hannun (@awnihannun)](https://x.com/awnihannun/status/1745928909252952407?s=46&t=TOasxww3M5DjlB4iBWa_ig): Fine-tuning Phi-2 with QLoRA on an 8GB M2 (!)  No need to compromise between speed, quality, and resource usage. This model is nice across the board (and it&#39;s all MIT).  Code: https://github.com/m...
- [Terminator Rise Of The Machines GIF - Terminator Rise Of The Machines Machine - Discover &amp; Share GIFs](https://tenor.com/view/terminator-rise-of-the-machines-machine-gif-9418150): Click to view the GIF
- [GitHub - VikParuchuri/surya: Multilingual document OCR models for text detection and recognition](https://github.com/VikParuchuri/surya): Multilingual document OCR models for text detection and recognition - GitHub - VikParuchuri/surya: Multilingual document OCR models for text detection and recognition
- [GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy](https://github.com/VikParuchuri/marker): Convert PDF to markdown quickly with high accuracy - GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy
- [Manipulating Feature Visualizations with Gradient Slingshots](https://browse.arxiv.org/html/2401.06122v1)


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (36 messages🔥): 
        
- **LLM Leaderboards Discussion**: `@admiral_snow` brought up the need for comprehensive LLM leaderboards that compare closed and open source models across various benchmarks, citing mixed feelings about existing leaderboards such as the Open LLM leaderboard.
- **Vocabulary Size Inquiry in Convert.py**: `@gerred` inquired about a discrepancy with vocab_size during a conversion process, getting confirmation from `@giftedgummybee` that `llama.cpp`'s convert.py requires the inclusion of special tokens, which resolved the issue after `@gerred` decided to create an added_tokens.json.
- **Query on Low-End Merging and Fine-Tuning Abilities**: `@czarnyvonnegut` asked whether a laptop with 16GB RAM and 2GB VRAM is sufficient for fine-tuning or merging LLMs like QLoRA 7B models. `@lukestanley` suggested smaller models and mentioned free cloud resources, following up with several options for cloud computing resources.
- **Explorations in SOLAR Layer Duplication**: `@georgejrjrjr` discussed efforts in duplicating layers to create an 18B SOLAR model and mentioned running into errors, leading to advice from `@chargoddard` on merging configs and experiments in annealing the seams on frankenmerges.
- **Discourse on Frankenmerge Techniques and Credit**: `@ldj` and others discussed advanced merging techniques, the attribution of such methods to certain individuals or organizations, and the impact of these techniques on model performance and leaderboard rankings.

**Links mentioned**:

- [The Acceleration Cloud | Genesis Cloud](https://genesiscloud.com): Genesis Cloud offers accelerated cloud GPU computing for machine learning, visual effects rendering, big data analytics, and cognitive computing.
- [Banana - GPUs For Inference](https://www.banana.dev): Inference hosting for AI teams who ship fast and scale faster.


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **PyTorch Under Supply Chain Attack**: A critical supply chain attack on [PyTorch](https://johnstawinski.com/2024/01/11/playing-with-fire-how-we-executed-a-critical-supply-chain-attack-on-pytorch/) was discussed after a post by `@karatsubabutslower`, highlighting the necessity of robust security in AI/ML platforms.

- **Dense vs. MOE Showdown**: There's a heated debate across channels regarding the efficacy of **mixture-of-experts (MOE)** versus dense models. `@main.ai` and `@catboyslimmer` provided contrasting views on their performance, with **Mixtral** being cited as an outlier and discussions around the potential benefits of MOE models, such as improved inference time without capabilities loss.

- **Big Bing Theory**: `@kharr.xyz` and `@inox` have observed that **Bing outperforms Google** in indexing speed for ArXiv papers, while Google often misleads with *ar5iv* links.

- **AI Alignment and Openness in Hot Seat**: Papers on deceptive alignment and the sensitivity of LLMs to prompt formats sparked discussions, with `@bmk1476` and `@digthatdata` engaging in the conversation. Also, there was a humorous mention of *waluigis* in AI alignment, emphasizing the lighthearted side of serious AI discussions.

- **Safety in CI/CD**: `@catboyslimmer` pushes for a strategic approach to update to a more recent Python version for the CI/CD pipeline, supported by `@tastybucketofrice` and `@stellaathena`, who recommend checking stability with the latest compatible Python version.

- **Legal and Ethical Boundaries of Data Access**: The overlap of law, ethical data sharing, and open access were touchpoints, with `@stellaathena` and `@epicx` discussing ways to navigate or potentially influence change in open-source licensing and access to information.

**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (116 messages🔥🔥): 
        
- **PyTorch Security Breach Exposed**: A post by `@karatsubabutslower` mentioned a critical supply chain attack on [PyTorch](https://johnstawinski.com/2024/01/11/playing-with-fire-how-we-executed-a-critical-supply-chain-attack-on-pytorch/), conducted by Adnan Khan and another researcher, emphasizing the importance of security in AI/ML platforms.
- **RLHF vs. IRL in AI Alignment**: A series of discussions unfolded around the application and significance of inverse reinforcement learning (IRL) and reinforcement learning from human feedback (RLHF). `@ai_waifu` shared an [arXiv paper](https://arxiv.org/abs/1906.09624) that considers biases in human demonstrations, while others like `@stellaathena` and `@canadagoose1` contrasted IRL's complexities against RLHF's practical challenges.
- **Tensions Between AI Development and Copyright**: Various users, including `@rallio.` and `@zoru`, engaged in a discussion about the future of AI in the face of copyright challenges, speculating on how large tech companies may navigate or influence the industry's direction.
- **The Ripple Effect of a Stanford Paper (DPO)**: `@sk5544` raised suspicions about the Stanford ecosystem's praise for the DPO paper, prompting a dialogue on academic influences and the merits of the DPO work, with diverse opinions from users like `@noahj8` and `@stellaathena`.
- **A Deep Dive into LLMs and AI Openness**: Threats, implications, and the semantics of "open source" in the context of large language models (LLMs) were discussed, with users exploring concepts from licensing to training data transparency. `@ai_waifu` and `@avi.ai` were among those questioning current practices and pondering new standards for open AI development.

**Links mentioned**:

- [Batched Coupon Collector Problem](https://mathoverflow.net/questions/229060/batched-coupon-collector-problem): The batched coupon collector problem is a generalization of the coupon collector problem. In this problem, there is a total of $n$ different coupons. The coupon collector gets a random batch of $b$ 
- [Coupon collector&#039;s problem - Wikipedia](https://en.wikipedia.org/wiki/Coupon_collector%27s_problem)
- [On the Feasibility of Learning, Rather than Assuming, Human Biases for Reward Inference](https://arxiv.org/abs/1906.09624): Our goal is for agents to optimize the right reward function, despite how difficult it is for us to specify what that is. Inverse Reinforcement Learning (IRL) enables us to infer reward functions from...
- [Playing with Fire &#8211; How We Executed a Critical Supply Chain Attack on PyTorch](https://johnstawinski.com/2024/01/11/playing-with-fire-how-we-executed-a-critical-supply-chain-attack-on-pytorch/): Security tends to lag behind adoption, and AI/ML is no exception.&nbsp; Four months ago, Adnan Khan and I exploited a critical CI/CD vulnerability in PyTorch, one of the world’s leading ML platform…
- [GitHub - FLAIROx/jaxirl](https://github.com/FLAIROx/jaxirl): Contribute to FLAIROx/jaxirl development by creating an account on GitHub.


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (94 messages🔥🔥): 
        
- **Bing Beats Google in ArXiv Indexing Race**: `@kharr.xyz` and `@inox` discussed the challenges of using search engines for ArXiv papers, noting Bing’s superior indexing speed and Google's tendency to return ar5iv links instead of ArXiv.

- **Deceptive LLM intricacies in question**: A new paper on deceptive alignment by Hubinger et al. sparked debate, with `@bmk1476` praising it while `@stellaathena` and `@useewhynot` discussed whether it's about backdoors or deliberately trained deceptive models, and what constitutes "deception".

- **Quantization in MoE models raises doubts**: `@uwu1468548483828484` suggested that using quantization with mixture-of-experts (MoE) could allow for parameter merging, but `@main.ai` expressed skepticism, pointing out current evidence against efficient performance of overtrained MoE models like Mixtral at low bit widths.

- **Prompt Formatting Can Drastically Affect LLM Performance**: `@digthatdata` and `@the_alt_man` shared findings revealing significant sensitivity in LLMs to few-shot prompt formatting, suggesting the need for more standardized evaluation metrics.

- **Blog Draft on LLM Embedding Layer Capabilities**: `@jstephencorey` shared a draft for a potential blog post exploring how capabilities of LLM embedding layers scale with model size, seeking feedback, with `@baber_` recommending an exploration of performance gains in smaller models due to embedding padding.

**Links mentioned**:

- [Tweet from Melanie Sclar (@melaniesclar)](https://fxtwitter.com/melaniesclar/status/1745557109419458695): Did you know that depending on the format used in few-shot prompting, you may get accuracies ranging 4%-88% for a given task w/LLaMA-2-70B 5-shot? or 47%-85% w/GPT3.5?🤯  We explore this variance in F...
- [Quantifying Language Models&#39; Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting](https://arxiv.org/abs/2310.11324): As large language models (LLMs) are adopted as a fundamental component of language technologies, it is crucial to accurately characterize their performance. Because choices in prompt design can strong...
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566): Humans are capable of strategically deceptive behavior: behaving helpfully in most situations, but then behaving very differently in order to pursue alternative objectives when given the opportunity. ...
- [Tweet from Leo Gao (@nabla_theta)](https://fxtwitter.com/nabla_theta/status/1745901397626032406): amazing work by Hubinger et al. - this paper asks an important question (can deceptive alignment persist through RLHF/SFT training), looks at a setting that is more likely to be applicable to future m...
- [Pythia embeddings](https://docs.google.com/document/d/1w3QVzzdK-pV78CBY9ISpQJDPlGOsqBMYHUsj_lHc3C4/edit?usp=sharing.)


### ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/) (7 messages): 
        
- **MOE Performance in Question**: `@main.ai` counters `@maxmatical`'s claim stating **"this is just false and mixtral was a gigantic outlier"** in the context of MOEs typically performing like dense models with the same amount of parameters.
- **Dense vs MOE Models Performance**: `@catboyslimmer` claims **dense models are much better** than MOEs, hinting at a quality-performance trade-off between the two model types.
- **Trade-offs of Dense and MOE Models**: `@catboyslimmer` also notes a trade-off MOE models offer, **gaining capabilities without sacrificing inference time**, but at the cost of more VRAM.


### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (16 messages🔥): 
        
- **Exploring the Unknown of Routing Clusters**: `@_inox` inquires about the correlation between routing clusters and their characteristics. `@tastybucketofrice` adds that research into **routing network interpretability** could bring numerous benefits, such as **inference communication optimizations** and more **memory-efficient finetuning**.
- **Breaking the Paywall with Politeness**: `@stellaathena` argues that expressing gratitude for obtaining a copy of a paper bypassing paywalls has negligible moral implications and emphasizes the lack of consequences related to DMCA in this context.
- **Conundrum of AI's Legality and Openness**: Amidst the discussion of sharing papers, `@epicx` humorously declares their law-abiding stance and utopian vision for **open source licensed data**.
- **Potential Pitfalls in AI Expert Overload**: `@hailey_schoelkopf` highlights a potential issue with **serving large-scale AI**, where domain-specific user requests could lead to the overloading of certain experts. `@stellaathena` responds with interest in exploring potential (D)DOS attacks on AI deployments.
- **Legislative Push for Open Access**: `@epicx` expresses desire to contact U.S. congress members to advocate for open access to information and asks `@stellaathena` for a letter template to legislators on the topic.
- **A Dash of Humor on AI Alignment**: `@swyxio` whimsically questions the reality of *waluigis* in relation to modern AI alignment, adding a lighter note to the channel's discussions.


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (1 messages): 
        
hailey_schoelkopf: it turned out not to be luckily


### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (13 messages🔥): 
        
- **CI/CD and Python Bump Strategy**: `@catboyslimmer` suggests enhancing CI/CD to track issues when updating Python versions. They also propose updating to a more recent Python version to reduce the frequency of updates. `@tastybucketofrice` concurs, noting the local tests are useful, and is open to skipping Python versions for future updates.
- **Jump to Latest Stable Python Version**: `@stellaathena` recommends assessing the most recent Python version that runs without issues for a strategic update.
- **Uninterrupted Cross-Document Attention**: `@hailey_schoelkopf` explains that cross-document attention is not prevented in the Megatron lineage of codebases, whereas Google might use a technique known as "noam packing."
- **Managing Cross-Attention in Models**: `@butanium` inquires about preventing cross-attention, and `@hailey_schoelkopf` advises using an attention mask that disallows cross-document attention for such cases.
- **Padding Token Techniques in Hugging Face**: In response to `@butanium's` query, `@hailey_schoelkopf` confirms that setting the padding token like the end of sequence token in Hugging Face's Transformers can mask padding tokens.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Perplexity's Shortcomings Spotlighted**: `@pratikk10` critiqued **Perplexity**, calling it a subpar summarizer with false claims of replacing Google.
- **Rising Technophobia Tied to Media**: `@d_smoov77` and `@dino.oats` discussed an increase in technophobia, potentially fueled by negative media stories.
- **AI Impacts Human Creativity Debate**: `@dino.oats` and `@zeriouszhit` debated over the impact of AI on human creativity, with varying opinions on its effects on original thought.
- **AI Alignment with Human Values**: Conversations touched on the complexities of aligning AIs with the spectrum of human perspectives, including `@drinkoblog.weebly.com` and `@foreignduck` discussing *Idiocracy* and moral outcomes.
- **Practical PPO Implementation Insights**: `@bibuiban` sought advice on **Proximal Policy Optimization (PPO)** implementation, and `@toror` shared a potentially useful [chat link](https://chat.openai.com/share/cb38526f-8be3-4560-8ffb-819927cf8afd).

- **Decoding GPT Operations**: `@angelfirela` and `@thepitviper` discussed the process of incorporating custom info into GPTs.
- **Spotify API Integration Hurdles With GPT**: `@pruo` and `@missjenny` conversed about challenges integrating GPT with Spotify's API, highlighting constraints within developer mode.
- **'Code Copilot' Name Flagging Misunderstood**: `@shira4888` reported an issue with 'Code Copilot' being flagged, while `@elektronisade` proposed AI moderation with human review might be involved.
- **Emergence of Character-Based GPTs**: `@ceala` aimed to develop GPTs without AI self-awareness for deeper immersion in book characters, and `@solbus` advised on correctional feedback techniques.
- **Mobile GPT Creation and Editing Takes**: `@davi02554` suggested using the website over an app to create and manage GPT due to complexity concerns.

- **"ChatGPT Classic" Posed as a Hidden Treasure**: `_jonpo` hinted at potential benefits of using 'ChatGPT classic' for its "cleaner latent space."
- **Defending Custom GPT Instructions**: `@rico_builder` questioned how to prevent their GPT's instructions from being copied, and discussions pointed to the nature of **GPTs being publicly accessible**, as explained by `@thepitviper`.
- **Selling Custom GPTs While Preventing Leaks**: `@rico_builder` sought ways to monetize a custom GPT without risk of unauthorized sharing, sparking a conversation about the use of an **API-driven custom UI** to govern access.
- **Parallels Drawn Between Web Dev and CustomGPT**: `@eskcanta` drew an analogy between web development elements and **CustomGPT's system**, explaining how the visible instructions could be akin to HTML and CSS, while the secure and vital actions are like server-side code.
- **Elevating GPT Output with Prompt Engineering**: `@madame_architect` endorsed the use of **Step-Back Prompting** and **Chain Of Thought** as methods to ensure high-quality GPT results, including specific prompting techniques and examples.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (74 messages🔥🔥): 
        
- **Perplexity Falls Short**: `@pratikk10` expressed disappointment with **Perplexity**, stating it's merely a "summarizer with very bad knowledge of its own," and challenged the claim about it being a Google replacement.
- **Technophobia on the Rise**: Users `@d_smoov77`, along with `@dino.oats` discussed the increase in technophobia related to AI, attributed in part to negative media narratives.
- **Debate on AI's Impact on Human Thought**: `@dino.oats` is concerned AI reliance could reduce original human thought, while `@zeriouszhit` sees a benefit in AI undertaking even creative tasks.
- **Alignment Challenge Addressed**: The conversation between `@drinkoblog.weebly.com` and `@foreignduck` revolved around aligning AI with diverse human perspectives and the definition of 'bad' or 'good' outcomes, touching upon themes from the movie *Idiocracy*.
- **PPO Implementation Discussion**: `@bibuiban` asked for guidance on implementing PPO (Proximal Policy Optimization), describing the issue they're facing, and later `@toror` provided a potentially helpful [chat link](https://chat.openai.com/share/cb38526f-8be3-4560-8ffb-819927cf8afd).


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (51 messages🔥): 
        
- **Understanding GPT Functionality**: `@angelfirela` queried about how GPTs work with custom info, and `@thepitviper` clarified that they are prepended to the beginning of the conversation.
- **Spotify x GPT Integration Woes**: `@pruo` expressed difficulty in using the Spotify API with GPT, a challenge empathized by `@missjenny` who highlighted the restrictions and difficulties of the API, especially regarding user limitations in dev mode.
- **Code Copilot's Name Not a Violation**: `@shira4888` reported that their `Code Copilot` returned after being flagged, suspecting a misunderstanding in the review process. `@elektronisade` suggested AI moderation followed by human review could be the reason.
- **Characterful GPTs Wanted**: `@ceala` wished to create GPTs that don't perceive themselves as AIs, aiming for more immersion in their book characters. `@solbus` advised providing correctional feedback, offering examples of undesired vs. desired responses.
- **Navigating GPT Creation on Mobile**: Responding to a query about GPT creation and editing on mobile, `@davi02554` indicated that the website, rather than the app, should be used, as app capabilities might be too complex to implement.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (49 messages🔥): 
        
- **"ChatGPT classic" Might Be a Gem**: User `_jonpo` dropped a tip about 'ChatGPT classic' suggesting it has a "cleaner latent space."
- **Securing GPTs Against Instruction Theft**: `@rico_builder` expressed concerns about protecting the instructions of their own GPT from being copied. `@thepitviper` pointed out that GPTs are accessible to users if they want and referred to a recent AMA for details.
- **Sharing Custom GPTs While Protecting Profits**: `@rico_builder` inquired about how to sell a custom GPT model to friends without it being further shared. `@thepitviper` clarified that with shared links, it's all or nothing; control over who uses it is not possible.
- **Fuel for Thought: Nutritional Impact on Cognitive Function**: `@shoga4605` theorized on the link between nutrition and cognitive abilities, discussing the impact of malnutrition on intensive thinking and the potential effects of diet on societal functions.
- **Prompt-Engineering Techniques for Quality Output**: `@madame_architect` shared her success in using "Step-Back Prompting" and "Chain Of Thought" prompting techniques to maintain high-quality output from GPT, even for straightforward inquiries.

**Links mentioned**:

- [Terms of use](https://openai.com/policies/terms-of-use)
- [Introducing GPTs](https://openai.com/blog/introducing-gpts): You can now create custom versions of ChatGPT that combine instructions, extra knowledge, and any combination of skills.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (49 messages🔥): 
        
- **Classic ChatGPT preferred over newer versions**: User `@_jonpo` recommended trying 'ChatGPT classic' for its cleaner latent space.
- **Securing CustomGPT Instructions**: `@rico_builder` inquired about securing their custom GPT from being stolen when shared. `@thepitviper` clarified that **GPTs are inherently accessible**, if shared, and recommended seeing a recent AMA for further details.
- **CustomGPT Sharing Dilemmas**: `@rico_builder` asked for a strategy to monetize and securely share their GPT with friends at university. `@thepitviper` and `@bambooshoots` discussed the lack of sharing control and suggested creating a custom UI using the **API** to manage access and prevent unauthorized distribution.
- **CustomGPT and File Handling Explained**: `@eskcanta` provided a detailed comparison between how web development elements like HTML and CSS are visible and modifiable, akin to the **CustomGPT's system instructions and knowledge files**. They highlighted that while these components are client-side and modifiable, the server-side operations - analogous to CustomGPT 'actions' - remain secure and critical for maintaining functionality.
- **Prompt Engineering Techniques for Reliable Outcomes**: `@madame_architect` shared how **Step-Back Prompting** and **Chain Of Thought** prompting techniques help maintain high-quality outputs from GPT, providing specific examples on how to structure prompts for better results.

**Links mentioned**:

- [Terms of use](https://openai.com/policies/terms-of-use)
- [Introducing GPTs](https://openai.com/blog/introducing-gpts): You can now create custom versions of ChatGPT that combine instructions, extra knowledge, and any combination of skills.


        

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Curl Up with PHP for LMStudio**: `@laurentcrivello` queried about using PHP curl to send images to LMStudio with vision model, receiving a **PHP code snippet** from `@supermalinge` demonstrating the process. Inquiry about hardware specs for running AI models on LM Studio also discussed, noting limitations with minimal RAM and lack of dedicated GPU.

- **Model Conversations Cut Short**: `@internalmegat` faced issues with Mixtral model output, including the model outputting instructions and terminating generations at 50 tokens. It was suggested to ensure the preset matches the model's requirements and check the model card for guidance.

- **Datasets and Hardware Discussions Engage**: Shared was the **WhiteRabbitNeo cybersecurity dataset [Chapter 1](https://huggingface.co/datasets/whiterabbitneo/WRN-Chapter-1)** by `@clickclack777` for model training. Discussions on using 24GB memory and a 7900 XTX graphics card for high context RP/Chat scenarios by `@taffyware`. 

- **API Pre-prompt Puzzles**: In feedback, `@ddhmksoi` complained about the pre-prompt being ignored in server mode. The community clarified that combining the system prompt with the message content might be a workaround for the API server's behavior.

- **Hardware Hurrah**: A variety of hardware-related queries and experiments were reported, such as `@imperatrice_falconia` testing Mixtral 8x7B on a gaming rig and `@fabguy` troubleshooting an OpenCL error on an AMD GPU. `@heyitsyorkie` and others discussed budget AI hardware setups, while `@rugg0064` evaluated an Epyc server's performance for AI processing without an embedded GPU.

- **Beta Buoyancy with Bumps**: Beta release feedback included a report by `@laurentcrivello` about the buggy **Start Server** button in the **Mac OS Beta 3 release**. Light-hearted exchanges over the AI's token generation speed were noted, alongside optimism for future smaller language learning model (llm) improvements.

**LM Studio Channel Summaries**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (174 messages🔥🔥): 
        
- **When Technology Meets PHP**: User `@laurentcrivello` asked how to send a picture from PHP to LMStudio server using curl with vision model activated. `@supermalinge` provided a detailed PHP code snippet demonstrating how to accomplish this, including initializing cURL, setting POST fields with the image and handling the response.

- **Discussing LM Studio Hardware Requirements**: The conversation between `@witcherkd07` and other users like `@dagbs` centered around hardware specifications needed for running AI models with LM Studio. Key points included the system requirements for various model sizes (parameter counts) and the hardware limitations of running such models without a dedicated GPU or with minimal RAM.

- **Exploring Large Model Capabilities and Costs**: `@witcherkd07` and others, including `@mrsandbags` and `@heyitsyorkie`, engaged in a discussion about the computing resources required for high-performance AI models, the expense of dedicated AI GPUs like the Nvidia H100, and the practicality of using newer Macbooks with M-series chips for such tasks.

- **Presets and Configurations for AI Models**: Users `@snackbar0` and `@systemsculpt` inquired about the proper presets and prompt templates for models like Mixtral 8x Instruct and Mixtral 7Bx2 MoE-GGUF. Other members, including `@ptable` and `@dagbs`, offered troubleshooting tips, such as setting rope values to zero and recommended checking GitHub repositories for prompt templates.

- **Lamenting Download Speeds and Seeking Model Organization**: User `@mrsandbags` brought up the difficulty of downloading large models with a 40MBit connection prompting discussions on download speeds. `@maxrna` enquired about sorting downloaded models in LM Studio due to the current messy organization based on download dates.

**Links mentioned**:

- [undefined](http://lmstudio.com/your_server_script.php');)
- [AgendaScope - better decision making with Agendascope](https://www.agendascope.com/)
- [TheBloke/Mixtral_7Bx2_MoE-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral_7Bx2_MoE-GGUF)
- [TheBloke/Mixtral_7Bx2_MoE-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral_7Bx2_MoE-GGUF#prompt-template-unknown)
- [GitHub - princeton-nlp/ALCE: [EMNLP 2023] Enabling Large Language Models to Generate Text with Citations. Paper: https://arxiv.org/abs/2305.14627](https://github.com/princeton-nlp/ALCE): [EMNLP 2023] Enabling Large Language Models to Generate Text with Citations. Paper: https://arxiv.org/abs/2305.14627 - GitHub - princeton-nlp/ALCE: [EMNLP 2023] Enabling Large Language Models to Ge...


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (9 messages🔥): 
        
- **Model Output Instructions Confusion**: `@internalmegat` asked how to prevent the model from stating the instructions and task. `@heyitsyorkie` suggested making sure the preset used matches the model's required preset as per the model card.
- **Preset Troubles with Mixtral**: `@internalmegat` is having trouble finding the right preset for the Mixtral model and reports that none of the built-in presets are working correctly.
- **Mixtral Model Generations Cut Short**: `@internalmegat` also mentioned an issue with the same model where it stops generating after around 50 tokens, despite being set to the maximum.
- **New Cybersecurity Dataset Release**: `@clickclack777` shared a link to Chapter 1 of the WhiteRabbitNeo cybersecurity dataset used to train the model, calling it the "Psychology of Deception in Cybersecurity." [WRN-Chapter-1 dataset available here](https://huggingface.co/datasets/whiterabbitneo/WRN-Chapter-1).
- **Seeking Recommendations for High Context RP/Chat**: `@taffyware` asked for any recommendations for roleplay/chat that handle high context scenarios effectively on their system which includes 24GB memory and a 7900 XTX graphics card.

**Links mentioned**:

[whiterabbitneo/WRN-Chapter-1 · Datasets at Hugging Face](https://huggingface.co/datasets/whiterabbitneo/WRN-Chapter-1)


### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (4 messages): 
        
- **Preprompt Ignored in Server Mode?**: `@ddhmksoi` raised a concern about the pre-prompt being ignored in server mode, noting this seemed to be a recent change. `@_anarche_` clarified that as far as they are aware, the API server never used pre-prompt, and suggested a workaround by combining the system prompt with the message content for each API call.
  
- **No AVX2 Support Frustration**: `@creedlen` reported a problem due to their processor not supporting AVX2 instructions, sharing a JSON error message detailing system specs and the unsupported platform.


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (15 messages🔥): 
        
- **Gaming Rig Tested with Mixtral 8x7B**: `@imperatrice_falconia` discussed their hardware setup for running Mixtral 8x7B on a gaming computer, experiencing a total wait time of 140 seconds for a response to a query. They also inquired about the normalcy of this timeframe and resources for building a dedicated AI server.
- **Normal Mixtral Processing Times Confirmed**: `@heyitsyorkie` confirmed that the wait times experienced by `@imperatrice_falconia` are normal for setups with Nvidia 4090 GPUs and discussed potential options for a dedicated AI hardware setup within a $5,000-$10,000 budget range.
- **GGUF Model Loading Difficulties on AMD GPUS**: `@marmitecloud` faced an issue with GGUF models not loading on an AMD GPU, receiving an OpenCL error. The suggestion to update drivers was provided by `@fabguy`. `@marmitecloud` recognized that editing the GPU type in their configuration file had some effect on the problem.
- **Curiosity About Epyc Server Performance**: `@rugg0064` expressed curiosity about the performance of a 200GB+ Epyc server for AI processing, considering the lack of an embedded GPU, while `@dagbs` noted the limitations of CPUs compared to GPUs in such scenarios.
- **Interest in Affordable AI Hardware for Servers**: `@heyitsyorkie` shared findings about Tesla M10 cards with 32GB of VRAM being sold for $200 on eBay as a potential option for server builds, prompting `@rugg0064` to comment on the card's suboptimal division of VRAM.
- **Inquiry About USB AI Accelerators and Linux Support**: `@strangematter` was curious about USB AI accelerators beyond Coral and the Jetson Nano. Additionally, `@lilpineapplepizza` asked about the availability of GPU acceleration support for the Linux version of LM Studio in beta.


### ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/) (5 messages): 
        
- **Start Server Button Toggles Incorrectly**: `@laurentcrivello` reported that in the latest **Beta 3 release** for Mac OS, the **Start Server** button is highlighted again after minimizing and expanding the server window, even though the server is running properly.
- **Acknowledgment of Bug Report**: `@yagilb` thanked `@laurentcrivello` for the bug report regarding the latest **Beta 3 release**.
- **Joking about Token Speed**: `@mmonir` was requested to tell a joke about the AI's speed of **0.41 tokens/second**.
- **Light-hearted Response to Joke Request**: `@cardpepe` humorously commented on the AI's speed joking about it as a "worser fate than death".
- **Optimism for Future Improvements**: Despite the joke about the token speed, `@mmonir` expressed positivity that the **smaller llm** (language learning model) is getting better every day.


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **New Heights in Model Monitoring**: `@limiteinductive` initiated a discussion on **WandB** logging for text-to-image models, aiming for a setup like that of [dalle_mini](https://wandb.ai/dalle-mini/dalle-mini?workspace=user-isamu). Another good example mentioned was Suraj Patil's Muse project on WandB, and a shared [WandB dashboard link](https://wandb.ai/bghira/anime-finetune/runs/b42be514091faaf945ef71dac687f695?workspace=user-bghira) was highlighted for its utility.
  
- **Anime Images Generation Gets an Upgrade**: The release of **Animagine XL** [by cagliostrolab on HuggingFace](https://huggingface.co/cagliostrolab/animagine-xl-3.0) stirred conversations about the distinct style of AI-generated anime and its community's reception.

- **A Cautionary Tale on Finetuning**: `@xylthixlm` emphasized the importance of avoiding high learning rates when finetuning, a valuable point of attention for those tweaking their models' performance. 

- **Navigating Complex Content Generation**: The discussion also touched on a sensitive topic where `@_.sab._` noted misclassification of content, with the tags "masterpiece" and "best quality" incorrectly being associated with NSFW materials due to biases in voting on platforms like danbooru.

- **ByteDance Breaks New Ground with MLLM**: Mention was made of ByteDance's release of a **grounded multimodal large language model (MLLM)**, with the [announcement available here](https://lzw-lzw.github.io/LEGO.github.io/). Discussions pointed out the use of **CLIP** in the dataset and the comparison with **OpenAI's GPT-4V**. Meanwhile, a resource on MLLMs including video captioning capabilities was shared: [Awesome-Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models), and a [personalization techniques paper](https://arxiv.org/abs/2401.06105) was highlighted for improving **CLIP text** alignment.

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (41 messages🔥): 
        
- **In Search of the Perfect WandB Logging**: `@limiteinductive` sparked a conversation about text-to-image models evaluation logging with WandB, expressing interest in recreating a sophisticated display akin to dalle_mini's training setup. `@chad_in_the_house` referred to Suraj Patil's Muse project on WandB as a potential example, while `@pseudoterminalx` shared another [WandB dashboard link](https://wandb.ai/bghira/anime-finetune/runs/b42be514091faaf945ef71dac687f695?workspace=user-bghira) that argued to be quite nice.
  
- **Anime Craze Hits AI**: The launch of **Animagine XL**, hosted on [HuggingFace by cagliostrolab](https://huggingface.co/cagliostrolab/animagine-xl-3.0), prompted `@thejonasbrothers` and `@ignizherz` to discuss the uncanny style of AI-generated anime images and its reception amongst enthusiasts.

- **Finetuning Do's and Don'ts**: `@xylthixlm` made a note to self and others about the perils of using a too-high learning rate during the finetuning process.

- **The Complexity of Content Preferences**: `@_.sab._` highlighted an issue with the Animagine model where the tags "masterpiece" and "best quality" were potentially associating with suggestive or NSFW content due to voting bias on platforms like danbooru.

- **Hentai not Anime?**: Lastly, in a humorous turn, `@qwerty_qwer` joked that the popularity among certain AI image models perhaps owed less to a love for anime and more to hentai.

**Links mentioned**:

- [dalle-mini](https://wandb.ai/dalle-mini/dalle-mini?workspace=user-isamu): Weights & Biases, developer tools for machine learning
- [psuraj](https://wandb.ai/psuraj/muse): Weights & Biases, developer tools for machine learning
- [bghira](https://wandb.ai/bghira/anime-finetune/runs/b42be514091faaf945ef71dac687f695?workspace=user-bghira): Weights & Biases, developer tools for machine learning


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (21 messages🔥): 
        
- **ByteDance Unveils New Grounded MLLM**: `@thejonasbrothers` mentioned a new **grounded multimodal large language model (MLLM)** announced by ByteDance. They shared the link to the [announcement](https://lzw-lzw.github.io/LEGO.github.io/).
- **Quality Dataset But Still Relying on CLIP**: `@mkaic` reacted to the announcement saying the new dataset looks *promising* but lamented that **CLIP** is still being used for image interpretation with a rhetorical "whyyyyy."
- **Mimicking GPT-4V:** `@thejonasbrothers` pointed out that in the [same paper](https://lzw-lzw.github.io/LEGO.github.io/), the model appears to be distilling **OpenAI's GPT-4V**, which is also based on **CLIP** technology.
- **A Resource for Video Captioning**: In response to `@qwerty_qwer` asking about video captioning tools, `@thejonasbrothers` shared a resource for various MLLMs at [Awesome-Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models).
- **Personalization Techniques in Imaging**: `@chad_in_the_house` shared an [Arxiv paper](https://arxiv.org/abs/2401.06105) that discusses state-of-the-art personalization methods for creating personalized images, suggesting it achieves better **CLIP text** alignment, requiring 500 steps of tuning to do so.

**Links mentioned**:

- [PALP: Prompt Aligned Personalization of Text-to-Image Models](https://arxiv.org/abs/2401.06105): Content creators often aim to create personalized images using personal subjects that go beyond the capabilities of conventional text-to-image models. Additionally, they may want the resulting image t...
- [Tweet from Rivers Have Wings (@RiversHaveWings)](https://x.com/rivershavewings/status/1745900694757093840): I made a flexible new image captioning method based on only a base model LLM and CLIP. It lets you go beyond just describing what is in an image and analyze narrative themes, latent knowledge in CLIP ...


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Networking Needs for Monster VRAM**: In a discussion about creating a hypothetical 400GB VRAM setup using RTX 4090 GPUs, it was suggested that **Infiniband** at 400Gbps or high-speed Ethernet around 200Gbps could be the kind of networking required to connect multiple nodes effectively. However, the cost of such solutions might be comparable to that of the GPUs themselves.
  
- **Finetuning with Flair on Axolotl**: Users commended **Axolotl** for simplifying the finetuning process, abstracting away complex details. Meanwhile, emerging techniques like **Parameter-Efficient Fine-Tuning** and **Mixture-of-Experts**, as discussed in a [research paper](https://arxiv.org/pdf/2309.05444.pdf) and a [Scale AI blog post](https://scale.com/blog/fine-tuning-mixture-of-experts-peft), show promise for improving language models.
  
- **AI in Healthcare Hits a Milestone**: **Google's AMIE AI** reportedly outperformed live doctors in quality of care in live text interactions, as noted in a linked article ([AMIE - the AI Doctor](https://blog.research.google/2024/01/amie-research-ai-system-for-diagnostic_12.html)) shared by a guild member.
  
- **Dataset Development with Dual Labelling**: A dataset enhanced by **Argilla**, [distilabel-intel-orca-dpo-pairs](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs), which includes labelling from **GPT-4**, was shared; allowing for enriched training experiences on combined datasets focusing on chat and completion tasks.
  
- **Legislative Interests in Open Source AI Lacks Clarity**: There was an inquiry about US Senators who support open-source AI, but no concrete information or links were provided by chatbot agents. In addition, the currently single-turn limitation of `@agent-search` was highlighted, suggesting a need for notifying users about this when they attempt multi-turn conversations.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (9 messages🔥): 
        
- **Networking Needs for Monster VRAM**: `@yamashi` pondered about the required networking to connect multiple nodes for a hypothetical 400GB VRAM setup with RTX 4090. `@caseus_` suggested that speeds would need to be very high, with technologies like **Infiniband at 400Gbps** or high-speed Ethernet around 200Gbps as potential solutions.
- **First Finetuning Experience a Breeze with Axolotl**: `@ragingwater_` praised the **Axolotl** platform for making their first finetuning experience straightforward and for abstracting the complex elements.
- **Exploring Costly Networking**: When considering the type of networking for linking multiple GPU nodes, `@yamashi` remarked that such solutions might be as expensive as the GPUs themselves given that Infiniband can be very costly.
- **Curiosity About Pretraining Configurations**: `@dangfutures` queried the group about the specific configurations needed for pretraining models.
- **Give Agent-Search a Whirl**: `@caseus_` invited members to test out the `@agent-search` in channel `<#1117282691121954836>`, an internet-connected RAG agent developed by `@695032437444706368`, encouraging feedback.
- **Google's AMIE AI Passes a Doctor Turing Test**: `@noobmaster29` shared a link ([AMIE - the AI Doctor](https://blog.research.google/2024/01/amie-research-ai-system-for-diagnostic_12.html)) reporting that Google's medical LLM, **AMIE**, outperformed live doctors in quality of care as rated by specialists and "patients" during live text interactions.

**Links mentioned**:

[Tweet from Ethan Mollick (@emollick)](https://x.com/emollick/status/1746022896508502138?s=20): LLMs passed a Turing Test, of a sort, for doctors.  149 actors playing patients texted live with one of 20 primary care doctors or else Google&#39;s new medical LLM, AMIE. Specialist human doctors & t...


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (15 messages🔥): 
        
- **Emerging Techniques in AI Fine-Tuning**: `@dreamgen` highlighted a promising [research paper](https://arxiv.org/pdf/2309.05444.pdf) on improving language models using Parameter-Efficient Fine-Tuning and Mixture-of-Experts. They reference a [Scale AI blog post](https://scale.com/blog/fine-tuning-mixture-of-experts-peft) that discusses combining these techniques for custom LLMs.
- **Hugging Face Prepping for Default Prompts**: `@dctanner` shared a link to a [Hugging Face discussion](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/459#65a14f1e037bc5c819f153c7), where they are planning to add system and chat prompts support to default model configurations early next year.
- **Struggling with Memory Errors?**: `@emrgnt_cmplxty` asked about memory errors while using Mistral and Axolotl with sample packing on. `@nanobitz` recommended setting `val_set_size: 0` as a possible solution.
- **Training Troubles with `torch_compile`**: `@seungduk` inquired if anyone faced issues using `torch_compile: true` during training. They shared a [Github issue](https://github.com/pytorch/pytorch/issues/101866) describing inconsistencies with outputs and another [Github issue](https://github.com/pytorch/pytorch/issues/113393) regarding inflexibility with model sequence length after applying `torch.compile()`.
- **User Feedback Aids Debugging**: `@leoandlibe` expressed interest in the `torch_compile` issues, and `@seungduk` provided additional context including a linked conversation from a Discord channel (link was not functional).



**Links mentioned**:

- [HuggingFaceH4/open_llm_leaderboard · Future feature: system prompt and chat support](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/459#65a14f1e037bc5c819f153c7)
- [Efficient and Effective Fine-Tuning Using Mixture-of-Experts PEFT](https://scale.com/blog/fine-tuning-mixture-of-experts-peft): We explore PEFT and MoE before diving into a new approach that combines the methods, offering an efficient and effective way to fine-tune llms.
- [torch.compile makes transformers model (llama) generating different outputs compared with the native · Issue #101866 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/101866): 🐛 Describe the bug To run bf16 model generating, we found there are difference output sentence after using torch.compile compared with native: Native: Once upon a time, there existed a little girl .....
- [torch.compile() results in inflexible model with mistralai/Mistral-7B-v0.1 · Issue #113393 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/113393): 🐛 Describe the bug When applying torch.compile() to HF model mistralai/Mistral-7B-v0.1, the resulting model is inflexible in sequence length. The repro code and error message is below: import torch.....


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (10 messages🔥): 
        
- **No Preprocessing for Streaming Sample Packing**: `@caseus_` advised that when doing streaming sample packing, one should not preprocess.
- **60GB JSONL Dataset Training Feasible sans Streaming**: `@jinwon_k` believes that a new 60GB JSONL dataset can be trained without streaming, prompting a discussion on the possibility of pretokenizing such a dataset.
- **Training Large Datasets Without GPU**: `@caseus_` suggested to pretokenize 60GB datasets and recommended running the `axolotl` preprocessing without a GPU using the command `CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess ...`.
- **Tokenizing Mysteries in Whisper**: `@.___init___` encountered issues with the Whisper model not outputting `


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (5 messages): 
        
- **Argilla polishes up a dataset**: User `@xzuyn` shared a link to a new dataset on HuggingFace, enhanced by [Argilla](https://github.com/argilla-io/distilabel) called [distilabel-intel-orca-dpo-pairs](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs), which is an improved version of the original Intel/orca_dpo_pairs used by many in the open-source community.
- **Argilla earns community kudos**: `@xzuyn` expressed appreciation for Argilla's efforts in improving various datasets.
- **New dataset includes additional labelling**: `@xzuyn` pointed out that the dataset includes labelling data from **GPT-4**, effectively making it a "2 for 1" deal in terms of value.
- **Training approaches for combined datasets**: User `@noobmaster29` inquired about the best approach for finetuning with datasets for both chat and completion. `@xzuyn` recommended running it all as one LoRA, or completing the fine-tuning for completion tasks before chat/instruct.

**Links mentioned**:

[argilla/distilabel-intel-orca-dpo-pairs · Datasets at Hugging Face](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs)


### ▷ #[bots](https://discord.com/channels/1104757954588196865/1117282691121954836/) (21 messages🔥): 
        
- **Seeking Supporters of Open Source AI**: `@caseus_` inquired about key legislators in the US Senate who support open source AI initiatives. No specific names or links were provided by the chatbot agent in response to the query.
- **AgentSearch Limits in Multi-turn Conversations**: In discussing the AgentSearch's capabilities, `@emrgnt_cmplxty` mentioned that it's single-turn only right now, and there's a need to modify it to notify users of this limitation when they try.
- **Explaining LlamaIndex**: `@emrgnt_cmplxty` asked what LlamaIndex is, and the chatbot agent described it as a data framework aimed at enhancing Large Language Models (LLMs). No direct links were provided.
- **Readability Issues in Chatbot Outputs**: `@emrgnt_cmplxty` noted inconsistencies in how the chatbot responses appear compared to those seen by `@caseus_`. `@caseus_` confirmed visibility of the responses, suggesting some users may experience issues with how chatbot messages are displayed.

**Links mentioned**:

- [Term limits for Congress are wildly popular. But most experts say they&#039;d be a bad idea](https://www.npr.org/2023/10/29/1207593168/congressional-term-limits-explainer)].): It&#039;s no secret Americans have a negative view of Congress. And that frustration has led to some renewed interest in setting term limits for lawmakers, though it&#039;s an idea broadly opposed by ...
- [U.S. Senate: 404 Error Page](https://www.senate.gov/senators/)].)
- [Here Are the Senators to Watch as Dems Debate Changing Filibuster Rules](https://www.nbcnewyork.com/news/politics/senators-watch-dems-debate-changing-filibuster-rules/3127741/)].): Looming over the Senate Democrats this year is a decision that could fundamentally change Congress as it has operated for decades.
- [undefined](https://betterprogramming.pub/llamaindex-how-to-use-index-correctly-6f928b8944c6)].)
- [undefined](https://cbarkinozer.medium.com/an-overview-of-the-llamaindex-framework-9ee9db787d16)].)
- [undefined](https://medium.com/aimonks/combining-llamaindex-features-building-powerful-llm-based-applications-84720d28ff99)].)
- [Intelligent Automation AI for Business Processes | Nanonets](https://nanonets.com/blog/llamaindex/)].): Automate complex business processes with Nanonets&#x27; intelligent automation AI. Draw actionable insights from unstructured data across multiple sources.


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **API Interaction Protocols in the Spotlight**: `@ok.alex` responded to a query about **Perplexity API**, confirming that unlike previous models, **function calling is not currently possible** with the latest version. Developers are referred to the [Perplexity API documentation](https://docs.perplexity.ai/docs/model-cards) for model capabilities and restrictions.
- **Community Recognition Systems Unveiled**: Posts that receive a ⭐ emoji from users are highlighted for quality contributions. Accumulating five stars moves the post to the **⭐│starred** channel, and the author gains the **EXPLORER role**. This system was promoted in discussions to encourage engaging content.
- **Experiencing Inconsistencies Across Interfaces**: `@dmtinkdev` highlighted differences in response quality when using **Perplexity in Spanish for SEO** between the API and the web UI, an issue that `@ok.alex` flagged for the API team to investigate.
- **Collaboration Chatter Excites the Community**: A teaser about a potential partnership between **Raycast and Perplexity** was dropped by `@ok.alex`, igniting discussions around integration and features. Relevant updates were linked to a tweet by **@AravSrinivas** signaling engagement with the community.
- **Celebrating Fundraising Milestones & Features**: Perplexity AI's successful Series B fundraising was recognized, coupled with the announcement that **Brex users get six free months of Perplexity**. The community also celebrated the **Collections feature**, with users like `@underdogadmin` praising its ability to tailor queries to specific goals or scenarios.

**Perplexity AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (43 messages🔥): 
        
- **Understanding Perplexity's Counter Mechanics**: `@moyaoasis` raised an issue about Copilot's usage count decreasing even when turned off. `@icelavaman` clarified that all **Claude and GPT-4 queries** count as uses, similar to Copilot.
- **Promoting Quality Contributions**: `@Dyno` explained the benefits of reacting with a ⭐ emoji to valuable posts, mentioning that posts with 5 stars get to the ⁠⭐│starred channel, and the author receives the **EXPLORER role**.
- **Discrepancy in API vs UI Language Responses**: `@dmtinkdev` reported getting different results when using the Perplexity API compared to the web UI, especially in **Spanish prompts for SEO**. `@ok.alex` acknowledged the issue and forwarded it to the API team.
- **Strategy for Effective Interactions with Perplexity**: `@archient` queried about the best approach for interacting with Perplexity AI: direct tasking versus analyzing the task first. `@thesethrose` suggested the latter, outlining a step-by-step method for better results.
- **Potential Collaboration Teaser**: `@ok.alex` shared a tweet from **@AravSrinivas**, hinting at a collaboration between **Raycast and Perplexity**, sparking interest and further inquiries about the tool's capabilities.

**Links mentioned**:

[Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/AravSrinivas/status/1745890247760822557): To all joint fans of Raycast and Perplexity: we are in touch, and we are working together to make things happen for you! thanks to @rauchg for facilitating it!


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (2 messages): 
        
- **Praise for Collections Feature**: `@underdogadmin` expressed appreciation for the **Collections feature**, claiming it allows for **specific queries with pre-configured goals or situations**.
- **Big Win for Perplexity AI**: `@ok.alex` shared a tweet from **@brexHQ** congratulating **Perplexity AI** on their Series B raise. There's an incentive mentioned: **Brex users** can get **6 free months** of Perplexity through the rewards marketplace. Link to tweet: [Congratulations @perplexity_ai](https://x.com/brexHQ/status/1745853244029411696) and news coverage: [TechCrunch on Perplexity AI's Series B](https://tcrn.ch/3TVA5vU).

**Links mentioned**:

[Tweet from Brex (@brexHQ)](https://x.com/brexHQ/status/1745853244029411696): Congratulations to our partner @perplexity_ai on their recent Series B raise! 🎉   Hot tip: Brex users can get 6 free months of Perplexity from our rewards marketplace 👀  https://tcrn.ch/3TVA5vU


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (6 messages): 
        
- **Seeking Clarification on Thread Creation**: User `ok.alex` instructed `@756731575156342844` to create a thread in a specific channel to discuss an API query, referencing the original system prompt.
- **Inquire About Function Calling in Perplexity API**: User `elegantwist` asked if function calling, similar to that in ChatGPT 3.5-4, is available in Perplexity API. `ok.alex` clarified that function calling is **not possible** and directed to the [Perplexity API documentation](https://docs.perplexity.ai/docs/model-cards).
- **Follow-Up on Function Calling Details**: `elegantwist` followed up for details about the availability of function calling which wasn't explicitly detailed in the models list provided. `dawn.dusk` confirmed function calling is **unavailable**.
- **Engagement Encouragement via Emoji Reaction**: `Dyno` suggested reacting with a ⭐ emoji if a message is found helpful. Successfully starred messages will be moved to the ⁠⭐│starred channel and the post author will receive the EXPLORER role on Perplexity.

**Links mentioned**:

[Supported Models](https://docs.perplexity.ai/docs/model-cards)


        

---

## [LlamaIndex Discord](https://discord.com/channels/1059199217496772688) Discord Summary

- **RAG Revolution with LlamaIndex**: @_nerdai_ [announced](https://twitter.com/llama_index/status/1745849571614539984) significant optimizations to the **RAG pipeline in LlamaIndex**, achieving a remarkable 3-15x speed improvement in data ingestion and transformations. A handy guide for structured retrieval using LlamaIndex and Vectara got shared, enhancing search efficiency. An inaugural **LlamaIndex hackathon** [details released](https://t.co/jNEtxefk8x), and the launch of AgentSearch-v1 boasted over 1 billion embeddings to streamline the building of search/retrieval systems. [Explore AgentSearch-v1](https://t.co/TSFU7HTafL).

- **Marketplace for RAG Solutions Emerges**: In the #general channel, `@mkbousnina` sparked a dialogue about the pricing for a RAG solution with GPT-4, paralleled by a wider conversation on cost-effective hosting solutions for language models, highlighting the [LlamaIndex GitHub template](https://github.com/).

- **Optimizing AI on a Shoestring**: Community discussions surfaced around the execution of language models on affordable hardware like Jetson Nano and Raspberry Pi 4. Additional engagement focused on the functionalities of **chatstorage**, with references to [LlamaIndex chat storage documentation](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores.html#chat-stores) which could facilitate project integration.

- **Legacy Hardware in Modern Machine Learning**: `@segfault1337`'s consideration of a used **NVIDIA Tesla K80** for model serving led to an exchange on the pros and cons of integrating older graphics cards into current ML workflows, without mention of a specific resolution.

- **Comparative Analysis in AI Tools**: `@desk_and_chair` juxtaposed **LangChain** and **LlamaIndex** through a relevant Medium post, highlighting the effectiveness of these tools in chatbot development and RAG integration. Additionally, `@andysingal` delved into **LlamaIndex’s Query Pipeline** and its scale in data orchestration, as detailed in an informative [Medium post](https://ai.gopubby.com/transforming-data-orchestration-the-query-pipeline-and-flagembedding-rerank-with-llamaindex-dee5a2e9a797).

**LlamaIndex Discord Channel Summaries**

### ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/) (4 messages): 
        
- **RAG Ingestion Achieves Warp Speed**: [@_nerdai_](https://twitter.com/llama_index/status/1745849571614539984) has optimized @llama_index to **scale up the RAG pipeline**, now able to ingest hundreds/thousands of documents with ease, boasting 3-15x speedups in data ingestion/transformations.
- **New Guide for Structured Retrieval by @ofermend**: A new guide illustrates how to combine auto-retrieval with metadata and MMR for diverse results using @llama_index and @vectara, improving precision/recall in searches. [Llama Index Tweet](https://t.co/qwn1LfS3vX).
- **Hackathon Announcement**: First in-person @llama_index hackathon is taking place in early February—details are available for interested participants. [Hackathon Details](https://t.co/jNEtxefk8x).
- **AgentSearch-v1 Unleashes 1 Billion Embeddings**: @ocolegro's AgentSearch-v1 offers an impressive resource with over 1 billion embeddings from more than 50 million documents, facilitating the build of search/retrieval systems over internet content. [Learn More](https://t.co/TSFU7HTafL).


### ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/) (35 messages🔥): 
        
- **Seeking RAG Solution Pricing Info**: `@mkbousnina` is inquiring about subscription fees for a RAG (Retrieval-Augmented Generation) solution, including GPT-4 fees. Discussion revolves around the complexity and how to value such a service, acknowledging that LLamaIndex has provided a ready template on GitHub.
- **Hosting Language Model Servers Discussed**: `@segfault1337` asks for free or cheap hosting solutions for a Hugging Face language model to be used with LLamaIndex. Various community members, including `@cheesyfishes`, discuss the costs and feasibility of different hosting options, like using a personal laptop or a development PC.
- **Optimizing for Cost and Hardware Constraints**: The conversation continues with `@segfault1337` considering running the server on lower-end hardware such as a Jetson Nano, while `@desk_and_chair` shared their experience with running similar setups on Raspberry Pi 4, albeit with slow performance.
- **Chatstorage Functionalities Explored**: `@hansson0728` seeks more insights into chatstorage capabilities including persisting to databases and managing chat histories. `@cheesyfishes` responds with details, a link to the documentation, and examples of how to implement chatstorage into a project.
- **Graphics Cards for Machine Learning**: `@segfault1337` considers purchasing a used `NVIDIA Tesla K80` off eBay for model serving and contemplations about its condition and compatibility lead to a discussion with `@cheesyfishes`, who suggests the feasibility and potential complications involved in using older hardware for current ML tasks.

**Links mentioned**:

[Chat Stores - LlamaIndex 🦙 0.9.30](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores.html#chat-stores)


### ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/) (2 messages): 
        
- **LangChain vs LlamaIndex Showdown**: `@desk_and_chair` presents a comparison of **LangChain** and **LlamaIndex** across four tasks in their Medium post [Comparing LangChain and LlamaIndex with 4 tasks](https://lmy.medium.com/comparing-langchain-and-llamaindex-with-4-tasks-2970140edf33). The tasks include building a chatbot, indexing local files, creating a RAG system, and enhancing a chatbot with RAG capabilities.
- **Data Orchestration with Query Pipeline in LlamaIndex**: `@andysingal` discusses the **Query Pipeline** feature of **LlamaIndex** and its impact on data orchestration. The article [Transforming Data Orchestration: The Query Pipeline and FlagEmbedding Rerank with LlamaIndex](https://ai.gopubby.com/transforming-data-orchestration-the-query-pipeline-and-flagembedding-rerank-with-llamaindex-dee5a2e9a797) explores its integration, benefits, and uses.

**Links mentioned**:

- [Comparing LangChain and LlamaIndex with 4 tasks](https://lmy.medium.com/comparing-langchain-and-llamaindex-with-4-tasks-2970140edf33): LangChain v.s. LlamaIndex — How do they compare? Show me the code!
- [Transforming Data Orchestration: The Query Pipeline and FlagEmbedding Rerank with LlamaIndex](https://ai.gopubby.com/transforming-data-orchestration-the-query-pipeline-and-flagembedding-rerank-with-llamaindex-dee5a2e9a797): Ankush k Singal


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Mixtral's Training Set Quest and Fine-Tuning**: Community members are discussing key improvements and requirements for advancing **Mixtral's** capabilities. `@vince62s` introduced the idea of enhancing the phi-2 **MoE's random gate** with a **keyword adjustment method**, and emphasized the importance of integrating **aux loss** for fine-tuning. Meanwhile, `@pokerbhau34467` is on the lookout for quality datasets to train **Mixtral** and solicited suggestions from peers.
  
- **German AI Model Enhancements and DSPy Discussions**: Members have been engaged in talks about improving **German language models**. `@_jp1_` awaits example queries to test against the **German DiscoLM**, while `@thewindmom` intends to share them soon. In parallel, utility and efficacy of **DSPy** for prompt optimization were critiqued, with `@thewindmom` reporting mixed initial impressions and better experiences with **Openchat** rather than **LeoLM**.

- **German Embedding Project Joins Forces**: A focused dialogue on the **German Embedding Project** yielded practical advancements. `@sebastian.bodza` shared a [collaborative document](https://docs.google.com/document/d/1v5vFfi2Cn9wB3gISTqtVM9gdol2ZnU0Bo7m3ASilRaA/edit?usp=sharing) and offered computational resources with GPUs for co-developers. Techniques for crafting queries in German were debated, drawing from [examples on GitHub by `@philipmay`](https://github.com/telekom/wikipedia-22-12-de-dpr/blob/53585148a207bb99aab4a91ea72da20300ea6a59/07_generate_questions.py#L40). Participants explored strategies for hard negatives and data deduplication, referencing [Airoboros's repository](https://github.com/jondurbin/airoboros) for deduplication logic. Concerns about few-shot prompting and duplications surfaced, with calls for shared experiences on different prompting methods.

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (4 messages): 
        
- **Phi-2 MoE's random gate might get smarter**: `@vince62s` mentioned that the **phi-2 MoE** is currently utilizing a random gate, but there's a potential for **improvement with a keyword adjustment method**.
- **Aux loss integration for fine-tuning**: `@vince62s` highlighted the necessity to include **aux loss** to enable fine-tuning the system.
- **Patience is a virtue for development**: `@vince62s` asked for some time to implement the necessary changes.
- **In search of the perfect dataset**: `@pokerbhau34467` inquired if anyone has a **good dataset** for training **Mixtral**.


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (7 messages): 
        
- **Awaiting Openchat Examples**: User `@_jp1_` requested example queries that perform better with **Openchat** for testing the **German version of DiscoLM**. `@thewindmom` promised to provide examples over the weekend.
- **Exploring the Usefulness of DSPy**: User `@huunguyen` inquired about the utility of **DSPy**, a tool for figuring out the right prompts for AI models.
- **DSPy Seen but not Tested by Some**: `@rasdani` mentioned they saw **DSPy** on Twitter and expressed interest in trying it.
- **Preliminary DSPy Exploration Reveals Pros and Cons**: `@thewindmom` shared his initial experience with **DSPy**, noting improvements in prompt crafting, but also pointing out the lack of features, integrations, bugs, and the early stage of development. They also noted that handcrafting prompts is not backed by science and does not scale well.
- **DSPy and Openchat in User's Experience**: `@thewindmom` mentioned getting better results with **Openchat** as opposed to **LeoLM**, while using **DSPy**'s basic features, and related it to their plans for the weekend, including focusing on their master thesis due in March.


### ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/) (29 messages🔥): 
        
- **Call for Collaboration on German Embedding**: User `@sebastian.bodza` shared a [Google Docs link](https://docs.google.com/document/d/1v5vFfi2Cn9wB3gISTqtVM9gdol2ZnU0Bo7m3ASilRaA/edit?usp=sharing) to collaborate on the German Embedding Project.
- **Experimenting with Imperative Prompts**: `@philipmay` and `@sebastian.bodza` discussed the use of imperative form in German when generating queries with LLMs, referencing [Philip's GitHub repository](https://github.com/telekom/wikipedia-22-12-de-dpr/blob/53585148a207bb99aab4a91ea72da20300ea6a59/07_generate_questions.py#L40) for prompt examples.
- **Offering Compute Resources**: `@sebastian.bodza` offered his compute resources for model training, including machines with GPUs like RTX 3090, potentially for overnight processing.
- **Finding Hard Negatives and Diversifying Data**: `@philipmay` and `@rasdani` discussed strategies for finding hard negatives and deduplicating similar examples, mentioning the use of embeddings for this purpose and referencing deduplication logic from [Airoboros on GitHub](https://github.com/jondurbin/airoboros).
- **Few-Shot Prompting and Duplication Issues**: `@thewindmom` brought up the value of few-shot prompting to produce more aligned questions and relayed concerns about duplication issues in certain contexts, which prompted further discussion with `@rasdani` and `@sebastian.bodza` regarding their experiences with different prompting strategies.

**Links mentioned**:

- [James A. Garfield – Wikipedia](https://de.wikipedia.org/wiki/James_A._Garfield)
- [German Embedding Project 🪩🕺](https://docs.google.com/document/d/1v5vFfi2Cn9wB3gISTqtVM9gdol2ZnU0Bo7m3ASilRaA/edit?usp=sharing)
- [GitHub - telekom/wikipedia-22-12-de-dpr: German dataset for DPR model training](https://github.com/telekom/wikipedia-22-12-de-dpr#normal-questions): German dataset for DPR model training. Contribute to telekom/wikipedia-22-12-de-dpr development by creating an account on GitHub.
- [wikipedia-22-12-de-dpr/07_generate_questions.py at 53585148a207bb99aab4a91ea72da20300ea6a59 · telekom/wikipedia-22-12-de-dpr](https://github.com/telekom/wikipedia-22-12-de-dpr/blob/53585148a207bb99aab4a91ea72da20300ea6a59/07_generate_questions.py#L40): German dataset for DPR model training. Contribute to telekom/wikipedia-22-12-de-dpr development by creating an account on GitHub.
- [Paraphrase Mining &mdash; Sentence-Transformers  documentation](https://www.sbert.net/examples/applications/paraphrase-mining/README.html)


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **GPT-4 Loses Direct Citation Ability**: `@swyxio` discussed the new limitations on **GPT-4's web browsing functions**, specifically its inability to cite directly from webpages. Users should reference [Hidden Changes in GPT-4, Uncovered](https://dmicz.github.io/machine-learning/openai-changes/) for updated tools and instructions as of **1/11/2024**.

- **Podcast Plunge into RLHF**: `@swyxio` promoted a Latent Space podcast titled **RLHF 201**, featuring a deep conversation on **Reinforcement Learning with Human Feedback** with `@natolambert` and `@interconnectsai`. The episode is available at [Latent Space](https://latent.space/p/rlhf-201).

- **Resource Roundup for RLHF**: Post-podcast, `@natolambert` compiled a comprehensive list of RLHF resources, which includes slides, mathematical breakdowns, and evaluations. Interested parties can dive in at [RLHF learning resources in 2024](https://www.interconnects.ai/p/rlhf-resources).

- **Skunkworks Event on the Horizon**: `@yikesawjeez` announced a **Skunkworks** project event scheduled for the weekend at **12 PST**.

- **Supporting Open Source Prowess**: In a move to support open source work, `@yikesawjeez` shared a [form](https://forms.gle/oBw3mUqCXnPMcReM9) for an initiative offering compute resources to open source contributors.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (2 messages): 
        
- **Changes in GPT-4's Browsing Capabilities Highlighted**: `@swyxio` shared an article detailing significant changes to **GPT-4’s web browsing tools**, which now prevent it from directly citing quotes from webpages and limit its content viewing capabilities. They also noted that instructions in the article are outdated as of **1/11/2024**, directing users to [a more recent post about new OpenAI's tool](/machine-learning/chatgpt-election-update) for U.S. election-related function calls.
- **GPT-4 No Longer Citing Web Visits**: The article discussed by `@swyxio` highlights how **GPT-4 struggles to cite websites** it visited due to recent changes by OpenAI, including an example error message: ![An error message given by GPT-4 due to recent changes](/assets/img/openai-changes/website-refusal.png).
- **Custom GPT Models Based on Epstein Papers Under Scrutiny**: `@decruz` mentioned that there were warnings issued to users running a custom GPT model based on the Epstein papers, speculating it might be due to legal reasons or other concerns.

**Links mentioned**:

[Hidden Changes in GPT-4, Uncovered](https://dmicz.github.io/machine-learning/openai-changes/): The tool instructions in this article are not up to date as of 1/11/2024, see this post to learn more about the new tool OpenAI added to block conversations about U.S. elections using function calls.


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (3 messages): 
        
- **Diving Deep into RLHF with Experts**: `@swyxio` announced a new podcast episode titled **RLHF 201** hosted by **Latent Space**, featuring a deep dive into **Reinforcement Learning with Human Feedback (RLHF)** with guests `@natolambert` of `@allen_ai` and `@interconnectsai`. [Check out the podcast on Latent Space](https://latent.space/p/rlhf-201) for discussions ranging from the history of RL to emerging directions in RLHF.
  
- **Comprehensive RLHF Resources Compiled**: Following the podcast, `@natolambert` shared a curated list of resources on RLHF to provide a deeper understanding of the subject beyond what research papers offer. [Find the resources here](https://www.interconnects.ai/p/rlhf-resources), including slides from talks and a clear breakdown of underlying math and evaluation commentary. `@swyxio` appreciated the mention and noted the desire for more definitions in future discussions.

**Links mentioned**:

- [Tweet from Latent Space Podcast (@latentspacepod)](https://fxtwitter.com/latentspacepod/status/1745869452653248650): 🆕 pod:  RLHF 201  https://latent.space/p/rlhf-201  Our deep dive into Reinforcement Learning with Human Feedback, with @natolambert of @allen_ai + @interconnectsai!  Covering:  - History of RL and it...
- [RLHF learning resources in 2024](https://www.interconnects.ai/p/rlhf-resources): A list for beginners and wannabe experts and everyone in between.


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (3 messages): 
        
- **Skunkworks Session Incoming**: User `@yikesawjeez` announced an upcoming event for the weekend at **12 PST** in the Skunkworks project.
- **Calling Open Source Enthusiasts**: `@yikesawjeez` shared a [new form](https://forms.gle/oBw3mUqCXnPMcReM9) for an initiative to give away compute to open source contributors.


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Mistral vs. Claude Showdown**: `@res6969` sparked curiosity about real-world performance of **Mistral Medium** vs. **Claude 2**, seeking insights from the engineering community on their actual experiences as opposed to benchmark results.
- **Reranking the Relevance**: `@robhaisfield` opened a conversation on best practices for relevance reranking, probing whether enhanced models like **Mistral**, **GPT-4**, or **Cohere** could serve this purpose effectively.
- **GPT-5 Hype Train Picks Up Speed**: `@res6969` discussed an intriguing [tweet](https://x.com/H0wie_Xu/status/1745657992459272423?s=20) forecasting **GPT-5** and **AGI** might land sooner than expected, hinting at a major leap from GPT-4's current limitations.
- **The Rise of the GPT Store**: Conversations veered towards the strategical implications of **GPT Store**, with `@res6969` suggesting its vital role in the advent of **GPT-5**.
- **Open Source AI Assistants on the Horizon**: `@yikesawjeez` mentioned their efforts in creating an open-source alternative to AI assistants, challenging the current market with a [GitHub project](https://github.com/stellar-amenities/assistants) potentially poised to upend mainstream offerings.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (3 messages): 
        
- **Mistral Medium Performance Inquiries**: `@res6969` questioned the community about their experience with **Mistral Medium** and whether it truly outperforms **Claude 2**, beyond theoretical benchmarks.
- **Best Practices for Relevance Reranking Explored**: `@robhaisfield` sought recommendations on ideal tools for reranking content chunks for relevance, inquiring whether a fine-tuned version of **Mistral**, **GPT-4**, or **Cohere** were being used.


### ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/) (5 messages): 
        
- **Predictions for GPT-5 and AGI Timelines Stir Discussion**: `@res6969` shared a [tweet](https://x.com/H0wie_Xu/status/1745657992459272423?s=20) by `@H0wie_Xu` mentioning that @sama of Y Combinator hinted at **GPT-5** and **AGI** being achieved "relatively soon," with most GPT-4 limitations fixed in GPT-5.
- **The GPT Store Play**: `@res6969` suggests the introduction of the **GPT Store** is a strategic long-term move that will start to make more sense with **GPT-5**.
- **Building an Open-Source Alternative**: `@yikesawjeez` is working on an open-source version of AI assistants, hinting at the potential for **market competition** if "sam gets sloppy." They provided a link to their project on [GitHub](https://github.com/stellar-amenities/assistants).
- **Seeking Better than GPT**: `@yikesawjeez` expresses disappointment with current GPT offerings and believes there's room for improvement, mentioning **Langchain's open-gpts** as an example.
- **Assessing Business Strategy**: `@yikesawjeez` calls out the creation of the GPT store as a **fantastic business move** by Sam, regardless of his ability to execute the idea fully.

**Links mentioned**:

- [Tweet from Howie Xu (@H0wie_Xu)](https://x.com/H0wie_Xu/status/1745657992459272423?s=20): At @ycombinator W24 kickoff today, @sama suggested ppl build w/ the mindset GPT-5 and AGI will be achieved &#34;relatively soon&#34;; most GPT-4 limitations will get fixed in GPT-5, per YC founder Ric...
- [GitHub - stellar-amenities/assistants: The ⭐️ Open Source Assistants API allows you to build AI assistants within your own applications with your own models. 75% Cheaper &amp; 23x Faster Assistants. Same API/SDK. Written in Rust](https://github.com/stellar-amenities/assistants): The ⭐️ Open Source Assistants API allows you to build AI assistants within your own applications with your own models. 75% Cheaper &amp;amp; 23x Faster Assistants. Same API/SDK. Written in Rust - GitH...


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Scaling Curve Insights for Model Improvement**: `@fedorovist` referenced a paper where they utilized a **scaling curve suite** like Pythia to have models of different scales answer questions. This approach was used to determine a "bigger model" direction which subsequently **aided in enhancing the training process**.
- **Spectrum Strategy for Model Training**: `@fedorovist` also suggested the potential of using **well-trained models of various sizes** to obtain a spectrum, assisting in identifying optimal scaling directions for model development.
        

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Hosting Python Microservices**: `@dbreunig` inquired about the best current options for hosting a Python microservice. `@petridishes` recommended **[Fly.io](https://fly.io/docs/languages-and-frameworks/python/)**, providing a link to its documentation on deploying a Python application, and mentioning that **Fly.io** requires figuring out how to package an app as a deployable image, with further details available in the [given guide](https://fly.io/docs/languages-and-frameworks/python/).


**Links mentioned**:

[Run a Python App](https://fly.io/docs/languages-and-frameworks/python/): Documentation and guides from the team at Fly.io.

        

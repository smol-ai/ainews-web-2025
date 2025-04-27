---
id: 2f78c3db-d078-41d9-96c1-9feb81775d69
title: '1/8/2024: The Four Wars of the AI Stack'
date: '2024-01-09T07:39:51.817056Z'
original_slug: ainews-182024-the-four-wars-of-the-ai-stack
description: >-
  **Nous Research AI** Discord discussions covered various topics including the
  **Obsidian Project** utilizing **DINO**, **CLIP**, and **CNNs**; challenges
  and solutions for cloud-based LLM services via distributed models like
  **DistAttention** and **DistKV-LLM**; and a paper on **self-extending LLM
  context windows** without tuning. The community also discussed AI models like
  **Mistral**, **Mixtral** (noted for its 32k context window), and **TinyLLAM**.
  Other highlights include the launch of the **AgentSearch-V1** dataset with one
  billion embedding vectors, applications of LLMs in the oil & gas industry, and
  fine-tuning techniques using synthetic data. The conversations reflect active
  research and experimentation in **long-context handling**, **distributed LLM
  architectures**, **hierarchical embeddings**, **agentic retrieval-augmented
  generation (RAG)**, and **mixture-of-experts (MoE)** implementations.
  *"Smaller models may hold more processing capabilities than they seem"* was a
  notable insight shared.
companies:
  - nous-research
  - openai
  - mistral-ai
  - huggingface
models:
  - mistral
  - mixtral
  - tinyllam
topics:
  - long-context
  - distributed-models
  - hierarchical-embeddings
  - agentic-rag
  - fine-tuning
  - mixture-of-experts
  - embedding-vectors
  - context-window
  - cloud-based-llm-services
people:
  - qnguyen3
  - maxwellandrews
  - kenakafrosty
  - adjectiveallison
  - leontello
  - deki04
  - vincentweisser
  - gabriel_syme
  - n8programs
  - realsedlyf
  - kapnap_n
  - teknium
  - ocolegro
---


<!-- buttondown-editor-mode: plaintext -->Not much happened in the discords today, so time to plug our Latent Space 2023 recap! 

https://www.latent.space/p/dec-2023


Enjoy!

---

**Table of Contents**

[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **"Obsidian Project - in short"** : In project Obsidian `@qnguyen3` mentioned briefly that they are utilizing DINO, CLIP and CNNs for the project.

- **"Cloud-Based LLM Concerns, Addressed"**: Discussion about the design challenges of cloud-based Large Language Models (LLMs) services captured attention when `@maxwellandrews` shared a [research paper](https://huggingface.co/papers/2401.02669) that proposed solutions through distributed models like DistAttention and DistKV-LLM. 

- **"Self-Extending Context Window Triumph"**: `@kenakafrosty` shared a paper titled 'Self-Extend LLM Context Window Without Tuning', arguing that existing LLMs are inherently capable of handling long context situations, sharing the [paper's link](https://arxiv.org/abs/2401.01325) and relevant [Twitter](https://twitter.com/sdand/status/1743695855545426362) and [GitHub](https://github.com/datamllab/LongLM) discussions.

- **"Your Morning Coffee, Brought by AI?"**: Users tossed jokes and musings about a tweet that `@adjectiveallison` shared regarding an AI robot called 'Figure-01' that claims to have **learned to make coffee** after observing humans. The conversation expanded into comparing the project to another AI program, **ALOHA**, shared by `@leontello`.

- **"LLMs that Learn and Teach"**: `@deki04` shared a link to a [GitHub repository](https://github.com/mlabonne/llm-course) with a comprehensive course on Large Language Models (LLMs), sparking a discussion about model improvements and their practical applications, led by `@leontello` and `@vincentweisser`.

- **"To Embed, or Not to Embed"**: `@gabriel_syme` suggested that hierarchical embeddings may still be a necessary addition to the OAI model, despite not showing an expected boost in performance.

- **"The Agentic RAG Trend"**: `@n8programs` announced plans to experiment with agentic RAG, a model that generates search queries based on input and collects data until enough has been accumulated. 

- **"AI Engineer's Guide to LLM Fine-Tuning"**: `@realsedlyf` requested insight on the best methods for creating synthetic data required to fine-tune a language model for a specific domain.

- **"Oil & Gas Industry Embraces LLM Analysis"**: `@kapnap_n` detailed the application of LLMs for an unusual domain - analyzing downhole wellbore data in the oil & gas industry.

- **"AgentSearch Dataset Launch!"**: The newly-released AgentSearch-V1 dataset was promoted by `@teknium` who shared a link to [a tweet](https://fxtwitter.com/ocolegro/status/1744207765671657923?s=46) by `@ocolegro`, announcing the availability of one billion embedding vectors encompassing Wikipedia, Arxiv, and more.

- **"LLM Talk - Exposés and Suggestions"**: The '#ask-about-llms' channel saw several captivating debates about different LLM facets like KV_Cache implementation, comparison between MoE and **Mistral**, and performance differences between TinyLLAM and Lite LLAMAS.

- **"Peeking into the Silver Lining"**: The notion that smaller models may hold more processing capabilities than they seem was introduced by `@kenakafrosty` leading to conversations about the saturation point of smaller models.

- **"Exploring Implementations of Merging Technology"**: Queries about notebooks for MoE (mixture of experts) implementations and restraints of PointCloud models led to insightful exchanges between `@teknium` and `.beowulfbr`, with a [Mergekit GitHub link](https://github.com/cg123/mergekit) shared for reference.

- **"Mixtral Favored for its Roomy Context Window"**: `@gabriel_syme` and `@teknium` expressed preference for the Mixtral model. Despite the availability of other models like Mistral and Marcoroni, Mixtral's larger context window of 32k was considered a standout advantage.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (2 messages): 
        
- **Exploring DistAttention for Cloud-Based LLM Services**: User `@maxwellandrews` shared a link to a research paper on DistAttention and DistKV-LLM, new distributed models that aim to alleviate the challenges of designing cloud-based Large Language Models (LLMs) services. The [link](https://huggingface.co/papers/2401.02669) leads to an abstract discussing how these models can dynamically manage Key-Value Cache and orchestrate all accessible GPUs.
  
- **LLMs and Long Context Situations**: User `@kenakafrosty` shared a [link](https://arxiv.org/abs/2401.01325) to a research paper entitled 'Self-Extend LLM Context Window Without Tuning'. The paper argues existing LLMs have the inherent ability to handle long contexts without fine-tuning training sequences.
  
- **Practical Applications of the Self-Extend Model**: `@kenakafrosty` noted that the 'Self-Extend LLM Context Window Without Tuning' concept is being implemented with seemingly good results, sharing relevant [Twitter](https://twitter.com/sdand/status/1743695855545426362) and [GitHub](https://github.com/datamllab/LongLM) links.

**Links mentioned**:

- [Paper page - Infinite-LLM: Efficient LLM Service for Long Context with DistAttention
  and Distributed KVCache](https://huggingface.co/papers/2401.02669)
- [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/abs/2401.01325): This work elicits LLMs&#39; inherent ability to handle long contexts without fine-tuning. The limited length of the training sequence during training may limit the application of Large Language Models...
- [GitHub - datamllab/LongLM: LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://github.com/datamllab/LongLM): LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning - GitHub - datamllab/LongLM: LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (5 messages): 
        
- **Is Coffee-Making AI a Big Deal?**: `@adjectiveallison` shared a tweet from `@adcock_brett` about an **AI robot** called 'Figure-01', claiming it had **learned to make coffee** by watching humans, and mentioned this is an end-to-end AI with video in, trajectories out. This was followed by skepticism and humor from `@teknium` and `@gabriel_syme`, questioning if the capability to make coffee is remarkable news.
  
- **Comparing AI Complexity**: `@leontello` compared the coffee-making AI to another project, **ALOHA** (linked to a tweet by `@tonyzzhao`), calling it quite lackluster, but later clarified their comment, recognizing the difference in contexts involving self-driving vs. robotic hardware setups.
  
- **Spotting Coincidences in AI Robotics**: In a humorous twist, `@adjectiveallison` shared another tweet, this time from `@atroyn`, noticing that the coffee machine used by the coffee-making AI looked very familiar, being previously seen in a video from **Chelsea Finn's** research project.

**Links mentioned**:

- [Tweet from anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn)](https://vxtwitter.com/atroyn/status/1744169452869140988?s=20): something about this demo video seemed very familiar, then i realized i had seen that same coffee machine before in one of @chelseabfinn&#39;s video from her group&#39;s paper   https://lucys0.github....
- [Tweet from Brett Adcock (@adcock_brett)](https://vxtwitter.com/adcock_brett/status/1743987597301399852?s=20): Figure-01 has learned to make coffee ☕️  Our AI learned this after watching humans make coffee  This is end-to-end AI: our neural networks are taking video in, trajectories out  Join us to train our r...


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (5 messages): 
        
- **Resource for LLM Course shared**: `@deki04` shared a [link](https://github.com/mlabonne/llm-course) to a **GitHub repository** offering a comprehensive course on Large Language Models with chatbot roadmaps and Colab notebooks.
- **Mixed Opinions on Model Improvement**: `@leontello` speculated on the feasibility of the introduction of *augmented models*, suggesting a potential increase in parameter counts which could pose practicality concerns.
- **Deep Dive into LLM Agents**: `@vincentweisser` shared an [article](https://borretti.me/article/thoughts-llm-agents) detailing a comprehensive analysis on LLM agents, ChatGPT and GitHub Copilot underlining their significant impact yet existing limitations in complex tasks due to a restricted context window.
- **LeCun's LLM research feature**: `@vincentweisser` also highlighted a [research paper](https://openreview.net/pdf?id=BZ5a1r-kVsf) related to LLM by Yann LeCun.

**Links mentioned**:

- [Thoughts on LLM Agents](https://borretti.me/article/thoughts-llm-agents): Entropy, criticality, and complexity classes of cellular automata.
- [GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.](https://github.com/mlabonne/llm-course): Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks. - GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (79 messages🔥🔥): 
        
- **Exploring Performance of Hierarchical Embeddings**: `@gabriel_syme` noted that hierarchical embeddings might still be needed in addition to the implemented OAI model, hinting at a lack of expected improvement in performance.
- **Handling Synthetic Data for LLM Fine-Tuning**: `@realsedlyf` inquired about the best current methods for creating synthetic data for language model fine-tuning in a specific domain. 
- **Agentic RAG Experimentation**: `@n8programs` announced plans to experiment with agentic RAG, an approach where the model generates various search queries based on an input question and collects information until a sufficient amount has been gathered. They cited **Mistral** as being particularly good for such tasks.
- **Industry Application - LLM and Wellbore Analysis**: `@kapnap_n` shared their approach to using language model to analyze downhole wellbore data in oil & gas industry. They also discussed how the data is represented and the potential benefits of this approach, sparking interest from other users like `@julianotto` and `@everyoneisgross`.
- **AgentSearch Dataset Announcement**: `@teknium` shared a link to `@ocolegro`'s tweet about the release of the AgentSearch-V1 dataset, consisting of over one billion embedding vectors covering Wikipedia, Arxiv, filtered common crawl, and more.


**Links mentioned**:

[Tweet from Owen Colegrove (@ocolegro)](https://fxtwitter.com/ocolegro/status/1744207765671657923?s=46): The full dataset for AgentSearch-V1 is now available on HF!!  Recommended: @qdrant_engine - for indexing and search @nomic_ai - for visualization  I&#39;m looking to expand what is indexed - agent spe...


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (54 messages🔥): 
        
- **Request for KV_Cache implementation**: `@lwasinam` asked for any links to implementations of KV_Cache for confirmation purposes.
- **Considering MoE for comparison with Mistral**: `@bernaferrari` suggested making a mixture of experts (MoE) out of phi and compare it to **Mistral**.
- **TinyLLAM vs Lite LLAMAS**: In a discussion between `@gabriel_syme` and `@teknium`, it was noted that TinyLLAM underperforms, leading to a decision to switch to Lite LLAMAS.
- **Processing capabilities of smaller models**: `@kenakafrosty` sparked a discussion on the notion that smaller models (6-20B range) might actually have more processing capability than appears at first glance, with the gap lying in instruction following. `@teknium` shared his opinion that 7B models are reaching their saturation point but also added that saturation point scales non-linearly with model size. 
- **Mergekit and MOE demonstration**: `@teknium` inquired about any existing notebooks for Mergekit MOE (mixture of experts) implementation. In response, `.beowulfbr` shared a [link](https://github.com/cg123/mergekit) to the Mixtral branch of Mergekit on GitHub for reference.
- **PointCloud model limitations**: In a conversation started by `@gabriel_syme` about PointCloud models, `@teknium` explained that if the base model supports 8k, it should be able to do 8k inputs but will not produce more than 4k outputs. 
- **Preference for Mixtral with larger context window**: `@gabriel_syme` and `@teknium` discussed various models, including Mistral, Marcoroni, and Mixtral. They expressed a preference for **Mixtral**, given its larger context window of 32k.


**Links mentioned**:

[GitHub - cg123/mergekit: Tools for merging pretrained large language models.](https://github.com/cg123/mergekit): Tools for merging pretrained large language models. - GitHub - cg123/mergekit: Tools for merging pretrained large language models.


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (1 messages): 
        
qnguyen3: DINO, CLIP and CNNs for now


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Discussing LAION's Decay**: `@stellaathena` and `@flow7450` considered the decay rate of datasets like LAION. Notably, **LAION400** experienced approximately 20% failure in download after about 2-2.5 years according to a [paper](https://arxiv.org/abs/2111.12359) cited by `@flow7450`.
- **Duel Over Duplicate Data**: `@uwu1468548483828484` and `@flow7450` debated the importance of duplicate data, weighing the merits of backups versus the need for unique samples.
- **ELO-Ribbing Models**: `@letterrip` proposed a project for creating an ELO rating for each question in models to form a testing benchmark subset for training runs. The discussion also touched on where to propose new projects.
- **The Axolotl DPO Conundrum**: `@karkomagor` asked if axolotl supports fine-tuning using DPO datasets, with `@main.ai` suggesting to ask on the Axolotl server instead.
- **T5 Breakdown**: `@stellaathena` confirmed T5 as an encoder-decoder in response to a question from `@ricklius`.
- **Rattling the Learning Rate Cage**: DeepSeek AI models' unusual stepwise decay learning rate schedule was examined by `@maxmatical` and `@ad8e`. This led `@ad8e` to propose testing swift final stretches at 0.1xLR, potentially negating the need for a constant decay.
- **Twisting Transformer Layers**: `@kram1032` suggested permuting layers in transformer architectures during training, hypothesizing that this might encourage more reliance on skip connections and lead to robust networks even when adding or removing layers.
- **Seeking MoE-mentous Scaling Laws**: `@bshlgrs` sought cutting-edge literature on **LM Scaling Laws** specific to Mixture Of Experts (MoE) models, with contributions from `@philpax` and `@main.ai`.
- **The Harness Snags**: From evaluating models like MMLU to implementing custom datasets and even considering adding a toxicity/bias grader, various functionalities of the `lm-eval-harness` were discussed by `@gson_arlo`, `@hyperion.ai`, `@ishigami6465`, and `@.johnnysands`. The importance of speculative decoding was emphasized by `@stellaathena` and `@hailey_schoelkopf`.

**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (29 messages🔥): 
        
- **Decaying Datasets: Pondering LAION's Longevity**: `@stellaathena` initiated a discussion about linkrot, specifically addressing the decay rate of datasets like LAION. `@flow7450` examined **LAION400's** decay, discovering that approximately [20% failed to download](https://arxiv.org/abs/2111.12359) after about 2-2.5 years.
- **Duplicates: A Debate on Backup Strategy vs. Uniqueness**: `@uwu1468548483828484` and `@flow7450` had a debate on the importance of duplicate data, arguing whether backups outweigh the need for unique samples.
- **A New Project Proposal on Model ELO Ranking**: `@letterrip` proposed a new project, suggesting creating an ELO rating for each question in current models to form a subset of benchmarks for testing during training runs. There was discussion about where to propose new projects, with `@flow7450` suggesting `<#1102787157866852402>` and `@ad8e` clarifying that community-projects is mainly for projects that one intends to drive rather than just ideas. `@letterrip` confirmed interest in driving the project. 
- **Axolotl and DPO datasets**: `@karkomagor` inquired if axolotl supports fine-tuning using DPO datasets, and `@main.ai` recommended asking this in the Axolotl server instead.
- **Inquiries about Managing Multiple Server Activities**: `@seon5448` opened a discussion about keeping track of activities across various servers, and was seeking advice on management tactics for multiple reading groups and project events. Suggestions for management tools or techniques were not mentioned in response.


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (33 messages🔥): 
        
- **T5 Encoder-Decoder Clarification**: `@ricklius` inquired if T5 was an encoder-decoder to which `@stellaathena` confirmed it is, and mentioned that sometimes people detach the encoder from the encoder-decoder for usage.

- **DeepSeek AI Models' Learning Rate Schedules**: `@maxmatical` discussed DeepSeek AI models, particularly an unusual stepwise decay learning rate schedule they used that had the same final loss as the traditional cosine decay. While this method allows for more flexible use of checkpoints during pre-training, `@ad8e` highlighted the potential dangers of large learning rate steps, pointing out that misuse of learning rates could lead to suboptimal outcomes or even divergence in model training.

- **Potential Model Training Experiment**: `@ad8e` revealed an intention to test the idea behind the above-discussed learning rate steps. The intention was to see if a swift final stretch at 0.1xLR was all that's needed, possibly negating the need for a constant decay.

- **Discussion on Weight Decay and Gaussian Weight Noise**:  `@fessus` brought up the subject of the possible effects of combining gaussian weight noise and weight decay in a network that doesn't have learned affines in normalization layers. They reported potential benefits in terms of pruning unnecessary network complexity in toy datasets.

- **Transformers with Permuted Layers Idea**: `@kram1032` proposed a unique idea of permuting layers during training within a transformer architecture with constant layer size. Their hypothesis is that this approach may encourage the network to rely more on skip connections and could lead to more robust networks when it comes to adding or removing layers.

**Links mentioned**:

- [Chess-GPT’s Internal World Model](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html): A Chess-GPT Linear Emergent World Representation
- [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954): The rapid development of open-source large language models (LLMs) has been truly remarkable. However, the scaling law described in previous literature presents varying conclusions, which casts a dark ...
- [ad8e](https://wandb.ai/ad8e/tinystories3/runs/30snj0x7/logs?workspace=user-ad8e;): Weights & Biases, developer tools for machine learning


### ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/) (5 messages): 
        
- **Looking for Reading Recommendations on LM Scaling Laws**: `@bshlgrs` sought recommendations for current state-of-the-art literature on **Language Model (LM) scaling laws**, specifically for Mixture of Experts (MoE) models. They specifically mentioned [this paper](https://arxiv.org/abs/2202.01169) which recommends MoE use only for LMs with less than 1 billion parameters, a claim appearing to be contested by practitioners.
- **Suggestion for LM Scaling Laws Paper**: `@philpax` highlighted a [recent paper](https://arxiv.org/abs/2401.00448) on LM scaling laws. Although this does not specifically address MoE models, it could offer relevant insights.
- **'Smaller and Longer' is the Key for Large Inference Demand**: `@bshlgrs` highlighted a key finding from the suggested paper, suggesting that for Language Model Manufacturers (LLM) with large inference demand (around 1 billion requests), the best strategy is to **train smaller models for a longer duration**.
- **Lack of High-Compute Budget Scaling Papers for MoE**: `@main.ai` pointed out the lack of papers addressing high compute budget scaling for MoE models.


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (16 messages🔥): 
        
- **Understanding lm-eval-harness Functions**: `@gson_arlo` asked how `lm-eval-harness` works with evaluation on datasets like MMLU (4-choice mcqa). `@baber_` confirmed that `output_type: generate_until` triggers a model's inference once, whereas `output_type: log_prob` calculates the likelihood four times, once for each probable completion.
- **Flexible Postprocessing for lm-eval-harness**: `@hyperion.ai` suggested enhancing `lm-eval-harness` with a loose and flexible post-processing factor, aligning closer to real-world practices where the answer output can be flexible yet correct. `@stellaathena` confirmed that the harness can handle such situations.
- **Implementing Custom Datasets in lm-eval-harness**: `@ishigami6465` inquired about the specific format of datasets needed for `lm-eval-harness`. `@hailey_schoelkopf` clarified that the user can define this in a task's configuration and explained how the configuration could work for different types of tasks.
- **Potential for Toxigen Grader in lm-eval-harness**: `@.johnnysands` brought up the idea of adding a toxicity/bias grader to lm-eval-harness, considering tools like LlamaGuard offer this functionality already. `@hailey_schoelkopf` affirmed that such grader models could be integrated, particularly if locally deployed to avoid disrupting the main evaluation model.
- **Considerations for Speculative Decoding**: `@stellaathena` highlighted the importance of speculative decoding, and `@hailey_schoelkopf` suggested that an inference library should handle this externally from lm-eval. Both believe the Hugging Face's TGI and tensorrt-llm currently manage this well.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Europe's Elderly Shift**: In the #prompt-engineering channel, `mysterious_guava_93336` and `ChatGPT` explored the demographic transition from the **"European baby boom"** of the last century to today's **"elderly Europe"**. The AI clarified that signs of fertility contraction started appearing in the late 1960s due to various factors, leading to an aging European population by the late 20th and early 21st centuries. This conversation was later echoed in the #api-discussions channel, reinforcing the shift's timeline and the factors uniting the demographic change.
   
- **Are PowerPoint Presentations Ruining AI?**: In #ai-discussions, an ongoing topic was about `ChatGPT` recently providing verbose, "PowerPoint-like" responses instead of concise, natural conversations. Despite advice from `@eskcanta` on refining system instructions to generate desired responses, `@mysterious_guava_93336` reported no significant improvement. The discussion also addressed how to use `ChatGPT` on Discord.

- **Staying in Your Domain**: Dealing with domain verification and GPT editor issues was a hot topic in #gpt-4-discussions. While `@darthgustav` attempted to guide `@anardude` through domain verification, `@anardude` struggled with the solution and sought OpenAI Support's help. `@.marcolp` expressed frustration at a persistent error disallowing access to the GPT editor until resolved.

- **When Training Hours Go to Waste**: Despair echoed in #gpt-4-discussions as `@moonlit_shadows` expressed disappointment in a 20-hour GPT-2 training session that ended unproductively.

- **Language-specific GPT Questions and Rule Sets**: `@codocoderson` queried about creating a GPT in English and having its descriptions and starter questions shown in the user's language worldwide in #gpt-4-discussions. `@jesuisjes` inquired if their GPT occasionally missed following an outlined process for strategy-making conversations.

- **Why Can't GPT Read my Files?**: The #gpt-4-discussions channel saw a discussion by `@cerebrocortex`, wondering why GPTs sometimes have issues reading .docx and .txt files. The issue was admittedly uncertain, with possible reasons being document size, token limits, or file corruption during /mnt timeouts.


**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (30 messages🔥): 
        
- **Concerns about recent updates to ChatGPT**:
    - `@mysterious_guava_93336` expressed dissatisfaction with recent updates to ChatGPT, stating that the AI now provides verbose, structured "PowerPoint" style responses instead of the concise, natural conversation style it used to have before the summer of 2023. They shared [an example of the type of answer they prefer](https://chat.openai.com/share/f7390818-627b-43e6-9f68-054afca55fbf) and asked for advice on how to instruct the AI to generate such responses.
- **Proposed Instructions for Desired Responses**:
    - `@eskcanta` suggested refining the system's custom instructions to be more specific and positive, using guidance techniques similar to dog training. They proposed using a pattern for opinions and encouraging the AI to challenge the user creatively. An [example of this approach can be seen on OpenAI's chat](https://chat.openai.com/share/f7390818-627b-43e6-9f68-054afca55fbf).
- **Disappointment with Proposed Changes**:
    - Despite changing the instructions, `@mysterious_guava_93336` noticed no major improvement, stating that the AI still generated "PowerPoint-like" outputs.
- **Using ChatGPT on Discord**:
    - `@lemon77_ps` inquired about how to use ChatGPT on Discord, and `@7877` explained that ChatGPT must be used via OpenAI's website.
- **Interaction on the Discord Server**:
    - `@michael_6138_97508` pointed out that the discord server is meant for interactions with real people, or equivalents, also mentioning the existence of a bot with specific knowledge of OpenAI API and documentation.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (26 messages🔥): 
        
- **Domain Verification Woes**: User `@anardude` asked how to verify his domain. `@darthgustav` provided instructions for verifying the domain via DNS records in the GPT editor but `@anardude` reported the solution didn't work and asked about how to contact OpenAI Support, to which `@darthgustav` suggested Googling "OpenAI Help and Support".
- **Long Hours of GPT-2 Training in Vain?**: User `@moonlit_shadows` shared their disappointment in a 20-hour training session that seemed to end unproductively due to an attribute error during saving.
- **Inquiries about Creating Language-Specific GPTs**: User `@codocoderson` asked about publishing a GPT in English and inquired whether its descriptions and starter questions will be shown in the user's language worldwide.
- **Issues with GPT Obeying Rulesets**: `@jesuisjes` sought to confirm expectations about their GPT occasionally missing processes, despite setting up rules to follow an outlined process for strategy-making conversations.
- **Concerns on GPT's Issues with Reading Document Files**: `@cerebrocortex` asked why GPTs sometimes have problems reading .Docx & .txt files. User `@michael_6138_97508` made an educated guess about document size and token limits being potential issues, and `@darthgustav` suggested /mnt timeouts during updates causing file corruption.
- **Troubles with GPT Editor**: `@.marcolp` expressed frustration over a persistent "error searching knowledge" problem, leading to an inability to even access the GPT editor, rendering further development of GPTs potentially useless until a fix is in place. `@darthgustav` offered a potential workaround involving removing and reattaching knowledge.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (1 messages): 
        
- **European Demographics Evolution Discussed**: User `mysterious_guava_93336` initiated a conversation with `ChatGPT` about the transition from the **"European baby boom"** of last century to today's **"elderly Europe"**. 
- **Fertility Contraction Timeline**: `ChatGPT` clarified that the first signs of a fertility contraction in Europe began emerging in the late 1960s and became significantly pronounced in the 1970s. 
- **Contributing Factors to Fertility Decline**: The reasons cited for this shift included economic changes, women's rights and workforce participation, increased access to contraception and family planning, and cultural shifts.
- **Resulting Aging Population**: By the late 20th and early 21st centuries, many European countries were experiencing birth rates below the replacement level of 2.1 children per woman, leading to an aging population.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (1 messages): 
        
- **The Transition from 'Baby Boom' to 'Elderly Europe'**: `mysterious_guava_93336` engaged ChatGPT in a discussion about when the first signs of a fertility contraction began in Europe after the post-World War II "baby boom". ChatGPT confirmed that this demographic transition started to appear in the late 1960s and became more pronounced in the 1970s, with various factors contributing, including economic changes, women's workforce participation, access to contraception, and cultural shifts.


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **File Upload Features on Mobile**: According to `@righthandofdoom` and `@giddz`, it's currently possible to only upload images from the **Perplexity mobile app** on iOS. Support for Android is reported to be **on the horizon**.
- **Locating the Writing Mode Feature**: `@ellestar_52679` queried about the location of the **Writing Mode** feature, and `@icelavaman` highlighted the navigation pathway: *click "Focus", then "writing"*.
- **Detailed Discussion on Features and Promotions**: `@debian3` queried about file upload features and available promotions for the **yearly plan**. `@icelavaman` clarified that savings could be achieved through referral links instead of promotions and redirected to a [FAQ link](https://blog.perplexity.ai/faq/how-does-file-upload-work) for more details on file upload.
- **Billing Issues with Perplexity**: `@ahmed7089` was upset about getting charged by **Perplexity** despite deleting their account. The situation was addressed by `@mares1317`, who provided a [relevant link](https://discord.com/channels/1047197230748151888/1118264005207793674/1188109891508908073).
- **Perplexity Access Restrictions**: `@byerk_enjoyer_sociology_enjoyer` voiced concerns about **Perplexity**'s inability to access posts on Pinterest, Instagram, or Tumblr.
- **In-depth Perplexity vs Pplx Model Comparison Analysis**: `@dw_0901` initiated discussions about **differences among pplx online models (7B/70B) and Perplexity**, questioning the differences in product design.
- **Contrasting Perplexity's Copilot and Normal version**: `@promoweb.2024` enquired about **Perplexity's Copilot and normal version** differences. Detailed information about Perplexity Copilot was shared by `@icelavaman` at this [link](https://blog.perplexity.ai/faq/what-is-copilot).
- **Troubleshoot using $5 Pro Credits on pplx-api**: `@blackwhitegrey` was guided to the application process for Pro credits on `pplx-api` by `@mares1317`, who also provided a [step-by-step guide](https://docs.perplexity.ai/docs/getting-started).
- **Clarity on Perplexity API User Friendliness**: `@blackwhitegrey` and `@brknclock1215` struggled with a perceived lack of user-friendliness in Perplexity API, primarily due to a lack of coding skills. `icelavaman` clarified that the **API is primarily meant for developers**.
- **Clarification on Pro Credits as an Extra Payment**: `@blackwhitegrey` initially misconstrued Pro credits as an additional payment for API access. `icelavaman` clarified that these are actually bonuses provided to developers.
- **Optimism for Non-Technical Users**: Despite the struggles, `@brknclock1215` ended with an optimistic view that people, who aren't necessarily coders but understand technology, could **benefit the most from its progression**.
- **Help requested to make a thread public**: `@me.lk` advised `<@1018532617479532608>` to make their thread public so others can view their content. `<@1018532617479532608>` followed this advice and made the thread publicly accessible.
- **Sharing Perplexity.AI Searches**: Searches on **how to use** and **how to draw** were shared by `@soanseng` and `@debian3` respectively, spreading their knowledge with the community.


**Perplexity AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (23 messages🔥): 
        
- **File Upload Functionality In Mobile App**: `@righthandofdoom` asked about the ability to upload files from the **mobile app**. `@giddz` clarified that it's currently possible only for images and is only available on iOS, with Android support **coming soon**.
- **Finding the "writing mode" feature**: `@ellestar_52679` was trying to locate the **writing mode** feature. `@icelavaman` advised them to click "Focus", then "writing".
- **Questions about Perplexity's functionality and promotions**: `@debian3` asked about the purpose of the file upload feature, the types of files that can be uploaded and if any promotions were available for the **yearly plan**. `@icelavaman` assured that while there was no promo, saving could be achieved through referral links. They also provided a [link](https://blog.perplexity.ai/faq/how-does-file-upload-work) for the queries on file uploads.
- **Issues with Account Deletion and Billing**: `@ahmed7089` complained about being billed by **Perplexity** even after deleting their account. `@mares1317` provided a [link](https://discord.com/channels/1047197230748151888/1118264005207793674/1188109891508908073) in response, presumably containing more information.
- **Perplexity and Social Media Platforms**: `@byerk_enjoyer_sociology_enjoyer` raised a concern about **Perplexity**'s inability to access Pinterest, Instagram, or Tumblr posts.
- **Perplexity and Pplx Model Comparison**: `@dw_0901`, a consultant, asked about the **differences between pplx online models (7B/70B) and Perplexity**, questioning if there were differences in the underlying product design.
- **Copilot vs Normal version of Perplexity**: `@promoweb.2024` enquired about the differences between using **Perplexity's Copilot and the normal version**. `@icelavaman` shared a [link](https://blog.perplexity.ai/faq/what-is-copilot) providing a detailed overview of Perplexity Copilot.


**Links mentioned**:

- [What is Perplexity Copilot?](https://blog.perplexity.ai/faq/what-is-copilot): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [How does File Upload work?](https://blog.perplexity.ai/faq/how-does-file-upload-work): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [What is Search Focus?](https://blog.perplexity.ai/faq/what-is-search-focus): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (4 messages): 
        
- **Sharing made public**: User `@me.lk` advised `<@1018532617479532608>` to **make their thread public** so others can see it, to which `<@1018532617479532608>` responded that they've made their thread public now.
- **Perplexity.AI Searches**: Users `@soanseng` and `@debian3` shared perplexity.ai searches:
    - `@soanseng`: shared a link on [how to use](https://www.perplexity.ai/search/how-to-use-jDLvEroNQke87WGjx4W.fA?s=c)
    - `@debian3`: shared a link on [how to draw](https://www.perplexity.ai/search/How-to-draw-PZIg5SS2QxuuK8xpHyHhGA?s=c#3d9220e5-24b6-431b-ae2b-cc691f21e118)


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (12 messages🔥): 
        
- **How to use $5 Pro Credits**: User `@blackwhitegrey` sought advice on using the Pro credits on `pplx-api`. `@mares1317` provided a link to [Perplexity API's Getting Started Guide](https://docs.perplexity.ai/docs/getting-started) and highlighted the steps including **providing payment info, purchasing credits, and generating an API key**.
- **Mismatched API knowledge and needs**: `@blackwhitegrey` expressed frustration due to their lack of coding skills and perceived the **API as not user-friendly**. `icelavaman` clarified that the **API is primarily intended for developers** and not for direct usage on websites. 
- **Pro Credits understood as an extra payment**: `@blackwhitegrey` initially assumed Pro users had to pay an extra $5 for the API access. However, `icelavaman` clarified that these are **not extra payments but bonuses for developers**. 
- **Practicality of using the API for Non-Developers**: `@brknclock1215` echoed `@blackwhitegrey`'s sentiments, expressing similar difficulties in implementing the API due to the lack coding skills. They also reasoned that trying to use advanced tools without the appropriate technical knowledge can be more time-consuming than beneficial.
- **Non-Technical users can still benefit**: `@brknclock1215` ended on an optimistic note, insinuating that people who are not necessarily coders, but who understand how to interact with technology, could potentially benefit the most from its progression.

**Links mentioned**:

- [Getting Started with pplx-api](https://docs.perplexity.ai/docs/getting-started)
- [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions)


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Priming the LLM Response:** In a conversation with `@bdambrosio`, `@i_am_dom` clarified that priming the Latent Language Model (LLM) for an appropriate response is indeed feasible, but the message set must end with a user message. This douses hopes for partially pre-writing responses with the official API as it would return an error.
- **On the Hunt for a Solid Chat Conversation Program:** `@jb_5579` posed a question to the community seeking recommendations on repositories that provide a robust chat conversation program, ideally optimized for the **Mistral API**, and featuring memory session alongside code-assist and code-completion.
- **The Case of Unknown Context Window Sizes:** `@tonyaichamp`'s inquiry about the context window sizes for different versions of API models was met with uncertainty by `@frosty04212`, emphasizing on the need for experimental explorations to better understand and harness the system.
- **Mistral-tiny Proves Its Mettle:** `@tonyaichamp` shared noteworthy success with the `mistral-tiny` model, leveraging it to extract content from a 16k token HTML page. Given its cost-effectiveness and speed, the user intends to apply it for similar tasks in the future.
- **Ai Agents Assemble:** `@10anant10` announced their project centered around **building AI agents**, to which `@.tanuj.` expressed interest and initiated direct communication.
- **Framework Endorsement:** User `@joselolol` recommended exploring the **MLX framework**.
- **Guardrailing Guide:** A useful resource on **guardrailing** is linked by `@akshay_1` with a URL: `https://docs.mistral.ai/platform/guardrailing/`
- **Hardware Limitations for Fine-tuning:** `@david78901` raised the feasibility of fine-tuning the **Mistral 7b** on a single 3090. Tuning a LoRA or QLoRA could be managed, but full fine-tuning would likely need multiple 3090s or a single A100 with `Axolotl`.
- **LLMcord - The Versatile Discord Bot:** `@jakobdylanc` showcased LLMcord, an open-source Discord bot compatible with **Mistral API** and personal hardware-run **Mistral models** via *LM Studio*. The project available on GitHub also scored mention.
- **Superiority of Mistral Over OpenAI:** `@joselolol` acknowledged Mistral's edge over OpenAI, recognizing it as faster, cheaper, and more effective for tasks.
- **Brains Baffled by Bicameral Mind:** A mention of the Bicameral Mind theory stirred a debate, with `@cognitivetech` recommending the cornerstone book by Julian Jaynes and a skeptical `@king_sleeze` equating the theory to pseudoscience.
- **Mistral API Feature Suggestion:** `@jakobdylanc` favored an enhancement to the **Mistral API** to handle edge cases of an empty content list, much like the **OpenAI API** presently does.
- **Functionality Expansion on the Horizon:** Function calling in **Mistral** is set to be a priority, as indicated by `@tom_lrd`.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (9 messages🔥): 
        
- **Prime the Response:** `@bdambrosio` discussed the technique of using a final assistant message to 'prime' the LLM for the appropriate response, and was seeking advice on its effectiveness when the server rejects a message set that ends with an assistant message. `@i_am_dom` clarified that response priming can be done, but it needs to end with a user message. They also mentioned that it is not possible to partially pre-write the response itself with the official API, as it would return an error.
- **Favorite repo for chat conversation:** `@jb_5579` asked the community for their favorite repositories for a solid chat conversation program - specifically one that is optimized for the Mistral API and features session memory, with a focus on Code Assist and Code Completion.
- **Context Window Sizes**: `@tonyaichamp` inquired about the context window sizes for different versions of API models, but `@frosty04212` responded that the sizes are not currently known. They urged for experimentation to understand and leverage the system better.
- **Quality of Mistral-tiny for extracting content**: `@tonyaichamp` shared a positive experience using the `mistral-tiny` model for extracting content from a 16k token HTML page. Given the model's cost-effectiveness and speed, `@tonyaichamp` intends to use it for similar tasks in the future.


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (2 messages): 
        
- **Building AI Agents**: User `@10anant10` announced that they are working on **building AI agents**.
- **Direct Communication Initiated**: User `@.tanuj.` responded to `@10anant10`'s comment, stating their intention to **shoot them a direct message**.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (1 messages): 
        
joselolol.: Hello good sir, consider using the MLX framework!


### ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/) (1 messages): 
        
akshay_1: https://docs.mistral.ai/platform/guardrailing/


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (2 messages): 
        
- **Feasibility of fine-tuning on a single 3090**: User `@david78901` mentioned that a single 3090 can possibly handle tuning a LoRA or QLoRA on **Mistral 7b**, but full fine-tuning is only feasible with 3x3090s or a single A100 using Axolotl.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (4 messages): 
        
- **Introducing LLMcord, a Versatile Discord Bot**: `@jakobdylanc` presented his open-source Discord bot, LLMcord, which supports both Mistral API and running Mistral models on personal hardware via [LM Studio](https://lmstudio.ai/). Features include a sophisticated chat system, compatibility with OpenAI API, streamed responses, and concise code contained in a single Python file. One can check out the project on [GitHub](https://github.com/jakobdylanc/llmcord).
- **Mistral Powers AI Backend**: `@joselolol` mentioned that they're using Mistral to support the backend of certain AI tasks.
- **Synthetic Data Generation and Model Evaluation**: `@joselolol` also shared that his system can generate synthetic data and provide evaluation for fine-tuned models, a potentially useful tool for developers.
- **Mistral vs OpenAI**: In `@joselolol`'s experience, Mistral outpaces OpenAI in most tasks, proving faster, cheaper, and more effective.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/)): Find, download, and experiment with local LLMs
- [GitHub - jakobdylanc/llmcord: A Discord AI chat bot | Choose your LLM | GPT-4 Turbo with vision | Mixtral 8X7B | OpenAI API | Mistral API | LM Studio | Streamed responses | And more 🔥](https://github.com/jakobdylanc/llmcord): A Discord AI chat bot | Choose your LLM | GPT-4 Turbo with vision | Mixtral 8X7B | OpenAI API | Mistral API | LM Studio | Streamed responses | And more 🔥 - GitHub - jakobdylanc/llmcord: A Discord A.....


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (5 messages): 
        
- **Bicameral Mind Theory Sparked Interest**: User `@blueridanus` appeared puzzled about something, which was soon clarified by `@cognitivetech` with a link to the [Wikipedia page](https://en.wikipedia.org/wiki/The_Origin_of_Consciousness_in_the_Breakdown_of_the_Bicameral_Mind) that discussed **The Origin of Consciousness in the Breakdown of the Bicameral Mind**. It's a 1976 publication by author Julian Jaynes that presents a theory on the origin of human consciousness.
- **Book Recommendation and Contention**: The same user `@cognitivetech` then highly recommended the book and highlighted its thought-provoking nature. In response, `@king_sleeze` expressed skepticism, arguing that Jaynes’s theory is based on circumstantial evidence, equating it to *pseudoscience*.
- **Skepticism Over Understanding Consciousness**: In another message, `@king_sleeze` pointed out the complexity of understanding human consciousness, drawing a comparison to the 'black box' nature of neural networks. They stated, "*no human I know of could tell me where or how their thoughts formed*", highlighting the mysterious nature of human thought formation.


**Links mentioned**:

[The Origin of Consciousness in the Breakdown of the Bicameral Mind - Wikipedia](https://en.wikipedia.org/wiki/The_Origin_of_Consciousness_in_the_Breakdown_of_the_Bicameral_Mind)


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (3 messages): 
        
- **Discussion on desired functions**: `@gbourdin` mentioned that the community is eagerly waiting for a certain function possibility. 
- **Mistral API enhancement suggestion**: `@jakobdylanc` suggested that the **Mistral API** should be able to handle the edge case of message.content as an empty list just like the **OpenAI API** does currently.
- **Function calling is on the horizon**: `@tom_lrd` mentioned that function calling has been **announced as a priority** for future development.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **German DPR Dataset from Wikipedia**: `@philipmay` shared his project for creating a German *Dense Passage Retrieval (DPR)* dataset based on the German Wikipedia. He posted the project on [GitHub](https://github.com/telekom/wikipedia-22-12-de-dpr) for public use. 

- **Debate on Contextual Length**: A discussion emerged about the appropriate length of document context for embedding. `@sebastian.bodza` questioned whether `@philipmay's` use of a maximum token count of 270 in his project was too short, and compared it to [Jina Embeddings](https://arize.com/blog-course/evaluation-of-llm-rag-chunking-strategy/) and other models that were trained on as many as 512 tokens. `@philipmay` and `@bjoernp` argued that longer contexts may become distracting or be more difficult for a BERT model to encode.

- **BAAI Training Data Suggestion**: `@sebastian.bodza` shared a link to BGE's training data hosted on [HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5) suggesting it might provide additional insights. 

- **E5's Training on 512 Tokens**: `@sebastian.bodza` noted that the E5 model was also trained on 512 tokens, further supporting the debate on the optimal contextual length. Detailed information about E5's training can be found [here](https://arxiv.org/abs/2212.03533).

**Links mentioned**:

- [Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923): Text embedding models have emerged as powerful tools for transforming sentences into fixed-sized feature vectors that encapsulate semantic information. While these models are essential for tasks like ...
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533): This paper presents E5, a family of state-of-the-art text embeddings that transfer well to a wide range of tasks. The model is trained in a contrastive manner with weak supervision signals from our cu...
- [GitHub - telekom/wikipedia-22-12-de-dpr: German dataset for DPR model training](https://github.com/telekom/wikipedia-22-12-de-dpr): German dataset for DPR model training. Contribute to telekom/wikipedia-22-12-de-dpr development by creating an account on GitHub.
- [BAAI/bge-large-en-v1.5 · Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Benchmarking Evaluation of LLM Retrieval Augmented Generation](https://arize.com/blog-course/evaluation-of-llm-rag-chunking-strategy/): Learn about what retrieval approaches work and chunking strategy. Includes test scripts and examples to parameterize retrieval on your own docs, determine performance with LLM evaluations and provide ...

        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Exploring the Role of AI as an Editor**: User `@slono` questioned if there are **AI writing tools that perform as an editor**, helping craft messages, reorganize structure, suggest or eliminate sentences. `@coffeebean6887` suggested that describing what you want to a custom GPT for a few minutes can achieve this. However, slono pointed out that it's not an ideal solution due to its cumbersome user interface.
- **Struggles with AI-Assisted Editing**: Responding to the above thread, `@swizec` shared their experience of implementing such a feature in **swiz-cms**. They remarked that the biggest challenge was effectively communicating what the AI should look for, implying the need for improved guidance or interface for content editing AI tools.
- **Insights into Large Language Models (LLMs)**: User `@swyxio` shared a link to a [LessWrong post](https://www.lesswrong.com/posts/5ZFgZbqp6Mi2xpYjK/an-explanation-for-every-token-using-an-llm-to-sample) that delves into the implications and potential safety benefits of using LLMs in creating AGIs, and a related [Arxiv paper](https://arxiv.org/abs/2401.02415) that proposes a post-pretraining method for LLMs to improve their knowledge without catastrophic forgetting.
- **Appreciation for AI Resources**: User `@thenoahhein` thanked `@swyxio` for providing resources, saying it had given him reading material for the week. The resources link to a [Twitter post](https://twitter.com/eugeneyan/status/1744179600056545300) from user Eugene Yan.

**Links mentioned**:

- [LLaMA Pro: Progressive LLaMA with Block Expansion](https://arxiv.org/abs/2401.02415?utm_source=ainews&utm_medium=email): Humans generally acquire new skills without compromising the old; however, the opposite holds for Large Language Models (LLMs), e.g., from LLaMA to CodeLLaMA. To this end, we propose a new post-pretra...
- [Mat’s Blog - Transformers From Scratch](https://blog.matdmiller.com/posts/2023-06-10_transformers/notebook.html)
- [An explanation for every token: using an LLM to sample another LLM — LessWrong](https://www.lesswrong.com/posts/5ZFgZbqp6Mi2xpYjK/an-explanation-for-every-token-using-an-llm-to-sample): Introduction Much has been written about the implications and potential safety benefits of building an AGI based on one or more Large Language Models…

        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **The Mysterious Case of LAION's Linkrot**: @stellaathena expressed interest in any studies examining the rate of *decay of a dataset like LAION due to linkrot*.
- **A Puzzling Inquiry on Stable Diffusion 1.6**: @pseudoterminalx asked about **Stable Diffusion 1.6**, and @nodja speculated it might combine 1.x architecture with improvements from sdxl and more.
- **Delving Deep into Aspect Ratio Bucketed SD 1.5**: @thejonasbrothers shared that **Aspect Ratio Bucketed SD 1.5** supports up to *1024x1024 pixels*.
- **CogVLM Steals the Spotlight**: @SegmentationFault gave a shoutout to **CogVLM** stating it's *highly impressive and undervalued in the AI community*.
- **A Dreamy Solution for Overfitting**: @progamergov posted a [research paper](https://www.cell.com/patterns/fulltext/S2666-3899(21)00064-7) arguing that *dreams could potentially act as an anti-overfitting mechanism in the human brain* by introducing random noise to overfitted concepts.
- **Sleep-deprived Memories Need More Research**: In response to the dream theory, @progamergov expressed wish for studies investigating the effects of *sleep deprivation on semantic and episodic memory formations* as supportive evidence.

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (8 messages🔥): 
        
- **Curiosity about Linkrot in LAION Dataset**: `@stellaathena` inquires if anyone has conducted or seen any study pertaining to the rate of decay of a dataset like LAION due to linkrot.
- **Questions about Stable Diffusion 1.6**:  `@pseudoterminalx` asks for information about Stable Diffusion 1.6. `@nodja` hypothesizes that it might be the 1.x architecture with some improvements from sdxl and some extras.
- **Insights into Aspect Ratio Bucketed SD 1.5**: `@thejonasbrothers` provides insights about Aspect Ratio Bucketed SD 1.5, they stated that it supports up to 1024x1024 pixels.
- **Appreciation towards CogVLM**: `@SegmentationFault` expresses their appreciation for CogVLM, stating that they believe it's highly impressive and undervalued in the AI community.


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (2 messages): 
        
- **Dreams as Anti-overfitting Mechanisms**: `@progamergov` shared a [research paper](https://www.cell.com/patterns/fulltext/S2666-3899(21)00064-7) suggesting that **dreaming might play a crucial role in preventing overfitting in the human brain**. According to the paper, dreams introduce random noise to overfitted concepts, thereby aiding in avoiding overfitting.
- **Call for Testing Effects of Sleep Deprivation on Memory**: `@progamergov` expressed a wish that the research extended to testing the impacts of **sleep deprivation on semantic and episodic memory formations**, asserting such tests could provide supporting evidence for the aforementioned hypothesis.


        
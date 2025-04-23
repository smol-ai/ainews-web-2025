---
id: 3d66beee-bc9f-4dad-a3fe-dcc91b1e9af1
title: 1699062952925 Discord Newsletter Test
date: '2023-11-04T02:01:49.247517Z'
status: sent
type: public
source: api
metadata: {}
original_slug: 1699062952925-discord-newsletter-test
---

<!-- buttondown-editor-mode: plaintext -->

## Guild: [Skunkworks AI](https://discord.com/channels/1131084849432768614)

- Inquiry into the usage of torch.nn.MultiheadAttention in different libraries:
    - User aniketmaurya questioned why many libraries build their own MultiheadAttention implementation instead of using torch.nn.MultiheadAttention.
    - User benjamin_w replied that this might be because torch.nn.MultiheadAttention only supports flash attention during inference and not during training.

- Discussion on SVD implementation and model quantization:
    - Talked about the problem of SVD implementations reshaping layer weights when a quantized model is loaded.
    - Suggestion of passing "full_matrices=false" to SVD, which may result in needing to reshape the outcome to set "lora_a" and "lora_b" weights with proper dimensions.

- Proposition on a potential solution for regular LoRA:
    - It was proposed that alterations to the regular LoRA might provide a solution to the above problem, contingent on the VRAM budget.

- General introductions and greetings among users in the welcome channel with fluctuating levels of excitement. No significant discussion topics or links to note. Involvement varied from users introducing themselves to isolated messages lacking contextual information.

--- Channel by Channel Summary --- 

### Channel: [general](https://discord.com/channels/1131084849432768614/1131084849906716735)

Summary: 
1. (specific topic title, e.g. "Usage of torch.nn.MultiheadAttention") (Excitement: N/A)
    - (specific discussion thread, e.g. "Most libraries having their own MultiHeadAttention implementation")
        - The user aniketmaurya expressed curiosity about why most libraries have their own MultiHeadAttention implementation instead of using `torch.nn.MultiheadAttention`.
    - (specific discussion thread, e.g. "Limitation of `torch.nn.MultiheadAttention`")
        - The user benjamin_w mentioned that `torch.nn.MultiheadAttention` currently only supports flash attention during inference, not training.
    - Links: N/A


### Channel: [moe-main](https://discord.com/channels/1131084849432768614/1139310171076706464)

Summary: 
1. SVD implementation issue with model quantization (Excitement: 4/10)
    - There was a discussion about the problem of SVD implementations reshaping layer weights when loading a quantized model.
    - It was mentioned that passing the argument "full_matrices=false" to SVD would require reshaing the result to subsequently set "lora_a" and "lora_b" weights with the proper dimensions.
2. Possible workaround for regular LoRA (Excitement: 5/10)
    - It was mentioned that using regular LoRA with a few alterations could potentially address the issue, depending on the VRAM budget.


### Channel: [welcome](https://discord.com/channels/1131084849432768614/1139363022150848612)

Summary: 
1. (Topic: General Introduction and Greetings) (Excitement: 3/10)
    - Users introduced themselves in the channel.
    - Some users exchanged greetings.
    - No specific discussion points or links were mentioned.

2. (Topic: General Discussion) (Excitement: 4/10)
    - Users engaged in a general conversation.
    - No specific discussion points or links were mentioned.

3. (Topic: Unspecified) (Excitement: 2/10)
    - A user named "paradisen" posted a single message with no context provided.
    - No further discussion or links were mentioned.

4. (Topic: Unspecified) (Excitement: 2/10)
    - A user named "roberto_there" posted a single message with no context provided.
    - No further discussion or links were mentioned.

5. (Topic: Unspecified) (Excitement: 2/10)
    - A user named "mike.bird" posted a single message with no context provided.
    - No further discussion or links were mentioned.

6. (Topic: Unspecified) (Excitement: 3/10)
    - A user named "whimsicalism" posted a single message with no context provided.
    - No further discussion or links were mentioned.
        

---


## Guild: [Nous Research AI](https://discord.com/channels/1053877538025386074)

- Members debated on the legitimacy of using 128k context lengths for AI models and questioned how to use gguf on a 4090 GPU for better efficiency.
- Technical discussions revolved around emulating ZRAM on MacOS and improving flash decoding methods, highlighted by a research paper on flash decoding.
- An admiration for Tsinghua's researchers was noted, while the application and performance of various AI models were actively discussed through benchmarks shared.
- The use of 8bit over 4bit for better efficiency was shared in a code snippet alongside exchange on the slow inference of 4bit and probable solutions to enhance the speed.
- Significant interest was displayed in trying new models, investigating their benchmarks, and understanding setbacks, with a notable focus on GPT-4's performance.
- A GitHub link was shared that exhibited example inference code using transformers for the new OpenHermes 2.5 model.
- Further inquiries were made about models fine-tuned for function calling, recommendations for quant or norm for data generation, and various user links shared for experimenting with different models.
- Focus on the incorporation of AI in art was expressed by a guild member.
- Discussions extended to the limitations, performance trade-offs, and model size concerns of quantized sizes.
- Community members also raised safety considerations in fine-tuning models, discussed continuous learning, model divergence, and the efficacy of rlhf methods.
- Laughter and humor were exhibited through programming language memes shared within the Discord community.

Relevant links:
- https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/transformers_inference.py
- https://colab.research.google.com/drive/1x8OsqBdHMUsQ5jlu_NwIPpOl1lPUWm-C?usp=sharing
- https://huggingface.co/datasets/jondurbin/airoboros-gpt4-2.0

Note: While summarizing, there were instances where the conversation context was not clear. In those cases, an assumption has been made based on guild norms and user post history. If something is unclear, please refer back to the original messages.

--- Channel by Channel Summary --- 

### Channel: [ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015)

Summary: 
1. (specific topic title, e.g. "128k Context Length Usage") (Excitement: 6/10)
    - (specific discussion thread, e.g. "Is it legit for the 128k context lengths?")
        - There is a discussion about the legitimacy of using 128k context lengths.
        - A link is shared regarding the utilization of 128k context lengths.
2. (specific topic title, e.g. "Using gguf on a 4090 GPU") (Excitement: 4/10)
    - (specific discussion thread, e.g. "Emozilla's experience with 4bit llamacpp gguf on a 4090 GPU")
        - Emozilla mentions being able to achieve up to 24~k on a 4090 GPU with 4bit llamacpp gguf.
        - ogmilady shares their experience with using gguf and looking for leads on possible issues.
    - (specific discussion thread, e.g. "Running the llama_cpp.server script locally")
        - ogmilady shares the command they used to run llama_cpp.server locally and asks for help.
3. (specific topic title, e.g. "Improvements to Flash Decoding") (Excitement: 5/10)
    - (specific discussion thread, e.g. "Apparent improvements to flash decoding")
        - ldj mentions that apparent improvements to flash decoding are already being made.
        - A link to a research paper on flash decoding is shared.
4. (specific topic title, e.g. "Tsinghua Researchers") (Excitement: 6/10)
    - (specific discussion thread, e.g. "Impressive researchers at Tsinghua")
        - conceptofmind expresses admiration for Tsinghua researchers.
    - (specific discussion thread, e.g. "Including our friend One")
        - ldj mentions that their friend "One" is among the impressive researchers.
5. (specific topic title, e.g. "Emulating ZRAM on MacOS") (Excitement: 3/10)
    - (specific discussion thread, e.g. "Code for emulating ZRAM on MacOS")
        - chadbrewbaker asks if there is code to emulate ZRAM on macOS.
        - A link to code for creating a RAM disk on macOS is shared.

Links:
- https://x.com/chillgates_/status/1720303678526267441?s=20
- https://arxiv.org/pdf/2311.01282.pdf
- https://apple.stackexchange.com/questions/461889/ram-disk-in-macos-ventura


### Channel: [off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928)

Summary: 
There were only a few messages in the off-topic channel:

1. (No specific topic mentioned) (Excitement: N/A)
   - No specific discussion thread or links were provided.

2. (No specific topic mentioned) (Excitement: N/A)
   - No specific discussion thread or links were provided.

3. (No specific topic mentioned) (Excitement: N/A)
   - No specific discussion thread or links were provided.

4. (No specific topic mentioned) (Excitement: N/A)
   - No specific discussion thread or links were provided.

5. (No specific topic mentioned) (Excitement: N/A)
   - No specific discussion thread or links were provided.

As there were no specific topics or discussions mentioned, and no excitement ratings or links provided, it is not possible to provide a detailed summary of the messages in the off-topic channel.


### Channel: [benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928)

Summary: 
1. Topic: GPT-4All Benchmark Set (Excitement: 7/10)
   - Discussion: teknium shared benchmark results for the GPT-4All model on various tasks including arc_challenge, arc_easy, boolq, hellaswag, openbookqa, piqa, and winogrande.
   - Average performance on the tasks was 73.12%.
   - Links: No specific links mentioned.

2. Topic: AGI-Eval (Excitement: 6/10)
   - Discussion: teknium shared benchmark results for AGI-Eval on tasks including aieval_aqua_rat, aieval_logiqa_en, aieval_lsat_ar, aieval_lsat_lr, aieval_lsat_rc, aieval_sat_en, aieval_sat_en_without_passage, and aieval_sat_math.
   - Average performance on the tasks was 43.07%.
   - Links: No specific links mentioned.

3. Topic: BigBench (Excitement: 7/10)
   - Discussion: teknium shared benchmark results for the BigBench model on various tasks including bigbench_causal_judgement, bigbench_date_understanding, bigbench_disambiguation_qa, bigbench_geometric_shapes, bigbench_logical_deduction_five_objects, bigbench_logical_deduction_seven_objects, bigbench_logical_deduction_three_objects, bigbench_movie_recommendation, bigbench_navigate, bigbench_reasoning_about_colored_objects, bigbench_ruin_names, bigbench_salient_translation_error_detection, bigbench_snarks, bigbench_sports_understanding, bigbench_temporal_sequences, bigbench_tracking_shuffled_objects_five_objects, bigbench_tracking_shuffled_objects_seven_objects, and bigbench_tracking_shuffled_objects_three_objects.
   - Average performance on the tasks was 40.96%.
   - Links: No specific links mentioned.

4. Topic: TruthfulQA (Excitement: 5/10)
   - Discussion: teknium shared benchmark results for the TruthfulQA model on the truthfulqa_mc task.
   - Accuracy for mc1 was 35.99% and for mc2 was 53.04%.
   - Links: No specific links mentioned.

5. Topic: DeepSeek Models (Excitement: 3/10)
   - Discussion: gabriel_syme asked if anyone has tested deepseek models. No further discussion or links were provided.


### Channel: [interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192)

Summary: 
1. Link: GPT-4 and partial completion (Excitement: N/A)
2. Link: Modify attention heads to query DB and outperform RAG (Excitement: N/A)
3. Link: Reference implementation for modified attention heads (Excitement: N/A)
4. Discussion: Appreciation for research with code and reproducibility (Excitement: N/A)
5. Link: Twitter thread on the potential of a new model surpassing GPT-3 (Excitement: N/A)
6. Discussion: Speculation and skepticism about surpassing GPT-4 (Excitement: N/A)
7. Link: Model with 34B parameters and high MMLU (Excitement: N/A)
8. Discussion: Confidence and skepticism about new model performance (Excitement: N/A)
9. Discussion: Questions on benchmark alignment and replicability (Excitement: N/A)
10. Link: Dataset contamination accusation (Excitement: N/A)
11. Discussion: Uncertainty and desire for more information (Excitement: N/A)
12. Discussion: Interest in trying out the new models (Excitement: N/A)
13. Discussion: Skepticism about future model improvements (Excitement: N/A)
14. Assorted humor and banter (Excitement: N/A)
15. Link: Benchmarks for GPT-4 and potential decline (Excitement: N/A)
16. Discussion: Discussion on GPT-4 performance in different tasks (Excitement: N/A)
17. Discussion: Experiences with OpenAI API and chat interface (Excitement: N/A)
18. Discussion: Strategies and challenges with using LLMs (Excitement: N/A)
19. Link: Paper on "Cringe Loss" (Excitement: N/A)
20. Discussion: Remarks on paper title and zoomer culture (Excitement: N/A)
21. Link: Paper on a different topic (Excitement: N/A)
22. Discussion: Iterative Newton's impact on LLMs (Excitement: N/A)
23. Discussion: Newton's convergence compared to SGD (Excitement: N/A)
24. Link: Another paper on a different topic (Excitement: N/A)
25. Discussion: Influence of emotions on LLMs (Excitement: N/A)


### Channel: [bots](https://discord.com/channels/1053877538025386074/1149866614590816256)

Summary: 
1. "Código em Python para gerar a sequência de Fibonacci" (Excitement: 7/10)
    - Discussion points and quotes:
        - Code snippet shared by <@597576664809078814> generates the Fibonacci sequence in Python
        - The code uses a while loop and a list to store the sequence
        - Sample output for the first 10 numbers in the Fibonacci sequence is provided
    - Links:
        - No links mentioned in the messages.


### Channel: [general](https://discord.com/channels/1053877538025386074/1149866623109439599)

Summary: 
1. gabriel_syme: Which open source models support are fine-tuned for function calling? Is Airoboros the only one?
2. teknium: Added example inference code with transformers to openhermes 2.5 repo: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/transformers_inference.py
3. yorth_night: top tier prompt
4. yorth_night: also device map = device map = auto
5. teknium: woops lol fixed
6. gabriel_syme: Would you recommend quant or normal for data gen? Noticeable difference?
7. teknium: 8bit in gguf seems 99.9% as good as fp16
8. yorth_night: 4bit does suffer some
9. yorth_night: https://colab.research.google.com/drive/1x8OsqBdHMUsQ5jlu_NwIPpOl1lPUWm-C?usp=sharing made a colab to run it
10. gabriel_syme: Awesome, I'm guessing awq similar right
11. gabriel_syme: Cause I'm doing vllm for 🏃‍♂️
12. yorth_night: btw, is there a way to solve this: warnings.warn(f'Input type into Linear4bit is torch.float16, but bnb_4bit_compute_type=torch.float32 (default). This will lead to slow inference or training speed.')
13. yorth_night: the inference does seem quite slow
14. yorth_night: it took 6 mins
15. yorth_night: maybe its just a 4 bit issue, I'll try with 8 bit
16. teknium: hmm, I've never seen that
17. teknium: let me see if it happens on my PC
18. yorth_night: maybe it has to do with my installation too
19. yorth_night: tested with 8 bit
20. yorth_night: it works fine with that
21. yorth_night: the stuff it generated was hilarious
22. gabriel_syme: I miss old political cartoons
23. euclaise: Change bnb_4bit_compute_dtype? Like it says so there
24. yorth_night: the inference does seem quite slow
25. teknium: lol
26. mars_eve: I know ZERO coding.. but I love to push the boundaries with art.. so PLZ loop me into any vision or art related aspects. Cheers!!!
27. yorth_night: Are there any links for these?
28. teknium: the dataset isn't out yet
29. teknium: Try sharegpt
30. giftedgummybee: openhermes2
31. euclaise: FLAN
32. giftedgummybee: 🧌
33. yorth_night: any link, there seems to be a lot of different datasets with that name
34. euclaise: Yeah but it's 500gb
35. yorth_night: <:stare:1019075524070481990>
36. giftedgummybee: real
37. yorth_night: I'm thinking of just finetuning yarn mistral on a simple dataset that everyone uses for great results
38. yorth_night: just to get a proof of concept
39. euclaise: Airoboros
40. yorth_night: https://huggingface.co/datasets/jondurbin/airoboros-gpt4-2.0 this one?
41. euclaise: The newer one
42. euclaise: I think he's at 3.1 or 3.2 now
43. yorth_night: oh, found it
44. yorth_night: 123 mb
45. yorth_night: cool stuff
46. euclaise: If you want something smaller, try everythinglm or dove
47. yorth_night: I just want something where an epoch isn't a full day
48. yorth_night: on an A100
49. euclaise: Airoboros shouldn't be a full day on an A100, I can do it on a 3090 in a day
50. yorth_night: yeah, colab does some weird stuff when you let it run for more than a day
51. yorth_night: how many epochs is usually good?
52. yorth_night: 3?
53. euclaise: 3 is a little


### Channel: [welcomes](https://discord.com/channels/1053877538025386074/1151415076033658931)

Summary: 
There were no specific topics or discussion points mentioned in the messages in the "welcomes" channel. Therefore, there are no links, blogposts, or excitement ratings to report.


### Channel: [ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927)

Summary: 
1. Llama.cpp with ROCm support (Excitement: 7/10)
    - Adjectiveallison shares their experience of using llama.cpp with ROCm support on a RX580 GPU.
    - They are impressed with the performance and the ability to run models locally.
    - They mention using different quantizations (Q2, Q6, and Q8) and observe a difference in token output speed.
    - They inquire about the expected behavior and the mechanism causing smaller quantized models to be slower.

2. Support for AMD with llama.cpp (Excitement: 5/10)
    - Teknium expresses surprise about the support for AMD with llama.cpp.

3. Performance and limitations of quantized sizes (Excitement: 6/10)
    - Adjectiveallison discusses the speed differences between smaller and larger quantized models and wonders about the underlying mechanism.
    - Giftedgummybee suggests that the slower speeds could be due to limitations in ROCm.
    - Wesh3104 suggests that the slower speed of Q6 compared to Q8 could be related to the quantization size.

4. Model sizes and speed trade-offs (Excitement: 4/10)
    - Adjectiveallison mentions trying different quantized sizes and expresses concerns about loss and model size.
    - Max_paperclips shares their preferred quantized size based on their laptop's performance.

5. Discussion on the best 7b model (Excitement: 5/10)
    - Jacquesthibs asks about the best current 7b model and if there is a fine-tuned version.
    - Teknium confirms that the Mistral model is currently available.
    - Jacquesthibs inquires about the fine-tuned chat version and suggests Nous-hermes-llama-2.

6. Safety considerations in fine-tuning models (Excitement: 4/10)
    - Jacquesthibs raises questions about the safety of instruct-tuned models and the potential for "jailbreaking."
    - Tecknium clarifies that safety measures were taken in the training of Hermes to exclude refusals.
    - Jacquesthibs explores the idea of running experiments to compare instruct-tuned and rlhf models in terms of their responses to interventions.

7. Continuous learning and model divergence (Excitement: 4/10)
    - Jacquesthibs discusses the challenges of continual learning and online learning.
    - Tecknium expresses skepticism about rlhf methods and suggests that current approaches might not be sufficient.
    - Jacquesthibs shares a paper on continual learning for further reference.

8. Tokenization in Mistral and llama-2 (Excitement: 3/10)
    - Jacquesthibs asks if Mistral and llama-2 use the same tokenizer.
    - Teknium confirms that they do.

Note: The excitement ratings are subjective and based on the overall tone and level of engagement in the conversation.


### Channel: [memes](https://discord.com/channels/1053877538025386074/1166105758635655270)

Summary: 
1. (specific topic title, e.g. "Memes about programming languages") (Excitement: 6/10)
    - (specific discussion thread, e.g. "Python vs JavaScript memes")
    - (specific discussion thread, e.g. "C++ memes")
    - Links:
        - {link 1 discussed in the source messages}
        - {link 2 discussed in the source messages}
        - {link 3 discussed in the source messages}
        

---


## Guild: [Alignment Lab AI](https://discord.com/channels/1087862276448595968)

- Transition around frontier models and distillation techniques, exploring whether Additional Reinforcement has a potential to boost the performance of distilled models, and the feasibility of distilling into increasingly smaller models.
- Expectations and speculations about the forthcoming GPT-4 and GPT-5 models, along with the discussion about the utilization of synthetic data mixes in their training.
- Comparisons between GPT-3 and other AI models, like the Llama 2 70b and Mistral 7b. Few members suggested that despite being smaller, these alternative models might outperform the larger GPT-3.
- A link from Reddit on LocalLLaMA was shared discussing the possible development of an open-source GPT-3 on Dev:
  - https://www.reddit.com/r/LocalLLaMA/comments/17mascq/we_could_be_getting_an_open_source_gpt3_on_dev/
- Examination of OpenChat 3.5 relative to Amazon's MistalLite. OpenChat 3.5 was noted to be slightly faster and can run on a Mac without Python dependencies.
- A shared resource on tokenization and comparison of model outputs, with incorporation of embeddings and cosine similarity as the method to measure the difference between expected and model outputs:
  - https://chat.openai.com/share/b6474b79-61d8-451a-b3a4-9ba3c574d424
- Sharing of Twitter link related to OpenChat 3.5, although not accompanied by additional discussion or details:
  - https://fxtwitter.com/Teknium1/status/1720188958154625296
- A reference to a Twitter link from the user's Mac experience running OpenChat 3.5:
  - https://twitter.com/realwasmedge/status/1720297412235804853
- A general request from a user to another to check their DMs.

--- Channel by Channel Summary --- 

### Channel: [general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841)

Summary: 
1. Scoot: Compare to GPT4? (Excitement: 0/10)
   - No further discussion or details provided.

2. ufghfigchv: Tokenization and comparison (Excitement: 0/10)
   - Shared a link (https://chat.openai.com/share/b6474b79-61d8-451a-b3a4-9ba3c574d424) discussing tokenization and comparison of model outputs.
   - Mentioned the use of embeddings and cosine similarity for measuring the difference between expected and model outputs.

3. teknium: Twitter link about OpenChat 3.5 (Excitement: 0/10)
   - Shared a link (https://fxtwitter.com/Teknium1/status/1720188958154625296) related to OpenChat 3.5.
   - No further discussion or details provided.

4. vivian_hu: Comparing OpenChat 3.5 to MistalLite (Excitement: 0/10)
   - Successfully ran OpenChat 3.5 on a Mac without Python dependencies.
   - Mentions that OpenChat 3.5 is slightly faster than Amazon's MistalLite.
   - Shared a Twitter link (https://twitter.com/realwasmedge/status/1720297412235804853) for reference.

5. fredipy: DM request (Excitement: 0/10)
   - Sent a message to another user requesting them to check their DMs.

No high levels of excitement were expressed in the given messages. The conversations primarily revolved around technical details, comparisons, and sharing of relevant links.


### Channel: [oo](https://discord.com/channels/1087862276448595968/1118217717984530553)

Summary: 
1. "Applied use cases, but also I’m not convinced you can’t go beyond what model you distill from distillation, especially if you incorporate rag if the model is sufficiently powerful, I’ve been given several leaks that gpt4 further trains on synthetic data mixes that it generates itself, but synthesizing more from rag is a viable option as well, and also, gpt5 will be out soon enough" (Excitement: 5/10)

2. "I guess we're viewing things from 2 different perspectives. My point is more about frontier models vs distilling from them. I agree with your point that in the case of distillation it's at this point more beneficial to just do sft. My point since the initial conversation has been that frontier models needed to go beyond sft to be more aligned shows a limitation of sft. So far there hasnt been any counterexamples to show evidence to the contrary" (Excitement: 3/10)

3. "Is gpt 3 175b useful at all now?" (Excitement: 2/10)

4. "Large and dumb 🙁" (Excitement: 1/10)

5. "Dumber than llama 2 70b which is 2-3x smaller" (Excitement: 2/10)

6. "It's even questionable that whether it's better than Mistral 7b" (Excitement: 1/10)

7. "No need of any questions it is definitely not" (Excitement: 1/10)

8. Link: [Reddit post on LocalLLaMA about getting an open-source GPT3 on Dev](https://www.reddit.com/r/LocalLLaMA/comments/17mascq/we_could_be_getting_an_open_source_gpt3_on_dev/) (Excitement: 3/10)
        

---
This guild has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---


## Guild: [LangChain AI](https://discord.com/channels/1038097195422978059)

- Introduction of the new course on using LangChain.js on the platform Scrimba, covering expression language, retrieval, and adding chat history. Direct course link was shared: [Scrimba Course](https://scrimba.com/learn/langchain?newLaunch)  
- Announcements of two new channels in the Discord guild for questions, comments, and ideas related to LangServe.
- Several queries and solutions related to LangChain popped up across channels including:  
  - Creating an agent that uses Vector database
  - Calculating tokens in JavaScript using LangChain
  - Deprecation of LangChain.chat_models and how to import ChatOpenAI from it
  - Calling ConversationalRetrievalChain with a Qdrant vector store as retriever
  - The ability of LangChain in supporting the build of an app using TypeScript, SQL, and Go. A linked document was mentioned for reference: Python.langchain.com/docs/integrations/document_loaders/source_code
  - Handling complex filtering sequences in an agent
  - Public GitHub repositories using LangChain or similar frameworks, with a public repository name 'Dataherald' being mentioned. Link: Github.com/Dataherald
  - Improving Retrieval-Augmented Generation (RAG) with a focus on the codebase retrieval aspect. A blog post was discussed in the context: Blog.continue.dev/accuracy-limits-of-codebase-retrieval/
  - Creating a wrapper class around the Tool class in LangChain
  - Whether the OpenAI_functions agent can be forced to use specific tools
  - Using a separate chain for determining tool requirements
  - The possibility of modifying the thought process of agents by creating a new class that inherits from Conversational Agent and overrides some functions
- Assistance requests regarding errands and issues encountered during coding, mainly in environmental setup of LangServe, and application of LangChain in a chatbot.
- User shared works: Calvix1 introduced a chess AI web application made with Next.js and LangChain, and Thebigbigbuddha discussed ongoing projects at Manifold Research Group including multimodal AI, robotics and biology.
- Conduct reminders and small talk.

--- Channel by Channel Summary --- 

### Channel: [announcements](https://discord.com/channels/1038097195422978059/1058033358799655042)

Summary: 
1. Course Launch (Excitement: 8/10)
    - Jacoblee93 announced a new interactive course on using the latest in LangChain.js
    - The course covers expression language, retrieval, and adding chat history
    - Scrimba is the platform offering the course
    - Link to the course: [Scrimba Course](https://scrimba.com/learn/langchain?newLaunch)

2. New Channels for LangServe (Excitement: 4/10)
    - Hwchase17 created a new channel, <#1170024642245832774>, for questions regarding LangServe
    - Another channel, <#1170025009960456282>, was created for questions, comments, and ideas related to LangServe


### Channel: [general](https://discord.com/channels/1038097195422978059/1038097196224086148)

Summary: 
1. Topic: Help needed with creating an agent that uses Vector database (Excitement: 5/10)
    - Discussion points and quotes:
        - A user asked for assistance in creating an agent that utilizes Vector database.
    - No links or specific discussion threads provided.

2. Topic: Token calculation in LangChain in JavaScript (Excitement: 7/10)
    - Discussion points and quotes:
        - A user requested help with token calculation in JavaScript using LangChain.
    - No links or specific discussion threads provided.

3. Topic: Deprecation of langchain.chat_models (Excitement: 4/10)
    - Discussion points and quotes:
        - A user asked if langchain.chat_models is deprecated.
    - No specific discussion threads or links provided.

4. Topic: Importing ChatOpenAI from langchain.chat_models (Excitement: 3/10)
    - Discussion points and quotes:
        - A user mentioned importing ChatOpenAI from langchain.chat_models.
    - No specific discussion threads or links provided.

5. Topic: Calling ConversationalRetrievalChain with a Qdrant vector store as retriever (Excitement: 6/10)
    - Discussion points and quotes:
        - A user asked how to call ConversationalRetrievalChain by passing a Qdrant vector store as a retriever.
    - No specific discussion threads or links provided.

6. Topic: Building an app for chat with TypeScript, SQL, and Go (Excitement: 5/10)
    - Discussion points and quotes:
        - A user inquired if LangChain supports building an app using TypeScript, SQL, and Go.
    - Links:
        - Python.langchain.com/docs/integrations/document_loaders/source_code

7. Topic: Handling complex filtering sequences in an agent (Excitement: 7/10)
    - Discussion points and quotes:
        - A user shared the challenge of an agent struggling to handle complex filtering sequences.
        - They asked for advice on improving the agent's capability in managing such situations.
    - No links or specific discussion threads provided.

8. Topic: GitHub repositories for startups using LangChain (Excitement: 6/10)
    - Discussion points and quotes:
        - A user asked if there are publicly visible GitHub repositories of startups using LangChain or similar frameworks.
        - They mentioned coming across the repository "Dataherald" that leverages LangChain and expressed interest in finding more examples.
    - Links:
        - Github.com/Dataherald

9. Topic: RAG (Retrieval-Augmented Generation) over a codebase (Excitement: 8/10)
    - Discussion points and quotes:
        - A user shared their experience with codebase retrieval and expressed interest in improving RAG by updating embeddings based on user feedback.
        - They mentioned the potential of predefined user documents and annotations for refining the results.
        - They asked if anyone has tried this approach.
    - Links:
        - Blog.continue.dev/accuracy-limits-of-codebase-retrieval/

10. Topic: Wrapper class for tools in LangChain (Excitement: 6/10)
    - Discussion points and quotes:
        - A user described creating a wrapper class around the Tool class in LangChain to enable data storage and retrieval when using tools.
        - They provided an example of wrapping the Reddit API to store both the LLM response and the Reddit posts used.
    - No links or specific discussion threads provided.

11. Topic: Forcing the OpenAI_functions agent to use specific tools (Excitement: 4/10)
    - Discussion points and quotes:
        - A user asked if there is a way to force the OpenAI_functions agent to use specific tools.
    - No specific discussion threads or links provided.

12. Topic: Using a separate chain to determine tool requirements (Excitement: 5/10)
    - Discussion points and quotes:
        - A user suggested sending a user message into a separate chain to ask the model if a specific tool is required.
        - They mentioned resending the request to the agent, specifying the tool to be used.
    - No specific discussion threads or links provided.

13. Topic: Modifying the thought process of agents by creating a custom agent class (Excitement: 5/10)
    - Discussion points and quotes:
        - A user mentioned testing a solution by creating a custom agent class that inherits from the Conversational Agent and overrides some of its functions.
        - They specifically mentioned modifying the plan and aplan functions.
    - No specific discussion threads or links provided.

14. Error: Requesting help with an error (Excitement: 3/10)
    - Discussion points and quotes:
        - A user asked for assistance with an error they encountered, without providing specific details.
    - No specific discussion threads or links provided.

15. General announcements and conversations (Excitement: 3/10)
    - Various users made general announcements, shared articles, and engaged in conversations.
    - No specific discussion threads or links provided.


### Channel: [langserve](https://discord.com/channels/1038097195422978059/1170024642245832774)

Summary: 
1. (topic: Setting Environment Variables in LangServe) (Excitement: 5/10)
   - User "attila_ibs" is experiencing difficulties setting environment variables after installing templates in LangServe.
   - User asks where to create the `.env` file with the environment variables.
   - User mentions trying to place the file in the root directory and the `my-app/app` directory, but it doesn't work.

2. (topic: Feedback and Issues with LangServe) (Excitement: 3/10)
   - User "veryboldbagel" requests feedback and asks for any issues encountered with LangServe.


### Channel: [share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729)

Summary: 
1. Building a Chess vs AI web with Next.js and LangChain (Excitement: 7/10)
    - Calvix1 shared a link to their work-in-progress chess vs AI web application
    - The project is built with Next.js and LangChain
    - Major features have been completed, but an agent to play the game needs to be implemented

2. Manifold Research Group working on multimodal AI projects (Excitement: 5/10)
    - Thebigbigbuddha introduced themselves as a member of Manifold Research Group
    - The group is focused on AI projects, particularly in the area of multimodal AI
    - They mentioned two active projects and plans for future projects in robotics and biology
    - The group's website and GitHub page were shared for more information


### Channel: [tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538)

Summary: 
1. "Token calculation in LangChain to JavaScript" (Excitement: 4/10)
    - mohitsakhiya077 is seeking help with calculating tokens in LangChain for the JavaScript language.

2. "Building a chatbot with LangChain" (Excitement: 5/10)
    - hamza_sarwar_ is working on a chatbot and wants to create different responses based on whether the question is related to a specific document or a general conversation.
    - They are interested in using LangChain to achieve this functionality.
        

---
This guild has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
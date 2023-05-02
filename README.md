[TOC]

[重要论文的总结](https://docs.google.com/presentation/d/13k5cs4p_OmMKkNB9YVuF2CkzZ3PTwUV5bcAIHL4sqjk/edit?usp=sharing)



- [Papers](#Papers)



# Papers

## Base Models

1. `LLaMA: Open and Efficient Foundation Language Models.` arxiv 2023
   Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. [[pdf]](https://arxiv.org/abs/2302.13971v1)

2. `Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling` 2023
   Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, Oskar van der Wal [[paper](https://arxiv.org/abs/2304.01373)] [[project](https://github.com/EleutherAI/pythia)]

   Model Size: 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B

3. `Opt-iml: Scaling language model instruction meta learning through the lens of generalization.` arxiv 202
   Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Da ́niel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Koura, et al. [[paper](https://arxiv.org/abs/2212.12017)]
4. 



## Method to Get Training Data

1. `Self-Instruct: Aligning Language Model with Self Generated Instructions` arXiv 2022
   *Wang, Yizhong , Kordi, Yeganeh , Mishra, Swaroop , Liu, Alisa , Smith, Noah A. , Khashabi, Daniel , Hajishirzi, Hannaneh*  [[pdf](https://arxiv.org/pdf/2212.10560.pdf)] [[project](https://github.com/yizhongw/self-instruct)]
2. 



## 实现ChatGPT的平替

> 主要跟Instruction Tuning有关，即构建 Instruction + N个Input-output examples。
> Alpaca, Vicuna, Dolly都没有使用RLHF

1. `Alpaca: A Strong, Replicable Instruction-Following Model` 2023
   Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto  [[blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[project](https://github.com/tatsu-lab/stanford_alpaca)]
   基于LLaMa训练的模型，训练数据来自于text-davinci-003的标注

2. `Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality` 2023
   Chiang, Wei-Lin, Li, Zhuohan, Lin, Zi, Sheng, Ying, Wu, Zhanghao, Zhang, Hao, Zheng, Lianmin, Zhuang, Siyuan, Zhuang, Yonghao, Gonzalez, Joseph E., Stoica, Ion, Xing, Eric P. [[blog](https://vicuna.lmsys.org/)] [[project](https://github.com/lm-sys/FastChat)]
   基于LLaMa训练的模型，训练数据来自于chatgpt的标注(share gpt)

3. `Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM`
   Mike Conover, Matt Hayes, Ankit Mathur, Xiangrui Meng, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, Reynold Xin [[blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)] [[project](https://github.com/databrickslabs/dolly)]

   基于Pythia 12b，训练数据来自于DataBricks员工的标注 `databricks-dolly-15k`

4. 



## PEFT (Parameter Efficient Fine-Tuning)



## Plug-And-Play



## 评估LLMs

1. `Instruction Tuning with GPT-4` arxiv 2023
   Peng, Baolin, Li Chunyuan , He Pengcheng , Galley Michel , Gao Jianfeng [[blog](https://instruction-tuning-with-gpt-4.github.io/)] [[paper](https://arxiv.org/pdf/2304.03277.pdf)] [[project](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)]
   
   基于LLaMA 7B进行SFT，基于OPT 1.3B训练得到Reward Model。
   
   - Instruction效果的评估，使用GPT-4生成Instruction-following data
   - 3种评估指标：
     - human evaluation on three alignment criteria: **Helpfulness** (比如越正确、越相关，答案可能就越有帮助), **Honesty**）（是否有虚假信息）, **Harmlessness**（是否有仇恨、暴力的内容）
     - automatic evaluation using GPT-4 feedback,  [让GPT-4给结果从1-10打分]
     - **ROUGE-L** on un-natural instructions
   
2.  `Holistic Evaluation of Language Models` arxiv 2023
   Liang P, Bommasani R, Lee T, et al. [[paper](https://arxiv.org/pdf/2211.09110.pdf)] [[project](https://github.com/stanford-crfm/helm)]

3. `Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models` arxiv 2023
   Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, et al. [[paper](https://arxiv.org/abs/2206.04615)] [[project](https://github.com/google/BIG-bench)]

## LLM带来的新方向



## Model Explanation

> 在模型中加入可解释性模块



## 大模型训练的技巧

硬件层面

-  **Flash Attention** (Dao et al., 2022)

模型层面

- **rotary embeddings**  (Su et al. 2021)

- **parallelized attention and feedforward technique** (Wang & Komatsuzaki 2021)
- **untied embedding / unembedding matrices** (Belrose et al., 2023)



# Tutorials



# Resources

- [LLM Zoo](https://github.com/FreedomIntelligence/LLMZoo): 

## Data

- [InstructionZoo](https://github.com/FreedomIntelligence/InstructionZoo): 各种Instruction-Tuning的数据集

- [awesome-chatgpt-dataset](https://github.com/voidful/awesome-chatgpt-dataset/): 各种Instruction-Tuning的数据集

- [Alpaca Training Data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

- [Vicuna Training Data: ShareGPT](https://github.com/lm-sys/FastChat/issues/90)

- Dolly 2.0 Training Data: [databricks-dolly-15k](https://github.com/databrickslabs/dolly/tree/master/data)



## Online Demo

- [Vicuna, Koala, FastChat-T5, OpenAssistant, ChatGLM, StableLM, Alpaca, LLaMa, Dolly](https://chat.lmsys.org/)



## Models




# TODO

Decoupling Knowledge From Memorization Retrieval-augmented Prompt Learning

[神奇LLM引擎上线：帮你把GPT-3直接调成ChatGPT](https://mp.weixin.qq.com/s/eBFjLfyLycdMIF6-ucgy1w)

[清华唐杰教授：从千亿模型到ChatGPT的⼀点思考](https://mp.weixin.qq.com/s/25cxLdYd37DHw6-UpZlayw)

# Else

`Language Models as Knowledge Bases?` EMNLP 2019.

*Fabio Petroni, Tim Rocktaschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel.* 2019.9 [[pdf](https://arxiv.org/abs/1909.01066)] [[project](https://arxiv.org/abs/1909.01066)]
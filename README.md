<h1 align="center">🌐 A Systematic Survey of Deep Research </h1>

> We will continuously update this repo.

If you like our project, please give us a star ⭐ on GitHub for the latest update. 

# Overview

# 📣 Latest News
[2025.11.25] 🎉🎉🎉 We release our survey [Deep Research: A systematic Survey](https://github.com/mangopy/Deep-Research-Survey/blob/main/Deep-Research-Survey.pdf). Thanks to my awesome co-authors🤩. Feel free to contact me if you are interested in this topic and want to discuss me.

# 🎬 Table of Content
> will be updated as soon as possible!

 ## 📑 Table of Contents
- [🌟 Overview](#-overview)
- [📊 Latest News](#-latest-news)
- [📚 Reading List](#-reading-list)
  - [Query Planning](#query-planning)
  - [Knowledge Acquisition](#knowledge-acquisition)
    - [Knowledge Boundary](#knowledge-boundary)
    - [Retrieval Timing](#timing)
  - [Agentic End-to-End Reinforcement Learning](#agentic-end-to-end-reinforcement-learning)
  - [Supervised Fine-tuning](#sft)
  - [Datasets & Benchmarks](#datasets--benchmarks)
 - [Acknowledgement](#ack)
 - [Contact](#contact)
 - [Citation](#citation)


# 📚 Reading List
> will be updated as soon as possible!

To get started with Deep Research, we recommend the representative and often seminal papers listed below. Reviewing this selection will provide a solid overview of the field.

## Query Planning
| Venue      | Date         | Paper Title                                                                 | URL |
|------------|--------------|------------------------------------------------------------------------------|-----|
| ICLR 2023  | 21 May 2022  | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models  | https://arxiv.org/abs/2205.10625 |
| NeurIPS 2023 | 17 May 2023 | Tree of Thoughts: Deliberate Problem Solving with Large Language Models     | https://arxiv.org/abs/2305.10601 |
| ACL 2024   | 21 Jun 2024  | Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering | https://aclanthology.org/2024.acl-long.397/ |
| Arxiv      | 20 Sep 2023  | Chain-of-Verification Reduces Hallucination in Large Language Models         | https://arxiv.org/abs/2309.11495 |
| EMNLP 2023 | 23 May 2023  | Query Rewriting for Retrieval-Augmented Large Language Models                | https://aclanthology.org/2023.emnlp-main.322/ |
| COLM 2025  | 28 Feb 2025  | DeepRetrieval: Hacking Real Search Engines and Retrievers with LLMs via RL  | https://arxiv.org/abs/2503.00223 |
| Arxiv      | 11 Oct 2025  | CardRewriter: Leveraging Knowledge Cards for Long-Tail Query Rewriting on Short-Video Platforms | https://arxiv.org/abs/2510.10095 |
| NeurIPS 2025 | 25 Jan 2025 | Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning | https://arxiv.org/abs/2501.15228 |
| NAACL 2024 | 14 Nov 2023  | LLatrieval: LLM-Verified Retrieval for Verifiable Generation                 | https://aclanthology.org/2024.naacl-long.305/ |
| ACL 2024   | —            | DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of LLMs | https://aclanthology.org/2024.acl-long.702/ |
| WWW 2025   | 18 Jul 2024  | Retrieve, Summarize, Plan: Advancing Multi-hop QA with an Iterative Approach | https://dl.acm.org/doi/10.1145/3701716.3716889 |
| Arxiv      | 10 Jun 2025  | RAISE: Enhancing Scientific Reasoning in LLMs via Step-by-Step Retrieval    | https://arxiv.org/abs/2506.08625 |
| Arxiv      | 20 May 2025  | s3: You Don’t Need That Much Data to Train a Search Agent via RL            | https://arxiv.org/abs/2505.14146 |
| Arxiv      | 28 Aug 2025  | AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective RL | https://arxiv.org/abs/2508.20368 |
| COLM 2025  | 12 Mar 2025  | Search-r1: Training LLMs to Reason and Leverage Search Engines with RL      | https://arxiv.org/abs/2503.09516 |
| Arxiv      | 7 Mar 2025   | R1-Searcher: Incentivizing the Search Capability in LLMs via RL             | https://arxiv.org/abs/2503.05592 |
| Arxiv      | 22 May 2025  | R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via RL | https://arxiv.org/abs/2505.17005 |
| NAACL 2025 | 17 Dec 2024  | RAG-Star: Enhancing Deliberative Reasoning with Retrieval-Augmented Verification and Refinement | https://aclanthology.org/2025.naacl-long.361/ |
| ACL 2025   | 21 Jan 2025  | Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation | https://aclanthology.org/2025.acl-long.1401/ |
| Arxiv      | 29 Jul 2025  | DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router               | https://arxiv.org/abs/2507.22050 |
| Arxiv      | 3 Feb 2025   | DeepRAG: Thinking to Retrieve Step by Step for Large Language Models        | https://arxiv.org/abs/2502.01142 |
| Arxiv      | 1 Aug 2025   | MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation | https://arxiv.org/abs/2508.01005 |

## Information Acquisition

### Knowledge Boundary
| Venue              | Date         | Paper Title                                                  | URL                                  |
| ------------------ | ------------ | ------------------------------------------------------------ | ------------------------------------ |
| ICML 2017          | 14 Jun 2017  | On Calibration of Modern Neural Networks                     | https://arxiv.org/abs/1706.04599     |
| EMNLP 2020         | 17 May 2020  | Calibration of Pre-trained Transformers                      | https://arxiv.org/pdf/2003.07892     |
| TACL 2021          | 2 Dec 2020   | How Can We Know When Language Models Know? On the Calibration of Language Models for Question Answering | https://arxiv.org/abs/2012.00955     |
| Anthropic          | 11 Jul 2022  | Language Models (Mostly) Know What They Know                 | https://arxiv.org/abs/2207.05221     |
| ACL 2024           | 3 Jul 2023   | Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models | https://arxiv.org/abs/2307.01379     |
| ICLR 2023          | 19 June 2024 | Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation | https://arxiv.org/abs/2302.09664     |
| EMNLP 2023         | 15 Mar 2023  | Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models | https://arxiv.org/abs/2303.08896     |
| EMNLP 2023         | 3 Nov 2023   | SAC3: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency | https://arxiv.org/abs/2311.01740     |
| EMNLP 2023         | 26 Apr 2023  | The Internal State of an LLM Knows When It's Lying           | https://arxiv.org/pdf/2304.13734     |
| ACL 2025           | 17 Feb 2025  | Towards Fully Exploiting LLM Internal States to Enhance Knowledge Boundary Perception | https://www.arxiv.org/abs/2502.11677 |
| TMLR 2022          | 28 May 2022  | Teaching Models to Express Their Uncertainty in Words        | https://arxiv.org/abs/2205.14334     |
| EMNLP 2023         | 24 May 2023  | Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback | https://arxiv.org/abs/2305.14975     |
| ICLR 2024          | 22 Jun 2023  | Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs | https://arxiv.org/abs/2306.13063     |
| NAACL 2024         | 7 Jun 2024   | R-Tuning: Instructing Large Language Models to Say ‘I Don’t Know’ | https://arxiv.org/pdf/2311.09677     |
| NeurIPS 2023       | 12 Dec 2023  | Alignment for Honesty                                        | https://arxiv.org/abs/2312.07000     |
| Arxiv              | 20 Oct 2025 | Annotation-Efficient Universal Honesty Alignment             | https://arxiv.org/abs/2510.17509     |

### Retrieval Timing
| Venue              | Date        | Paper Title                                                  | URL                                  |
| ------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------ |
| EMNLP 2023         | 11 May 2023 | Active Retrieval Augmented Generation                        | https://arxiv.org/abs/2305.06983     |
| ACL 2024           | 18 Feb 2024 | When Do LLMs Need Retrieval Augmentation? Mitigating LLMs' Overconfidence Helps Retrieval Augmentation |                                      |
| ACL 2024           | 12 Mar 2024 | DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of Large Language Models | https://arxiv.org/pdf/2403.10081     |
| SIGIR-AP 2025      | 16 Feb 2024 | Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models | https://arxiv.org/abs/2402.10612     |
| ACL 2025           | 29 May 2024 | CtrlA: Adaptive Retrieval-Augmented Generation via Inherent Control | https://arxiv.org/abs/2405.18727     |
| EMNLP 2024         | 18 Jun 2024 | Unified active retrieval for retrieval augmented generation  | https://arxiv.org/abs/2406.12534     |
| ICLR 2023          | 6 Oct 2022  | ReAct: Synergizing Reasoning and Acting in Language Models   | https://arxiv.org/abs/2210.03629     |
| ACL 2023           | 20 Dec 2022 | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions | https://arxiv.org/abs/2212.10509     |
| ICLR 2024          | 17 Oct 2023 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | https://arxiv.org/abs/2310.11511     |
| EMNLP 2025         | 9 Jan 2025  | Search-o1: Agentic Search-Enhanced Large Reasoning Models    | https://arxiv.org/abs/2501.05366     |
| COLM 2025          | 12 Mar 2025 | Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | https://arxiv.org/abs/2503.09516     |
| EMNLP 2025         | 22 May 2025 | Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty | https://arxiv.org/abs/2505.17281     |
| EMNLP 2025         | 21 May 2025 | StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization | https://arxiv.org/abs/2505.15107     |


## Agentic End-to-End Reinforcement Learning

| Venue      | Date         | Paper Title                                                                | URL |
|------------|--------------|----------------------------------------------------------------------------|-----|
| Arxiv      | 30 Sep 2025  | Planner-R1: Reward Shaping Enables Efficient Agentic RL with Smaller LLMs  | https://arxiv.org/abs/2509.25779 |
| Arxiv      | 21 May 2025  | ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via RL | https://arxiv.org/abs/2505.15776 |
| AAAI 2026  | 22 Aug 2025  | OPERA: A RL-Enhanced Orchestrated Planner-Executor Architecture for Multi-Hop Retrieval | https://arxiv.org/abs/2508.16438 |
| Arxiv      | 1 Aug 2025   | MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation | https://arxiv.org/abs/2508.01005 |
| Arxiv      | 28 Aug 2025  | AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective RL | https://arxiv.org/abs/2508.20368 |
| Arxiv      | 7 Mar 2025   | R1-Searcher: Incentivizing the Search Capability in LLMs via RL            | https://arxiv.org/abs/2503.05592 |
| Arxiv      | 22 May 2025  | R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via RL | https://arxiv.org/abs/2505.17005 |
| Arxiv      | 4 Apr 2025   | DeepResearcher: Scaling Deep Research via RL in Real-world Environments    | https://arxiv.org/abs/2504.03160 |
| Arxiv      | 25 Jun 2025  | MMSearch-R1: Incentivizing LMMs to Search                                  | https://arxiv.org/abs/2506.20670 |
| COLM 2025  | 12 Mar 2025  | Search-r1: Training LLMs to Reason and Leverage Search Engines with RL     | https://arxiv.org/abs/2503.09516 |
| Arxiv      | 21 May 2025  | An Empirical Study on RL for Reasoning-Search Interleaved LLM Agents      | https://arxiv.org/abs/2505.15117 |
| Arxiv      | 4 Jun 2025   | R-Search: Empowering LLM Reasoning with Search via Multi-Reward RL         | https://arxiv.org/abs/2506.04185 |
| Arxiv      | 7 May 2025   | ZeroSearch: Incentivize the Search Capability of LLMs without Searching     | https://arxiv.org/abs/2505.04588 |
| Arxiv      | 22 May 2025  | O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended QA   | https://arxiv.org/abs/2505.16582 |
| Arxiv      | 11 Aug 2025  | HierSearch: A Hierarchical Enterprise Deep Search Framework                | https://arxiv.org/abs/2508.08088 |
| Arxiv      | 29 Jul 2025  | Graph-R1: Towards Agentic GraphRAG Framework via End-to-end RL             | https://arxiv.org/abs/2507.21892 |
| Arxiv      | 23 Jul 2025  | DynaSearcher: Dynamic Knowledge Graph Augmented Search Agent via Multi-Reward RL | https://arxiv.org/abs/2507.17365 |
| Arxiv      | 28 May 2025  | WebDancer: Towards Autonomous Information Seeking Agency                    | https://arxiv.org/abs/2505.22648 |
| Arxiv      | 16 Sep 2025  | WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data & RL | https://arxiv.org/abs/2509.13305 |
| Arxiv      | 28 Jul 2025  | Kimi k2: Open Agentic Intelligence                                          | https://arxiv.org/abs/2507.20534 |
| Arxiv      | 11 Aug 2025  | Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Async RL | https://arxiv.org/abs/2508.07976 |
| Arxiv      | 30 May 2025  | Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web RL         | https://arxiv.org/html/2505.24332v1 |
| Arxiv      | 6 Aug 2025   | Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation & RL | https://arxiv.org/abs/2508.13167 |
| Arxiv      | 22 May 2025  | Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via RL               | https://arxiv.org/abs/2505.16410 |
| Arxiv      | 26 Jul 2025  | Agentic Reinforced Policy Optimization                                     | https://arxiv.org/abs/2507.19849 |
| Arxiv      | 16 Oct 2025  | Agentic Entropy-Balanced Policy Optimization                               | https://arxiv.org/abs/2510.14545 |

## Datasets & Benchmarks


# ❤️ Acknowledgement

This project benefits from [deepresearch](https://github.com/scienceaix/deepresearch), [Tongyi-DeepResearch](https://github.com/Alibaba-NLP/DeepResearch), [Search Agent](https://github.com/YunjiaXi/Awesome-Search-Agent-Papers), and [Knowledge-Boundary](https://github.com/ShiyuNee/Awesome-LMs-Perception-of-Their-Knowledge-Boundaries-Papers) Thanks for their wonderful works and collective efforts.


# 📞 Contact

Feel free to contact us if there are any problems: zhengliang.shii@gmail.com; shizhl@mail.sdu.edu.cn


# 🥳 Citation
If you find this work useful, please cite:
```txt
@misc{shi2025deepresearch,
  title = {Deep Research: A Systematic Survey},
  author = {Shi, Zhengliang and Chen, Yiqun and Li, Haitao and Sun, Weiwei and Ni, Shiyu and Lyu, Yougang and Fan, Run-Ze and Jin, Bowen and Weng, Yixuan and Zhu, Minjun and Xie, Qiujie and Guo, Xinyu and Yang, Qu and Wu, Jiayi and Zhao, Jujia and Tang, Xiaqiang and Ma, Xinbei and Wang, Cunxiang and Mao, Jiaxin and Ai, Qingyao and Huang, Jen-Tse and Wang, Wenxuan and Zhang, Yue and Yang, Yiming and Tu, Zhaopeng and Ren, Zhaochun},
  year = {2025},
  howpublished = {\url{https://github.com/mangopy/Deep-Research-Survey}},
  note = {Accessed: 2025-11-22}
}
```

# A Comparative Analysis of Guardrail Frameworks for Large Language Models and Enhancement with Ensemble Techniques

## Abstract

In the swiftly evolving field of artificial intelligence, large language models (LLMs) have become powerful tools for crafting human-like text. However, their integration into real-world settings raises ethical, safety, regulatory, and legal concerns due to the potential for generating inappropriate, misleading, or biased content. To address these issues, guardrails designed for LLMs regulate information flow within these systems to prevent or mitigate undesirable outcomes. Our study compares two primary guardrail frameworks: Llama Guard by Meta and NeMo Guardrails by NVIDIA, representing LLM-based and vector similarity search methodologies, respectively. Through empirical evaluation, we assess the efficacy of these models in practical contexts, highlighting the importance of robust content moderation. Furthermore, we propose a novel integration of these frameworks using ensemble techniques that markedly enhances performance. The resulting ensemble models harness the strengths of Llama Guard and NeMo, reducing both false positives and false negatives, and ensuring accurate identification of unsafe prompts. Incorporating prompt embeddings, we further improve performance, emphasizing the role of contextual information in prompt classification. On the test dataset, Llama Guard achieves 89.0% accuracy while NeMo Guardrails reaches 97.0%. Using ensemble methods such as Random Forest and K-Nearest Neighbors with prompt embeddings, performance increases to 99.4%. This study advances responsible AI usage by enhancing user interaction safeguards with LLMs, focusing on deployment, model effectiveness, and ensemble techniques for guardrail enforcement.

## Documentation

The repository contains all documentation, including our datasets and code.

- [/data](/data): our custom training and test data
- [/llamaguard](/llamaguard): training and fine tuning Llama Guard model using our data and use case
- [/nemo](/nemo): our experimentation with NeMo Guard using our data and use case
- [/rephrase_eval](/rephrase_eval): study how well NeMo can capture similar semantic representations of a single config prompt
- [/ensemble](/ensemble): applying ensemble methods to integrate models in attempt to increase overall accuracy

Our PEFT Llama Guard model, which we trained using our custom dataset, can be found at [https://huggingface.co/neurips/llama-guard-finetuned-1ep-1000](https://huggingface.co/neurips/llama-guard-finetuned-1ep-1000).

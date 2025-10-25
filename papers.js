// This file just defines the data.
// It must be loaded *before* app.js in the HTML.
const papers = [
    // PHASE 1: Mathematical & ML Foundations
    {
        phase: "Phase 1: Mathematical & ML Foundations",
        phaseDescription: "Build the mathematical intuition before diving into deep learning",
        number: 1,
        title: "A Few Useful Things to Know About Machine Learning",
        description: "Essential principles and common pitfalls every ML practitioner should understand",
        tags: ["ML Fundamentals"],
        difficulty: "beginner",
        mandatory: true,
        link: "https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf"
    },

    // PHASE 2: Deep Learning Foundations
    {
        phase: "Phase 2: Deep Learning Foundations",
        phaseDescription: "Master neural networks, CNNs, and the building blocks of modern AI",
        number: 2,
        title: "Deep Learning (Nature Paper)",
        description: "The Nature paper that summarizes deep learning's breakthrough - foundational reading",
        tags: ["ML Fundamentals", "Deep Learning"],
        difficulty: "beginner",
        mandatory: true,
        link: "https://www.nature.com/articles/nature14539"
    },
    {
        phase: "Phase 2: Deep Learning Foundations",
        phaseDescription: "Master neural networks, CNNs, and the building blocks of modern AI",
        number: 3,
        title: "ImageNet Classification with Deep CNNs (AlexNet)",
        description: "The paper that started the deep learning revolution in computer vision",
        tags: ["Computer Vision", "Deep Learning"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf"
    },
    {
        phase: "Phase 2: Deep Learning Foundations",
        phaseDescription: "Master neural networks, CNNs, and the building blocks of modern AI",
        number: 4,
        title: "Very Deep Convolutional Networks (VGG)",
        description: "Simple but powerful architecture that showed depth matters in CNNs",
        tags: ["Computer Vision", "Deep Learning"],
        difficulty: "intermediate",
        mandatory: false,
        link: "https://arxiv.org/abs/1409.1556"
    },
    {
        phase: "Phase 2: Deep Learning Foundations",
        phaseDescription: "Master neural networks, CNNs, and the building blocks of modern AI",
        number: 5,
        title: "Deep Residual Learning (ResNet)",
        description: "Revolutionary skip connections that enabled training of 100+ layer networks",
        tags: ["Computer Vision", "Deep Learning"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://arxiv.org/abs/1512.03385"
    },
    {
        phase: "Phase 2: Deep Learning Foundations",
        phaseDescription: "Master neural networks, CNNs, and the building blocks of modern AI",
        number: 6,
        title: "Dropout: Preventing Neural Networks from Overfitting",
        description: "Simple yet effective regularization technique used in almost every modern architecture",
        tags: ["ML Fundamentals", "Optimization"],
        difficulty: "beginner",
        mandatory: true,
        link: "https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf"
    },
    {
        phase: "Phase 2: Deep Learning Foundations",
        phaseDescription: "Master neural networks, CNNs, and the building blocks of modern AI",
        number: 7,
        title: "Batch Normalization",
        description: "Stabilizes training and enables higher learning rates - essential for deep networks",
        tags: ["ML Fundamentals", "Optimization"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://arxiv.org/abs/1502.03167"
    },
    {
        phase: "Phase 2: Deep Learning Foundations",
        phaseDescription: "Master neural networks, CNNs, and the building blocks of modern AI",
        number: 8,
        title: "Adam: A Method for Stochastic Optimization",
        description: "Most popular optimizer in deep learning - adaptive learning rates made simple",
        tags: ["Optimization", "ML Fundamentals"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://arxiv.org/abs/1412.6980"
    },

    // PHASE 3: Sequence Models & Pre-Attention Era
    {
        phase: "Phase 3: Sequence Models & Early NLP",
        phaseDescription: "Understand RNNs and the evolution toward attention mechanisms",
        number: 9,
        title: "Long Short-Term Memory (LSTM)",
        description: "Foundational recurrent architecture that solved vanishing gradients in sequences",
        tags: ["NLP", "Deep Learning"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory"
    },
    {
        phase: "Phase 3: Sequence Models & Early NLP",
        phaseDescription: "Understand RNNs and the evolution toward attention mechanisms",
        number: 10,
        title: "Learning Phrase Representations (GRU)",
        description: "Simplified alternative to LSTM with fewer parameters but similar performance",
        tags: ["NLP", "Deep Learning"],
        difficulty: "intermediate",
        mandatory: false,
        link: "https://arxiv.org/abs/1406.1078"
    },
    {
        phase: "Phase 3: Sequence Models & Early NLP",
        phaseDescription: "Understand RNNs and the evolution toward attention mechanisms",
        number: 11,
        title: "Efficient Estimation of Word Representations (Word2Vec)",
        description: "Breakthrough in word embeddings - words with similar meanings have similar vectors",
        tags: ["NLP", "ML Fundamentals"],
        difficulty: "beginner",
        mandatory: true,
        link: "https://arxiv.org/abs/1301.3781"
    },
    {
        phase: "Phase 3: Sequence Models & Early NLP",
        phaseDescription: "Understand RNNs and the evolution toward attention mechanisms",
        number: 12,
        title: "GloVe: Global Vectors for Word Representation",
        description: "Alternative to Word2Vec using global word co-occurrence statistics",
        tags: ["NLP", "ML Fundamentals"],
        difficulty: "intermediate",
        mandatory: false,
        link: "https://nlp.stanford.edu/pubs/glove.pdf"
    },
    {
        phase: "Phase 3: Sequence Models & Early NLP",
        phaseDescription: "Understand RNNs and the evolution toward attention mechanisms",
        number: 13,
        title: "Neural Machine Translation by Attention",
        description: "The first attention mechanism - bridge between RNNs and Transformers",
        tags: ["NLP", "Transformers"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://arxiv.org/abs/1409.0473"
    },

    // PHASE 4: THE TRANSFORMER REVOLUTION
    {
        phase: "Phase 4: The Transformer Revolution 🔥",
        phaseDescription: "THE MOST CRITICAL PHASE - Foundation of all modern AI",
        number: 14,
        title: "⭐⭐⭐ Attention Is All You Need",
        description: "THE breakthrough paper - created the architecture behind GPT, BERT, and all modern LLMs",
        tags: ["Transformers", "NLP", "Deep Learning"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/1706.03762"
    },
    {
        phase: "Phase 4: The Transformer Revolution 🔥",
        phaseDescription: "THE MOST CRITICAL PHASE - Foundation of all modern AI",
        number: 15,
        title: "BERT: Bidirectional Transformer Pre-training",
        description: "Revolutionized NLP with bidirectional context understanding and transfer learning",
        tags: ["Transformers", "NLP"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/1810.04805"
    },
    {
        phase: "Phase 4: The Transformer Revolution 🔥",
        phaseDescription: "THE MOST CRITICAL PHASE - Foundation of all modern AI",
        number: 16,
        title: "Improving Language Understanding (GPT-1)",
        description: "OpenAI's first GPT - decoder-only transformer for language generation",
        tags: ["Transformers", "NLP", "GenAI"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf"
    },
    {
        phase: "Phase 4: The Transformer Revolution 🔥",
        phaseDescription: "THE MOST CRITICAL PHASE - Foundation of all modern AI",
        number: 17,
        title: "Language Models are Unsupervised Multitask Learners (GPT-2)",
        description: "Demonstrated zero-shot learning emergence through scale",
        tags: ["Transformers", "NLP", "GenAI"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
    },
    {
        phase: "Phase 4: The Transformer Revolution 🔥",
        phaseDescription: "THE MOST CRITICAL PHASE - Foundation of all modern AI",
        number: 18,
        title: "Language Models are Few-Shot Learners (GPT-3)",
        description: "175B parameters - showed emergent abilities and in-context learning at scale",
        tags: ["Transformers", "NLP", "GenAI"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2005.14165"
    },
    {
        phase: "Phase 4: The Transformer Revolution 🔥",
        phaseDescription: "THE MOST CRITICAL PHASE - Foundation of all modern AI",
        number: 19,
        title: "ELMo: Deep Contextualized Word Representations",
        description: "Pre-BERT contextualized embeddings using bidirectional LSTMs",
        tags: ["NLP", "Deep Learning"],
        difficulty: "intermediate",
        mandatory: false,
        link: "https://arxiv.org/abs/1802.05365"
    },
    {
        phase: "Phase 4: The Transformer Revolution 🔥",
        phaseDescription: "THE MOST CRITICAL PHASE - Foundation of all modern AI",
        number: 20,
        title: "XLNet: Generalized Autoregressive Pretraining",
        description: "Permutation language modeling - combines benefits of BERT and GPT",
        tags: ["Transformers", "NLP"],
        difficulty: "advanced",
        mandatory: false,
        link: "https://arxiv.org/abs/1906.08237"
    },
    {
        phase: "Phase 4: The Transformer Revolution 🔥",
        phaseDescription: "THE MOST CRITICAL PHASE - Foundation of all modern AI",
        number: 21,
        title: "RoBERTa: A Robustly Optimized BERT Approach",
        description: "Improved BERT training - shows training methodology matters as much as architecture",
        tags: ["Transformers", "NLP"],
        difficulty: "intermediate",
        mandatory: false,
        link: "https://arxiv.org/abs/1907.11692"
    },
    {
        phase: "Phase 4: The Transformer Revolution 🔥",
        phaseDescription: "THE MOST CRITICAL PHASE - Foundation of all modern AI",
        number: 22,
        title: "Exploring the Limits of Transfer Learning (T5)",
        description: "Unified text-to-text framework - every NLP task becomes sequence-to-sequence",
        tags: ["Transformers", "NLP"],
        difficulty: "advanced",
        mandatory: false,
        link: "https://arxiv.org/abs/1910.10683"
    },

    // PHASE 5: Vision Transformers & Multimodal
    {
        phase: "Phase 5: Vision Transformers & Multimodal AI",
        phaseDescription: "Transformers beyond NLP - vision, speech, and multimodal models",
        number: 23,
        title: "An Image is Worth 16x16 Words (ViT)",
        description: "Applied pure transformers to vision - no convolutions needed",
        tags: ["Computer Vision", "Transformers"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2010.11929"
    },
    {
        phase: "Phase 5: Vision Transformers & Multimodal AI",
        phaseDescription: "Transformers beyond NLP - vision, speech, and multimodal models",
        number: 24,
        title: "Learning Transferable Visual Models (CLIP)",
        description: "Contrastive learning of vision-language models - revolutionized image-text understanding",
        tags: ["Computer Vision", "NLP", "Transformers"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2103.00020"
    },

    // PHASE 6: Generative Models
    {
        phase: "Phase 6: Generative Models",
        phaseDescription: "GANs, VAEs, and Diffusion - creating new content from learned distributions",
        number: 25,
        title: "Generative Adversarial Networks (GANs)",
        description: "Revolutionary adversarial training - generator vs discriminator game theory",
        tags: ["GenAI", "Deep Learning"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/1406.2661"
    },
    {
        phase: "Phase 6: Generative Models",
        phaseDescription: "GANs, VAEs, and Diffusion - creating new content from learned distributions",
        number: 26,
        title: "Auto-Encoding Variational Bayes (VAE)",
        description: "Probabilistic approach to generative modeling using variational inference",
        tags: ["GenAI", "Deep Learning"],
        difficulty: "advanced",
        mandatory: false,
        link: "https://arxiv.org/abs/1312.6114"
    },
    {
        phase: "Phase 6: Generative Models",
        phaseDescription: "GANs, VAEs, and Diffusion - creating new content from learned distributions",
        number: 27,
        title: "Unsupervised Representation Learning (DCGAN)",
        description: "Convolutional architecture for GANs - stable training and high-quality images",
        tags: ["GenAI", "Computer Vision"],
        difficulty: "intermediate",
        mandatory: false,
        link: "https://arxiv.org/abs/1511.06434"
    },
    {
        phase: "Phase 6: Generative Models",
        phaseDescription: "GANs, VAEs, and Diffusion - creating new content from learned distributions",
        number: 28,
        title: "Denoising Diffusion Probabilistic Models",
        description: "Foundation of modern image generation - iterative denoising process",
        tags: ["GenAI", "Deep Learning"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2006.11239"
    },
    {
        phase: "Phase 6: Generative Models",
        phaseDescription: "GANs, VAEs, and Diffusion - creating new content from learned distributions",
        number: 29,
        title: "High-Resolution Image Synthesis (Stable Diffusion)",
        description: "Latent diffusion models - efficient high-quality text-to-image generation",
        tags: ["GenAI", "Computer Vision"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2112.10752"
    },
    {
        phase: "Phase 6: Generative Models",
        phaseDescription: "GANs, VAEs, and Diffusion - creating new content from learned distributions",
        number: 30,
        title: "Segment Anything (SAM)",
        description: "Universal image segmentation model - foundation model for vision",
        tags: ["Computer Vision", "Deep Learning"],
        difficulty: "intermediate",
        mandatory: false,
        link: "https://arxiv.org/abs/2304.02643"
    },

    // PHASE 7: LLM Fine-Tuning & Alignment
    {
        phase: "Phase 7: LLM Fine-Tuning & Alignment",
        phaseDescription: "Making LLMs helpful, harmless, and efficient",
        number: 31,
        title: "Training Language Models to Follow Instructions (InstructGPT)",
        description: "RLHF technique that made ChatGPT possible - aligning LLMs with human preferences",
        tags: ["GenAI", "NLP", "Transformers"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2203.02155"
    },
    {
        phase: "Phase 7: LLM Fine-Tuning & Alignment",
        phaseDescription: "Making LLMs helpful, harmless, and efficient",
        number: 32,
        title: "Constitutional AI: Harmlessness from AI Feedback",
        description: "AI safety through self-critique - training models to be helpful and harmless",
        tags: ["GenAI", "NLP"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2212.08073"
    },
    {
        phase: "Phase 7: LLM Fine-Tuning & Alignment",
        phaseDescription: "Making LLMs helpful, harmless, and efficient",
        number: 33,
        title: "LoRA: Low-Rank Adaptation of LLMs",
        description: "Parameter-efficient fine-tuning - train massive models on consumer hardware",
        tags: ["GenAI", "NLP", "Optimization"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://arxiv.org/abs/2106.09685"
    },
    {
        phase: "Phase 7: LLM Fine-Tuning & Alignment",
        phaseDescription: "Making LLMs helpful, harmless, and efficient",
        number: 34,
        title: "QLoRA: Efficient Finetuning of Quantized LLMs",
        description: "Quantization + LoRA - fine-tune 65B models on a single GPU",
        tags: ["GenAI", "NLP", "Optimization"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2305.14314"
    },
    {
        phase: "Phase 7: LLM Fine-Tuning & Alignment",
        phaseDescription: "Making LLMs helpful, harmless, and efficient",
        number: 35,
        title: "Prefix-Tuning: Optimizing Continuous Prompts",
        description: "Alternative PEFT method - learn task-specific prefixes instead of full fine-tuning",
        tags: ["GenAI", "NLP", "Optimization"],
        difficulty: "intermediate",
        mandatory: false,
        link: "https://arxiv.org/abs/2101.00190"
    },

    // PHASE 8: Prompt Engineering & Reasoning
    {
        phase: "Phase 8: Prompt Engineering & Reasoning",
        phaseDescription: "Unlocking reasoning capabilities through better prompting",
        number: 36,
        title: "Chain-of-Thought Prompting Elicits Reasoning",
        description: "Let's think step by step - dramatically improves complex reasoning",
        tags: ["GenAI", "NLP"],
        difficulty: "beginner",
        mandatory: true,
        link: "https://arxiv.org/abs/2201.11903"
    },
    {
        phase: "Phase 8: Prompt Engineering & Reasoning",
        phaseDescription: "Unlocking reasoning capabilities through better prompting",
        number: 37,
        title: "Tree of Thoughts: Deliberate Problem Solving",
        description: "Explore multiple reasoning paths - search through thought trees",
        tags: ["GenAI", "NLP"],
        difficulty: "intermediate",
        mandatory: false,
        link: "https://arxiv.org/abs/2305.10601"
    },
    {
        phase: "Phase 8: Prompt Engineering & Reasoning",
        phaseDescription: "Unlocking reasoning capabilities through better prompting",
        number: 38,
        title: "ReAct: Synergizing Reasoning and Acting",
        description: "Combine reasoning with actions - foundation of modern AI agents",
        tags: ["GenAI", "NLP"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://arxiv.org/abs/2210.03629"
    },
    {
        phase: "Phase 8: Prompt Engineering & Reasoning",
        phaseDescription: "Unlocking reasoning capabilities through better prompting",
        number: 39,
        title: "Large Language Models are Zero-Shot Reasoners",
        description: "Simple zero-shot CoT - just add 'Let's think step by step' to your prompts",
        tags: ["GenAI", "NLP"],
        difficulty: "beginner",
        mandatory: false,
        link: "https://arxiv.org/abs/2205.11916"
    },

    // PHASE 9: RAG & Retrieval
    {
        phase: "Phase 9: RAG & Knowledge Augmentation",
        phaseDescription: "Combining LLMs with external knowledge sources",
        number: 40,
        title: "Retrieval-Augmented Generation for Knowledge-Intensive Tasks",
        description: "Foundation of RAG systems - retrieve relevant docs then generate answers",
        tags: ["GenAI", "NLP"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://arxiv.org/abs/2005.11401"
    },

    // PHASE 10: Open Source LLMs
    {
        phase: "Phase 10: Open Source LLMs",
        phaseDescription: "Democratizing access to powerful language models",
        number: 41,
        title: "LLaMA: Open and Efficient Foundation Models",
        description: "Meta's open foundation models - efficient training at scale",
        tags: ["GenAI", "NLP", "Transformers"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2302.13971"
    },
    {
        phase: "Phase 10: Open Source LLMs",
        phaseDescription: "Democratizing access to powerful language models",
        number: 42,
        title: "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        description: "Instruction-tuned Llama with safety improvements - commercially available",
        tags: ["GenAI", "NLP", "Transformers"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2307.09288"
    },
    {
        phase: "Phase 10: Open Source LLMs",
        phaseDescription: "Democratizing access to powerful language models",
        number: 43,
        title: "The Llama 3 Herd of Models",
        description: "Latest Llama generation - improved performance and multilingual capabilities",
        tags: ["GenAI", "NLP", "Transformers"],
        difficulty: "advanced",
        mandatory: false,
        link: "https://arxiv.org/abs/2407.21783"
    },
    {
        phase: "Phase 10: Open Source LLMs",
        phaseDescription: "Democratizing access to powerful language models",
        number: 44,
        title: "Mixtral of Experts",
        description: "Sparse mixture of experts - efficient scaling through conditional computation",
        tags: ["GenAI", "NLP", "Transformers"],
        difficulty: "advanced",
        mandatory: false,
        link: "https://arxiv.org/abs/2401.04088"
    },
    {
        phase: "Phase 10: Open Source LLMs",
        phaseDescription: "Democratizing access to powerful language models",
        number: 45,
        title: "DeepSeek-V3 Technical Report",
        description: "Cost-efficient MoE training - achieving competitive performance at lower cost",
        tags: ["GenAI", "NLP", "Transformers"],
        difficulty: "advanced",
        mandatory: false,
        link: "https://arxiv.org/abs/2412.19437"
    },

    // PHASE 11: Attention Optimization
    {
        phase: "Phase 11: Efficient Attention Mechanisms",
        phaseDescription: "Making transformers faster and handling longer contexts",
        number: 46,
        title: "FlashAttention: Fast and Memory-Efficient Attention",
        description: "IO-aware attention algorithm - 2-4x faster with less memory",
        tags: ["Optimization", "Transformers"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/2205.14135"
    },
    {
        phase: "Phase 11: Efficient Attention Mechanisms",
        phaseDescription: "Making transformers faster and handling longer contexts",
        number: 47,
        title: "FlashAttention-2: Faster Attention with Better Parallelism",
        description: "Improved FlashAttention - better GPU utilization and speed",
        tags: ["Optimization", "Transformers"],
        difficulty: "advanced",
        mandatory: false,
        link: "https://arxiv.org/abs/2307.08691"
    },
    {
        phase: "Phase 11: Efficient Attention Mechanisms",
        phaseDescription: "Making transformers faster and handling longer contexts",
        number: 48,
        title: "Ring Attention with Blockwise Transformers",
        description: "Distributed attention for million-token contexts - breakthrough for long sequences",
        tags: ["Optimization", "Transformers"],
        difficulty: "advanced",
        mandatory: false,
        link: "https://arxiv.org/abs/2310.01889"
    },

    // PHASE 12: Reinforcement Learning
    {
        phase: "Phase 12: Reinforcement Learning Foundations",
        phaseDescription: "RL basics and its application to LLM training",
        number: 49,
        title: "Playing Atari with Deep Reinforcement Learning (DQN)",
        description: "Deep Q-Networks - combining deep learning with RL for game playing",
        tags: ["Reinforcement Learning", "Deep Learning"],
        difficulty: "advanced",
        mandatory: false,
        link: "https://arxiv.org/abs/1312.5602"
    },
    {
        phase: "Phase 12: Reinforcement Learning Foundations",
        phaseDescription: "RL basics and its application to LLM training",
        number: 50,
        title: "Proximal Policy Optimization (PPO)",
        description: "Most popular policy gradient method - used in RLHF for ChatGPT",
        tags: ["Reinforcement Learning", "Optimization"],
        difficulty: "advanced",
        mandatory: true,
        link: "https://arxiv.org/abs/1707.06347"
    },

    // PHASE 13: AI Safety & Ethics
    {
        phase: "Phase 13: AI Safety & Ethics",
        phaseDescription: "Understanding risks and responsible AI development",
        number: 51,
        title: "On the Dangers of Stochastic Parrots",
        description: "Critical analysis of LLM risks - environmental, social, and ethical concerns",
        tags: ["GenAI", "NLP"],
        difficulty: "beginner",
        mandatory: true,
        link: "https://dl.acm.org/doi/10.1145/3442188.3445922"
    },
    {
        phase: "Phase 13: AI Safety & Ethics",
        phaseDescription: "Understanding risks and responsible AI development",
        number: 52,
        title: "Ethical and Social Risks of Harm from Language Models",
        description: "Comprehensive taxonomy of LLM risks and mitigation strategies",
        tags: ["GenAI", "NLP"],
        difficulty: "intermediate",
        mandatory: true,
        link: "https://arxiv.org/abs/2112.04359"
    }
];
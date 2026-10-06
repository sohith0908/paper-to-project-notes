"""Hand-labeled gold set (25 papers across task types — includes niches, not only famous ones)."""

from __future__ import annotations

import json
from pathlib import Path

# Each item: id, title, text, task_type, datasets, metrics, components, paper_type
GOLD_PAPERS: list[dict] = [
    {
        "id": "g01_leaf_disease_cnn",
        "title": "Lightweight CNN for Tomato Leaf Disease Classification",
        "paper_type": "method",
        "task_type": "image_classification",
        "datasets": ["PlantVillage"],
        "metrics": ["accuracy", "F1"],
        "components": ["CNN", "data augmentation", "dropout"],
        "text": """Lightweight CNN for Tomato Leaf Disease Classification
Abstract
We propose a lightweight CNN for image classification of tomato leaf diseases. On a PlantVillage subset we report accuracy 0.94 and macro-F1 0.91 using data augmentation and dropout. We do not address object detection.
""",
    },
    {
        "id": "g02_unet_lung",
        "title": "U-Net for Lung CT Semantic Segmentation",
        "paper_type": "method",
        "task_type": "semantic_segmentation",
        "datasets": ["Cityscapes"],
        "metrics": ["mIoU"],
        "components": ["U-Net", "encoder-decoder", "skip connection"],
        "text": """U-Net for Lung CT Semantic Segmentation
Abstract
We apply a U-Net encoder-decoder with skip connections for semantic segmentation of lung CT. We report mIoU 0.78 on a Cityscapes-style medical split. Pixel-wise labels are required.
""",
    },
    {
        "id": "g03_toy_dcgan",
        "title": "A Tiny DCGAN for Fashion Thumbnail Synthesis",
        "paper_type": "method",
        "task_type": "generative_models",
        "datasets": ["Fashion-MNIST"],
        "metrics": ["FID"],
        "components": ["GAN", "generator", "discriminator"],
        "text": """A Tiny DCGAN for Fashion Thumbnail Synthesis
Abstract
We train a generative adversarial network (GAN) with a generator and discriminator on Fashion-MNIST thumbnails. We report FID 42.1. This is not a classification study.
""",
    },
    {
        "id": "g04_diffusion_faces",
        "title": "Denoising Diffusion for Low-Res Face Sketches",
        "paper_type": "method",
        "task_type": "generative_models",
        "datasets": ["CelebA"],
        "metrics": ["FID", "Inception Score"],
        "components": ["diffusion", "U-Net denoiser"],
        "text": """Denoising Diffusion for Low-Res Face Sketches
Abstract
We present a denoising diffusion model with a U-Net denoiser for sketch faces on CelebA. FID and Inception Score are reported. Training diffusion models from scratch is compute heavy.
""",
    },
    {
        "id": "g05_ppo_cartpole",
        "title": "PPO Stability Tricks on Classic Control",
        "paper_type": "method",
        "task_type": "reinforcement_learning",
        "datasets": [],
        "metrics": ["episodic reward"],
        "components": ["PPO", "policy network", "advantage estimation"],
        "text": """PPO Stability Tricks on Classic Control
Abstract
We study proximal policy optimization (PPO) reinforcement learning on OpenAI Gym CartPole. We plot episodic reward and discuss the reward function and actor-critic style updates.
""",
    },
    {
        "id": "g06_gcn_cora",
        "title": "Shallow GCN for Citation Node Classification",
        "paper_type": "method",
        "task_type": "graph_ml",
        "datasets": ["Cora"],
        "metrics": ["accuracy"],
        "components": ["GCN", "message passing", "graph convolution"],
        "text": """Shallow GCN for Citation Node Classification
Abstract
We train a graph convolutional network (GCN) with message passing for node classification on Cora. Accuracy reaches 0.81. Link prediction is out of scope.
""",
    },
    {
        "id": "g07_whisper_tiny",
        "title": "Domain Adaptation Notes for Tiny ASR",
        "paper_type": "method",
        "task_type": "speech",
        "datasets": ["LibriSpeech"],
        "metrics": ["WER"],
        "components": ["ASR", "mel-spectrogram", "encoder-decoder"],
        "text": """Domain Adaptation Notes for Tiny ASR
Abstract
We fine-tune an automatic speech recognition (ASR) encoder-decoder on LibriSpeech clean-100 using mel-spectrograms. Word error rate (WER) improves from 12.4 to 9.8.
""",
    },
    {
        "id": "g08_sentiment_tfidf",
        "title": "TF-IDF Linear SVM for Course Review Sentiment",
        "paper_type": "method",
        "task_type": "nlp_text",
        "datasets": ["IMDB"],
        "metrics": ["F1", "accuracy"],
        "components": ["TF-IDF", "SVM"],
        "text": """TF-IDF Linear SVM for Course Review Sentiment
Abstract
We perform text classification / sentiment analysis with TF-IDF features and a linear SVM on an IMDB-like review set. Accuracy 0.88 and F1 0.87.
""",
    },
    {
        "id": "g09_bert_multitask",
        "title": "Multi-task BERT Fine-Tuning Notes",
        "paper_type": "method",
        "task_type": "nlp_text",
        "datasets": ["GLUE", "SQuAD"],
        "metrics": ["F1", "accuracy"],
        "components": ["BERT", "transformer", "attention", "masked language modeling"],
        "text": """Multi-task BERT Fine-Tuning Notes
Abstract
We fine-tune BERT transformers with attention for text classification, named entity recognition, and question answering on GLUE and SQuAD after masked language modeling pretraining. This NLP study is multi-task.
""",
    },
    {
        "id": "g10_xgboost_churn",
        "title": "XGBoost for Telecom Churn Tabular Prediction",
        "paper_type": "method",
        "task_type": "classic_tabular_ml",
        "datasets": ["Adult"],
        "metrics": ["AUC", "logloss"],
        "components": ["XGBoost", "gradient boosting", "feature importance"],
        "text": """XGBoost for Telecom Churn Tabular Prediction
Abstract
We apply XGBoost gradient boosting on a tabular churn dataset similar to Adult. We report AUC 0.84 and logloss, with feature importance analysis. No YOLO or vision modules.
""",
    },
    {
        "id": "g11_arima_energy",
        "title": "ARIMA Forecasting for Campus Energy Load",
        "paper_type": "method",
        "task_type": "time_series",
        "datasets": [],
        "metrics": ["MAE", "RMSE"],
        "components": ["ARIMA", "lag features"],
        "text": """ARIMA Forecasting for Campus Energy Load
Abstract
We forecast campus energy load as a time-series problem using ARIMA and lag features. MAE and RMSE are reported on a rolling origin split.
""",
    },
    {
        "id": "g12_cf_movies",
        "title": "Item-Item Collaborative Filtering for Campus Movies",
        "paper_type": "method",
        "task_type": "recommender",
        "datasets": [],
        "metrics": ["RMSE", "Precision@K"],
        "components": ["collaborative filtering", "matrix factorization"],
        "text": """Item-Item Collaborative Filtering for Campus Movies
Abstract
We build a recommender system with collaborative filtering and matrix factorization on student movie ratings. RMSE and Precision@K are reported.
""",
    },
    {
        "id": "g13_yolo_campus",
        "title": "YOLOv8n for Bike Detection on Campus Paths",
        "paper_type": "method",
        "task_type": "object_detection_tracking",
        "datasets": ["COCO"],
        "metrics": ["mAP50"],
        "components": ["YOLOv8n", "NMS"],
        "text": """YOLOv8n for Bike Detection on Campus Paths
Abstract
We fine-tune YOLOv8n for object detection of bicycles with bounding boxes. mAP50 is 0.61 on a COCO-format campus set. Tracking is future work.
""",
    },
    {
        "id": "g14_bytetrack_lobby",
        "title": "ByteTrack for Lobby Pedestrian Flow",
        "paper_type": "method",
        "task_type": "object_detection_tracking",
        "datasets": ["MOT17"],
        "metrics": ["MOTA", "IDF1"],
        "components": ["ByteTrack", "Re-ID"],
        "text": """ByteTrack for Lobby Pedestrian Flow
Abstract
We combine a detector with ByteTrack multiple object tracking and light Re-ID. On MOT17-style clips MOTA is 0.55 and IDF1 0.48.
""",
    },
    {
        "id": "g15_resnet_survey",
        "title": "Deep Residual Networks: A Survey",
        "paper_type": "survey",
        "task_type": "image_classification",
        "datasets": ["ImageNet", "CIFAR-10"],
        "metrics": [],
        "components": ["ResNet", "residual block", "skip connection", "bottleneck", "batch normalization"],
        "text": """Deep Residual Networks: A Survey
Abstract
Why residual networks work is not fully understood yet. This survey reviews ResNet variants, residual blocks, skip connections, bottleneck designs, and batch normalization. Literature often uses ImageNet and CIFAR-10. No new method is proposed.
""",
    },
    {
        "id": "g16_he_resnet",
        "title": "Deep Residual Learning for Image Recognition",
        "paper_type": "method",
        "task_type": "image_classification",
        "datasets": ["ImageNet", "CIFAR-10"],
        "metrics": ["top-5 error", "3.57%"],
        "components": ["residual learning", "identity shortcut", "bottleneck", "ResNet"],
        "text": """Deep Residual Learning for Image Recognition
Abstract
Deeper neural networks are more difficult to train. We present a residual learning framework with identity skip connections. On ImageNet an ensemble achieves 3.57% top-5 error. CIFAR-10 uses 45k training and 5k validation. The degradation problem is addressed. plain-18 27.94% plain-34 28.54%.
""",
    },
    {
        "id": "g17_maskrcnn_lite",
        "title": "Lite Mask R-CNN Notes for Shelf Instance Segmentation",
        "paper_type": "method",
        "task_type": "semantic_segmentation",
        "datasets": ["COCO"],
        "metrics": ["mIoU"],
        "components": ["Mask R-CNN", "instance segmentation"],
        "text": """Lite Mask R-CNN Notes for Shelf Instance Segmentation
Abstract
We study Mask R-CNN for instance segmentation of retail shelves on a COCO-format set and report mIoU. Semantic segmentation baselines are compared.
""",
    },
    {
        "id": "g18_dqn_grid",
        "title": "DQN on a Tiny Gridworld",
        "paper_type": "method",
        "task_type": "reinforcement_learning",
        "datasets": [],
        "metrics": ["success rate"],
        "components": ["DQN", "replay buffer", "Q-learning"],
        "text": """DQN on a Tiny Gridworld
Abstract
We apply deep Q-learning (DQN) reinforcement learning with a replay buffer on a tiny gridworld. Success rate and episodic reward are tracked.
""",
    },
    {
        "id": "g19_graphsage",
        "title": "GraphSAGE-style Embeddings for Club Networks",
        "paper_type": "method",
        "task_type": "graph_ml",
        "datasets": ["CiteSeer"],
        "metrics": ["F1"],
        "components": ["GNN", "node embedding", "message passing"],
        "text": """GraphSAGE-style Embeddings for Club Networks
Abstract
We learn node embeddings with a graph neural network (GNN) message passing model on CiteSeer for node classification. Macro-F1 is 0.74.
""",
    },
    {
        "id": "g20_tts_toy",
        "title": "A Toy Text-to-Speech Pipeline for Bell Schedules",
        "paper_type": "method",
        "task_type": "speech",
        "datasets": [],
        "metrics": ["MOS"],
        "components": ["TTS", "mel-spectrogram"],
        "text": """A Toy Text-to-Speech Pipeline for Bell Schedules
Abstract
We prototype text-to-speech (TTS) using mel-spectrograms for short bell announcements and collect MOS listening scores. This is speech synthesis, not ASR.
""",
    },
    {
        "id": "g21_ner_biomed",
        "title": "Biomedical Named Entity Recognition with Transformers",
        "paper_type": "method",
        "task_type": "nlp_text",
        "datasets": [],
        "metrics": ["F1"],
        "components": ["transformer", "tokenizer", "NER"],
        "text": """Biomedical Named Entity Recognition with Transformers
Abstract
We fine-tune a transformer tokenizer pipeline for named entity recognition in biomedical abstracts and report entity-level F1. NLP evaluation only.
""",
    },
    {
        "id": "g22_lightgbm_fraud",
        "title": "LightGBM for Transaction Fraud Tabular Scores",
        "paper_type": "method",
        "task_type": "classic_tabular_ml",
        "datasets": ["Higgs"],
        "metrics": ["AUC"],
        "components": ["LightGBM", "gradient boosting"],
        "text": """LightGBM for Transaction Fraud Tabular Scores
Abstract
LightGBM gradient boosting on a Higgs-like tabular fraud set yields AUC 0.91. Feature importance guides thresholds.
""",
    },
    {
        "id": "g23_lstm_weather",
        "title": "LSTM Weather Nowcasting on Campus Sensors",
        "paper_type": "method",
        "task_type": "time_series",
        "datasets": [],
        "metrics": ["RMSE", "MAE"],
        "components": ["LSTM", "lag features"],
        "text": """LSTM Weather Nowcasting on Campus Sensors
Abstract
We cast campus sensor weather as time-series forecasting with an LSTM and lag features. RMSE and MAE beat a persistence baseline.
""",
    },
    {
        "id": "g24_mf_books",
        "title": "Matrix Factorization for Library Book Recommender",
        "paper_type": "method",
        "task_type": "recommender",
        "datasets": [],
        "metrics": ["NDCG", "RMSE"],
        "components": ["matrix factorization", "collaborative filtering"],
        "text": """Matrix Factorization for Library Book Recommender
Abstract
A campus library recommender system uses matrix factorization collaborative filtering. NDCG and RMSE are primary metrics.
""",
    },
    {
        "id": "g25_unknown_optics",
        "title": "Notes on Optical Bench Calibration Drift",
        "paper_type": "method",
        "task_type": "unknown",
        "datasets": [],
        "metrics": [],
        "components": [],
        "text": """Notes on Optical Bench Calibration Drift
Abstract
We document calibration drift of an optical bench over humidity cycles. No machine learning model is proposed; this is instrumentation notes.
""",
    },
    {
        "id": "g26_resnet_coco_multitask",
        "title": "Residual Backbones for Classification and COCO Detection",
        "paper_type": "method",
        "task_type": "image_classification",
        "datasets": ["ImageNet", "COCO"],
        "metrics": ["top-1 error", "mAP"],
        "components": ["ResNet", "residual block", "FPN"],
        "text": """Residual Backbones for Classification and COCO Detection
Abstract
We pretrain ResNet residual blocks on ImageNet image classification (top-1 error) and transfer the backbone with an FPN for COCO object detection mAP. Both protocols are reported.
""",
    },
    {
        "id": "g27_ambiguous_network",
        "title": "On Networks and Performance in Lab Infrastructure",
        "paper_type": "method",
        "task_type": "unknown",
        "datasets": [],
        "metrics": [],
        "components": [],
        "text": """On Networks and Performance in Lab Infrastructure
Abstract
We discuss campus network switches and performance monitoring dashboards. Accuracy here means uptime accounting, not classifiers.
""",
    },
    {
        "id": "g28_speech_vs_nlp_trap",
        "title": "Transcribing Lectures with Whisper-tiny",
        "paper_type": "method",
        "task_type": "speech",
        "datasets": ["LibriSpeech"],
        "metrics": ["WER"],
        "components": ["Whisper", "ASR", "mel-spectrogram"],
        "text": """Transcribing Lectures with Whisper-tiny
Abstract
We use Whisper for automatic speech recognition (ASR) on lecture audio related to LibriSpeech-style evaluation. Word error rate (WER) is the metric. Mel-spectrograms are intermediate features. This is not text classification.
""",
    },
    {
        "id": "g29_bertscore_metric",
        "title": "BERTScore: Evaluating Text Generation with BERT",
        "paper_type": "method",
        "task_type": "nlp_text",
        "contribution": "new_metric",
        "datasets": ["WMT", "PAWS", "COCO Captions"],
        "metrics": ["Pearson", "Kendall", "BLEU"],
        "components": [
            "BERTScore",
            "cosine similarity",
            "greedy matching",
            "IDF weighting",
            "BERT",
        ],
        "text": """BERTScore: Evaluating Text Generation with BERT
Abstract
We propose BERTScore, an automatic evaluation metric for text generation. Existing metrics rely on exact matches and fail to capture paraphrases. Instead of exact matches, we compute similarity using contextual embeddings from pretrained BERT with greedy matching and optional IDF weighting. BERTScore correlates better with human judgments than BLEU and METEOR on WMT metrics shared tasks, COCO Captions, and adversarial paraphrase detection (PAWS). Finally, we use baseline rescaling for readability. Pearson and Kendall correlations are reported. No classifier is fine-tuned.
""",
    },
]


def gold_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "gold" / "papers"


def write_gold_set() -> Path:
    d = gold_dir()
    d.mkdir(parents=True, exist_ok=True)
    for p in GOLD_PAPERS:
        (d / f"{p['id']}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    (d.parent / "README.md").write_text(
        "# Gold set\n\nHand-labeled papers for measuring task/component/metric accuracy.\n"
        "Not limited to famous benchmarks — includes niche campus-scale abstracts.\n",
        encoding="utf-8",
    )
    return d

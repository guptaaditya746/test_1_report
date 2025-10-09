# anodex

CLI Orchestrator for Time-Series Anomaly Explanation & Reporting.

## Usage

```bash
anodex --help
```
Of course. Here is the combined data in a single Markdown table:

| Method | Backend | Type | Data | Training Set |
| :--- | :--- | :--- | :--- | :--- |
| **COMTE** | TF, PYT, SK, BB | InstanceBased | multi | y |
| **LEFTIST** | TF, PYT, SK, BB | FeatureAttribution | uni | y |
| **NUN-CF / Native-Guide** | TF, PYT, SK | InstanceBased | uni | y |
| **SETS (shapelet-based CF)** | TF | InstanceBased | uni & multi | y |
| **TSEvo** | TF, PYT | InstanceBased | uni & multi | y |
| **TSR (Temporal Saliency Rescaling)** | TF, PYT | FeatureAttribution | multi | n |
| **Glacier (locally-constrained CF)** | TF, PYT | InstanceBased | uni | n |
| **TimeX (barycenter-guided CF)** | BB | InstanceBased | uni | n |
| **DiscoX (discord-based CF)** | BB | InstanceBased | uni & multi | y |
| **CELS (saliency-guided CF)** | TF, PYT | InstanceBased | uni | y |
| **M-CELS / Info-CELS** | TF, PYT | InstanceBased | multi | y |
| **AB-CF (attention-based CF)** | TF, PYT | InstanceBased | multi | y |
| **SPARCE (GAN-based sparse CF)** | TF, PYT | InstanceBased | multi | y |
| **Sub-SpaCE (subsequence-sparse CF)** | BB | InstanceBased | uni | n |
| **Multi-SpaCE (multi-objective subseq. CF)** | BB | InstanceBased | multi | n |
| **TX-Gen (NSGA-II CF)** | BB | InstanceBased | uni & multi | n |
| **CFWoT (CF *without* training data)** | BB | InstanceBased | multi | n |
| **ForecastCF (forecasting CF)** | TF, PYT | InstanceBased | uni & multi | n |
| **CounTS (self-interpretable predictor w/ CF)** | TF, PYT | InstanceBased | uni & multi | y |
| **LASTS (subsequence explainer w/ counter-exemplars & rules)** | PYT, BB | FeatureAttribution | uni & multi | y |
| **TimeSHAP** | TF, PYT, SK, BB | FeatureAttribution | uni & multi | n |
| **DynaMask** | TF, PYT, BB | FeatureAttribution | multi | n |
| **WinTSR (windowed TSR)** | TF, PYT | FeatureAttribution | uni & multi | n |
| **Motif-Guided TS Counterfactuals (MGCE)** | PYT, BB | InstanceBased | uni | y |
| **Instance-based CF for TSC (Delaney et al.)** | PYT, BB | InstanceBased | uni | y |
| **LIMESegment** | BB | FeatureAttribution | uni & multi | y |
| **DEMUX (Class-Specific Explainability)** | PYT | FeatureAttribution | uni & multi | n |
| **Meaningful Perturbation & Optimisation (ETSC)** | PYT | FeatureAttribution | uni & multi | **y** *(uses a generative prior)* |
| **Virtual Inspection Layers (VIL)** | TF/PYT | FeatureAttribution | uni & multi | n |
| **TSInsight** | BB | FeatureAttribution | uni & multi | n |
| **ALOE (Agnostic Local Explanation for TSC)** | BB | FeatureAttribution | uni & multi | n |
| **Time-is-Not-Enough (Time-Frequency XAI)** | BB | FeatureAttribution | uni & multi | n |
| **Translating Image XAI to MTS** | PYT | FeatureAttribution | multi | n |
| **timeXplain (framework)** | SK/PYT | FeatureAttribution | uni & multi | n |


Of course. Here is the content from the LaTeX `longtable` converted into a structured Markdown format with separate tables for each section.

### 2.1 Feature Attribution and Saliency Methods

#### 2.1.1 Foundational Approaches and Their Adaptation

| Method | Core Principle | Explanation Type | Task Focus | Data Dim. | Backend / Dependencies | Primary Reference(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LIME** | Approximates a black-box model locally with an interpretable surrogate model (e.g., linear regression) trained on perturbations. | Feature Attribution (Segment Importance) | Classification | Univariate, Multivariate | Model-Agnostic (Python, Scikit-learn) | Paper: 28 <br> Repo: 111 |
| **SHAP** | Computes optimal feature importance scores (Shapley values) based on cooperative game theory. | Feature Attribution (Point/Segment Importance) | Classification, Forecasting | Univariate, Multivariate | Model-Agnostic (Python) | Paper: 110 <br> Repo: 12 |
| **Gradient (GRAD)** | Calculates the gradient of the output with respect to the input features to determine importance. | Feature Attribution (Saliency Map) | Classification | Multivariate | Deep Learning (PyTorch, TensorFlow) | Paper: 26 |
| **Integrated Gradients (IG)** | Averages gradients along a path from a non-informative baseline to the input instance. | Feature Attribution (Saliency Map) | Classification, Forecasting | Multivariate | Deep Learning (PyTorch, TensorFlow) | Paper: 13 |
| **DeepLIFT (DL)** | Attributes predictions to features by backpropagating contribution scores based on the difference from a reference activation. | Feature Attribution (Saliency Map) | Classification | Multivariate | Deep Learning (PyTorch, TensorFlow) | Paper: 13 |
| **Feature Occlusion (FO)** | Computes attribution as the difference in model output after replacing a contiguous region with a baseline value. | Feature Attribution (Saliency Map) | Classification | Multivariate | Model-Agnostic | Paper: 13 |

#### 2.1.2 Time-Series Native Saliency

| Method | Core Principle | Explanation Type | Task Focus | Data Dim. | Backend / Dependencies | Primary Reference(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TSR (Temporal Saliency Rescaling)** | Two-step meta-method that first calculates time-relevance scores, then feature-relevance scores within important time steps to decouple domains. | Feature Attribution (Saliency Map) | Classification | Multivariate | Model-Agnostic (operates on saliency maps) | Paper: 17 |
| **WinTSR (Windowed TSR)** | Extends TSR using a sliding window to capture temporal dependencies and delayed impacts more effectively. | Feature Attribution (Saliency Map) | Classification, Regression | Multivariate | Model-Agnostic (operates on saliency maps) | Paper: 32 |
| **FIT** | Measures an observation's importance by the KL-divergence of the predictive distribution at the same time step when the observation is removed. | Feature Attribution (Saliency Map) | Classification | Multivariate | Model-Agnostic | Paper: 13 |
| **WinIT** | Extends FIT by using a window to aggregate importance over future time steps, capturing delayed feature impacts. | Feature Attribution (Saliency Map) | Classification | Multivariate | Model-Agnostic | Paper: 13 |
| **Dynamask** | Fits a dynamic perturbation mask to the input sequence to produce instance-wise importance scores for each feature at each time step. | Feature Attribution (Saliency Mask) | Classification, Forecasting | Multivariate | Model-Agnostic (PyTorch) | Paper: 35 <br> Repo: 36 |

#### 2.1.3 Segment- and Shapelet-Based Attribution

| Method | Core Principle | Explanation Type | Task Focus | Data Dim. | Backend / Dependencies | Primary Reference(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LIMESegment** | Adapts LIME to time series by defining meaningful segmentation, realistic perturbations, and a temporal neighborhood. | Feature Attribution (Segment Importance) | Classification | Univariate, Multivariate | Model-Agnostic | Paper: 14 <br> Repo: 72 |
| **LEFTIST** | A LIME-like framework that uses shapelets (discriminative subsequences) as the interpretable features. | Feature Attribution (Shapelet Importance) | Classification | Univariate, Multivariate | Model-Agnostic (TSInterpret library) | Paper: 38 <br> Repo: 113 |
| **TimeSHAP** | Extends KernelSHAP to sequential data, computing event-, feature-, and cell-level attributions with a temporal pruning algorithm. | Feature Attribution (Event/Feature Importance) | Classification, Forecasting | Multivariate | Model-Agnostic (PyTorch, TensorFlow) | Paper: 31 <br> Repo: 44 |

#### 2.1.4 Advanced Saliency Concepts

| Method | Core Principle | Explanation Type | Task Focus | Data Dim. | Backend / Dependencies | Primary Reference(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Series Saliency** | Treats a time-series window as a 2D "series image" and uses a learnable mask for both data augmentation and interpretation. | Feature Attribution (Saliency Map) | Forecasting | Multivariate | Model-Agnostic (Deep Learning) | Paper: 27 <br> Repo: 115 |
| **TF-LIME** | Extends LIME to the time-frequency domain using STFT and a custom segmentation algorithm (TFHS) to explain models based on frequency components. | Feature Attribution (Time-Frequency Importance) | Classification | Univariate | Model-Agnostic | Paper: 37 <br> Repo: 116 |

### 2.2 Counterfactual Explanations

#### 2.2.2 Instance-Based and Substitution Approaches

| Method | Core Principle | Explanation Type | Task Focus | Data Dim. | Backend / Dependencies | Primary Reference(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CoMTE** | Generates counterfactuals for multivariate time series by substituting entire feature channels from a "distractor" instance from the target class. | Counterfactual Instance | Classification | Multivariate | Model-Agnostic (Python) | Paper: 20 <br> Repo: 19 |
| **Native-Guide / NUN-CF** | Finds the Nearest Unlike Neighbor (NUN) and adapts it by modifying discriminative areas, often using DTW Barycenter Averaging. | Counterfactual Instance | Classification | Univariate | Model-Agnostic (Keras) | Paper: 10 <br> Repo: 55 |
| **TimeX** | Optimization-based method using a Dynamic Barycenter Averaging (DBA) loss term to guide perturbations toward the centroid of the target class. | Counterfactual Instance | Classification | Univariate | Model-Agnostic (PyTorch) | Paper: 51 |
| **DiscoX** | Uses Matrix Profile to find and replace anomalous subsequences (discords) with nearest neighbor subsequences from the target class. | Counterfactual Instance | Classification | Univariate, Multivariate | Model-Agnostic | Paper: 58 |
| **AB-CF** | Uses an attention mechanism to identify and replace only the most important segments of a multivariate time series with segments from a NUN. | Counterfactual Instance | Classification | Multivariate | Model-Agnostic | Paper: 61 <br> Repo: 61 |

#### 2.2.3 Optimization-Based Approaches

| Method | Core Principle | Explanation Type | Task Focus | Data Dim. | Backend / Dependencies | Primary Reference(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Glacier** | Model-agnostic gradient search in original or latent space to generate locally-constrained counterfactuals. | Counterfactual Instance | Classification | Univariate | Model-Agnostic (Python, CUDA) | Paper: 67 <br> Repo: 66 |
| **TSEvo** | Multi-objective evolutionary algorithm (NSGA-II) that generates a Pareto front of diverse counterfactuals for multivariate time series. | Counterfactual Instances (Pareto Front) | Classification | Univariate, Multivariate | Model-Agnostic (PyTorch, TensorFlow) | Paper: 56 <br> Repo: 70 |
| **Sub-SpaCE / Multi-SpaCE** | Evolutionary method (NSGA-II) focused on generating sparse counterfactuals by modifying subsequences. Multi-SpaCE extends to multivariate data and guarantees validity. | Counterfactual Instances (Pareto Front) | Classification | Univariate, Multivariate | Model-Agnostic | Paper: 72 <br> Repo: 72 |
| **TX-Gen** | Multi-objective evolutionary algorithm (NSGA-II) that modifies a single subsequence, using an AR model to generate plausible content. | Counterfactual Instances (Pareto Front) | Classification | Univariate | Model-Agnostic | Paper: 77 |

#### 2.2.4 Generative and Saliency-Guided Approaches

| Method | Core Principle | Explanation Type | Task Focus | Data Dim. | Backend / Dependencies | Primary Reference(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SPARCE** | GAN-based architecture with a custom sparsity layer to generate sparse counterfactuals for multivariate time series. | Counterfactual Instance | Classification | Multivariate | Deep Learning (GAN, LSTM) | Paper: 81 |
| **CELS / Info-CELS / M-CELS** | Learns a saliency map to guide NUN-based perturbations to only the most important time steps. Info-CELS improves validity; M-CELS handles multivariate data. | Counterfactual Instance | Classification | Univariate, Multivariate | Model-Agnostic | Paper: 72 <br> Repo: 72 |
| **SETS / Time-CF** | Leverages shapelets to guide perturbations, either by removing original-class shapelets or introducing target-class shapelets. Time-CF uses a TimeGAN for plausible generation. | Counterfactual Instance | Classification | Multivariate | Model-Agnostic | Paper: 89 <br> Repo: 72 |

#### 2.2.5 Specialized Counterfactual Frameworks

| Method | Core Principle | Explanation Type | Task Focus | Data Dim. | Backend / Dependencies | Primary Reference(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ForecastCF** | Gradient-based optimization to alter input history so the future forecast falls within a desired range. First CF method for forecasting. | Counterfactual Instance | Forecasting | Multivariate | Deep Learning (PyTorch, TensorFlow) | Paper: 95 <br> Repo: 95 |
| **CFWoT** | Reinforcement learning-based method that generates counterfactuals without access to the model's training data. | Counterfactual Instance | Classification | Static, Multivariate | Model-Agnostic (RL) | Paper: 100 |

### 2.3 Exemplar, Rule-Based, and Ante-Hoc Explanations

| Method | Core Principle | Explanation Type | Task Focus | Data Dim. | Backend / Dependencies | Primary Reference(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LASTS** | Provides a hybrid explanation: saliency map, exemplar/counter-exemplar instances from a latent space, and factual/counterfactual rules based on subsequences. | Saliency Map, Exemplars, Rules | Classification | Univariate, Multivariate | Model-Agnostic (VAE) | Paper: 30 <br> Repo: 72 |
| **Temporal Fusion Transformer (TFT)** | Ante-hoc interpretable attention-based architecture for forecasting that reveals globally important features and temporal patterns. | Feature Importance, Attention Patterns | Forecasting | Multivariate | Ante-Hoc (PyTorch) | Paper: 12 |
| **CounTS** | Self-interpretable variational Bayesian model that provides both predictions and actionable counterfactual explanations by design. | Counterfactual Instance | Prediction (Classification/ Regression) | Multivariate | Ante-Hoc (PyTorch) | Paper: 108 <br> Repo: 108 |
| **COMTE-LEFTIST** | Hybrid post-hoc method combining CoMTE's counterfactuals with LEFTIST's shapelet analysis to explain the impact of counterfactual changes. | Counterfactual + Shapelet Importance | Classification | Multivariate | Model-Agnostic | Paper: 42 |

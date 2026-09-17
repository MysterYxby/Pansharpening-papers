# Pansharpening-papers
A collection of papers on pansharpening methods for remote sensing image fusion.
仅讨论 **低分辨率多光谱 MS + 高分辨率全色 PAN → 高分辨率多光谱 HRMS**。不收录高光谱 pansharpening、HSI–MSI 融合及其数据集。

## 1. 收录与分类口径

- **代码公开要求**：必须能找到实际实现文件；仅有论文 PDF、空仓库或“Code coming soon”不算。公开实现不等于已经复现成功，本次未安装环境或运行训练。
- **代码来源**：`作者`为作者/论文关联仓库；`复现`为公开第三方或 benchmark 实现，不能当作原作者代码。PNN、PanNet、DiCNN 保留可用的统一工具箱入口；原始方法年份不等于复现仓库发布时间。
- **开源用语**：这里按科研常用的“源代码公开”口径收录。部分仓库仅允许非营利研究，部分未明确许可证，因此不代表全部符合严格 OSI 开源定义或允许商用。
- **监督学习**：训练时使用成对目标 HRMS，包括通过 Wald 降采样构造的参考 MS。参考目标是模拟的，也仍属于监督学习。
- **非监督学习**：不依赖成对 HRMS 标签，通过空间/光谱一致性、退化模型或对抗约束训练；包括此任务中通常称为 self-supervised 的方法。
- **边界方法**：零样本不自动等于无监督。使用目标图像生成降尺度训练对，再进行无监督适配的方法，单独注明半监督/混合训练。
- **年份**：优先采用正式会议或期刊卷年；已知预印本、在线发表和卷年不一致时附注，避免把一篇文章重复计数。

## 2. 监督学习：按年份梳理

### 2.1 2016—2021：CNN、残差与细节注入

| 年份 / 出处               | 方法与论文                                                   | 核心思路                                          | 公开代码及核查备注                                           |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------ |
| 2016 · Remote Sensing     | **PNN** — [Pansharpening by Convolutional Neural Networks](https://www.mdpi.com/2072-4292/8/7/594) | 将 CNN 用于 PAN/MS 到 HRMS 的映射，是本清单的起点 | [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox)：复现，含 `models/PNN`；不要与后来的 A-PNN、Z-PNN 混为一篇 |
| 2017 · GRSL               | **DRPNN** — [Boosting the Accuracy of Multispectral Image Pansharpening by Learning a Deep Residual Network](https://github.com/Decri/Multi-Scale-and-Depth-CNN-for-Pan-sharpening#readme) | 深层残差学习                                      | [作者联合代码仓库](https://github.com/Decri/Multi-Scale-and-Depth-CNN-for-Pan-sharpening)：包含 DRPNN/MSDCNN 模型、训练资源和 MATLAB demo；论文书目信息见 README |
| 2017 · ICCV               | **PanNet** — [PanNet: A Deep Network Architecture for Pan-Sharpening](https://xueyangfu.github.io/paper/2017/iccv/YangFuetal2017.pdf) | 高频域学习空间细节，残差路径保留光谱信息          | [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox)：复现，含 `models/PanNet` |
| 2018 · JSTARS             | **MSDCNN** — [A Multiscale and Multidepth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening](https://arxiv.org/abs/1712.09809) | 多尺度特征提取与不同深度分支                      | [作者代码](https://github.com/Decri/Multi-Scale-and-Depth-CNN-for-Pan-sharpening)：MatConvNet/Caffe；预印本为 2017 年 |
| 2018 · TGRS               | **A-PNN / PNN+** — [Target-Adaptive CNN-Based Pansharpening](https://ieeexplore.ieee.org/document/8334206) | 残差 PNN 加目标图像适配；使用降尺度训练对         | [作者代码](https://github.com/sergiovitale/pansharpening-cnn)：旧 Theano/Python 2.7 环境；非营利许可；不归入纯无监督 |
| 2019 · JSTARS             | **DiCNN** — [Pansharpening via Detail Injection Based Convolutional Neural Networks](https://doi.org/10.1109/JSTARS.2019.2898574) | 显式学习待注入 MS 的细节；论文区分 DiCNN1/DiCNN2  | [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox)：复现，含 `models/DiCNN`，比较前核对实现变体；[预印本](https://arxiv.org/abs/1806.08898)为 2018 年 |
| 2020 · Information Fusion | **TFNet** — [Remote Sensing Image Fusion Based on Two-stream Fusion Network](https://arxiv.org/abs/1711.02549) | PAN/MS 双流特征提取与特征级融合                   | [公开实现](https://github.com/liouxy/tfnet_pytorch)；[明确标注的第三方复现](https://github.com/xyc19970716/Deep-Learning-PanSharpening)；预印本始于 2017 年 |
| 2021 · TGRS（2020 在线）  | **FusionNet** — [Detail Injection-Based Deep Convolutional Neural Networks for Pansharpening](https://doi.org/10.1109/TGRS.2020.3031366) | 把传统细节注入思想融入 CNN                        | [作者代码](https://github.com/liangjiandeng/FusionNet)：README 采用在线年份 2020，正式卷年为 2021 |
| 2021 · CVPR               | **GPPNN** — [Deep Gradient Projection Networks for Pan-sharpening](https://arxiv.org/abs/2103.04584) | 将梯度投影迭代转化为可学习网络                    | [作者代码](https://github.com/xsxjtu/GPPNN)；采用此处论文链接，仓库 README 的 arXiv 链接存在误指 |
| 2021 · ICCV               | **DCFNet** — [Dynamic Cross Feature Fusion for Remote Sensing Pansharpening](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Dynamic_Cross_Feature_Fusion_for_Remote_Sensing_Pansharpening_ICCV_2021_paper.html) | 动态跨尺度、跨分支特征融合                        | [作者维护框架中的实现](https://github.com/XiaoXiao-Woo/PanCollection/tree/main/pancollection/models/DCFNet)：原独立仓库链接已失效，现框架包含 DCFNet 模型文件 |

### 2.2 2022—2024：模型展开、自适应融合、频域与扩散

| 年份 / 出处              | 方法与论文                                                   | 核心思路                                                     | 公开代码及核查备注                                           |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2022 · IJCAI             | **ADKNet** — [Source-Adaptive Discriminative Kernels based Network for Remote Sensing Pansharpening](https://www.ijcai.org/proceedings/2022/179) | 根据输入模态生成适配的判别卷积核                             | [作者代码](https://github.com/liangjiandeng/ADKNet/tree/main/ADKNet_for_pansharpening)：仅使用多光谱 pansharpening 子目录；README 很简略 |
| 2022 · ECCV              | **MMNet** — [Memory-Augmented Model-Driven Network for Pansharpening](https://doi.org/10.1007/978-3-031-19800-7_18) | 模型驱动迭代结合记忆机制                                     | [作者代码](https://github.com/Keyu-Yan/MMNet)：已存在模型与辅助代码，但 README 仍保留准备上传字样，运行文档有限 |
| 2022 · TGRS              | **D2TNet** — [D2TNet: A ConvLSTM Network With Dual-Direction Transfer for Pan-Sharpening](https://doi.org/10.1109/TGRS.2022.3169134) | ConvLSTM 在尺度和网络层级间双向传递信息，学习 PAN/MS 融合映射 | [作者代码](https://github.com/Meiqi-Gong/D2TNet)：含 model.py、train.py、test.py 和预训练模型；文件名没有 code，但实际已公开 |
| 2022 · CVPR              | **MDCUN** — [Memory-augmented Deep Conditional Unfolding Network for Pan-sharpening](https://doi.org/10.1109/CVPR52688.2022.00183) | 记忆增强的条件展开网络，结合非局部先验与跨迭代信息传递       | [作者代码](https://github.com/yggame/MDCUN)：含 model、solver、main.py 和配置 |
| 2022 · CVPR              | **MIDPS / MutInf** — [Mutual Information-Driven Pan-Sharpening](<D:/科研/论文阅读/图像融合/Pansharpening/2022-MIDPS-CVPR.pdf>) | 最小化模态特征间互信息，减少冗余并鼓励 PAN/MS 互补信息学习   | [作者代码](https://github.com/manman1995/Mutual-Information-driven-Pan-sharpening)：含 models、training；MIDPS、MutInf 对应同一论文 |
| 2022 · TGRS（2021 在线） | **VP-Net** — [VP-Net: An Interpretable Deep Network for Variational Pansharpening](https://doi.org/10.1109/TGRS.2021.3089868) | 把变分 pansharpening 求解步骤展开成可训练网络                | [作者代码](https://github.com/likun97/VP-Net)：TensorFlow 实现，含 fusion_net.py、train.py、test.py 和权重 |
| 2023 · CVPR              | **PGCU** — [Probability-Based Global Cross-Modal Upsampling for Pansharpening](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Probability-Based_Global_Cross-Modal_Upsampling_for_Pansharpening_CVPR_2023_paper.html) | 概率建模的全局跨模态上采样，可嵌入不同主干                   | [作者代码](https://github.com/Zeyu-Zhu/PGCU)：包含模块、数据处理和 WV2/WV3 预训练模型 |
| 2023 · ACM MM            | **BiMPan** — [Bidomain Modeling Paradigm for Pansharpening](https://liangjiandeng.github.io/papers/2023/hou-acmmm2023.pdf) | 空间域建模局部波段特征，傅里叶域恢复全局细节                 | [作者代码](https://github.com/coder-qicao/BiMPan)：训练、测试入口均公开，使用 PanCollection |
| 2023 · TGRS              | **FAFNet** — [Pansharpening via Frequency-Aware Fusion Network With Explicit Similarity Constraints](https://doi.org/10.1109/TGRS.2023.3281829) | 频率感知融合，通过显式相似性约束保持空间与光谱信息           | [作者代码](https://github.com/YinghuiXing/FAFNet)：小波模块、模型、训练和 RR/FR 测试均已公开 |
| 2023 · IJCAI             | **LGTEUN** — [Local-Global Transformer Enhanced Unfolding Network for Pan-sharpening](<D:/科研/论文阅读/图像融合/Pansharpening/2023-LGTEUN-IJCAI-code.pdf>) | 将近端梯度迭代展开，局部–全局 Transformer 作为图像先验模块   | [作者代码](https://github.com/lms-07/LGTEUN)：含模型、配置、日志及自建 GF2/WV2/WV3 数据入口；代码别名 UnlgFormer |
| 2023 · ACM MM            | **U2Net** — [U2Net: A General Framework with Spatial-Spectral-Integrated Double U-Net for Image Fusion](https://doi.org/10.1145/3581783.3612084) | 空间–光谱交互的双 U-Net，统一处理多类图像融合任务            | [作者代码](https://github.com/PSRben/U2Net)：仅收录 PAN+MS 实验；不整理其高光谱任务。含 model、train.py、test.py |
| 2024 · AAAI              | **FAME** — [Frequency-Adaptive Pan-Sharpening with Mixture of Experts](https://ojs.aaai.org/index.php/AAAI/article/download/27984/27985) | 频率自适应专家混合                                           | [作者代码](https://github.com/alexhe101/FAME-Net)            |
| 2024 · NeurIPS           | **SSDiff** — [SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening](https://arxiv.org/abs/2404.11537) | 空间/光谱分支扩散、交替投影融合和频率调制                    | [作者代码](https://github.com/Z-ypnos/SSdiff_main)：包含训练与采样；按会议年份记 2024，勿因仓库 BibTeX 写 2025 重复收录 |
| 2024 · TNNLS 在线        | **VBPN** — [Deep Variational Network for Blind Pansharpening](https://doi.org/10.1109/TNNLS.2024.3436850) | 在贝叶斯变分框架中联合估计退化与融合结果                     | [作者代码](https://github.com/ZhiyuanZhang-WHU/VBPN)：训练/测试和退化模拟公开；“blind”不等于“unsupervised” |
| 2024 · Remote Sensing    | **CMFNet / PanBench** — [Towards Robust Pansharpening: A Large-Scale High-Resolution Multi-Scene Dataset and Novel Approach](https://www.mdpi.com/2072-4292/16/16/2899) | 面向多卫星、多场景融合，同时发布 PanBench                    | [作者代码与数据](https://github.com/XavierJiezou/Pansharpening)：训练、评测、模型和数据入口 |
| 2024 · CVPR              | **CANConv / CANNet** — [Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening](<D:/科研/论文阅读/图像融合/Pansharpening/2024-CANConv-CVPR-code.pdf>) | 按内容聚类的非局部卷积，兼顾远距离关联与动态滤波             | [作者代码](https://github.com/Duanyll/CANConv)：含 canconv、CUDA/C++ 扩展及权重；复现需编译依赖 |
| 2024 · GRSL              | **CMT** — [CMT: Cross Modulation Transformer With Hybrid Loss for Pansharpening](https://doi.org/10.1109/LGRS.2024.3435143) | 跨模态调制 Transformer，结合混合损失学习融合                 | [作者代码](https://github.com/WenjieShu/CMT)：含 model_cmt.py、train_cmt.py、数据与 UDL 框架 |
| 2024 · TGRS              | **DCPNet** — [DCPNet: A Dual-Task Collaborative Promotion Network for Pansharpening](https://doi.org/10.1109/TGRS.2024.3377635) | 融合任务与辅助 MS 超分辨率任务协同训练，相互促进特征恢复     | [作者代码](https://github.com/lhf12278/DCPNet)：实现位于 UDL/pansharpening/models/UPNet，包含模型和训练入口；目录名与论文名不同 |
| 2024 · TGRS              | **PEMAE** — [Pixel-Wise Ensembled Masked Autoencoder for Multispectral Pansharpening](https://doi.org/10.1109/TGRS.2024.3450688) | 将 MS 像素分散为多个掩码模式，集成 MAE 重建，并使用线性交叉注意力 | [作者代码](https://github.com/yc-cui/PEMAE)：正文 IV-A 明确为监督训练：原始 MS 作重建参考；不采用传统 Wald 输入构造。含 src/model/PEMAE.py、train.py |
| 2024 · TGRS              | **TMDiff** — [Empower Generalizability for Pansharpening Through Text-Modulated Diffusion Model](https://doi.org/10.1109/TGRS.2024.3434431) | 文本信息调制扩散网络，增强跨卫星泛化                         | [作者代码](https://github.com/codgodtao/TMDiff)：含扩散与融合实现；注意额外文本/预训练条件及采样开销 |
| 2024 · TGRS              | **UTeRM** — [Deep Unfolding Tensor Rank Minimization With Generalized Detail Injection for Pansharpening](https://doi.org/10.1109/TGRS.2024.3392215) | 张量秩最小化展开结合广义细节注入，包含 CS/MRA/CNN 等变体     | [作者代码](https://github.com/mtntruong/UTeRM)：监督项使用参考 HRMS 的 L1 损失；不同细节注入版本需分开报告 |

### 2.3 2025—2026：状态空间、预训练迁移与新近工作

| 年份 / 出处                                      | 方法与论文                                                   | 核心思路                                                     | 公开代码及核查备注                                           |
| ------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2025 · Information Fusion（2024 预印本）         | **Pan-Mamba** — [Pan-Mamba: Effective pan-sharpening with state space model](https://www.sciencedirect.com/science/article/pii/S1566253524005578) | 状态空间长距离建模、通道交换与跨模态交互                     | [作者代码](https://github.com/alexhe101/Pan-Mamba)：主干模块、训练、推理与权重说明；[预印本](https://arxiv.org/abs/2402.12192) |
| 2025 · AAAI                                      | **WFANet** — [Wavelet-Assisted Multi-Frequency Attention Network for Pansharpening](https://arxiv.org/abs/2502.04903) | 小波分解、多频注意力和多尺度融合                             | [作者代码](https://github.com/Jie-1203/WFANet)：已公开训练/测试代码和权重，关联 WV3/GF2/QB/WV2 |
| 2025 · AAAI（2024 预印本）                       | **PanAdapter** — [PanAdapter: Two-Stage Fine-Tuning with Spatial-Spectral Priors Injecting for Pansharpening](https://ojs.aaai.org/index.php/AAAI/article/view/32912) | 空间/光谱先验结合 adapter，迁移预训练图像处理模型            | [公开复现](https://github.com/RC-Wu/PanAdapter)：两阶段训练说明；**未确认为原作者仓库**，依赖 IPT/EDSR 权重 |
| 2025 · CVPR                                      | **ADWM** — [A General Adaptive Dual-level Weighting Mechanism for Remote Sensing Pansharpening](<D:/科研/论文阅读/图像融合/Pansharpening/2025-ADWM-CVPR-code.pdf>) | 自适应双层权重机制，结合通道协方差与图像/通道频率加权        | [作者代码](https://github.com/Jie-1203/ADWM)：作者代码公开；属通用机制，实验应注明搭配主干 |
| 2025 · CVPR                                      | **ARConv / ARNet** — [Adaptive Rectangular Convolution for Remote Sensing Pansharpening](<D:/科研/论文阅读/图像融合/Pansharpening/2025-ARNet-CVPR-code.pdf>) | 根据内容调整矩形卷积的高宽及采样位置，适配遥感地物形状       | [作者代码](https://github.com/WangXueyang-uestc/ARConv)：含 models、scripts、trainer.py；ARConv 模块与 ARNet 方法别名合并计数 |
| 2025 · TGRS                                      | **BPDUN** — [BPDUN: Bidirectional Progressive Deep Unfolding Network for Pansharpening](<D:/科研/论文阅读/图像融合/Pansharpening/2025-BPDUN-TGRS.pdf>) | 双向渐进深度展开，增强迭代阶段间的空间–光谱信息传递          | [作者代码](https://github.com/fliery/BPDUN)：含 BPDUN_main/BPDUN.py；PDF 中网址的 ﬂ 连字须还原为 fl |
| 2025 · ISPRS JPRS（2024 在线）                   | **PreMix** — [Pansharpening via predictive filtering with element-wise feature mixing](https://doi.org/10.1016/j.isprsjprs.2024.10.029) | 预测滤波结合逐元素特征混合，学习空间信息注入                 | [作者代码](https://github.com/yc-cui/PreMix)：含 src、train.py、test.py 和依赖说明 |
| 2025 · arXiv（v3：2025-11-12）                   | **PanTiny** — [Rethinking Pan-sharpening: A New Training Process for Full-Resolution Generalization](https://arxiv.org/abs/2507.15059) | 统一 WV2/WV3/GF2 多数据集训练，轻量模型结合复合重建损失改善 FR 泛化 | [作者代码](https://github.com/Zirconium233/PanTiny)：本地名写 2026，实际 arXiv:2507.15059 始于 2025；README 题名有版本差异，尚不据此认定 2026 正式发表 |
| 2026 · AAAI（2025 预印本）                       | **MMMamba** — [MMMamba: A Versatile Cross-Modal in Context Fusion Framework for Pan-Sharpening and Zero-Shot Image Enhancement](https://ojs.aaai.org/index.php/AAAI/article/download/38933/42895) | 交错扫描与跨模态上下文融合                                   | [作者代码](https://github.com/Gracewangyy/MMMamba)：含模型、训练和测试；标题中的 zero-shot image enhancement 不意味着其 pansharpening 训练无监督 |
| 2026 · TGRS                                      | **TMGformer** — [TMGformer: Text-Modulated Multiscale Guidance Transformer for Pansharpening](https://doi.org/10.1109/TGRS.2026.3651576) | 文本语义调制视觉特征并指导多尺度融合                         | [作者代码](https://github.com/PeterZhaoXJTU/TMGformer)：训练、文本特征与数据准备说明；比较时需考虑额外预训练模型/文本信息 |
| 2026 · TGRS（仓库年份冲突，见注）                | **TS-Pan** — [Two-Step Pansharpening: A High-Frequency-Guided Spatial-Spectral Enhancement Network Based on Mixture of Experts](https://ieeexplore.ieee.org/document/11316500) | 两步空间–光谱增强、高频引导和专家混合                        | [作者代码](https://github.com/lzm-01/TS-Pan)：训练、测试、效率统计和 MATLAB 评测；仓库标题为 2026，BibTeX year 为 2025，正式引用需以出版记录为准 |
| 2026 · Remote Sensing（7 月 9 日）               | **PanDiM** — [PanDiM: A Diffusion Mamba Network for High-Fidelity Pansharpening](https://www.mdpi.com/2072-4292/18/14/2299) | 扩散与 Mamba 结合                                            | [作者代码](https://github.com/HongshiXu/PanDiM)：已有 `models`、`diffusion`、`train.py`；README 文档不足；论文使用 PanCollection 的 WV3/GF2/QB |
| 2026 · Infrared Physics & Technology（9 月卷期） | **NIRFreq** — [NIRFreq: A frequency-aware and spectrally guided framework for multispectral pansharpening](https://doi.org/10.1016/j.infrared.2026.106789) | RGB–NIR 相关性引导和频率感知融合                             | [作者代码](https://github.com/Spa2k1e/NIRFreq)：仓库称 NIRFreqNet，提供实现及 checkpoint；GF1/IKONOS/WV2，关联 NBU 数据 |
| 2026 · EAAI（2026-05-27 在线）                   | **PDTUN** — [Progressive dynamic Taylor unfolding network for multi-modal image fusion](https://doi.org/10.1016/j.engappai.2026.115218) | 用 Taylor 展开构建动态细节注入网络，递进建模高阶信息         | [作者代码](https://github.com/RSMagneto/PDTUN)：含 PDTUN.py 模型；仅整理多光谱 RIF 分支（参考图像 MAE 监督），不混入其红外/医学无监督任务 |
| 2026 · AAAI                                      | **Pan-TCR** — [Pansharpening for Thin-Cloud Contaminated Remote Sensing Images: A Unified Framework and Benchmark Dataset](<D:/科研/论文阅读/图像融合/Pansharpening/2026-PANTCR-AAAI-2026.pdf>) | 联合薄云去除与 pansharpening，融合物理先验并发布 PanTCR-GF2  | [作者代码](https://github.com/dusongcheng/PanTCR-GF2)：含 net.py、train.py、test.py、数据下载入口；这是带云扩展任务，不能直接与普通清晰场景结果横比 |

## 3. 非监督学习：按年份梳理

> 这里的“无监督”针对是否有 HRMS 配对标签，而非是否完全不使用训练数据、预训练权重或传感器知识。全分辨率适配的方法还应报告逐图优化耗时。

| 年份 / 出处                              | 方法与论文                                                   | 训练信号与核心思路                                           | 公开代码及核查备注                                           |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2020 · Information Fusion                | **Pan-GAN** — [Pan-GAN: An unsupervised pan-sharpening method for remote sensing image fusion](https://doi.org/10.1016/j.inffus.2020.04.006) | 对抗学习及空间/光谱约束，无配对 HRMS 监督                    | [作者代码](https://github.com/yuwei998/PanGAN)；注意区分监督型 PSGAN |
| 2021 · JSTARS                            | **PGMAN** — [PGMAN: An Unsupervised Generative Multiadversarial Network for Pansharpening](https://arxiv.org/abs/2012.09054) | 多对抗约束平衡空间与光谱质量                                 | [作者代码](https://github.com/zhysora/PGMAN)：数据构建、训练和测试均有入口；预印本 2020 |
| 2022 · JSTARS（2021 预印本）             | **LDP-Net** — [LDP-Net: An Unsupervised Pansharpening Network Based on Learnable Degradation Processes](https://arxiv.org/abs/2111.12483) | 学习空间重模糊与光谱灰度化退化，以观测一致性约束输出         | [论文对应代码](https://github.com/suifenglian/LDP-Net)：含网络、数据集和主程序；不要使用同名但属于 few-shot learning 的 NWPUZhoufei 仓库 |
| 2022 · TGRS                              | **UCGAN** — [Unsupervised Cycle-Consistent Generative Adversarial Networks for Pan Sharpening](https://doi.org/10.1109/TGRS.2022.3166528) | 循环一致性与对抗学习                                         | [作者代码](https://github.com/zhysora/UCGAN)                 |
| 2022 · TGRS                              | **Z-PNN** — [Pansharpening by Convolutional Neural Networks in the Full Resolution Framework](https://doi.org/10.1109/TGRS.2022.3163887) | 在全分辨率上利用光谱与空间保真损失训练并做目标适配           | [作者代码](https://github.com/matciotola/Z-PNN)：含环境、运行脚本与示例说明；非营利研究许可 |
| 2023 · Remote Sensing                    | **Fast Z-PNN** — [Fast Full-Resolution Target-Adaptive CNN-Based Pansharpening Framework](https://doi.org/10.3390/rs15020319) | 加速全分辨率目标适配；可用于不同 CNN 主干                    | [作者代码](https://github.com/matciotola/fast-z-pnn)：含 Fast/Faster 配置相关实现；非营利研究许可 |
| 2023 · TGRS                              | **λ-PNN / Lambda-PNN** — [Unsupervised Deep Learning-Based Pansharpening With Jointly Enhanced Spectral and Spatial Fidelity](https://doi.org/10.1109/TGRS.2023.3299356) | 残差注意力网络与联合空间/光谱保真损失                        | [作者代码](https://github.com/matciotola/Lambda-PNN)：训练、推理、配准与权重公开；非营利研究许可；[预印本](https://arxiv.org/abs/2307.14403) |
| 2024 · TIP                               | **CrossDiff** — [CrossDiff: Exploring Self-Supervised Representation of Pansharpening via Cross-Predictive Diffusion Model](https://doi.org/10.1109/TIP.2024.3461476) | 先以跨模态预测扩散学习表示，再以无监督空间/光谱损失训练融合头 | [作者代码](https://github.com/codgodtao/CrossDiff)：含扩散模型及 fusion 系列入口；分类依据最终融合阶段损失，不只看 self-supervised 标题 |
| 2024 · TGRS                              | **PSDip** — [Variational Zero-Shot Multispectral Pansharpening](https://doi.org/10.1109/TGRS.2024.3492059) | 单幅图像对的零样本变分优化，借助深度先验估计融合及非线性关系 | [作者代码](https://github.com/xyrui/PSDip)：实际实现已公开；应报告逐图优化开销，而非仅统计网络前向时间 |
| 2024 · Information Fusion（2023 在线）   | **Zero-Sharpen** — [Zero-Sharpen: A universal pansharpening method across satellites for reducing scale-variance gap via zero-shot variation](https://doi.org/10.1016/j.inffus.2023.102003) | 零样本神经网络与变分模型协同更新，针对当前 PAN/MS 对减少尺度差异 | [作者代码](https://github.com/Baixuzx7/ZeroSharpen)：不需预先训练或 HRMS 参考样本；结合观测约束进行逐图优化 |
| 2026 · Information Fusion（DOI 含 2025） | **UCL** — [Unsupervised coefficient learning framework for variational pansharpening](https://doi.org/10.1016/j.inffus.2025.103790) | 通过无监督非线性系数学习改进变分融合，只需当前 PAN/MS 图像对 | [作者代码](https://github.com/Jin-liangXiao/UCL)：已有 `models`、`tools` 和 `test_reduced.py`；使用说明较少，不能据此承诺完整一键训练 |
| 2026 · AAAI（2025 预印本）               | **CLIPPan** — [CLIPPan: Adapting CLIP as a Supervisor for Unsupervised Pansharpening](https://ojs.aaai.org/index.php/AAAI/article/download/37451/41413) | 将适配后的 CLIP 作为语义监督信号，约束全分辨率融合；需区分基础模型预训练与目标融合训练 | [作者代码](https://github.com/Jiabo-Liu/CLIPPan)：已存在 `train_stage_I.py`、`train_stage_II.py` 和 `models/clip_adapt.py`；README 的 coming soon 说明尚未同步更新 |

### 3.1 边界补充：半监督与混合训练

| 年份 / 出处                            | 方法与论文                                                   | 为什么单列                                                   | 公开代码                                                     |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2023 · Information Fusion（2022 在线） | **P2Sharpen** — [P2Sharpen: A progressive pansharpening network with deep spectral transformation](https://doi.org/10.1016/j.inffus.2022.10.010) | 深度光谱变换与渐进融合，先降尺度监督学习，再在真实尺度适配   | [作者代码](https://github.com/Baixuzx7/P2Sharpen)：分阶段含监督与无监督约束，单列混合训练；含 Model、训练/评测入口 |
| 2023 · TGRS                            | **CrossNet** — [Cross-Resolution Semi-Supervised Adversarial Learning for Pansharpening](https://doi.org/10.1109/TGRS.2023.3271658) | 跨分辨率半监督对抗学习，联合降尺度有参考数据和全分辨率无参考数据 | [作者代码](https://github.com/RSMagneto/CrossNet)：目前有 net.py 模型；未见完整训练入口，不能写成一键复现 |
| 2024 · Information Fusion（2023 在线） | **ZS-Pan** — [Zero-shot semi-supervised learning for pansharpening](https://doi.org/10.1016/j.inffus.2023.102001) | 单个 PAN/MS 图像对，先构建降尺度监督训练，再进行全分辨率无监督生成；论文明确自称 semi-supervised，不应标为纯无监督 | [作者代码](https://github.com/coder-qicao/ZS-Pan)：包含 RSP、SDE、训练与测试相关入口 |

## 4. 数据集与下载入口（仅多光谱）

### 4.1 优先使用的公开数据资源

| 数据资源                     | 组成 / 特点                                                  | 用途与注意事项                                               | 下载 / 原始来源                                              |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **PanCollection**            | WV3、QB、GF2 提供训练与测试；经典数据发布页的 WV2 主要提供测试；RR/FR、H5/MAT 格式 | 很多近期方法使用；适合统一 baseline 和跨传感器测试。经典发布页列出各传感器 20 个测试样例，训练规模及划分以具体版本为准 | [数据仓库](https://github.com/liangjiandeng/PanCollection)，README 内有 Google Drive/百度网盘；[训练评测框架](https://github.com/XiaoXiao-Woo/PanCollection)是另一个仓库 |
| **PanBench**                 | 5,898 对；10 类卫星来源：GF1/GF2/GF6、IKONOS、Landsat7/8、QB、WV2/3/4；统一 RGB+NIR 四波段，MS 256×256、PAN 1024×1024 | 适合多场景、多卫星泛化；其 WV2/WV3 是四波段配置，不可直接与 PanCollection 的八波段结果合并比较 | [论文](https://www.mdpi.com/2072-4292/16/16/2899)、[作者仓库](https://github.com/XavierJiezou/Pansharpening)、[Hugging Face 下载](https://huggingface.co/datasets/XavierJiezou/pansharpening-datasets/blob/main/PanBench.zip) |
| **NBU_PansharpRSData**       | 2021 论文发布 2,270 对：IKONOS 200、QB 500、GF1 410、WV4 500、WV2 500、WV3 160；另有主题场景集合 | 可用于多来源测试及质量评价；使用发布的说明核对传感器、波段、裁块和划分，不将它与 PanCollection 当成同一数据集 | [作者数据仓库](https://github.com/starboot/NBU_PansharpRSData)、[论文 DOI](https://doi.org/10.1109/MGRS.2020.2976696)；仓库有百度网盘主/备份入口，数据限学术用途 |
| **PAirMax / PAirMax-Airbus** | 2021 PAirMax 论文基准为 14 对 PAN/MS；另有 PAirMax-Airbus 发布包（Pléiades、SPOT-7，5 个 FR + 5 个 RR 测试案例），两者不要混为一个规模 | 更适合标准化评测与真实场景泛化，不宜直接当作大规模训练集；Airbus 压缩包有密码，需按发布页流程接受许可并申请 | [协议与论文](https://openremotesensing.net/knowledgebase/a-benchmarking-protocol-for-pansharpening-dataset-preprocessing-and-quality-assessment/)、[Airbus Zenodo 数据](https://zenodo.org/records/7706102) |
| **LGTEUN 配套数据**          | GF2/WV2/WV3；RR 训练分别 1036/1012/910 对，RR 测试 136/145/144 对，FR 测试各 120 对 | 自建场景划分，不能将其结果直接当作同卫星名的 PanCollection 结果；README 的 WF-2 是 WV2 笔误 | [作者发布说明与 Google Drive 入口](https://github.com/lms-07/LGTEUN#datasets-and-file-hierarchy) |
| **PanTCR-GF2**               | Pan-TCR 的 GF2 薄云污染 PAN/MS 基准；含联合去云与融合所需数据 | 用于带云扩展任务；需核对作者的云合成、参考图像及训练测试划分 | [代码/数据仓库及百度网盘入口](https://github.com/dusongcheng/PanTCR-GF2) |


### 4.2 常见卫星名与实验配置

| 缩写     | 卫星 / 传感器来源 | 本清单常见 MS 波段配置                     | 实验中需要核对                                             |
| -------- | ----------------- | ------------------------------------------ | ---------------------------------------------------------- |
| QB       | QuickBird         | 4                                          | 常用于经典 CNN 和 PanCollection baseline                   |
| GF2      | GaoFen-2          | 4                                          | PanCollection 常采用 1023 归一化，须以实际数据量化方式为准 |
| WV2      | WorldView-2       | 8；PanBench 为选取的 4 波段                | 同名卫星不保证相同波段、场景或 train/test split            |
| WV3      | WorldView-3       | 常用 VNIR 8 波段；PanBench 为选取的 4 波段 | 常见研究配置不是把该卫星所有传感器波段一起使用             |
| GE1 / IK | GeoEye-1 / IKONOS | 常见 4 波段                                | 可用于全分辨率与跨传感器泛化；以具体数据发布为准           |

在 PanCollection 的典型 ×4 设置中，LRMS 为 `H×W×C`，PAN 为 `4H×4W×1`，目标 HRMS 为 `4H×4W×C`。这一比例不能推广为所有卫星原始产品的固定比例，尤其不要直接套到 Landsat 数据。

## 5. 训练与评测协议：比较论文前先统一

### 5.1 RR 与 FR

- **Reduced Resolution（RR）**：采用传感器相关低通/MTF 与降采样构造更低尺度输入，以原始 MS 作参考目标，报告有参考指标。
- **Full Resolution（FR）**：直接融合原始 PAN/MS，通常不存在真实 HRMS 标签，报告无参考指标、视觉结果和实际应用表现。
- **训练范式与评测协议是两件事**：监督模型也能在 FR 测试，无监督模型也能在 RR 用参考指标评价。不能把 RR 叫“监督指标”、FR 叫“无监督指标”。

| 场景       | 常见指标               | 方向                                |
| ---------- | ---------------------- | ----------------------------------- |
| RR，有参考 | SAM、ERGAS             | 越低越好                            |
| RR，有参考 | Q/Q2n、PSNR、SSIM、SCC | 越高越好；明确具体实现和数值范围    |
| FR，无参考 | Dλ、Ds                 | 越低越好                            |
| FR，无参考 | QNR、HQNR              | 越高越好；不同定义/实现不能直接混用 |

协议来源：[A Benchmarking Protocol for Pansharpening: Dataset, Preprocessing, and Quality Assessment](https://openremotesensing.net/knowledgebase/a-benchmarking-protocol-for-pansharpening-dataset-preprocessing-and-quality-assessment/)。实现入口：[DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox)。

公平比较至少统一：数据版本、按场景划分、波段数与顺序、归一化、MTF/降采样、配准、边界裁剪、指标实现。对于扩散与目标自适应方法，额外报告采样步数、逐图优化次数、耗时与显存；不要只比较参数量。

## 6. 快速阅读与复现顺序

以下是按学习成本和技术覆盖面给出的建议，不是性能排名。

1. **建立任务认识**：PNN → PanNet → DiCNN/FusionNet，理解直接映射、高频残差与细节注入。
2. **理解结构改进**：GPPNN/VP-Net → MDCUN/LGTEUN → PGCU/CANConv，理解模型展开、上采样和频域融合。
3. **理解真实尺度问题**：Pan-GAN/PGMAN → LDP-Net → Z-PNN → λ-PNN → CrossDiff/PSDip/Zero-Sharpen → UCL，重点看损失、退化假设与适配成本。
4. **跟踪新方法**：SSDiff → Pan-Mamba/WFANet/ADWM/ARConv → MMMamba/TMGformer/PanDiM/NIRFreq，核对新增模块是否带来一致的 RR 与 FR 收益。
5. **开展实验**：优先以 PanCollection + DLPan-Toolbox 建立统一结果，再加入 PanBench 或 PAirMax 做跨场景/真实尺度验证。

一个较紧凑的起始 baseline 组合：监督侧 **PNN、PanNet、FusionNet、GPPNN、PGCU、SSDiff、Pan-Mamba、WFANet**；非监督侧 **Pan-GAN、LDP-Net、Z-PNN、λ-PNN、UCL**。ZS-Pan 作为半监督补充单独对照。



### 7 数据、综述与评价资料（10 篇）

这些资料用于数据来源、领域背景与评价协议，不计入 61 篇方法数。覆盖多任务的综述只使用其多光谱 pansharpening 内容。

| #    | 本地文件                                                     | 正文题名                                                     | 纳入位置 / 用途                                              |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [公开数据集\NBU_PansharpRSData-2021.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/公开数据集/NBU_PansharpRSData-2021.pdf>) | A Large-Scale Benchmark Data Set for Evaluating Pansharpening Performance: Overview and Implementation | 第 4 节数据/benchmark 来源                                   |
| 2    | [公开数据集\PairMax-2021.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/公开数据集/PairMax-2021.pdf>) | A Benchmarking Protocol for Pansharpening: Dataset, Preprocessing, and Quality Assessment | 第 4 节数据/benchmark 来源                                   |
| 3    | [公开数据集\Pancollection-2021.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/公开数据集/Pancollection-2021.pdf>) | A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods | 第 4 节命名纠正：Vivone 等 benchmark，非 PanCollection 数据论文 |
| 4    | [综述论文\Deng-MGRS-2021.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/综述论文/Deng-MGRS-2021.pdf>) | Machine Learning in Pansharpening: A benchmark, from shallow to deep networks | 综述背景：辅助梳理技术路线，不当作独立方法                   |
| 5    | [综述论文\Deng-中图学报-2023.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/综述论文/Deng-中图学报-2023.pdf>) | 遥感图像全色锐化的卷积神经网络方法研究进展                   | 综述背景：辅助梳理技术路线，不当作独立方法                   |
| 6    | [综述论文\GC-MGRS-2024.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/综述论文/GC-MGRS-2024.pdf>) | Deep Learning in Remote Sensing Image Fusion: Methods, protocols, data, and future perspectives | 综述背景：辅助梳理技术路线，不当作独立方法                   |
| 7    | [综述论文\Hassan-INFFus-2016.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/综述论文/Hassan-INFFus-2016.pdf>) | A Review of Remote Sensing Image Fusion Methods              | 综述背景：辅助梳理技术路线，不当作独立方法                   |
| 8    | [综述论文\Li-JAG-2022.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/综述论文/Li-JAG-2022.pdf>) | Deep learning in multimodal remote sensing data fusion: A comprehensive review | 综述背景：辅助梳理技术路线，不当作独立方法                   |
| 9    | [评价指标\1367002.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/评价指标/1367002.pdf>) | A proposal for full-scale processing and assessment of MS pansharpening | 第 5 节 FR 质量评估补充；关注配准、harmonization 与指标定义  |
| 10   | [评价指标\2022-HQNR-MGRS.pdf](<D:/科研/论文阅读/图像融合/Pansharpening/评价指标/2022-HQNR-MGRS.pdf>) | Full-Resolution Quality Assessment of Pansharpening: Theoretical and hands-on approaches | 第 5 节 FR 质量评估补充；关注配准、harmonization 与指标定义  |

## 8. 核查边界与后续维护

- 论文依据：本地正文的题名、方法、损失及实验设置，辅以 DOI、会议出版页、arXiv 与作者仓库。文件名中的 code、年份和简称不能单独作为证据。
- 代码依据：核查仓库是否能访问、是否有匹配模型/训练实现；未运行训练，不承诺复现精度。作者仓库、第三方复现、仅模型公开的差别已逐项注明。
- 数据依据：核查论文规模及发布入口，未下载全部数据；不得把不同卫星版本、不同波段数和不同划分合并比较。
- 更新时至少记录：年份/出处、完整题名、最终任务训练信号、论文链接、代码来源与实际文件、数据划分、核心思路。确认实现后，再将第 7.1 节条目移入主表。

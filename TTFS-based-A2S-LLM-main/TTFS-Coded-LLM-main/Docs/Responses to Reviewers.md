# Responses to Reviewers 



### Reviewer uzd3 (6, 3)
> **W1**:It is still unclear what advantages this method has compared to the baseline, i.e., the quantized LLM.

We thank the reviewer for the question. We would like to emphasize that the key distinction lies in the **target hardware**:
1. Our SNN-based method is designed for **in-memory neuromorphic hardware**. Event-driven computation and co-optimized memory access give large energy and latency benefits. These advantages are confirmed in prior works [1,2]

2. Quantized LLMs aim to reduce **digital computation cost** on **traditional GPUs** through lower-precision arithmetic. Therefore, while quantization **does not exploit** the architectural advantages of neuromorphic platforms. Our method, by leveraging SNN dynamics, achieves a fundamentally different efficiency benefit tailored to neuromorphic systems, which cannot be captured by merely quantizing LLMs on conventional GPUs.


[1] Yao M. et al., “Spike‑based dynamic computing with asynchronous event‑driven, sparse and dynamic neuromorphic system on chip”, Nature Communications, 2024.
[2] Muir D.R., Sheik S., “The road to commercial success for neuromorphic technologies”, Nature Communications, 2025.


1. **Non-Von-Neumann:** Existing quantized LLM baselines are designed for a conventional Von Neumann architecture, where computation and memory are separated, high-throughput digital arithmetic is assumed, and tokens are processed sequentially. These methods aim to reduce digital computation cost, but they do not take into account the compute–memory fusion and event-driven characteristics of non-Von-Neumann neuromorphic chips. 
2. **Neuromorphic Hardware**: In contrast, our method is an algorithm–architecture co-design specifically targeting neuromorphic hardware. Neuromorphic processors rely on in-memory computing, asynchronous execution, and massive parallelism, which fundamentally change the data representation, operator execution, and computational pathways. Directly deploying a quantized LLM onto such hardware is not efficient and often leads to significant latency and energy degradation.

Therefore, the advantage of our method does not lie in being a simple variant of traditional quantized LLMs; rather, it lies in its ability to **fully exploit the native properties of non-Von-Neumann neuromorphic chips to achieve substantial latency and energy improvements**.

> **W2**:The guiding idea of this method (i.e., converting the quantized ANN into an SNN) may not be original. Similar ideas have been proposed in previous works, and this work may only extend this idea to SNNs with TTFS encoding.[1][2]
> [1] Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks
> [2] SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking

We sincerely thank the reviewer for the careful reading and accurate understanding of our work. As the reviewer correctly pointed out, our method indeed involves mapping representations from an ANN into an SNN. We appreciate the reviewer’s clear grasp of the high-level motivation behind our approach. Our contributions lie in two fundamental aspects that have not been explored in prior TTFS, ANN-SNN conversion：
1. **New Discretization-aware Theoretical Framework:** Existing TTFS-based conversion methods rely heavily on continuous-time assumptions, which are not aligned with actual hardware implementations that inevitably operate in discrete time steps. Our framework is the first to:
(1) identify the fundamental discrepancy between continuous-time TTFS formulations and discrete hardware dynamics
(2) reveal the formal equivalence between TTFS conversion errors and quantization error bounds.
This discretization-aware TTFS theory did not exist in any prior TTFS-based conversion works and forms a core theoretical innovation of our paper.
2. **Deployment-friendly Discrete-equivalent Representation:**
prior work基于连续的假设，忽视了误差的影响，所以需要相当大的latency来维持对浮点激活的近似。（1024）不能做到LLM
正是因为基于这样的离散分析框架，以及对误差的清晰刻画，我们才可以应用平滑技术缓解误差，从而实现low latency的时序脉冲大模型。
Building on our theoretical framework, we introduce a discrete-equivalent representation that is natively compatible with real hardware, rather than relying on continuous approximations. This representation:
(1) establishes a strict discrete correspondence between quantized Transformer activations and TTFS spiking representations,
(2) ensures that all layers, intermediate states, and attention operations are represented in hardware-native discrete temporal events,
Unlike prior work that uses continuous surrogates or pseudo-spike approximations, our discrete-equivalent formulation is the first truly deployment-friendly representation for temporal-coding spiking LLMs.


> **W3**:The lack of a pseudocode description of the algorithm makes it insufficiently clear.

We thank the reviewer for pointing this out. To address this issue, we will include a clear and concise pseudocode block in the revised version to summarize the entire QC-A2S pipeline

> **Q1**:What specific quantization methods are used when quantizing LLM in this paper? Are specific quantization methods required? Are both weights and activations quantized? For operations in the nonlinear parts, are the calculations performed using quantized activations and weights? Please elaborate in detail.

We adopt PrefixQuant in this paper, as our method requires static quantization. Both weights and activations are quantized, and all nonlinear operations are computed using the quantized weights and quantized activations.

> **Q2**:Please explain the advantages of converting the quantized LLM into an SNN? Does it improve inference efficiency, or does it reduce energy consumption? Can data support be provided?

Sample to W1.

<br>

<br>


### Reviewer Rkrn (4, 4)
>**W1**: Theorems 1 and 2 are essentially discrete versions of the continuous theory proposed by Zhao et al. (2025). The extension from continuous to discrete is relatively straightforward, with limited theoretical innovation. The error analysis in Theorem 3 lacks comparative validation between the theoretically predicted error and the actual observed error. The comparison in Figure 2 is insufficiently clear and requires more detailed explanation.

**Discrete TTFS Is Not a Trivial Extension:** We fully agree that our theorems are conceptually connected to the continuous-time TTFS framework of Zhao et al. (2025). However, the reviewer’s observation that the discrete extension is “straightforward” does not fully reflect the technical challenges involved. In fact, moving from continuous time to discrete time fundamentally changes the behavior of TTFS neurons because:
(1) the spike time can no longer be solved in closed form,
(2) the membrane dynamics become piecewise-constant,
(3) the equivalence between spike timing and ANN activation no longer holds unless discrete constraints are explicitly enforced.
These issues do not appear in any continuous-time analysis, including Zhao et al. (2025). Thus, Theorems 1 and 2 are not merely a “discrete counterpart,” but the first theoretical framework that explicitly resolves the mismatch between continuous TTFS theory and real, discrete timesteps used by hardware.
**Discrete TTFS Error Structure:** Our contribution goes beyond extending previous continuous formulations. In particular:

1. We show that discretization error is formally equivalent to quantization error in integer LLMs (a new theoretical link).
2. We characterize the exact rounding-induced deviation between spike timing and ANN activations, which does not exist in continuous theory.
3. We provide the first discrete-time error bound that is compatible with hardware implementation.

These results constitute theoretical innovations that are not present in Zhao et al. (2025) or prior TTFS literature.

**Comparative validation for Theorem 3**: In addition, we would like to clarify the dependence of the upper bound in Theorem 3 on the timestep $T$ and the number of layers $L$: As shown in our theoretical bound, a larger $T$ results in a smaller discretization error. This directly leads to improved empirical accuracy, which is consistent with our observations. Although deeper models contain more layers, the accuracy does not continue to increase. This is because the error bound in Theorem 3 accumulates with respect to the number of layers $L$; hence, when $L$ becomes large, the accumulated discretization error offsets the benefit of increased model capacity. This explains why scaling from 7B to 13B does not yield substantial accuracy gains in our TTFS-SNN setting.

Finally, to improve clarity, we have added a more detailed explanation of the algorithmic workflow in the caption of Figure 2.

> **W2**: The experimental setup has issues. The absence of comparisons for TTFSFormer at T=64 prevents a fair evaluation of method performance. Table 3 lacks comparisons with other models, diminishing its persuasiveness. Table 3 only investigates the impact of T, neglecting the influence of quantization bit depth. We recommend adding an analysis of the trade-off between quantization bit depth and latency.

In accordance with the reviewer’s suggestions, we have further improved the experimental section as follows:
1. **T=64 for TTFSFormer:** To ensure a fair comparison, the revised version now includes TTFSFormer evaluated under the temporal resolution T=64, alongside our model.(<span style="color:red;">to_do:Experiment</span>)
2. **Additional LLaMA-3-8B in Table 3:** We have added LLaMA-3-8B experiments analyzing the effect of different $T$ values, similar to Table 3.(<span style="color:red;">to_do:Experiment</span>)

**Clarification of Trade-off**: We clarify that in our conversion framework, the quantization bit depth directly corresponds to the latency (maximum time steps) in TTFS. Because our framework establishes a discrete-equivalent mapping, a higher quantization bit depth provides finer discrete spike-time resolution, enabling a larger T and thus lower discretization error, while a lower bit depth implies coarser temporal resolution, yielding a smaller T and therefore lower latency but potentially higher error.


> **W3**: Experiments max out at 70B parameters, with no discussion of scalability for larger models.

We added the experiment with LLaMA-2-70B.(<span style="color:red;">to_do:Experiment</span>)


> **W4**: Recommendations: Add charts, visualizations, pseudocode, etc., to enhance narrative clarity.

We have added the pseudocode for the QC-A2S framework.

> **Q1**: The specific implementation details of the method remain somewhat ambiguous. Section 4.4 states, “we first apply established techniques, such as post-training quantization,” but does not explicitly specify which PTQ method is used. Is it PrefixQuant? If so, what distinguishes this approach from directly employing PrefixQuant?

In Section 4.4, we have added the algorithm pseudocode along with detailed descriptions of the procedure. The PTQ method used in our framework is PrefixQuant. Same to W1.

> **Q2**: How is the trade-off between quantization bit depth n and time window T selected? What is the rationale for choosing T=64? Is this choice empirical or based on theoretical analysis?

We clarify that in our conversion framework, the quantization bit depth directly corresponds to the latency (maximum time steps) in TTFS. We select the time window T based on its inherent correspondence with the quantization bit depth. We also tested $T=32$ and $T=16$.(<span style="color:red;">to_do:Experiment</span>)

<br>

<br>


### Reviewer EyDs (4, 3)
> W1:Although an error bound is proposed (Theorem 3), there is no systematic experimental validation of its tightness, nor is the actual distribution of errors across layers and time-steps presented

We thank the reviewer for the question. We clarify the following:
1. In TTFS coding, the output is a single spike time rather than the number of spikes per timestep; hence, there is no timestep-wise accumulation error. TTFS produces exactly one spike per neuron, and the representation is encoded solely in the firing time. Therefore, the discretization error arises from the mapping between continuous-valued activations and discrete spike times, not from errors accumulated across individual timesteps.
2. We have added layer-wise error visualizations in the revised version. To validate Theorem 3, we plot the actual discretization error for each layer, which clearly shows how the error behaves across layers.


> W2:In the comparison with TTFSFormer, only the high-latency setting (T=8192) is used, failing to show TTFSFormer's performance at medium latencies. this makes it difficult to assert the absolute advantage of QC-A2S

To ensure a fair comparison, the revised version now includes TTFSFormer evaluated under the temporal resolution T=64,512,2048 alongside our model.(<span style="color:red;">to_do:Experiment</span>)

> W3:Despite the proposed method is emphasized hardware-friendly, there are no actual deployment tests on any neuromorphic hardware or FPGAs. the claimed energy efficiency advantages remain theoretical speculation

类脑芯片部署，一类问题

> W4:In expeirments, the quantitative analysis of the energy-accuracy trade-off is missing, so it is difficult to judge whether the method is truly superior to pure quantization methods in practical applications

1. In TTFS, each neuron emits at most one spike, and energy is dominated by the number of spikes, not the number of timesteps. Therefore, larger $T$ only increases inference time, but does not increase spike-based energy cost, while simultaneously improving accuracy by offering finer temporal resolution. This energy–accuracy decoupling is a unique advantage of TTFS-based computation.

2. Our method is designed for non–Von Neumann neuromorphic architectures, where computation is event-driven and in-memory.

> W5:The experiments is limited to the LLaMA architecture. The applicability to other Transformer variants (eg. encoder-only model) or non-autoregressive models is not tested, it is suggested to test more architectures

添加Qwen实验 (<span style="color:red;">to_do:Experiment</span>)

> Q1:does the error bound in Theorem 3 have practical guiding significance?

In addition, we would like to clarify the dependence of the upper bound in Theorem 3 on the timestep $T$ and the number of layers $L$: As shown in our theoretical bound, a larger $T$ results in a smaller discretization error. This directly leads to improved empirical accuracy, which is consistent with our observations. Although deeper models contain more layers, the accuracy does not continue to increase. This is because the error bound in Theorem 3 accumulates with respect to the number of layers $L$; hence, when $L$ becomes large, the accumulated discretization error offsets the benefit of increased model capacity. This explains why scaling from 7B to 13B does not yield substantial accuracy gains in our TTFS-SNN setting.


> Q2:Why wasn't TTFSFormer compared at medium latencies (T=256 or T=1024)? Is it because its performance would still be competitive with or superior to QC-A2S?

补充TTFSFormer中低延迟实验，一类问题

> Q3:Are there plans to validate energy efficiency on real neuromorphic hardware? If not, isn't the claim of being energy-efficient overly optimistic?

类脑芯片部署，一类问题

> Q4:Is QC-A2S applicable to non-LLaMA architectures?

添加Qwen实验，一类问题

<br>

<br>

### Reviewer M5Md (4, 4)

> **Q1**: Figure 1 illustrates a layer-by-layer serial inference pipeline. If the number of layers $L$ is very large (as in LLMs), the total inference latency will be $L \times T$ (where $T$ is the time window), which may be unacceptably long. Is it possible to implement a parallel TTFS inference scheme (e.g., similar to the mechanism shown in Figure 2 of the paper "Efficiently Training Time-to-First-Spike Spiking Neural Networks from Scratch")? If not, could the authors discuss the primary difficulties in implementing such parallel TTFS-SNN inference?

We thank the reviewer for the deep understanding of this issue. Indeed, there exist prior works that explore parallelized TTFS inference in simpler feedforward SNNs. However, none of these approaches have been successfully extended to large-scale LLMs, whose layer dependencies and attention mechanisms fundamentally complicate such parallelization. Moreover, attempting to parallelize TTFS inference in LLMs would inevitably introduce additional approximation errors, as partial or predicted spike times must be used before full membrane integration is completed. These approximations typically require extra calibration procedures to maintain accuracy, which undermines the theoretical guarantees provided by standard TTFS coding.


> **Q2**: One of the paper's core motivations is the energy efficiency of SNNs. However, the advantages of the proposed TTFS-SNN over its corresponding QANN (Quantized ANN) in terms of computation and storage are not clear to me: a) Computational Complexity: In Equation 7 of section 4.1, it seems that to perform the forward of TTFS-based SNN, we first need to encode the time information into an integer (from spike time to an integer $(t-t_{j}^{l-1})$ and then multiply it by the weight and $\eta$. What is the advantage of this compared to multiplying the weight by the integer input in an ANN? The traditional hardware advantage of SNNs is replacing high-energy MAC operations with simple ACC operations. However, in this paper's framework (e.g., Eq. 7 and Appendix E), multiplication operations do not seem to have been eliminated. I observed that the Softmax and RMSNorm implementations in Appendix E also appear to contain complex multiplication and power operations. b) Storage Cost: The authors use $T=64$ in their experiments. If represented in a one-hot fashion, storing this spike train might require 64 bits. In contrast, the QNN baselines (W8A6) use only 6-bit activations, which is much smaller. If we don't consider storing the spike train by using dataflow-type neuromorphic chip, the single spike information also need auxiliary routing information like the neuron id which effect the energy efficient. c) Leakage energy: T=64 will lead to much larger static energy consumption compare with 1 in ANN. Can author discuss it also? Further, since rate-based SNN can reduce the time window size to a super small number like 1 or 4. Then is TTFS still have advantage?

We thank the reviewer for these detailed and insightful questions. We address points $(a)$, $(b)$, and $(c)$ below：

a) **Computational Complexity and Multiplications：** The key advantage does not lie at the level of digital simulation, but in how the computation is physically realized on non–Von Neumann neuromorphic hardware. In our TTFS formulation, the term $(t-t_{j}^{l-1})$ corresponds to a spike arrival time difference, which is naturally represented as an event delay rather than a digital integer. Likewise, the weight is encoded directly as the conductance of a crossbar synapse, and the operation in Eq. (7) is carried out through analog in-memory current integration, not digital multiplication. Thus, although the GPU simulation shows a multiply, the actual hardware execution eliminates digital MACs entirely: the timing difference is inherent in the event dynamics, and the “multiplication” is realized physically by the crossbar’s weighted charge accumulation. This provides a fundamentally lower-energy computation pathway compared to multiplying quantized integers in a traditional ANN running on Von Neumann hardware. Regarding the reviewer’s concern about the power operations and multiplications in Softmax and RMSNorm, we clarify that these operations appear only in the GPU/CPU simulation and theoretical equivalence formulation, and do not represent how the functions would be implemented on neuromorphic hardware. On non–Von Neumann chips, such nonlinearities are typically realized through lookup tables (LUTs), piecewise-linear approximations, membrane-potential normalization circuits, or simple spike-time comparisons, rather than per-element high-precision multiplications or power functions. Moreover, in LLMs, the dominant energy cost comes from large matrix MACs; the few multiplications or power operations in Softmax/RMSNorm are only O(d) and negligible compared to the  O(d2) cost of attention and FFN layers. Under TTFS semantics, these nonlinearities correspond more closely to relative spike-time ordering and normalization, not literal exp/pow/mul computations.

b) **Storage Cost and Spike Representation：** TTFS does not require storing a 64-bit one-hot spike train. In TTFS, each neuron emits exactly one spike, represented by a single integer time value. Thus, storing activation requires only: $log_2(T) = 6$ bits
for $T=64$, which is identical to W8A6 activation precision. Furthermore, in neuromorphic hardware, spike routing is performed using Address-Event Representation (AER), where each spike is encoded as an event containing only the neuron ID and its firing time. This means that TTFS does not require storing or transmitting a T-length spike train; instead, each neuron emits only a single AER packet, whose cost is constant and independent of the time window T. Therefore, the routing and storage overhead remains lightweight and does not scale with the temporal resolution, which is fundamentally different from how activations are represented in conventional quantized ANNs.

c) **Leakage / Static Energy and Advantage over Rate-based SNNs:** Regarding static (leakage) energy, we note that the static power of neuromorphic hardware is extremely low, often several orders of magnitude smaller than the dynamic MAC cost in conventional ANN accelerators, and therefore is typically negligible in the overall energy budget. Furthermore, while rate-based SNNs can reduce their time window to very small values (e.g., 1 or 4), this does not necessarily translate into energy efficiency in practice. Rate-based models usually require complex decoding mechanisms to recover analog values from multiple spikes, and these decoding operations themselves incur substantial energy and routing overhead. In contrast, TTFS always uses exactly one spike per neuron, achieving both high representational precision and minimal event-driven energy. This makes TTFS fundamentally more energy-efficient and more hardware-friendly than rate-based SNNs, even when the latter operate with very short time windows.

> **Q3.1**: If this TTFS-SNN perform a similar number of complex multiplications as a sparse QANN, and may even be at a disadvantage in storage and memory access, what is its true advantage in SNN compared to a QNN that can directly leverage hardware sparsity ？

类脑芯片部署，一类问题

> **Q3**: Given the issues above, I wish to see a direct comparison between the proposed TTFS-SNN and the QNN it was converted from, in terms of energy consumption and actual wall-clock inference time. If the SNN merely matches the QNN's accuracy but cannot demonstrate an advantage in energy or speed, what is the significance of this conversion? The experiment of SNNs seem conduct on A100 also, one more question is : do experiments conducted on NVIDIA A100 GPU truly reflect the real advantages of the SNN?

类脑芯片部署，一类问题

> **Q4**: One more minor question: In Equation (1) and (4), what is the blue color intended to emphasize?

The blue color is used solely to highlight the components that differ from the continuous-time TTFS formulation. These terms represent the discrete-time modifications introduced by our discretization-aware framework and are marked in blue to help readers clearly distinguish them from the corresponding continuous expressions. We will clarify this intention explicitly in the revised version. 

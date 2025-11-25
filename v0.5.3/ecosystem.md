---
layout: page
---

<script setup>
import { VPTeamPage, VPTeamPageTitle, VPTeamMembers, VPTeamPageSection } from 'vitepress/theme'

const extends_lux = [
  {
    avatar: 'https://github.com/SciML.png',
    name: 'DiffEqFlux.jl',
    desc: 'Universal neural differential equations with O(1) backprop, GPUs, and stiff+non-stiff DE solvers, demonstrating scientific machine learning (SciML) and physics-informed machine learning methods',
    links: [
      { icon: 'github', link: 'https://github.com/SciML/DiffEqFlux.jl' }
    ]
  },
  {
    avatar: 'https://github.com/SciML.png',
    name: 'SciMLSensitivity.jl',
    desc: 'A component of the DiffEq ecosystem for enabling sensitivity analysis for scientific machine learning (SciML). Optimize-then-discretize, discretize-then-optimize, adjoint methods, and more for ODEs, SDEs, DDEs, DAEs, etc.',
    links: [
      { icon: 'github', link: 'https://github.com/SciML/SciMLSensitivity.jl' }
    ]
  },
  {
    avatar: 'https://github.com/SciML.png',
    name: 'NeuralPDE.jl',
    desc: 'Physics-Informed Neural Networks (PINN) and Deep BSDE Solvers of Differential Equations for Scientific Machine Learning (SciML) accelerated simulation',
    links: [
      { icon: 'github', link: 'https://github.com/SciML/NeuralPDE.jl' }
    ]
  },
  {
    avatar: 'https://github.com/SciML.png',
    name: 'NeuralLyapunov.jl',
    desc: 'A library for searching for neural Lyapunov functions in Julia',
    links: [
      { icon: 'github', link: 'https://github.com/SciML/NeuralLyapunov.jl' }
    ]
  },
  {
    avatar: 'https://github.com/SciML.png',
    name: 'DeepEquilibriumNetworks.jl',
    desc: 'Implicit Layer Machine Learning via Deep Equilibrium Networks, O(1) backpropagation with accelerated convergence',
    links: [
      { icon: 'github', link: 'https://github.com/SciML/DeepEquilibriumNetworks.jl' }
    ]
  },
  {
    avatar: 'https://github.com/CosmologicalEmulators.png',
    name: 'AbstractCosmologicalEmulators.jl',
    desc: 'Repository containing the abstract interface to the emulators used in the CosmologicalEmulators organization',
    links: [
      { icon: 'github', link: 'https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl' }
    ]
  },
  {
    avatar: 'https://github.com/impICNF.png',
    name: 'ContinuousNormalizingFlows.jl',
    desc: 'Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia',
    links: [
      { icon: 'github', link: 'https://github.com/impICNF/ContinuousNormalizingFlows.jl' }
    ]
  },
  {
    avatar: 'https://github.com/YichengDWu.png',
    name: 'Sophon.jl',
    desc: 'Efficient, Accurate, and Streamlined Training of Physics-Informed Neural Networks',
    links: [
      { icon: 'github', link: 'https://github.com/YichengDWu/Sophon.jl' }
    ]
  },
  {
    avatar: 'https://github.com/SciML.png',
    name: 'DataDrivenDiffEq.jl',
    desc: 'Data driven modeling and automated discovery of dynamical systems for the SciML Scientific Machine Learning organization',
    links: [
      { icon: 'github', link: 'https://github.com/SciML/DataDrivenDiffEq.jl' }
    ]
  },
  {
    avatar: 'https://github.com/YichengDWu.png',
    name: 'NeuralGraphPDE.jl',
    desc: 'Integrating Neural Ordinary Differential Equations, the Method of Lines, and Graph Neural Networks',
    links: [
      { icon: 'github', link: 'https://github.com/YichengDWu/NeuralGraphPDE.jl' }
    ]
  },
  {
    avatar: 'https://github.com/vavrines.png',
    name: 'Solaris.jl',
    desc: 'Lightweight module for fusing physical and neural models',
    links: [
      { icon: 'github', link: 'https://github.com/vavrines/Solaris.jl' }
    ]
  },
  {
    avatar: 'https://github.com/avik-pal.png',
    name: 'FluxMPI.jl',
    desc: 'Distributed Data Parallel Training of Deep Neural Networks',
    links: [
      { icon: 'github', link: 'https://github.com/avik-pal/FluxMPI.jl' }
    ]
  },
  {
    avatar: 'https://github.com/LuxDL.png',
    name: 'Boltz.jl',
    desc: ' Accelerate your ML research using pre-built Deep Learning Models with Lux',
    links: [
      { icon: 'github', link: 'https://github.com/LuxDL/Boltz.jl' }
    ]
  },
  {
    avatar: 'https://as1.ftcdn.net/jpg/01/09/84/42/220_F_109844212_NnLGUrn3RgMHQIuqSiLGlc9d419eK2dX.jpg',
    name: 'Want to Add Your Package?',
    desc: 'Open a PR in <u><a href="https://github.com/LuxDL/luxdl.github.io">LuxDL/luxdl.github.io</a></u>'
  }
];

const autodiff = [
  {
    avatar: 'https://github.com/FluxML.png',
    name: 'Zygote.jl',
    desc: 'Lux.jl default choice for AD',
    links: [
      { icon: 'github', link: 'https://github.com/FluxML/Zygote.jl' }
    ]
  },
  {
    avatar: 'https://github.com/FluxML.png',
    name: 'Tracker.jl',
    desc: 'Well tested and robust AD library (might fail on edge cases)',
    links: [
      { icon: 'github', link: 'https://github.com/FluxML/Tracker.jl' }
    ]
  },
  {
    avatar: 'https://github.com/JuliaDiff.png',
    name: 'ForwardDiff.jl',
    desc: 'For forward mode AD support',
    links: [
      { icon: 'github', link: 'https://github.com/JuliaDiff/ForwardDiff.jl' }
    ]
  },
  {
    avatar: 'https://github.com/JuliaDiff.png',
    name: 'ReverseDiff.jl',
    desc: "Tape based reverse mode AD (might fail on edge cases and doesn't work on GPU)",
    links: [
      { icon: 'github', link: 'https://github.com/JuliaDiff/ReverseDiff.jl' }
    ]
  },
  {
    avatar: 'https://github.com/EnzymeAD.png',
    name: 'Enzyme.jl',
    desc: 'Experimental Support but will become the Future Default',
    links: [
      { icon: 'github', link: 'https://github.com/EnzymeAD/Enzyme.jl' }
    ]
  }
];

const dataload = [
  {
    avatar: 'https://github.com/evizero.png',
    name: 'Augmentor.jl',
    desc: 'Data augmentation for machine learning',
    links: [
      { icon: 'github', link: 'https://github.com/evizero/Augmentor.jl' }
    ]
  },
  {
    avatar: 'https://github.com/JuliaML.png',
    name: 'MLUtils.jl',
    desc: 'Utilities and abstractions for Machine Learning tasks',
    links: [
      { icon: 'github', link: 'https://github.com/JuliaML/MLUtils.jl' }
    ]
  },
  {
    avatar: 'https://github.com/JuliaML.png',
    name: 'MLDatasets.jl',
    desc: 'Utility package for accessing common Machine Learning datasets in Julia',
    links: [
      { icon: 'github', link: 'https://github.com/JuliaML/MLDatasets.jl' }
    ]
  },
  {
    avatar: 'https://github.com/JuliaImages.png',
    name: 'Images.jl',
    desc: 'An image library for Julia',
    links: [
      { icon: 'github', link: 'ttps://github.com/JuliaImages/Images.jl' }
    ]
  },
  {
    avatar: 'https://github.com/FluxML.png',
    name: 'DataAugmentation.jl',
    desc: 'Flexible data augmentation library for machine and deep learning',
    links: [
      { icon: 'github', link: 'https://github.com/FluxML/DataAugmentation.jl' }
    ]
  }
];

const nnprimitives = [
  {
    avatar: 'https://github.com/FluxML.png',
    name: 'NNlib.jl',
    desc: 'Neural Network primitives with multiple backends',
    links: [
      { icon: 'github', link: 'https://github.com/FluxML/NNlib.jl' }
    ]
  },
  {
    avatar: 'https://github.com/LuxDL.png',
    name: 'LuxLib.jl',
    desc: 'Backend for Lux.jl',
    links: [
      { icon: 'github', link: 'https://github.com/LuxDL/LuxLib.jl' }
    ]
  }
];

const optimization = [
  {
    avatar: 'https://github.com/SciML.png',
    name: 'Optimization.jl',
    desc: 'Unified API for Optimization in Julia',
    links: [
      { icon: 'github', link: 'https://github.com/SciML/Optimization.jl' }
    ]
  },
  {
    avatar: 'https://github.com/FluxML.png',
    name: 'Optimisers.jl',
    desc: 'Optimisers.jl defines many standard optimisers and utilities for learning loops',
    links: [
      { icon: 'github', link: 'https://github.com/FluxML/Optimisers.jl' }
    ]
  },
  {
    avatar: 'https://github.com/FluxML.png',
    name: 'ParameterSchedulers.jl',
    desc: 'Common hyperparameter scheduling for ML',
    links: [
      { icon: 'github', link: 'https://github.com/FluxML/ParameterSchedulers.jl' }
    ]
  },
];

const param_manipulation = [
  {
    avatar: 'https://github.com/FluxML.png',
    name: 'Functors.jl',
    desc: 'Parameterise all the things',
    links: [
      { icon: 'github', link: 'https://github.com/FluxML/Functors.jl' }
    ]
  },
  {
    avatar: 'https://github.com/jonniedie.png',
    name: 'ComponentArrays.jl',
    desc: 'Arrays with arbitrarily nested named components',
    links: [
      { icon: 'github', link: 'https://github.com/jonniedie/ComponentArrays.jl' }
    ]
  }
];

const serialization = [
  {
    avatar: 'https://github.com/JuliaLang.png',
    name: 'Serialization.jl',
    desc: 'Provides serialization of Julia objects',
    links: [
      { icon: 'github', link: 'https://github.com/JuliaLang/julia/tree/master/stdlib/Serialization' }
    ]
  },
  {
    avatar: 'https://github.com/JuliaIO.png',
    name: 'JLD2.jl',
    desc: 'HDF5-compatible file format in pure Julia',
    links: [
      { icon: 'github', link: 'https://github.com/JuliaIO/JLD2.jl' }
    ]
  }
];

const test_utils = [
  {
    avatar: 'https://github.com/JuliaDiff.png',
    name: 'FiniteDiff.jl',
    desc: 'Fast non-allocating calculations of gradients, Jacobians, and Hessians with sparsity support',
    links: [
      { icon: 'github', link: 'https://github.com/JuliaDiff/FiniteDiff.jl' }
    ]
  },
  {
    avatar: 'https://github.com/JuliaDiff.png',
    name: 'FiniteDifferences.jl',
    desc: 'High accuracy derivatives, estimated via numerical finite differences (formerly FDM.jl)',
    links: [
      { icon: 'github', link: 'https://github.com/JuliaDiff/FiniteDifferences.jl' }
    ]
  },
  {
    avatar: 'https://github.com/aviatesk.png',
    name: 'JET.jl',
    desc: 'JET employs Julia\'s type inference system to detect potential bugs and type instabilities',
    links: [
      { icon: 'github', link: 'https://github.com/aviatesk/JET.jl' }
    ]
  },
  {
    avatar: 'https://github.com/LuxDL.png',
    name: 'LuxTestUtils.jl',
    desc: 'Collection of Functions useful for testing various packages in the Lux Ecosystem',
    links: [
      { icon: 'github', link: 'https://github.com/LuxDL/LuxTestUtils.jl' }
    ]
  }
];

const trainvis = [
  {
    avatar: 'https://github.com/JuliaAI.png',
    name: 'MLFlowClient.jl',
    desc: 'Julia client for MLFlow',
    links: [
      { icon: 'github', link: 'https://github.com/JuliaAI/MLFlowClient.jl' }
    ]
  },
  {
    avatar: 'https://github.com/JuliaLogging.png',
    name: 'TensorBoardLogger.jl',
    desc: 'Easy peasy logging to TensorBoard with Julia',
    links: [
      { icon: 'github', link: 'https://github.com/JuliaLogging/TensorBoardLogger.jl' }
    ]
  },
  {
    avatar: 'https://github.com/avik-pal.png',
    name: 'Wandb.jl',
    desc: 'Unofficial Julia bindings for logging experiments to wandb.ai',
    links: [
      { icon: 'github', link: 'https://github.com/avik-pal/Wandb.jl' }
    ]
  }
];
</script>

<VPTeamPage>
  <VPTeamPageTitle>
    <template #title>Ecosystem</template>
  </VPTeamPageTitle>

  <VPTeamPageSection>
    <template #title>Frameworks Extending Lux.jl</template>
    <template #members>
      <VPTeamMembers size="small" :members="extends_lux" />
    </template>
  </VPTeamPageSection>

  <VPTeamPageSection>
    <template #title>Automatic Differentiation</template>
    <template #members>
      <VPTeamMembers size="small" :members="autodiff" />
    </template>
  </VPTeamPageSection>

  <VPTeamPageSection>
    <template #title>Data Manipulation, Data Loading & Datasets</template>
    <template #members>
      <VPTeamMembers size="small" :members="dataload" />
    </template>
  </VPTeamPageSection>

  <VPTeamPageSection>
    <template #title>Neural Network Primitives</template>
    <template #members>
      <VPTeamMembers size="small" :members="nnprimitives" />
    </template>
  </VPTeamPageSection>

  <VPTeamPageSection>
    <template #title>Optimization</template>
    <template #members>
      <VPTeamMembers size="small" :members="optimization" />
    </template>
  </VPTeamPageSection>

  <VPTeamPageSection>
    <template #title>Parameter Manipulation</template>
    <template #members>
      <VPTeamMembers size="small" :members="param_manipulation" />
    </template>
  </VPTeamPageSection>

  <VPTeamPageSection>
    <template #title>Serialization</template>
    <template #members>
      <VPTeamMembers size="small" :members="serialization" />
    </template>
  </VPTeamPageSection>

  <VPTeamPageSection>
    <template #title>Testing Utilities</template>
    <template #members>
      <VPTeamMembers size="small" :members="test_utils" />
    </template>
  </VPTeamPageSection>

  <VPTeamPageSection>
    <template #title>Training Visualization & Logging</template>
    <template #members>
      <VPTeamMembers size="small" :members="trainvis" />
    </template>
  </VPTeamPageSection>
</VPTeamPage>



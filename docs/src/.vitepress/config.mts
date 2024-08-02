import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import { transformerMetaWordHighlight } from '@shikijs/transformers';

// https://vitepress.dev/reference/site-config
export default defineConfig({
    base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
    title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    description: 'Documentation for LuxDL Repositories',
    cleanUrls: true,
    outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...

    markdown: {
        math: true,
        config(md) {
            md.use(tabsMarkdownPlugin),
                md.use(mathjax3),
                md.use(footnote)
        },
        theme: {
            light: "github-light",
            dark: "github-dark"
        },
        codeTransformers: [transformerMetaWordHighlight(),],
    },

    head: [
        [
            "script",
            { async: "", src: "https://www.googletagmanager.com/gtag/js?id=G-Q8GYTEVTZ2" },
        ],
        [
            "script",
            {},
            `window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'G-Q8GYTEVTZ2');`,
        ],
        ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: '/apple-touch-icon.png' }],
        ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon-32x32.png' }],
        ['link', { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/favicon-16x16.png' }],
        ['link', { rel: 'icon', href: '/favicon.ico' }],
        ['link', { rel: 'manifest', href: '/site.webmanifest' }],
    ],

    themeConfig: {
        outline: 'deep',
        // https://vitepress.dev/reference/default-theme-config
        logo: {
            'light': '/lux-logo.svg',
            'dark': '/lux-logo-dark.svg'
        },
        search: {
            provider: 'local',
            options: {
                detailedView: true
            }
        },
        nav: [
            { text: 'Home', link: '/' },
            { text: 'Getting Started', link: '/introduction' },
            { text: 'Benchmarks', link: 'https://lux.csail.mit.edu/benchmarks/' },
            { text: 'Tutorials', link: '/tutorials/' },
            { text: 'Manual', link: '/manual/interface' },
            {
                text: 'API', items: [
                    {
                        text: 'Lux', items: [
                            { text: 'Built-In Layers', link: '/api/Lux/layers' },
                            { text: 'Automatic Differentiation', link: '/api/Lux/autodiff' },
                            { text: 'Utilities', link: '/api/Lux/utilities' },
                            { text: 'Experimental', link: '/api/Lux/contrib' },
                            { text: 'InterOp', link: '/api/Lux/interop' },
                            { text: 'DistributedUtils', link: '/api/Lux/distributed_utils' }
                        ]
                    },
                    {
                        text: 'Accelerator Support', items: [
                            { text: 'LuxDeviceUtils', link: '/api/Accelerator_Support/LuxDeviceUtils' },
                            { text: 'MLDataDevices', link: '/api/Accelerator_Support/MLDataDevices' }
                        ]
                    },
                    {
                        text: 'Building Blocks', items: [
                            { text: 'LuxCore', link: '/api/Building_Blocks/LuxCore' },
                            { text: 'LuxLib', link: '/api/Building_Blocks/LuxLib' },
                            { text: 'WeightInitializers', link: '/api/Building_Blocks/WeightInitializers' },
                            { text: 'NNlib', link: 'https://fluxml.ai/NNlib.jl/dev/' },
                            { text: 'Activation Functions', link: 'https://fluxml.ai/NNlib.jl/dev/reference/#Activation-Functions' }
                        ]
                    },
                    {
                        text: 'Domain Specific Modeling', items: [
                            { text: 'Boltz', link: '/api/Domain_Specific_Modeling/Boltz' }
                        ]
                    },
                    {
                        text: 'Testing Functionality', items: [
                            { text: 'LuxTestUtils', link: '/api/Testing_Functionality/LuxTestUtils' }
                        ]
                    }
                ]
            },
            {
                text: 'Versions', items: [
                    { text: 'Stable', link: 'https://lux.csail.mit.edu/stable' },
                    { text: 'Dev', link: 'https://lux.csail.mit.edu/dev' }
                ]
            }
        ],
        sidebar: {
            "/introduction/": {
                text: 'Getting Started', collapsed: false, items: [
                    { text: 'Introduction', link: '/introduction' },
                    { text: 'Overview', link: '/introduction/overview' },
                    { text: 'Resources', link: '/introduction/resources' },
                    { text: 'Citation', link: '/introduction/citation' }]
            },
            "/tutorials/": {
                text: 'Tutorials', collapsed: false, items: [
                    { text: 'Overview', link: '/tutorials/' },
                    {
                        text: 'Beginner', collapsed: false, items: [
                            { text: 'Julia & Lux for the Uninitiated', link: '/tutorials/beginner/1_Basics' },
                            { text: 'Fitting a Polynomial using MLP', link: '/tutorials/beginner/2_PolynomialFitting' },
                            { text: 'Training a Simple LSTM', link: '/tutorials/beginner/3_SimpleRNN' },
                            { text: 'MNIST Classification with SimpleChains', link: '/tutorials/beginner/4_SimpleChains' }]
                    },
                    {
                        text: 'Intermediate', collapsed: false, items: [
                            { text: 'MNIST Classification using Neural ODEs', link: '/tutorials/intermediate/1_NeuralODE' },
                            { text: 'Bayesian Neural Network', link: '/tutorials/intermediate/2_BayesianNN' },
                            { text: 'Training a HyperNetwork on MNIST and FashionMNIST', link: '/tutorials/intermediate/3_HyperNet' }]
                    },
                    {
                        text: 'Advanced', collapsed: false, items: [
                            { text: 'Training a Neural ODE to Model Gravitational Waveforms', link: '/tutorials/advanced/1_GravitationalWaveForm' },
                            { text: 'Solving Optimal Control Problems with Symbolic UDEs', link: '/tutorials/advanced/2_SymbolicOptimalControl' },]
                    },
                    {
                        text: 'Large Models', collapsed: true, items: [
                            { text: 'Training Image Classification Models on ImageNet with Distributed Data Parallel Training', link: 'https://github.com/LuxDL/Lux.jl/tree/main/examples/ImageNet' },
                            { text: 'Training a DDIM (Diffusion Model) for Image Generation', link: 'https://github.com/LuxDL/Lux.jl/tree/main/examples/DDIM' },
                            { text: 'ConvMixer on CIFAR-10', link: 'https://github.com/LuxDL/Lux.jl/tree/main/examples/ConvMixer' },
                        ]
                    },
                    {
                        text: '3rd Party Tutorials', collapsed: true, items: [
                            {
                                text: 'PINNs (NeuralPDE.jl)', link: 'https://docs.sciml.ai/NeuralPDE/stable/tutorials/pdesystem/'
                            },
                            {
                                text: 'UDEs (SciMLSensitivity.jl)', link: 'https://docs.sciml.ai/SciMLSensitivity/stable/tutorials/data_parallel/'
                            },
                            {
                                text: 'Neural DEs (DiffEqFlux.jl)', link: 'https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/'
                            },
                            {
                                text: 'DEQs (DeepEquilibriumNetworks.jl)', link: 'https://docs.sciml.ai/DeepEquilibriumNetworks/stable/tutorials/basic_mnist_deq/'
                            },
                            {
                                text: 'Medical Image Segmentation', link: 'https://github.com/Dale-Black/ComputerVisionTutorials.jl/'
                            },
                            {
                                text: 'Neural closure models', link: 'https://github.com/agdestein/NeuralClosureTutorials/'
                            }
                        ]
                    }]
            },
            "/manual/": {
                text: 'Manual', collapsed: false, items: [
                    {
                        text: 'Basics', items: [
                            { text: 'Lux Interface', link: '/manual/interface' },
                            { text: 'Freezing Parameters', link: '/manual/freezing_model_parameters' },
                            { text: 'GPU Management', link: '/manual/gpu_management' },
                            { text: 'Initializing Weights', link: '/manual/weight_initializers' },
                        ]
                    },
                    {
                        text: 'Automatic Differentiation', items: [
                            { text: 'Automatic Differentiation', link: '/manual/autodiff' },
                            { text: 'Nested AutoDiff', link: '/manual/nested_autodiff' },
                        ]
                    },
                    {
                        text: 'Debugging / Performance Enhancement Tools',
                        items: [
                            { text: 'Debugging Lux Models', link: '/manual/debugging' },
                            { text: 'Performance Pitfalls', link: '/manual/performance_pitfalls' },
                        ]
                    },
                    {
                        text: 'Migration Guides', items: [
                            { text: 'Migrating from Flux', link: '/manual/migrate_from_flux' },
                        ]
                    },
                    {
                        text: 'Advanced Usage', items: [
                            { text: 'Custom Input Types', link: '/manual/dispatch_custom_input' },
                            { text: 'Configuration via Preferences', link: '/manual/preferences' },
                            { text: 'Distributed Training', link: '/manual/distributed_utils' },
                            { text: 'Lux In GPU Kernels', link: '/manual/nn_inside_gpu_kernels' },
                        ]
                    },
                ]
            },
            "/api/": {
                text: 'API Reference', collapsed: false, items: [
                    {
                        text: 'Lux', collapsed: false, items: [
                            { text: 'Built-In Layers', link: '/api/Lux/layers' },
                            { text: 'Automatic Differentiation', link: '/api/Lux/autodiff' },
                            { text: 'Utilities', link: '/api/Lux/utilities' },
                            { text: 'Experimental Features', link: '/api/Lux/contrib' },
                            { text: 'Interoperability', link: '/api/Lux/interop' },
                            { text: 'DistributedUtils', link: '/api/Lux/distributed_utils' }]
                    },
                    {
                        text: 'Accelerator Support', collapsed: false, items: [
                            { text: 'LuxDeviceUtils', link: '/api/Accelerator_Support/LuxDeviceUtils' },
                            { text: 'MLDataDevices', link: '/api/Accelerator_Support/MLDataDevices' }]
                    },
                    {
                        text: 'Building Blocks', collapsed: false, items: [
                            { text: 'LuxCore', link: '/api/Building_Blocks/LuxCore' },
                            { text: 'LuxLib', link: '/api/Building_Blocks/LuxLib' },
                            { text: 'WeightInitializers', link: '/api/Building_Blocks/WeightInitializers' },
                            { text: 'NNlib', link: 'https://fluxml.ai/NNlib.jl/dev/' },
                            { text: 'Activation Functions', link: 'https://fluxml.ai/NNlib.jl/dev/reference/#Activation-Functions' }]
                    },
                    {
                        text: 'Domain Specific Modeling', collapsed: false, items: [
                            { text: 'Boltz', link: '/api/Domain_Specific_Modeling/Boltz' },
                            { text: 'Computer Vision', link: '/api/Domain_Specific_Modeling/Boltz_Vision' },
                            { text: 'Boltz Layers', link: '/api/Domain_Specific_Modeling/Boltz_Layers' }]
                    },
                    {
                        text: 'Testing Functionality', collapsed: false, items: [
                            { text: 'LuxTestUtils', link: '/api/Testing_Functionality/LuxTestUtils' }]
                    }]
            }
        },
        editLink: {
            pattern: 'https://github.com/LuxDL/Lux.jl/edit/main/docs/src/:path',
            text: 'Edit this page on GitHub'
        },
        socialLinks: [
            { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' },
            { icon: 'twitter', link: 'https://twitter.com/avikpal1410' },
            { icon: 'slack', link: 'https://julialang.org/slack/' }
        ],
        footer: {
            message: 'Made with <a href="https://documenter.juliadocs.org/stable/" target="_blank"><strong>Documenter.jl</strong></a>, <a href="https://vitepress.dev" target="_blank"><strong>VitePress</strong></a> and <a href="https://luxdl.github.io/DocumenterVitepress.jl/stable" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>Released under the MIT License. Powered by the <a href="https://www.julialang.org">Julia Programming Language</a>.<br>',
            copyright: `© Copyright ${new Date().getUTCFullYear()} Avik Pal.`
        },
        lastUpdated: {
            text: 'Updated at',
            formatOptions: {
                dateStyle: 'full',
                timeStyle: 'medium'
            }
        },
    }
})

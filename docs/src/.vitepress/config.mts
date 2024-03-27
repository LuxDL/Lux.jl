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
    themeConfig: {
        outline: 'deep',
        // https://vitepress.dev/reference/default-theme-config
        logo: {
            'light': '/assets/lux-logo.svg',
            'dark': '/assets/lux-logo-dark.svg'
        },
        search: {
            provider: 'local',
            options: {
                detailedView: true
            }
        },
        nav: [
            { text: 'Home', link: '/' },
            { text: 'Getting Started', link: '/introduction/index' },
            { text: 'Ecosystem', link: '/ecosystem' },
            { text: 'Tutorials', link: '/tutorials/index' },
            { text: 'Manual', link: '/manual/interface' },
            {
                text: 'API', items: [
                    {
                        text: 'Lux', items: [
                            { text: 'Built-In Layers', link: '/api/Lux/layers' },
                            { text: 'Utilities', link: '/api/Lux/utilities' },
                            { text: 'Experimental', link: '/api/Lux/contrib' },
                            { text: 'InterOp', link: '/api/Lux/switching_frameworks' }
                        ]
                    },
                    {
                        text: 'Accelerator Support', items: [
                            { text: 'LuxAMDGPU', link: '/api/Accelerator_Support/LuxAMDGPU' },
                            { text: 'LuxCUDA', link: '/api/Accelerator_Support/LuxCUDA' },
                            { text: 'LuxDeviceUtils', link: '/api/Accelerator_Support/LuxDeviceUtils' }
                        ]
                    },
                    {
                        text: 'Building Blocks', items: [
                            { text: 'LuxCore', link: '/api/Building_Blocks/LuxCore' },
                            { text: 'LuxLib', link: '/api/Building_Blocks/LuxLib' },
                            { text: 'WeightInitializers', link: '/api/Building_Blocks/WeightInitializers' }
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
                    { text: 'Stable', link: 'https://lux.csail.mit.edu/stable/' },
                    { text: 'Dev', link: 'https://lux.csail.mit.edu/dev/' }
                ]
            }
        ],
        sidebar: {
            "/introduction/": {
                text: 'Getting Started', collapsed: false, items: [
                    { text: 'Introduction', link: '/introduction/index' },
                    { text: 'Overview', link: '/introduction/overview' },
                    { text: 'Resources', link: '/introduction/resources' },
                    { text: 'Citation', link: '/introduction/citation' }]
            },
            "/tutorials/": {
                text: 'Tutorials', collapsed: false, items: [
                    { text: 'Overview', link: '/tutorials/index' },
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
                            { text: 'Training a Neural ODE to Model Gravitational Waveforms', link: '/tutorials/advanced/1_GravitationalWaveForm' }]
                    }]
            },
            "/manual/": {
                text: 'Manual', collapsed: false, items: [
                    { text: 'Lux Interface', link: '/manual/interface' },
                    { text: 'Debugging Lux Models', link: '/manual/debugging' },
                    { text: 'Dispatching on Custom Input Types', link: '/manual/dispatch_custom_input' },
                    { text: 'Freezing Model Parameters', link: '/manual/freezing_model_parameters' },
                    { text: 'GPU Management', link: '/manual/gpu_management' },
                    { text: 'Migrating from Flux to Lux', link: '/manual/migrate_from_flux' },
                    { text: 'Initializing Weights', link: '/manual/weight_initializers' }]
            },
            "/api/": {
                text: 'API Reference', collapsed: false, items: [
                    {
                        text: 'Lux', collapsed: false, items: [
                            { text: 'Built-In Layers', link: '/api/Lux/layers' },
                            { text: 'Utilities', link: '/api/Lux/utilities' },
                            { text: 'Experimental Features', link: '/api/Lux/contrib' },
                            { text: 'Switching between Deep Learning Frameworks', link: '/api/Lux/switching_frameworks' }]
                    },
                    {
                        text: 'Accelerator Support', collapsed: false, items: [
                            { text: 'LuxAMDGPU', link: '/api/Accelerator_Support/LuxAMDGPU' },
                            { text: 'LuxCUDA', link: '/api/Accelerator_Support/LuxCUDA' },
                            { text: 'LuxDeviceUtils', link: '/api/Accelerator_Support/LuxDeviceUtils' }]
                    },
                    {
                        text: 'Building Blocks', collapsed: false, items: [
                            { text: 'LuxCore', link: '/api/Building_Blocks/LuxCore' },
                            { text: 'LuxLib', link: '/api/Building_Blocks/LuxLib' },
                            { text: 'WeightInitializers', link: '/api/Building_Blocks/WeightInitializers' }]
                    },
                    {
                        text: 'Domain Specific Modeling', collapsed: false, items: [
                            { text: 'Boltz', link: '/api/Domain_Specific_Modeling/Boltz' }]
                    },
                    {
                        text: 'Testing Functionality', collapsed: false, items: [
                            { text: 'LuxTestUtils', link: '/api/Testing_Functionality/LuxTestUtils' }]
                    }]
            }
        }, // TODO: Once https://github.com/LuxDL/DocumenterVitepress.jl/issues/48 is fixed we can use the default sidebar --- 'REPLACE_ME_DOCUMENTER_VITEPRESS'
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
            ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: '/assets/apple-touch-icon.png' }],
            ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/assets/favicon-32x32.png' }],
            ['link', { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/assets/favicon-16x16.png' }],
            ['link', { rel: 'manifest', href: '/assets/site.webmanifest' }],
        ],
        lastUpdated: {
            text: 'Updated at',
            formatOptions: {
                dateStyle: 'full',
                timeStyle: 'medium'
            }
        },
    }
})
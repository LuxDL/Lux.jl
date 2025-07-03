import { V as VPTeamPageTitle, a as VPTeamPageSection, b as VPTeamMembers, c as VPTeamPage } from "./chunks/theme.BTVu2Ezd.js";
import { c as createElementBlock, J as createVNode, w as withCtx, p as unref, o as openBlock, a as createTextVNode } from "./chunks/framework.C7s4kFEZ.js";
const __pageData = JSON.parse('{"title":"","description":"","frontmatter":{"layout":"page"},"headers":[],"relativePath":"tutorials/index.md","filePath":"tutorials/index.md","lastUpdated":null}');
const __default__ = { name: "tutorials/index.md" };
const _sfc_main = /* @__PURE__ */ Object.assign(__default__, {
  setup(__props) {
    const githubSvg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 512"><path d="M392.8 1.2c-17-4.9-34.7 5-39.6 22l-128 448c-4.9 17 5 34.7 22 39.6s34.7-5 39.6-22l128-448c4.9-17-5-34.7-22-39.6zm80.6 120.1c-12.5 12.5-12.5 32.8 0 45.3L562.7 256l-89.4 89.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l112-112c12.5-12.5 12.5-32.8 0-45.3l-112-112c-12.5-12.5-32.8-12.5-45.3 0zm-306.7 0c-12.5-12.5-32.8-12.5-45.3 0l-112 112c-12.5 12.5-12.5 32.8 0 45.3l112 112c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L77.3 256l89.4-89.4c12.5-12.5 12.5-32.8 0-45.3z"/></svg>';
    const beginners = [
      {
        avatar: "https://github.com/LuxDL.png",
        name: "Julia & Lux for the Uninitiated",
        desc: "A tutorial on how to get started with Julia and Lux for those who have never used Julia before.",
        links: [
          {
            icon: {
              svg: githubSvg
            },
            link: "beginner/1_Basics"
          }
        ]
      },
      {
        avatar: "https://github.com/LuxDL.png",
        name: "Fitting a Polynomial using MLP",
        desc: "Learn the Basics of Lux by fitting a Multi-Layer Perceptron to a Polynomial.",
        links: [
          {
            icon: {
              svg: githubSvg
            },
            link: "beginner/2_PolynomialFitting"
          }
        ]
      },
      {
        avatar: "https://github.com/LuxDL.png",
        name: "Training a Simple LSTM",
        desc: "Learn the API for defining Recurrent Models in Lux.",
        links: [
          {
            icon: {
              svg: githubSvg
            },
            link: "beginner/3_SimpleRNN"
          }
        ]
      },
      {
        avatar: "https://github.com/PumasAI.png",
        name: "Use SimpleChains.jl as a Backend",
        desc: "Learn how to train small neural networks really fast",
        links: [
          {
            icon: {
              svg: githubSvg
            },
            link: "beginner/4_SimpleChains"
          }
        ]
      }
    ];
    const intermediate = [
      {
        avatar: "https://github.com/SciML.png",
        name: "MNIST Classification using Neural ODE",
        desc: "Train a Neural ODE to classify MNIST Images.",
        links: [
          {
            icon: {
              svg: githubSvg
            },
            link: "intermediate/1_NeuralODE"
          }
        ]
      },
      {
        avatar: "https://github.com/TuringLang.png",
        name: "Bayesian Neural Networks",
        desc: "Figure out how to use Probabilistic Programming Frameworks like Turing with Lux.",
        links: [
          {
            icon: {
              svg: githubSvg
            },
            link: "intermediate/2_BayesianNN"
          }
        ]
      },
      {
        avatar: "https://github.com/LuxDL.png",
        name: "Training a HyperNetwork",
        desc: "In this tutorial we will train a hypernetwork to work on multiple datasets by predicting neural network parameters.",
        orgLink: "intermediate/3_HyperNet",
        links: [
          {
            icon: {
              svg: githubSvg
            },
            link: "intermediate/3_HyperNet"
          }
        ]
      }
    ];
    const advanced = [
      {
        avatar: "https://github.com/SciML.png",
        name: "Neural ODE to Model Gravitational Waveforms",
        desc: "Training a Neural ODE to fit simulated data of gravitational waveforms.",
        links: [
          {
            icon: {
              svg: githubSvg
            },
            link: "advanced/1_GravitationalWaveForm"
          }
        ]
      }
    ];
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", null, [
        createVNode(unref(VPTeamPage), null, {
          default: withCtx(() => [
            createVNode(unref(VPTeamPageTitle), null, {
              title: withCtx(() => [
                createTextVNode("Tutorials")
              ]),
              _: 1
            }),
            createVNode(unref(VPTeamPageSection), null, {
              title: withCtx(() => [
                createTextVNode("Beginners Tutorials")
              ]),
              members: withCtx(() => [
                createVNode(unref(VPTeamMembers), {
                  size: "small",
                  members: beginners
                })
              ]),
              _: 1
            }),
            createVNode(unref(VPTeamPageSection), null, {
              title: withCtx(() => [
                createTextVNode("Intermediate Tutorials")
              ]),
              members: withCtx(() => [
                createVNode(unref(VPTeamMembers), {
                  size: "small",
                  members: intermediate
                })
              ]),
              _: 1
            }),
            createVNode(unref(VPTeamPageSection), null, {
              title: withCtx(() => [
                createTextVNode("Advanced Tutorials")
              ]),
              members: withCtx(() => [
                createVNode(unref(VPTeamMembers), {
                  size: "small",
                  members: advanced
                })
              ]),
              _: 1
            })
          ]),
          _: 1
        })
      ]);
    };
  }
});
export {
  __pageData,
  _sfc_main as default
};

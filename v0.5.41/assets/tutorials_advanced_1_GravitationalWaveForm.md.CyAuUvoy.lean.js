import { _ as _export_sfc, c as createElementBlock, l as createBaseVNode, a as createTextVNode, a3 as createStaticVNode, o as openBlock } from "./chunks/framework.DVlmWVQO.js";
const __pageData = JSON.parse('{"title":"Training a Neural ODE to Model Gravitational Waveforms","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/advanced/1_GravitationalWaveForm.md","filePath":"tutorials/advanced/1_GravitationalWaveForm.md","lastUpdated":null}');
const _sfc_main = { name: "tutorials/advanced/1_GravitationalWaveForm.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode("", 7);
const _hoisted_8 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_9 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.339ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "10.819ex",
  height: "1.658ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -583 4782.1 733",
  "aria-hidden": "true"
};
const _hoisted_10 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_11 = [
  _hoisted_10
];
const _hoisted_12 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "r"),
    /* @__PURE__ */ createBaseVNode("mo", null, "="),
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "r"),
      /* @__PURE__ */ createBaseVNode("mn", null, "1")
    ]),
    /* @__PURE__ */ createBaseVNode("mo", null, "−"),
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "r"),
      /* @__PURE__ */ createBaseVNode("mn", null, "2")
    ])
  ])
], -1);
const _hoisted_13 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_14 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.339ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "2.008ex",
  height: "1.339ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -442 887.6 592",
  "aria-hidden": "true"
};
const _hoisted_15 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_16 = [
  _hoisted_15
];
const _hoisted_17 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "r"),
      /* @__PURE__ */ createBaseVNode("mn", null, "1")
    ])
  ])
], -1);
const _hoisted_18 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_19 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.339ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "2.008ex",
  height: "1.339ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -442 887.6 592",
  "aria-hidden": "true"
};
const _hoisted_20 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_21 = [
  _hoisted_20
];
const _hoisted_22 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "r"),
      /* @__PURE__ */ createBaseVNode("mn", null, "2")
    ])
  ])
], -1);
const _hoisted_23 = /* @__PURE__ */ createStaticVNode("", 2);
const _hoisted_25 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_26 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.566ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "24.527ex",
  height: "2.262ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -750 10840.9 1000",
  "aria-hidden": "true"
};
const _hoisted_27 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_28 = [
  _hoisted_27
];
const _hoisted_29 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mi", null, "χ"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mi", null, "t"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")"),
    /* @__PURE__ */ createBaseVNode("mo", null, ","),
    /* @__PURE__ */ createBaseVNode("mi", null, "ϕ"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mi", null, "t"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "↦"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mi", null, "x"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mi", null, "t"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")"),
    /* @__PURE__ */ createBaseVNode("mo", null, ","),
    /* @__PURE__ */ createBaseVNode("mi", null, "y"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mi", null, "t"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")")
  ])
], -1);
const _hoisted_30 = /* @__PURE__ */ createStaticVNode("", 13);
const _hoisted_43 = {
  class: "MathJax",
  jax: "SVG",
  display: "true",
  style: { "direction": "ltr", "display": "block", "text-align": "center", "margin": "1em 0", "position": "relative" }
};
const _hoisted_44 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.566ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "8.117ex",
  height: "2.262ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -750 3587.6 1000",
  "aria-hidden": "true"
};
const _hoisted_45 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_46 = [
  _hoisted_45
];
const _hoisted_47 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "block",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "overflow": "hidden", "width": "100%" }
}, [
  /* @__PURE__ */ createBaseVNode("math", {
    xmlns: "http://www.w3.org/1998/Math/MathML",
    display: "block"
  }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "u"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "["),
    /* @__PURE__ */ createBaseVNode("mn", null, "1"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "]"),
    /* @__PURE__ */ createBaseVNode("mo", null, "="),
    /* @__PURE__ */ createBaseVNode("mi", null, "χ")
  ])
], -1);
const _hoisted_48 = {
  class: "MathJax",
  jax: "SVG",
  display: "true",
  style: { "direction": "ltr", "display": "block", "text-align": "center", "margin": "1em 0", "position": "relative" }
};
const _hoisted_49 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.566ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "8.049ex",
  height: "2.262ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -750 3557.6 1000",
  "aria-hidden": "true"
};
const _hoisted_50 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_51 = [
  _hoisted_50
];
const _hoisted_52 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "block",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "overflow": "hidden", "width": "100%" }
}, [
  /* @__PURE__ */ createBaseVNode("math", {
    xmlns: "http://www.w3.org/1998/Math/MathML",
    display: "block"
  }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "u"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "["),
    /* @__PURE__ */ createBaseVNode("mn", null, "2"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "]"),
    /* @__PURE__ */ createBaseVNode("mo", null, "="),
    /* @__PURE__ */ createBaseVNode("mi", null, "ϕ")
  ])
], -1);
const _hoisted_53 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_54 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.439ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.138ex",
  height: "1.439ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -442 503 636",
  "aria-hidden": "true"
};
const _hoisted_55 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D45D",
        d: "M23 287Q24 290 25 295T30 317T40 348T55 381T75 411T101 433T134 442Q209 442 230 378L240 387Q302 442 358 442Q423 442 460 395T497 281Q497 173 421 82T249 -10Q227 -10 210 -4Q199 1 187 11T168 28L161 36Q160 35 139 -51T118 -138Q118 -144 126 -145T163 -148H188Q194 -155 194 -157T191 -175Q188 -187 185 -190T172 -194Q170 -194 161 -194T127 -193T65 -192Q-5 -192 -24 -194H-32Q-39 -187 -39 -183Q-37 -156 -26 -148H-6Q28 -147 33 -136Q36 -130 94 103T155 350Q156 355 156 364Q156 405 131 405Q109 405 94 377T71 316T59 280Q57 278 43 278H29Q23 284 23 287ZM178 102Q200 26 252 26Q282 26 310 49T356 107Q374 141 392 215T411 325V331Q411 405 350 405Q339 405 328 402T306 393T286 380T269 365T254 350T243 336T235 326L232 322Q232 321 229 308T218 264T204 212Q178 106 178 102Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_56 = [
  _hoisted_55
];
const _hoisted_57 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "p")
  ])
], -1);
const _hoisted_58 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_59 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "0" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "2.378ex",
  height: "1.545ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -683 1051 683",
  "aria-hidden": "true"
};
const _hoisted_60 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D440",
        d: "M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_61 = [
  _hoisted_60
];
const _hoisted_62 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "M")
  ])
], -1);
const _hoisted_63 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_64 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.025ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.054ex",
  height: "1.025ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -442 466 453",
  "aria-hidden": "true"
};
const _hoisted_65 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D452",
        d: "M39 168Q39 225 58 272T107 350T174 402T244 433T307 442H310Q355 442 388 420T421 355Q421 265 310 237Q261 224 176 223Q139 223 138 221Q138 219 132 186T125 128Q125 81 146 54T209 26T302 45T394 111Q403 121 406 121Q410 121 419 112T429 98T420 82T390 55T344 24T281 -1T205 -11Q126 -11 83 42T39 168ZM373 353Q367 405 305 405Q272 405 244 391T199 357T170 316T154 280T149 261Q149 260 169 260Q282 260 327 284T373 353Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_66 = [
  _hoisted_65
];
const _hoisted_67 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "e")
  ])
], -1);
const _hoisted_68 = /* @__PURE__ */ createStaticVNode("", 14);
const _hoisted_82 = {
  class: "MathJax",
  jax: "SVG",
  display: "true",
  style: { "direction": "ltr", "display": "block", "text-align": "center", "margin": "1em 0", "position": "relative" }
};
const _hoisted_83 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.566ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "8.117ex",
  height: "2.262ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -750 3587.6 1000",
  "aria-hidden": "true"
};
const _hoisted_84 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_85 = [
  _hoisted_84
];
const _hoisted_86 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "block",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "overflow": "hidden", "width": "100%" }
}, [
  /* @__PURE__ */ createBaseVNode("math", {
    xmlns: "http://www.w3.org/1998/Math/MathML",
    display: "block"
  }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "u"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "["),
    /* @__PURE__ */ createBaseVNode("mn", null, "1"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "]"),
    /* @__PURE__ */ createBaseVNode("mo", null, "="),
    /* @__PURE__ */ createBaseVNode("mi", null, "χ")
  ])
], -1);
const _hoisted_87 = {
  class: "MathJax",
  jax: "SVG",
  display: "true",
  style: { "direction": "ltr", "display": "block", "text-align": "center", "margin": "1em 0", "position": "relative" }
};
const _hoisted_88 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.566ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "8.049ex",
  height: "2.262ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -750 3557.6 1000",
  "aria-hidden": "true"
};
const _hoisted_89 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_90 = [
  _hoisted_89
];
const _hoisted_91 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "block",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "overflow": "hidden", "width": "100%" }
}, [
  /* @__PURE__ */ createBaseVNode("math", {
    xmlns: "http://www.w3.org/1998/Math/MathML",
    display: "block"
  }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "u"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "["),
    /* @__PURE__ */ createBaseVNode("mn", null, "2"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "]"),
    /* @__PURE__ */ createBaseVNode("mo", null, "="),
    /* @__PURE__ */ createBaseVNode("mi", null, "ϕ")
  ])
], -1);
const _hoisted_92 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_93 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.439ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.138ex",
  height: "1.439ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -442 503 636",
  "aria-hidden": "true"
};
const _hoisted_94 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D45D",
        d: "M23 287Q24 290 25 295T30 317T40 348T55 381T75 411T101 433T134 442Q209 442 230 378L240 387Q302 442 358 442Q423 442 460 395T497 281Q497 173 421 82T249 -10Q227 -10 210 -4Q199 1 187 11T168 28L161 36Q160 35 139 -51T118 -138Q118 -144 126 -145T163 -148H188Q194 -155 194 -157T191 -175Q188 -187 185 -190T172 -194Q170 -194 161 -194T127 -193T65 -192Q-5 -192 -24 -194H-32Q-39 -187 -39 -183Q-37 -156 -26 -148H-6Q28 -147 33 -136Q36 -130 94 103T155 350Q156 355 156 364Q156 405 131 405Q109 405 94 377T71 316T59 280Q57 278 43 278H29Q23 284 23 287ZM178 102Q200 26 252 26Q282 26 310 49T356 107Q374 141 392 215T411 325V331Q411 405 350 405Q339 405 328 402T306 393T286 380T269 365T254 350T243 336T235 326L232 322Q232 321 229 308T218 264T204 212Q178 106 178 102Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_95 = [
  _hoisted_94
];
const _hoisted_96 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "p")
  ])
], -1);
const _hoisted_97 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_98 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "0" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "2.378ex",
  height: "1.545ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -683 1051 683",
  "aria-hidden": "true"
};
const _hoisted_99 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D440",
        d: "M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_100 = [
  _hoisted_99
];
const _hoisted_101 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "M")
  ])
], -1);
const _hoisted_102 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_103 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.025ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "1.054ex",
  height: "1.025ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -442 466 453",
  "aria-hidden": "true"
};
const _hoisted_104 = /* @__PURE__ */ createBaseVNode("g", {
  stroke: "currentColor",
  fill: "currentColor",
  "stroke-width": "0",
  transform: "scale(1,-1)"
}, [
  /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "math" }, [
    /* @__PURE__ */ createBaseVNode("g", { "data-mml-node": "mi" }, [
      /* @__PURE__ */ createBaseVNode("path", {
        "data-c": "1D452",
        d: "M39 168Q39 225 58 272T107 350T174 402T244 433T307 442H310Q355 442 388 420T421 355Q421 265 310 237Q261 224 176 223Q139 223 138 221Q138 219 132 186T125 128Q125 81 146 54T209 26T302 45T394 111Q403 121 406 121Q410 121 419 112T429 98T420 82T390 55T344 24T281 -1T205 -11Q126 -11 83 42T39 168ZM373 353Q367 405 305 405Q272 405 244 391T199 357T170 316T154 280T149 261Q149 260 169 260Q282 260 327 284T373 353Z",
        style: { "stroke-width": "3" }
      })
    ])
  ])
], -1);
const _hoisted_105 = [
  _hoisted_104
];
const _hoisted_106 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "e")
  ])
], -1);
const _hoisted_107 = /* @__PURE__ */ createStaticVNode("", 31);
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, [
    _hoisted_1,
    createBaseVNode("p", null, [
      createTextVNode("We need a very crude 2-body path. Assume the 1-body motion is a newtonian 2-body position vector "),
      createBaseVNode("mjx-container", _hoisted_8, [
        (openBlock(), createElementBlock("svg", _hoisted_9, _hoisted_11)),
        _hoisted_12
      ]),
      createTextVNode(" and use Newtonian formulas to get "),
      createBaseVNode("mjx-container", _hoisted_13, [
        (openBlock(), createElementBlock("svg", _hoisted_14, _hoisted_16)),
        _hoisted_17
      ]),
      createTextVNode(", "),
      createBaseVNode("mjx-container", _hoisted_18, [
        (openBlock(), createElementBlock("svg", _hoisted_19, _hoisted_21)),
        _hoisted_22
      ]),
      createTextVNode(" (e.g. Theoretical Mechanics of Particles and Continua 4.3)")
    ]),
    _hoisted_23,
    createBaseVNode("p", null, [
      createTextVNode("Next we define a function to perform the change of variables: "),
      createBaseVNode("mjx-container", _hoisted_25, [
        (openBlock(), createElementBlock("svg", _hoisted_26, _hoisted_28)),
        _hoisted_29
      ])
    ]),
    _hoisted_30,
    createBaseVNode("mjx-container", _hoisted_43, [
      (openBlock(), createElementBlock("svg", _hoisted_44, _hoisted_46)),
      _hoisted_47
    ]),
    createBaseVNode("mjx-container", _hoisted_48, [
      (openBlock(), createElementBlock("svg", _hoisted_49, _hoisted_51)),
      _hoisted_52
    ]),
    createBaseVNode("p", null, [
      createTextVNode("where, "),
      createBaseVNode("mjx-container", _hoisted_53, [
        (openBlock(), createElementBlock("svg", _hoisted_54, _hoisted_56)),
        _hoisted_57
      ]),
      createTextVNode(", "),
      createBaseVNode("mjx-container", _hoisted_58, [
        (openBlock(), createElementBlock("svg", _hoisted_59, _hoisted_61)),
        _hoisted_62
      ]),
      createTextVNode(", and "),
      createBaseVNode("mjx-container", _hoisted_63, [
        (openBlock(), createElementBlock("svg", _hoisted_64, _hoisted_66)),
        _hoisted_67
      ]),
      createTextVNode(" are constants")
    ]),
    _hoisted_68,
    createBaseVNode("mjx-container", _hoisted_82, [
      (openBlock(), createElementBlock("svg", _hoisted_83, _hoisted_85)),
      _hoisted_86
    ]),
    createBaseVNode("mjx-container", _hoisted_87, [
      (openBlock(), createElementBlock("svg", _hoisted_88, _hoisted_90)),
      _hoisted_91
    ]),
    createBaseVNode("p", null, [
      createTextVNode("where, "),
      createBaseVNode("mjx-container", _hoisted_92, [
        (openBlock(), createElementBlock("svg", _hoisted_93, _hoisted_95)),
        _hoisted_96
      ]),
      createTextVNode(", "),
      createBaseVNode("mjx-container", _hoisted_97, [
        (openBlock(), createElementBlock("svg", _hoisted_98, _hoisted_100)),
        _hoisted_101
      ]),
      createTextVNode(", and "),
      createBaseVNode("mjx-container", _hoisted_102, [
        (openBlock(), createElementBlock("svg", _hoisted_103, _hoisted_105)),
        _hoisted_106
      ]),
      createTextVNode(" are constants")
    ]),
    _hoisted_107
  ]);
}
const _1_GravitationalWaveForm = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  _1_GravitationalWaveForm as default
};

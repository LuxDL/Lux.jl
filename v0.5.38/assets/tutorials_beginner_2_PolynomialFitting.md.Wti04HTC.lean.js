import { _ as _export_sfc, c as createElementBlock, l as createBaseVNode, a as createTextVNode, a4 as createStaticVNode, o as openBlock } from "./chunks/framework.Cu_l04oX.js";
const __pageData = JSON.parse('{"title":"Fitting a Polynomial using MLP","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/2_PolynomialFitting.md","filePath":"tutorials/beginner/2_PolynomialFitting.md","lastUpdated":null}');
const _sfc_main = { name: "tutorials/beginner/2_PolynomialFitting.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode("", 5);
const _hoisted_6 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_7 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.464ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "11.599ex",
  height: "2.351ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -833.9 5126.6 1038.9",
  "aria-hidden": "true"
};
const _hoisted_8 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_9 = [
  _hoisted_8
];
const _hoisted_10 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "y"),
    /* @__PURE__ */ createBaseVNode("mo", null, "="),
    /* @__PURE__ */ createBaseVNode("msup", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "x"),
      /* @__PURE__ */ createBaseVNode("mn", null, "2")
    ]),
    /* @__PURE__ */ createBaseVNode("mo", null, "−"),
    /* @__PURE__ */ createBaseVNode("mn", null, "2"),
    /* @__PURE__ */ createBaseVNode("mi", null, "x")
  ])
], -1);
const _hoisted_11 = /* @__PURE__ */ createStaticVNode("", 38);
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, [
    _hoisted_1,
    createBaseVNode("p", null, [
      createTextVNode("Generate 128 datapoints from the polynomial "),
      createBaseVNode("mjx-container", _hoisted_6, [
        (openBlock(), createElementBlock("svg", _hoisted_7, _hoisted_9)),
        _hoisted_10
      ]),
      createTextVNode(".")
    ]),
    _hoisted_11
  ]);
}
const _2_PolynomialFitting = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  _2_PolynomialFitting as default
};

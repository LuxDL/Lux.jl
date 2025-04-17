import{_ as r,C as T,c as l,o as d,j as t,a,al as s,G as i,w as n}from"./chunks/framework.BCN3FD2k.js";const j=JSON.parse('{"title":"Automatic Differentiation Helpers","description":"","frontmatter":{},"headers":[],"relativePath":"api/Lux/autodiff.md","filePath":"api/Lux/autodiff.md","lastUpdated":null}'),Q={name:"api/Lux/autodiff.md"},p={class:"jldocstring custom-block"},c={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},u={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-1.469ex"},xmlns:"http://www.w3.org/2000/svg",width:"6.812ex",height:"4.07ex",role:"img",focusable:"false",viewBox:"0 -1149.5 3010.7 1799","aria-hidden":"true"},m={class:"jldocstring custom-block"},h={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},g={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-1.469ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.126ex",height:"4.536ex",role:"img",focusable:"false",viewBox:"0 -1355.3 3591.5 2004.8","aria-hidden":"true"},f={class:"jldocstring custom-block"};function b(x,e,k,y,_,w){const o=T("Badge");return d(),l("div",null,[e[22]||(e[22]=t("h1",{id:"autodiff-lux-helpers",tabindex:"-1"},[a("Automatic Differentiation Helpers "),t("a",{class:"header-anchor",href:"#autodiff-lux-helpers","aria-label":'Permalink to "Automatic Differentiation Helpers {#autodiff-lux-helpers}"'},"​")],-1)),e[23]||(e[23]=t("h2",{id:"JVP-and-VJP-Wrappers",tabindex:"-1"},[a("JVP & VJP Wrappers "),t("a",{class:"header-anchor",href:"#JVP-and-VJP-Wrappers","aria-label":'Permalink to "JVP &amp; VJP Wrappers {#JVP-and-VJP-Wrappers}"'},"​")],-1)),t("details",p,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Lux.jacobian_vector_product",href:"#Lux.jacobian_vector_product"},[t("span",{class:"jlbinding"},"Lux.jacobian_vector_product")],-1)),e[1]||(e[1]=a()),i(o,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[7]||(e[7]=s("",1)),t("p",null,[e[4]||(e[4]=a("Compute the Jacobian-Vector Product ")),t("mjx-container",c,[(d(),l("svg",u,e[2]||(e[2]=[s("",1)]))),e[3]||(e[3]=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mrow",{"data-mjx-texclass":"INNER"},[t("mo",{"data-mjx-texclass":"OPEN"},"("),t("mfrac",null,[t("mrow",null,[t("mi",null,"∂"),t("mi",null,"f")]),t("mrow",null,[t("mi",null,"∂"),t("mi",null,"x")])]),t("mo",{"data-mjx-texclass":"CLOSE"},")")]),t("mi",null,"u")])],-1))]),e[5]||(e[5]=a(". This is a wrapper around AD backends but allows us to compute gradients of jacobian-vector products efficiently using mixed-mode AD."))]),e[8]||(e[8]=s("",7)),i(o,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[6]||(e[6]=[t("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/src/autodiff/api.jl#L43",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",m,[t("summary",null,[e[9]||(e[9]=t("a",{id:"Lux.vector_jacobian_product",href:"#Lux.vector_jacobian_product"},[t("span",{class:"jlbinding"},"Lux.vector_jacobian_product")],-1)),e[10]||(e[10]=a()),i(o,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[16]||(e[16]=s("",1)),t("p",null,[e[13]||(e[13]=a("Compute the Vector-Jacobian Product ")),t("mjx-container",h,[(d(),l("svg",g,e[11]||(e[11]=[s("",1)]))),e[12]||(e[12]=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msup",null,[t("mrow",{"data-mjx-texclass":"INNER"},[t("mo",{"data-mjx-texclass":"OPEN"},"("),t("mfrac",null,[t("mrow",null,[t("mi",null,"∂"),t("mi",null,"f")]),t("mrow",null,[t("mi",null,"∂"),t("mi",null,"x")])]),t("mo",{"data-mjx-texclass":"CLOSE"},")")]),t("mi",null,"T")]),t("mi",null,"u")])],-1))]),e[14]||(e[14]=a(". This is a wrapper around AD backends but allows us to compute gradients of vector-jacobian products efficiently using mixed-mode AD."))]),e[17]||(e[17]=s("",7)),i(o,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[15]||(e[15]=[t("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/src/autodiff/api.jl#L1",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e[24]||(e[24]=t("h2",{id:"Batched-AD",tabindex:"-1"},[a("Batched AD "),t("a",{class:"header-anchor",href:"#Batched-AD","aria-label":'Permalink to "Batched AD {#Batched-AD}"'},"​")],-1)),t("details",f,[t("summary",null,[e[18]||(e[18]=t("a",{id:"Lux.batched_jacobian",href:"#Lux.batched_jacobian"},[t("span",{class:"jlbinding"},"Lux.batched_jacobian")],-1)),e[19]||(e[19]=a()),i(o,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[21]||(e[21]=s("",10)),i(o,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[20]||(e[20]=[t("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/src/autodiff/api.jl#L81-L114",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e[25]||(e[25]=t("h2",{id:"Nested-2nd-Order-AD",tabindex:"-1"},[a("Nested 2nd Order AD "),t("a",{class:"header-anchor",href:"#Nested-2nd-Order-AD","aria-label":'Permalink to "Nested 2nd Order AD {#Nested-2nd-Order-AD}"'},"​")],-1)),e[26]||(e[26]=t("p",null,[a("Consult the "),t("a",{href:"/dev/manual/nested_autodiff#nested_autodiff"},"manual page on Nested AD"),a(" for information on nested automatic differentiation.")],-1))])}const v=r(Q,[["render",b]]);export{j as __pageData,v as default};

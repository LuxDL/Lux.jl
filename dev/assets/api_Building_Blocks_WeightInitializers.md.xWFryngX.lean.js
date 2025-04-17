import{_ as k,C as p,c as h,o as r,al as l,j as s,G as t,a as n,w as e}from"./chunks/framework.BCN3FD2k.js";const Y=JSON.parse('{"title":"WeightInitializers","description":"","frontmatter":{},"headers":[],"relativePath":"api/Building_Blocks/WeightInitializers.md","filePath":"api/Building_Blocks/WeightInitializers.md","lastUpdated":null}'),d={name:"api/Building_Blocks/WeightInitializers.md"},o={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},E={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},y={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"6.612ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 2922.7 1000","aria-hidden":"true"},u={class:"jldocstring custom-block"},c={class:"jldocstring custom-block"},F={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"},C={class:"jldocstring custom-block"},A={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"},f={class:"jldocstring custom-block"},_={class:"jldocstring custom-block"},T={class:"jldocstring custom-block"},z={class:"jldocstring custom-block"},D={class:"jldocstring custom-block"},j={class:"jldocstring custom-block"},x={class:"jldocstring custom-block"},v={class:"jldocstring custom-block"},I={class:"jldocstring custom-block"},B={class:"jldocstring custom-block"},L={class:"jldocstring custom-block"},W={class:"jldocstring custom-block"},R={class:"jldocstring custom-block"},S={class:"jldocstring custom-block"},P={class:"jldocstring custom-block"},N={class:"jldocstring custom-block"},w={class:"jldocstring custom-block"},V={class:"jldocstring custom-block"},Q={class:"jldocstring custom-block"},G={class:"jldocstring custom-block"},U={class:"jldocstring custom-block"},O={class:"jldocstring custom-block"},M={class:"jldocstring custom-block"};function q(H,i,X,Z,$,J){const a=p("Badge");return r(),h("div",null,[i[140]||(i[140]=l("",6)),s("details",o,[s("summary",null,[i[0]||(i[0]=s("a",{id:"WeightInitializers.glorot_normal",href:"#WeightInitializers.glorot_normal"},[s("span",{class:"jlbinding"},"WeightInitializers.glorot_normal")],-1)),i[1]||(i[1]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[3]||(i[3]=l("",4)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[2]||(i[2]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L38-L51",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",g,[s("summary",null,[i[4]||(i[4]=s("a",{id:"WeightInitializers.glorot_uniform",href:"#WeightInitializers.glorot_uniform"},[s("span",{class:"jlbinding"},"WeightInitializers.glorot_uniform")],-1)),i[5]||(i[5]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[17]||(i[17]=l("",1)),s("p",null,[i[8]||(i[8]=n("Return an ")),i[9]||(i[9]=s("code",null,"AbstractArray{T}",-1)),i[10]||(i[10]=n(" of the given ")),i[11]||(i[11]=s("code",null,"size",-1)),i[12]||(i[12]=n(" containing random numbers drawn from a uniform distribution on the interval ")),s("mjx-container",E,[(r(),h("svg",y,i[6]||(i[6]=[l("",1)]))),i[7]||(i[7]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mo",{stretchy:"false"},"["),s("mo",null,"−"),s("mi",null,"x"),s("mo",null,","),s("mi",null,"x"),s("mo",{stretchy:"false"},"]")])],-1))]),i[13]||(i[13]=n(", where ")),i[14]||(i[14]=s("code",null,"x = gain * sqrt(6 / (fan_in + fan_out))",-1)),i[15]||(i[15]=n(". This method is described in [1] and also known as Xavier initialization."))]),i[18]||(i[18]=s("p",null,[s("strong",null,"References")],-1)),i[19]||(i[19]=s("p",null,[n('[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." '),s("em",null,"Proceedings of the thirteenth international conference on artificial intelligence and statistics"),n(". 2010.")],-1)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[16]||(i[16]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L13-L27",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",u,[s("summary",null,[i[20]||(i[20]=s("a",{id:"WeightInitializers.identity_init",href:"#WeightInitializers.identity_init"},[s("span",{class:"jlbinding"},"WeightInitializers.identity_init")],-1)),i[21]||(i[21]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[23]||(i[23]=l("",12)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[22]||(i[22]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L250-L311",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",c,[s("summary",null,[i[24]||(i[24]=s("a",{id:"WeightInitializers.kaiming_normal",href:"#WeightInitializers.kaiming_normal"},[s("span",{class:"jlbinding"},"WeightInitializers.kaiming_normal")],-1)),i[25]||(i[25]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[27]||(i[27]=l("",4)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[26]||(i[26]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L84-L96",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",F,[s("summary",null,[i[28]||(i[28]=s("a",{id:"WeightInitializers.kaiming_uniform",href:"#WeightInitializers.kaiming_uniform"},[s("span",{class:"jlbinding"},"WeightInitializers.kaiming_uniform")],-1)),i[29]||(i[29]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[31]||(i[31]=l("",4)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[30]||(i[30]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L61-L73",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",b,[s("summary",null,[i[32]||(i[32]=s("a",{id:"WeightInitializers.sparse_init",href:"#WeightInitializers.sparse_init"},[s("span",{class:"jlbinding"},"WeightInitializers.sparse_init")],-1)),i[33]||(i[33]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[35]||(i[35]=l("",11)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[34]||(i[34]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L173-L218",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",C,[s("summary",null,[i[36]||(i[36]=s("a",{id:"WeightInitializers.truncated_normal",href:"#WeightInitializers.truncated_normal"},[s("span",{class:"jlbinding"},"WeightInitializers.truncated_normal")],-1)),i[37]||(i[37]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[39]||(i[39]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[38]||(i[38]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L106-L113",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",A,[s("summary",null,[i[40]||(i[40]=s("a",{id:"WeightInitializers.orthogonal",href:"#WeightInitializers.orthogonal"},[s("span",{class:"jlbinding"},"WeightInitializers.orthogonal")],-1)),i[41]||(i[41]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[43]||(i[43]=l("",8)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[42]||(i[42]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L132-L157",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),i[141]||(i[141]=s("h3",{id:"Other-Convenience-Functions",tabindex:"-1"},[n("Other Convenience Functions "),s("a",{class:"header-anchor",href:"#Other-Convenience-Functions","aria-label":'Permalink to "Other Convenience Functions {#Other-Convenience-Functions}"'},"​")],-1)),i[142]||(i[142]=s("div",{class:"warning custom-block"},[s("p",{class:"custom-block-title"},"Beware"),s("p",null,"Unlike the other functions these ones don't take a type argument.")],-1)),s("details",m,[s("summary",null,[i[44]||(i[44]=s("a",{id:"WeightInitializers.zeros16",href:"#WeightInitializers.zeros16"},[s("span",{class:"jlbinding"},"WeightInitializers.zeros16")],-1)),i[45]||(i[45]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[47]||(i[47]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[46]||(i[46]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",f,[s("summary",null,[i[48]||(i[48]=s("a",{id:"WeightInitializers.ones16",href:"#WeightInitializers.ones16"},[s("span",{class:"jlbinding"},"WeightInitializers.ones16")],-1)),i[49]||(i[49]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[51]||(i[51]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[50]||(i[50]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",_,[s("summary",null,[i[52]||(i[52]=s("a",{id:"WeightInitializers.rand16",href:"#WeightInitializers.rand16"},[s("span",{class:"jlbinding"},"WeightInitializers.rand16")],-1)),i[53]||(i[53]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[55]||(i[55]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[54]||(i[54]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",T,[s("summary",null,[i[56]||(i[56]=s("a",{id:"WeightInitializers.randn16",href:"#WeightInitializers.randn16"},[s("span",{class:"jlbinding"},"WeightInitializers.randn16")],-1)),i[57]||(i[57]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[59]||(i[59]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[58]||(i[58]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",z,[s("summary",null,[i[60]||(i[60]=s("a",{id:"WeightInitializers.zeros32",href:"#WeightInitializers.zeros32"},[s("span",{class:"jlbinding"},"WeightInitializers.zeros32")],-1)),i[61]||(i[61]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[63]||(i[63]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[62]||(i[62]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",D,[s("summary",null,[i[64]||(i[64]=s("a",{id:"WeightInitializers.ones32",href:"#WeightInitializers.ones32"},[s("span",{class:"jlbinding"},"WeightInitializers.ones32")],-1)),i[65]||(i[65]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[67]||(i[67]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[66]||(i[66]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",j,[s("summary",null,[i[68]||(i[68]=s("a",{id:"WeightInitializers.rand32",href:"#WeightInitializers.rand32"},[s("span",{class:"jlbinding"},"WeightInitializers.rand32")],-1)),i[69]||(i[69]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[71]||(i[71]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[70]||(i[70]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",x,[s("summary",null,[i[72]||(i[72]=s("a",{id:"WeightInitializers.randn32",href:"#WeightInitializers.randn32"},[s("span",{class:"jlbinding"},"WeightInitializers.randn32")],-1)),i[73]||(i[73]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[75]||(i[75]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[74]||(i[74]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",v,[s("summary",null,[i[76]||(i[76]=s("a",{id:"WeightInitializers.zeros64",href:"#WeightInitializers.zeros64"},[s("span",{class:"jlbinding"},"WeightInitializers.zeros64")],-1)),i[77]||(i[77]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[79]||(i[79]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[78]||(i[78]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",I,[s("summary",null,[i[80]||(i[80]=s("a",{id:"WeightInitializers.ones64",href:"#WeightInitializers.ones64"},[s("span",{class:"jlbinding"},"WeightInitializers.ones64")],-1)),i[81]||(i[81]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[83]||(i[83]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[82]||(i[82]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",B,[s("summary",null,[i[84]||(i[84]=s("a",{id:"WeightInitializers.rand64",href:"#WeightInitializers.rand64"},[s("span",{class:"jlbinding"},"WeightInitializers.rand64")],-1)),i[85]||(i[85]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[87]||(i[87]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[86]||(i[86]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",L,[s("summary",null,[i[88]||(i[88]=s("a",{id:"WeightInitializers.randn64",href:"#WeightInitializers.randn64"},[s("span",{class:"jlbinding"},"WeightInitializers.randn64")],-1)),i[89]||(i[89]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[91]||(i[91]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[90]||(i[90]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",W,[s("summary",null,[i[92]||(i[92]=s("a",{id:"WeightInitializers.zerosC16",href:"#WeightInitializers.zerosC16"},[s("span",{class:"jlbinding"},"WeightInitializers.zerosC16")],-1)),i[93]||(i[93]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[95]||(i[95]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[94]||(i[94]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",R,[s("summary",null,[i[96]||(i[96]=s("a",{id:"WeightInitializers.onesC16",href:"#WeightInitializers.onesC16"},[s("span",{class:"jlbinding"},"WeightInitializers.onesC16")],-1)),i[97]||(i[97]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[99]||(i[99]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[98]||(i[98]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",S,[s("summary",null,[i[100]||(i[100]=s("a",{id:"WeightInitializers.randC16",href:"#WeightInitializers.randC16"},[s("span",{class:"jlbinding"},"WeightInitializers.randC16")],-1)),i[101]||(i[101]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[103]||(i[103]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[102]||(i[102]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",P,[s("summary",null,[i[104]||(i[104]=s("a",{id:"WeightInitializers.randnC16",href:"#WeightInitializers.randnC16"},[s("span",{class:"jlbinding"},"WeightInitializers.randnC16")],-1)),i[105]||(i[105]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[107]||(i[107]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[106]||(i[106]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",N,[s("summary",null,[i[108]||(i[108]=s("a",{id:"WeightInitializers.zerosC32",href:"#WeightInitializers.zerosC32"},[s("span",{class:"jlbinding"},"WeightInitializers.zerosC32")],-1)),i[109]||(i[109]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[111]||(i[111]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[110]||(i[110]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",w,[s("summary",null,[i[112]||(i[112]=s("a",{id:"WeightInitializers.onesC32",href:"#WeightInitializers.onesC32"},[s("span",{class:"jlbinding"},"WeightInitializers.onesC32")],-1)),i[113]||(i[113]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[115]||(i[115]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[114]||(i[114]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",V,[s("summary",null,[i[116]||(i[116]=s("a",{id:"WeightInitializers.randC32",href:"#WeightInitializers.randC32"},[s("span",{class:"jlbinding"},"WeightInitializers.randC32")],-1)),i[117]||(i[117]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[119]||(i[119]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[118]||(i[118]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",Q,[s("summary",null,[i[120]||(i[120]=s("a",{id:"WeightInitializers.randnC32",href:"#WeightInitializers.randnC32"},[s("span",{class:"jlbinding"},"WeightInitializers.randnC32")],-1)),i[121]||(i[121]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[123]||(i[123]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[122]||(i[122]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",G,[s("summary",null,[i[124]||(i[124]=s("a",{id:"WeightInitializers.zerosC64",href:"#WeightInitializers.zerosC64"},[s("span",{class:"jlbinding"},"WeightInitializers.zerosC64")],-1)),i[125]||(i[125]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[127]||(i[127]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[126]||(i[126]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",U,[s("summary",null,[i[128]||(i[128]=s("a",{id:"WeightInitializers.onesC64",href:"#WeightInitializers.onesC64"},[s("span",{class:"jlbinding"},"WeightInitializers.onesC64")],-1)),i[129]||(i[129]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[131]||(i[131]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[130]||(i[130]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",O,[s("summary",null,[i[132]||(i[132]=s("a",{id:"WeightInitializers.randC64",href:"#WeightInitializers.randC64"},[s("span",{class:"jlbinding"},"WeightInitializers.randC64")],-1)),i[133]||(i[133]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[135]||(i[135]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[134]||(i[134]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",M,[s("summary",null,[i[136]||(i[136]=s("a",{id:"WeightInitializers.randnC64",href:"#WeightInitializers.randnC64"},[s("span",{class:"jlbinding"},"WeightInitializers.randnC64")],-1)),i[137]||(i[137]=n()),t(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),i[139]||(i[139]=l("",2)),t(a,{type:"info",class:"source-link",text:"source"},{default:e(()=>i[138]||(i[138]=[s("a",{href:"https://github.com/LuxDL/Lux.jl/blob/626f137dd178d7841dba44465dab61058f85ba73/lib/WeightInitializers/src/initializers.jl#L7-L12",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const ii=k(d,[["render",q]]);export{Y as __pageData,ii as default};

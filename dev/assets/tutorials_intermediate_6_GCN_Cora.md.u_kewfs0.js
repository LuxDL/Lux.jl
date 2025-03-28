import{_ as a,c as n,o as e,al as p}from"./chunks/framework.BCN3FD2k.js";const d=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),c={name:"tutorials/intermediate/6_GCN_Cora.md"};function i(t,s,r,l,f,o){return e(),n("div",null,s[0]||(s[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Reactant,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    MLDatasets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Statistics,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Enzyme,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    GNNGraphs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ConcreteStructs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Printf,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    OneHotArrays,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Optimisers</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; force</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(::MLDataDevices.CPUDevice) (generic function with 1 method)</span></span></code></pre></div><h2 id="Loading-Cora-Dataset" tabindex="-1">Loading Cora Dataset <a class="header-anchor" href="#Loading-Cora-Dataset" aria-label="Permalink to &quot;Loading Cora Dataset {#Loading-Cora-Dataset}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Cora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">graphs[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gnngraph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GNNGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_index; ndata</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data, edata</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_data, gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">num_nodes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">features,</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">targets, data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">metadata[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;classes&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use a dense matrix here to avoid incompatibility with Reactant</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adjacency_matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gnngraph)),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use this since Reactant doesn&#39;t yet support gather adjoint</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">140</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">141</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">640</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1709</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2708</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>loadcora (generic function with 1 method)</span></span></code></pre></div><h2 id="Model-Definition" tabindex="-1">Model Definition <a class="header-anchor" href="#Model-Definition" aria-label="Permalink to &quot;Model Definition {#Model-Definition}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dense</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> adj</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, h_dim, out_dim; nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    layer_sizes </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, [h_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nb_layers])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn_layers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (in_dim, out_dim) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> zip</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)], layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    last_layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dropout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dropout)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; gcn_layers, dropout, last_layer) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj, mask)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gcn_layers</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> relu</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> last_layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj))[:, mask]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>GCN (generic function with 1 method)</span></span></code></pre></div><h2 id="Helper-Functions" tabindex="-1">Helper Functions <a class="header-anchor" href="#Helper-Functions" aria-label="Permalink to &quot;Helper Functions {#Helper-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y, adj, mask))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj, mask), ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; agg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">mean, logits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))(y_pred, y[:, mask])</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st, (; y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    hidden_dim</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    use_bias</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lr</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.001</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    weight_decay</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    patience</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    epochs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">200</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">seed!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    features, targets, adj, (train_idx, val_idx, test_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(features, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), hidden_dim, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); nb_layers, dropout, use_bias)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, gcn))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    opt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(weight_decay) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lr) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AdamW</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; eta</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lr, lambda</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">weight_decay)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gcn, ps, st, opt)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Total Trainable Parameters: %0.4f M</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ps) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0e6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_loss_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gcn, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st), (features, targets, adj, val_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, train_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, val_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Inf</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">epochs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss_function,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (features, targets, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            return_gradients</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                train_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, train_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            val_loss_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                gcn,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                (features, targets, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                val_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, val_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch %3d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Val Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 Val Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss train_acc val_loss val_acc</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> best_loss_val</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> patience</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Early Stopping at Epoch %d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                break</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            gcn,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (features, targets, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, test_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Test Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Test Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_loss test_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-03-28 03:49:27.947899: I external/xla/xla/service/service.cc:152] XLA service 0x79b9620 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-28 03:49:27.948267: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1743133767.949136 2952737 se_gpu_pjrt_client.cc:1039] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1743133767.949211 2952737 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743133767.949260 2952737 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1743133767.966335 2952737 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-03-28 03:50:23.692108: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 336 bytes spill stores, 336 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:23.722833: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:23.743560: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:23.900394: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 128 bytes spill stores, 128 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:24.100549: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 432 bytes spill stores, 432 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:24.159247: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 8264 bytes spill stores, 8320 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:24.212492: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 8324 bytes spill stores, 8340 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:24.213512: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 760 bytes spill stores, 760 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:24.373459: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 124 bytes spill stores, 124 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:24.592796: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 5128 bytes spill stores, 5120 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:24.913074: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 16 bytes spill stores, 16 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.031219: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 1108 bytes spill stores, 1108 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.138452: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 124 bytes spill stores, 124 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.150130: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 1864 bytes spill stores, 1900 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.184734: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 24 bytes spill stores, 24 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.317750: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 4724 bytes spill stores, 4768 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.477295: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.559923: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 4088 bytes spill stores, 4036 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.640248: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.805456: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 320 bytes spill stores, 320 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:25.985870: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 2872 bytes spill stores, 2848 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:26.846989: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 24 bytes spill stores, 24 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:27.097616: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 648 bytes spill stores, 652 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:27.197826: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:28.148091: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_29&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:28.233436: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 1176 bytes spill stores, 1148 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:28.414066: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:28.424762: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:50:29.815157: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_19&#39;, 15432 bytes spill stores, 15608 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743133829.971044 2952737 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1743133829.971710 2952737 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1743133829.971719 2952737 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1743133829.971726 2952737 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1743133829.971733 2952737 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1743133829.971739 2952737 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1743133829.971746 2952737 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1743133829.971752 2952737 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1743133829.971759 2952737 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1743133829.971766 2952737 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-28 03:50:29.971780: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.974486 2952737 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1743133829.974515 2952737 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1743133829.974523 2952737 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1743133829.974530 2952737 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1743133829.974537 2952737 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1743133829.974543 2952737 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1743133829.974550 2952737 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1743133829.974556 2952737 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1743133829.974563 2952737 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1743133829.974569 2952737 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-28 03:50:29.974580: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.976812 2952737 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1743133829.976827 2952737 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1743133829.976832 2952737 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1743133829.976836 2952737 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1743133829.976840 2952737 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1743133829.976844 2952737 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1743133829.976848 2952737 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1743133829.976852 2952737 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1743133829.976856 2952737 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1743133829.976861 2952737 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-28 03:50:29.976867: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.979092 2952737 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1743133829.979108 2952737 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1743133829.979113 2952737 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1743133829.979117 2952737 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1743133829.979121 2952737 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1743133829.979125 2952737 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1743133829.979129 2952737 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1743133829.979133 2952737 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1743133829.979137 2952737 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1743133829.979142 2952737 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-03-28 03:50:29.979148: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.981444 2952737 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1743133829.981460 2952737 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1743133829.981464 2952737 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1743133829.981468 2952737 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1743133829.981473 2952737 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1743133829.981479 2952737 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1743133829.981483 2952737 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1743133829.981487 2952737 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1743133829.981491 2952737 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1743133829.981495 2952737 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-03-28 03:50:29.981502: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.983884 2952737 buffer_comparator.cc:156] Difference at 0: 16.5257, expected 14.4011</span></span>
<span class="line"><span>E0000 00:00:1743133829.984083 2952737 buffer_comparator.cc:156] Difference at 1: 19.4064, expected 15.9904</span></span>
<span class="line"><span>E0000 00:00:1743133829.984087 2952737 buffer_comparator.cc:156] Difference at 2: 16.1909, expected 13.4103</span></span>
<span class="line"><span>E0000 00:00:1743133829.984090 2952737 buffer_comparator.cc:156] Difference at 6: 13.1689, expected 11.4953</span></span>
<span class="line"><span>E0000 00:00:1743133829.984093 2952737 buffer_comparator.cc:156] Difference at 9: 16.2882, expected 14.2452</span></span>
<span class="line"><span>E0000 00:00:1743133829.984096 2952737 buffer_comparator.cc:156] Difference at 11: 15.6385, expected 13.739</span></span>
<span class="line"><span>E0000 00:00:1743133829.984099 2952737 buffer_comparator.cc:156] Difference at 12: 20.6748, expected 16.297</span></span>
<span class="line"><span>E0000 00:00:1743133829.984102 2952737 buffer_comparator.cc:156] Difference at 13: 17.2352, expected 14.372</span></span>
<span class="line"><span>E0000 00:00:1743133829.984105 2952737 buffer_comparator.cc:156] Difference at 14: 14.761, expected 12.4213</span></span>
<span class="line"><span>E0000 00:00:1743133829.984108 2952737 buffer_comparator.cc:156] Difference at 16: 17.262, expected 15.1227</span></span>
<span class="line"><span>2025-03-28 03:50:29.984113: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.986234 2952737 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743133829.986246 2952737 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743133829.986249 2952737 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743133829.986252 2952737 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743133829.986255 2952737 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743133829.986258 2952737 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743133829.986261 2952737 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743133829.986264 2952737 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743133829.986267 2952737 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743133829.986269 2952737 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-28 03:50:29.986274: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.988399 2952737 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743133829.988410 2952737 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743133829.988413 2952737 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743133829.988416 2952737 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743133829.988419 2952737 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743133829.988422 2952737 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743133829.988425 2952737 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743133829.988428 2952737 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743133829.988431 2952737 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743133829.988436 2952737 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-28 03:50:29.988440: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.990542 2952737 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1743133829.990553 2952737 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1743133829.990556 2952737 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1743133829.990559 2952737 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1743133829.990562 2952737 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1743133829.990565 2952737 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1743133829.990568 2952737 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1743133829.990571 2952737 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1743133829.990574 2952737 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1743133829.990577 2952737 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-28 03:50:29.990581: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.992703 2952737 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1743133829.992714 2952737 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1743133829.992717 2952737 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1743133829.992720 2952737 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1743133829.992723 2952737 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1743133829.992726 2952737 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1743133829.992729 2952737 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1743133829.992732 2952737 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1743133829.992735 2952737 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1743133829.992738 2952737 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-28 03:50:29.992743: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.994859 2952737 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1743133829.994871 2952737 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1743133829.994874 2952737 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1743133829.994877 2952737 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1743133829.994880 2952737 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1743133829.994883 2952737 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1743133829.994886 2952737 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1743133829.994888 2952737 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1743133829.994891 2952737 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1743133829.994894 2952737 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-28 03:50:29.994899: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.997012 2952737 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1743133829.997027 2952737 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1743133829.997030 2952737 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1743133829.997033 2952737 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1743133829.997036 2952737 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1743133829.997039 2952737 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1743133829.997042 2952737 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1743133829.997045 2952737 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1743133829.997047 2952737 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1743133829.997050 2952737 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-28 03:50:29.997055: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133829.999171 2952737 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743133829.999182 2952737 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743133829.999186 2952737 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743133829.999189 2952737 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743133829.999192 2952737 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743133829.999194 2952737 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743133829.999197 2952737 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743133829.999200 2952737 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743133829.999203 2952737 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743133829.999206 2952737 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-28 03:50:29.999211: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.001329 2952737 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743133830.001341 2952737 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743133830.001345 2952737 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743133830.001348 2952737 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743133830.001350 2952737 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743133830.001353 2952737 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743133830.001356 2952737 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743133830.001359 2952737 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743133830.001362 2952737 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743133830.001365 2952737 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-28 03:50:30.001370: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.003483 2952737 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743133830.003496 2952737 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743133830.003499 2952737 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743133830.003503 2952737 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743133830.003505 2952737 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743133830.003510 2952737 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743133830.003513 2952737 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743133830.003516 2952737 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743133830.003519 2952737 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743133830.003522 2952737 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-28 03:50:30.003526: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.005652 2952737 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1743133830.005665 2952737 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1743133830.005668 2952737 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1743133830.005671 2952737 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1743133830.005674 2952737 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1743133830.005677 2952737 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1743133830.005680 2952737 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1743133830.005683 2952737 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1743133830.005686 2952737 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1743133830.005689 2952737 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-28 03:50:30.005693: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.032562 2952737 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.032609 2952737 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743133830.032614 2952737 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743133830.032619 2952737 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743133830.032623 2952737 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743133830.032628 2952737 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743133830.032632 2952737 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133830.032636 2952737 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743133830.032641 2952737 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743133830.032645 2952737 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.032656: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.035805 2952737 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.035824 2952737 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743133830.035828 2952737 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743133830.035833 2952737 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743133830.035837 2952737 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743133830.035841 2952737 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743133830.035846 2952737 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133830.035850 2952737 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743133830.035855 2952737 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743133830.035859 2952737 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.035866: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.039193 2952737 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.039212 2952737 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743133830.039217 2952737 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743133830.039221 2952737 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743133830.039225 2952737 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743133830.039230 2952737 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743133830.039234 2952737 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133830.039238 2952737 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743133830.039242 2952737 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743133830.039247 2952737 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.039253: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.042401 2952737 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.042423 2952737 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743133830.042428 2952737 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743133830.042432 2952737 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743133830.042436 2952737 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743133830.042441 2952737 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743133830.042445 2952737 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133830.042449 2952737 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743133830.042453 2952737 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743133830.042458 2952737 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.042464: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.045721 2952737 buffer_comparator.cc:156] Difference at 0: 1139.71, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1743133830.045735 2952737 buffer_comparator.cc:156] Difference at 1: 1404.8, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1743133830.045738 2952737 buffer_comparator.cc:156] Difference at 2: 2132.23, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1743133830.045741 2952737 buffer_comparator.cc:156] Difference at 3: 1838.84, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1743133830.045744 2952737 buffer_comparator.cc:156] Difference at 4: 1307.39, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1743133830.045747 2952737 buffer_comparator.cc:156] Difference at 5: 2064.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.045750 2952737 buffer_comparator.cc:156] Difference at 6: 1480.82, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1743133830.045753 2952737 buffer_comparator.cc:156] Difference at 7: 1113.19, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1743133830.045756 2952737 buffer_comparator.cc:156] Difference at 8: 1358.65, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1743133830.045759 2952737 buffer_comparator.cc:156] Difference at 9: 2048.24, expected 1833.76</span></span>
<span class="line"><span>2025-03-28 03:50:30.045765: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.048851 2952737 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.048868 2952737 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1743133830.048871 2952737 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1743133830.048874 2952737 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1743133830.048877 2952737 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1743133830.048880 2952737 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1743133830.048883 2952737 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133830.048886 2952737 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1743133830.048889 2952737 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1743133830.048892 2952737 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.048897: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.051964 2952737 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1743133830.051977 2952737 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743133830.051980 2952737 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1743133830.051984 2952737 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1743133830.051987 2952737 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743133830.051990 2952737 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743133830.051993 2952737 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1743133830.051996 2952737 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1743133830.051999 2952737 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1743133830.052002 2952737 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-03-28 03:50:30.052007: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.055016 2952737 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1743133830.055030 2952737 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743133830.055033 2952737 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1743133830.055036 2952737 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1743133830.055039 2952737 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743133830.055042 2952737 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743133830.055045 2952737 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1743133830.055048 2952737 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1743133830.055052 2952737 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1743133830.055055 2952737 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-03-28 03:50:30.055059: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.058098 2952737 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1743133830.058112 2952737 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743133830.058115 2952737 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1743133830.058118 2952737 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1743133830.058121 2952737 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743133830.058124 2952737 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743133830.058127 2952737 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1743133830.058130 2952737 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1743133830.058133 2952737 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1743133830.058136 2952737 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-03-28 03:50:30.058141: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.061225 2952737 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743133830.061238 2952737 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133830.061242 2952737 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743133830.061245 2952737 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743133830.061248 2952737 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.061251 2952737 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743133830.061254 2952737 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743133830.061257 2952737 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743133830.061260 2952737 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133830.061263 2952737 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-28 03:50:30.061268: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.064277 2952737 buffer_comparator.cc:156] Difference at 0: 1057.27, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1743133830.064290 2952737 buffer_comparator.cc:156] Difference at 1: 1319.15, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1743133830.064294 2952737 buffer_comparator.cc:156] Difference at 2: 2004.43, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1743133830.064297 2952737 buffer_comparator.cc:156] Difference at 3: 1745.74, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1743133830.064300 2952737 buffer_comparator.cc:156] Difference at 4: 1252.2, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1743133830.064303 2952737 buffer_comparator.cc:156] Difference at 7: 1175.57, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1743133830.064306 2952737 buffer_comparator.cc:156] Difference at 8: 1398.75, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1743133830.064309 2952737 buffer_comparator.cc:156] Difference at 9: 2125.62, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1743133830.064312 2952737 buffer_comparator.cc:156] Difference at 10: 1878.38, expected 1592.37</span></span>
<span class="line"><span>E0000 00:00:1743133830.064315 2952737 buffer_comparator.cc:156] Difference at 11: 1362.67, expected 1121.95</span></span>
<span class="line"><span>2025-03-28 03:50:30.064320: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.067383 2952737 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743133830.067400 2952737 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133830.067403 2952737 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743133830.067406 2952737 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743133830.067409 2952737 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.067413 2952737 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743133830.067416 2952737 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743133830.067419 2952737 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743133830.067422 2952737 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133830.067425 2952737 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-28 03:50:30.067430: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.070497 2952737 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743133830.070511 2952737 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133830.070515 2952737 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743133830.070518 2952737 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743133830.070521 2952737 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.070524 2952737 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743133830.070527 2952737 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743133830.070530 2952737 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743133830.070533 2952737 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133830.070536 2952737 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-28 03:50:30.070541: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.073591 2952737 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743133830.073604 2952737 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133830.073607 2952737 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743133830.073610 2952737 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743133830.073614 2952737 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.073617 2952737 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743133830.073620 2952737 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743133830.073623 2952737 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743133830.073626 2952737 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133830.073629 2952737 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-28 03:50:30.073634: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.076674 2952737 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1743133830.076689 2952737 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133830.076693 2952737 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1743133830.076698 2952737 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1743133830.076701 2952737 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.076704 2952737 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1743133830.076707 2952737 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1743133830.076710 2952737 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1743133830.076714 2952737 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133830.076717 2952737 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-28 03:50:30.076721: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.079921 2952737 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743133830.079937 2952737 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743133830.079940 2952737 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.079943 2952737 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743133830.079946 2952737 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743133830.079949 2952737 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.079952 2952737 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743133830.079955 2952737 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743133830.079958 2952737 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743133830.079961 2952737 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-28 03:50:30.079966: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.083152 2952737 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743133830.083166 2952737 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743133830.083169 2952737 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.083172 2952737 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743133830.083175 2952737 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743133830.083178 2952737 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.083181 2952737 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743133830.083184 2952737 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743133830.083187 2952737 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743133830.083190 2952737 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-28 03:50:30.083195: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.086309 2952737 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743133830.086327 2952737 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743133830.086331 2952737 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.086334 2952737 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743133830.086337 2952737 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743133830.086340 2952737 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.086344 2952737 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743133830.086347 2952737 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743133830.086350 2952737 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743133830.086353 2952737 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-28 03:50:30.086357: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.089526 2952737 buffer_comparator.cc:156] Difference at 896: 1198.32, expected 958.128</span></span>
<span class="line"><span>E0000 00:00:1743133830.089542 2952737 buffer_comparator.cc:156] Difference at 897: 1047.69, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743133830.089546 2952737 buffer_comparator.cc:156] Difference at 898: 733.669, expected 1826.79</span></span>
<span class="line"><span>E0000 00:00:1743133830.089549 2952737 buffer_comparator.cc:156] Difference at 899: 1177.34, expected 1593.43</span></span>
<span class="line"><span>E0000 00:00:1743133830.089552 2952737 buffer_comparator.cc:156] Difference at 900: 842.502, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743133830.089555 2952737 buffer_comparator.cc:156] Difference at 901: 627.594, expected 1796.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.089558 2952737 buffer_comparator.cc:156] Difference at 902: 792.637, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743133830.089561 2952737 buffer_comparator.cc:156] Difference at 903: 1202.18, expected 941.479</span></span>
<span class="line"><span>E0000 00:00:1743133830.089564 2952737 buffer_comparator.cc:156] Difference at 904: 1049.9, expected 1202.97</span></span>
<span class="line"><span>E0000 00:00:1743133830.089567 2952737 buffer_comparator.cc:156] Difference at 905: 739.86, expected 1817.41</span></span>
<span class="line"><span>2025-03-28 03:50:30.089572: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.092982 2952737 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1743133830.092997 2952737 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743133830.093000 2952737 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.093003 2952737 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743133830.093006 2952737 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1743133830.093010 2952737 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1743133830.093013 2952737 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1743133830.093016 2952737 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1743133830.093019 2952737 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1743133830.093022 2952737 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-03-28 03:50:30.093027: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.096063 2952737 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1743133830.096078 2952737 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743133830.096081 2952737 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.096084 2952737 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743133830.096087 2952737 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1743133830.096090 2952737 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1743133830.096094 2952737 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1743133830.096098 2952737 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1743133830.096101 2952737 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1743133830.096104 2952737 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-03-28 03:50:30.096109: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.099078 2952737 buffer_comparator.cc:156] Difference at 1792: 1216.65, expected 926.778</span></span>
<span class="line"><span>E0000 00:00:1743133830.099092 2952737 buffer_comparator.cc:156] Difference at 1793: 1058.09, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743133830.099095 2952737 buffer_comparator.cc:156] Difference at 1794: 743.338, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743133830.099098 2952737 buffer_comparator.cc:156] Difference at 1795: 1184.75, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743133830.099101 2952737 buffer_comparator.cc:156] Difference at 1796: 852.404, expected 1101.04</span></span>
<span class="line"><span>E0000 00:00:1743133830.099104 2952737 buffer_comparator.cc:156] Difference at 1797: 626.131, expected 1756.21</span></span>
<span class="line"><span>E0000 00:00:1743133830.099107 2952737 buffer_comparator.cc:156] Difference at 1798: 799.781, expected 1272.34</span></span>
<span class="line"><span>E0000 00:00:1743133830.099111 2952737 buffer_comparator.cc:156] Difference at 1799: 1209.98, expected 944.465</span></span>
<span class="line"><span>E0000 00:00:1743133830.099114 2952737 buffer_comparator.cc:156] Difference at 1800: 1057.15, expected 1200.58</span></span>
<span class="line"><span>E0000 00:00:1743133830.099117 2952737 buffer_comparator.cc:156] Difference at 1801: 742.39, expected 1808.36</span></span>
<span class="line"><span>2025-03-28 03:50:30.099122: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.106734 2952737 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1743133830.106748 2952737 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1743133830.106751 2952737 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1743133830.106754 2952737 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1743133830.106757 2952737 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1743133830.106760 2952737 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1743133830.106763 2952737 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1743133830.106766 2952737 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1743133830.106769 2952737 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1743133830.106772 2952737 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-28 03:50:30.106776: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.112017 2952737 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1743133830.112031 2952737 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1743133830.112034 2952737 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1743133830.112037 2952737 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1743133830.112040 2952737 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1743133830.112043 2952737 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1743133830.112046 2952737 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1743133830.112049 2952737 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1743133830.112052 2952737 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1743133830.112055 2952737 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-28 03:50:30.112060: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.116448 2952737 buffer_comparator.cc:156] Difference at 64: 0, expected 1106.21</span></span>
<span class="line"><span>E0000 00:00:1743133830.116462 2952737 buffer_comparator.cc:156] Difference at 65: 0, expected 1087.83</span></span>
<span class="line"><span>E0000 00:00:1743133830.116465 2952737 buffer_comparator.cc:156] Difference at 66: 0, expected 1090.54</span></span>
<span class="line"><span>E0000 00:00:1743133830.116468 2952737 buffer_comparator.cc:156] Difference at 67: 0, expected 1104.23</span></span>
<span class="line"><span>E0000 00:00:1743133830.116471 2952737 buffer_comparator.cc:156] Difference at 68: 0, expected 1104.3</span></span>
<span class="line"><span>E0000 00:00:1743133830.116474 2952737 buffer_comparator.cc:156] Difference at 69: 0, expected 1093.45</span></span>
<span class="line"><span>E0000 00:00:1743133830.116477 2952737 buffer_comparator.cc:156] Difference at 70: 0, expected 1091.52</span></span>
<span class="line"><span>E0000 00:00:1743133830.116480 2952737 buffer_comparator.cc:156] Difference at 71: 0, expected 1110.4</span></span>
<span class="line"><span>E0000 00:00:1743133830.116483 2952737 buffer_comparator.cc:156] Difference at 72: 0, expected 1106.92</span></span>
<span class="line"><span>E0000 00:00:1743133830.116486 2952737 buffer_comparator.cc:156] Difference at 73: 0, expected 1088.44</span></span>
<span class="line"><span>2025-03-28 03:50:30.116490: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.120712 2952737 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743133830.120726 2952737 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743133830.120729 2952737 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743133830.120732 2952737 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743133830.120735 2952737 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743133830.120738 2952737 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743133830.120741 2952737 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743133830.120744 2952737 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743133830.120747 2952737 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743133830.120749 2952737 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.120754: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.125069 2952737 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743133830.125084 2952737 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743133830.125087 2952737 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743133830.125090 2952737 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743133830.125093 2952737 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743133830.125096 2952737 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743133830.125099 2952737 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743133830.125102 2952737 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743133830.125105 2952737 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743133830.125108 2952737 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.125113: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.129166 2952737 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743133830.129180 2952737 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743133830.129184 2952737 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743133830.129187 2952737 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743133830.129190 2952737 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743133830.129193 2952737 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743133830.129196 2952737 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743133830.129199 2952737 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743133830.129202 2952737 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743133830.129205 2952737 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.129209: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.133177 2952737 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743133830.133191 2952737 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743133830.133194 2952737 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743133830.133197 2952737 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743133830.133200 2952737 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743133830.133203 2952737 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743133830.133206 2952737 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743133830.133209 2952737 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743133830.133212 2952737 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743133830.133214 2952737 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.133219: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.137285 2952737 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743133830.137299 2952737 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743133830.137302 2952737 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743133830.137305 2952737 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743133830.137308 2952737 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1743133830.137311 2952737 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1743133830.137314 2952737 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1743133830.137316 2952737 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1743133830.137319 2952737 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1743133830.137322 2952737 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-28 03:50:30.137327: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133830.141492 2952737 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1743133830.141508 2952737 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1743133830.141512 2952737 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1743133830.141514 2952737 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1743133830.141517 2952737 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>Epoch   1	Train Loss: 17.101273	Train Acc: 20.0000%	Val Loss: 7.181659	Val Acc: 27.2000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 9.362435	Train Acc: 24.2857%	Val Loss: 3.449754	Val Acc: 28.8000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 3.877512	Train Acc: 43.5714%	Val Loss: 2.159894	Val Acc: 33.6000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 1.769575	Train Acc: 52.8571%	Val Loss: 2.090940	Val Acc: 39.8000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 1.496373	Train Acc: 61.4286%	Val Loss: 2.008781	Val Acc: 41.4000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.250796	Train Acc: 69.2857%	Val Loss: 1.817183	Val Acc: 46.4000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.082509	Train Acc: 72.1429%	Val Loss: 1.635154	Val Acc: 52.8000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 0.964871	Train Acc: 74.2857%	Val Loss: 1.548043	Val Acc: 56.4000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 0.943460	Train Acc: 77.8571%	Val Loss: 1.524287	Val Acc: 58.6000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 0.878995	Train Acc: 78.5714%	Val Loss: 1.531219	Val Acc: 57.6000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 0.797675	Train Acc: 80.7143%	Val Loss: 1.547833	Val Acc: 58.8000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.737401	Train Acc: 80.7143%	Val Loss: 1.561798	Val Acc: 60.6000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.691695	Train Acc: 81.4286%	Val Loss: 1.572292	Val Acc: 61.6000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.660892	Train Acc: 82.1429%	Val Loss: 1.576262	Val Acc: 61.6000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.628364	Train Acc: 82.1429%	Val Loss: 1.573994	Val Acc: 62.8000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.595277	Train Acc: 82.1429%	Val Loss: 1.570692	Val Acc: 62.4000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.571132	Train Acc: 82.8571%	Val Loss: 1.570269	Val Acc: 63.6000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.551259	Train Acc: 83.5714%	Val Loss: 1.570376	Val Acc: 63.4000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.532083	Train Acc: 85.0000%	Val Loss: 1.570112	Val Acc: 64.0000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.512467	Train Acc: 86.4286%	Val Loss: 1.570012	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.491931	Train Acc: 86.4286%	Val Loss: 1.571555	Val Acc: 63.6000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.471314	Train Acc: 87.8571%	Val Loss: 1.579162	Val Acc: 64.4000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.451298	Train Acc: 88.5714%	Val Loss: 1.592630	Val Acc: 64.2000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.433356	Train Acc: 88.5714%	Val Loss: 1.611834	Val Acc: 64.0000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.417691	Train Acc: 87.1429%	Val Loss: 1.636085	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.403636	Train Acc: 87.1429%	Val Loss: 1.661440	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.391145	Train Acc: 88.5714%	Val Loss: 1.687567	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.379763	Train Acc: 90.0000%	Val Loss: 1.711942	Val Acc: 64.4000%</span></span>
<span class="line"><span>Epoch  29	Train Loss: 0.368857	Train Acc: 90.0000%	Val Loss: 1.734173	Val Acc: 64.2000%</span></span>
<span class="line"><span>Early Stopping at Epoch 29</span></span>
<span class="line"><span>2025-03-28 03:51:33.312989: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:51:33.352009: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 24 bytes spill stores, 24 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:51:33.635981: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743133893.853580 2952737 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743133893.853642 2952737 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743133893.853651 2952737 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743133893.853658 2952737 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.853665 2952737 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743133893.853673 2952737 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743133893.853680 2952737 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133893.853687 2952737 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743133893.853694 2952737 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.853701 2952737 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-28 03:51:33.853716: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.857227 2952737 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743133893.857254 2952737 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743133893.857262 2952737 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743133893.857269 2952737 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.857276 2952737 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743133893.857283 2952737 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743133893.857290 2952737 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133893.857297 2952737 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743133893.857304 2952737 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.857311 2952737 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-28 03:51:33.857322: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.860834 2952737 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743133893.860854 2952737 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743133893.860858 2952737 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743133893.860861 2952737 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.860864 2952737 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743133893.860867 2952737 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743133893.860870 2952737 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133893.860873 2952737 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743133893.860879 2952737 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.860882 2952737 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-28 03:51:33.860888: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.864211 2952737 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743133893.864228 2952737 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743133893.864231 2952737 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743133893.864235 2952737 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.864238 2952737 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743133893.864241 2952737 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743133893.864244 2952737 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133893.864247 2952737 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743133893.864250 2952737 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.864253 2952737 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-28 03:51:33.864259: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.867526 2952737 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743133893.867544 2952737 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743133893.867547 2952737 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743133893.867550 2952737 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.867554 2952737 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743133893.867557 2952737 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743133893.867560 2952737 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133893.867563 2952737 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743133893.867566 2952737 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.867569 2952737 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-28 03:51:33.867574: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.870982 2952737 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1743133893.871017 2952737 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1743133893.871021 2952737 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1743133893.871024 2952737 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.871027 2952737 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1743133893.871030 2952737 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1743133893.871034 2952737 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1743133893.871037 2952737 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1743133893.871040 2952737 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.871043 2952737 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-28 03:51:33.871053: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.874347 2952737 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1743133893.874373 2952737 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743133893.874376 2952737 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1743133893.874379 2952737 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1743133893.874383 2952737 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743133893.874386 2952737 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743133893.874389 2952737 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1743133893.874392 2952737 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1743133893.874395 2952737 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1743133893.874398 2952737 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-03-28 03:51:33.874405: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.877785 2952737 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743133893.877848 2952737 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1743133893.877852 2952737 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1743133893.877856 2952737 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743133893.877859 2952737 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743133893.877862 2952737 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1743133893.877865 2952737 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1743133893.877868 2952737 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1743133893.877871 2952737 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1743133893.877874 2952737 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-28 03:51:33.877884: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.881424 2952737 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1743133893.881474 2952737 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1743133893.881477 2952737 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1743133893.881480 2952737 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1743133893.881484 2952737 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1743133893.881487 2952737 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1743133893.881490 2952737 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1743133893.881493 2952737 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1743133893.881496 2952737 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1743133893.881499 2952737 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-28 03:51:33.881509: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.884832 2952737 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133893.884847 2952737 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743133893.884850 2952737 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743133893.884853 2952737 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133893.884857 2952737 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743133893.884860 2952737 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743133893.884863 2952737 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.884866 2952737 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.884869 2952737 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743133893.884872 2952737 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-28 03:51:33.884877: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.888063 2952737 buffer_comparator.cc:156] Difference at 7: 1058.92, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1743133893.888078 2952737 buffer_comparator.cc:156] Difference at 11: 1263.92, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1743133893.888082 2952737 buffer_comparator.cc:156] Difference at 179: 1223.75, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1743133893.888086 2952737 buffer_comparator.cc:156] Difference at 266: 1047.35, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1743133893.888089 2952737 buffer_comparator.cc:156] Difference at 270: 1246.8, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1743133893.888093 2952737 buffer_comparator.cc:156] Difference at 417: 1222.47, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1743133893.888096 2952737 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133893.888100 2952737 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743133893.888103 2952737 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743133893.888106 2952737 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-03-28 03:51:33.888309: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.891535 2952737 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133893.891548 2952737 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743133893.891551 2952737 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743133893.891554 2952737 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133893.891557 2952737 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743133893.891561 2952737 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743133893.891564 2952737 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.891567 2952737 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.891570 2952737 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743133893.891573 2952737 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-28 03:51:33.891578: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.894935 2952737 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133893.894949 2952737 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743133893.894953 2952737 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743133893.894957 2952737 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133893.894960 2952737 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743133893.894963 2952737 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743133893.894966 2952737 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.894969 2952737 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.894972 2952737 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743133893.894975 2952737 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-28 03:51:33.894980: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.898194 2952737 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133893.898208 2952737 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743133893.898211 2952737 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743133893.898214 2952737 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133893.898218 2952737 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743133893.898221 2952737 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743133893.898224 2952737 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.898227 2952737 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.898230 2952737 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743133893.898233 2952737 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-28 03:51:33.898238: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.901439 2952737 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1743133893.901451 2952737 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1743133893.901455 2952737 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1743133893.901458 2952737 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1743133893.901461 2952737 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1743133893.901464 2952737 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1743133893.901467 2952737 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.901470 2952737 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1743133893.901473 2952737 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1743133893.901477 2952737 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-28 03:51:33.901481: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.904836 2952737 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743133893.904852 2952737 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743133893.904855 2952737 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743133893.904862 2952737 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743133893.904865 2952737 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743133893.904868 2952737 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.904871 2952737 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743133893.904874 2952737 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743133893.904877 2952737 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743133893.904880 2952737 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-28 03:51:33.904886: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.908208 2952737 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743133893.908221 2952737 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743133893.908225 2952737 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743133893.908228 2952737 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743133893.908231 2952737 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743133893.908234 2952737 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.908237 2952737 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743133893.908240 2952737 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743133893.908243 2952737 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743133893.908247 2952737 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-28 03:51:33.908251: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.911497 2952737 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743133893.911511 2952737 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743133893.911515 2952737 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743133893.911518 2952737 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743133893.911521 2952737 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743133893.911524 2952737 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.911527 2952737 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743133893.911531 2952737 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1743133893.911534 2952737 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1743133893.911537 2952737 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-28 03:51:33.911541: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.914788 2952737 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1743133893.914807 2952737 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1743133893.914811 2952737 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1743133893.914814 2952737 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1743133893.914817 2952737 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1743133893.914823 2952737 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1743133893.914826 2952737 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1743133893.914829 2952737 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1743133893.914832 2952737 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1743133893.914835 2952737 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-03-28 03:51:33.914841: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.918311 2952737 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1743133893.918328 2952737 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743133893.918331 2952737 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743133893.918335 2952737 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.918338 2952737 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1743133893.918341 2952737 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1743133893.918344 2952737 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1743133893.918347 2952737 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1743133893.918350 2952737 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.918353 2952737 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-28 03:51:33.918359: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.921801 2952737 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1743133893.921814 2952737 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743133893.921818 2952737 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743133893.921821 2952737 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.921824 2952737 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1743133893.921827 2952737 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1743133893.921831 2952737 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1743133893.921834 2952737 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1743133893.921837 2952737 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.921840 2952737 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-28 03:51:33.921845: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133893.925238 2952737 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1743133893.925252 2952737 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1743133893.925256 2952737 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1743133893.925260 2952737 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.925263 2952737 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1743133893.925266 2952737 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1743133893.925269 2952737 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1743133893.925275 2952737 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1743133893.925278 2952737 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1743133893.925281 2952737 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-28 03:51:33.925286: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-03-28 03:51:35.606678: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:51:35.770463: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 24 bytes spill stores, 24 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-28 03:51:35.859486: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1743133895.866778 2952737 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743133895.866867 2952737 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743133895.866880 2952737 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743133895.866887 2952737 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.866894 2952737 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743133895.866901 2952737 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743133895.866908 2952737 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743133895.866915 2952737 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743133895.866922 2952737 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.866928 2952737 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-28 03:51:35.866943: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.870660 2952737 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743133895.870687 2952737 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743133895.870695 2952737 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743133895.870702 2952737 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.870709 2952737 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743133895.870716 2952737 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743133895.870723 2952737 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743133895.870730 2952737 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743133895.870736 2952737 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.870743 2952737 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-28 03:51:35.870754: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.874060 2952737 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743133895.874072 2952737 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743133895.874076 2952737 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743133895.874079 2952737 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.874084 2952737 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743133895.874087 2952737 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743133895.874090 2952737 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743133895.874094 2952737 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743133895.874097 2952737 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.874099 2952737 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-28 03:51:35.874104: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.877566 2952737 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743133895.877578 2952737 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743133895.877581 2952737 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743133895.877585 2952737 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.877588 2952737 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743133895.877591 2952737 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743133895.877594 2952737 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743133895.877597 2952737 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743133895.877600 2952737 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.877603 2952737 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-28 03:51:35.877607: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.880866 2952737 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743133895.880878 2952737 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743133895.880881 2952737 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743133895.880884 2952737 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.880887 2952737 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743133895.880890 2952737 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743133895.880893 2952737 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743133895.880896 2952737 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743133895.880899 2952737 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.880902 2952737 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-28 03:51:35.880907: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.884222 2952737 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1743133895.884234 2952737 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1743133895.884238 2952737 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1743133895.884241 2952737 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.884244 2952737 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1743133895.884247 2952737 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1743133895.884250 2952737 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1743133895.884253 2952737 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1743133895.884258 2952737 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.884261 2952737 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-28 03:51:35.884266: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.887479 2952737 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1743133895.887491 2952737 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1743133895.887494 2952737 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1743133895.887497 2952737 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1743133895.887501 2952737 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1743133895.887504 2952737 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1743133895.887507 2952737 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1743133895.887510 2952737 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1743133895.887513 2952737 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1743133895.887516 2952737 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-28 03:51:35.887520: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.890692 2952737 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1743133895.890704 2952737 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1743133895.890708 2952737 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1743133895.890711 2952737 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1743133895.890714 2952737 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1743133895.890717 2952737 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1743133895.890720 2952737 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1743133895.890723 2952737 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1743133895.890726 2952737 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1743133895.890729 2952737 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-28 03:51:35.890734: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.893955 2952737 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1743133895.893967 2952737 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1743133895.893970 2952737 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1743133895.893973 2952737 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1743133895.893977 2952737 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1743133895.893980 2952737 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1743133895.893983 2952737 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1743133895.893986 2952737 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1743133895.893989 2952737 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1743133895.893992 2952737 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-28 03:51:35.893996: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.897227 2952737 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743133895.897239 2952737 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743133895.897243 2952737 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743133895.897246 2952737 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743133895.897249 2952737 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743133895.897252 2952737 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743133895.897255 2952737 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743133895.897258 2952737 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743133895.897261 2952737 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743133895.897264 2952737 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-28 03:51:35.897269: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.900448 2952737 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743133895.900460 2952737 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743133895.900463 2952737 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743133895.900466 2952737 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743133895.900470 2952737 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743133895.900473 2952737 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743133895.900476 2952737 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743133895.900479 2952737 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743133895.900482 2952737 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743133895.900485 2952737 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-28 03:51:35.900489: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.903704 2952737 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743133895.903716 2952737 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743133895.903719 2952737 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743133895.903722 2952737 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743133895.903726 2952737 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743133895.903729 2952737 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743133895.903732 2952737 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743133895.903735 2952737 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743133895.903738 2952737 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743133895.903741 2952737 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-28 03:51:35.903745: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.906974 2952737 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743133895.906986 2952737 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743133895.906990 2952737 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743133895.906994 2952737 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743133895.906998 2952737 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743133895.907001 2952737 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743133895.907004 2952737 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743133895.907007 2952737 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743133895.907010 2952737 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743133895.907013 2952737 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-28 03:51:35.907017: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.910216 2952737 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743133895.910228 2952737 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743133895.910231 2952737 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743133895.910235 2952737 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743133895.910238 2952737 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743133895.910241 2952737 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743133895.910244 2952737 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743133895.910247 2952737 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743133895.910250 2952737 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743133895.910252 2952737 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-28 03:51:35.910257: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.913444 2952737 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1743133895.913455 2952737 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1743133895.913459 2952737 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1743133895.913462 2952737 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1743133895.913465 2952737 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1743133895.913468 2952737 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1743133895.913471 2952737 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1743133895.913474 2952737 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1743133895.913477 2952737 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1743133895.913480 2952737 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-28 03:51:35.913484: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.916852 2952737 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743133895.916864 2952737 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.916868 2952737 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.916871 2952737 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743133895.916874 2952737 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743133895.916877 2952737 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743133895.916880 2952737 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743133895.916885 2952737 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743133895.916888 2952737 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743133895.916891 2952737 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-28 03:51:35.916896: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.920211 2952737 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743133895.920222 2952737 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.920226 2952737 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.920229 2952737 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743133895.920232 2952737 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743133895.920235 2952737 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743133895.920238 2952737 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743133895.920241 2952737 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743133895.920244 2952737 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743133895.920247 2952737 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-28 03:51:35.920252: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.923484 2952737 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743133895.923496 2952737 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.923499 2952737 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.923502 2952737 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743133895.923505 2952737 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743133895.923508 2952737 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743133895.923511 2952737 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743133895.923514 2952737 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743133895.923517 2952737 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743133895.923520 2952737 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-28 03:51:35.923525: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.926747 2952737 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1743133895.926759 2952737 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1743133895.926762 2952737 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1743133895.926766 2952737 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1743133895.926769 2952737 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1743133895.926772 2952737 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1743133895.926775 2952737 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1743133895.926778 2952737 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1743133895.926781 2952737 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1743133895.926784 2952737 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-28 03:51:35.926788: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.930247 2952737 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1743133895.930259 2952737 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1743133895.930262 2952737 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1743133895.930265 2952737 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1743133895.930269 2952737 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1743133895.930272 2952737 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1743133895.930275 2952737 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1743133895.930278 2952737 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1743133895.930281 2952737 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1743133895.930284 2952737 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-28 03:51:35.930288: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.933735 2952737 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1743133895.933747 2952737 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1743133895.933750 2952737 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1743133895.933753 2952737 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1743133895.933756 2952737 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1743133895.933759 2952737 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1743133895.933762 2952737 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1743133895.933765 2952737 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1743133895.933768 2952737 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1743133895.933771 2952737 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-28 03:51:35.933776: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1743133895.937176 2952737 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1743133895.937187 2952737 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1743133895.937191 2952737 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1743133895.937194 2952737 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1743133895.937197 2952737 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1743133895.937200 2952737 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1743133895.937203 2952737 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1743133895.937206 2952737 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1743133895.937209 2952737 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1743133895.937212 2952737 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-28 03:51:35.937217: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.552145	Test Acc: 66.2000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MLDataDevices)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.4</span></span>
<span class="line"><span>Commit 8561cc3d68d (2025-03-10 11:36 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LLVM: libLLVM-16.0.6 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,21)]))}const E=a(c,[["render",i]]);export{d as __pageData,E as default};

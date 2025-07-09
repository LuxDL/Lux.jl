import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.CYuWMjW_.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function h(e,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.68/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.68/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.68/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.68/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.73673</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.70158</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.63264</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60179</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56753</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54071</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50607</span></span>
<span class="line"><span>Validation: Loss 0.47787 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47686 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47749</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45250</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43124</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40331</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38042</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37468</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35117</span></span>
<span class="line"><span>Validation: Loss 0.33918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.33153 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32913</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32219</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30392</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.27819</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.27343</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25045</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23073</span></span>
<span class="line"><span>Validation: Loss 0.23854 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.23018 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22694</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22916</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21602</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.19636</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.19386</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18130</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17992</span></span>
<span class="line"><span>Validation: Loss 0.17364 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16782 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16791</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16248</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15579</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14473</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14685</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13079</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13320</span></span>
<span class="line"><span>Validation: Loss 0.12662 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12291 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12614</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12016</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10961</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10798</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10401</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10008</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08811</span></span>
<span class="line"><span>Validation: Loss 0.09265 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09001 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09078</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08520</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08291</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08204</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07455</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07242</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06951</span></span>
<span class="line"><span>Validation: Loss 0.06840 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06612 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06580</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06198</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06142</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05998</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05557</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05431</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05521</span></span>
<span class="line"><span>Validation: Loss 0.05141 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04937 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05207</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04888</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04669</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04152</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04283</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04094</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03625</span></span>
<span class="line"><span>Validation: Loss 0.04003 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03828 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03828</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03910</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03497</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03545</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03477</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03153</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03228</span></span>
<span class="line"><span>Validation: Loss 0.03246 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03021</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02952</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02825</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02870</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02959</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02824</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02882</span></span>
<span class="line"><span>Validation: Loss 0.02713 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02587 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02659</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02676</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02497</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02342</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02317</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02217</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02518</span></span>
<span class="line"><span>Validation: Loss 0.02319 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02212 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02281</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02095</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02106</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02160</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02065</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02028</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01900</span></span>
<span class="line"><span>Validation: Loss 0.02023 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01931 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01852</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01824</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01854</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01904</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01759</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01812</span></span>
<span class="line"><span>Validation: Loss 0.01796 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01715 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01727</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01769</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01605</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01674</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01585</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01592</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01627</span></span>
<span class="line"><span>Validation: Loss 0.01615 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01542 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01573</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01489</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01580</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01321</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01322</span></span>
<span class="line"><span>Validation: Loss 0.01466 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01399 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01267</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01355</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01417</span></span>
<span class="line"><span>Validation: Loss 0.01342 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01280 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01241</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01284</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01235</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01083</span></span>
<span class="line"><span>Validation: Loss 0.01236 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01179 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01325</span></span>
<span class="line"><span>Validation: Loss 0.01145 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01091 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01059</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01032</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01039</span></span>
<span class="line"><span>Validation: Loss 0.01064 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01014 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00961</span></span>
<span class="line"><span>Validation: Loss 0.00994 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00946 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00937</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00904</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01035</span></span>
<span class="line"><span>Validation: Loss 0.00931 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00886 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00958</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Validation: Loss 0.00874 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00832 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00835</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00842</span></span>
<span class="line"><span>Validation: Loss 0.00823 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00783 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00820</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00724</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00707</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Validation: Loss 0.00777 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00739 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.73144</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.67839</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.64396</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59659</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57715</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53814</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50134</span></span>
<span class="line"><span>Validation: Loss 0.47486 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47575 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47732</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44956</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42565</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40488</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38910</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36951</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35933</span></span>
<span class="line"><span>Validation: Loss 0.33479 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.32591 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32664</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31709</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29801</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28722</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26316</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25891</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23415</span></span>
<span class="line"><span>Validation: Loss 0.23345 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22358 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23313</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22011</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20610</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20197</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18954</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18644</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16965</span></span>
<span class="line"><span>Validation: Loss 0.16929 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16289 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17248</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15691</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14939</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15211</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13778</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13366</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12813</span></span>
<span class="line"><span>Validation: Loss 0.12354 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11952 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12129</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11648</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11025</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11038</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10240</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09844</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09438</span></span>
<span class="line"><span>Validation: Loss 0.09038 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08732 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08690</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08627</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08206</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08071</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07612</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07176</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06464</span></span>
<span class="line"><span>Validation: Loss 0.06657 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06384 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06580</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06142</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06099</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05952</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05652</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05247</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04920</span></span>
<span class="line"><span>Validation: Loss 0.04998 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04735 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04984</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04869</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04513</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04462</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04283</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03819</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04063</span></span>
<span class="line"><span>Validation: Loss 0.03902 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03663 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03702</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03864</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03552</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03625</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03313</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03199</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03259</span></span>
<span class="line"><span>Validation: Loss 0.03162 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02954 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03234</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02995</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03071</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02743</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02662</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02684</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02702</span></span>
<span class="line"><span>Validation: Loss 0.02637 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02460 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02498</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02625</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02586</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02266</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02356</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02290</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02310</span></span>
<span class="line"><span>Validation: Loss 0.02257 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02104 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02235</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02041</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02204</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02106</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02094</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01886</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02162</span></span>
<span class="line"><span>Validation: Loss 0.01971 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01839 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01974</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01898</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01828</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01841</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01823</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01780</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01521</span></span>
<span class="line"><span>Validation: Loss 0.01749 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01632 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01649</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01711</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01699</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01602</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01727</span></span>
<span class="line"><span>Validation: Loss 0.01573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01468 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01557</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01498</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01444</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01454</span></span>
<span class="line"><span>Validation: Loss 0.01428 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01333 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01433</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01287</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01397</span></span>
<span class="line"><span>Validation: Loss 0.01307 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01219 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01247</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01323</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01210</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01216</span></span>
<span class="line"><span>Validation: Loss 0.01203 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01122 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01078</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00963</span></span>
<span class="line"><span>Validation: Loss 0.01114 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01038 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01135</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01027</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01033</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00998</span></span>
<span class="line"><span>Validation: Loss 0.01036 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00965 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01011</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00977</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00943</span></span>
<span class="line"><span>Validation: Loss 0.00968 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00901 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00942</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00896</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00799</span></span>
<span class="line"><span>Validation: Loss 0.00907 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00844 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00664</span></span>
<span class="line"><span>Validation: Loss 0.00852 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00792 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00725</span></span>
<span class="line"><span>Validation: Loss 0.00804 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00747 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00787</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00745</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00770</span></span>
<span class="line"><span>Validation: Loss 0.00760 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00705 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxDeviceUtils)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.5</span></span>
<span class="line"><span>Commit 6f3fdf7b362 (2024-08-27 14:19 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 4 default, 0 interactive, 2 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 4</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.2+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.5</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.484 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",h]]);export{r as __pageData,d as default};

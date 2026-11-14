import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DEmcUXGC.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function h(e,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.0.4/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.0.4/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.0.4/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.0.4/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62119</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59574</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55560</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54413</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52242</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49230</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50040</span></span>
<span class="line"><span>Validation: Loss 0.47404 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46642 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47304</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44008</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43130</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43225</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42277</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39227</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40997</span></span>
<span class="line"><span>Validation: Loss 0.37889 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36971 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37536</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36507</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34466</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32613</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32422</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30613</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30507</span></span>
<span class="line"><span>Validation: Loss 0.29538 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28527 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28244</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27678</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27830</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26213</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23831</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23514</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21314</span></span>
<span class="line"><span>Validation: Loss 0.22583 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21560 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21782</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21827</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20122</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18831</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19369</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16890</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14980</span></span>
<span class="line"><span>Validation: Loss 0.16984 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16043 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16845</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15413</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15137</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14031</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12981</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13301</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13494</span></span>
<span class="line"><span>Validation: Loss 0.12594 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11818 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11420</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11860</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10908</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10439</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10327</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09212</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09798</span></span>
<span class="line"><span>Validation: Loss 0.09058 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08486 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08829</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08023</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07346</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07633</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07117</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06883</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05656</span></span>
<span class="line"><span>Validation: Loss 0.06288 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05908 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05933</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05994</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05380</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05270</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05178</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04477</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04302</span></span>
<span class="line"><span>Validation: Loss 0.04644 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04801</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04085</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03921</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04068</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03650</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03830</span></span>
<span class="line"><span>Validation: Loss 0.03753 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03539 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03724</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03387</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03543</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03485</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03291</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02831</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03010</span></span>
<span class="line"><span>Validation: Loss 0.03185 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03000 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02944</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03070</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02904</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02895</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02837</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02685</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02608</span></span>
<span class="line"><span>Validation: Loss 0.02774 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02609 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02728</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02725</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02421</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02515</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02518</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02340</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02039</span></span>
<span class="line"><span>Validation: Loss 0.02455 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02307 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02285</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02289</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02128</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02238</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02189</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02535</span></span>
<span class="line"><span>Validation: Loss 0.02201 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02065 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02104</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02060</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01973</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02028</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01927</span></span>
<span class="line"><span>Validation: Loss 0.01987 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01863 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01791</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01925</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01882</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01698</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01830</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01788</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01814</span></span>
<span class="line"><span>Validation: Loss 0.01808 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01692 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01725</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01620</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01844</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01692</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01524</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01521</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01724</span></span>
<span class="line"><span>Validation: Loss 0.01654 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01547 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01577</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01542</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01538</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01547</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01366</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01408</span></span>
<span class="line"><span>Validation: Loss 0.01521 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01422 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01476</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01374</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01302</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01413</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01314</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01465</span></span>
<span class="line"><span>Validation: Loss 0.01406 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01313 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01434</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01379</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01218</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01407</span></span>
<span class="line"><span>Validation: Loss 0.01303 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01217 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01191</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01214</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01207</span></span>
<span class="line"><span>Validation: Loss 0.01205 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01126 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01220</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01123</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01023</span></span>
<span class="line"><span>Validation: Loss 0.01105 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01033 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01136</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00999</span></span>
<span class="line"><span>Validation: Loss 0.00992 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00929 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00930</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00894</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00753</span></span>
<span class="line"><span>Validation: Loss 0.00878 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00824 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00741</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00745</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00824</span></span>
<span class="line"><span>Validation: Loss 0.00790 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00743 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62722</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59413</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55655</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54395</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51816</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51010</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47961</span></span>
<span class="line"><span>Validation: Loss 0.46258 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46519 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47281</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45069</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44288</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42149</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41961</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40148</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37439</span></span>
<span class="line"><span>Validation: Loss 0.36427 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36735 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37005</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36556</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35175</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32889</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31650</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31270</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30203</span></span>
<span class="line"><span>Validation: Loss 0.27841 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28163 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28210</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29894</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25652</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26337</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23870</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22994</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21905</span></span>
<span class="line"><span>Validation: Loss 0.20832 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21135 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21729</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20556</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18958</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18994</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18850</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18489</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17659</span></span>
<span class="line"><span>Validation: Loss 0.15314 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15580 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16271</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15928</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13819</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13685</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13191</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13796</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13965</span></span>
<span class="line"><span>Validation: Loss 0.11145 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11353 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12304</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11471</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10947</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10706</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09683</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08594</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08188</span></span>
<span class="line"><span>Validation: Loss 0.07944 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08091 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07766</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07953</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08340</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07306</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06519</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06961</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05725</span></span>
<span class="line"><span>Validation: Loss 0.05554 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05651 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05919</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05915</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04997</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05109</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05020</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04629</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04627</span></span>
<span class="line"><span>Validation: Loss 0.04135 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04209 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04544</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03921</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04085</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04055</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03785</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03856</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03724</span></span>
<span class="line"><span>Validation: Loss 0.03345 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03409 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03828</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03170</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03526</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03154</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03262</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03114</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03046</span></span>
<span class="line"><span>Validation: Loss 0.02835 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02892 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02955</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03087</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02767</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02730</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02769</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02786</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03004</span></span>
<span class="line"><span>Validation: Loss 0.02464 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02515 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02595</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02547</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02543</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02584</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02205</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02475</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02738</span></span>
<span class="line"><span>Validation: Loss 0.02176 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02223 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02314</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02237</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02317</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02203</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02092</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02244</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01936</span></span>
<span class="line"><span>Validation: Loss 0.01944 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01987 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02146</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01903</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01873</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01967</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02077</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01741</span></span>
<span class="line"><span>Validation: Loss 0.01753 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01793 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01718</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01893</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01770</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01792</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01866</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01826</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01704</span></span>
<span class="line"><span>Validation: Loss 0.01592 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01629 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01653</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01671</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01832</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01495</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01549</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01853</span></span>
<span class="line"><span>Validation: Loss 0.01454 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01488 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01510</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01550</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01345</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01668</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01537</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01487</span></span>
<span class="line"><span>Validation: Loss 0.01335 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01367 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01261</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01495</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01373</span></span>
<span class="line"><span>Validation: Loss 0.01231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01261 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01293</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01379</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01300</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01287</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01283</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01317</span></span>
<span class="line"><span>Validation: Loss 0.01138 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01166 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00907</span></span>
<span class="line"><span>Validation: Loss 0.01048 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01074 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01074</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00826</span></span>
<span class="line"><span>Validation: Loss 0.00953 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01066</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00893</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00774</span></span>
<span class="line"><span>Validation: Loss 0.00850 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00869 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00960</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00757</span></span>
<span class="line"><span>Validation: Loss 0.00761 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00777 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00751</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00768</span></span>
<span class="line"><span>Validation: Loss 0.00696 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00710 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",h]]);export{r as __pageData,d as default};

import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.p5L7N9Bt.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function h(e,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.0.0/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.0.0/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.0.0/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.0.0/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62971</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59257</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57339</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53836</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51698</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49490</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48043</span></span>
<span class="line"><span>Validation: Loss 0.46506 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47192 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47233</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45569</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43496</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41765</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41143</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40128</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40212</span></span>
<span class="line"><span>Validation: Loss 0.36659 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37494 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36967</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36134</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36210</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31757</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31068</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31522</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29133</span></span>
<span class="line"><span>Validation: Loss 0.28104 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29017 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29200</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27371</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26770</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25882</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23682</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22155</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24595</span></span>
<span class="line"><span>Validation: Loss 0.21095 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22010 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20887</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20335</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19391</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19752</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17516</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17962</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19653</span></span>
<span class="line"><span>Validation: Loss 0.15569 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16394 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15946</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15349</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15102</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12940</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14021</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12752</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11737</span></span>
<span class="line"><span>Validation: Loss 0.11334 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11994 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12843</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11030</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09183</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09374</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10322</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09652</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08432</span></span>
<span class="line"><span>Validation: Loss 0.08088 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08567 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08357</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07563</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07285</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07308</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06987</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06369</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06499</span></span>
<span class="line"><span>Validation: Loss 0.05648 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06047</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05518</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05308</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04792</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05168</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04425</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04354</span></span>
<span class="line"><span>Validation: Loss 0.04230 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04463 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04263</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04192</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03763</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04138</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04109</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03844</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02794</span></span>
<span class="line"><span>Validation: Loss 0.03440 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03633 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03736</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03441</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03303</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03283</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03098</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03135</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03072</span></span>
<span class="line"><span>Validation: Loss 0.02928 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03057</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03019</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02785</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03096</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02820</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02510</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02299</span></span>
<span class="line"><span>Validation: Loss 0.02552 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02703 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02736</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02528</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02305</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02419</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02659</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02531</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01943</span></span>
<span class="line"><span>Validation: Loss 0.02262 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02398 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02304</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02327</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02130</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02117</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02221</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02344</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02041</span></span>
<span class="line"><span>Validation: Loss 0.02029 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02153 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02125</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01979</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02100</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01977</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02040</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01939</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01631</span></span>
<span class="line"><span>Validation: Loss 0.01833 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01949 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01886</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01843</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01784</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01743</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01837</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01897</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01607</span></span>
<span class="line"><span>Validation: Loss 0.01668 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01775 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01815</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01744</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01513</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01683</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01694</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01578</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01466</span></span>
<span class="line"><span>Validation: Loss 0.01525 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01625 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01646</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01507</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01578</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01481</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01464</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01462</span></span>
<span class="line"><span>Validation: Loss 0.01401 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01494 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01330</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01391</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01227</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01185</span></span>
<span class="line"><span>Validation: Loss 0.01292 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01224</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01354</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01341</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01209</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01357</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01296</span></span>
<span class="line"><span>Validation: Loss 0.01193 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01273 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01229</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01350</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01101</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01006</span></span>
<span class="line"><span>Validation: Loss 0.01094 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01167 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01066</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01069</span></span>
<span class="line"><span>Validation: Loss 0.00986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01050 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00960</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00979</span></span>
<span class="line"><span>Validation: Loss 0.00872 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00927 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00895</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00905</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00906</span></span>
<span class="line"><span>Validation: Loss 0.00782 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00830 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00774</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00714</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00774</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00719</span></span>
<span class="line"><span>Validation: Loss 0.00718 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00761 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63452</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58475</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56772</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55285</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51931</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49952</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48290</span></span>
<span class="line"><span>Validation: Loss 0.46231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46178 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47137</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45927</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44311</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41728</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41874</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40626</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38031</span></span>
<span class="line"><span>Validation: Loss 0.36362 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36289 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38143</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36440</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35107</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33287</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32499</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30182</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28698</span></span>
<span class="line"><span>Validation: Loss 0.27742 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27669 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29334</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28473</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26456</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25983</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24079</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23357</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21942</span></span>
<span class="line"><span>Validation: Loss 0.20696 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20640 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21394</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20307</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20392</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19862</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18408</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17642</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18840</span></span>
<span class="line"><span>Validation: Loss 0.15176 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15144 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16435</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14796</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15781</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13945</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14118</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12911</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11438</span></span>
<span class="line"><span>Validation: Loss 0.11011 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11000 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12858</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11691</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11066</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09406</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09806</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09013</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09469</span></span>
<span class="line"><span>Validation: Loss 0.07858 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07857 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08926</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08194</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07845</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07024</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06963</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06261</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06569</span></span>
<span class="line"><span>Validation: Loss 0.05492 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05490 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05588</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06106</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05314</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05496</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04832</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04531</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04504</span></span>
<span class="line"><span>Validation: Loss 0.04101 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04096 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04245</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04303</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04039</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04299</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03709</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03847</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03740</span></span>
<span class="line"><span>Validation: Loss 0.03325 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03321 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03502</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03638</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03548</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03220</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03175</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03207</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02855</span></span>
<span class="line"><span>Validation: Loss 0.02820 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02816 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03196</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03134</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02825</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03009</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02764</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02594</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02112</span></span>
<span class="line"><span>Validation: Loss 0.02453 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02450 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02542</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02675</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02781</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02498</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02497</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02328</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02023</span></span>
<span class="line"><span>Validation: Loss 0.02172 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02169 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02173</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02518</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02282</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02205</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02231</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02158</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02110</span></span>
<span class="line"><span>Validation: Loss 0.01946 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01943 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02132</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02250</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01980</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02040</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01835</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01931</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02113</span></span>
<span class="line"><span>Validation: Loss 0.01757 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01755 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01860</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01819</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02129</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01852</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01811</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01542</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01988</span></span>
<span class="line"><span>Validation: Loss 0.01596 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01595 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01624</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01804</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01797</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01623</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01677</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01463</span></span>
<span class="line"><span>Validation: Loss 0.01458 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01456 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01502</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01573</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01539</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01517</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01612</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01453</span></span>
<span class="line"><span>Validation: Loss 0.01338 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01337 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01422</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01446</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01388</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01454</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01439</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01395</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01237</span></span>
<span class="line"><span>Validation: Loss 0.01233 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01232 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01311</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01429</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01194</span></span>
<span class="line"><span>Validation: Loss 0.01141 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01140 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01358</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01150</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01649</span></span>
<span class="line"><span>Validation: Loss 0.01055 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01055 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01181</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00839</span></span>
<span class="line"><span>Validation: Loss 0.00965 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00965 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01029</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01018</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00839</span></span>
<span class="line"><span>Validation: Loss 0.00867 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00868 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00934</span></span>
<span class="line"><span>Validation: Loss 0.00771 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00771 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00835</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00703</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00672</span></span>
<span class="line"><span>Validation: Loss 0.00697 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00697 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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

import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DZYPd6QR.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function h(e,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.0.1/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.0.1/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.0.1/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.0.1/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63363</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58107</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56740</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54022</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51598</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50952</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47641</span></span>
<span class="line"><span>Validation: Loss 0.46745 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47185 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46529</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46127</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45251</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42456</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40951</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39444</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37673</span></span>
<span class="line"><span>Validation: Loss 0.37011 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37545 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36862</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36597</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34827</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34519</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31592</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30139</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31022</span></span>
<span class="line"><span>Validation: Loss 0.28488 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29073 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28752</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26677</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27206</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25701</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25002</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23381</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23789</span></span>
<span class="line"><span>Validation: Loss 0.21455 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22029 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21389</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20979</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19148</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19602</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19112</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17698</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16302</span></span>
<span class="line"><span>Validation: Loss 0.15852 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16358 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16230</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14705</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14988</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13514</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14176</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13110</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12789</span></span>
<span class="line"><span>Validation: Loss 0.11559 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11961 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11763</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10926</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10829</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10551</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09851</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08757</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10091</span></span>
<span class="line"><span>Validation: Loss 0.08221 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08509 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09275</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08028</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07877</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07199</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06543</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05869</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05268</span></span>
<span class="line"><span>Validation: Loss 0.05708 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05897 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05457</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05363</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04984</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05663</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04950</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04771</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04671</span></span>
<span class="line"><span>Validation: Loss 0.04292 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04429 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04077</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04255</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04058</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03803</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03853</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04149</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03777</span></span>
<span class="line"><span>Validation: Loss 0.03493 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03606 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03434</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03377</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03296</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03385</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03444</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03208</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02881</span></span>
<span class="line"><span>Validation: Loss 0.02969 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03068 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03047</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03008</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02824</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02843</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02883</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02710</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02491</span></span>
<span class="line"><span>Validation: Loss 0.02586 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02674 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02587</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02448</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02584</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02408</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02560</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02540</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02372</span></span>
<span class="line"><span>Validation: Loss 0.02290 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02369 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02355</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02294</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02439</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02041</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02226</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02133</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02041</span></span>
<span class="line"><span>Validation: Loss 0.02050 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02123 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02183</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02021</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02045</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01869</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01932</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01994</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02126</span></span>
<span class="line"><span>Validation: Loss 0.01851 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01918 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01871</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01825</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01783</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01827</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01952</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01935</span></span>
<span class="line"><span>Validation: Loss 0.01682 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01744 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01635</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01703</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01601</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01643</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01731</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01647</span></span>
<span class="line"><span>Validation: Loss 0.01538 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01595 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01678</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01599</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01377</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01432</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01569</span></span>
<span class="line"><span>Validation: Loss 0.01413 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01466 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01679</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01354</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01308</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01291</span></span>
<span class="line"><span>Validation: Loss 0.01304 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01354 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01342</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01246</span></span>
<span class="line"><span>Validation: Loss 0.01208 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01254 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01122</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01133</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01236</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01061</span></span>
<span class="line"><span>Validation: Loss 0.01118 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01160 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01060</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01126</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00975</span></span>
<span class="line"><span>Validation: Loss 0.01024 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01062 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00994</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00955</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00844</span></span>
<span class="line"><span>Validation: Loss 0.00919 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00952 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00930</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00689</span></span>
<span class="line"><span>Validation: Loss 0.00817 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00845 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00835</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00598</span></span>
<span class="line"><span>Validation: Loss 0.00740 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00766 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.60294</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58756</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57470</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54895</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52199</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49149</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48506</span></span>
<span class="line"><span>Validation: Loss 0.48276 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47895 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47011</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45964</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43511</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42398</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41160</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38680</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40415</span></span>
<span class="line"><span>Validation: Loss 0.38807 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38384 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36732</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34996</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34527</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33681</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31621</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32185</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30775</span></span>
<span class="line"><span>Validation: Loss 0.30521 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.30071 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29681</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27822</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26056</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25291</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23603</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24225</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22994</span></span>
<span class="line"><span>Validation: Loss 0.23551 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.23110 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21795</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21422</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20792</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19499</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17770</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17183</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16068</span></span>
<span class="line"><span>Validation: Loss 0.17835 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17435 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16007</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15957</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14442</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14187</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13934</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12529</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14995</span></span>
<span class="line"><span>Validation: Loss 0.13267 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12941 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11240</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10309</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10744</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11145</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10243</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10379</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08507</span></span>
<span class="line"><span>Validation: Loss 0.09567 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09324 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08379</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08044</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08166</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07579</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06885</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06613</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05985</span></span>
<span class="line"><span>Validation: Loss 0.06640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06474 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06068</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05652</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05485</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05233</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04971</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04684</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04336</span></span>
<span class="line"><span>Validation: Loss 0.04867 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04754 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04424</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04479</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03874</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03961</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04071</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03695</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03219</span></span>
<span class="line"><span>Validation: Loss 0.03922 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03833 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03558</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03376</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03523</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03155</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03110</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03231</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03356</span></span>
<span class="line"><span>Validation: Loss 0.03330 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03253 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03067</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02975</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03041</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02751</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02625</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02595</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03005</span></span>
<span class="line"><span>Validation: Loss 0.02897 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02829 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02238</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02684</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02452</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02701</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02408</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02390</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02650</span></span>
<span class="line"><span>Validation: Loss 0.02564 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02502 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02371</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02208</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02316</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02074</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02283</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02029</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02055</span></span>
<span class="line"><span>Validation: Loss 0.02294 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02238 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02121</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02013</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02057</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02009</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01830</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01877</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01847</span></span>
<span class="line"><span>Validation: Loss 0.02073 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02021 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01681</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01782</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01825</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01853</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01726</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01804</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02012</span></span>
<span class="line"><span>Validation: Loss 0.01888 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01840 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01707</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01651</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01774</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01724</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01593</span></span>
<span class="line"><span>Validation: Loss 0.01728 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01683 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01529</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01551</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01548</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01480</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01289</span></span>
<span class="line"><span>Validation: Loss 0.01590 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01549 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01486</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01357</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01424</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01461</span></span>
<span class="line"><span>Validation: Loss 0.01471 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01432 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01377</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01101</span></span>
<span class="line"><span>Validation: Loss 0.01363 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01327 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01349</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01187</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01055</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01207</span></span>
<span class="line"><span>Validation: Loss 0.01263 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01229 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01154</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01233</span></span>
<span class="line"><span>Validation: Loss 0.01158 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01127 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01168</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00993</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00844</span></span>
<span class="line"><span>Validation: Loss 0.01037 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01009 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00722</span></span>
<span class="line"><span>Validation: Loss 0.00916 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00893 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00845</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00741</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Validation: Loss 0.00824 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00804 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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

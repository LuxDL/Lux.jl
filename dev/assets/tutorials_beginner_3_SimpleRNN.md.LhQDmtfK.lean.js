import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.Cg02roKE.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##230&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61224</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59839</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56825</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54056</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51875</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49626</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47547</span></span>
<span class="line"><span>Validation: Loss 0.47234 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47135 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46921</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45312</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43362</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41934</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41250</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40465</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36605</span></span>
<span class="line"><span>Validation: Loss 0.37603 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37474 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37266</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35038</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34109</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34270</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31697</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30348</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30246</span></span>
<span class="line"><span>Validation: Loss 0.29223 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29090 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27721</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28109</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26379</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26772</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23971</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22962</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20398</span></span>
<span class="line"><span>Validation: Loss 0.22261 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22147 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20470</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21311</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19256</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19309</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19007</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16966</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18436</span></span>
<span class="line"><span>Validation: Loss 0.16670 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16590 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15487</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16371</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14789</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13918</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13460</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12096</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13676</span></span>
<span class="line"><span>Validation: Loss 0.12270 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12224 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12161</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11115</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11563</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09413</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09214</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10037</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07018</span></span>
<span class="line"><span>Validation: Loss 0.08798 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08777 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08746</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08297</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07599</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06778</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06807</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06705</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05588</span></span>
<span class="line"><span>Validation: Loss 0.06136 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06128 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05576</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05737</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05504</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05137</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04150</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05179</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04985</span></span>
<span class="line"><span>Validation: Loss 0.04505 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04493 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04221</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04580</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04102</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03873</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03420</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03706</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03715</span></span>
<span class="line"><span>Validation: Loss 0.03616 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03604 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03727</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03192</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03486</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03113</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03164</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02995</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02731</span></span>
<span class="line"><span>Validation: Loss 0.03059 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03048 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02689</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02805</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02769</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02781</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02865</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02854</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02446</span></span>
<span class="line"><span>Validation: Loss 0.02661 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02651 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02443</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02698</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02395</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02352</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02476</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02351</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01991</span></span>
<span class="line"><span>Validation: Loss 0.02352 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02343 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02411</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02080</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02392</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02129</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02004</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01992</span></span>
<span class="line"><span>Validation: Loss 0.02105 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02284</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01735</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01931</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01791</span></span>
<span class="line"><span>Validation: Loss 0.01900 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01893 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01899</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01872</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01831</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01694</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01621</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01680</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01438</span></span>
<span class="line"><span>Validation: Loss 0.01728 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01721 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01675</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01592</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01542</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01612</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01893</span></span>
<span class="line"><span>Validation: Loss 0.01582 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01576 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01641</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01605</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01482</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01465</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01324</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01091</span></span>
<span class="line"><span>Validation: Loss 0.01455 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01449 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01294</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01316</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01303</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01496</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01300</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01412</span></span>
<span class="line"><span>Validation: Loss 0.01347 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01342 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01247</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01167</span></span>
<span class="line"><span>Validation: Loss 0.01250 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01245 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01214</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01130</span></span>
<span class="line"><span>Validation: Loss 0.01160 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01155 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01100</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00799</span></span>
<span class="line"><span>Validation: Loss 0.01069 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01065 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00840</span></span>
<span class="line"><span>Validation: Loss 0.00971 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00937</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00878</span></span>
<span class="line"><span>Validation: Loss 0.00863 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00861 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00820</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00736</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00839</span></span>
<span class="line"><span>Validation: Loss 0.00773 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00770 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62413</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.61233</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56293</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53998</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51566</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48981</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47048</span></span>
<span class="line"><span>Validation: Loss 0.46709 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.48035 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47229</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44753</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44980</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42927</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40470</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38982</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38987</span></span>
<span class="line"><span>Validation: Loss 0.36986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38590 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36905</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36049</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34151</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33279</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31992</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31227</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29895</span></span>
<span class="line"><span>Validation: Loss 0.28462 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.30215 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27368</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27566</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26971</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26429</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24907</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22839</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21440</span></span>
<span class="line"><span>Validation: Loss 0.21459 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.23211 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21549</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21439</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19644</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18597</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18067</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17839</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16351</span></span>
<span class="line"><span>Validation: Loss 0.15905 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17519 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16012</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15676</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14560</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14280</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12607</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13410</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12361</span></span>
<span class="line"><span>Validation: Loss 0.11621 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12952 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11278</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10890</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10789</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10362</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09902</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09333</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09837</span></span>
<span class="line"><span>Validation: Loss 0.08284 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09262 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08576</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08026</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07265</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07269</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06902</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06214</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06579</span></span>
<span class="line"><span>Validation: Loss 0.05755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06398 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06162</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05385</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05143</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05008</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04929</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04658</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04520</span></span>
<span class="line"><span>Validation: Loss 0.04297 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04754 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04585</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03907</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03776</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04328</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03875</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03644</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03516</span></span>
<span class="line"><span>Validation: Loss 0.03483 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03856 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03510</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03422</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03453</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03287</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03161</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03009</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03279</span></span>
<span class="line"><span>Validation: Loss 0.02958 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03282 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03261</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03046</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02835</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02606</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02798</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02578</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02481</span></span>
<span class="line"><span>Validation: Loss 0.02573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02860 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02731</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02614</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02457</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02480</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02095</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02523</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02524</span></span>
<span class="line"><span>Validation: Loss 0.02277 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02536 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02339</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02300</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02242</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02169</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02251</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01930</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02335</span></span>
<span class="line"><span>Validation: Loss 0.02039 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02275 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01945</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01935</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02089</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02130</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02028</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01822</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01831</span></span>
<span class="line"><span>Validation: Loss 0.01841 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02058 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01806</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01935</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01849</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01824</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01734</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01729</span></span>
<span class="line"><span>Validation: Loss 0.01674 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01875 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01635</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01614</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01594</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01696</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01584</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01576</span></span>
<span class="line"><span>Validation: Loss 0.01530 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01717 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01470</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01506</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01473</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01454</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01522</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01617</span></span>
<span class="line"><span>Validation: Loss 0.01404 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01578 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01528</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01441</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01356</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01313</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01374</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01280</span></span>
<span class="line"><span>Validation: Loss 0.01291 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01454 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01288</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01217</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01128</span></span>
<span class="line"><span>Validation: Loss 0.01188 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01338 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01255</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01018</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01210</span></span>
<span class="line"><span>Validation: Loss 0.01085 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01220 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00995</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01084</span></span>
<span class="line"><span>Validation: Loss 0.00970 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01088 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00893</span></span>
<span class="line"><span>Validation: Loss 0.00860 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00961 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00899</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00886</span></span>
<span class="line"><span>Validation: Loss 0.00777 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00866 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00730</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00728</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Validation: Loss 0.00717 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00798 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.0</span></span>
<span class="line"><span>Commit 501a4f25c2b (2024-10-07 11:40 UTC)</span></span>
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
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.6, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.6.3</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.3+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.3+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.11.0</span></span>
<span class="line"><span>- LLVM: 16.0.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2 devices:</span></span>
<span class="line"><span>  0: Quadro RTX 5000 (sm_75, 14.921 GiB / 16.000 GiB available)</span></span>
<span class="line"><span>  1: Quadro RTX 5000 (sm_75, 15.556 GiB / 16.000 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};

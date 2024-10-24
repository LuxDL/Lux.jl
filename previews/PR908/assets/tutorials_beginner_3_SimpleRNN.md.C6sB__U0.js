import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BB8b_vP3.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR908/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR908/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR908/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR908/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61491</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58423</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56624</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53952</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52475</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49803</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48406</span></span>
<span class="line"><span>Validation: Loss 0.46978 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.48340 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46364</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45412</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44571</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42214</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40617</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40201</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36349</span></span>
<span class="line"><span>Validation: Loss 0.37288 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38870 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37061</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34872</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34332</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33674</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32395</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30443</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30854</span></span>
<span class="line"><span>Validation: Loss 0.28890 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.30672 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29515</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26033</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26939</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24992</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24233</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23925</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22205</span></span>
<span class="line"><span>Validation: Loss 0.21925 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.23700 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20720</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21132</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20044</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18786</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18820</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17376</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16400</span></span>
<span class="line"><span>Validation: Loss 0.16331 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17952 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16916</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15216</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12934</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13895</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14431</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12580</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13664</span></span>
<span class="line"><span>Validation: Loss 0.12011 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13331 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11168</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11151</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11220</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10757</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09185</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09151</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09940</span></span>
<span class="line"><span>Validation: Loss 0.08614 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09575 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08901</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07910</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08081</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06433</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06984</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06456</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05825</span></span>
<span class="line"><span>Validation: Loss 0.05980 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06613 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05821</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05156</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05289</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05287</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04668</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04937</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04665</span></span>
<span class="line"><span>Validation: Loss 0.04398 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04848 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04431</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03845</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04196</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03506</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03761</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04067</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03611</span></span>
<span class="line"><span>Validation: Loss 0.03542 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03910 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03454</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03531</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03202</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03124</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03050</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03261</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02842</span></span>
<span class="line"><span>Validation: Loss 0.02998 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03319 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03023</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02765</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02731</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02793</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02830</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02593</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02632</span></span>
<span class="line"><span>Validation: Loss 0.02606 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02892 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02300</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02580</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02489</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02490</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02546</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02137</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02646</span></span>
<span class="line"><span>Validation: Loss 0.02303 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02562 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02283</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02458</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02128</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02039</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01988</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02121</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01924</span></span>
<span class="line"><span>Validation: Loss 0.02057 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02294 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02037</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01956</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02030</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01948</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01850</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01823</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01798</span></span>
<span class="line"><span>Validation: Loss 0.01856 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02075 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01806</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01713</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01754</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01844</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01685</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01757</span></span>
<span class="line"><span>Validation: Loss 0.01687 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01891 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01532</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01772</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01627</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01577</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01497</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01430</span></span>
<span class="line"><span>Validation: Loss 0.01544 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01734 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01406</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01470</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01534</span></span>
<span class="line"><span>Validation: Loss 0.01423 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01600 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01300</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01420</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01362</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01384</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01267</span></span>
<span class="line"><span>Validation: Loss 0.01317 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01483 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01289</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01239</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01079</span></span>
<span class="line"><span>Validation: Loss 0.01223 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01140</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01208</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01272</span></span>
<span class="line"><span>Validation: Loss 0.01139 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01283 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01082</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01090</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01056</span></span>
<span class="line"><span>Validation: Loss 0.01057 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01190 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00898</span></span>
<span class="line"><span>Validation: Loss 0.00971 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01091 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00876</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00980</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00914</span></span>
<span class="line"><span>Validation: Loss 0.00874 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00978 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00835</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00785</span></span>
<span class="line"><span>Validation: Loss 0.00778 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00867 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61790</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59421</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56931</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53677</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51776</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49078</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49102</span></span>
<span class="line"><span>Validation: Loss 0.47731 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46988 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46288</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44483</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43890</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42637</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40994</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40088</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38376</span></span>
<span class="line"><span>Validation: Loss 0.38187 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37284 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36644</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34889</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34618</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33898</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31021</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31073</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29141</span></span>
<span class="line"><span>Validation: Loss 0.29786 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28788 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29343</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26626</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25433</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25218</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24175</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23096</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22924</span></span>
<span class="line"><span>Validation: Loss 0.22791 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21783 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22179</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21040</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18919</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18671</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17098</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17417</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16264</span></span>
<span class="line"><span>Validation: Loss 0.17118 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16198 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16445</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15263</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14397</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13413</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13374</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12614</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10184</span></span>
<span class="line"><span>Validation: Loss 0.12651 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11896 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11680</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10798</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10416</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10196</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09471</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09268</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08812</span></span>
<span class="line"><span>Validation: Loss 0.09111 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08546 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08142</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07916</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07517</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07291</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06836</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06189</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06166</span></span>
<span class="line"><span>Validation: Loss 0.06324 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05946 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05998</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05294</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05261</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05274</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04581</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04668</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03904</span></span>
<span class="line"><span>Validation: Loss 0.04647 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04388 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04530</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04012</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04235</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03535</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03794</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03628</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03237</span></span>
<span class="line"><span>Validation: Loss 0.03748 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03541 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03685</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03237</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03424</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03265</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02929</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02937</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02921</span></span>
<span class="line"><span>Validation: Loss 0.03182 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03003 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02957</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03018</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02648</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02732</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02588</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02738</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02441</span></span>
<span class="line"><span>Validation: Loss 0.02772 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02612 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02725</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02482</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02310</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02268</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02414</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02332</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02390</span></span>
<span class="line"><span>Validation: Loss 0.02454 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02309 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02174</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02178</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02046</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02125</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02111</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02302</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01916</span></span>
<span class="line"><span>Validation: Loss 0.02198 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02066 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01886</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02020</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01937</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02100</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01759</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01853</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01926</span></span>
<span class="line"><span>Validation: Loss 0.01986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01864 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01850</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01823</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01912</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01661</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01685</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01811</span></span>
<span class="line"><span>Validation: Loss 0.01806 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01693 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01602</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01643</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01546</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01566</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01459</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01425</span></span>
<span class="line"><span>Validation: Loss 0.01655 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01549 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01479</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01515</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01437</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01475</span></span>
<span class="line"><span>Validation: Loss 0.01526 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01427 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01443</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01467</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01319</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01237</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01059</span></span>
<span class="line"><span>Validation: Loss 0.01413 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01321 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01336</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01345</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01125</span></span>
<span class="line"><span>Validation: Loss 0.01314 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01228 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01265</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01183</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01114</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01267</span></span>
<span class="line"><span>Validation: Loss 0.01224 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01144 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01177</span></span>
<span class="line"><span>Validation: Loss 0.01137 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01063 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01047</span></span>
<span class="line"><span>Validation: Loss 0.01046 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00977 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00919</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00781</span></span>
<span class="line"><span>Validation: Loss 0.00941 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00899</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00845</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00725</span></span>
<span class="line"><span>Validation: Loss 0.00836 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00784 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};

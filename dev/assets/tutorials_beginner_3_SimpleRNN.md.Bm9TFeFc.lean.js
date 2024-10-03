import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.CZ4L7dST.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.60722</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58440</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58100</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54911</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51710</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50018</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49053</span></span>
<span class="line"><span>Validation: Loss 0.46199 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.49707 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47159</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46456</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44336</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42007</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41171</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39815</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36684</span></span>
<span class="line"><span>Validation: Loss 0.36529 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.40374 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36191</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36769</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35300</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32958</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31822</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32191</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30201</span></span>
<span class="line"><span>Validation: Loss 0.28095 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.32266 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28751</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28925</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27703</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24959</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24578</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23550</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21775</span></span>
<span class="line"><span>Validation: Loss 0.21183 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25287 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22290</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21317</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19582</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19554</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17962</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18605</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18406</span></span>
<span class="line"><span>Validation: Loss 0.15677 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.19432 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16313</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14862</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15596</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13971</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14090</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14061</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12828</span></span>
<span class="line"><span>Validation: Loss 0.11467 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14551 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11370</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11697</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10871</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10562</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09908</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10390</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09867</span></span>
<span class="line"><span>Validation: Loss 0.08219 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10502 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09189</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08264</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08002</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07064</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06501</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07081</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06886</span></span>
<span class="line"><span>Validation: Loss 0.05719 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07234 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06419</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05609</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05607</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05572</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04361</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04868</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04521</span></span>
<span class="line"><span>Validation: Loss 0.04218 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05272 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04662</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04365</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04112</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03744</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03794</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04069</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03219</span></span>
<span class="line"><span>Validation: Loss 0.03396 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04251 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03633</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03429</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03227</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03374</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03239</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03164</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03661</span></span>
<span class="line"><span>Validation: Loss 0.02872 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03616 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03055</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02682</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02979</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02818</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02837</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02806</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03084</span></span>
<span class="line"><span>Validation: Loss 0.02490 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03151 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02561</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02443</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02692</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02513</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02508</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02274</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02809</span></span>
<span class="line"><span>Validation: Loss 0.02196 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02790 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02301</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02138</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02429</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02271</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02212</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02042</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02124</span></span>
<span class="line"><span>Validation: Loss 0.01960 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02500 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02037</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02006</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02069</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01947</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02044</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01924</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01843</span></span>
<span class="line"><span>Validation: Loss 0.01766 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02263 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01852</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01846</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01819</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01947</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01770</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01682</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01542</span></span>
<span class="line"><span>Validation: Loss 0.01603 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02063 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01765</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01630</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01481</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01655</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01469</span></span>
<span class="line"><span>Validation: Loss 0.01464 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01894 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01661</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01549</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01339</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01543</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01467</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01492</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01517</span></span>
<span class="line"><span>Validation: Loss 0.01344 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01746 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01437</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01325</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01392</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01362</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01079</span></span>
<span class="line"><span>Validation: Loss 0.01239 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01615 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01358</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01339</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01247</span></span>
<span class="line"><span>Validation: Loss 0.01145 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01497 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01233</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01190</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01222</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01008</span></span>
<span class="line"><span>Validation: Loss 0.01056 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01380 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01224</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01094</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00966</span></span>
<span class="line"><span>Validation: Loss 0.00960 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01252 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00978</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00884</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00972</span></span>
<span class="line"><span>Validation: Loss 0.00856 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01108 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00964</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00771</span></span>
<span class="line"><span>Validation: Loss 0.00763 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00978 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00811</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00835</span></span>
<span class="line"><span>Validation: Loss 0.00696 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00888 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61452</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59444</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56803</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53794</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51557</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50351</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47547</span></span>
<span class="line"><span>Validation: Loss 0.46657 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47851 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46900</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45620</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42628</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42479</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40709</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40254</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39721</span></span>
<span class="line"><span>Validation: Loss 0.36884 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38301 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36593</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35614</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34504</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32370</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32273</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31495</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29725</span></span>
<span class="line"><span>Validation: Loss 0.28379 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29918 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28744</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27778</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26189</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24392</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23824</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24195</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21870</span></span>
<span class="line"><span>Validation: Loss 0.21386 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22921 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20348</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21080</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19691</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19774</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17330</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17967</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16232</span></span>
<span class="line"><span>Validation: Loss 0.15853 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17260 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15718</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15880</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14885</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13708</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12893</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13282</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09764</span></span>
<span class="line"><span>Validation: Loss 0.11617 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12768 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11550</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11020</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10235</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10739</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09667</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08803</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10376</span></span>
<span class="line"><span>Validation: Loss 0.08341 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09197 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08571</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07857</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07732</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06960</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06742</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06498</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05920</span></span>
<span class="line"><span>Validation: Loss 0.05801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06369 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05955</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04993</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05257</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05204</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04905</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04856</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04058</span></span>
<span class="line"><span>Validation: Loss 0.04286 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04685 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04012</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04176</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03746</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04041</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03796</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03776</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04280</span></span>
<span class="line"><span>Validation: Loss 0.03458 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03784 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03318</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03315</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03421</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02971</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03184</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03341</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02922</span></span>
<span class="line"><span>Validation: Loss 0.02927 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03209 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03003</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02867</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02759</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02727</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02626</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02754</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02525</span></span>
<span class="line"><span>Validation: Loss 0.02542 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02792 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02521</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02431</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02326</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02532</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02326</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02465</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02307</span></span>
<span class="line"><span>Validation: Loss 0.02246 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02472 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02226</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02209</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02174</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02120</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02216</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02129</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01623</span></span>
<span class="line"><span>Validation: Loss 0.02008 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02215 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02027</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02031</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02106</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01902</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01695</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01520</span></span>
<span class="line"><span>Validation: Loss 0.01811 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02003 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01862</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01803</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01745</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01738</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01714</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01707</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01501</span></span>
<span class="line"><span>Validation: Loss 0.01647 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01826 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01611</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01559</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01616</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01625</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01635</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01660</span></span>
<span class="line"><span>Validation: Loss 0.01508 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01676 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01407</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01604</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01430</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01441</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01466</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01518</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01147</span></span>
<span class="line"><span>Validation: Loss 0.01389 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01545 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01371</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01329</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01568</span></span>
<span class="line"><span>Validation: Loss 0.01286 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01432 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01362</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01440</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01265</span></span>
<span class="line"><span>Validation: Loss 0.01193 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01329 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01179</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01190</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01086</span></span>
<span class="line"><span>Validation: Loss 0.01108 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01235 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01094</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01029</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01127</span></span>
<span class="line"><span>Validation: Loss 0.01026 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01143 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00850</span></span>
<span class="line"><span>Validation: Loss 0.00938 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01043 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00899</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00930</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00927</span></span>
<span class="line"><span>Validation: Loss 0.00840 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00931 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00769</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00713</span></span>
<span class="line"><span>Validation: Loss 0.00749 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00827 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- Julia: 1.10.5</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};

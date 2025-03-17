import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DGp6rkXP.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.2.2/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.2.2/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.2.2/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.2.2/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62418</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58579</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56593</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54239</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51930</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50690</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49584</span></span>
<span class="line"><span>Validation: Loss 0.46165 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46952 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47788</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45420</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43943</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42512</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40691</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40282</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38509</span></span>
<span class="line"><span>Validation: Loss 0.36373 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37269 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37104</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38009</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34337</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32877</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31390</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31147</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29863</span></span>
<span class="line"><span>Validation: Loss 0.27834 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28801 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29062</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27988</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26162</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25111</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25346</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23974</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21767</span></span>
<span class="line"><span>Validation: Loss 0.20848 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21814 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20867</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21213</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19058</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19469</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19829</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18660</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15299</span></span>
<span class="line"><span>Validation: Loss 0.15361 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16244 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16677</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16597</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14248</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14270</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13129</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13175</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13294</span></span>
<span class="line"><span>Validation: Loss 0.11233 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11952 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11907</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10307</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11472</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10291</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10372</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10140</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09050</span></span>
<span class="line"><span>Validation: Loss 0.08060 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08589 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08299</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08522</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08371</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07357</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06734</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06675</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06284</span></span>
<span class="line"><span>Validation: Loss 0.05620 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05972 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06122</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05772</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05625</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05107</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04973</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04731</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04114</span></span>
<span class="line"><span>Validation: Loss 0.04140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04387 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04597</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04468</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04027</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03766</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03869</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03671</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03978</span></span>
<span class="line"><span>Validation: Loss 0.03330 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03531 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03641</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03406</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03407</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03293</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03303</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03120</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02747</span></span>
<span class="line"><span>Validation: Loss 0.02813 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02988 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03035</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02948</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02886</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02897</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02867</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02558</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02645</span></span>
<span class="line"><span>Validation: Loss 0.02441 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02597 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02640</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02505</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02586</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02413</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02457</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02392</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02495</span></span>
<span class="line"><span>Validation: Loss 0.02154 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02295 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02393</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02394</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02140</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02144</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02072</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02172</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02228</span></span>
<span class="line"><span>Validation: Loss 0.01923 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02051 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02086</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02128</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01953</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02066</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01842</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01920</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01809</span></span>
<span class="line"><span>Validation: Loss 0.01732 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01850 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01951</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01857</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01764</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01806</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01846</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01676</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01449</span></span>
<span class="line"><span>Validation: Loss 0.01571 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01681 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01790</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01653</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01615</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01641</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01455</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01555</span></span>
<span class="line"><span>Validation: Loss 0.01435 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01504</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01486</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01653</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01422</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01420</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01710</span></span>
<span class="line"><span>Validation: Loss 0.01319 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01415 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01415</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01302</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01433</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01432</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01416</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01478</span></span>
<span class="line"><span>Validation: Loss 0.01219 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01308 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01346</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01252</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01251</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01081</span></span>
<span class="line"><span>Validation: Loss 0.01130 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01213 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01128</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01175</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01185</span></span>
<span class="line"><span>Validation: Loss 0.01049 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01126 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01004</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01071</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00929</span></span>
<span class="line"><span>Validation: Loss 0.00970 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01041 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01052</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00873</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00930</span></span>
<span class="line"><span>Validation: Loss 0.00885 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00948 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00732</span></span>
<span class="line"><span>Validation: Loss 0.00790 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00844 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00744</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00710</span></span>
<span class="line"><span>Validation: Loss 0.00707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00754 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61648</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60569</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55924</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53742</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51968</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50163</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48797</span></span>
<span class="line"><span>Validation: Loss 0.46968 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46657 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47505</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45209</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43445</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43346</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40767</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39600</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38576</span></span>
<span class="line"><span>Validation: Loss 0.37245 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36874 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36002</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36527</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33697</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34172</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32651</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30933</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29770</span></span>
<span class="line"><span>Validation: Loss 0.28741 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28346 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27948</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27427</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25578</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26128</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26132</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22517</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24097</span></span>
<span class="line"><span>Validation: Loss 0.21734 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21362 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21696</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20913</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19235</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19339</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18199</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18324</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14668</span></span>
<span class="line"><span>Validation: Loss 0.16142 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15829 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16548</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16349</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14657</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14099</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12364</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13009</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11028</span></span>
<span class="line"><span>Validation: Loss 0.11837 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11600 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10952</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11348</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11021</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10278</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09154</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10022</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09825</span></span>
<span class="line"><span>Validation: Loss 0.08504 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08340 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08598</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08123</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07999</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06669</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07046</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06645</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05322</span></span>
<span class="line"><span>Validation: Loss 0.05916 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05808 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06442</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05365</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05147</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05029</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04758</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04685</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04956</span></span>
<span class="line"><span>Validation: Loss 0.04375 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04287 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04564</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04405</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03796</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03963</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03850</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03521</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03610</span></span>
<span class="line"><span>Validation: Loss 0.03532 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03456 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03507</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03105</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03418</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03227</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03260</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03136</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03553</span></span>
<span class="line"><span>Validation: Loss 0.02996 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02930 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03098</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03034</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02851</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02817</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02666</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02503</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02634</span></span>
<span class="line"><span>Validation: Loss 0.02603 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02544 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02473</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02426</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02448</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02658</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02259</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02529</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02406</span></span>
<span class="line"><span>Validation: Loss 0.02302 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02249 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02143</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02288</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02464</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02272</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01895</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02051</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02293</span></span>
<span class="line"><span>Validation: Loss 0.02060 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02011 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02189</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01995</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01945</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02023</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01856</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01759</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02073</span></span>
<span class="line"><span>Validation: Loss 0.01858 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01813 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01992</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01900</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01808</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01701</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01582</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01750</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01509</span></span>
<span class="line"><span>Validation: Loss 0.01688 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01646 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01581</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01652</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01684</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01589</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01615</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01434</span></span>
<span class="line"><span>Validation: Loss 0.01544 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01506 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01574</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01368</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01452</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01589</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01487</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01538</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01043</span></span>
<span class="line"><span>Validation: Loss 0.01421 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01385 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01446</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01391</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01384</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01473</span></span>
<span class="line"><span>Validation: Loss 0.01314 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01281 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01289</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01313</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01236</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01370</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01173</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01218</span></span>
<span class="line"><span>Validation: Loss 0.01217 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01187 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01133</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01086</span></span>
<span class="line"><span>Validation: Loss 0.01125 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01191</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00967</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00951</span></span>
<span class="line"><span>Validation: Loss 0.01027 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01003 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00996</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00890</span></span>
<span class="line"><span>Validation: Loss 0.00918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00899 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00876</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00889</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00889</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00845</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00914</span></span>
<span class="line"><span>Validation: Loss 0.00816 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00799 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00762</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00899</span></span>
<span class="line"><span>Validation: Loss 0.00740 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00725 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.6</span></span>
<span class="line"><span>Commit 67dffc4a8ae (2024-10-28 12:23 UTC)</span></span>
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
<span class="line"><span>CUDA driver 12.6</span></span>
<span class="line"><span>NVIDIA driver 560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.6.3</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.3+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.3+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.6</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};

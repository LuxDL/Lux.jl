import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.H3Pxo4Yr.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63319</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59393</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57072</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54135</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51537</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49392</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48809</span></span>
<span class="line"><span>Validation: Loss 0.46265 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47218 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47357</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45490</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44293</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42035</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41429</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39933</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36456</span></span>
<span class="line"><span>Validation: Loss 0.36376 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37475 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37654</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36986</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35627</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32722</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31184</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30162</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.27532</span></span>
<span class="line"><span>Validation: Loss 0.27783 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29010 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28801</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27127</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26407</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26110</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24018</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24419</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20260</span></span>
<span class="line"><span>Validation: Loss 0.20812 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22083 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21453</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20476</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19651</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19452</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18578</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17929</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17903</span></span>
<span class="line"><span>Validation: Loss 0.15362 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16552 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17298</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15294</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15183</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14500</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12692</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12735</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11935</span></span>
<span class="line"><span>Validation: Loss 0.11205 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12185 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11833</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09899</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10212</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11255</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10092</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09958</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10254</span></span>
<span class="line"><span>Validation: Loss 0.08021 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08751 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08249</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08225</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07170</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07924</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07150</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06319</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06671</span></span>
<span class="line"><span>Validation: Loss 0.05588 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06071 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05893</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05670</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05395</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05145</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05209</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04614</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03778</span></span>
<span class="line"><span>Validation: Loss 0.04142 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04477 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04748</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03993</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04165</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03853</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03826</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03746</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03377</span></span>
<span class="line"><span>Validation: Loss 0.03346 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03618 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03605</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03318</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03314</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03347</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03405</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03117</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02544</span></span>
<span class="line"><span>Validation: Loss 0.02835 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03073 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02987</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02899</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02719</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02953</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02706</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02769</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03016</span></span>
<span class="line"><span>Validation: Loss 0.02467 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02679 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02692</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02532</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02658</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02527</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02567</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02109</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02089</span></span>
<span class="line"><span>Validation: Loss 0.02179 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02370 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02441</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02203</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02266</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02092</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02272</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02105</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01938</span></span>
<span class="line"><span>Validation: Loss 0.01949 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02124 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02152</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02010</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02037</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02076</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01941</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01722</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02094</span></span>
<span class="line"><span>Validation: Loss 0.01758 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01920 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01894</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01802</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02042</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01784</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01726</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01525</span></span>
<span class="line"><span>Validation: Loss 0.01596 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01746 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01613</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01545</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01756</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01652</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01677</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01507</span></span>
<span class="line"><span>Validation: Loss 0.01459 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01599 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01557</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01480</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01492</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01390</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01627</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01497</span></span>
<span class="line"><span>Validation: Loss 0.01340 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01472 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01450</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01557</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01281</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01346</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01371</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01121</span></span>
<span class="line"><span>Validation: Loss 0.01237 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01360 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01392</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01135</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01286</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01234</span></span>
<span class="line"><span>Validation: Loss 0.01147 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01262 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01217</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01227</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01189</span></span>
<span class="line"><span>Validation: Loss 0.01065 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01172 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01133</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01010</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01186</span></span>
<span class="line"><span>Validation: Loss 0.00985 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01084 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01017</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01023</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01060</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01121</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00817</span></span>
<span class="line"><span>Validation: Loss 0.00898 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00986 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00919</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01022</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00930</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00703</span></span>
<span class="line"><span>Validation: Loss 0.00803 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00879 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00813</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00745</span></span>
<span class="line"><span>Validation: Loss 0.00720 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00785 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61958</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58894</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57488</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53939</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51395</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49772</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50451</span></span>
<span class="line"><span>Validation: Loss 0.46954 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46901 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46639</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45009</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44407</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42564</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41331</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40562</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36186</span></span>
<span class="line"><span>Validation: Loss 0.37308 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37232 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37315</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36342</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33747</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33466</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31260</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31985</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30922</span></span>
<span class="line"><span>Validation: Loss 0.28910 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28799 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27143</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27561</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25191</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26373</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25626</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25540</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21676</span></span>
<span class="line"><span>Validation: Loss 0.21997 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21510</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20196</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20299</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19098</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18306</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18473</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19814</span></span>
<span class="line"><span>Validation: Loss 0.16449 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16313 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17904</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15928</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15115</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13397</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12892</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12915</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13066</span></span>
<span class="line"><span>Validation: Loss 0.12098 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11978 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12747</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12187</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10482</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10755</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08709</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09507</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09342</span></span>
<span class="line"><span>Validation: Loss 0.08671 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08580 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08749</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08399</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07920</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07393</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06196</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06892</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06528</span></span>
<span class="line"><span>Validation: Loss 0.06024 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05966 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05740</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05590</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05480</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05277</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05005</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04776</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04887</span></span>
<span class="line"><span>Validation: Loss 0.04458 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04419 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04644</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04308</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04106</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04145</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03600</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03710</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03676</span></span>
<span class="line"><span>Validation: Loss 0.03602 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03571 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03366</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03371</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03486</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03342</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03323</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03250</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03097</span></span>
<span class="line"><span>Validation: Loss 0.03056 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03029 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03265</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02942</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03081</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02580</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02938</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02571</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02311</span></span>
<span class="line"><span>Validation: Loss 0.02658 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02634 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02658</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02634</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02585</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02520</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02455</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02285</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02289</span></span>
<span class="line"><span>Validation: Loss 0.02353 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02331 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02314</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02215</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02157</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02283</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02168</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02231</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02353</span></span>
<span class="line"><span>Validation: Loss 0.02108 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02088 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02134</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01859</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02145</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01805</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01888</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01988</span></span>
<span class="line"><span>Validation: Loss 0.01903 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01884 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01825</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01747</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01946</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01820</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01711</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01802</span></span>
<span class="line"><span>Validation: Loss 0.01730 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01713 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01651</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01660</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01623</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01759</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01772</span></span>
<span class="line"><span>Validation: Loss 0.01583 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01567 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01598</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01489</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01575</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01586</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01519</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01400</span></span>
<span class="line"><span>Validation: Loss 0.01455 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01440 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01470</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01440</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01437</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01350</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01391</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01240</span></span>
<span class="line"><span>Validation: Loss 0.01345 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01331 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01250</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01200</span></span>
<span class="line"><span>Validation: Loss 0.01248 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01235 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01271</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01160</span></span>
<span class="line"><span>Validation: Loss 0.01158 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01146 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01171</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01065</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01164</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01028</span></span>
<span class="line"><span>Validation: Loss 0.01069 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01057 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01016</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01027</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00963</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01016</span></span>
<span class="line"><span>Validation: Loss 0.00970 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00960 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01022</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00894</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00798</span></span>
<span class="line"><span>Validation: Loss 0.00862 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00853 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00663</span></span>
<span class="line"><span>Validation: Loss 0.00772 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00764 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};

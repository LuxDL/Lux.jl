import{_ as a,o as n,c as i,a2 as p}from"./chunks/framework.ZBMMAXEM.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return n(),i("div",null,s[0]||(s[0]=[p(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.3.3/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.3.3/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.3.3/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.3.3/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62228</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58571</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56973</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54320</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52264</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49863</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50773</span></span>
<span class="line"><span>Validation: Loss 0.46027 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47027 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47163</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45379</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44359</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43374</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39947</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40825</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38567</span></span>
<span class="line"><span>Validation: Loss 0.36253 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37395 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37715</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35866</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35465</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33274</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32503</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31172</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28375</span></span>
<span class="line"><span>Validation: Loss 0.27681 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28966 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29246</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27398</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25867</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26764</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25732</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23090</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23431</span></span>
<span class="line"><span>Validation: Loss 0.20692 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22037 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22287</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20974</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20163</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19467</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18014</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18608</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17436</span></span>
<span class="line"><span>Validation: Loss 0.15213 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16489 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16569</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15666</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15815</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13737</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13432</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14018</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11618</span></span>
<span class="line"><span>Validation: Loss 0.11088 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12165 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12116</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12185</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11245</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10089</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10328</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09578</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07751</span></span>
<span class="line"><span>Validation: Loss 0.07946 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08757 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08631</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07977</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07880</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07758</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06987</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06798</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07269</span></span>
<span class="line"><span>Validation: Loss 0.05564 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06111 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06551</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05592</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05594</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05511</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05059</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04331</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04586</span></span>
<span class="line"><span>Validation: Loss 0.04106 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04490 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04347</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04279</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04173</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04170</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03853</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04025</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03299</span></span>
<span class="line"><span>Validation: Loss 0.03307 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03621 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03841</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03623</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03336</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03336</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03205</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03028</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03195</span></span>
<span class="line"><span>Validation: Loss 0.02795 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03069 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03101</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03167</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02849</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02917</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02664</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02826</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02401</span></span>
<span class="line"><span>Validation: Loss 0.02426 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02670 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02837</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02402</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02581</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02668</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02360</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02448</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02282</span></span>
<span class="line"><span>Validation: Loss 0.02141 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02362 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02389</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02247</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02272</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02337</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02260</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02167</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01731</span></span>
<span class="line"><span>Validation: Loss 0.01914 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02116 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02064</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02031</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02182</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01812</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02050</span></span>
<span class="line"><span>Validation: Loss 0.01726 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01913 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01952</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01856</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01918</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01822</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01782</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01707</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01800</span></span>
<span class="line"><span>Validation: Loss 0.01566 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01739 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01710</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01705</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01684</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01678</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01609</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01646</span></span>
<span class="line"><span>Validation: Loss 0.01429 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01591 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01580</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01545</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01417</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01573</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01572</span></span>
<span class="line"><span>Validation: Loss 0.01311 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01463 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01424</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01548</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01387</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01609</span></span>
<span class="line"><span>Validation: Loss 0.01209 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01350 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01353</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01413</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01271</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01241</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01193</span></span>
<span class="line"><span>Validation: Loss 0.01117 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01248 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01276</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01187</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00963</span></span>
<span class="line"><span>Validation: Loss 0.01031 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01153 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01066</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01131</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01095</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01006</span></span>
<span class="line"><span>Validation: Loss 0.00943 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01053 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00964</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01027</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00980</span></span>
<span class="line"><span>Validation: Loss 0.00844 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00939 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00922</span></span>
<span class="line"><span>Validation: Loss 0.00750 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00831 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00719</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00914</span></span>
<span class="line"><span>Validation: Loss 0.00682 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00753 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61509</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60108</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55682</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54901</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49920</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51514</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51064</span></span>
<span class="line"><span>Validation: Loss 0.46906 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46767 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47772</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45928</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43215</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42920</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41722</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39954</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38827</span></span>
<span class="line"><span>Validation: Loss 0.37399 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37236 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36867</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35577</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34842</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34695</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32450</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31801</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32518</span></span>
<span class="line"><span>Validation: Loss 0.28976 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28805 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28978</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27647</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26276</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25823</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25967</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24144</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25356</span></span>
<span class="line"><span>Validation: Loss 0.22011 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21852 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22836</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21406</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20441</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19362</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18782</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17750</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17369</span></span>
<span class="line"><span>Validation: Loss 0.16410 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16279 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17219</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15211</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15563</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14199</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14171</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13298</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12676</span></span>
<span class="line"><span>Validation: Loss 0.12054 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11958 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12465</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11327</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10910</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09648</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10936</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10177</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08885</span></span>
<span class="line"><span>Validation: Loss 0.08635 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08568 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09532</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08611</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07121</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07892</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06568</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06907</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05675</span></span>
<span class="line"><span>Validation: Loss 0.05999 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05956 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05736</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05612</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05023</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05699</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05182</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05200</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04850</span></span>
<span class="line"><span>Validation: Loss 0.04473 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04442 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04791</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04287</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04373</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03918</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03873</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03829</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04021</span></span>
<span class="line"><span>Validation: Loss 0.03626 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03600 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03885</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03412</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03265</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03425</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03439</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03421</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02723</span></span>
<span class="line"><span>Validation: Loss 0.03079 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03057 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03052</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03016</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02903</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02899</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03049</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02760</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03159</span></span>
<span class="line"><span>Validation: Loss 0.02682 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02662 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.03079</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02633</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02706</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02541</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02263</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02406</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02344</span></span>
<span class="line"><span>Validation: Loss 0.02371 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02353 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02772</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02266</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02230</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02132</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02350</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02066</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02339</span></span>
<span class="line"><span>Validation: Loss 0.02123 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02107 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02165</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02132</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02082</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02090</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01924</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01966</span></span>
<span class="line"><span>Validation: Loss 0.01918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01903 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02048</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01983</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01884</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01768</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01726</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01914</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01576</span></span>
<span class="line"><span>Validation: Loss 0.01743 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01729 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01949</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01637</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01750</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01575</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01605</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01714</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01758</span></span>
<span class="line"><span>Validation: Loss 0.01592 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01579 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01689</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01667</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01582</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01393</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01568</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01611</span></span>
<span class="line"><span>Validation: Loss 0.01456 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01444 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01441</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01467</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01491</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01454</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01197</span></span>
<span class="line"><span>Validation: Loss 0.01328 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01318 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01368</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01418</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01122</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01230</span></span>
<span class="line"><span>Validation: Loss 0.01197 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01188 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01184</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01250</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01107</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01045</span></span>
<span class="line"><span>Validation: Loss 0.01055 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01047 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01101</span></span>
<span class="line"><span>Validation: Loss 0.00932 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00925 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00994</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00873</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00933</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00895</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00866</span></span>
<span class="line"><span>Validation: Loss 0.00844 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00838 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00738</span></span>
<span class="line"><span>Validation: Loss 0.00781 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00775 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00760</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00745</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00725</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00717</span></span>
<span class="line"><span>Validation: Loss 0.00732 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00726 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- CUDA_Runtime_jll: 0.15.4+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.6</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.422 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};

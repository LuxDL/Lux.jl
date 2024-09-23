import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DJ5UDqQb.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR902/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR902/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR902/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR902/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63006</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59919</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56212</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54347</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51502</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49814</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47976</span></span>
<span class="line"><span>Validation: Loss 0.46461 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46686 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46215</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45582</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44483</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44264</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40212</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38583</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40885</span></span>
<span class="line"><span>Validation: Loss 0.36648 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36900 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38259</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35660</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33686</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32916</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31813</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31293</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31901</span></span>
<span class="line"><span>Validation: Loss 0.28153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28389 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28908</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28880</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25648</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26070</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24039</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23153</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22024</span></span>
<span class="line"><span>Validation: Loss 0.21197 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21395 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20815</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22124</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20864</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19342</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17806</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16787</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17373</span></span>
<span class="line"><span>Validation: Loss 0.15708 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15858 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15885</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15658</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14186</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13888</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13535</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14058</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12788</span></span>
<span class="line"><span>Validation: Loss 0.11511 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11609 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12241</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11597</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10010</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10197</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10616</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09254</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08235</span></span>
<span class="line"><span>Validation: Loss 0.08231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08287 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08304</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08333</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07563</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07331</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06809</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06771</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05969</span></span>
<span class="line"><span>Validation: Loss 0.05740 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05779 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05847</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05758</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05262</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05035</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04999</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04808</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04875</span></span>
<span class="line"><span>Validation: Loss 0.04281 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04319 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04466</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04103</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04075</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04051</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03864</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03848</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03904</span></span>
<span class="line"><span>Validation: Loss 0.03467 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03502 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03709</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03468</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03615</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03385</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03035</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02939</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03406</span></span>
<span class="line"><span>Validation: Loss 0.02941 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02972 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03204</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02904</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02874</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02938</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02733</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02538</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03237</span></span>
<span class="line"><span>Validation: Loss 0.02558 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02585 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02628</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02807</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02488</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02487</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02336</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02552</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01860</span></span>
<span class="line"><span>Validation: Loss 0.02261 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02286 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02178</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02492</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02028</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02214</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02338</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02206</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02193</span></span>
<span class="line"><span>Validation: Loss 0.02026 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02048 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02064</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02070</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01968</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02083</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02038</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01882</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01982</span></span>
<span class="line"><span>Validation: Loss 0.01830 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01850 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01813</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01805</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02087</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01908</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01731</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01660</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01670</span></span>
<span class="line"><span>Validation: Loss 0.01663 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01682 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01598</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01658</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01797</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01805</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01605</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01465</span></span>
<span class="line"><span>Validation: Loss 0.01521 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01641</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01470</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01569</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01447</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01402</span></span>
<span class="line"><span>Validation: Loss 0.01397 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01413 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01412</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01437</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01446</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01413</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01503</span></span>
<span class="line"><span>Validation: Loss 0.01290 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01304 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01289</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01314</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01366</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01543</span></span>
<span class="line"><span>Validation: Loss 0.01193 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01206 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01202</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01184</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01011</span></span>
<span class="line"><span>Validation: Loss 0.01101 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01111 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01202</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01097</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01117</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01097</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01091</span></span>
<span class="line"><span>Validation: Loss 0.01005 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01012 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01065</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01050</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00930</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00940</span></span>
<span class="line"><span>Validation: Loss 0.00898 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00903 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00889</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00838</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00867</span></span>
<span class="line"><span>Validation: Loss 0.00799 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00804 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00727</span></span>
<span class="line"><span>Validation: Loss 0.00725 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00731 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61659</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60509</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57319</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53192</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51396</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50094</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47924</span></span>
<span class="line"><span>Validation: Loss 0.47097 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46898 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46369</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45125</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44132</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42275</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42220</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39699</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38405</span></span>
<span class="line"><span>Validation: Loss 0.37399 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37151 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36925</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35045</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33542</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33156</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32906</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32295</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29111</span></span>
<span class="line"><span>Validation: Loss 0.28904 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28622 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27582</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28504</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27167</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25198</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24247</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23329</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21407</span></span>
<span class="line"><span>Validation: Loss 0.21913 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21623 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21731</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20174</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19711</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18470</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18060</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18105</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19266</span></span>
<span class="line"><span>Validation: Loss 0.16337 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16062 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15480</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14532</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14735</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12882</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14858</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13769</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12321</span></span>
<span class="line"><span>Validation: Loss 0.11986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11752 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12184</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11339</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10997</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10488</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08952</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09010</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08512</span></span>
<span class="line"><span>Validation: Loss 0.08552 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08377 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08397</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08469</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07960</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06749</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06641</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06353</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05486</span></span>
<span class="line"><span>Validation: Loss 0.05941 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05829 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05785</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05604</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05305</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05078</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05020</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04632</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03761</span></span>
<span class="line"><span>Validation: Loss 0.04413 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04337 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04735</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04154</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03994</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03999</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03597</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03710</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03014</span></span>
<span class="line"><span>Validation: Loss 0.03573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03512 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03568</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03229</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03344</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03345</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03004</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03173</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03567</span></span>
<span class="line"><span>Validation: Loss 0.03038 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02985 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02932</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02922</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02988</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02568</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02678</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02815</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02920</span></span>
<span class="line"><span>Validation: Loss 0.02642 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02528</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02527</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02461</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02332</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02457</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02582</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02168</span></span>
<span class="line"><span>Validation: Loss 0.02336 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02295 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02177</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02218</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02142</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02246</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02303</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02120</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02008</span></span>
<span class="line"><span>Validation: Loss 0.02092 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02054 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02090</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01952</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01822</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01831</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02151</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02020</span></span>
<span class="line"><span>Validation: Loss 0.01889 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01854 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01771</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01813</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01737</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01893</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01674</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01827</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01708</span></span>
<span class="line"><span>Validation: Loss 0.01717 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01684 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01685</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01662</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01703</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01573</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01550</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01679</span></span>
<span class="line"><span>Validation: Loss 0.01569 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01392</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01588</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01440</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01521</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01483</span></span>
<span class="line"><span>Validation: Loss 0.01441 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01412 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01563</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01356</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01354</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01336</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01336</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01242</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01420</span></span>
<span class="line"><span>Validation: Loss 0.01328 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01301 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01239</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01146</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01407</span></span>
<span class="line"><span>Validation: Loss 0.01226 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01201 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01267</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01165</span></span>
<span class="line"><span>Validation: Loss 0.01127 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01103 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01100</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00897</span></span>
<span class="line"><span>Validation: Loss 0.01018 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00996 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00893</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00909</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01002</span></span>
<span class="line"><span>Validation: Loss 0.00904 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00884 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00745</span></span>
<span class="line"><span>Validation: Loss 0.00807 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00791 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00653</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00769</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00814</span></span>
<span class="line"><span>Validation: Loss 0.00739 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00725 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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

import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DrUBLjQW.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR903/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR903/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR903/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR903/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62883</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59724</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55827</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54071</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50833</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49903</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48308</span></span>
<span class="line"><span>Validation: Loss 0.47753 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47304 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46342</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45074</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43079</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42580</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41020</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40500</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39409</span></span>
<span class="line"><span>Validation: Loss 0.38196 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37660 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37346</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36698</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34443</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32032</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30073</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31619</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30853</span></span>
<span class="line"><span>Validation: Loss 0.29748 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29181 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27638</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28090</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26475</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25169</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25101</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22191</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21906</span></span>
<span class="line"><span>Validation: Loss 0.22707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22153 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20860</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20875</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18874</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19470</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18836</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16830</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15695</span></span>
<span class="line"><span>Validation: Loss 0.17014 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16515 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15557</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16090</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14265</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13913</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12589</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12626</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12274</span></span>
<span class="line"><span>Validation: Loss 0.12519 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12117 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11783</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10879</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11521</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08952</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09587</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08860</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09094</span></span>
<span class="line"><span>Validation: Loss 0.08919 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08630 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08029</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08047</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08039</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06834</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06162</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06240</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06287</span></span>
<span class="line"><span>Validation: Loss 0.06179 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05982 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05736</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05470</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05223</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04920</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04563</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04856</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04191</span></span>
<span class="line"><span>Validation: Loss 0.04624 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04476 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04153</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04195</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04001</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03941</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03894</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03659</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03264</span></span>
<span class="line"><span>Validation: Loss 0.03765 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03640 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03471</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03302</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03357</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03086</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03180</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03218</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03274</span></span>
<span class="line"><span>Validation: Loss 0.03210 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03100 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02902</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02991</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02755</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02924</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02917</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02409</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02670</span></span>
<span class="line"><span>Validation: Loss 0.02799 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02701 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02571</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02391</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02501</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02501</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02436</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02351</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02518</span></span>
<span class="line"><span>Validation: Loss 0.02481 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02392 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02149</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02400</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02121</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02146</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02211</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02139</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02027</span></span>
<span class="line"><span>Validation: Loss 0.02224 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02142 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01979</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02004</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01940</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01981</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01936</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01961</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01901</span></span>
<span class="line"><span>Validation: Loss 0.02011 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01936 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01821</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01879</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01804</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01711</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01704</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01779</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01661</span></span>
<span class="line"><span>Validation: Loss 0.01830 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01761 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01622</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01568</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01626</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01620</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01755</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01594</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01333</span></span>
<span class="line"><span>Validation: Loss 0.01677 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01612 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01538</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01516</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01585</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01166</span></span>
<span class="line"><span>Validation: Loss 0.01545 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01484 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01416</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01497</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01711</span></span>
<span class="line"><span>Validation: Loss 0.01429 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01373 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01377</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01258</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01385</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01179</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01340</span></span>
<span class="line"><span>Validation: Loss 0.01323 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01271 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01247</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01198</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01125</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00918</span></span>
<span class="line"><span>Validation: Loss 0.01222 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01175 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01087</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01119</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01262</span></span>
<span class="line"><span>Validation: Loss 0.01118 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01076 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01065</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00800</span></span>
<span class="line"><span>Validation: Loss 0.00998 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00963 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00912</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00896</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00746</span></span>
<span class="line"><span>Validation: Loss 0.00885 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00854 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00727</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00812</span></span>
<span class="line"><span>Validation: Loss 0.00801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00774 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62708</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58025</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58181</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53561</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51663</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50634</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46787</span></span>
<span class="line"><span>Validation: Loss 0.47616 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47763</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46042</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44031</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43836</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40603</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38532</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37587</span></span>
<span class="line"><span>Validation: Loss 0.38117 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36188 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36868</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35408</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35713</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32341</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33460</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32052</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28949</span></span>
<span class="line"><span>Validation: Loss 0.29822 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27695 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27402</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27921</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28370</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25648</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24783</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24378</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22391</span></span>
<span class="line"><span>Validation: Loss 0.22912 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20792 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21497</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21507</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21203</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19958</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18312</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17444</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16676</span></span>
<span class="line"><span>Validation: Loss 0.17279 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15340 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16217</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16054</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14376</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14753</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14033</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14061</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10872</span></span>
<span class="line"><span>Validation: Loss 0.12799 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11198 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12604</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10790</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11917</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09819</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09434</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10430</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09473</span></span>
<span class="line"><span>Validation: Loss 0.09196 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08010 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08610</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07770</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07510</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08028</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07077</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07208</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05460</span></span>
<span class="line"><span>Validation: Loss 0.06365 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05578 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06325</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05653</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05054</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05477</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04808</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04983</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04801</span></span>
<span class="line"><span>Validation: Loss 0.04699 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04140 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04628</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04360</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03973</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03918</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04097</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03750</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03835</span></span>
<span class="line"><span>Validation: Loss 0.03796 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03340 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03551</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03672</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03317</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03266</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03247</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03420</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02689</span></span>
<span class="line"><span>Validation: Loss 0.03220 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02825 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03301</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03051</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02651</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02978</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02720</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02766</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02618</span></span>
<span class="line"><span>Validation: Loss 0.02803 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02452 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02640</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02567</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02544</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02497</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02502</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02457</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02569</span></span>
<span class="line"><span>Validation: Loss 0.02482 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02165 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02333</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02169</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02196</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02447</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02086</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02273</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02294</span></span>
<span class="line"><span>Validation: Loss 0.02222 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01934 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01990</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01998</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02102</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02016</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01966</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02048</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02073</span></span>
<span class="line"><span>Validation: Loss 0.02008 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01743 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01847</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01839</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01761</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01892</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01843</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01618</span></span>
<span class="line"><span>Validation: Loss 0.01826 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01581 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01711</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01623</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01748</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01677</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01719</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01541</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01597</span></span>
<span class="line"><span>Validation: Loss 0.01671 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01442 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01470</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01544</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01602</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01522</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01490</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01759</span></span>
<span class="line"><span>Validation: Loss 0.01537 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01324 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01576</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01427</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01300</span></span>
<span class="line"><span>Validation: Loss 0.01419 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01220 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01103</span></span>
<span class="line"><span>Validation: Loss 0.01316 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01130 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01229</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01154</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01401</span></span>
<span class="line"><span>Validation: Loss 0.01222 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01049 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01235</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01296</span></span>
<span class="line"><span>Validation: Loss 0.01129 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00970 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01099</span></span>
<span class="line"><span>Validation: Loss 0.01026 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00884 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01016</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00893</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00846</span></span>
<span class="line"><span>Validation: Loss 0.00911 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00789 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00820</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00762</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00779</span></span>
<span class="line"><span>Validation: Loss 0.00812 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00707 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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

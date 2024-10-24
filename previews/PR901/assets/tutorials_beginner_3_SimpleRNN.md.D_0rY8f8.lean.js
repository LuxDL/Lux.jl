import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.B85O5OJK.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR901/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR901/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR901/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR901/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63467</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59127</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55949</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54212</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52052</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50006</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49581</span></span>
<span class="line"><span>Validation: Loss 0.46522 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46275 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46721</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46170</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44020</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42758</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41142</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39956</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38887</span></span>
<span class="line"><span>Validation: Loss 0.36714 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36407 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38249</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35473</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34213</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33922</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31899</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30584</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31311</span></span>
<span class="line"><span>Validation: Loss 0.28110 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27775 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28561</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26966</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26892</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24235</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24683</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24995</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24067</span></span>
<span class="line"><span>Validation: Loss 0.21066 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20745 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21955</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20149</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20598</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18746</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17585</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18458</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17264</span></span>
<span class="line"><span>Validation: Loss 0.15510 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15244 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15489</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15282</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16283</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13406</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12947</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13595</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11791</span></span>
<span class="line"><span>Validation: Loss 0.11290 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11094 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11363</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12732</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10278</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10350</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09346</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09550</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07517</span></span>
<span class="line"><span>Validation: Loss 0.08066 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07933 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07978</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07758</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07694</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07189</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07257</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06475</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07439</span></span>
<span class="line"><span>Validation: Loss 0.05646 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05561 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05904</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05229</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05119</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05159</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05279</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04947</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04212</span></span>
<span class="line"><span>Validation: Loss 0.04201 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04142 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04053</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04191</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04267</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03942</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04232</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03495</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03968</span></span>
<span class="line"><span>Validation: Loss 0.03401 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03354 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03399</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03626</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03456</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03456</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02946</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03173</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03046</span></span>
<span class="line"><span>Validation: Loss 0.02884 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02845 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02754</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03323</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02740</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02878</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02775</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02861</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02097</span></span>
<span class="line"><span>Validation: Loss 0.02509 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02475 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02554</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02687</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02527</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02377</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02421</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02486</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02421</span></span>
<span class="line"><span>Validation: Loss 0.02221 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02190 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02671</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02304</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02369</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02035</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01970</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02393</span></span>
<span class="line"><span>Validation: Loss 0.01987 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01959 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02191</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02086</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01982</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01900</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01995</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01931</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01748</span></span>
<span class="line"><span>Validation: Loss 0.01793 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01768 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01810</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01767</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01945</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01821</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01836</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01715</span></span>
<span class="line"><span>Validation: Loss 0.01630 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01607 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01661</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01751</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01610</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01725</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01387</span></span>
<span class="line"><span>Validation: Loss 0.01490 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01469 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01474</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01427</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01512</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01654</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01520</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01612</span></span>
<span class="line"><span>Validation: Loss 0.01370 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01350 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01384</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01534</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01365</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01429</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01332</span></span>
<span class="line"><span>Validation: Loss 0.01265 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01247 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01341</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01237</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01346</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01381</span></span>
<span class="line"><span>Validation: Loss 0.01171 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01155 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01253</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01187</span></span>
<span class="line"><span>Validation: Loss 0.01083 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01068 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01104</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01189</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01052</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01246</span></span>
<span class="line"><span>Validation: Loss 0.00990 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00977 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00978</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01001</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01009</span></span>
<span class="line"><span>Validation: Loss 0.00886 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00875 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00888</span></span>
<span class="line"><span>Validation: Loss 0.00786 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00776 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00824</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00737</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00668</span></span>
<span class="line"><span>Validation: Loss 0.00713 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00704 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61989</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59445</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57251</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53817</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52356</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49596</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48740</span></span>
<span class="line"><span>Validation: Loss 0.46533 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46810 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47191</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44878</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43721</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42953</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41019</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40261</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37991</span></span>
<span class="line"><span>Validation: Loss 0.36677 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37007 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35948</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35291</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34714</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33204</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33203</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31656</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28422</span></span>
<span class="line"><span>Validation: Loss 0.28082 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28461 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28631</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26921</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26587</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25131</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24598</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24237</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20481</span></span>
<span class="line"><span>Validation: Loss 0.21062 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21453 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22513</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21419</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19889</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19699</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17525</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16009</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15663</span></span>
<span class="line"><span>Validation: Loss 0.15536 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15904 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16413</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14673</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15066</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13976</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13264</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13195</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10978</span></span>
<span class="line"><span>Validation: Loss 0.11349 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11659 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12152</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12611</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10573</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09787</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09527</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08352</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09047</span></span>
<span class="line"><span>Validation: Loss 0.08130 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08364 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08900</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08033</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07362</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06748</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07316</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06411</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05745</span></span>
<span class="line"><span>Validation: Loss 0.05681 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05836 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06049</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05636</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05211</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04888</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04730</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04882</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04476</span></span>
<span class="line"><span>Validation: Loss 0.04208 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04312 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04187</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04463</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04130</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03630</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03759</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03801</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03692</span></span>
<span class="line"><span>Validation: Loss 0.03400 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03484 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03402</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03453</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03418</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03391</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03057</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03063</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02914</span></span>
<span class="line"><span>Validation: Loss 0.02879 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02953 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03102</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02916</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02739</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02844</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02765</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02580</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02528</span></span>
<span class="line"><span>Validation: Loss 0.02503 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02568 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02571</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02566</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02435</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02391</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02554</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02289</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02309</span></span>
<span class="line"><span>Validation: Loss 0.02212 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02271 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02116</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02371</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02123</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02219</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02092</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02116</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02499</span></span>
<span class="line"><span>Validation: Loss 0.01978 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02033 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01854</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01878</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02006</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01978</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01933</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02092</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02057</span></span>
<span class="line"><span>Validation: Loss 0.01784 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01834 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01933</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02019</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01750</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01694</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01718</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01538</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01809</span></span>
<span class="line"><span>Validation: Loss 0.01618 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01664 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01664</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01507</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01606</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01645</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01739</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01487</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01770</span></span>
<span class="line"><span>Validation: Loss 0.01477 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01520 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01506</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01502</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01581</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01434</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01470</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01414</span></span>
<span class="line"><span>Validation: Loss 0.01357 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01397 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01355</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01401</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01437</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01370</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01005</span></span>
<span class="line"><span>Validation: Loss 0.01253 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01291 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01276</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01259</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01256</span></span>
<span class="line"><span>Validation: Loss 0.01162 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01197 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01231</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01208</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01217</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01104</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01168</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01208</span></span>
<span class="line"><span>Validation: Loss 0.01077 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01110 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01150</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01007</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00955</span></span>
<span class="line"><span>Validation: Loss 0.00991 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01022 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01139</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00934</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00911</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00971</span></span>
<span class="line"><span>Validation: Loss 0.00897 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00924 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00934</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00912</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00897</span></span>
<span class="line"><span>Validation: Loss 0.00798 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00822 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00851</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00811</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00721</span></span>
<span class="line"><span>Validation: Loss 0.00718 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00738 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
